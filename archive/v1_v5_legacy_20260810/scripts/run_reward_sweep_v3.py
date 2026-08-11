"""Run the exploratory PPO reward/training recovery sweep.

The runner develops only on cells a-d, preserves the frozen v2 outputs, ranks
policies on a fixed full-dollar objective, and evaluates e-h once after the
winning US and Global configurations are frozen.  MPC is explicitly out of
scope.

Usage:
    python scripts/run_reward_sweep_v3.py --phase preflight
    python scripts/run_reward_sweep_v3.py --phase round1 --workers 16
    python scripts/run_reward_sweep_v3.py --phase round2 --workers 16
    python scripts/run_reward_sweep_v3.py --phase full --workers 16
    python scripts/run_reward_sweep_v3.py --phase budget --workers 16
    python scripts/run_reward_sweep_v3.py --phase final
    python scripts/run_reward_sweep_v3.py --phase all --workers 16
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines import (  # noqa: E402
    DrainImmediatelyPolicy,
    RoundRobinPolicy,
    StatusQuoPolicy,
)
from env.reward import RewardConfig  # noqa: E402
from evaluate import compute_summary, run_episode  # noqa: E402
from train_v3 import (  # noqa: E402
    DEFAULT_DEADLINE_BUCKET_EDGES,
    make_recovery_env,
)


PROTOCOL_PATH = ROOT / "env" / "protocols" / "v3_reward_sweep.yaml"
CONFIG = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
MODEL_ROOT = ROOT / "models" / "ppo_v3_reward_sweep"
LOG_ROOT = ROOT / "logs" / "ppo_v3_reward_sweep"
OUT_ROOT = ROOT / "output" / "ppo_v3_reward_sweep"
PROTOCOL_SNAPSHOT = OUT_ROOT / "protocol.json"
ALPHA = float(CONFIG["environment"]["peak_penalty_weight"])
N_STEPS = int(CONFIG["ppo"]["n_steps"])
NET_ARCH = tuple(int(width) for width in CONFIG["ppo"]["net_arch"])
DEADLINE_EDGES = tuple(
    int(edge) for edge in CONFIG["environment"]["deadline_bucket_edges"]
)

SOURCE_FILES = [
    "baselines.py",
    "evaluate.py",
    "train_v3.py",
    "env/data_loader.py",
    "env/dc_site.py",
    "env/multi_dc_env.py",
    "env/power_model.py",
    "env/reward.py",
    "env/workload_generator.py",
    "env/protocols/v3_reward_sweep.yaml",
    "scripts/run_reward_sweep_v3.py",
    "scripts/smoke_test_demand_charge.py",
    "scripts/smoke_test_ppo_v3.py",
    "scripts/preflight_energy_model_v2.py",
    "requirements.txt",
    "env/scenarios/us_model_v2_2025.yaml",
    "env/scenarios/us_model_eh_v2_2025.yaml",
    "env/scenarios/global_model_v2_2025.yaml",
    "env/scenarios/global_model_eh_v2_2025.yaml",
]


@dataclass(frozen=True)
class Job:
    stage: str
    region: str
    candidate: str
    variant: str | None
    seed: int
    timesteps: int
    gae_lambda: float
    normalize_observations: bool
    learning_rate_schedule: str
    target_kl: float | None
    batch_size: int
    subtract_idle_cost: bool
    service_backlog_weight: float
    batch_completion_weight: float
    urgency_potential_weight: float

    @property
    def combo(self) -> str:
        return (
            self.candidate
            if self.variant is None
            else f"{self.candidate}_{self.variant}"
        )

    @property
    def label(self) -> str:
        return f"{self.stage}:{self.region}:{self.combo}:s{self.seed}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def scenario_path(region: str, evaluation: bool = False) -> Path:
    scope = CONFIG["development_scope"]
    key = "evaluation_scenarios" if evaluation else "train_scenarios"
    return ROOT / scope[key][region]


def protocol() -> dict[str, Any]:
    package_versions = {
        package: importlib.metadata.version(package)
        for package in (
            "numpy",
            "pandas",
            "torch",
            "gymnasium",
            "stable-baselines3",
            "scipy",
        )
    }
    source_hashes = {
        path: sha256(ROOT / path) for path in SOURCE_FILES
    }
    data_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in sorted((ROOT / "data").rglob("*"))
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}
    }
    return {
        "protocol": CONFIG,
        "protocol_config": str(PROTOCOL_PATH.relative_to(ROOT)).replace(
            "\\", "/"
        ),
        "protocol_config_sha256": sha256(PROTOCOL_PATH),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "Post-hoc exploratory PPO development on a-d only. The e-h "
            "evaluation is one-time but not fresh confirmatory evidence."
        ),
        "excluded": [
            "model_predictive_control",
            "spatial_only_training",
        ],
        "provenance": {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                text=True,
            ).strip(),
            "git_status_at_freeze": subprocess.check_output(
                ["git", "status", "--short"],
                cwd=ROOT,
                text=True,
            ).splitlines(),
            "python": sys.version,
            "packages": package_versions,
            "source_sha256": source_hashes,
            "data_sha256": data_hashes,
        },
    }


def write_protocol() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    current = protocol()
    if PROTOCOL_SNAPSHOT.exists():
        existing = json.loads(PROTOCOL_SNAPSHOT.read_text(encoding="utf-8"))
        existing["provenance"].pop("git_status_at_freeze", None)
        current["provenance"].pop("git_status_at_freeze", None)
        old_immutable = {
            key: value
            for key, value in existing.items()
            if key != "frozen_at_utc"
        }
        new_immutable = {
            key: value
            for key, value in current.items()
            if key != "frozen_at_utc"
        }
        if old_immutable != new_immutable:
            raise RuntimeError(
                "v3 protocol snapshot differs from current source/data; "
                "refusing to mix experiments"
            )
        return
    PROTOCOL_SNAPSHOT.write_text(
        json.dumps(current, indent=2),
        encoding="utf-8",
    )


def assert_provenance() -> None:
    if not PROTOCOL_SNAPSHOT.exists():
        raise RuntimeError("run --phase preflight before training")
    frozen = json.loads(
        PROTOCOL_SNAPSHOT.read_text(encoding="utf-8")
    )["provenance"]
    if sys.version != frozen["python"]:
        raise RuntimeError("Python runtime changed after v3 protocol freeze")
    for package, expected in frozen["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(
                f"package changed after v3 freeze: {package}"
            )
    for path, expected in frozen["source_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(
                f"source changed after v3 freeze: {path}"
            )
    for path, expected in frozen["data_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"data changed after v3 freeze: {path}")


def validate_config() -> None:
    if CONFIG["excluded"] != [
        "model_predictive_control",
        "spatial_only_training",
    ]:
        raise ValueError(
            "v3 protocol must exclude MPC and spatial-only training"
        )
    if CONFIG["controller_scope"]["active"] != (
        "joint_temporal_and_spatial"
    ):
        raise ValueError("v3 trains only the joint controller")
    if not math.isclose(
        float(CONFIG["environment"]["gamma"]),
        1.0,
        abs_tol=1e-12,
    ):
        raise ValueError("v3 requires gamma=1")
    if DEADLINE_EDGES != DEFAULT_DEADLINE_BUCKET_EDGES:
        raise ValueError(
            "protocol and training deadline buckets must match"
        )
    for section in ("round_1", "round_2", "full_development"):
        timesteps = int(CONFIG[section]["timesteps"])
        if timesteps <= 0 or timesteps % N_STEPS != 0:
            raise ValueError(
                f"{section} timesteps must be positive and rollout-aligned"
            )
    round1_total = (
        2
        * len(CONFIG["round_1"]["candidates"])
        * len(CONFIG["round_1"]["seeds"])
    )
    round2_total = (
        2
        * int(CONFIG["round_1"]["promote_per_region"])
        * len(CONFIG["round_2"]["variants"])
        * len(CONFIG["round_2"]["seeds"])
    )
    full_total = 2 * len(CONFIG["full_development"]["seeds"])
    if (round1_total, round2_total, full_total) != (36, 40, 20):
        raise ValueError(
            "successive-halving job counts must be 36/40/20, got "
            f"{round1_total}/{round2_total}/{full_total}"
        )
    budgets = [
        int(value) for value in CONFIG["budget_scaling"]["timesteps"]
    ]
    if budgets != [151_552, 501_760, 1_003_520]:
        raise ValueError("budget-scaling curve must use the frozen three budgets")
    if any(value % N_STEPS != 0 for value in budgets):
        raise ValueError("all budget-scaling points must be rollout-aligned")


def base_candidate(candidate: str) -> dict[str, Any]:
    return dict(CONFIG["round_1"]["candidates"][candidate])


def reward_for(
    candidate: str,
    weights: dict[str, Any],
) -> dict[str, Any]:
    ppo = base_candidate(candidate)
    return {
        "service_backlog_weight": float(
            weights["service_backlog_weight"]
        ),
        "batch_completion_weight": float(
            weights["batch_completion_weight"]
        ),
        "reward_scale": float(CONFIG["environment"]["reward_scale"]),
        "subtract_idle_cost": ppo["reward_mode"] == "idle_subtracted",
        "urgency_potential_weight": float(
            weights["urgency_potential_weight"]
        ),
    }


def make_job(
    stage: str,
    region: str,
    candidate: str,
    variant: str | None,
    seed: int,
    timesteps: int,
    reward: dict[str, Any],
) -> Job:
    ppo = base_candidate(candidate)
    return Job(
        stage=stage,
        region=region,
        candidate=candidate,
        variant=variant,
        seed=int(seed),
        timesteps=int(timesteps),
        gae_lambda=float(ppo["gae_lambda"]),
        normalize_observations=bool(
            ppo["normalize_observations"]
        ),
        learning_rate_schedule=str(
            ppo["learning_rate_schedule"]
        ),
        target_kl=(
            None if ppo["target_kl"] is None else float(ppo["target_kl"])
        ),
        batch_size=int(ppo["batch_size"]),
        subtract_idle_cost=bool(reward["subtract_idle_cost"]),
        service_backlog_weight=float(
            reward["service_backlog_weight"]
        ),
        batch_completion_weight=float(
            reward["batch_completion_weight"]
        ),
        urgency_potential_weight=float(
            reward["urgency_potential_weight"]
        ),
    )


def round1_jobs() -> list[Job]:
    section = CONFIG["round_1"]
    reward = section["reward"]
    return [
        make_job(
            "round1",
            region,
            candidate,
            None,
            seed,
            section["timesteps"],
            reward_for(candidate, reward),
        )
        for region in ("us", "global")
        for candidate in section["candidates"]
        for seed in section["seeds"]
    ]


def read_selection(stage: str) -> dict[str, Any]:
    path = OUT_ROOT / f"{stage}_selection.json"
    if not path.exists():
        raise RuntimeError(f"missing selection: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def round2_jobs() -> list[Job]:
    selection = read_selection("round1")
    section = CONFIG["round_2"]
    jobs = []
    for region in ("us", "global"):
        for selected in selection["selected"][region]:
            candidate = selected["candidate"]
            for variant, weights in section["variants"].items():
                reward = reward_for(candidate, weights)
                for seed in section["seeds"]:
                    jobs.append(
                        make_job(
                            "round2",
                            region,
                            candidate,
                            variant,
                            seed,
                            section["timesteps"],
                            reward,
                        )
                    )
    return jobs


def full_jobs() -> list[Job]:
    selection = read_selection("round2")
    section = CONFIG["full_development"]
    jobs = []
    for region in ("us", "global"):
        selected = selection["selected"][region][0]
        candidate = selected["candidate"]
        variant = selected["variant"]
        weights = CONFIG["round_2"]["variants"][variant]
        reward = reward_for(candidate, weights)
        for seed in section["seeds"]:
            jobs.append(
                make_job(
                    "full",
                    region,
                    candidate,
                    variant,
                    seed,
                    section["timesteps"],
                    reward,
                )
            )
    return jobs


def selected_combo(region: str) -> tuple[str, str, dict[str, Any]]:
    selected = read_selection("round2")["selected"][region][0]
    candidate = selected["candidate"]
    variant = selected["variant"]
    if variant is None:
        raise RuntimeError("round-two winner must include a reward variant")
    return (
        candidate,
        variant,
        reward_for(
            candidate,
            CONFIG["round_2"]["variants"][variant],
        ),
    )


def make_budget_job(
    region: str,
    timesteps: int,
    seed: int,
) -> Job:
    candidate, variant, reward = selected_combo(region)
    return make_job(
        f"budget_{timesteps}",
        region,
        candidate,
        variant,
        seed,
        timesteps,
        reward,
    )


def budget_comparison_jobs() -> list[Job]:
    budgets = [
        int(value) for value in CONFIG["budget_scaling"]["timesteps"]
    ]
    seeds = [
        int(value)
        for value in CONFIG["budget_scaling"]["comparison_seeds"]
    ]
    existing_full = {
        (job.region, job.seed): job for job in full_jobs()
    }
    jobs = []
    for region in ("us", "global"):
        for timesteps in budgets:
            for seed in seeds:
                if timesteps == int(
                    CONFIG["full_development"]["timesteps"]
                ):
                    jobs.append(existing_full[(region, seed)])
                else:
                    jobs.append(
                        make_budget_job(region, timesteps, seed)
                    )
    return jobs


def budget_training_jobs() -> list[Job]:
    return [
        job
        for job in budget_comparison_jobs()
        if job.stage != "full"
    ]


def selected_budget_jobs() -> list[Job]:
    selection = read_selection("budget")
    seeds = [
        int(value)
        for value in CONFIG["budget_scaling"][
            "selected_budget_replication_seeds"
        ]
    ]
    existing_full = {
        (job.region, job.seed): job for job in full_jobs()
    }
    jobs = []
    for region in ("us", "global"):
        timesteps = int(
            selection["selected"][region][0]["timesteps"]
        )
        for seed in seeds:
            if timesteps == int(
                CONFIG["full_development"]["timesteps"]
            ):
                jobs.append(existing_full[(region, seed)])
            else:
                jobs.append(make_budget_job(region, timesteps, seed))
    return jobs


def job_dir(job: Job) -> Path:
    return (
        MODEL_ROOT
        / job.stage
        / job.region
        / job.combo
        / f"s{job.seed}"
    )


def model_path(job: Job) -> Path:
    return job_dir(job) / "model.zip"


def vecnormalize_path(job: Job) -> Path | None:
    return (
        job_dir(job) / "vecnormalize.pkl"
        if job.normalize_observations
        else None
    )


def diagnostics_path(job: Job) -> Path:
    return job_dir(job) / "training.json"


def completion_path(job: Job) -> Path:
    return job_dir(job) / "complete.json"


def log_path(job: Job) -> Path:
    return LOG_ROOT / job.stage / job.region / f"{job.combo}_s{job.seed}.log"


def job_spec(job: Job) -> dict[str, Any]:
    spec = {
        **asdict(job),
        "protocol_sha256": sha256(PROTOCOL_SNAPSHOT),
        "train_scenario": str(
            scenario_path(job.region).relative_to(ROOT)
        ).replace("\\", "/"),
        "algorithm": "PPO",
        "learning_rate": float(CONFIG["ppo"]["learning_rate"]),
        "n_steps": N_STEPS,
        "n_epochs": int(CONFIG["ppo"]["n_epochs"]),
        "gamma": 1.0,
        "net_arch": list(NET_ARCH),
        "deadline_bucket_edges": list(DEADLINE_EDGES),
        "evaluation_objective": {
            "service_backlog_weight": 1000.0,
            "batch_completion_weight": 1000.0,
            "full_dollar_cost_unchanged": True,
        },
    }
    spec["job_fingerprint"] = canonical_hash(spec)
    return spec


def validate_completed_job(job: Job) -> bool:
    paths = [
        model_path(job),
        diagnostics_path(job),
        completion_path(job),
    ]
    stats_path = vecnormalize_path(job)
    if stats_path is not None:
        paths.append(stats_path)
    existence = [path.exists() for path in paths]
    if not any(existence):
        return False
    if not all(existence):
        raise RuntimeError(
            f"incomplete prior output for {job.label}: "
            + ", ".join(str(path) for path in paths)
        )
    record = json.loads(
        completion_path(job).read_text(encoding="utf-8")
    )
    expected = job_spec(job)
    if record.get("job_fingerprint") != expected["job_fingerprint"]:
        raise RuntimeError(f"job fingerprint mismatch: {job.label}")
    for key, path in (
        ("model_sha256", model_path(job)),
        ("diagnostics_sha256", diagnostics_path(job)),
    ):
        if record.get(key) != sha256(path):
            raise RuntimeError(f"{key} mismatch: {job.label}")
    if stats_path is not None and record.get(
        "vecnormalize_sha256"
    ) != sha256(stats_path):
        raise RuntimeError(
            f"VecNormalize hash mismatch: {job.label}"
        )
    model = PPO.load(model_path(job))
    if model.num_timesteps != job.timesteps:
        raise RuntimeError(
            f"timestep mismatch for {job.label}: "
            f"{model.num_timesteps} != {job.timesteps}"
        )
    return True


def train_command(job: Job, temp_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "train_v3.py",
        "--scenario",
        str(scenario_path(job.region)),
        "--model-path",
        str(temp_dir / "model.zip"),
        "--diagnostics-path",
        str(temp_dir / "training.json"),
        "--timesteps",
        str(job.timesteps),
        "--seed",
        str(job.seed),
        "--lr",
        str(CONFIG["ppo"]["learning_rate"]),
        "--lr-schedule",
        job.learning_rate_schedule,
        "--n-steps",
        str(N_STEPS),
        "--batch-size",
        str(job.batch_size),
        "--n-epochs",
        str(CONFIG["ppo"]["n_epochs"]),
        "--gamma",
        "1.0",
        "--gae-lambda",
        str(job.gae_lambda),
        "--net-arch",
        *[str(width) for width in NET_ARCH],
        "--peak-penalty-weight",
        str(ALPHA),
        "--service-backlog-weight",
        str(job.service_backlog_weight),
        "--batch-completion-weight",
        str(job.batch_completion_weight),
        "--reward-scale",
        str(CONFIG["environment"]["reward_scale"]),
        "--urgency-potential-weight",
        str(job.urgency_potential_weight),
        "--deadline-bucket-edges",
        *[str(edge) for edge in DEADLINE_EDGES],
    ]
    if job.target_kl is not None:
        cmd += ["--target-kl", str(job.target_kl)]
    if job.subtract_idle_cost:
        cmd.append("--subtract-idle-cost")
    if job.normalize_observations:
        cmd += [
            "--normalize-observations",
            "--vecnormalize-path",
            str(temp_dir / "vecnormalize.pkl"),
        ]
    return cmd


def train_one(job: Job) -> tuple[str, bool, float]:
    assert_provenance()
    if validate_completed_job(job):
        return job.label, True, 0.0

    final_dir = job_dir(job)
    final_dir.mkdir(parents=True, exist_ok=True)
    log = log_path(job)
    log.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = (
        MODEL_ROOT
        / "_tmp"
        / f"{job.stage}_{job.region}_{job.combo}_s{job.seed}_{uuid.uuid4().hex}"
    )
    temp_dir.mkdir(parents=True, exist_ok=False)
    command = train_command(job, temp_dir)
    lock = final_dir / "training.lock"
    lock_fd: int | None = None
    started = time.time()
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with log.open("w", encoding="utf-8") as handle:
            handle.write(
                f"# {job.label}\n"
                f"# job: {json.dumps(job_spec(job), sort_keys=True)}\n"
                f"# command: {' '.join(command)}\n"
                f"# started: {datetime.now(timezone.utc).isoformat()}\n\n"
            )
            handle.flush()
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONIOENCODING": "utf-8",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
            )
        elapsed = time.time() - started
        temp_model = temp_dir / "model.zip"
        temp_diagnostics = temp_dir / "training.json"
        temp_stats = temp_dir / "vecnormalize.pkl"
        if (
            result.returncode != 0
            or not temp_model.exists()
            or not temp_diagnostics.exists()
            or (job.normalize_observations and not temp_stats.exists())
        ):
            return job.label, False, elapsed

        loaded = PPO.load(temp_model)
        if loaded.num_timesteps != job.timesteps:
            raise RuntimeError(
                f"trained timestep mismatch for {job.label}"
            )
        assert_provenance()
        os.replace(temp_model, model_path(job))
        os.replace(temp_diagnostics, diagnostics_path(job))
        final_stats = vecnormalize_path(job)
        if final_stats is not None:
            os.replace(temp_stats, final_stats)
        record = {
            **job_spec(job),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "model_path": str(
                model_path(job).relative_to(ROOT)
            ).replace("\\", "/"),
            "model_bytes": model_path(job).stat().st_size,
            "model_sha256": sha256(model_path(job)),
            "diagnostics_sha256": sha256(diagnostics_path(job)),
            "vecnormalize_sha256": (
                sha256(final_stats) if final_stats is not None else None
            ),
            "log_path": str(log.relative_to(ROOT)).replace("\\", "/"),
            "log_sha256": sha256(log),
        }
        temp_record = completion_path(job).with_suffix(".tmp")
        temp_record.write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )
        os.replace(temp_record, completion_path(job))
        return job.label, True, elapsed
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            lock.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


def train_all(jobs: list[Job], workers: int) -> None:
    failures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as pool:
        future_jobs = {
            pool.submit(train_one, job): job for job in jobs
        }
        for future in concurrent.futures.as_completed(future_jobs):
            label, ok, elapsed = future.result()
            print(
                f"[{'OK' if ok else 'FAIL'}] {label} "
                f"({elapsed / 60:.1f} min)",
                flush=True,
            )
            if not ok:
                failures.append(label)
    if failures:
        raise RuntimeError(
            "training failed: " + ", ".join(failures)
        )


def reward_config(job: Job) -> RewardConfig:
    return RewardConfig(
        service_backlog_weight=job.service_backlog_weight,
        batch_completion_weight=job.batch_completion_weight,
        evaluation_service_backlog_weight=1000.0,
        evaluation_batch_completion_weight=1000.0,
        reward_scale=float(CONFIG["environment"]["reward_scale"]),
        subtract_idle_cost=job.subtract_idle_cost,
        urgency_potential_weight=job.urgency_potential_weight,
    )


def evaluate_job(
    job: Job,
    *,
    evaluation_cells: bool = False,
) -> dict[str, Any]:
    env = make_recovery_env(
        scenario_path(job.region, evaluation=evaluation_cells),
        seed=42,
        peak_penalty_weight=ALPHA,
        service_backlog_weight=job.service_backlog_weight,
        batch_completion_weight=job.batch_completion_weight,
        reward_scale=float(CONFIG["environment"]["reward_scale"]),
        subtract_idle_cost=job.subtract_idle_cost,
        urgency_potential_weight=job.urgency_potential_weight,
        domain_randomization=False,
        deadline_bucket_edges=DEADLINE_EDGES,
    )
    max_steps = env.max_steps
    vec_env = DummyVecEnv([lambda: env])
    stats_path = vecnormalize_path(job)
    if stats_path is not None:
        vec_env = VecNormalize.load(stats_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(model_path(job))
    obs = vec_env.reset()
    history: list[dict[str, Any]] = []
    action_values = 0
    saturated_actions = 0
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        values = np.asarray(action, dtype=np.float64)
        action_values += values.size
        saturated_actions += int(
            np.count_nonzero(np.abs(values) >= 2.95)
        )
        obs, _, dones, infos = vec_env.step(action)
        history.append(infos[0])
        if bool(dones[0]):
            break
    vec_env.close()
    if len(history) != max_steps:
        raise RuntimeError(
            f"evaluation length mismatch for {job.label}: "
            f"{len(history)} != {max_steps}"
        )
    summary = compute_summary(history, batch_enabled=True)
    summary["action_saturation_fraction"] = (
        saturated_actions / action_values if action_values else 0.0
    )
    summary["safe"] = safe(summary)
    summary["job"] = asdict(job)
    summary["model_sha256"] = sha256(model_path(job))
    summary["evaluation_cells"] = "e-h" if evaluation_cells else "a-d"
    return summary


def baseline_summaries(
    region: str,
    *,
    evaluation_cells: bool = False,
) -> dict[str, Any]:
    output = {}
    for baseline_cls in (
        StatusQuoPolicy,
        RoundRobinPolicy,
        DrainImmediatelyPolicy,
    ):
        env = make_recovery_env(
            scenario_path(region, evaluation=evaluation_cells),
            seed=42,
            peak_penalty_weight=ALPHA,
            service_backlog_weight=1000.0,
            batch_completion_weight=1000.0,
            reward_scale=float(CONFIG["environment"]["reward_scale"]),
            subtract_idle_cost=False,
            urgency_potential_weight=0.0,
            domain_randomization=False,
            deadline_bucket_edges=DEADLINE_EDGES,
        )
        policy = baseline_cls()
        _, history = run_episode(
            env,
            policy.predict,
            is_sb3=False,
        )
        output[policy.name] = compute_summary(
            history,
            batch_enabled=True,
        )
    return output


def safe(summary: dict[str, Any]) -> bool:
    selection = CONFIG["selection"]
    service = (
        summary["service_served_total"]
        / summary["service_demand_total"]
        if summary["service_demand_total"] > 0.0
        else 1.0
    )
    terminal_tolerance = max(
        1e-6,
        float(selection["terminal_pool_fraction_tolerance"])
        * summary.get("batch_arrival_total", 0.0),
    )
    return bool(
        service >= float(selection["service_completion_floor"])
        and summary.get("batch_completion_fraction", 1.0)
        >= float(selection["batch_completion_floor"])
        and summary.get("total_batch_expired", 0.0)
        <= float(selection["expired_tolerance"])
        and summary.get("terminal_batch_pool", 0.0)
        <= terminal_tolerance
        and summary.get("max_backlog", 0.0)
        <= float(selection["max_service_backlog"])
    )


def aggregate_group(
    summaries: dict[str, dict[str, Any]],
    status_quo_cost: float,
) -> dict[str, Any]:
    ordered = [summaries[key] for key in sorted(summaries, key=int)]
    costs = np.asarray(
        [summary["total_cost"] for summary in ordered],
        dtype=np.float64,
    )
    safe_values = np.asarray(
        [bool(summary["safe"]) for summary in ordered],
        dtype=bool,
    )
    savings = 100.0 * (status_quo_cost - costs) / status_quo_cost
    return {
        "seed_count": len(ordered),
        "safe_seed_count": int(safe_values.sum()),
        "all_seeds_safe": bool(safe_values.all()),
        "worst_seed_total_cost": float(costs.max()),
        "mean_total_cost": float(costs.mean()),
        "std_total_cost": float(
            costs.std(ddof=1) if len(costs) > 1 else 0.0
        ),
        "mean_savings_vs_status_quo_pct": float(savings.mean()),
        "worst_savings_vs_status_quo_pct": float(savings.min()),
        "minimum_batch_completion": float(
            min(
                summary["batch_completion_fraction"]
                for summary in ordered
            )
        ),
        "maximum_service_backlog": float(
            max(summary["max_backlog"] for summary in ordered)
        ),
        "maximum_terminal_batch_pool": float(
            max(
                summary["terminal_batch_pool"]
                for summary in ordered
            )
        ),
        "total_expired": float(
            sum(
                summary["total_batch_expired"]
                for summary in ordered
            )
        ),
        "mean_action_saturation_fraction": float(
            np.mean(
                [
                    summary["action_saturation_fraction"]
                    for summary in ordered
                ]
            )
        ),
    }


def rank_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if summary["all_seeds_safe"] else 1,
        -summary["safe_seed_count"],
        summary["worst_seed_total_cost"],
        summary["mean_total_cost"],
    )


def evaluate_stage(
    stage: str,
    jobs: list[Job],
    *,
    evaluation_cells: bool = False,
    immutable: bool = False,
) -> dict[str, Any]:
    output_path = OUT_ROOT / (
        f"{stage}_eh_results.json"
        if evaluation_cells
        else f"{stage}_results.json"
    )
    if immutable and output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    for job in jobs:
        if not validate_completed_job(job):
            raise RuntimeError(f"missing completed model: {job.label}")

    result: dict[str, Any] = {
        "stage": stage,
        "evaluation_cells": "e-h" if evaluation_cells else "a-d",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL_SNAPSHOT),
        "regions": {},
    }
    for region in ("us", "global"):
        baselines = baseline_summaries(
            region,
            evaluation_cells=evaluation_cells,
        )
        status_quo_cost = baselines[
            "Status Quo (local, no deferral)"
        ]["total_cost"]
        region_jobs = [job for job in jobs if job.region == region]
        groups: dict[str, Any] = {}
        for job in region_jobs:
            groups.setdefault(
                job.combo,
                {
                    "candidate": job.candidate,
                    "variant": job.variant,
                    "seeds": {},
                },
            )
            summary = evaluate_job(
                job,
                evaluation_cells=evaluation_cells,
            )
            summary["savings_vs_status_quo_pct"] = float(
                100.0
                * (status_quo_cost - summary["total_cost"])
                / status_quo_cost
            )
            groups[job.combo]["seeds"][str(job.seed)] = summary
        for group in groups.values():
            group["summary"] = aggregate_group(
                group["seeds"],
                status_quo_cost,
            )
        result["regions"][region] = {
            "scenario": str(
                scenario_path(
                    region,
                    evaluation=evaluation_cells,
                ).relative_to(ROOT)
            ).replace("\\", "/"),
            "baselines": baselines,
            "candidates": groups,
            "ranking": sorted(
                groups,
                key=lambda combo: rank_key(
                    groups[combo]["summary"]
                ),
            ),
        }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def evaluate_budget_curve(jobs: list[Job]) -> dict[str, Any]:
    """Evaluate the selected joint config at three matched-seed budgets."""
    for job in jobs:
        if not validate_completed_job(job):
            raise RuntimeError(f"missing completed model: {job.label}")
    result: dict[str, Any] = {
        "stage": "budget_scaling",
        "evaluation_cells": "a-d",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL_SNAPSHOT),
        "regions": {},
    }
    for region in ("us", "global"):
        baselines = baseline_summaries(region)
        status_quo_cost = baselines[
            "Status Quo (local, no deferral)"
        ]["total_cost"]
        groups: dict[str, Any] = {}
        for job in [item for item in jobs if item.region == region]:
            key = str(job.timesteps)
            groups.setdefault(
                key,
                {
                    "timesteps": job.timesteps,
                    "candidate": job.candidate,
                    "variant": job.variant,
                    "seeds": {},
                },
            )
            summary = evaluate_job(job)
            summary["savings_vs_status_quo_pct"] = float(
                100.0
                * (status_quo_cost - summary["total_cost"])
                / status_quo_cost
            )
            groups[key]["seeds"][str(job.seed)] = summary
        for group in groups.values():
            group["summary"] = aggregate_group(
                group["seeds"],
                status_quo_cost,
            )
        ranking = sorted(
            groups,
            key=lambda key: (
                *rank_key(groups[key]["summary"]),
                groups[key]["timesteps"],
            ),
        )
        result["regions"][region] = {
            "scenario": str(
                scenario_path(region).relative_to(ROOT)
            ).replace("\\", "/"),
            "baselines": baselines,
            "budgets": groups,
            "ranking": ranking,
        }
    path = OUT_ROOT / "budget_curve_results.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_budget_selection(results: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for region in ("us", "global"):
        winner = results["regions"][region]["ranking"][0]
        group = results["regions"][region]["budgets"][winner]
        selected[region] = [
            {
                "candidate": group["candidate"],
                "variant": group["variant"],
                "timesteps": group["timesteps"],
                "summary": group["summary"],
            }
        ]
    selection = {
        "stage": "budget",
        "results_sha256": canonical_hash(
            {
                key: value
                for key, value in results.items()
                if key != "generated_at_utc"
            }
        ),
        "selected": selected,
        "selection_rule": CONFIG["budget_scaling"]["selection_order"],
    }
    path = OUT_ROOT / "budget_selection.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != selection:
            raise RuntimeError("budget selection is frozen and would change")
    else:
        path.write_text(
            json.dumps(selection, indent=2),
            encoding="utf-8",
        )
    return selection


def write_selection(
    stage: str,
    results: dict[str, Any],
    count: int,
) -> dict[str, Any]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for region in ("us", "global"):
        candidates = results["regions"][region]["candidates"]
        ranking = results["regions"][region]["ranking"][:count]
        selected[region] = [
            {
                "combo": combo,
                "candidate": candidates[combo]["candidate"],
                "variant": candidates[combo]["variant"],
                "summary": candidates[combo]["summary"],
            }
            for combo in ranking
        ]
    selection = {
        "stage": stage,
        "results_sha256": canonical_hash(
            {
                key: value
                for key, value in results.items()
                if key != "generated_at_utc"
            }
        ),
        "selected": selected,
        "selection_rule": CONFIG["selection"]["lexicographic_order"],
    }
    path = OUT_ROOT / f"{stage}_selection.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != selection:
            raise RuntimeError(
                f"{stage} selection is frozen and would change"
            )
    else:
        path.write_text(
            json.dumps(selection, indent=2),
            encoding="utf-8",
        )
    return selection


def optimizer_ci(
    joint: list[float],
    baseline_cost: float,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if len(joint) < 2:
        raise ValueError("optimizer interval requires multiple seeds")
    improvement = baseline_cost - np.asarray(joint, dtype=np.float64)
    rng = np.random.default_rng(0)
    sample_indices = rng.integers(
        0,
        len(improvement),
        size=(20_000, len(improvement)),
    )
    bootstrap = improvement[sample_indices].mean(axis=1)
    return {
        "interpretation": (
            "Positive USD means joint PPO costs less than deterministic Status "
            "Quo. The interval describes optimizer-seed variability only."
        ),
        "mean_improvement_usd": float(improvement.mean()),
        "optimizer_bootstrap_ci95_usd": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "all_seed_improvements_positive": bool(
            np.all(improvement > 0.0)
        ),
    }


def full_gate(
    full_results: dict[str, Any],
    *,
    evaluation_cells: bool = False,
) -> dict[str, Any]:
    gate = {
        "evaluation_cells": "e-h" if evaluation_cells else "a-d",
        "regions": {},
    }
    for region in ("us", "global"):
        combo = full_results["regions"][region]["ranking"][0]
        group = full_results["regions"][region]["candidates"][combo]
        joint_costs = [
            group["seeds"][seed]["total_cost"]
            for seed in sorted(group["seeds"], key=int)
        ]
        status_quo = full_results["regions"][region]["baselines"][
            "Status Quo (local, no deferral)"
        ]
        comparison = optimizer_ci(
            joint_costs,
            float(status_quo["total_cost"]),
        )
        passed = bool(
            group["summary"]["all_seeds_safe"]
            and comparison["optimizer_bootstrap_ci95_usd"][0]
            > 0.0
        )
        gate["regions"][region] = {
            "combo": combo,
            "candidate_summary": group["summary"],
            "primary_baseline": {
                "name": "Status Quo (local, no deferral)",
                "summary": status_quo,
            },
            "optimizer_comparison": comparison,
            "passed": passed,
        }
    gate["passed"] = bool(
        all(
            gate["regions"][region]["passed"]
            for region in ("us", "global")
        )
    )
    return gate


def write_manifest() -> None:
    entries = []
    for path in sorted(MODEL_ROOT.glob("**/model.zip")):
        complete = path.parent / "complete.json"
        if not complete.exists():
            raise RuntimeError(f"missing completion record: {path}")
        record = json.loads(complete.read_text(encoding="utf-8"))
        if record["model_sha256"] != sha256(path):
            raise RuntimeError(f"model hash mismatch: {path}")
        entries.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "job_fingerprint": record["job_fingerprint"],
            }
        )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL_SNAPSHOT),
        "model_count": len(entries),
        "models": entries,
    }
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    (MODEL_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def preflight() -> None:
    validate_config()
    for script in (
        "scripts/smoke_test_demand_charge.py",
        "scripts/smoke_test_ppo_v3.py",
        "scripts/preflight_energy_model_v2.py",
    ):
        subprocess.run(
            [sys.executable, script],
            cwd=ROOT,
            check=True,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    checks = {}
    for region in ("us", "global"):
        env = make_recovery_env(
            scenario_path(region),
            seed=201,
            peak_penalty_weight=ALPHA,
            service_backlog_weight=700.0,
            batch_completion_weight=1000.0,
            reward_scale=float(CONFIG["environment"]["reward_scale"]),
            subtract_idle_cost=True,
            urgency_potential_weight=0.0,
            domain_randomization=True,
            deadline_bucket_edges=DEADLINE_EDGES,
        )
        obs, _ = env.reset(seed=201)
        if env.observation_space.shape != (81,) or obs.shape != (81,):
            raise RuntimeError(
                f"unexpected v3 observation shape for {region}"
            )
        if not np.isfinite(obs).all():
            raise RuntimeError(f"non-finite v3 observation for {region}")
        if not np.allclose(obs[-2:], [0.0, 1.0]):
            raise RuntimeError(
                f"episode context missing at reset for {region}"
            )
        checks[region] = {
            "max_steps": env.max_steps,
            "observation_shape": list(env.observation_space.shape),
            "action_shape": list(env.action_space.shape),
            "computed_economic_floor": (
                env.computed_economic_penalty_floor
            ),
            "evaluation_weights": {
                "service": env.backlog_weight,
                "batch": env.batch_completion_weight,
            },
            "training_weights": {
                "service": env.reward_service_backlog_weight,
                "batch": env.reward_batch_completion_weight,
            },
        }
    write_protocol()
    assert_provenance()
    (OUT_ROOT / "preflight.json").write_text(
        json.dumps(checks, indent=2),
        encoding="utf-8",
    )
    print(
        f"PPO v3 protocol ready: {PROTOCOL_SNAPSHOT}",
        flush=True,
    )


def run_stage(stage: str, jobs: list[Job], workers: int) -> dict[str, Any]:
    assert_provenance()
    train_all(jobs, workers)
    results = evaluate_stage(stage, jobs)
    if stage == "round1":
        write_selection(
            stage,
            results,
            int(CONFIG["round_1"]["promote_per_region"]),
        )
    elif stage == "round2":
        write_selection(stage, results, 1)
    elif stage == "full":
        gate = full_gate(results)
        (OUT_ROOT / "full_gate.json").write_text(
            json.dumps(gate, indent=2),
            encoding="utf-8",
        )
    write_manifest()
    return results


def run_budget_scaling(workers: int) -> dict[str, Any]:
    """Train the short/long points, select a budget, then reach ten seeds."""
    assert_provenance()
    train_all(budget_training_jobs(), workers)
    curve = evaluate_budget_curve(budget_comparison_jobs())
    write_budget_selection(curve)
    selected_jobs = selected_budget_jobs()
    train_all(selected_jobs, workers)
    selected_results = evaluate_stage(
        "budget_selected",
        selected_jobs,
    )
    gate = full_gate(selected_results)
    (OUT_ROOT / "budget_selected_gate.json").write_text(
        json.dumps(gate, indent=2),
        encoding="utf-8",
    )
    write_manifest()
    return selected_results


def final_evaluation() -> dict[str, Any]:
    assert_provenance()
    jobs = selected_budget_jobs()
    results = evaluate_stage(
        "budget_selected",
        jobs,
        evaluation_cells=True,
        immutable=True,
    )
    transfer = {
        "interpretation": (
            "Post-selection descriptive transfer check only. Cells e-h were "
            "already exposed by frozen v2 and are not fresh confirmatory data."
        ),
        "headline_eligible": False,
        "results_sha256": canonical_hash(
            {
                key: value
                for key, value in results.items()
                if key != "generated_at_utc"
            }
        ),
        "regions": {
            region: {
                "ranking": results["regions"][region]["ranking"],
                "candidates": results["regions"][region]["candidates"],
                "primary_baseline": results["regions"][region]["baselines"][
                    "Status Quo (local, no deferral)"
                ],
            }
            for region in ("us", "global")
        },
    }
    path = OUT_ROOT / "final_eh_transfer.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != transfer:
            raise RuntimeError("one-time e-h transfer check would change")
    else:
        path.write_text(
            json.dumps(transfer, indent=2),
            encoding="utf-8",
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=[
            "preflight",
            "round1",
            "round2",
            "full",
            "budget",
            "final",
            "all",
        ],
        default="all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 16),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    if args.phase in ("preflight", "all"):
        preflight()
    if args.phase in ("round1", "all"):
        run_stage("round1", round1_jobs(), args.workers)
    if args.phase in ("round2", "all"):
        run_stage("round2", round2_jobs(), args.workers)
    if args.phase in ("full", "all"):
        run_stage("full", full_jobs(), args.workers)
    if args.phase in ("budget", "all"):
        run_budget_scaling(args.workers)
    if args.phase in ("final", "all"):
        final_evaluation()


if __name__ == "__main__":
    main()
