"""Run the post-v3 hard-safety PPO v4 campaign.

Phases:
    python scripts/run_safety_campaign_v4.py --phase preflight
    python scripts/run_safety_campaign_v4.py --phase replay --workers 16
    python scripts/run_safety_campaign_v4.py --phase short --workers 16
    python scripts/run_safety_campaign_v4.py --phase medium --workers 16
    python scripts/run_safety_campaign_v4.py --phase full --workers 16
    python scripts/run_safety_campaign_v4.py --phase final --workers 16
    python scripts/run_safety_campaign_v4.py --phase all --workers 16
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
from env.data_loader import load_scenario  # noqa: E402
from env.reward import RewardConfig  # noqa: E402
from env.safety_layer import SafetyConfig, SafetyInfeasibleError  # noqa: E402
from evaluate import compute_summary  # noqa: E402
from scripts.run_reward_sweep_v3 import (  # noqa: E402
    model_path as v3_model_path,
    selected_budget_jobs,
    vecnormalize_path as v3_vecnormalize_path,
)
from train_v3 import DEFAULT_DEADLINE_BUCKET_EDGES  # noqa: E402
from train_v4 import make_safe_env  # noqa: E402

PROTOCOL_PATH = ROOT / "env" / "protocols" / "v4_safety.yaml"
CONFIG = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))

MODEL_ROOT = ROOT / "models" / "ppo_v4_safety"
LOG_ROOT = ROOT / "logs" / "ppo_v4_safety"
OUT_ROOT = ROOT / "output" / "ppo_v4_safety"
PROTOCOL_SNAPSHOT = OUT_ROOT / "protocol.json"

V3_OUTPUT_ROOT = ROOT / "output" / "ppo_v3_reward_sweep"
V3_PROTOCOL_SNAPSHOT = V3_OUTPUT_ROOT / "protocol.json"
V3_BUDGET_SELECTION = V3_OUTPUT_ROOT / "budget_selection.json"
V3_MODEL_MANIFEST = ROOT / "models" / "ppo_v3_reward_sweep" / "manifest.json"
V3_PROTOCOL = json.loads(V3_PROTOCOL_SNAPSHOT.read_text(encoding="utf-8"))
V3_PEAK_PENALTY_WEIGHT = float(
    V3_PROTOCOL["protocol"]["environment"]["peak_penalty_weight"]
)

PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "torch",
    "gymnasium",
    "stable-baselines3",
    "scipy",
    "PyYAML",
)
SOURCE_FILES = [
    "baselines.py",
    "evaluate.py",
    "train_v3.py",
    "train_v4.py",
    "env/data_loader.py",
    "env/dc_site.py",
    "env/multi_dc_env.py",
    "env/power_model.py",
    "env/reward.py",
    "env/safe_multi_dc_env.py",
    "env/safety_layer.py",
    "env/workload_generator.py",
    "env/protocols/v4_safety.yaml",
    "env/protocols/v3_reward_sweep.yaml",
    "env/protocols/v2_2025.yaml",
    "env/protocol.py",
    "scripts/preflight_energy_model_v2.py",
    "scripts/compute_qp_optimum.py",
    "scripts/run_reward_sweep_v3.py",
    "scripts/run_safety_campaign_v4.py",
    "scripts/smoke_test_demand_charge.py",
    "scripts/smoke_test_ppo_v3.py",
    "scripts/smoke_test_safety_v4.py",
    "requirements.txt",
    "env/scenarios/us_model_v2_2025.yaml",
    "env/scenarios/us_model_eh_v2_2025.yaml",
    "env/scenarios/global_model_v2_2025.yaml",
    "env/scenarios/global_model_eh_v2_2025.yaml",
]
ARCHIVED_INPUTS = {
    "v3_protocol": V3_PROTOCOL_SNAPSHOT,
    "v3_budget_selection": V3_BUDGET_SELECTION,
    "v3_model_manifest": V3_MODEL_MANIFEST,
}
SMOKE_TESTS = (
    "scripts/smoke_test_safety_v4.py",
    "scripts/smoke_test_demand_charge.py",
    "scripts/smoke_test_ppo_v3.py",
    "scripts/preflight_energy_model_v2.py",
)

PRIMARY_MODE = "safety_only"
ABLATION_MODE = "safety_plus_negative_demand_flush"
PRIMARY_BASELINE_NAME = "Status Quo (local, no deferral)"
DEVELOPMENT_LABEL = "a-d"
TRANSFER_LABEL = "e-h"
DEADLINE_EDGES = tuple(int(edge) for edge in DEFAULT_DEADLINE_BUCKET_EDGES)
EVALUATION_SEED = 42
SERVICE_EQUALITY_TOL = 1e-9
BATCH_EQUALITY_TOL = 1e-9
TERMINAL_TOL = 1e-8
TRANSPORT_TOL = 1e-8
PROMOTION_CHECK_KEYS = [
    "service_completion_equals_1",
    "batch_completion_equals_1",
    "total_expired_equals_0",
    "terminal_batch_pool_lte_1e_8",
    "terminal_service_backlog_lte_1e_8",
    "safety_infeasibility_certificates_equals_0",
    "transport_conservation_error_lte_1e-8",
]


@dataclass(frozen=True)
class Job:
    stage: str
    region: str
    seed: int
    timesteps: int

    @property
    def combo(self) -> str:
        return str(selected_config(self.region)["source_combo"])

    @property
    def label(self) -> str:
        return f"{self.stage}:{self.region}:{self.combo}:s{self.seed}"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


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
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


def package_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in PACKAGE_NAMES
    }


def source_hashes() -> dict[str, str]:
    return {path: sha256(ROOT / path) for path in SOURCE_FILES}


def data_hashes() -> dict[str, str]:
    return {
        to_rel(path): sha256(path)
        for path in sorted((ROOT / "data").rglob("*"))
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}
    }


def subprocess_env() -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_job_lock(path: Path) -> int:
    """Acquire an exclusive lock, recovering only provably stale locks."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        age = time.time() - path.stat().st_mtime
        try:
            payload = read_json(path)
            owner_pid = int(payload["pid"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            if age < 300.0:
                raise RuntimeError(
                    f"job lock exists but is too new to classify as stale: {path}"
                )
            owner_pid = -1
        if _pid_is_alive(owner_pid):
            raise RuntimeError(
                f"job is already active under PID {owner_pid}: {path}"
            )
        path.unlink()
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    payload = json.dumps(
        {"pid": os.getpid(), "created_at_utc": now_utc()}
    ).encode("utf-8")
    os.write(fd, payload)
    os.fsync(fd)
    return fd


def selected_config(region: str) -> dict[str, Any]:
    configs = CONFIG["selected_configs"]
    if region not in configs:
        raise KeyError(f"unknown region: {region}")
    return dict(configs[region])


def scenario_path(region: str, *, evaluation_cells: bool = False) -> Path:
    config = selected_config(region)
    key = "descriptive_transfer_scenario" if evaluation_cells else "scenario"
    return ROOT / str(config[key])


def replay_source_jobs() -> list[Any]:
    jobs = selected_budget_jobs()
    counts = {
        region: len([job for job in jobs if job.region == region])
        for region in ("us", "global")
    }
    if counts != {"us": 10, "global": 10}:
        raise RuntimeError(
            f"expected 10 selected v3 jobs per region, got {counts}"
        )
    expected_seeds = [int(seed) for seed in CONFIG["replay"]["selected_v3_seeds"]]
    for region in ("us", "global"):
        region_jobs = sorted(
            [job for job in jobs if job.region == region],
            key=lambda item: int(item.seed),
        )
        actual_seeds = [int(job.seed) for job in region_jobs]
        if actual_seeds != expected_seeds:
            raise RuntimeError(
                f"selected v3 seeds mismatch for {region}: {actual_seeds}"
            )
        config = selected_config(region)
        expected_combo = str(config["source_combo"])
        if any(str(job.combo) != expected_combo for job in region_jobs):
            raise RuntimeError(
                f"selected v3 combo mismatch for {region}: expected {expected_combo}"
            )
        expected_budget = int(config["final_budget"])
        if any(int(job.timesteps) != expected_budget for job in region_jobs):
            raise RuntimeError(
                f"selected v3 budget mismatch for {region}: expected {expected_budget}"
            )
    return jobs


def short_jobs() -> list[Job]:
    section = CONFIG["stages"]["short"]
    return [
        Job(
            stage="short",
            region=region,
            seed=int(seed),
            timesteps=int(section["timesteps"]),
        )
        for region in ("us", "global")
        for seed in section["seeds"]
    ]


def medium_jobs() -> list[Job]:
    section = CONFIG["stages"]["medium"]
    return [
        Job(
            stage="medium",
            region=region,
            seed=int(seed),
            timesteps=int(section["timesteps"]),
        )
        for region in ("us", "global")
        for seed in section["seeds"]
    ]


def full_jobs() -> list[Job]:
    section = CONFIG["stages"]["full"]
    return [
        Job(
            stage="full",
            region=region,
            seed=int(seed),
            timesteps=int(selected_config(region)["final_budget"]),
        )
        for region in ("us", "global")
        for seed in section["seeds"]
    ]


def jobs_for_stage(stage: str) -> list[Job]:
    if stage == "short":
        return short_jobs()
    if stage == "medium":
        return medium_jobs()
    if stage == "full":
        return full_jobs()
    raise ValueError(f"unsupported stage: {stage}")


def safety_config_for_mode(mode: str) -> SafetyConfig:
    if mode not in {PRIMARY_MODE, ABLATION_MODE}:
        raise ValueError(f"unsupported safety mode: {mode}")
    safety = CONFIG["safety"]
    return SafetyConfig(
        service_envelope_total=float(safety["service_envelope_total"]),
        batch_arrival_envelope_total=float(
            safety["batch_arrival_envelope_total"]
        ),
        future_fleet_capacity_total=float(
            safety["future_fleet_capacity_total"]
        ),
        envelope_id=str(safety["envelope_id"]),
        envelope_scope=str(safety["envelope_scope"]),
        negative_demand_flush=(mode == ABLATION_MODE),
    )


def reward_config_for_region(region: str) -> RewardConfig:
    reward = selected_config(region)["reward"]
    return RewardConfig(
        service_backlog_weight=float(reward["service_backlog_weight"]),
        batch_completion_weight=float(reward["batch_completion_weight"]),
        evaluation_service_backlog_weight=1000.0,
        evaluation_batch_completion_weight=1000.0,
        reward_scale=float(reward["reward_scale"]),
        subtract_idle_cost=bool(reward["subtract_idle_cost"]),
        urgency_potential_weight=float(reward["urgency_potential_weight"]),
    )


def ppo_config_for_region(region: str) -> dict[str, Any]:
    return dict(selected_config(region)["ppo"])


def protocol() -> dict[str, Any]:
    archived = {
        name: {
            "path": to_rel(path),
            "sha256": sha256(path),
        }
        for name, path in ARCHIVED_INPUTS.items()
    }
    return {
        "protocol": CONFIG,
        "protocol_config": to_rel(PROTOCOL_PATH),
        "protocol_config_sha256": sha256(PROTOCOL_PATH),
        "frozen_at_utc": now_utc(),
        "job_counts": {
            "replay": len(replay_source_jobs()),
            "short": len(short_jobs()),
            "medium": len(medium_jobs()),
            "full": len(full_jobs()),
        },
        "controller_scope": {
            "primary_training_mode": PRIMARY_MODE,
            "replay_ablation_mode": ABLATION_MODE,
            "descriptive_transfer_mode": PRIMARY_MODE,
        },
        "archived_v3_inputs": archived,
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
            "packages": package_versions(),
            "source_sha256": source_hashes(),
            "data_sha256": data_hashes(),
        },
    }


def write_protocol() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    current = protocol()
    if PROTOCOL_SNAPSHOT.exists():
        existing = read_json(PROTOCOL_SNAPSHOT)
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
                "v4 protocol snapshot differs from current source/data/package "
                "or archived v3 inputs; refusing to mix experiments"
            )
        return
    atomic_write_json(PROTOCOL_SNAPSHOT, current)


def assert_provenance() -> None:
    if not PROTOCOL_SNAPSHOT.exists():
        raise RuntimeError("run --phase preflight before replay or training")
    frozen = read_json(PROTOCOL_SNAPSHOT)
    provenance = frozen["provenance"]
    if sys.version != provenance["python"]:
        raise RuntimeError("Python runtime changed after v4 protocol freeze")
    for package, expected in provenance["packages"].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(
                f"package changed after v4 freeze: {package} {actual} != {expected}"
            )
    for path, expected in provenance["source_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"source changed after v4 freeze: {path}")
    for path, expected in provenance["data_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"data changed after v4 freeze: {path}")
    for archived in frozen["archived_v3_inputs"].values():
        archived_path = ROOT / archived["path"]
        if sha256(archived_path) != archived["sha256"]:
            raise RuntimeError(
                f"archived v3 input changed after v4 freeze: {archived['path']}"
            )


def validate_config() -> None:
    parent_commit = str(CONFIG["parent_git_commit"])
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent_commit, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if len(parent_commit) < 7 or ancestry.returncode != 0:
        raise ValueError(
            "the committed v3 checkpoint must be an ancestor of v4 HEAD: "
            f"{parent_commit}"
        )
    if CONFIG["replay"]["modes"] != [PRIMARY_MODE, ABLATION_MODE]:
        raise ValueError("v4 replay modes must be safety_only and flush ablation")
    if list(CONFIG["stages"]["short"]["seeds"]) != [301, 302, 303]:
        raise ValueError("short stage must use seeds 301-303")
    if int(CONFIG["stages"]["short"]["timesteps"]) != 151552:
        raise ValueError("short stage must use 151552 timesteps")
    if list(CONFIG["stages"]["medium"]["seeds"]) != [301, 302, 303, 304, 305]:
        raise ValueError("medium stage must use seeds 301-305")
    if int(CONFIG["stages"]["medium"]["timesteps"]) != 301056:
        raise ValueError("medium stage must use 301056 timesteps")
    if list(CONFIG["stages"]["full"]["seeds"]) != [
        301,
        302,
        303,
        304,
        305,
        306,
        307,
        308,
        309,
        310,
    ]:
        raise ValueError("full stage must use seeds 301-310")
    counts = {
        "short": len(short_jobs()),
        "medium": len(medium_jobs()),
        "full": len(full_jobs()),
    }
    if counts != {"short": 6, "medium": 10, "full": 20}:
        raise ValueError(f"unexpected v4 job counts: {counts}")
    for region in ("us", "global"):
        ppo = ppo_config_for_region(region)
        reward = reward_config_for_region(region)
        if not math.isclose(float(ppo["gamma"]), 1.0, abs_tol=1e-12):
            raise ValueError(f"v4 requires gamma=1 for {region}")
        n_steps = int(ppo["n_steps"])
        if n_steps != 2048:
            raise ValueError(f"v4 requires n_steps=2048 for {region}")
        reward.validate(0.0)
        stage_timesteps = [
            int(CONFIG["stages"]["short"]["timesteps"]),
            int(CONFIG["stages"]["medium"]["timesteps"]),
            int(selected_config(region)["final_budget"]),
        ]
        if any(value <= 0 or value % n_steps != 0 for value in stage_timesteps):
            raise ValueError(
                f"stage timesteps must be positive and rollout-aligned for {region}"
            )
    budgets = {
        region: int(selected_config(region)["final_budget"])
        for region in ("us", "global")
    }
    if budgets != {"us": 151552, "global": 1003520}:
        raise ValueError(f"unexpected v4 full budgets: {budgets}")
    safety_config_for_mode(PRIMARY_MODE).validate(4)
    if not math.isclose(
        safety_config_for_mode(
            PRIMARY_MODE
        ).guaranteed_carried_batch_capacity,
        float(CONFIG["safety"]["guaranteed_carried_batch_capacity"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("configured carried-batch guarantee is inconsistent")
    required_checks = list(
        CONFIG["promotion_gate"]["require_all_seeds"]
    )
    if required_checks != PROMOTION_CHECK_KEYS:
        raise ValueError(
            "promotion gate keys differ from executable checks: "
            f"{required_checks} != {PROMOTION_CHECK_KEYS}"
        )
    probe_checks = gate_seed(
        {
            "service_completion": 1.0,
            "batch_completion": 1.0,
            "total_batch_expired": 0.0,
            "terminal_batch_pool": 0.0,
            "terminal_service_backlog": 0.0,
            "safety": {
                "infeasibility_certificates": 0,
                "max_transport_conservation_error": 0.0,
            },
        }
    )["checks"]
    if list(probe_checks) != PROMOTION_CHECK_KEYS:
        raise ValueError(
            "runtime gate checks differ from protocol keys: "
            f"{list(probe_checks)} != {PROMOTION_CHECK_KEYS}"
        )
    replay_source_jobs()


def job_dir(job: Job) -> Path:
    return MODEL_ROOT / job.stage / job.region / job.combo / f"s{job.seed}"


def model_path(job: Job) -> Path:
    return job_dir(job) / "model.zip"


def diagnostics_path(job: Job) -> Path:
    return job_dir(job) / "training.json"


def vecnormalize_path(job: Job) -> Path | None:
    if bool(ppo_config_for_region(job.region)["normalize_observations"]):
        return job_dir(job) / "vecnormalize.pkl"
    return None


def completion_path(job: Job) -> Path:
    return model_path(job).with_suffix(".complete.json")


def log_path(job: Job) -> Path:
    return LOG_ROOT / job.stage / job.region / f"{job.combo}_s{job.seed}.log"


def evaluation_record_path(
    stage: str,
    region: str,
    combo: str,
    seed: int,
    *,
    mode: str,
    evaluation_cells: bool,
) -> Path:
    scope = TRANSFER_LABEL if evaluation_cells else DEVELOPMENT_LABEL
    return (
        OUT_ROOT
        / "_eval_records"
        / stage
        / scope
        / mode
        / region
        / combo
        / f"s{seed}.json"
    )


def replay_record_path(
    region: str,
    combo: str,
    seed: int,
    *,
    mode: str,
) -> Path:
    return evaluation_record_path(
        "replay",
        region,
        combo,
        seed,
        mode=mode,
        evaluation_cells=False,
    )


def baseline_cache_path(
    region: str,
    *,
    mode: str,
    evaluation_cells: bool,
) -> Path:
    scope = TRANSFER_LABEL if evaluation_cells else DEVELOPMENT_LABEL
    return OUT_ROOT / "_baselines" / scope / mode / f"{region}.json"


def job_spec(job: Job) -> dict[str, Any]:
    selected = selected_config(job.region)
    ppo = ppo_config_for_region(job.region)
    reward = reward_config_for_region(job.region)
    spec = {
        "stage": job.stage,
        "region": job.region,
        "combo": job.combo,
        "seed": int(job.seed),
        "timesteps": int(job.timesteps),
        "protocol_sha256": sha256(PROTOCOL_SNAPSHOT),
        "train_scenario": to_rel(scenario_path(job.region)),
        "transfer_scenario": to_rel(
            scenario_path(job.region, evaluation_cells=True)
        ),
        "source_combo": str(selected["source_combo"]),
        "ppo": {
            "learning_rate": float(ppo["learning_rate"]),
            "learning_rate_schedule": str(ppo["learning_rate_schedule"]),
            "n_steps": int(ppo["n_steps"]),
            "batch_size": int(ppo["batch_size"]),
            "n_epochs": int(ppo["n_epochs"]),
            "gamma": float(ppo["gamma"]),
            "gae_lambda": float(ppo["gae_lambda"]),
            "target_kl": (
                None if ppo["target_kl"] is None else float(ppo["target_kl"])
            ),
            "normalize_observations": bool(ppo["normalize_observations"]),
            "net_arch": [int(width) for width in ppo["net_arch"]],
        },
        "reward": reward.as_dict(),
        "safety": safety_config_for_mode(PRIMARY_MODE).as_dict(),
        "peak_penalty_weight": V3_PEAK_PENALTY_WEIGHT,
        "deadline_bucket_edges": list(DEADLINE_EDGES),
        "controller_scope": {
            "training": PRIMARY_MODE,
            "replay_ablation": ABLATION_MODE,
        },
    }
    spec["job_fingerprint"] = canonical_hash(spec)
    return spec


def validate_completed_job(job: Job) -> bool:
    paths = [model_path(job), diagnostics_path(job), completion_path(job)]
    stats = vecnormalize_path(job)
    if stats is not None:
        paths.append(stats)
    exists = [path.exists() for path in paths]
    if not any(exists):
        return False
    if not all(exists):
        raise RuntimeError(
            f"incomplete prior output for {job.label}: "
            + ", ".join(str(path) for path in paths)
        )
    record = read_json(completion_path(job))
    expected = job_spec(job)
    if record.get("job_fingerprint") != expected["job_fingerprint"]:
        raise RuntimeError(f"job fingerprint mismatch: {job.label}")
    if record.get("model_sha256") != sha256(model_path(job)):
        raise RuntimeError(f"model hash mismatch: {job.label}")
    if record.get("diagnostics_sha256") != sha256(diagnostics_path(job)):
        raise RuntimeError(f"diagnostics hash mismatch: {job.label}")
    if stats is not None and record.get("vecnormalize_sha256") != sha256(stats):
        raise RuntimeError(f"VecNormalize hash mismatch: {job.label}")
    model = PPO.load(model_path(job))
    if model.num_timesteps != int(job.timesteps):
        raise RuntimeError(
            f"timestep mismatch for {job.label}: "
            f"{model.num_timesteps} != {job.timesteps}"
        )
    return True


def training_command(job: Job, temp_dir: Path, python: Path) -> list[str]:
    ppo = ppo_config_for_region(job.region)
    reward = reward_config_for_region(job.region)
    command = [
        str(python),
        "train_v4.py",
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
        str(float(ppo["learning_rate"])),
        "--lr-schedule",
        str(ppo["learning_rate_schedule"]),
        "--n-steps",
        str(int(ppo["n_steps"])),
        "--batch-size",
        str(int(ppo["batch_size"])),
        "--n-epochs",
        str(int(ppo["n_epochs"])),
        "--gamma",
        str(float(ppo["gamma"])),
        "--gae-lambda",
        str(float(ppo["gae_lambda"])),
        "--net-arch",
        *[str(int(width)) for width in ppo["net_arch"]],
        "--peak-penalty-weight",
        str(V3_PEAK_PENALTY_WEIGHT),
        "--service-backlog-weight",
        str(float(reward.service_backlog_weight)),
        "--batch-completion-weight",
        str(float(reward.batch_completion_weight)),
        "--reward-scale",
        str(float(reward.reward_scale)),
        "--urgency-potential-weight",
        str(float(reward.urgency_potential_weight)),
        "--service-envelope-total",
        str(float(CONFIG["safety"]["service_envelope_total"])),
        "--batch-arrival-envelope-total",
        str(float(CONFIG["safety"]["batch_arrival_envelope_total"])),
        "--future-fleet-capacity-total",
        str(float(CONFIG["safety"]["future_fleet_capacity_total"])),
        "--envelope-id",
        str(CONFIG["safety"]["envelope_id"]),
        "--envelope-scope",
        str(CONFIG["safety"]["envelope_scope"]),
    ]
    if ppo["target_kl"] is not None:
        command += ["--target-kl", str(float(ppo["target_kl"]))]
    if reward.subtract_idle_cost:
        command.append("--subtract-idle-cost")
    if bool(ppo["normalize_observations"]):
        command += [
            "--normalize-observations",
            "--vecnormalize-path",
            str(temp_dir / "vecnormalize.pkl"),
        ]
    return command


def train_one(job: Job, python: Path) -> tuple[str, bool, float]:
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
    command = training_command(job, temp_dir, python)
    lock = final_dir / "training.lock"
    lock_fd: int | None = None
    started = time.time()
    try:
        lock_fd = acquire_job_lock(lock)
        with log.open("w", encoding="utf-8") as handle:
            handle.write(
                f"# {job.label}\n"
                f"# job: {json.dumps(job_spec(job), sort_keys=True)}\n"
                f"# command: {' '.join(command)}\n"
                f"# started: {now_utc()}\n\n"
            )
            handle.flush()
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=subprocess_env(),
            )
        elapsed = time.time() - started
        temp_model = temp_dir / "model.zip"
        temp_diagnostics = temp_dir / "training.json"
        temp_stats = temp_dir / "vecnormalize.pkl"
        need_stats = vecnormalize_path(job) is not None
        if (
            result.returncode != 0
            or not temp_model.exists()
            or not temp_diagnostics.exists()
            or (need_stats and not temp_stats.exists())
        ):
            if temp_diagnostics.exists():
                failure = final_dir / "failure.json"
                shutil.copyfile(temp_diagnostics, failure)
            return job.label, False, elapsed
        model = PPO.load(temp_model)
        if model.num_timesteps != int(job.timesteps):
            raise RuntimeError(f"trained timestep mismatch for {job.label}")
        assert_provenance()
        os.replace(temp_model, model_path(job))
        os.replace(temp_diagnostics, diagnostics_path(job))
        final_stats = vecnormalize_path(job)
        if final_stats is not None:
            os.replace(temp_stats, final_stats)
        record = {
            **job_spec(job),
            "completed_at_utc": now_utc(),
            "elapsed_seconds": elapsed,
            "model_path": to_rel(model_path(job)),
            "model_bytes": model_path(job).stat().st_size,
            "model_sha256": sha256(model_path(job)),
            "diagnostics_sha256": sha256(diagnostics_path(job)),
            "vecnormalize_sha256": (
                sha256(final_stats) if final_stats is not None else None
            ),
            "log_path": to_rel(log),
            "log_sha256": sha256(log),
        }
        atomic_write_json(completion_path(job), record)
        return job.label, True, elapsed
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            lock.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


def train_all(jobs: list[Job], python: Path, workers: int) -> None:
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_jobs = {
            pool.submit(train_one, job, python): job for job in jobs
        }
        for future in concurrent.futures.as_completed(future_jobs):
            label, ok, elapsed = future.result()
            print(
                f"[{'OK' if ok else 'FAIL'}] {label} ({elapsed / 60:.1f} min)",
                flush=True,
            )
            if not ok:
                failures.append(label)
    if failures:
        raise RuntimeError("training failed: " + ", ".join(failures))


def baseline_fingerprint(
    region: str,
    *,
    mode: str,
    evaluation_cells: bool,
) -> str:
    payload = {
        "region": region,
        "mode": mode,
        "evaluation_cells": evaluation_cells,
        "scenario": to_rel(scenario_path(region, evaluation_cells=evaluation_cells)),
        "reward": reward_config_for_region(region).as_dict(),
        "safety": safety_config_for_mode(mode).as_dict(),
        "protocol_sha256": sha256(PROTOCOL_SNAPSHOT),
        "seed": EVALUATION_SEED,
        "baselines": [
            PRIMARY_BASELINE_NAME,
            "Round Robin",
            "Drain Immediately",
        ],
    }
    return canonical_hash(payload)


def stage_result_path(stage: str, *, evaluation_cells: bool = False) -> Path:
    if stage == "final":
        return OUT_ROOT / "final_results.json"
    suffix = "_eh_results.json" if evaluation_cells else "_results.json"
    return OUT_ROOT / f"{stage}{suffix}"


def gate_path(stage: str) -> Path:
    return OUT_ROOT / f"{stage}_gate.json"


def policy_action_stats(values: Any) -> tuple[int, int]:
    array = np.asarray(values, dtype=np.float64)
    return array.size, int(np.count_nonzero(np.abs(array) >= 2.95))


def enrich_summary(summary: dict[str, Any]) -> dict[str, Any]:
    summary = dict(summary)
    demand = float(summary.get("service_demand_total", 0.0))
    served = float(summary.get("service_served_total", 0.0))
    batch_arrival = float(summary.get("batch_arrival_total", 0.0))
    batch_completed = float(summary.get("batch_completed_total", 0.0))
    summary["service_completion"] = (
        served / demand if demand > 0.0 else 1.0
    )
    summary["batch_completion"] = (
        batch_completed / batch_arrival if batch_arrival > 0.0 else 1.0
    )
    summary["terminal_service_backlog"] = float(
        summary.get("terminal_backlog", 0.0)
    )
    if "safety" not in summary:
        summary["safety"] = {
            "enabled": False,
            "infeasibility_certificates": 0,
            "max_transport_conservation_error": None,
        }
    return summary


def run_policy_episode(
    *,
    env_factory: Callable[[], Any],
    predict_fn: Callable[..., Any],
    is_sb3: bool,
    vecnormalize_file: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = env_factory()
    history: list[dict[str, Any]] = []
    action_values = 0
    saturated_actions = 0
    if is_sb3 and vecnormalize_file is not None:
        vec_env = DummyVecEnv([env_factory])
        vec_env = VecNormalize.load(vecnormalize_file, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        try:
            vec_env.seed(EVALUATION_SEED)
            obs = vec_env.reset()
            for _ in range(env.max_steps):
                action, _ = predict_fn(obs, deterministic=True)
                total, saturated = policy_action_stats(action)
                action_values += total
                saturated_actions += saturated
                obs, _, dones, infos = vec_env.step(action)
                history.append(infos[0])
                if bool(dones[0]):
                    break
        finally:
            vec_env.close()
            env.close()
    else:
        try:
            obs, _ = env.reset(seed=EVALUATION_SEED)
            while True:
                if is_sb3:
                    action, _ = predict_fn(obs, deterministic=True)
                else:
                    action = predict_fn(obs, env)
                total, saturated = policy_action_stats(action)
                action_values += total
                saturated_actions += saturated
                obs, _, terminated, truncated, info = env.step(action)
                history.append(info)
                if terminated or truncated:
                    break
        finally:
            env.close()
    if len(history) != env.max_steps:
        raise RuntimeError(
            "evaluation length mismatch: "
            f"{len(history)} != {env.max_steps}"
        )
    summary = enrich_summary(compute_summary(history, batch_enabled=True))
    summary["action_saturation_fraction"] = (
        saturated_actions / action_values if action_values else 0.0
    )
    return summary, history


def baseline_summaries(
    region: str,
    *,
    mode: str,
    evaluation_cells: bool = False,
) -> dict[str, Any]:
    path = baseline_cache_path(
        region,
        mode=mode,
        evaluation_cells=evaluation_cells,
    )
    fingerprint = baseline_fingerprint(
        region,
        mode=mode,
        evaluation_cells=evaluation_cells,
    )
    if path.exists():
        existing = read_json(path)
        if existing.get("fingerprint") == fingerprint:
            if existing.get("status") != "ok":
                raise RuntimeError(
                    f"cached baseline failure for {region} {mode}: {path}"
                )
            return existing["baselines"]
    baselines: dict[str, Any] = {}
    safety = safety_config_for_mode(mode)
    reward = reward_config_for_region(region)
    for baseline_cls in (
        StatusQuoPolicy,
        RoundRobinPolicy,
        DrainImmediatelyPolicy,
    ):
        policy = baseline_cls()

        def factory() -> Any:
            return make_safe_env(
                scenario_path(region, evaluation_cells=evaluation_cells),
                seed=EVALUATION_SEED,
                peak_penalty_weight=V3_PEAK_PENALTY_WEIGHT,
                reward_config=reward,
                safety_config=safety,
                domain_randomization=False,
                deadline_bucket_edges=DEADLINE_EDGES,
            )

        try:
            summary, _ = run_policy_episode(
                env_factory=factory,
                predict_fn=policy.predict,
                is_sb3=False,
            )
        except SafetyInfeasibleError as exc:
            atomic_write_json(
                path,
                {
                    "status": "safety_infeasible",
                    "fingerprint": fingerprint,
                    "baseline": policy.name,
                    "certificate": exc.certificate,
                    "generated_at_utc": now_utc(),
                },
            )
            raise RuntimeError(
                f"baseline {policy.name} infeasible in {region} {mode}"
            ) from exc
        baselines[policy.name] = summary
    atomic_write_json(
        path,
        {
            "status": "ok",
            "fingerprint": fingerprint,
            "generated_at_utc": now_utc(),
            "baselines": baselines,
        },
    )
    return baselines


def model_hashes_from_record(
    *,
    model_file: Path,
    completion_record: dict[str, Any],
    diagnostics_file: Path | None = None,
    vecnormalize_file: Path | None = None,
) -> dict[str, Any]:
    hashes = {
        "model_path": to_rel(model_file),
        "model_sha256": sha256(model_file),
        "completion_record_path": to_rel(
            model_file.with_suffix(".complete.json")
            if diagnostics_file is None
            else (model_file.with_suffix(".complete.json"))
        ),
        "job_fingerprint": completion_record.get("job_fingerprint"),
    }
    if diagnostics_file is not None:
        hashes["diagnostics_path"] = to_rel(diagnostics_file)
        hashes["diagnostics_sha256"] = sha256(diagnostics_file)
    if vecnormalize_file is not None:
        hashes["vecnormalize_path"] = to_rel(vecnormalize_file)
        hashes["vecnormalize_sha256"] = sha256(vecnormalize_file)
    for key in (
        "model_sha256",
        "diagnostics_sha256",
        "vecnormalize_sha256",
    ):
        if key in completion_record and completion_record.get(key) != hashes.get(key):
            raise RuntimeError(f"completion-record hash mismatch for {model_file}: {key}")
    return hashes


def evaluation_fingerprint(
    *,
    stage: str,
    region: str,
    combo: str,
    seed: int,
    mode: str,
    evaluation_cells: bool,
    model_hashes: dict[str, Any],
    baselines: dict[str, Any],
) -> str:
    payload = {
        "stage": stage,
        "region": region,
        "combo": combo,
        "seed": seed,
        "mode": mode,
        "evaluation_cells": evaluation_cells,
        "protocol_sha256": sha256(PROTOCOL_SNAPSHOT),
        "scenario": to_rel(scenario_path(region, evaluation_cells=evaluation_cells)),
        "reward": reward_config_for_region(region).as_dict(),
        "safety": safety_config_for_mode(mode).as_dict(),
        "model_hashes": model_hashes,
        "baselines_sha256": canonical_hash(baselines),
        "seed_fixed": EVALUATION_SEED,
    }
    return canonical_hash(payload)


def evaluate_v4_seed(
    job: Job,
    *,
    baselines: dict[str, Any],
    mode: str = PRIMARY_MODE,
    evaluation_cells: bool = False,
) -> dict[str, Any]:
    if mode != PRIMARY_MODE:
        raise ValueError("v4 training stages evaluate only safety_only")
    if not validate_completed_job(job):
        raise RuntimeError(f"missing completed model: {job.label}")
    completion = read_json(completion_path(job))
    vec_file = vecnormalize_path(job)
    hashes = model_hashes_from_record(
        model_file=model_path(job),
        completion_record=completion,
        diagnostics_file=diagnostics_path(job),
        vecnormalize_file=vec_file,
    )
    path = evaluation_record_path(
        job.stage,
        job.region,
        job.combo,
        job.seed,
        mode=mode,
        evaluation_cells=evaluation_cells,
    )
    fingerprint = evaluation_fingerprint(
        stage=job.stage if not evaluation_cells else "final",
        region=job.region,
        combo=job.combo,
        seed=job.seed,
        mode=mode,
        evaluation_cells=evaluation_cells,
        model_hashes=hashes,
        baselines=baselines,
    )
    if path.exists():
        existing = read_json(path)
        if existing.get("evaluation_fingerprint") == fingerprint:
            return existing

    reward = reward_config_for_region(job.region)
    safety = safety_config_for_mode(mode)

    def factory() -> Any:
        return make_safe_env(
            scenario_path(job.region, evaluation_cells=evaluation_cells),
            seed=EVALUATION_SEED,
            peak_penalty_weight=V3_PEAK_PENALTY_WEIGHT,
            reward_config=reward,
            safety_config=safety,
            domain_randomization=False,
            deadline_bucket_edges=DEADLINE_EDGES,
        )

    model = PPO.load(model_path(job))
    try:
        summary, _ = run_policy_episode(
            env_factory=factory,
            predict_fn=model.predict,
            is_sb3=True,
            vecnormalize_file=vec_file,
        )
    except SafetyInfeasibleError as exc:
        record = {
            "status": "safety_infeasible",
            "stage": job.stage if not evaluation_cells else "final",
            "region": job.region,
            "combo": job.combo,
            "seed": int(job.seed),
            "mode": mode,
            "evaluation_cells": TRANSFER_LABEL if evaluation_cells else DEVELOPMENT_LABEL,
            "scenario": to_rel(
                scenario_path(job.region, evaluation_cells=evaluation_cells)
            ),
            "protocol_sha256": sha256(PROTOCOL_SNAPSHOT),
            "evaluation_fingerprint": fingerprint,
            "model_hashes": hashes,
            "baselines": baselines,
            "certificate": exc.certificate,
            "generated_at_utc": now_utc(),
        }
        atomic_write_json(path, record)
        return record

    status_quo_cost = float(baselines[PRIMARY_BASELINE_NAME]["total_cost"])
    record = {
        "status": "ok",
        "stage": job.stage if not evaluation_cells else "final",
        "region": job.region,
        "combo": job.combo,
        "seed": int(job.seed),
        "mode": mode,
        "evaluation_cells": TRANSFER_LABEL if evaluation_cells else DEVELOPMENT_LABEL,
        "scenario": to_rel(
            scenario_path(job.region, evaluation_cells=evaluation_cells)
        ),
        "protocol_sha256": sha256(PROTOCOL_SNAPSHOT),
        "evaluation_fingerprint": fingerprint,
        "model_hashes": hashes,
        "baselines": baselines,
        "summary": summary,
        "savings_vs_status_quo_pct": float(
            100.0 * (status_quo_cost - summary["total_cost"]) / status_quo_cost
        ),
        "generated_at_utc": now_utc(),
    }
    atomic_write_json(path, record)
    return record


def validate_v3_replay_source(job: Any) -> dict[str, Any]:
    model_file = v3_model_path(job)
    record_path = model_file.parent / "complete.json"
    if not model_file.exists() or not record_path.exists():
        raise RuntimeError(f"missing archived v3 model or completion record: {job}")
    record = read_json(record_path)
    if record.get("model_sha256") != sha256(model_file):
        raise RuntimeError(f"archived v3 model hash mismatch: {model_file}")
    vec_file = v3_vecnormalize_path(job)
    if vec_file is not None:
        if not vec_file.exists():
            raise RuntimeError(f"missing archived VecNormalize file: {vec_file}")
        if record.get("vecnormalize_sha256") != sha256(vec_file):
            raise RuntimeError(f"archived v3 VecNormalize hash mismatch: {vec_file}")
    manifest = read_json(V3_MODEL_MANIFEST)
    manifest_map = {row["path"]: row for row in manifest["models"]}
    row = manifest_map.get(to_rel(model_file))
    if row is None:
        raise RuntimeError(f"archived v3 manifest entry missing: {model_file}")
    if row["sha256"] != record["model_sha256"]:
        raise RuntimeError(f"archived v3 manifest hash mismatch: {model_file}")
    return record


def evaluate_v3_replay_seed(
    job: Any,
    *,
    baselines: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    if mode not in {PRIMARY_MODE, ABLATION_MODE}:
        raise ValueError(f"unsupported replay mode: {mode}")
    completion = validate_v3_replay_source(job)
    model_file = v3_model_path(job)
    vec_file = v3_vecnormalize_path(job)
    hashes = {
        "model_path": to_rel(model_file),
        "model_sha256": sha256(model_file),
        "completion_record_path": to_rel(model_file.parent / "complete.json"),
        "completion_record_sha256": sha256(model_file.parent / "complete.json"),
        "job_fingerprint": completion.get("job_fingerprint"),
        "source_protocol_sha256": V3_PROTOCOL["protocol_config_sha256"]
        if "protocol_config_sha256" in V3_PROTOCOL
        else None,
    }
    if vec_file is not None:
        hashes["vecnormalize_path"] = to_rel(vec_file)
        hashes["vecnormalize_sha256"] = sha256(vec_file)
    path = replay_record_path(
        job.region,
        str(job.combo),
        int(job.seed),
        mode=mode,
    )
    fingerprint = evaluation_fingerprint(
        stage="replay",
        region=str(job.region),
        combo=str(job.combo),
        seed=int(job.seed),
        mode=mode,
        evaluation_cells=False,
        model_hashes=hashes,
        baselines=baselines,
    )
    if path.exists():
        existing = read_json(path)
        if existing.get("evaluation_fingerprint") == fingerprint:
            return existing

    reward = reward_config_for_region(str(job.region))
    safety = safety_config_for_mode(mode)

    def factory() -> Any:
        return make_safe_env(
            scenario_path(str(job.region)),
            seed=EVALUATION_SEED,
            peak_penalty_weight=V3_PEAK_PENALTY_WEIGHT,
            reward_config=reward,
            safety_config=safety,
            domain_randomization=False,
            deadline_bucket_edges=DEADLINE_EDGES,
        )

    model = PPO.load(model_file)
    try:
        summary, _ = run_policy_episode(
            env_factory=factory,
            predict_fn=model.predict,
            is_sb3=True,
            vecnormalize_file=vec_file,
        )
    except SafetyInfeasibleError as exc:
        record = {
            "status": "safety_infeasible",
            "stage": "replay",
            "region": str(job.region),
            "combo": str(job.combo),
            "seed": int(job.seed),
            "mode": mode,
            "evaluation_cells": DEVELOPMENT_LABEL,
            "scenario": to_rel(scenario_path(str(job.region))),
            "protocol_sha256": sha256(PROTOCOL_SNAPSHOT),
            "evaluation_fingerprint": fingerprint,
            "model_hashes": hashes,
            "baselines": baselines,
            "certificate": exc.certificate,
            "generated_at_utc": now_utc(),
        }
        atomic_write_json(path, record)
        return record

    status_quo_cost = float(baselines[PRIMARY_BASELINE_NAME]["total_cost"])
    record = {
        "status": "ok",
        "stage": "replay",
        "region": str(job.region),
        "combo": str(job.combo),
        "seed": int(job.seed),
        "mode": mode,
        "evaluation_cells": DEVELOPMENT_LABEL,
        "scenario": to_rel(scenario_path(str(job.region))),
        "protocol_sha256": sha256(PROTOCOL_SNAPSHOT),
        "evaluation_fingerprint": fingerprint,
        "model_hashes": hashes,
        "baselines": baselines,
        "summary": summary,
        "savings_vs_status_quo_pct": float(
            100.0 * (status_quo_cost - summary["total_cost"]) / status_quo_cost
        ),
        "generated_at_utc": now_utc(),
    }
    atomic_write_json(path, record)
    return record


def aggregate_records(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    successful = [records[key] for key in sorted(records, key=int)]
    costs = np.asarray(
        [row["summary"]["total_cost"] for row in successful],
        dtype=np.float64,
    )
    savings = np.asarray(
        [row["savings_vs_status_quo_pct"] for row in successful],
        dtype=np.float64,
    )
    service_completion = np.asarray(
        [row["summary"]["service_completion"] for row in successful],
        dtype=np.float64,
    )
    batch_completion = np.asarray(
        [row["summary"]["batch_completion"] for row in successful],
        dtype=np.float64,
    )
    intervention_rates = np.asarray(
        [
            float(row["summary"]["safety"].get("intervention_rate", 0.0))
            for row in successful
        ],
        dtype=np.float64,
    )
    return {
        "seed_count": len(successful),
        "mean_total_cost": float(costs.mean()),
        "std_total_cost": float(costs.std(ddof=1) if len(costs) > 1 else 0.0),
        "worst_seed_total_cost": float(costs.max()),
        "mean_savings_vs_status_quo_pct": float(savings.mean()),
        "worst_savings_vs_status_quo_pct": float(savings.min()),
        "minimum_service_completion": float(service_completion.min()),
        "minimum_batch_completion": float(batch_completion.min()),
        "maximum_terminal_batch_pool": float(
            max(
                row["summary"].get("terminal_batch_pool", 0.0)
                for row in successful
            )
        ),
        "maximum_terminal_service_backlog": float(
            max(
                row["summary"].get("terminal_service_backlog", 0.0)
                for row in successful
            )
        ),
        "total_expired": float(
            sum(row["summary"].get("total_batch_expired", 0.0) for row in successful)
        ),
        "maximum_transport_conservation_error": float(
            max(
                row["summary"]["safety"].get(
                    "max_transport_conservation_error",
                    0.0,
                )
                for row in successful
            )
        ),
        "safety_infeasibility_certificates": int(
            sum(
                int(
                    row["summary"]["safety"].get(
                        "infeasibility_certificates",
                        0,
                    )
                )
                for row in successful
            )
        ),
        "mean_safety_intervention_rate": float(intervention_rates.mean()),
        "max_projection_l2": float(
            max(
                row["summary"]["safety"].get("max_projection_l2", 0.0)
                for row in successful
            )
        ),
    }


def optimizer_ci_vs_deterministic(
    candidate_costs: list[float],
    baseline_cost: float,
) -> dict[str, Any]:
    if len(candidate_costs) < 2:
        raise ValueError("bootstrap CI requires multiple optimizer seeds")
    costs = np.asarray(candidate_costs, dtype=np.float64)
    improvement = baseline_cost - costs
    rng = np.random.default_rng(0)
    indices = rng.integers(
        0,
        len(improvement),
        size=(20_000, len(improvement)),
    )
    bootstrap = improvement[indices].mean(axis=1)
    return {
        "interpretation": (
            "Positive USD means the candidate costs less than deterministic "
            "Status Quo. The interval reflects optimizer-seed variability only."
        ),
        "mean_improvement_usd": float(improvement.mean()),
        "optimizer_bootstrap_ci95_usd": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "all_seed_improvements_positive": bool(np.all(improvement > 0.0)),
    }


def optimizer_ci_vs_archived_sample(
    candidate_costs: list[float],
    archived_costs: list[float],
) -> dict[str, Any]:
    if len(candidate_costs) < 2 or len(archived_costs) < 2:
        raise ValueError("archived replay bootstrap CI requires multiple seeds")
    candidate = np.asarray(candidate_costs, dtype=np.float64)
    archived = np.asarray(archived_costs, dtype=np.float64)
    rng = np.random.default_rng(1)
    candidate_idx = rng.integers(
        0,
        len(candidate),
        size=(20_000, len(candidate)),
    )
    archived_idx = rng.integers(
        0,
        len(archived),
        size=(20_000, len(archived)),
    )
    bootstrap = archived[archived_idx].mean(axis=1) - candidate[candidate_idx].mean(axis=1)
    return {
        "interpretation": (
            "Positive USD means v4 costs less than the archived v3 safety-only "
            "replay. This is diagnostic, not a promotion gate."
        ),
        "mean_improvement_usd": float(archived.mean() - candidate.mean()),
        "optimizer_bootstrap_ci95_usd": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def gate_seed(summary: dict[str, Any]) -> dict[str, Any]:
    service_completion = float(summary["service_completion"])
    batch_completion = float(summary["batch_completion"])
    total_expired = float(summary.get("total_batch_expired", 0.0))
    terminal_pool = float(summary.get("terminal_batch_pool", 0.0))
    terminal_backlog = float(summary.get("terminal_service_backlog", 0.0))
    safety = summary["safety"]
    max_transport_error = float(
        safety.get("max_transport_conservation_error", math.inf)
    )
    certificates = int(safety.get("infeasibility_certificates", 0))
    checks = {
        "service_completion_equals_1": math.isclose(
            service_completion,
            1.0,
            rel_tol=0.0,
            abs_tol=SERVICE_EQUALITY_TOL,
        ),
        "batch_completion_equals_1": math.isclose(
            batch_completion,
            1.0,
            rel_tol=0.0,
            abs_tol=BATCH_EQUALITY_TOL,
        ),
        "total_expired_equals_0": abs(total_expired) <= SERVICE_EQUALITY_TOL,
        "terminal_batch_pool_lte_1e_8": terminal_pool <= TERMINAL_TOL,
        "terminal_service_backlog_lte_1e_8": terminal_backlog <= TERMINAL_TOL,
        "safety_infeasibility_certificates_equals_0": certificates == 0,
        "transport_conservation_error_lte_1e-8": max_transport_error
        <= TRANSPORT_TOL,
    }
    return {
        "service_completion": service_completion,
        "batch_completion": batch_completion,
        "total_expired": total_expired,
        "terminal_batch_pool": terminal_pool,
        "terminal_service_backlog": terminal_backlog,
        "safety_infeasibility_certificates": certificates,
        "max_transport_conservation_error": max_transport_error,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def load_replay_reference_costs() -> dict[str, list[float]]:
    path = stage_result_path("replay")
    if not path.exists():
        raise RuntimeError("run --phase replay before short/medium/full")
    replay = read_json(path)
    costs: dict[str, list[float]] = {}
    for region in ("us", "global"):
        seeds = replay["regions"][region]["modes"][PRIMARY_MODE]["seeds"]
        costs[region] = [
            float(seeds[key]["summary"]["total_cost"])
            for key in sorted(seeds, key=int)
        ]
    return costs


def build_gate(stage: str, results: dict[str, Any]) -> dict[str, Any]:
    replay_costs = load_replay_reference_costs()
    gate: dict[str, Any] = {
        "stage": stage,
        "evaluation_cells": DEVELOPMENT_LABEL,
        "gate_rule": {
            "service_completion": f"exactly 1 within {SERVICE_EQUALITY_TOL}",
            "batch_completion": f"exactly 1 within {BATCH_EQUALITY_TOL}",
            "total_expired": "exactly 0",
            "terminal_batch_pool": f"<= {TERMINAL_TOL}",
            "terminal_service_backlog": f"<= {TERMINAL_TOL}",
            "safety_infeasibility_certificates": "exactly 0",
            "max_transport_conservation_error": f"<= {TRANSPORT_TOL}",
        },
        "regions": {},
    }
    for region in ("us", "global"):
        region_result = results["regions"][region]
        seeds = region_result["seeds"]
        per_seed = {
            seed: gate_seed(row["summary"]) for seed, row in seeds.items()
        }
        status_quo_cost = float(
            region_result["baselines"][PRIMARY_BASELINE_NAME]["total_cost"]
        )
        candidate_costs = [
            float(seeds[key]["summary"]["total_cost"])
            for key in sorted(seeds, key=int)
        ]
        gate["regions"][region] = {
            "combo": region_result["combo"],
            "seeds": per_seed,
            "aggregate": region_result["aggregate"],
            "primary_baseline": {
                "name": PRIMARY_BASELINE_NAME,
                "summary": region_result["baselines"][PRIMARY_BASELINE_NAME],
            },
            "optimizer_vs_status_quo": optimizer_ci_vs_deterministic(
                candidate_costs,
                status_quo_cost,
            ),
            "diagnostic_vs_archived_v3_safety_only": optimizer_ci_vs_archived_sample(
                candidate_costs,
                replay_costs[region],
            ),
            "passed": bool(all(item["passed"] for item in per_seed.values())),
        }
    gate["passed"] = bool(
        all(gate["regions"][region]["passed"] for region in ("us", "global"))
    )
    return gate


def ensure_gate_passed(stage: str) -> None:
    path = gate_path(stage)
    if not path.exists():
        raise RuntimeError(f"missing promotion gate: {path}")
    gate = read_json(path)
    if not gate.get("passed", False):
        raise RuntimeError(f"{stage} gate failed; refusing to continue")


def check_for_seed_failures(records: dict[str, dict[str, Any]], *, stage: str) -> None:
    failures = [
        f"s{seed}"
        for seed, record in sorted(records.items(), key=lambda item: int(item[0]))
        if record.get("status") != "ok"
    ]
    if failures:
        raise RuntimeError(
            f"{stage} encountered safety infeasibility certificates for "
            + ", ".join(failures)
        )


def evaluate_training_stage(
    stage: str,
    jobs: list[Job],
    *,
    workers: int,
    evaluation_cells: bool = False,
) -> dict[str, Any]:
    mode = PRIMARY_MODE
    result: dict[str, Any] = {
        "stage": stage,
        "generated_at_utc": now_utc(),
        "protocol_sha256": sha256(PROTOCOL_SNAPSHOT),
        "evaluation_cells": TRANSFER_LABEL if evaluation_cells else DEVELOPMENT_LABEL,
        "mode": mode,
        "headline_eligible": False if evaluation_cells else None,
        "regions": {},
    }
    for region in ("us", "global"):
        region_jobs = [job for job in jobs if job.region == region]
        baselines = baseline_summaries(
            region,
            mode=mode,
            evaluation_cells=evaluation_cells,
        )
        records: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_jobs = {
                pool.submit(
                    evaluate_v4_seed,
                    job,
                    baselines=baselines,
                    mode=mode,
                    evaluation_cells=evaluation_cells,
                ): job
                for job in region_jobs
            }
            for future in concurrent.futures.as_completed(future_jobs):
                record = future.result()
                records[str(record["seed"])] = record
        check_for_seed_failures(records, stage=stage)
        aggregate = aggregate_records(records)
        region_result = {
            "scenario": to_rel(
                scenario_path(region, evaluation_cells=evaluation_cells)
            ),
            "combo": selected_config(region)["source_combo"],
            "selected_config": selected_config(region),
            "baselines": baselines,
            "seeds": {
                key: records[key]
                for key in sorted(records, key=int)
            },
            "aggregate": aggregate,
        }
        if not evaluation_cells:
            candidate_costs = [
                float(region_result["seeds"][key]["summary"]["total_cost"])
                for key in sorted(region_result["seeds"], key=int)
            ]
            status_quo_cost = float(
                baselines[PRIMARY_BASELINE_NAME]["total_cost"]
            )
            region_result["optimizer_vs_status_quo"] = optimizer_ci_vs_deterministic(
                candidate_costs,
                status_quo_cost,
            )
            replay_reference = load_replay_reference_costs()
            region_result["diagnostic_vs_archived_v3_safety_only"] = (
                optimizer_ci_vs_archived_sample(
                    candidate_costs,
                    replay_reference[region],
                )
            )
        result["regions"][region] = region_result
    if evaluation_cells:
        result["headline_eligible"] = False
        result["interpretation"] = (
            "Descriptive non-confirmatory transfer only. Cells e-h were already "
            "exposed by frozen v2/v3 and are not headline-eligible."
        )
    atomic_write_json(
        stage_result_path(stage, evaluation_cells=evaluation_cells),
        result,
    )
    return result


def run_replay(workers: int) -> dict[str, Any]:
    assert_provenance()
    result: dict[str, Any] = {
        "stage": "replay",
        "generated_at_utc": now_utc(),
        "protocol_sha256": sha256(PROTOCOL_SNAPSHOT),
        "evaluation_cells": DEVELOPMENT_LABEL,
        "headline_eligible": False,
        "regions": {},
    }
    source_jobs = replay_source_jobs()
    for region in ("us", "global"):
        region_jobs = [job for job in source_jobs if job.region == region]
        region_result: dict[str, Any] = {
            "scenario": to_rel(scenario_path(region)),
            "source_combo": selected_config(region)["source_combo"],
            "modes": {},
        }
        for mode in (PRIMARY_MODE, ABLATION_MODE):
            baselines = baseline_summaries(region, mode=mode, evaluation_cells=False)
            records: dict[str, dict[str, Any]] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                future_jobs = {
                    pool.submit(
                        evaluate_v3_replay_seed,
                        job,
                        baselines=baselines,
                        mode=mode,
                    ): job
                    for job in region_jobs
                }
                for future in concurrent.futures.as_completed(future_jobs):
                    record = future.result()
                    records[str(record["seed"])] = record
            check_for_seed_failures(records, stage=f"replay:{region}:{mode}")
            region_result["modes"][mode] = {
                "baselines": baselines,
                "seeds": {
                    key: records[key]
                    for key in sorted(records, key=int)
                },
                "aggregate": aggregate_records(records),
                "optimizer_vs_status_quo": optimizer_ci_vs_deterministic(
                    [
                        float(records[key]["summary"]["total_cost"])
                        for key in sorted(records, key=int)
                    ],
                    float(baselines[PRIMARY_BASELINE_NAME]["total_cost"]),
                ),
            }
        result["regions"][region] = region_result
    atomic_write_json(stage_result_path("replay"), result)
    write_manifest()
    return result


def training_stage(stage: str, jobs: list[Job], python: Path, workers: int) -> dict[str, Any]:
    assert_provenance()
    if stage == "medium":
        ensure_gate_passed("short")
    elif stage == "full":
        ensure_gate_passed("medium")
    load_replay_reference_costs()
    train_all(jobs, python, workers)
    results = evaluate_training_stage(stage, jobs, workers=workers)
    gate = build_gate(stage, results)
    atomic_write_json(gate_path(stage), gate)
    write_manifest()
    return results


def final_evaluation(workers: int) -> dict[str, Any]:
    assert_provenance()
    ensure_gate_passed("full")
    results = evaluate_training_stage(
        "final",
        full_jobs(),
        workers=workers,
        evaluation_cells=True,
    )
    write_manifest()
    return results


def envelope_scan(region: str) -> dict[str, Any]:
    env = make_safe_env(
        scenario_path(region),
        seed=301,
        peak_penalty_weight=V3_PEAK_PENALTY_WEIGHT,
        reward_config=reward_config_for_region(region),
        safety_config=safety_config_for_mode(PRIMARY_MODE),
        domain_randomization=False,
        deadline_bucket_edges=DEADLINE_EDGES,
    )
    try:
        obs, _ = env.reset(seed=301)
        if env.observation_space.shape != (81,) or obs.shape != (81,):
            raise RuntimeError(f"unexpected observation shape for {region}")
        if env.action_space.shape != (12,):
            raise RuntimeError(f"unexpected action shape for {region}")
        max_service = 0.0
        max_batch = 0.0
        for t in range(env.max_steps):
            total_service = float(
                sum(site.get_service_demand(t) for site in env.sites)
            )
            total_batch = float(
                sum(site.get_batch_demand(t) for site in env.sites)
            )
            max_service = max(max_service, total_service)
            max_batch = max(max_batch, total_batch)
        if max_service > float(CONFIG["safety"]["service_envelope_total"]) + 1e-12:
            raise RuntimeError(
                f"service envelope violated in development cells for {region}: {max_service}"
            )
        if max_batch > float(CONFIG["safety"]["batch_arrival_envelope_total"]) + 1e-12:
            raise RuntimeError(
                f"batch envelope violated in development cells for {region}: {max_batch}"
            )
        return {
            "max_steps": int(env.max_steps),
            "observation_shape": list(env.observation_space.shape),
            "action_shape": list(env.action_space.shape),
            "max_total_service_demand": float(max_service),
            "max_total_batch_arrival": float(max_batch),
            "envelopes": {
                "service_envelope_total": float(
                    CONFIG["safety"]["service_envelope_total"]
                ),
                "batch_arrival_envelope_total": float(
                    CONFIG["safety"]["batch_arrival_envelope_total"]
                ),
            },
        }
    finally:
        env.close()


def preflight(python: Path) -> dict[str, Any]:
    validate_config()
    for script in SMOKE_TESTS:
        subprocess.run(
            [str(python), script],
            cwd=ROOT,
            check=True,
            env=subprocess_env(),
        )
    checks = {
        "generated_at_utc": now_utc(),
        "job_counts": {
            "replay": len(replay_source_jobs()),
            "short": len(short_jobs()),
            "medium": len(medium_jobs()),
            "full": len(full_jobs()),
        },
        "development_cells": {
            region: envelope_scan(region) for region in ("us", "global")
        },
    }
    write_protocol()
    assert_provenance()
    atomic_write_json(OUT_ROOT / "preflight.json", checks)
    write_manifest()
    print(f"v4 protocol ready: {PROTOCOL_SNAPSHOT}", flush=True)
    return checks


def write_manifest() -> None:
    entries: list[dict[str, Any]] = []
    for path in sorted(MODEL_ROOT.glob("**/model.zip")):
        record_path = path.with_suffix(".complete.json")
        if not record_path.exists():
            raise RuntimeError(f"missing completion record: {path}")
        record = read_json(record_path)
        if record["model_sha256"] != sha256(path):
            raise RuntimeError(f"model hash mismatch: {path}")
        entries.append(
            {
                "path": to_rel(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "completion_record_path": to_rel(record_path),
                "completion_record_sha256": sha256(record_path),
                "job_fingerprint": record["job_fingerprint"],
            }
        )
    manifest = {
        "generated_at_utc": now_utc(),
        "protocol_sha256": (
            sha256(PROTOCOL_SNAPSHOT) if PROTOCOL_SNAPSHOT.exists() else None
        ),
        "model_count": len(entries),
        "models": entries,
    }
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(MODEL_ROOT / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["preflight", "replay", "short", "medium", "full", "final", "all"],
        default="all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 16),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.python.resolve() != Path(sys.executable).resolve():
        raise ValueError(
            "--python must match the launcher interpreter so frozen package "
            "provenance applies to every subprocess"
        )
    if args.phase in {"preflight", "all"}:
        preflight(args.python)
    if args.phase in {"replay", "all"}:
        run_replay(args.workers)
    if args.phase in {"short", "all"}:
        training_stage("short", short_jobs(), args.python, args.workers)
    if args.phase in {"medium", "all"}:
        training_stage("medium", medium_jobs(), args.python, args.workers)
    if args.phase in {"full", "all"}:
        training_stage("full", full_jobs(), args.python, args.workers)
    if args.phase in {"final", "all"}:
        final_evaluation(args.workers)


if __name__ == "__main__":
    main()
