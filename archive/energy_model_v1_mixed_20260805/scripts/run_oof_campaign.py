"""Two-fold out-of-fold PPO campaign across Google cells a-h.

Protocol (frozen before training):

* Fold ``ad_to_eh`` trains on cells a-d and evaluates frozen policies on e-h.
* Fold ``eh_to_ad`` trains on cells e-h and evaluates frozen policies on a-d.
* Every environment retains four DCs and the complete 8,917-step month.
* PPO hyperparameters, reward, 501,760-step rollout-aligned budget, 10 seeds,
  and metrics are fixed; no
  validation-based checkpoint or hyperparameter selection is performed.
* Domain-randomization ranges come only from the current training scenario.
* Headline metrics are held-out savings versus the held-out Status Quo,
  completion, and gap to a held-out clairvoyant QP lower bound.

Usage:
    python scripts/run_oof_campaign.py --phase train --workers 16
    python scripts/run_oof_campaign.py --phase evaluate
    python scripts/run_oof_campaign.py --phase all --workers 16
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines import (  # noqa: E402
    DrainImmediatelyPolicy,
    RoundRobinPolicy,
    StatusQuoPolicy,
)
from env.data_loader import load_scenario  # noqa: E402
from env.multi_dc_env import MultiDCEnv  # noqa: E402
from evaluate import compute_summary, run_episode  # noqa: E402
from scripts.compute_qp_optimum import build_inputs, solve_clarabel  # noqa: E402

ALPHA = 0.015
TIMESTEPS = 501_760  # 245 complete PPO rollouts at n_steps=2,048
SEEDS = list(range(101, 111))
SERVICE_COMPLETION_FLOOR = 0.9999
BATCH_COMPLETION_FLOOR = 0.9999
COMPLETION_PENALTY = 1_000.0

FOLDS = {
    "ad_to_eh": {
        "train": {
            "us": "env/scenarios/us_model.yaml",
            "global": "env/scenarios/global_model.yaml",
        },
        "test": {
            "us": "env/scenarios/us_model_eh.yaml",
            "global": "env/scenarios/global_model_eh.yaml",
        },
    },
    "eh_to_ad": {
        "train": {
            "us": "env/scenarios/us_model_eh.yaml",
            "global": "env/scenarios/global_model_eh.yaml",
        },
        "test": {
            "us": "env/scenarios/us_model.yaml",
            "global": "env/scenarios/global_model.yaml",
        },
    },
}

CONFIGS = [
    ("us_spatial", "us", False),
    ("us_batch", "us", True),
    ("global_spatial", "global", False),
    ("global_batch", "global", True),
]

MODEL_ROOT = ROOT / "models" / "oof"
LOG_ROOT = ROOT / "logs" / "oof"
OUT_ROOT = ROOT / "output" / "oof"

SOURCE_FILES = [
    "baselines.py",
    "evaluate.py",
    "train.py",
    "env/data_loader.py",
    "env/dc_site.py",
    "env/multi_dc_env.py",
    "env/power_model.py",
    "env/workload_generator.py",
    "scripts/compute_qp_optimum.py",
    "scripts/run_oof_campaign.py",
    "requirements.txt",
    "env/scenarios/us_model.yaml",
    "env/scenarios/us_model_eh.yaml",
    "env/scenarios/global_model.yaml",
    "env/scenarios/global_model_eh.yaml",
]


def scenario_path(path: str) -> Path:
    return ROOT / path


def model_path(
    fold: str,
    region: str,
    batch: bool,
    seed: int,
) -> Path:
    stem = scenario_path(FOLDS[fold]["train"][region]).stem
    name = f"ppo_{stem}" + ("_batch" if batch else "")
    if batch:
        name += "_completion_guard"
    return MODEL_ROOT / fold / f"s{seed}" / f"{name}.zip"


def log_path(fold: str, cfg: str, seed: int) -> Path:
    return LOG_ROOT / fold / f"{cfg}_s{seed}.log"


def completion_record_path(model: Path) -> Path:
    return model.with_suffix(".complete.json")


def protocol_hash() -> str:
    return sha256(OUT_ROOT / "protocol.json")


def job_spec(
    fold: str,
    cfg: str,
    region: str,
    batch: bool,
    seed: int,
    timesteps: int,
) -> dict[str, Any]:
    spec = {
        "protocol_sha256": protocol_hash(),
        "fold": fold,
        "config": cfg,
        "region": region,
        "batch": batch,
        "seed": seed,
        "timesteps": timesteps,
        "train_scenario": FOLDS[fold]["train"][region],
        "test_scenario": FOLDS[fold]["test"][region],
        "domain_randomization": True,
        "enforce_batch_completion": batch,
        "completion_penalty": (
            COMPLETION_PENALTY if batch else None
        ),
        "gamma": 1.0,
        "alpha": ALPHA,
        "baseline_scope": {
            "primary": "Status Quo (local, no deferral)",
            "routing_sanity": "Round Robin",
            "batch_no_deferral_ablation": "Drain Immediately",
            "headroom_only": "clairvoyant fluid QP lower bound",
            "excluded_from_headline": "DQN/CFWS and algorithm-superiority tests",
        },
    }
    encoded = json.dumps(
        spec, sort_keys=True, separators=(",", ":")
    ).encode()
    spec["job_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return spec


def validate_completed_job(
    model: Path,
    record_path: Path,
    spec: dict[str, Any],
) -> bool:
    from stable_baselines3 import PPO

    if not model.exists() and not record_path.exists():
        return False
    if not model.exists() or not record_path.exists():
        raise RuntimeError(
            f"incomplete prior output for {model}; remove both model and "
            "completion record before retrying"
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("job_fingerprint") != spec["job_fingerprint"]:
        raise RuntimeError(f"protocol mismatch for existing model {model}")
    if record.get("model_sha256") != sha256(model):
        raise RuntimeError(f"model hash mismatch for {model}")
    loaded = PPO.load(model)
    if loaded.num_timesteps != spec["timesteps"]:
        raise RuntimeError(
            f"model timestep mismatch for {model}: "
            f"{loaded.num_timesteps} != {spec['timesteps']}"
        )
    return True


def training_jobs() -> list[tuple[str, str, str, bool, int]]:
    return [
        (fold, cfg, region, batch, seed)
        for fold in FOLDS
        for cfg, region, batch in CONFIGS
        for seed in SEEDS
    ]


def training_bounds(scenario: Path, batch: bool) -> dict[str, list[float]]:
    sites, power_model, batch_config = load_scenario(
        scenario, batch_enabled=batch, seed=42
    )
    env = MultiDCEnv(
        sites,
        power_model,
        batch_enabled=batch,
        peak_penalty_weight=ALPHA,
        flexibility_factor=batch_config.get("flexibility_factor", 1.0),
        deadline_penalty_weight=batch_config.get(
            "deadline_penalty_weight", 2.0
        ),
        urgency_horizon_steps=batch_config.get("urgency_horizon_steps", 12),
        domain_randomization=True,
        enforce_batch_completion=batch,
        completion_penalty_weight=(
            COMPLETION_PENALTY if batch else None
        ),
    )
    return {
        "idle": list(env._domain_idle_bounds),
        "slope": list(env._domain_slope_bounds),
        "computed_economic_floor": float(
            env.computed_economic_penalty_floor
        ),
        "fixed_completion_penalty": float(env.economic_penalty_floor),
    }


def protocol() -> dict[str, Any]:
    source_hashes = {
        path: sha256(ROOT / path) for path in SOURCE_FILES
    }
    data_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in sorted((ROOT / "data").rglob("*"))
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}
    }
    package_versions = {}
    for package in (
        "numpy",
        "pandas",
        "torch",
        "gymnasium",
        "stable-baselines3",
        "scipy",
    ):
        package_versions[package] = importlib.metadata.version(package)
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT
    )
    return {
        "protocol": "cross-cell-two-fold-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "folds": FOLDS,
        "configs": [c[0] for c in CONFIGS],
        "seeds": SEEDS,
        "timesteps": TIMESTEPS,
        "algorithm": "PPO",
        "ppo": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "net_arch": [128, 128],
            "gamma": 1.0,
            "action_logit_bound": 3.0,
        },
        "domain_randomization": True,
        "domain_randomization_bounds": {
            fold: {
                cfg: training_bounds(
                    scenario_path(FOLDS[fold]["train"][region]), batch
                )
                for cfg, region, batch in CONFIGS
            }
            for fold in FOLDS
        },
        "alpha": ALPHA,
        "batch_completion_guard": {
            "enabled": True,
            "dense_cost": "lambda_x * (arrivals - completions)",
            "terminal_pool_penalized": True,
            "economic_penalty_floor": True,
            "fixed_completion_penalty_usd_per_unit": COMPLETION_PENALTY,
        },
        "validation_selection": None,
        "test_scope": "full 8,917-step month on opposite four-cell fold",
        "headline_metrics": [
            "held_out_total_cost",
            "held_out_savings_vs_status_quo_pct",
            "service_completion",
            "batch_completion",
            "batch_expired",
            "terminal_batch_pool",
            "billed_peak_sum_mw",
            "load_factor",
            "gap_to_held_out_qp_pct",
        ],
        "success_rule": {
            "savings": (
                "95% optimizer-bootstrap CI lower bound is positive in both "
                "folds"
            ),
            "service_completion_floor": SERVICE_COMPLETION_FLOOR,
            "batch_completion_floor": BATCH_COMPLETION_FLOOR,
        },
        "interpretation": (
            "Cross-cell out-of-fold evaluation with 10 optimizer seeds per "
            "fold. Bootstrap intervals describe optimizer variability only; "
            "no population-level significance claim or cross-algorithm pairing."
        ),
        "provenance": {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_diff_binary_sha256": hashlib.sha256(diff).hexdigest(),
            "python": sys.version,
            "packages": package_versions,
            "source_sha256": source_hashes,
            "data_sha256": data_hashes,
        },
    }


def write_protocol() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "protocol.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        immutable = {
            k: v for k, v in existing.items() if k != "frozen_at_utc"
        }
        current = {
            k: v for k, v in protocol().items() if k != "frozen_at_utc"
        }
        if immutable != current:
            raise RuntimeError(
                "output/oof/protocol.json differs from the current protocol; "
                "refusing to change frozen metrics after training began"
            )
        return
    path.write_text(json.dumps(protocol(), indent=2), encoding="utf-8")


def assert_frozen_provenance() -> None:
    frozen = json.loads(
        (OUT_ROOT / "protocol.json").read_text(encoding="utf-8")
    )["provenance"]
    if sys.version != frozen["python"]:
        raise RuntimeError("Python runtime changed after protocol freeze")
    for package, expected in frozen["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(
                f"package version changed after freeze: {package}"
            )
    for path, expected in frozen["source_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(
                f"training source changed after protocol freeze: {path}"
            )
    for path, expected in frozen["data_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(
                f"training data changed after protocol freeze: {path}"
            )


def train_one(
    job: tuple[str, str, str, bool, int],
    python: Path,
    timesteps: int,
) -> tuple[str, bool, float]:
    fold, cfg, region, batch, seed = job
    assert_frozen_provenance()
    model = model_path(fold, region, batch, seed)
    label = f"{fold}:{cfg}:s{seed}"
    spec = job_spec(fold, cfg, region, batch, seed, timesteps)
    record_path = completion_record_path(model)
    if validate_completed_job(model, record_path, spec):
        return label, True, 0.0

    model.parent.mkdir(parents=True, exist_ok=True)
    log = log_path(fold, cfg, seed)
    log.parent.mkdir(parents=True, exist_ok=True)
    scenario = scenario_path(FOLDS[fold]["train"][region])
    temp_dir = (
        MODEL_ROOT / "_tmp" / f"{fold}_{cfg}_s{seed}_{uuid.uuid4().hex}"
    )
    temp_dir.mkdir(parents=True, exist_ok=False)
    cmd = [
        str(python),
        "train.py",
        "--scenario",
        str(scenario),
        "--timesteps",
        str(timesteps),
        "--peak-penalty-weight",
        str(ALPHA),
        "--seed",
        str(seed),
        "--output-dir",
        str(temp_dir),
        "--domain-rand",
    ]
    if batch:
        cmd += [
            "--batch-mode",
            "--enforce-batch-completion",
            "--completion-penalty",
            str(COMPLETION_PENALTY),
        ]
    cmd += ["--gamma", "1.0"]

    lock = model.with_suffix(".lock")
    lock_fd: int | None = None
    started = time.time()
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with log.open("w", encoding="utf-8") as handle:
            handle.write(
                f"# {label}\n# job: {json.dumps(spec, sort_keys=True)}\n"
                f"# command: {' '.join(cmd)}\n"
                f"# started: {datetime.now(timezone.utc).isoformat()}\n\n"
            )
            handle.flush()
            result = subprocess.run(
                cmd,
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
        temp_model = temp_dir / model.name
        if result.returncode != 0 or not temp_model.exists():
            return label, False, elapsed

        from stable_baselines3 import PPO

        loaded = PPO.load(temp_model)
        if loaded.num_timesteps != timesteps:
            raise RuntimeError(
                f"trained timestep mismatch for {label}: "
                f"{loaded.num_timesteps} != {timesteps}"
            )
        assert_frozen_provenance()
        model_hash = sha256(temp_model)
        os.replace(temp_model, model)
        record = {
            **spec,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "model_path": str(model.relative_to(ROOT)).replace("\\", "/"),
            "model_bytes": model.stat().st_size,
            "model_sha256": model_hash,
            "log_path": str(log.relative_to(ROOT)).replace("\\", "/"),
            "log_sha256": sha256(log),
        }
        record_tmp = record_path.with_suffix(".json.tmp")
        record_tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(record_tmp, record_path)
        return label, True, elapsed
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            lock.unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


def train_all(python: Path, workers: int, timesteps: int) -> None:
    jobs = training_jobs()
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_jobs = {
            pool.submit(train_one, job, python, timesteps): job for job in jobs
        }
        for future in concurrent.futures.as_completed(future_jobs):
            label, ok, elapsed = future.result()
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {label} ({elapsed / 60:.1f} min)", flush=True)
            if not ok:
                failures.append(label)
    if failures:
        raise RuntimeError(f"training failed: {', '.join(failures)}")


def make_env(scenario: Path, batch: bool) -> MultiDCEnv:
    sites, power_model, batch_config = load_scenario(
        scenario, batch_enabled=batch, seed=42
    )
    return MultiDCEnv(
        sites,
        power_model,
        batch_enabled=batch,
        peak_penalty_weight=ALPHA,
        flexibility_factor=batch_config.get("flexibility_factor", 1.0),
        deadline_penalty_weight=batch_config.get(
            "deadline_penalty_weight", 2.0
        ),
        urgency_horizon_steps=batch_config.get("urgency_horizon_steps", 12),
        enforce_batch_completion=batch,
        completion_penalty_weight=(
            COMPLETION_PENALTY if batch else None
        ),
    )


def completion(summary: dict[str, Any]) -> tuple[float, float]:
    service = (
        summary["service_served_total"] / summary["service_demand_total"]
        if summary["service_demand_total"] > 0
        else 1.0
    )
    batch = summary.get("batch_completion_fraction", 1.0)
    return float(service), float(batch)


def bootstrap_mean_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    samples: int = 20_000,
) -> list[float]:
    rng = np.random.default_rng(0)
    means = rng.choice(
        values, size=(samples, len(values)), replace=True
    ).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    ]


def feasible(summary: dict[str, Any]) -> bool:
    service, batch = completion(summary)
    return (
        service >= SERVICE_COMPLETION_FLOOR
        and batch >= BATCH_COMPLETION_FLOOR
    )


def evaluate_fold_config(
    fold: str,
    cfg: str,
    region: str,
    batch: bool,
) -> dict[str, Any]:
    from stable_baselines3 import PPO

    test_scenario = scenario_path(FOLDS[fold]["test"][region])
    train_scenario = scenario_path(FOLDS[fold]["train"][region])
    entry: dict[str, Any] = {
        "fold": fold,
        "config": cfg,
        "train_scenario": str(train_scenario.relative_to(ROOT)),
        "test_scenario": str(test_scenario.relative_to(ROOT)),
        "baselines": {},
        "ppo_seeds": {},
    }

    baseline_classes = [StatusQuoPolicy, RoundRobinPolicy]
    if batch:
        baseline_classes.append(DrainImmediatelyPolicy)
    for baseline_cls in baseline_classes:
        policy = baseline_cls()
        _, history = run_episode(
            make_env(test_scenario, batch), policy.predict, is_sb3=False
        )
        entry["baselines"][policy.name] = compute_summary(
            history, batch_enabled=batch
        )

    status_quo = entry["baselines"][
        "Status Quo (local, no deferral)"
    ]["total_cost"]
    for seed in SEEDS:
        path = model_path(fold, region, batch, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        model = PPO.load(path)
        _, history = run_episode(
            make_env(test_scenario, batch), model.predict, is_sb3=True
        )
        summary = compute_summary(history, batch_enabled=batch)
        summary["savings_vs_status_quo_pct"] = (
            100.0 * (status_quo - summary["total_cost"]) / status_quo
        )
        summary["feasible"] = feasible(summary)
        entry["ppo_seeds"][str(seed)] = summary

    qp = solve_clarabel(
        build_inputs(
            str(test_scenario),
            batch,
            enforce_batch_completion=batch,
            completion_penalty_weight=(
                COMPLETION_PENALTY if batch else None
            ),
        ),
        batch,
    )
    entry["qp"] = qp

    seeds = list(entry["ppo_seeds"].values())
    savings = np.array(
        [s["savings_vs_status_quo_pct"] for s in seeds], dtype=float
    )
    costs = np.array([s["total_cost"] for s in seeds], dtype=float)
    service = np.array([completion(s)[0] for s in seeds], dtype=float)
    batch_completion = np.array(
        [completion(s)[1] for s in seeds], dtype=float
    )
    feasible_baselines = {
        name: summary
        for name, summary in entry["baselines"].items()
        if feasible(summary)
    }
    best_baseline = min(
        feasible_baselines,
        key=lambda name: feasible_baselines[name]["total_cost"],
    )
    mean_cost = float(costs.mean())
    entry["summary"] = {
        "status_quo_cost": float(status_quo),
        "ppo_mean_cost": mean_cost,
        "ppo_std_cost": float(costs.std(ddof=1)),
        "ppo_mean_savings_pct": float(savings.mean()),
        "ppo_std_savings_pct": float(savings.std(ddof=1)),
        "ppo_mean_savings_ci95_optimizer": bootstrap_mean_ci(savings),
        "ppo_min_savings_pct": float(savings.min()),
        "ppo_max_savings_pct": float(savings.max()),
        "all_seeds_positive": bool(np.all(savings > 0.0)),
        "minimum_service_completion": float(service.min()),
        "minimum_batch_completion": float(batch_completion.min()),
        "all_seeds_feasible": bool(all(s["feasible"] for s in seeds)),
        "best_feasible_baseline": best_baseline,
        "best_feasible_baseline_cost": float(
            feasible_baselines[best_baseline]["total_cost"]
        ),
        "ppo_gap_to_qp_pct": float(
            100.0
            * (mean_cost - qp["optimum_total"])
            / qp["optimum_total"]
        ),
    }
    return entry


def evaluate_all() -> None:
    assert_frozen_provenance()
    for fold, cfg, region, batch, seed in training_jobs():
        model = model_path(fold, region, batch, seed)
        spec = job_spec(
            fold, cfg, region, batch, seed, TIMESTEPS
        )
        if not validate_completed_job(
            model, completion_record_path(model), spec
        ):
            raise RuntimeError(f"missing completed model: {model}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fold_results: dict[str, Any] = {}
    for fold in FOLDS:
        fold_results[fold] = {}
        for cfg, region, batch in CONFIGS:
            print(f"[eval] {fold}:{cfg}", flush=True)
            entry = evaluate_fold_config(fold, cfg, region, batch)
            fold_results[fold][cfg] = entry
            output = OUT_ROOT / f"{fold}_{cfg}.json"
            output.write_text(json.dumps(entry, indent=2), encoding="utf-8")

    aggregate: dict[str, Any] = {
        "protocol": json.loads(
            (OUT_ROOT / "protocol.json").read_text(encoding="utf-8")
        ),
        "folds": fold_results,
        "configs": {},
    }
    for cfg, _, _ in CONFIGS:
        fold_summaries = {
            fold: fold_results[fold][cfg]["summary"] for fold in FOLDS
        }
        fold_means = np.array([
            fold_summaries[fold]["ppo_mean_savings_pct"] for fold in FOLDS
        ])
        aggregate["configs"][cfg] = {
            "folds": fold_summaries,
            "macro_mean_savings_pct": float(fold_means.mean()),
            "worst_fold_mean_savings_pct": float(fold_means.min()),
            "positive_in_both_folds": bool(np.all(fold_means > 0.0)),
            "optimizer_ci_positive_in_both_folds": bool(all(
                fold_summaries[fold][
                    "ppo_mean_savings_ci95_optimizer"
                ][0] > 0.0
                for fold in FOLDS
            )),
            "all_seeds_positive_in_both_folds": bool(all(
                fold_summaries[fold]["all_seeds_positive"] for fold in FOLDS
            )),
            "all_seeds_feasible_in_both_folds": bool(all(
                fold_summaries[fold]["all_seeds_feasible"] for fold in FOLDS
            )),
        }

    aggregate["headline_joint_shaping_supported"] = bool(
        aggregate["configs"]["us_batch"][
            "optimizer_ci_positive_in_both_folds"
        ]
        and aggregate["configs"]["global_batch"][
            "optimizer_ci_positive_in_both_folds"
        ]
        and aggregate["configs"]["us_batch"][
            "all_seeds_feasible_in_both_folds"
        ]
        and aggregate["configs"]["global_batch"][
            "all_seeds_feasible_in_both_folds"
        ]
    )
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    write_manifest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> None:
    models = []
    for path in sorted(MODEL_ROOT.glob("**/*.zip")):
        record_path = completion_record_path(path)
        if not record_path.exists():
            raise RuntimeError(f"missing completion record for {path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["model_sha256"] != sha256(path):
            raise RuntimeError(f"manifest hash mismatch for {path}")
        models.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "job_fingerprint": record["job_fingerprint"],
            "completion_record": str(
                record_path.relative_to(ROOT)
            ).replace("\\", "/"),
        })
    frozen_protocol = json.loads(
        (OUT_ROOT / "protocol.json").read_text(encoding="utf-8")
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(OUT_ROOT / "protocol.json"),
        "frozen_provenance": frozen_protocol["provenance"],
        "models": models,
    }
    (MODEL_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["train", "evaluate", "all"], default="all"
    )
    parser.add_argument(
        "--workers", type=int, default=min(os.cpu_count() or 1, 16)
    )
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    if args.python.resolve() != Path(sys.executable).resolve():
        raise ValueError(
            "--python must match the launcher interpreter so frozen package "
            "provenance applies to every training subprocess"
        )
    if args.timesteps != TIMESTEPS:
        raise ValueError(
            f"the frozen protocol requires {TIMESTEPS} timesteps"
        )
    write_protocol()
    assert_frozen_provenance()
    if args.phase in ("train", "all"):
        train_all(args.python, args.workers, args.timesteps)
    if args.phase in ("evaluate", "all"):
        evaluate_all()


if __name__ == "__main__":
    main()
