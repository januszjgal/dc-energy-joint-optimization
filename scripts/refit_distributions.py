"""Refit batch job distributions from full Google ClusterData 2019 extracts.

Uses MLE fitting on 200k batch jobs per cell (from batch_raw_*.csv).
Falls back to quantile-matching from summary statistics only if raw data
is unavailable.

Produces updated batch_distributions_*.json with log-normal, Weibull, gamma
fits. For tasks_per_job, uses a mixture model: point-mass at 1 + continuous tail.

Following Da Costa et al. (2016) and Grange et al. (2018) methodology for
synthetic workload generation from fitted distributions.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JOBS_DIR = DATA_DIR / "jobs"

CELLS = ["a", "b", "c", "d"]

# Candidate continuous distributions for MLE
CANDIDATES = {
    "lognorm": stats.lognorm,
    "weibull_min": stats.weibull_min,
    "gamma": stats.gamma,
    "expon": stats.expon,
}

# Minimum sample size for reliable MLE fitting
MIN_SAMPLES_MLE = 20

# Max samples for MLE fitting (subsampling for speed on large datasets)
MAX_SAMPLES_FIT = 50000

# Per-cell batch fraction = beb jobs / total jobs.
# WARNING: these values are STALE — they were computed under the old, incorrect
# `scheduling_class <= 1 AND priority < 200` classification, which conflates the
# production tier (priority 200) with batch. The correct definition is the Borg
# best-effort batch (beb) tier, priority 110-115 (Tirmazi et al. 2020 §2). These
# MUST be recomputed from the re-extracted beb data (colab_extract_batch_jobs.py
# now prints the correct beb fraction per cell) before the numbers are trusted.
# TODO(rerun): replace with beb fractions from the re-extraction.
BATCH_FRACTIONS = {
    "a": 0.04304,
    "b": 0.48353,
    "c": 0.12955,
    "d": 0.20146,
}


def fit_from_samples(
    data: np.ndarray,
    subsample: int = MAX_SAMPLES_FIT,
    trim_percentile: tuple[float, float] = (1, 99),
) -> dict | None:
    """Fit multiple distributions to raw sample data, return best by KS.

    Trims to [p1, p99] before fitting to avoid extreme outliers biasing
    MLE (especially for heavy-tailed inter-arrival data with bursts).
    Subsamples to `subsample` for MLE speed.
    """
    data = data[np.isfinite(data)]
    data = data[data > 0]

    if len(data) < MIN_SAMPLES_MLE:
        return None

    # Trim extreme outliers before fitting
    lo, hi = np.percentile(data, trim_percentile)
    trimmed = data[(data >= lo) & (data <= hi)]
    if len(trimmed) < MIN_SAMPLES_MLE:
        trimmed = data

    # Subsample for fitting (MLE on 200k points is slow)
    rng = np.random.default_rng(42)
    fit_data = trimmed
    if len(trimmed) > subsample:
        fit_data = rng.choice(trimmed, subsample, replace=False)

    # Use a subsample for KS test (on untrimmed data for honest evaluation)
    ks_data = data
    if len(data) > 5000:
        ks_data = rng.choice(data, 5000, replace=False)

    best = None
    for name, dist in CANDIDATES.items():
        try:
            params = dist.fit(fit_data)

            # Sanity check: fitted mean should be within 10x of sample mean
            fitted_mean = dist(*params).mean()
            sample_mean = data.mean()
            if fitted_mean > sample_mean * 10 or fitted_mean < sample_mean * 0.1:
                continue

            ks_stat, ks_p = stats.kstest(ks_data, name, args=params)
            if np.isnan(ks_stat):
                continue
            entry = {
                "distribution": name,
                "params": [float(p) for p in params],
                "ks_statistic": round(float(ks_stat), 6),
                "ks_pvalue": round(float(ks_p), 6),
            }
            if best is None or ks_stat < best["ks_statistic"]:
                best = entry
        except Exception:
            continue
    return best


def fit_from_quantiles(summary: dict) -> dict:
    """Fit a lognormal distribution from summary statistics (method of moments).

    For data with mean >> median, lognormal is the natural choice:
        X ~ LogNormal(mu, sigma)
        median = exp(mu)
        mean = exp(mu + sigma^2/2)
    """
    mean = summary["mean"]
    median = summary["median"]

    if median <= 0 or mean <= 0:
        return {
            "distribution": "expon",
            "params": [0.0, mean],
            "ks_statistic": 1.0,
            "ks_pvalue": 0.0,
            "fit_method": "fallback",
        }

    mu = np.log(median)
    sigma_sq = 2.0 * (np.log(mean) - mu)

    if sigma_sq <= 0:
        sigma_sq = 0.01

    sigma = np.sqrt(sigma_sq)
    params = [sigma, 0.0, np.exp(mu)]

    # Approximate KS via quantile comparison
    dist = stats.lognorm(*params)
    quantile_probs = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantile_keys = ["p5", "p25", "median", "p75", "p95"]
    errors = []
    for p, key in zip(quantile_probs, quantile_keys):
        obs = summary.get(key)
        if obs is not None and obs > 0:
            exp = dist.ppf(p)
            errors.append(min(abs(obs - exp) / obs, 1.0))
    approx_ks = np.mean(errors) if errors else 0.5

    return {
        "distribution": "lognorm",
        "params": [float(p) for p in params],
        "ks_statistic": round(min(approx_ks, 1.0), 6),
        "ks_pvalue": None,
        "fit_method": "quantile_matching",
        "n_samples_original": summary.get("n_samples", 0),
    }


def fit_tasks_mixture(raw_data: np.ndarray | None = None,
                      summary: dict | None = None) -> dict:
    """Fit tasks_per_job as a mixture: point-mass at 1 + lognormal tail."""
    if raw_data is not None and len(raw_data) >= MIN_SAMPLES_MLE:
        raw = raw_data[np.isfinite(raw_data)]
        raw = raw[raw >= 1]
        point_mass_weight = float((raw <= 1.0).sum() / len(raw))
        multi = raw[raw > 1.0]
        n_samples = len(raw)
        mean_val = float(raw.mean())
        median_val = float(np.median(raw))
        std_val = float(raw.std())
    elif summary is not None:
        if summary.get("p95", 1.0) <= 1.0:
            point_mass_weight = 0.95
        elif summary.get("p75", 1.0) <= 1.0:
            point_mass_weight = 0.75
        else:
            point_mass_weight = 0.5
        multi = None
        n_samples = summary.get("n_samples", 0)
        mean_val = summary["mean"]
        median_val = summary["median"]
        std_val = summary["std"]
    else:
        return {"distribution": "mixture", "point_mass": {"value": 1, "weight": 0.95}}

    result = {
        "distribution": "mixture",
        "point_mass": {"value": 1, "weight": round(point_mass_weight, 6)},
        "n_samples": n_samples,
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "name": "tasks_per_job",
    }

    # Fit the tail
    if multi is not None and len(multi) >= MIN_SAMPLES_MLE:
        tail_fit = fit_from_samples(multi)
        if tail_fit:
            tail_fit["n_samples"] = int(len(multi))
            result["tail"] = tail_fit
            return result

    # Estimate tail from summary stats
    tail_mean = mean_val / max(1.0 - point_mass_weight, 0.01)
    if tail_mean > 2:
        result["tail"] = {
            "distribution": "lognorm",
            "params": [1.5, 1.0, tail_mean * 0.3],
            "fit_method": "estimated",
        }
    else:
        result["tail"] = {"distribution": "empirical", "mean": tail_mean}

    return result


def compute_stats(data: np.ndarray) -> dict:
    """Compute summary statistics."""
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {"n_samples": 0}
    return {
        "n_samples": int(len(data)),
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "std": float(data.std()),
        "p5": float(np.percentile(data, 5)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
        "p95": float(np.percentile(data, 95)),
    }


def fit_variable(name: str, data: np.ndarray, label: str) -> dict:
    """Fit a distribution to a variable, printing results."""
    print(f"  {label}: ", end="")
    data = data[np.isfinite(data)]
    data = data[data > 0]

    if len(data) >= MIN_SAMPLES_MLE:
        fit = fit_from_samples(data)
        if fit:
            fit.update(compute_stats(data))
            fit["name"] = name
            print(f"{fit['distribution']} KS={fit['ks_statistic']:.4f} "
                  f"(MLE from {len(data):,} samples)")
            return fit

    # Fallback: lognormal from stats
    s = compute_stats(data)
    fit = fit_from_quantiles(s)
    fit["name"] = name
    fit.update({k: v for k, v in s.items() if k not in fit})
    print(f"{fit['distribution']} (quantile-matched, approx KS={fit['ks_statistic']:.4f})")
    return fit


def process_cell(cell: str) -> dict:
    """Process one cell using full batch_raw data."""
    batch_fraction = BATCH_FRACTIONS[cell]

    # Try full raw data first, then truncated
    raw_path = JOBS_DIR / f"batch_raw_{cell}.csv"
    trunc_path = JOBS_DIR / f"jobs_{cell}_truncated.csv"

    if raw_path.exists():
        df = pd.read_csv(raw_path)
        print(f"  Loaded {len(df):,} batch jobs from batch_raw_{cell}.csv")
        source = "raw"
    elif trunc_path.exists():
        df = pd.read_csv(trunc_path)
        df = df[df["job_type"] == "batch"] if "job_type" in df.columns else df
        print(f"  Loaded {len(df):,} batch jobs from truncated CSV (fallback)")
        source = "truncated"
    else:
        print(f"  WARNING: No data files found for cell {cell}")
        return {}

    result = {
        "workload_mix": {
            "total_jobs": int(len(df)),
            "batch_fraction": batch_fraction,
        },
    }

    # --- Inter-arrival ---
    print(f"  inter_arrival: ", end="")
    submit_col = "submit_time_us" if "submit_time_us" in df.columns else "submit_time"
    sorted_submit = df[submit_col].sort_values().values
    # Convert to seconds
    if submit_col == "submit_time_us":
        ia_sec = np.diff(sorted_submit) / 1e6
    else:
        ia_sec = np.diff(sorted_submit).astype(float)
    # Remove zeros and negatives
    ia_sec = ia_sec[ia_sec > 0]

    if len(ia_sec) >= MIN_SAMPLES_MLE:
        fit = fit_from_samples(ia_sec)
        if fit:
            fit.update(compute_stats(ia_sec))
            fit["name"] = "inter_arrival_sec"
            print(f"{fit['distribution']} KS={fit['ks_statistic']:.4f} "
                  f"(MLE from {len(ia_sec):,} samples)")
        else:
            s = compute_stats(ia_sec)
            fit = fit_from_quantiles(s)
            fit["name"] = "inter_arrival_sec"
            fit.update({k: v for k, v in s.items() if k not in fit})
            print(f"{fit['distribution']} (quantile-matched)")
    else:
        print(f"insufficient inter-arrival data ({len(ia_sec)} positive values)")
        fit = {"distribution": "expon", "params": [0.0, 1.0],
               "name": "inter_arrival_sec", "fit_method": "fallback"}
    result["inter_arrival"] = fit

    # --- Duration ---
    dur_col = "duration_sec"
    raw_dur = df[dur_col].dropna().values.astype(float)
    result["duration"] = fit_variable("duration_sec", raw_dur, "duration")

    # --- CPU request ---
    cpu_col = "avg_cpu_request"
    raw_cpu = df[cpu_col].dropna().values.astype(float)
    result["cpu_request"] = fit_variable("cpu_request", raw_cpu, "cpu_request")

    # --- Memory request ---
    mem_col = "avg_mem_request"
    raw_mem = df[mem_col].dropna().values.astype(float)
    result["memory_request"] = fit_variable("memory_request", raw_mem, "memory_request")

    # --- Tasks per job ---
    print(f"  tasks_per_job: ", end="")
    raw_tasks = df["num_tasks"].dropna().values.astype(float)
    fit = fit_tasks_mixture(raw_data=raw_tasks)
    result["tasks_per_job"] = fit
    pm_w = fit["point_mass"]["weight"]
    tail_dist = fit.get("tail", {}).get("distribution", "none")
    tail_ks = fit.get("tail", {}).get("ks_statistic", "N/A")
    print(f"mixture (P(1)={pm_w:.3f}, tail={tail_dist} KS={tail_ks})")

    return result


def main() -> None:
    print("Refitting batch distributions from full Google ClusterData 2019 extracts...")
    print(f"Strategy: MLE fitting on up to {MAX_SAMPLES_FIT:,} samples per variable")
    print()

    for cell in CELLS:
        print(f"Cell {cell}:")
        result = process_cell(cell)

        out_path = JOBS_DIR / f"batch_distributions_{cell}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  -> Written to {out_path.name}")
        print()

    # Update cross-validation in workload_generator_params.json
    print("Updating workload_generator_params.json...")
    cross_val = {}
    for cell in CELLS:
        dist_path = JOBS_DIR / f"batch_distributions_{cell}.json"
        with open(dist_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        cross_val[cell] = {
            "batch_fraction": d.get("workload_mix", {}).get("batch_fraction", 0),
            "mean_duration": d.get("duration", {}).get("mean", 0),
            "duration_distribution": d.get("duration", {}).get("distribution", "unknown"),
            "inter_arrival_distribution": d.get("inter_arrival", {}).get("distribution", "unknown"),
        }

    params_path = DATA_DIR / "workload_generator_params.json"
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    params["cross_validation"] = cross_val
    params["description"] = (
        "Workload generator parameters fitted to Google ClusterData2019. "
        "Distributions fitted via MLE on 200k batch jobs per cell. "
        "Following Da Costa et al. (2016) and Grange et al. (2018) methodology."
    )
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print("  Updated")
    print("\nDone.")


if __name__ == "__main__":
    main()
