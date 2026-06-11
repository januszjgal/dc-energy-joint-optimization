"""Recompute batch_fraction and refit batch-job distributions LOCALLY from the
full jobs_*.csv extracts, under the 'no-SLO tiers' deferrable definition.

Deferrable ("batch") = Borg free tier (priority <= 99) OR best-effort batch
tier (priority 100-115), i.e. simply priority <= 115 -- both tiers explicitly
have NO SLOs. Tier bounds per the authoritative trace documentation (Wilkes,
"Google cluster-usage traces v3", 2020-08 revision), which explicitly corrects
the beb range "mistakenly reported as 110-115" in Tirmazi et al. (2020),
"Borg: the Next Generation" (EuroSys '20) §2. Everything else (mid 116-119,
production 120-359, monitoring >=360) is non-deferrable service.

No BigQuery needed: reads data/jobs/jobs_{a..d}.csv directly. Mirrors the
fitting logic in extract_clusterdata2019_full.ipynb (continuous fits selected
by KS D statistic; tasks-per-job fit with a discrete model).

batch_fraction is reported three ways and the *CPU-time* proxy is stored as the
primary value, because the env splits the CPU-*usage* curve, not job counts:
  - job-count : share of jobs that are deferrable                (ignores job size)
  - cpu-req   : share of requested CPU that is deferrable        (ignores duration)
  - cpu-time  : share of CPU-seconds (cpu_request x duration)    (usage proxy)
    Unfinished jobs are charged duration = trace_end - submit_time so that
    long-running services are not dropped (which would inflate the deferrable share).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(__file__).resolve().parent.parent / "data"
JOBS = DATA / "jobs"
CELLS = ["a", "b", "c", "d"]

FINISH = 6  # collection_events terminal type for FINISH


def fit_best_distribution(data, name, candidates=("expon", "lognorm", "gamma", "weibull_min")):
    data = data[np.isfinite(data) & (data > 0)]
    if len(data) < 100:
        return {"distribution": "insufficient_data", "n_samples": int(len(data))}
    best, best_d = None, np.inf
    for dn in candidates:
        try:
            dist = getattr(stats, dn)
            params = dist.fit(data)
            ks_stat, ks_p = stats.kstest(data, dn, args=params)
            loglik = float(np.sum(dist.logpdf(data, *params)))
            aic = 2 * len(params) - 2 * loglik
            if np.isfinite(ks_stat) and ks_stat < best_d:
                best_d = ks_stat
                best = {
                    "distribution": dn,
                    "params": [float(p) for p in params],
                    "ks_statistic": float(ks_stat),
                    "ks_pvalue": float(ks_p),
                    "aic": float(aic) if np.isfinite(aic) else None,
                    "loglik": loglik if np.isfinite(loglik) else None,
                }
        except Exception:
            continue
    if best is None:
        best = {"distribution": "fit_failed"}
    best.update(_summary(data, name, "continuous"))
    return best


def fit_best_discrete_distribution(data, name):
    data = np.round(data[np.isfinite(data)]).astype(int)
    data = data[data >= 0]
    if len(data) < 100:
        return {"distribution": "insufficient_data", "n_samples": int(len(data))}
    n = len(data)
    mean, var = float(data.mean()), float(data.var())
    xs = np.arange(data.min(), data.max() + 1)
    emp = np.searchsorted(np.sort(data), xs, side="right") / n
    cand = {}
    if mean > 0:
        cand["poisson"] = (stats.poisson, (mean,))
    if mean >= 1:
        cand["geom"] = (stats.geom, (1.0 / mean,))
    if var > mean > 0:
        p = mean / var
        r = mean * p / (1.0 - p)
        if r > 0 and 0.0 < p < 1.0:
            cand["nbinom"] = (stats.nbinom, (r, p))
    best, best_d = None, np.inf
    for dn, (dist, params) in cand.items():
        try:
            ks = float(np.max(np.abs(emp - dist.cdf(xs, *params))))
            loglik = float(np.sum(dist.logpmf(data, *params)))
            if np.isfinite(ks) and ks < best_d:
                best_d = ks
                best = {
                    "distribution": dn,
                    "params": [float(x) for x in params],
                    "ks_statistic": ks,
                    "aic": float(2 * len(params) - 2 * loglik) if np.isfinite(loglik) else None,
                    "loglik": loglik if np.isfinite(loglik) else None,
                }
        except Exception:
            continue
    if best is None:
        best = {"distribution": "fit_failed"}
    best.update(_summary(data.astype(float), name, "discrete"))
    return best


def _summary(data, name, kind):
    return {
        "name": name,
        "distribution_kind": kind,
        "n_samples": int(len(data)),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "p5": float(np.percentile(data, 5)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
        "p95": float(np.percentile(data, 95)),
    }


def process_cell(cell):
    df = pd.read_csv(
        JOBS / f"jobs_{cell}.csv",
        usecols=[
            "submit_time", "end_time", "duration_sec", "priority", "terminal_type",
            "num_tasks", "avg_cpu_request", "avg_mem_request", "total_cpu_request",
        ],
    )
    pr = df["priority"]
    deferrable = pr <= 115  # free (<=99) + beb (100-115); trace docs v3

    # --- batch_fraction proxies ---
    job_frac = float(deferrable.mean())
    cpu = df["total_cpu_request"].fillna(0.0)
    cpu_req_frac = float(cpu[deferrable].sum() / cpu.sum()) if cpu.sum() > 0 else 0.0

    # CPU-time (usage proxy): charge unfinished jobs duration to trace end
    trace_end = float(df["submit_time"].max())
    finished = (df["terminal_type"] == FINISH) & (df["duration_sec"] > 0)
    eff_dur = np.where(
        finished, df["duration_sec"].fillna(0.0),
        np.maximum((trace_end - df["submit_time"]) / 1e6, 0.0),
    )
    cputime = cpu.values * eff_dur
    cpu_time_frac = float(cputime[deferrable.values].sum() / cputime.sum()) if cputime.sum() > 0 else 0.0

    d = df[deferrable].copy()
    completed = d[(d["terminal_type"] == FINISH) & (d["duration_sec"] > 0)]

    profiles = {}

    submit = np.sort(d["submit_time"].values)
    ia = np.diff(submit) / 1e6
    profiles["inter_arrival"] = fit_best_distribution(ia[ia > 0], "inter_arrival_sec")
    profiles["duration"] = fit_best_distribution(completed["duration_sec"].values, "duration_sec")
    profiles["cpu_request"] = fit_best_distribution(d["avg_cpu_request"].dropna().values, "cpu_request")
    profiles["memory_request"] = fit_best_distribution(d["avg_mem_request"].dropna().values, "memory_request")
    profiles["tasks_per_job"] = fit_best_discrete_distribution(d["num_tasks"].dropna().values, "tasks_per_job")

    profiles["workload_mix"] = {
        "definition": "no-SLO tiers: priority <= 115 (free <=99 + beb 100-115; trace docs v3, correcting Tirmazi 2020's 110-115 erratum)",
        "total_jobs": int(len(df)),
        "batch_count": int(deferrable.sum()),
        "completed_batch": int(len(completed)),
        # primary value used by the env = CPU-time (usage) proxy
        "batch_fraction": cpu_time_frac,
        "batch_fraction_jobcount": job_frac,
        "batch_fraction_cpurequest": cpu_req_frac,
        "batch_fraction_cputime": cpu_time_frac,
    }

    with open(JOBS / f"batch_distributions_{cell}.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    return profiles


def main():
    print(f"{'cell':<5}{'jobs':>10}{'defer':>9}{'bf_job':>9}{'bf_cpureq':>11}{'bf_cputime':>12}"
          f"  {'duration':>10}{'iat':>10}{'tasks':>8}")
    print("-" * 96)
    for c in CELLS:
        p = process_cell(c)
        wm = p["workload_mix"]
        print(f"{c:<5}{wm['total_jobs']:>10,}{wm['batch_count']:>9,}"
              f"{wm['batch_fraction_jobcount']:>9.4f}{wm['batch_fraction_cpurequest']:>11.4f}"
              f"{wm['batch_fraction_cputime']:>12.4f}"
              f"  {p['duration']['distribution']:>10}{p['inter_arrival']['distribution']:>10}"
              f"{p['tasks_per_job']['distribution']:>8}")
    print("\nWrote data/jobs/batch_distributions_{a..d}.json (batch_fraction = CPU-time proxy).")


if __name__ == "__main__":
    main()
