"""Grange et al. (2018) synthetic batch-workload generator — reproduction of Listing 1.

Faithful reproduction of the workload generator published as Listing 1 of:
  Léo Grange, Georges Da Costa, Patricia Stolf (2018),
  "Green IT scheduling for data center powered with renewable energy",
  Future Generation Computer Systems 86, pp. 99-120.
  https://doi.org/10.1016/j.future.2018.03.049
which instantiates the parameterized model of:
  G. Da Costa, L. Grange, I. De Courchelle (2016),
  "Modeling and generating large-scale Google-like workload",
  7th Int. Green and Sustainable Computing Conf. (IGSC), Hangzhou, pp. 1-7.
  https://doi.org/10.1109/IGCC.2016.7892623

Per-task draws (scipy.stats), exactly as in Listing 1:
  inter-arrival    ~ modified Pareto  (pareto(4, loc=1) * 3 * dynamism)
  execution time   ~ log-normal       (makespan; median = mass/disparity, mean = mass)
  priority         ~ truncated exponential (rate 6, kept in [0, 1]) -> due-date class

Tunable knobs (Grange's reported values in brackets):
  mass      = mean task execution time, seconds         [1700]
  disparity = mean / median execution-time ratio        [3.8]
  dynamism  = inter-arrival scale parameter, seconds     [72]
  ratioTask = batch fraction (1.0 = batch only)          [1.0]

This is the *2011-era* Google parameterization we benchmark our pipeline against.
Our own fits to the 2019 trace (free+beb tier) live in
data/jobs/batch_distributions_*.json and are sampled by
env/workload_generator.py:BatchArrivalGenerator. See output/thesis_overview.md
§2.2 / §3.8 for how the two relate (same generator family; we re-fit the newer
trace and add discrete task-count + per-task resource-request distributions).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import expon, lognorm, pareto

# Grange et al. (2018) reported parameters
MASS = 1700.0       # mean task execution time (s)
DISPARITY = 3.8     # mean / median execution-time ratio
DYNAMISM = 72.0     # inter-arrival scale (s)


def lognorm_params_from(mass: float = MASS, disparity: float = DISPARITY) -> tuple[float, float]:
    """Derive scipy lognorm (s, scale) from Grange's mass/disparity.

    median = scale = mass / disparity, and mean = scale * exp(s^2/2) = mass
    => s = sqrt(2 * ln(disparity)). For (1700, 3.8) this reproduces Grange's
    literal Listing 1 values s=1.634, scale=447.
    """
    s = float(np.sqrt(2.0 * np.log(disparity)))
    scale = mass / disparity
    return s, scale


def get_next_task(
    prev_submission: float,
    dynamism: float = DYNAMISM,
    mass: float = MASS,
    disparity: float = DISPARITY,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """One task = (submission_time, priority, makespan). Mirrors Listing 1."""
    if rng is None:
        rng = np.random.default_rng()
    arrival = float(pareto.rvs(4, loc=1, random_state=rng)) * 3 * dynamism  # modified Pareto
    submission = prev_submission + arrival
    while True:  # truncated exponential, rate 6, kept in [0, 1]
        priority = float(expon.rvs(scale=1.0 / 6.0, random_state=rng))
        if priority <= 1.0:
            break
    s, scale = lognorm_params_from(mass, disparity)
    makespan = float(lognorm.rvs(s, scale=scale, random_state=rng))
    return submission, priority, makespan


def generate_workload(
    duration_s: float,
    seed: int = 42,
    dynamism: float = DYNAMISM,
    mass: float = MASS,
    disparity: float = DISPARITY,
) -> list[tuple[float, float, float]]:
    """Generate (submission, priority, makespan) tasks until submission > duration_s."""
    rng = np.random.default_rng(seed)
    tasks: list[tuple[float, float, float]] = []
    t = 0.0
    while True:
        t, prio, mk = get_next_task(t, dynamism, mass, disparity, rng)
        if t > duration_s:
            break
        tasks.append((t, prio, mk))
    return tasks


def main() -> None:
    tasks = generate_workload(7 * 24 * 3600, seed=42)  # one simulated week
    sub = np.array([x[0] for x in tasks])
    mk = np.array([x[2] for x in tasks])
    iat = np.diff(np.sort(sub))
    s, scale = lognorm_params_from()
    print("Grange et al. (2018) Listing 1 generator — 1 week")
    print(f"  tasks generated: {len(tasks)}")
    print(f"  derived lognorm: s={s:.3f}, scale={scale:.1f}  (Grange: s=1.634, scale=447)")
    print(f"  inter-arrival:   mean={iat.mean():.1f}s  median={np.median(iat):.1f}s")
    print(f"  makespan:        mean={mk.mean():.1f}s  median={np.median(mk):.1f}s"
          f"  (targets: mass={MASS:.0f}, median={MASS/DISPARITY:.0f})")


if __name__ == "__main__":
    main()
