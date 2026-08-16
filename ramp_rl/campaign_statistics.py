"""Seed-level paired summaries for the locked PPO campaign."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import numpy as np
from scipy.stats import rankdata


BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260811


def hodges_lehmann_paired_shift(differences: list[float] | np.ndarray) -> float:
    """Median of Walsh averages for policy-minus-status-quo seed differences."""
    values = np.asarray(differences, dtype=float)
    if values.size == 0:
        raise ValueError("at least one paired difference is required")
    walsh = [(values[i] + values[j]) / 2.0 for i in range(values.size) for j in range(i, values.size)]
    return float(np.median(walsh))


def exact_two_sided_wilcoxon(differences: list[float] | np.ndarray) -> dict[str, Any]:
    """Exact sign-enumeration Wilcoxon result, with zeros excluded by definition."""
    values = np.asarray(differences, dtype=float)
    nonzero = values[values != 0.0]
    if nonzero.size == 0:
        return {
            "method": "exact two-sided Wilcoxon signed-rank",
            "nonzero_pair_count": 0,
            "positive_rank_sum": 0.0,
            "negative_rank_sum": 0.0,
            "p_value": 1.0,
        }
    ranks = rankdata(np.abs(nonzero), method="average")
    positive_rank_sum = float(ranks[nonzero > 0].sum())
    total_rank_sum = float(ranks.sum())
    centered_observed = abs(positive_rank_sum - total_rank_sum / 2.0)
    possible_positive_sums = np.asarray(
        [sum(rank for rank, sign in zip(ranks, signs) if sign) for signs in product((False, True), repeat=len(ranks))],
        dtype=float,
    )
    p_value = float(np.mean(np.abs(possible_positive_sums - total_rank_sum / 2.0) >= centered_observed - 1e-12))
    return {
        "method": "exact two-sided Wilcoxon signed-rank",
        "nonzero_pair_count": int(nonzero.size),
        "positive_rank_sum": positive_rank_sum,
        "negative_rank_sum": float(total_rank_sum - positive_rank_sum),
        "p_value": p_value,
    }


def paired_bootstrap_intervals(
    differences: list[float] | np.ndarray,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Deterministic seed-resampling percentile intervals for paired estimands."""
    values = np.asarray(differences, dtype=float)
    if values.size == 0:
        raise ValueError("at least one paired difference is required")
    if draws < 10_000:
        raise ValueError("paired bootstrap requires at least 10,000 resamples")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(draws, values.size))]
    means = samples.mean(axis=1)
    shifts = np.asarray([hodges_lehmann_paired_shift(sample) for sample in samples])

    def interval(samples: np.ndarray, estimate: float) -> dict[str, float]:
        return {
            "estimate": float(estimate),
            "lower_95": float(np.quantile(samples, 0.025)),
            "upper_95": float(np.quantile(samples, 0.975)),
        }

    return {
        "method": "seed-level paired percentile bootstrap",
        "draws": draws,
        "seed": seed,
        "mean_difference": interval(means, float(values.mean())),
        "hodges_lehmann_paired_shift": interval(
            shifts, hodges_lehmann_paired_shift(values)
        ),
    }


def paired_seed_summary(differences: list[float] | np.ndarray) -> dict[str, Any]:
    """Summarize policy-minus-status-quo impact; negative values favor the policy."""
    values = np.asarray(differences, dtype=float)
    if values.size == 0:
        raise ValueError("at least one paired difference is required")
    ranks = rankdata(np.abs(values[values != 0.0]), method="average")
    nonzero = values[values != 0.0]
    rank_total = float(ranks.sum())
    # This orientation makes positive values indicate lower policy impact (improvement).
    rank_biserial = (
        float((ranks[nonzero < 0].sum() - ranks[nonzero > 0].sum()) / rank_total)
        if rank_total else 0.0
    )
    wins = int(np.sum(values < 0.0))
    losses = int(np.sum(values > 0.0))
    ties = int(np.sum(values == 0.0))
    return {
        "difference_definition": "policy minus status-quo mean incremental ramp impact; negative favors policy because lower impact is better",
        "seed_count": int(values.size),
        "mean_paired_difference": float(values.mean()),
        "median_paired_difference": float(np.median(values)),
        "hodges_lehmann_paired_shift": hodges_lehmann_paired_shift(values),
        "seed_win_count": wins,
        "seed_loss_count": losses,
        "seed_tie_count": ties,
        "common_language_effect": {
            "definition": "share of optimizer seeds with lower policy impact than status quo, with ties counted as one half",
            "value": float((wins + 0.5 * ties) / values.size),
        },
        "rank_biserial_correlation": {
            "definition": "(negative-difference rank sum minus positive-difference rank sum) / total nonzero rank sum; positive favors policy",
            "value": rank_biserial,
        },
        "wilcoxon_signed_rank": exact_two_sided_wilcoxon(values),
        "paired_bootstrap_95": paired_bootstrap_intervals(values),
    }


def slope_diagnostic(
    interaction_counts: list[int] | np.ndarray,
    impacts: list[float] | np.ndarray,
    *,
    selector: Callable[[np.ndarray], np.ndarray],
    label: str,
) -> dict[str, Any]:
    """A descriptive linear slope only; it intentionally has no p-value."""
    x = np.asarray(interaction_counts, dtype=float)
    y = np.asarray(impacts, dtype=float)
    mask = selector(x)
    if int(mask.sum()) < 2:
        return {
            "window": label,
            "point_count": int(mask.sum()),
            "impact_slope_per_interaction": None,
            "interpretation": "insufficient aligned rollout points; no inferential claim",
        }
    slope = float(np.polyfit(x[mask], y[mask], 1)[0])
    return {
        "window": label,
        "point_count": int(mask.sum()),
        "impact_slope_per_interaction": slope,
        "interpretation": (
            "negative slope means continued raw incremental-impact improvement; "
            "this descriptive learning-curve diagnostic makes no significance claim"
        ),
    }
