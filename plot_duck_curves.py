"""Plot duck curves (average daily price patterns) per region.

Shows how grid stress varies throughout the day in each region,
illustrating why CAISO (California) presents the strongest duck curve
and why routing workloads away from CA during the evening ramp is valuable.

Usage:
    python plot_duck_curves.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Data files ──────────────────────────────────────────────────────────────
PRICE_FILES = {
    "CAISO (US-West / California)": "data/prices/caiso.csv",
    "MISO (US-Central / Iowa)": "data/prices/miso.csv",
    "Southern Co (US-SE-1 / Georgia)": "data/prices/southern_co.csv",
    "Duke Carolinas (US-SE-2 / SC)": "data/prices/duke_carolinas.csv",
}

GLOBAL_FILES = {
    "ENTSO-E NL (Global-EU / Netherlands)": "data/prices/entso_e_nl.csv",
    "EMA Singapore (Global-Asia)": "data/prices/ema_singapore.csv",
}

STEPS_PER_DAY = 288   # 5-min intervals × 24h
ROOT = Path(__file__).resolve().parent


def compute_duck_score(price: np.ndarray, window: int = 288) -> np.ndarray:
    s = pd.Series(price.astype(np.float64))
    roll_mean = s.rolling(window, center=True, min_periods=1).mean()
    roll_std  = s.rolling(window, center=True, min_periods=1).std().fillna(1.0)
    score = (s - roll_mean) / (roll_std + 1e-6)
    return np.clip(score.values, -3.0, 3.0).astype(np.float32)


def load_daily_profiles(files: dict[str, str]) -> dict[str, dict]:
    profiles = {}
    for label, path in files.items():
        df = pd.read_csv(ROOT / path)
        price = df["price_usd_kwh"].values.astype(np.float64)
        duck  = compute_duck_score(price.astype(np.float32))

        n_days = len(price) // STEPS_PER_DAY
        price_mat = price[: n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)
        duck_mat  = duck[: n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)

        profiles[label] = {
            "price_mean": price_mat.mean(axis=0),
            "price_p25":  np.percentile(price_mat, 25, axis=0),
            "price_p75":  np.percentile(price_mat, 75, axis=0),
            "duck_mean":  duck_mat.mean(axis=0),
            "duck_p25":   np.percentile(duck_mat, 25, axis=0),
            "duck_p75":   np.percentile(duck_mat, 75, axis=0),
            "price_raw":  price,
        }
    return profiles


def make_figure(us_profiles: dict, global_profiles: dict, out_path: Path) -> None:
    hours = np.linspace(0, 24, STEPS_PER_DAY, endpoint=False)

    # ── colour palette ───────────────────────────────────────────────────────
    us_colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]
    gl_colors = ["#911eb4", "#42d4f4"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Regional Duck Curves: Average Daily Grid Stress by Location\n"
        "(Duck score = rolling 24-h price z-score; positive = grid stressed above average)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Panel A: Raw price curves (US) ───────────────────────────────────────
    ax = axes[0, 0]
    for (label, data), color in zip(us_profiles.items(), us_colors):
        short = label.split("(")[0].strip()
        ax.plot(hours, data["price_mean"] * 1000, color=color, lw=2, label=short)
        ax.fill_between(
            hours,
            data["price_p25"] * 1000,
            data["price_p75"] * 1000,
            color=color, alpha=0.15,
        )
    ax.set_title("A  –  US Regions: Raw Electricity Price", fontweight="bold")
    ax.set_ylabel("Price ($/MWh)")
    ax.set_xlabel("Hour of Day")
    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axvspan(17, 21, color="red", alpha=0.07, label="CA evening ramp")

    # ── Panel B: Duck score curves (US) ─────────────────────────────────────
    ax = axes[0, 1]
    for (label, data), color in zip(us_profiles.items(), us_colors):
        short = label.split("(")[0].strip()
        ax.plot(hours, data["duck_mean"], color=color, lw=2, label=short)
        ax.fill_between(
            hours,
            data["duck_p25"],
            data["duck_p75"],
            color=color, alpha=0.15,
        )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvspan(17, 21, color="red", alpha=0.07)
    ax.set_title("B  –  US Regions: Duck Curve Stress Score", fontweight="bold")
    ax.set_ylabel("Duck Score (z-score, clipped ±3)")
    ax.set_xlabel("Hour of Day")
    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.annotate(
        "CA evening ramp\n(17:00–21:00)",
        xy=(19, 1.4), xytext=(21, 1.8),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red", fontsize=8,
    )

    # ── Panel C: Duck score (global) ─────────────────────────────────────────
    ax = axes[1, 0]
    all_global = {**us_profiles, **global_profiles}
    all_colors  = us_colors + gl_colors
    for (label, data), color in zip(all_global.items(), all_colors):
        short = label.split("(")[0].strip()
        ax.plot(hours, data["duck_mean"], color=color, lw=2, label=short)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("C  –  All Regions: Duck Score Comparison", fontweight="bold")
    ax.set_ylabel("Duck Score (z-score, clipped ±3)")
    ax.set_xlabel("Hour of Day")
    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Panel D: Peak stress hour heatmap ────────────────────────────────────
    ax = axes[1, 1]
    all_labels = list(all_global.keys())
    short_labels = [l.split("(")[0].strip() for l in all_labels]
    duck_matrix = np.array([all_global[l]["duck_mean"] for l in all_labels])

    im = ax.imshow(
        duck_matrix,
        aspect="auto",
        extent=[0, 24, len(all_labels) - 0.5, -0.5],
        cmap="RdYlGn_r",
        vmin=-1.5, vmax=1.5,
    )
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Hour of Day")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.set_title("D  –  Duck Score Heatmap by Region & Hour", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Duck Score", fraction=0.046, pad=0.04)
    ax.axvline(17, color="red", lw=1.2, ls="--", alpha=0.7)
    ax.axvline(21, color="red", lw=1.2, ls="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("Loading price data...")
    us_profiles     = load_daily_profiles(PRICE_FILES)
    global_profiles = load_daily_profiles(GLOBAL_FILES)

    out = ROOT / "output" / "duck_curves.png"
    out.parent.mkdir(exist_ok=True)
    make_figure(us_profiles, global_profiles, out)

    # Print summary stats
    print("\nPeak duck score hour per region:")
    all_profiles = {**us_profiles, **global_profiles}
    for label, data in all_profiles.items():
        peak_step = int(np.argmax(data["duck_mean"]))
        peak_hour = peak_step / STEPS_PER_DAY * 24
        peak_score = data["duck_mean"][peak_step]
        amplitude = data["duck_mean"].max() - data["duck_mean"].min()
        print(f"  {label.split('(')[0].strip():<35} peak={peak_hour:.1f}h  "
              f"score={peak_score:+.2f}  amplitude={amplitude:.2f}")
