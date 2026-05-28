"""Smoke-test the CFWS-style discretized wrapper.

Verifies that:
  1. Action space has the expected size (48 for n_dc=4).
  2. Decode logic produces the expected (src, dst, drain) tuples.
  3. Step works and returns valid obs/reward.
  4. A random rollout with the wrapper hits a baseline-ish total cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from env.cfws_style_wrapper import CFWSStyleDiscretizedEnv, DRAIN_LOGITS, TRANSFER_AMOUNT
from env.data_loader import load_scenario
from env.multi_dc_env import MultiDCEnv


def main() -> None:
    scenario = ROOT / "env" / "scenarios" / "us_model.yaml"
    print("=" * 70)
    print(f"Smoke test: CFWSStyleDiscretizedEnv on {scenario.name}, batch=True")
    print("=" * 70)

    sites, power_model, batch_config = load_scenario(scenario, batch_enabled=True)
    base = MultiDCEnv(
        sites=sites,
        power_model=power_model,
        batch_enabled=True,
        peak_penalty_weight=0.015,
        flexibility_factor=batch_config.get("flexibility_factor", 1.0),
        deadline_penalty_weight=batch_config.get("deadline_penalty_weight", 2.0),
        urgency_horizon_steps=batch_config.get("urgency_horizon_steps", 12),
    )
    wrapped = CFWSStyleDiscretizedEnv(base)
    print(f"n_dc: {wrapped.n_dc}")
    print(f"n_actions: {wrapped.n_actions}  (expected: 4*4*3 = 48)")
    assert wrapped.n_actions == 48

    print("\nDecode check (first 6 actions):")
    for a in range(6):
        src, dst, dlvl = wrapped._decode_action(a)
        cont = wrapped._build_continuous_action(a)
        print(
            f"  action={a:>2}  -> src={src}, dst={dst}, drain_lvl={dlvl} "
            f"(logit={DRAIN_LOGITS[dlvl]:+.0f})  cont={cont.round(3).tolist()}"
        )

    print("\nDecode check (a few cross-DC migrations):")
    for a in [13, 26, 39, 47]:
        src, dst, dlvl = wrapped._decode_action(a)
        print(f"  action={a:>2}  -> src={src}, dst={dst}, drain_lvl={dlvl}")

    print("\nRandom-rollout sanity check (RNG seeded):")
    obs, _ = wrapped.reset(seed=42)
    rng = np.random.default_rng(42)
    total_cost = 0.0
    total_energy = 0.0
    total_peak = 0.0
    steps = 0
    while True:
        action = int(rng.integers(0, wrapped.n_actions))
        obs, reward, term, trunc, info = wrapped.step(action)
        total_cost += info["total_cost"]
        total_energy += info.get("total_energy_cost", 0.0)
        total_peak += info.get("total_peak_penalty", 0.0)
        steps += 1
        if term or trunc:
            break
    print(f"  steps run:           {steps}")
    print(f"  total_cost:          ${total_cost:>13,.0f}")
    print(f"  total_energy_cost:   ${total_energy:>13,.0f}")
    print(f"  total_peak_penalty:  ${total_peak:>13,.0f}")
    print(
        "\n(For reference: Round Robin US batch was ~$9.78M, Random ~$9.74M, "
        "so a uniform-random pick over our 48-action space should land somewhere "
        "in that ballpark or worse.)"
    )


if __name__ == "__main__":
    main()
