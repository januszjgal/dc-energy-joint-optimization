# Recovered-policy V4R corrected provenance

Status: **canonical corrected provenance**.

This package supersedes only the checkout-byte hash chain under `output/ramp_rl_v6/recovered_v4r/`.
The protocol, controller, model, data, factory, forecast identities, chronology, and numerical results are unchanged.

- Hash contract: `dc-energy-provenance-sha256-v2`
- Recovery commit: `d1d4828c32bada1ba1e852d0479c37bb71164561`
- Experimental source commit: `46329fe765f596f84eeac71f061dcbd191a90583`
- Validation strict pass: **True**
- Validation mean incremental ramp impact: `-1.4086907197938803e-05`
- Validation energy cost ratio: `0.9858280672779312`
- Sealed test opened exactly once: **True**
- Sealed test strict pass: **True**
- Test mean incremental ramp impact: `-1.4258710514255097e-05`
- Test energy cost ratio: `0.9775584402489343`
- One-GW robustness mean incremental ramp impact: `-1.707526521658319e-05`
- C-H robustness mean incremental ramp impact: `-1.713557824325034e-05`
- Training, retraining, retuning, validation evaluation, and sealed-test evaluation performed by this reseal: **false**

Verify with:

```text
python scripts/reseal_ramp_rl_v4r_provenance.py verify
```
