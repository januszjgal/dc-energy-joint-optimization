# Canonical v5 TD3+BC evidence package

- Source commit: `81d50713b85e5f96809b37c16855289d13b1ad4d`
- Protocol: `v5-offpolicy-td3bc-frozen-v3` (`af656597d178f64ccf1f37ce635e3d1c9dc9167f13efe104deee86b44c4e1349`)
- BC-only campaign: `td3bc_bconly_frozen_v3`
- Post-RL campaign: `td3bc_postrl_frozen_v3`
- Global manifest: `output\offpolicy_v5_continuous\final_td3bc_manifest_global_v3.json`
- US manifest: `output\offpolicy_v5_continuous\final_td3bc_manifest_us_v3.json`
- Build command: `python scripts\build_offpolicy_evidence_v5.py --bc-campaign td3bc_bconly_frozen_v3 --postrl-campaign td3bc_postrl_frozen_v3 --seeds 301 302 303 304 305 --workers 4 --suffix v3`
- Teacher is absent during reward updates and inference.
- Normal constraint-decoder adjustment and emergency fallback are reported separately.
