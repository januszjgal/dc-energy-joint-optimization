# V4R corrected-provenance verification commands

No command below trains, tunes, selects, evaluates, or opens the sealed test.

```powershell
python -m unittest tests.ramp_v6.test_provenance_hash_contract -v
python scripts\reseal_ramp_rl_v4r_provenance.py verify `
  --log output\ramp_rl_v6\recovered_v4r_resealed_v2\current_worktree_verification.log
```

Clean checkout verification used detached worktrees at evidence commit `aab5e6e`.
The gitignored binary root was supplied read-only from the current worktree.

```powershell
$binaryRoot = (Resolve-Path 'models\ramp_rl_v6\recovered_v4r').Path

git config extensions.worktreeConfig true
git -c core.autocrlf=false worktree add --detach $falsePath aab5e6e
git -C $falsePath config --worktree core.autocrlf false
git -C $falsePath status --porcelain
Push-Location $falsePath
python scripts\reseal_ramp_rl_v4r_provenance.py verify `
  --binary-root $binaryRoot --log $falseLog
Pop-Location

git -c core.autocrlf=true worktree add --detach $truePath aab5e6e
git -C $truePath config --worktree core.autocrlf true
git -C $truePath status --porcelain
Push-Location $truePath
python scripts\reseal_ramp_rl_v4r_provenance.py verify `
  --binary-root $binaryRoot --log $trueLog
Pop-Location
```

Regression commands:

```powershell
python -m unittest discover -s tests\ramp_v6 -t . -p "test_*.py" -v
python -m unittest tests.ramp_v6.test_rl_integration.RampRLIntegrationTests -v
python -m unittest `
  tests.ramp_v6.test_v4_ensemble.FrozenV4ProtocolTests `
  tests.ramp_v6.test_v4r_recovered_ensemble.RecoveredV4RProtocolTests -v
```
