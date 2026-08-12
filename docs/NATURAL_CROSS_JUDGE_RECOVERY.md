# Natural Cross-Judge Ceiling Recovery

## Trigger

The frozen GPT-5 Pro synthetic fixture completed in one call for `$0.05307`, but
the first benchmark request did not complete within the 512-token output ceiling.
The runtime accepted and saved **zero** benchmark responses. It wrote one fail-closed
marker with a `$0.096` conservative cost reservation and stopped automatically.

No reader answer, Claude label, GPT label, or provider response content was inspected
to choose the recovery. The original fixture and failure marker remain byte-for-byte
preserved and are hash-bound in the recovery protocol.

## Recovery

- Preserve the exact 200-output sample, prompt, JSON schema, metrics, and 0.85 gate.
- Migrate to pinned `gpt-5.1-2025-11-13` before any benchmark label is accepted.
- Keep `high` reasoning and increase the output ceiling to 4,096 tokens.
- Keep zero transport retries, zero output repair, and zero selective reruns.
- Use a separate private runtime directory.
- Count the prior `$0.05307` fixture and `$0.096` failed-call reservation against the
  original cumulative `$20.00` hard cap.

At published prices, each recovery request reserves `$0.04384`: 2,304 input tokens
at `$1.25/M` plus 4,096 output tokens at `$10/M`. The prior budget, new `$0.10`
fixture allowance, and all 200 call reservations total `$9.01707`. This is a budgeted
provider-compatibility recovery, not a response-dependent model comparison.

## Commands

```powershell
python scripts/run_natural_cross_judge_audit.py validate `
  --protocol experiments/natural_cross_judge_recovery_protocol.json

python scripts/run_natural_cross_judge_audit.py fixture `
  --protocol experiments/natural_cross_judge_recovery_protocol.json `
  --runtime tmp/natural_cross_judge_recovery

python scripts/run_natural_cross_judge_audit.py execute `
  --protocol experiments/natural_cross_judge_recovery_protocol.json `
  --runtime tmp/natural_cross_judge_recovery

python scripts/run_natural_cross_judge_audit.py analyze `
  --protocol experiments/natural_cross_judge_recovery_protocol.json `
  --runtime tmp/natural_cross_judge_recovery
```
