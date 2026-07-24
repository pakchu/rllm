# SABLE-8 source-support retirement

Date: 2026-07-25

## Decision

**RETIRE the exact SABLE-8 candidate before reward construction.**

The official write-once command was:

```bash
.venv/bin/python -m training.build_sable8_symbolic_alpha_braid_support
```

It exited during physical market projection with:

```text
RuntimeError: SABLE fresh open_interest must be positive
```

The frozen source contains at least one row where
`open_interest_available=1` does not imply positive open interest. That breaks
the field-level source invariant fixed by the boundary and preregistration.
The implementation therefore cannot construct the physically valid pre-2024
market cut required by Stage 0.

Machine-readable evidence:
[`sable8_symbolic_alpha_braid_source_retirement_2026-07-25.json`](../results/sable8_symbolic_alpha_braid_source_retirement_2026-07-25.json).

## Why this is not repaired

Possible repairs are rejected for the exact candidate:

- treating zero/nonpositive OI as fresh changes the availability semantics;
- silently replacing it with missing changes the frozen fused source contract;
- dropping offending rows changes the canonical eight-hour process and rank
  history;
- replacing the OI source after seeing the failure creates a new candidate;
- excluding OI changes the fourteen-token language; and
- lowering a support threshold does not repair a contradictory source flag.

Any future symbolic target-position candidate must either bind a source where
fresh OI is strictly positive or preregister OI as independently missing before
reading support values. It must use a new identity rather than `SABLE-8`.

## Outcome boundary

The failure happened before:

```text
token incidence calculation
future return construction
reward construction
funding cash-flow evaluation
market outcome evaluation
model training
2023 candidate outcome opening
2024+ source or outcome opening
```

No cut, token table, support report, reward, checkpoint, or economic result was
persisted. The partial source read is disclosed in the retirement JSON and is
not relabeled as a completed Stage 0 run.

## Result

SABLE-8 provides no alpha evidence. It is a source-contract rejection, not a
negative profitability backtest. Stage 0.5, Gemma training, and historical
evaluation are permanently unauthorized for this identity.
