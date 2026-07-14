# CATCH-12 clock freeze — 2026-07-14

## Frozen clock

The selected support clock has been reproduced from committed, outcome-blind
code and serialized before any return evaluator is run.

- selected quantile: `0.975`
- clock:
  `results/cash_auction_transfer_catchup_handoff_clock_2026-07-14.csv`
- clock SHA-256:
  `066bf8e08267a043cc191eb436f0aa33105ab948de9f9f1edfde4d9c30de46d1`
- manifest:
  `results/cash_auction_transfer_catchup_handoff_clock_manifest_2026-07-14.json`
- manifest SHA-256:
  `f461529e14539ea4aa6e4b498ef8738f918dfb095ebbfcefb43ed127fe272ca6`
- support SHA-256:
  `454c54cb234a34b51fca12a810332039d7b21e4395f47f4e6b8ad6375370be02`
- source SHA-256:
  `00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc`

## Clock facts

- rows: **3,957**
- first signal: `2020-01-08 02:35:00`
- last signal: `2023-12-31 15:25:00`
- long / short events: **1,742 / 2,215**
- entry position: exactly one five-minute row after the signal;
- exit position: exactly 12 rows after entry;
- overlap: no entry occurs before the preceding fixed-hold exit;
- every action is `-1` or `+1`;
- scheduled exit remains strictly before `2024-01-01`;
- columns contain identifiers, timestamps, side, branch, and hold only—no
  return, future, PnL, CAGR, or MDD field.

Running the committed freeze program twice produced the same clock SHA-256.
The manifest records `outcomes_opened=false` and reproduces the frozen support
counts exactly.

## Next boundary

The event clock is now immutable. A separate evaluator implementation and its
tests must be committed and hashed before reading any post-entry price path.
That evaluator may open only 2020–2023: train is 2020–2022 and selection is
2023 with fixed H1/H2. It may not alter the clock, side, one-hour hold, cost
model, strict-MDD path convention, or statistical gate. 2024+ remains sealed.
