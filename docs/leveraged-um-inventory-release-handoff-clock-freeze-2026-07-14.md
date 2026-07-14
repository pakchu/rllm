# LURI-48 clock freeze — 2026-07-14

## Frozen clock

The selected support clock has been reproduced from committed, outcome-blind
code and serialized before any LURI return or realized funding is read.

- selected positive-basis quantile: `0.40`
- clock:
  `results/leveraged_um_inventory_release_handoff_clock_2026-07-14.csv`
- clock SHA-256:
  `50765cfed0c3ec6a0d1df18857c4e0a3e574d1aa449538c9b89cfac1fff67095`
- manifest:
  `results/leveraged_um_inventory_release_handoff_clock_manifest_2026-07-14.json`
- manifest SHA-256:
  `c4ce91ee395fcff8ffd5321ef94bfc2aabd8980f0778d7d4369d22bd81cbca68`
- support SHA-256:
  `58a8c3dfd1727b48fe7548ec3b59290060f690f7d0cce8bdb5b13a0bdfd4e3a9`
- feature source SHA-256:
  `00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc`

## Clock facts

- rows: **432**
- first signal: `2020-01-16 04:20:00`
- last signal: `2023-12-27 16:25:00`
- long / short events: **180 / 252**
- entry position: exactly one five-minute row after the completed signal;
- exit position: exactly 48 rows after entry;
- overlap: no entry occurs before the preceding fixed-hold exit;
- every action is `-1` or `+1`, branch is `luri48`, and hold is 48;
- every scheduled exit remains strictly before `2024-01-01`;
- columns contain only row identifiers, timestamps, side, branch, and hold—no
  OHLC, future return, funding, PnL, CAGR, or MDD field.

Running the committed freeze program twice produced byte-identical clock and
manifest SHA-256 values. The manifest records `outcomes_opened=false` and
reproduces all frozen support counts exactly.

## Next boundary

The 432-event primary clock is now immutable. Before any return is opened, a
separate pre-2024 realized-funding source and evaluator implementation must be
validated, committed, and hashed. The evaluator may open only 2020–2023:
train is 2020–2022 and selection is 2023 with fixed H1/H2. It may not alter
the clock, side, four-hour hold, funding endpoints, cost model, strict-MDD path
convention, or statistical gate. Calendar 2024 and later remains sealed.
