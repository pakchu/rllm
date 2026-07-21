# BLSR-288 source-support rejection

## Verdict

`BLSR-288` is **rejected before novelty and before every market outcome**.
The committed and hash-frozen evaluator produced only 54 accepted primary
entries across train and selection, well below the preregistered support
floors. The candidate identity is retired without threshold, deadline, rank,
side, hold, or support-floor repair.

Frozen artifacts:

- preregistration:
  `results/blockspace_load_settlement_relay_preregistration_2026-07-21.json`;
- evaluator freeze:
  `results/blockspace_load_settlement_relay_evaluator_freeze_2026-07-21.json`;
- source-support report:
  `results/blockspace_load_settlement_relay_support_2026-07-21.json`; and
- source-support report SHA-256:
  `82cec44fe766a406272678721b0ff5ec997dda0bae4092701a30e28b8c27f672`.

## Source and causal checks

The source path itself passed:

- 213,095 allowed-column ledger rows read;
- 2,959 complete absolute-height 72-block packets;
- first/last complete packet heights `610704..823751`;
- all six-successor confirmations contained;
- packet availability strictly increasing;
- 2,958 valid one-packet feature changes; and
- 2,838 rank-ready changes under the frozen 180/120 strict-prior midrank.

The evaluator materialized only `height`, hashes, `timestamp`, `weight`,
`total_fees`, `total_inputs`, and `total_outputs`. It did not materialize
`mediantime`, `tx_count`, `size`, or `utxo_set_change` as BLSR source values.

## Failed support

| Window / subperiod | Accepted entries | Frozen minimum |
|---|---:|---:|
| train 2021–2022 | 34 | 80 |
| 2021 | 15 | 30 |
| 2022 | 19 | 30 |
| 2021-H1 / H2 | 7 / 8 | 12 each |
| 2022-H1 / H2 | 13 / 6 | 12 each |
| selection 2023 | 20 | 35 |
| 2023-H1 / H2 | 9 / 11 | 14 each |
| 2023-Q1 / Q2 / Q3 / Q4 | 8 / 1 / 7 / 4 | 6 each |

Side support also failed:

- train LONG/SHORT: `14 / 20`, versus at least `24 / 24`;
- selection LONG/SHORT: `11 / 9`, versus at least `12 / 12`; and
- 2021 LONG/SHORT: `4 / 11`, violating the per-side/per-year floor.

Concentration failed independently:

- train maximum UTC-entry weekday share: `29.41%`, above `25%`; and
- selection maximum month share: `30%`, above `20%`.

The relay formed 68 same-sign candidates before containment/non-overlap and 54
accepted entries. All nine control clocks passed their structural, causal,
24-hour hold, split-containment, and global non-overlap checks; those controls
cannot rescue a failed primary.

## Unopened evidence boundary

Because support failed, the evaluator stopped before comparator incidence:

- comparator event rows read: `0`;
- novelty evaluated: `false`;
- BTC market rows loaded: `0`;
- funding rows loaded: `0`;
- return/PnL rows read: `0`; and
- economic simulations run: `0`.

The report's generic `failed_stages` list contains `novelty` because an
unevaluated stage has `passed=false`; the authoritative fields are
`novelty.evaluated=false` and `comparator_event_rows_read=0`. No orthogonality
or profitability conclusion was produced.

## Disposition

BLSR-288 is terminally retired under its frozen identity. Its sequential
ledger-relay concept may inform a genuinely new preregistered mechanism, but
the observed support counts may not be used to relax this candidate's packet,
rank, deadline, side, latency, hold, concentration, or support rules.
