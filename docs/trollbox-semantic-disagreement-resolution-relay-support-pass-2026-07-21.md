# TSDR-72 source-support result — pass

## Result

TSDR-72 passed every frozen source-only support check. This stage opened no BTC
market row, funding row, return, PnL, private message text, or post-2022
semantic row.

- immutable result:
  `results/trollbox_semantic_disagreement_resolution_support_2026-07-21.json`;
- result hash:
  `475633a3262ee46bee30b56c6accdb71dc5ce61b40e0ba575b224cdb0ea71589`;
- artifact SHA-256:
  `e4da84347ba903c14e479b24754029352a7b7913eac32061a1820bd9428d660e`;
- primary pure-clock commitment:
  `c5bb79709c6c032712af57a30a729bbe40d20aeb29c4d8e623829590480640e0`;
- parameter search: false; and
- post-failure repair: false.

## Primary incidence

The frozen first-resolution state machine produced 255 raw candidates. Split
containment dropped none; chronological six-hour non-overlap suppressed 17;
238 remained accepted.

| Window | Events | LONG | SHORT | Active weeks | Max month share | Max weekday share |
|---|---:|---:|---:|---:|---:|---:|
| train, 2020-07 through 2021 | 163 | 70 | 93 | 67 | 9.82% | 16.56% |
| selection, 2022 | 75 | 35 | 40 | 36 | 13.33% | 20.00% |

Train quarter counts were `27, 24, 39, 21, 19, 33` from 2020-Q3 through
2021-Q4. Selection quarter counts were `22, 23, 13, 17`. The smallest quarter,
both directional sleeves, active-week breadth, month concentration, and weekday
concentration all remained inside their preregistered limits.

## Control incidence

Controls were constructed before market access and cannot replace the primary.

| Control | Train events | Selection events |
|---|---:|---:|
| exact initial plurality | 163 | 75 |
| exact direction flip | 163 | 75 |
| exact deterministic random side | 163 | 75 |
| clear-after-clear / no-disagreement | 502 | 283 |
| unresolved disagreement | 93 | 60 |
| exact relay with one-hour execution delay | 163 | 75 |

This incidence is useful but is not alpha evidence. In particular, the primary
resolution entry is drawn from the same broad attention source as retired
TBASR-24. The next mandatory stage is pure-clock export plus execution-level
novelty against TBASR and live sleeves. Market outcomes remain sealed until
that gate passes.

## Evidence boundary

The support artifact records:

```text
market_rows_loaded            = 0
funding_rows_loaded           = 0
outcome_rows_loaded           = 0
return_or_pnl_fields_read     = 0
raw_private_text_opened       = false
raw_private_text_committed    = false
post_2022_semantic_rows_loaded = 0
outcomes_opened               = false
```

The pass means only that TSDR-72 is sufficiently populated, two-sided, and
calendar-distributed to justify the already frozen novelty stage. It does not
authorize a backtest, LLM re-label, threshold search, hold search, sign change,
or production order.
