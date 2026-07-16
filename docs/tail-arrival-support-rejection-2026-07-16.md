# Tail–Arrival Absorption/Release support decision (2026-07-16)

## Decision

**Reject TAAR v1 before opening any forward trade return.** The four frozen
policies all had sufficient and reasonably balanced event clocks, but the
frozen source-quality gate failed. Consequently no 2020–2022 trade PnL, CAGR,
strict MDD, cost stress, 2023 holdout, or portfolio-correlation value was
computed.

This is a data-support rejection, not evidence that the economic mechanism is
profitable or unprofitable.

## Frozen mechanism

TAAR used an axis absent from the promoted portfolio:

- event-notional tail span (`p50`, `p99`, `max`),
- event-size dispersion (`std / mean`),
- aggregate-trade arrival irregularity (`interarrival std / mean`), and
- contemporaneous aggressive-event size asymmetry and price response.

The absorption branch faded an unusually large, irregular packet that failed
to move price in its own direction. The release branch followed one that did.
All thresholds used only eligible observations strictly before the signal bar.
The signal reserved two five-minute bars before assumed entry.

## Outcome-blind support result

| Policy | Branch | Hold | Events | 2020 | 2021 | 2022 | Short/long | Largest month |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| T01 | absorption fade | 12 bars | 230 | 80 | 83 | 67 | 121 / 109 | 6.09% |
| T02 | absorption fade | 36 bars | 220 | 77 | 81 | 62 | 114 / 106 | 6.36% |
| T03 | release follow | 12 bars | 978 | 142 | 560 | 276 | 536 / 442 | 7.87% |
| T04 | release follow | 36 bars | 846 | 130 | 474 | 242 | 466 / 380 | 7.21% |

Every policy passed the preregistered event-count, per-year count, side-share,
and event-month-concentration gates.

## Failing source gate

- Frozen grid: 315,648 five-minute rows, 2020-01-01 through 2022-12-31.
- Missing feature bars: 33.
- Full source-gap days: 5 (1,440 rows).
- Missing/gap rows plus the following 24-bar quarantine: 1,655 rows (0.5243%).
- Global unavailable limit: 2%; **passed**.
- Worst month: 2021-02, unavailable fraction 7.7381%.
- Monthly unavailable limit: 5%; **failed**.

Two source-gap days occurred in the short February 2021 calendar month. The
preregistration required each verified aggregate-trade ID gap to quarantine
the full affected day and the next 24 bars, so the threshold cannot be relaxed
after observing support. `passing_policy_ids` is therefore empty and the
selection evaluator must not read market OHLC or realized funding.

## Leakage and execution audit

- Support code read only the frozen aggTrade feature prefix ending before
  2023-01-01.
- It did not read market OHLC, future return, or post-signal funding.
- Rolling 30-day quantiles were shifted by one bar and required seven days of
  eligible prior observations.
- Episode starts excluded any same-branch active bar in the prior 12 completed
  buckets.
- Non-overlap and tail bounds reserved the frozen two-bar execution delay.
- Output hashes and Git anchors are recorded in
  `results/tail_arrival_support_manifest_2026-07-16.json`.

## Artifacts

- Preregistration: `results/tail_arrival_absorption_preregistration_2026-07-16.json`
- Frozen source manifest: `results/tail_arrival_source_manifest_2026-07-16.json`
- Support feature stream: `data/tail_arrival_support_features_2020_2022.csv.gz`
- Frozen policy clocks: `data/tail_arrival_support_clocks_2020_2022.csv.gz`
- Support decision: `results/tail_arrival_support_manifest_2026-07-16.json`

The next alpha search should use a different, reliably complete observation
axis rather than re-registering TAAR with a post-hoc weaker data-quality gate.
