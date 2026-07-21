# RPDS-576 refined-product divergence shock — outcome-blind preregistration

## Decision and evidence boundary

Freeze **RPDS-576**, a 48-hour BTC policy driven only by issue-local changes
in U.S. commercial-crude, gasoline, and distillate inventories from the EIA
Weekly Petroleum Status Report.

This freeze occurs before RPDS source incidence, side counts, calendar
concentration, comparator overlap metrics, BTC market data, funding, return,
PnL, absolute return, CAGR, or strict MDD is opened. The source's total 259
issues, 258 complete rows, one quarantined row, and prior EPSB
concordant-state counts were already published.

During design, one 2019 EIA source row and the first ten rows of the live pure-
clock comparator were printed to verify schemas. No RPDS predicate, mixed-state
count, RPDS clock, comparator overlap, correlation, or outcome was computed
from them. The preregistration records those 1/10 row exposures rather than
claiming a false zero-row clean room. Neither exposure selected the physical
sign topology, 48-hour hold, or novelty limits.

RPDS is not a threshold or hold repair of EPSB-1. EPSB traded only releases in
which crude, gasoline, and distillate all moved in the same direction. RPDS
uses the mutually exclusive physical state in which the two refined-product
stocks agree while crude moves oppositely. The broader branch is historically
exposed, so even a pass would be candidate-level retrospective evidence that
requires forward shadow validation, not a pristine discovery.

## Frozen immutable source

Use only:

```text
data/eia_petroleum_stock_breadth_2019_2023/
  eia_petroleum_stock_breadth_2019_2023.csv.gz
```

- panel SHA-256:
  `26cbe6a91079a64fd9bbcb1cb5e1f81e15df25e45ed2171f7c464d048b34757b`;
- source manifest SHA-256:
  `3969288900528d103016cdb0870a11269c1b352b9077faffdc61427f7fce29fb`;
- build manifest SHA-256:
  `d6813b1a5677c9222a1197343900d6b03381f35ff9db8688892b77e4cd9c0661`;
- complete rows: 258; quarantined rows: 1; and
- source interval: release years 2019–2023 only.

The support evaluator may materialize only:

```text
release_date
available_time_utc
source_complete
published_difference_consistent
commercial_crude_change_mmbbl
gasoline_change_mmbbl
distillate_change_mmbbl
```

It may not read stock levels, arithmetic recomputations, URLs, BTC prices,
funding, prior strategy returns, or 2024+ rows. A row is eligible only when
both source quality flags are true and all three published changes are finite
and nonzero. The quarantined issue cannot emit a signal or seed a delayed
control.

## Frozen physical-state mechanism

For one eligible WPSR issue, define:

```text
C = sign(commercial_crude_change_mmbbl)
G = sign(gasoline_change_mmbbl)
D = sign(distillate_change_mmbbl)
```

RPDS emits exactly when:

```text
G == D
G != 0
C == -G
```

Direction is immutable:

```text
G = D = +1 and C = -1  -> LONG BTC
G = D = -1 and C = +1  -> SHORT BTC
```

The LONG state is a crude draw accompanied by refined-product builds: refinery
throughput has converted upstream inventory into downstream surplus, a
tentative disinflationary/risk-liquidity impulse. The SHORT state is a crude
build accompanied by refined-product draws: downstream tightness persists
despite upstream accumulation, a tentative inflationary/risk-off impulse.

This is a falsifiable sign-topology hypothesis. Inventory changes do not prove
refinery throughput, final demand, inflation, monetary easing, or BTC buying.
No magnitude threshold, rolling statistic, seasonal adjustment, release
surprise, price reaction, LLM judgment, regime gate, or parameter grid is
permitted.

## Frozen causal clock and execution

- signal time: exact source `available_time_utc`, already set to 13:00 UTC on
  the calendar day after the official release date;
- entry: `signal_time + 5 minutes`, one complete five-minute latency bar;
- scheduled exit: entry plus exactly 576 five-minute bars / 48 hours;
- exposure for later evaluation: 0.5x BTCUSDT USD-M perpetual;
- chronology: sort by `(entry_time, release_date)`;
- global non-overlap: accept only an entry at or after the previous accepted
  scheduled exit; and
- split containment: release, signal, entry, and exit must all lie inside one
  declared split.

There is no stop, take profit, trailing exit, early close, score priority, or
future-source cancellation.

## Frozen source-support gates

Support uses 2019 only as source history/diagnostic. It selects nothing.

### Train: `[2020-01-01, 2023-01-01)` UTC

Every condition is required:

- 24–75 accepted events;
- at least 5 accepted events in each of 2020, 2021, and 2022;
- LONG and SHORT each at least 25% of events;
- no calendar month above 25% of events; and
- all source identity, quality, clock, containment, and non-overlap checks pass.

### Source-only selection: `[2023-01-01, 2024-01-01)` UTC

Every condition is required:

- 8–24 accepted events;
- at least 3 events in each half-year;
- at least one LONG and one SHORT;
- no calendar month above 25%; and
- all source identity, quality, clock, containment, and non-overlap checks pass.

A support failure retires RPDS-576 without changing the sign state, side, hold,
latency, split, or floor. Control density cannot replace primary support.

## Frozen source-only controls

Each control has its own chronological scheduler and the same 48-hour hold.
Controls diagnose but cannot substitute for RPDS:

1. `direction_flip`: exact side reversal on primary clocks;
2. `refined_only`: every eligible `G == D != 0` release, side `sign(G)`;
3. `crude_only`: every eligible nonzero crude release, side `-sign(C)`;
4. `epsb_concordance_48h`: all three signs equal and nonzero, side equal to
   that common sign;
5. `one_release_delay`: previous eligible RPDS side is emitted at the next
   eligible WPSR signal time; quarantined/incomplete rows clear pending state;
6. `deterministic_random_side`: SHA-256 of policy ID plus release date maps
   primary clocks to LONG/SHORT; and
7. `latency_plus_5m`: primary side with entry and exit shifted exactly five
   minutes later.

## Frozen novelty cohort and limits

Novelty is evaluated only after all primary support gates pass. Before reading
timestamps, bind these exact outcome-free clock artifacts:

| Cohort | Path | SHA-256 |
|---|---|---|
| EPSB source/control clocks | `results/eia_petroleum_stock_breadth_clocks_2026-07-17.csv.gz` | `6c6470ba90e8bd826bb566e5952755dd8a872b29c1ba0643d29e08ab23e44400` |
| Live research sleeves | `results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz` | `73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08` |
| FAR cohort | `results/cchr_far_pure_clocks_2026-07-21.csv.gz` | `2203bdb6122fbbc4eaf28b0ddf626362a6cde1a1153ff13c74722eba340f3ccf` |
| DTV cohort | `results/cchr_dtv_pure_clocks_2026-07-21.csv.gz` | `798e442f8ff4867232079cd6b500f388326b42c920297d407abbd1c4c85df225` |
| PDLH cohort | `results/cchr_pdlh_pure_clocks_2026-07-21.csv.gz` | `5001efba77620c45a4784a71a7d5ab5a3127a4549be926a581e6597ed3e0c9fa` |

On the single combined `[2020-01-01, 2024-01-01)` UTC grid, RPDS must satisfy
against each individual comparator candidate:

- exact-entry Jaccard at most 0.10;
- maximum bidirectional one-to-one containment within plus/minus six hours at
  most 0.25; and
- absolute signed occupied-exposure correlation on the five-minute union grid
  at most 0.35.

No denominator is truncated to a comparator's first or last observed event.
An unavailable comparator exposure remains zero on the combined grid; the
maximum bidirectional containment prevents that zero prefix from hiding a
high fraction of comparator events near RPDS.

Against EPSB `primary`, exact release-date and exact-entry overlap must be zero;
any nonzero overlap is an implementation error because the sign states are
mutually exclusive. A missing comparator, hash/schema drift, duplicate clock,
invalid side, timestamp outside the frozen splits, or outcome field fails
closed. No comparator return or performance artifact may be read.

## Later outcome boundary

Only a full support and novelty pass may authorize a separately committed
strict evaluator. It must use exact next-open execution, realized funding,
6 bp base and 10 bp stress cost per notional side, full-calendar CAGR including
idle cash, and strict held-path MDD from the global pre-entry high-water mark
through costs, funding, every held five-minute high/low, virtual adverse exit
cost, and actual exit.

The sequential opening is train 2020–2022, then 2023. A stage must have positive
absolute and stress return, CAGR/strict-MDD at least 3.0, strict MDD at most
15%, positive contained subperiod return, weekly clustered sign-flip
`p <= 0.10` for train (`<= 0.20` for 2023), and at least 0.25 ratio margin over
the strongest finite mechanism control. Report absolute return with every
CAGR/MDD statistic.

Only an unchanged 2023 pass may authorize a point-in-time EIA source extension
and separately sealed 2024, 2025, and recent-2026 stages. No later row may be
used to repair RPDS-576.

## Bound references

- `docs/eia-petroleum-stock-breadth-source-audit-2026-07-17.md`
- `docs/eia-petroleum-stock-breadth-preregistration-2026-07-17.md`
- `training/build_eia_petroleum_stock_breadth.py`
- `results/eia_petroleum_stock_breadth_preregistration_2026-07-17.json`
- `results/eia_petroleum_stock_breadth_support_2026-07-17.json`
