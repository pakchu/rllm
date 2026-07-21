# ORPB-21 — ON RRP participant-breadth residual preregistration

## Decision and evidence boundary

Freeze exactly one source-only candidate, **ORPB-21**, before computing its
event incidence, clock overlap, BTC return, funding, PnL, absolute return,
CAGR, or strict MDD.

ORPB asks whether the number of counterparties accepting the New York Fed's
daily overnight reverse repo is unusually broad relative to the total dollars
accepted. It is not a threshold or hold repair of ORFR-1: ORFR used accepted-
amount innovation, while ORPB's primary variable is the residual participation
breadth left after explaining breadth with same-facility accepted dollars.

Before this freeze, an architecture review opened the source header and two
source rows for schema confirmation. This session also opened the previously
committed ORFR preregistration/support JSON and inspected eight prospective
comparator CSV headers. Seven comparator artifacts were ultimately bound; the
header-inspection count is an evidence-access disclosure, not the number of
bound files.

During preregistration review, the evaluator cohort was made fail-closed by
projecting only candidate/clock identifiers from all seven bound artifacts:
7,104 identifier rows covering 45 required individual clocks. Entry, exit,
side, ORPB transform, residual, rank, incidence, overlap, BTC, funding, and
outcome fields remained unopened. Those artifacts disclose prior ORFR
source-only event counts, but no ORFR Stage-1 economic result was opened for
this decision.

## Frozen source

Use only the immutable panel:

```text
data/new_york_fed_overnight_rrp_2018_2023/
  new_york_fed_overnight_rrp_2018-01-01_2023-12-31.csv.gz
```

- panel SHA-256:
  `49f67ed44b7eb81fd35c17a8209cf14d6a8019d7e9f77fce8c343d1a7fb66b27`;
- build-manifest file SHA-256:
  `4f87e2219da71c94832c8708086ba01387efc145e3488b62cd3b3d07c62d8fee`;
- internal build-manifest hash:
  `de6708a85fd7626e19adb48bf89a27cf2e50cbc09f8caddb9a6f67c03ca7140a`;
- 1,498 normal afternoon operations, 1,489 complete and nine quarantined;
- operation dates 2018-01-02 through 2023-12-29; and
- official source and mechanics:
  [New York Fed Markets API](https://markets.newyorkfed.org/static/docs/markets-api.html)
  and [ON RRP FAQ](https://www.newyorkfed.org/markets/rrp_faq.html).

The support evaluator may materialize only:

```text
operation_date
result_available_at_utc
total_amount_accepted_usd
participating_counterparties
accepted_counterparties
source_complete
```

It may not read submitted amount, operation ID, later-update metadata, BTC,
funding, prior strategy returns, or post-2023 source rows. A quarantined row
exposes only its operation and availability clocks; its values remain blank,
it emits no signal, and it clears the feature window. No window may bridge it.

## Frozen primary feature

For a complete operation `t`, define:

```text
A[t] = log1p(total_amount_accepted_usd[t] / 1e9)
B[t] = log1p(accepted_counterparties[t])
```

Require exactly 21 consecutive complete prior operations since the last
quarantine. Fit ordinary least squares with an intercept on those prior rows:

```text
B[i] = alpha[t] + beta[t] * A[i] + epsilon[i],  i in t-21 ... t-1
```

The current row is excluded from the fit. If prior `A` variance is zero or any
fit value is non-finite, `t` emits no signal but remains an ordinary complete
row for the next window. Compute the out-of-sample current residual:

```text
E[t] = B[t] - (alpha[t] + beta[t] * A[t])
```

Rank `E[t]` against the 21 in-sample prior residuals from that exact fit:

```text
R[t] = (count(epsilon[i] < E[t]) + 0.5 * count(epsilon[i] == E[t])) / 21
```

Direction and abstention are immutable:

```text
R[t] <= 0.10  -> LONG BTC
R[t] >= 0.90  -> SHORT BTC
otherwise     -> abstain
```

A high residual means more accepting counterparties than accepted dollars
alone predict: a tentative systemwide breadth of cash parking and risk
aversion, mapped SHORT. A low residual means unusually concentrated facility
usage, mapped LONG as a tentative breadth release. This interpretation is
falsifiable; accepted-counterparty breadth does not prove reserve scarcity,
forced selling, or future BTC direction.

No alternative lookback, ridge penalty, robust regression, tail, interaction,
calendar filter, amount floor, level regime, LLM judgment, or crypto feature is
permitted.

## Frozen causal clock and execution

- decision: exact `result_available_at_utc`, already normal-operation close
  plus the source's conservative 15-minute publication allowance;
- entry: decision plus one complete five-minute bucket;
- exit: the next normal ON RRP operation's availability plus five minutes,
  whether that next row is complete or quarantined;
- the final source operation has no bounded next-operation exit and is omitted;
- fixed exposure: 0.5x BTCUSDT USD-M perpetual;
- base cost: 6 bp/notional/side;
- stress cost: 10 bp/notional/side;
- exact realized funding on `[entry, exit)`; and
- no stop, take profit, early close, score priority, or overlapping position.

## Frozen source-only controls

Every transform uses the same 21-prior-operation segment rule, tails, causal
clock, and next-operation exit. Controls diagnose and cannot substitute for
the primary:

1. `amount_only_tail`: strict-prior midrank of current `A` against prior `A`;
   high maps SHORT and low maps LONG;
2. `raw_accepted_breadth_tail`: strict-prior midrank of current `B` against
   prior `B`, with the same direction;
3. `participating_breadth_residual`: replace accepted counterparties with
   `log1p(participating_counterparties)` in the exact primary regression;
4. `direction_flip`: reverse every primary side on the same clock;
5. `one_release_delay`: emit each primary side at the next normal operation's
   entry and exit at the operation after that; an unavailable terminal exit is
   omitted; and
6. `deterministic_random_side`: on primary entries, SHA-256 of
   `ORPB-21-random-side-20260721|<entry ISO>`; first byte below 128 is LONG,
   otherwise SHORT.

## Frozen source-support gates

History through 2020 supplies strictly prior warm-up only and selects nothing.

### Train `[2021-01-01, 2023-01-01)`

All conditions are required:

- 50–130 accepted primary events;
- at least 20 events in each of 2021 and 2022;
- LONG and SHORT each at least 25% of train events;
- no calendar month above 20% of train events; and
- exact source identity, quarantine reset, OLS/rank, clock, split containment,
  and non-overlap checks pass.

### Source-only selection `[2023-01-01, 2024-01-01)`

All conditions are required:

- 25–80 accepted primary events;
- at least eight events in each half-year;
- LONG and SHORT each at least 20% of selection events;
- no calendar month above 25%; and
- the same integrity checks pass.

A support failure retires ORPB-21 unchanged before comparator clocks or market
outcomes are read. Control density cannot replace primary support.

## Frozen novelty cohort and limits

Only after support passes may the evaluator read the following outcome-free
clock artifacts, bound by the preregistration JSON:

- ORFR primary/control and feature clocks;
- FLCC federal-liquidity clocks;
- DFFB Treasury fiscal-flow primary/control clocks;
- SFRD secured-funding-rate clocks; and
- BDRC bank-deposit/secured-repo clocks.

The JSON freezes the exact identifier set and full-file row count for every
bound artifact. The cohort comprises three ORFR clocks (1,001 rows), one ORFR
feature clock (328), 24 FLCC clocks (3,098), one DFFB primary clock (112), six
DFFB controls (1,502), one SFRD clock (158), and nine BDRC clocks (905).
Missing or extra identifiers, missing or extra rows, or a changed per-clock
row count fails closed before any novelty statistic is computed.

On the combined `[2021-01-01, 2024-01-01)` grid, ORPB primary must satisfy:

### Same-source ORFR comparators

- exact-entry Jaccard at most 0.15 against each ORFR clock;
- maximum bidirectional one-to-one containment within one normal RRP operation
  at most 0.35;
- absolute signed occupied-exposure correlation at most 0.35; and
- absolute Spearman correlation between ORPB residual and the frozen ORFR
  amount innovation on common operation dates at most 0.35.

### Other macro comparators

Against each individual candidate clock:

- exact-entry Jaccard at most 0.10;
- maximum bidirectional one-to-one containment within plus/minus six hours at
  most 0.25; and
- absolute signed occupied-exposure correlation at most 0.35.

No denominator may be truncated to a comparator's observed prefix. Missing,
empty-required, hash-drifted, malformed, overlapping, off-grid, outcome-bearing,
or post-2023 comparator data fails closed. Novelty failure retires ORPB before
economic evaluation and cannot be repaired by dropping a comparator.

## Later sequential economic boundary

Only a full support and novelty pass may authorize a separately committed
strict evaluator. It opens 2021–2022 first and 2023 only after an exact,
unchanged Stage-1 pass.

Each stage requires positive base and stress absolute return, positive return
in every contained year/half, CAGR/strict-MDD at least 3.0, strict MDD at most
15%, weekly-cluster sign-flip `p <= 0.10`, mean gross underlying return at
least 35 bp, the frozen minimum trade/side counts, and at least 0.25
CAGR/strict-MDD margin over the strongest finite mechanism control. Absolute
return must be reported beside every CAGR/MDD statistic.

Strict MDD uses the global pre-entry high-water mark, entry/exit costs, funding,
every held five-minute high/low, favorable-before-adverse ordering, virtual
adverse exit cost, and realized exit. Full-calendar CAGR includes warm-up and
idle cash.

Only an unchanged 2023 pass may later authorize sealed post-2023 evaluation.
No support, novelty, or outcome failure permits a tail, lookback, side, hold,
calendar, regime, or feature repair.
