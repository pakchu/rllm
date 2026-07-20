# DFFB-601 outcome-blind mechanism preregistration — 2026-07-21

## Status and boundary

This document freezes one singleton **Daily Fiscal Flow Breadth** mechanism,
`DFFB-601`, after the DTS source audit passed and before any DFFB signal
incidence or BTC outcome is opened.  It is subordinate to the source-axis
decision in
`daily-treasury-fiscal-flow-breadth-source-axis-decision-2026-07-20.md`.

This work unit may read the frozen source and audit JSON metadata, hash the
frozen source bytes, and read only the header of each prior-strategy comparator
clock.  It may not read normalized DTS value rows, schema-transition rows,
comparator clock rows, market or funding rows, returns, labels, PnL, equity,
CAGR, or MDD.  In particular, this document contains no event count, direction
count, support result, correlation, or performance result.

The generator enforces this boundary with a physical-path and operation
allowlist. Every full-byte hash, JSON metadata parse, and header-only read is
recorded in a deterministic access ledger; an unlisted path or operation fails.
The emitted boundary counters are derived from that ledger rather than written
as an unaudited assertion. The generator imports no network, database, or
subprocess client.

## Frozen hypothesis

Many simultaneous unusually large Treasury withdrawals and debt redemptions
represent a broad fiscal-liquidity injection, whereas many simultaneous
unusually large deposits and debt issues represent a broad fiscal-liquidity
drain.  DFFB-601 goes long only when cash-flow and debt-flow breadth both occupy
their strict-prior upper tails, and short only when both occupy their
strict-prior lower tails.  Total TGA change and total net cash are prohibited
from the primary signal.

## Frozen input universe

The primary mechanism uses only `today_amount_usd_millions` from the hash-bound
pre-2024 DTS normalized source. The parser preserves the signed integer printed
in the PDF; it does not apply a cash-flow sign convention. The table side is the
economic orientation: a positive withdrawal or redemption is injection
activity, a positive deposit or issue is drain activity, and a negative printed
amount is a correction/reversal on that named side and remains negative. No
absolute value or side-dependent sign transform is applied. Month-to-date and
fiscal-year-to-date amounts are audit-only and prohibited. The four sides are:

| Table | Side |
|---|---|
| `II` | `deposit` |
| `II` | `withdrawal` |
| `IIIA` | `issue` |
| `IIIA` | `redemption` |

A row is admissible only when `row_kind == "detail"`, its table/side pair is in
the table above, and neither its raw nor normalized label is excluded by the
following frozen rules:

1. form `exclusion_key` by NFC normalization, Unicode-dash conversion to `-`,
   case-folding, whitespace collapse, and trim; reject keys beginning `total `,
   `sub-total `, `subtotal `, `net change`, or `change in balance`;
2. reject exact keys `treasury general account total deposits`, `treasury
   general account total withdrawals`, `sub-total deposits`, and `sub-total
   withdrawals`;
3. reject Table-II keys beginning `public debt cash issues` or `public debt cash
   redemp`; and
4. reject exact keys `transfers from depositaries`, `transfers to depositaries`,
   `transfers from federal reserve account (table v)`, `transfers to federal
   reserve account (table v)`, `transfers from tga (table v)`, and `transfers to
   tga (table v)`.

The exclusion is tested against both raw and normalized labels.  It is not
legal to add another exclusion after source incidence is observed.

## Causal category identity and missingness

The category identity is
`(table_id, side, canonical_parent_section, canonical_category_label)`. Both
`parent_section` and `normalized_category_label` first receive the exact
`exclusion_key` transform above, then their identity key removes every character
other than ASCII `a-z` and `0-9`. The raw label is never used as the identity;
it is only a second exclusion check. A collision of two retained rows under one
identity in the same report fails the support build.

This intentionally merges only spelling changes that are identical after
case/punctuation normalization.  A substantive label rename, parent-section
change, table move, or side move creates a new identity.  The full-sample schema
transition artifact is audit evidence only; it may not retrospectively create,
merge, kill, or backdate a category in feature computation.  Thus category
birth is the first causal report in which the identity is printed.

For a category already born strictly before report `t`, absence of its row at
`t` means an economic zero for that report.  A row that is printed with a
missing/null amount is not absence and is never converted to zero.  First
appearance at `t` is not prior knowledge and cannot be ranked at `t`.

## Strict-prior category ranks

For every known category and report `t`, use the exact preceding 60 DTS report
dates after that category's causal birth.  The current report is excluded.
Within those dates, a printed finite signed amount is its integer million-dollar
value and an absent known category is zero.  A printed null in the current or
60-report window makes the category rank non-computable at `t`.

A category is history-eligible only when at least 12 of the exact 60 prior
reports print a non-null amount for that category.  Its strict-prior midrank is:

```text
rank60(t) = (
    count(prior_value < current_value)
    + 0.5 * count(prior_value == current_value)
) / 60
```

Equality is exact integer equality.  There is no rounding, clipping,
interpolation, forward fill, epsilon, winsorization, or self-inclusion.

For each side at `t`, the denominator is every history-eligible category on
that side.  If the denominator is empty, or any denominator member has a
non-computable rank, the side breadth is non-computable.  Otherwise:

```text
side_breadth(t) = mean(rank60_i(t) >= 0.75)
cash_impulse(t)  = withdrawal_breadth(t) - deposit_breadth(t)
debt_impulse(t)  = redemption_breadth(t) - issue_breadth(t)
```

This equal-category breadth construction prevents a single large category or a
published total from dominating the primary signal.

## Strict-prior impulse ranks and singleton event

Each impulse is ranked against the exact preceding 126 **computable impulse
values of the same series**.  The current value is excluded.  Reports with a
non-computable impulse do not enter that impulse's 126-value history.  The
formula is:

```text
rank126(x_t) = (
    count(prior_x < x_t) + 0.5 * count(prior_x == x_t)
) / 126
```

The frozen singleton signal is:

```text
LONG  iff cash_rank126 >= 0.75 and debt_rank126 >= 0.75
SHORT iff cash_rank126 <= 0.25 and debt_rank126 <= 0.25
NONE  otherwise
```

No alternate sign, rank threshold, lookback, category grouping, vote rule, or
model is registered.

## Causal execution

- Warm-up reports may populate priors but cannot emit an event.
- Only source rows labelled `train` or `selection` may emit an event.
- A `boundary_quarantine` row can neither populate a pre-2024 feature nor emit
  an event.
- Decision time is the frozen `source_available_not_before_utc`.
- Entry time is the frozen `earliest_execution_time_utc`, normally five minutes
  later.
- Entry is the next five-minute bar open at that timestamp.
- Scheduled exit is exactly 24 hours, or 288 five-minute bars, after entry.
- Entry and scheduled exit must both be contained in the declared UTC split;
  otherwise the candidate is dropped.
- Candidate events sort by `(entry_time, record_date)`; accept the first and
  suppress every later candidate with `entry_time < previous_exit_time`.
  Entry exactly at the previous exit is allowed.  No score priority,
  replacement, or overlapping position is allowed.
- Notional leverage is fixed at `0.5x`.
- Base cost is 6 bp per notional per side; stress cost is 10 bp per notional per
  side.  Exact funding is entry-inclusive and exit-exclusive with fixed entry
  quantity.

## Frozen source-only controls

Every control uses the same causal availability, stage boundaries, 24-hour
hold, chronological non-overlap, and `0.5x` size unless stated otherwise.

1. **cash-only:** upper/lower 0.75/0.25 tails of `cash_rank126`, without debt
   confirmation;
2. **debt-only:** upper/lower 0.75/0.25 tails of `debt_rank126`, without cash
   confirmation;
3. **total-net-cash:** on each report compute published Table-II total
   withdrawals minus published Table-II total deposits, rank it against exactly
   126 prior computable total-net-cash reports with the same midrank, and apply
   the same 0.75/0.25 tails; this total is prohibited from the primary signal;
4. **direction-flip:** the accepted primary clock with every side multiplied by
   `-1`, diagnostic only;
5. **one-report-delay:** apply each accepted primary side at the next report's
   frozen entry time and exit 24 hours later; drop when no next in-stage report
   exists and rebuild chronological non-overlap; and
6. **deterministic-random-side:** use the accepted primary entry/exit clock and
   side `LONG` when the first byte of
   `SHA256("DFFB-601|20260721|" + entry_time_utc)` is below 128, otherwise
   `SHORT`.

Controls are frozen before their incidence or outcomes are read.  Controls do
not replace a failed primary candidate.

## Frozen source-only comparator clocks

The support builder may read only the allowlisted columns below and must prove
that its materialized column set equals the relevant allowlist. It must reject
any requested, selected, or materialized column whose lowercase alphanumeric
normalized name contains
`return`, `pnl`, `profit`, `equity`, `cagr`, `mdd`, `drawdown`, `sharpe`,
`sortino`, `price`, `funding`, or `outcome`.

1. **FLCC/H.4.1-TGA primary clocks:** read
   `candidate_id,clock_name,signal_time,entry_time,exit_time,side` from the
   hash-bound FLCC clock; filter `clock_name == "primary"`; create one
   comparator per `candidate_id` and a union decision-date comparator.  No
   candidate is selected by return.
2. **TADI primary clock:** read
   `auction_date,decision_time,entry_time,scheduled_exit_time,side,clock_mode`
   from the hash-bound TADI clock and filter `clock_mode == "primary"`.
3. **Official auction/settlement calendar:** use the hash-bound normalized
   original nominal-auction panel only to define eligible `(auction_date,
   cusip)` keys, then join the hash-bound raw TreasuryDirect pages on
   `(auctionDate,cusip)`.  Read only `auctionDate,issueDate,cusip,securityType,
   originalSecurityTerm,reopening`; retain original nominal fixed-rate
   2y/3y/5y/7y/10y/20y/30y rows represented in the normalized panel; the
   comparator date set is the union of auction and issue dates.
4. **DTS total-net-cash control:** use the control definition above.

For every non-empty comparator, form unique New York decision dates.  DFFB uses
the New York date of `source_available_not_before_utc`; FLCC uses `signal_time`,
TADI uses `decision_time`, and official calendar dates retain their published
New York civil date.  Before any outcome is opened, reject DFFB-601 if **any**
comparator has either:

```text
decision_date_jaccard > 0.30
fraction_of_DFFB_dates_within_plus_or_minus_one_US_business_day > 0.50
```

An empty comparator, missing allowlisted field, duplicate-key ambiguity, or
unparseable timestamp fails.  U.S. business-day distance uses the same frozen
federal-holiday calendar as the DTS source builder.

After the 24-hour hold is frozen and still before outcomes, build signed
occupied-exposure series on a complete five-minute UTC grid for the common
clock span of DFFB and each prior primary strategy.  An interval is
entry-inclusive and exit-exclusive; flat is 0, long is +1, and short is -1.
Reject when absolute Pearson correlation exceeds `0.40`, or when overlap is
empty or either series has zero variance.  Apply this separately to every FLCC
primary `candidate_id` and to TADI primary.

## Frozen source-only support floors

Counts are accepted primary entries after rank readiness, stage containment,
and global chronological non-overlap.

| Gate | Floor/cap |
|---|---:|
| train 2021-2022 total | at least 24 |
| each train calendar year | at least 8 |
| train long | at least 6 |
| train short | at least 6 |
| train maximum single-month share | at most 0.25 |
| selection 2023 total | at least 12 |
| each selection half-year | at least 4 |
| selection long | at least 3 |
| selection short | at least 3 |
| selection maximum single-month share | at most 0.33 |

Every novelty and occupied-exposure gate above must also pass.  Failure rejects
DFFB-601 without opening BTC data.  There is no support-floor relaxation or
incidence-driven repair.

## Frozen outcome sequence and performance gates

Only after a committed source-only support artifact passes may a separate
strict evaluator be committed and hash-frozen.  It then opens outcomes in this
order and stops permanently at the first failure:

1. train `[2021-01-01, 2023-01-01)`;
2. selection `[2023-01-01, 2024-01-01)`; and
3. only after both pass, a source extension under this exact parser and policy
   may create later report-only windows.  Later windows cannot select or repair
   the candidate.

Each opened constituent must have positive absolute return, CAGR/strict-MDD at
least `3.0`, strict MDD at most `15%`, positive stress-cost absolute return, and
a one-sided weekly entry-cluster sign-flip p-value at most `0.10` using 100,000
draws and seed `20260721`.  Train years 2021 and 2022 and selection half-years
2023H1 and 2023H2 must each have positive absolute return.  Train long-only and
train short-only contributions must each be positive.

On both train and selection, primary CAGR/strict-MDD must exceed every
support-ready cash-only, debt-only, and total-net-cash control by at least
`0.25`.  Direction-flip, one-report-delay, and deterministic-random-side are
reported diagnostics and cannot replace the primary.  CAGR uses the entire
declared wall-clock window including idle cash.  Strict MDD uses the global and
pre-entry high-water mark over every held five-minute bar, with adverse OHLC
ordering, exact funding, entry/exit costs, and hypothetical adverse-exit cost.

## Stop rule

Stop at the first source binding, support, novelty, occupied-exposure, train, or
selection failure.  Once source incidence is derived, no sign flip, threshold
search, label regrouping, exclusion edit, rank-window edit, support-floor edit,
hold grid, latency change, control replacement, model selection, or failed-rule
inversion is permitted.  DFFB-601 is research-only and cannot be promoted to
live or shadow trading by this preregistration.
