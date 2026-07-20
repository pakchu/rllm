# BDRC-864 mechanism decision — bank-deposit / secured-repo concordance

## Decision and evidence boundary

The next standalone BTC candidate is **BDRC-864**, a 72-hour policy that acts
only when two independently published USD-liquidity observations agree:

1. Federal Reserve H.8 bank balance-sheet stress or relief; and
2. New York Fed SOFR tightening or easing over five published observations.

This file freezes the economic object, direction, timing, support boundary,
controls, and no-repair rule before opening BDRC source incidence. It reads no
BTC OHLC, funding cash flow, forward return, PnL, equity, CAGR, MDD, existing
alpha outcome, or post-2023 source value.

BDRC is authorized only for outcome-blind preregistration and support testing
at this stage. It is not authorized for market evaluation or trading.

## Why this is a separate mechanism

H8DM-1 tested a tail of the H.8 composite alone with a 48-hour hold and failed
its 2020-2022 performance gate. SFRD-1 tested tail changes in SOFR alone with a
five-day hold and also failed. FLCC-1 used H.4.1 central-bank balance-sheet
components rather than commercial-bank H.8 conditions.

BDRC does not lower a failed H8DM or SFRD threshold, invert either failed
policy, or reuse either hold. Its falsifiable claim is narrower: a weak H.8
balance-sheet condition is actionable only when the independently published
secured-repo rate moves in the same stress direction. The SOFR leg is a
confirmation variable, not a second capital sleeve or an outcome-selected
gate.

The two parent signals are known to be weak. Agreement can reduce false
positives, but it can also merely select a regime or duplicate the H.8 clock.
The frozen component, stale-state, disagreement, and parent-clock diagnostics
below are required to distinguish those cases.

## Frozen source contract

### Federal Reserve H.8

- panel:
  `data/fed_h8_deposit_migration_2017_2023/fed_h8_deposit_migration_2017_2023.csv.gz`
- panel SHA-256:
  `c8d1bfb0bbd13ef6d35f09ad7367ef8d2d5bb28981376223b735746ade68a572`
- build-manifest SHA-256:
  `1f0a194e628ab9c44c23fc4a923145dcf89a62bface745cc36872eeee919eda9`
- rows / frozen coverage: 365 releases, 2017-01-06 through 2023-12-29
- official release archive: <https://www.federalreserve.gov/releases/h8/>
- H.8 definitions: <https://www.federalreserve.gov/releases/h8/about.htm>
- technical and revision notes:
  <https://www.federalreserve.gov/releases/h8/h8_technical_qa.html>

Only the seasonally adjusted values printed in each dated archive release are
primary inputs. Current-vintage FRED/DDP history is forbidden because H.8 is
benchmarked and revised.

### New York Fed SOFR

- panel:
  `data/new_york_fed_sofr_distribution_2018_2023/new_york_fed_sofr_distribution_2018-04-02_2023-12-28.csv.gz`
- panel SHA-256:
  `4993eda2b659e346b4d7b6e3aa0e2ff31cacf868f0e1fe2e1a5a76a03d1b5852`
- build-manifest SHA-256:
  `873afb5234fd013e3bc454a83713abf34d9f4a4bffc9895683add7891c636598`
- emitted rows / frozen coverage: 1,436 observations, 2018-04-02 through
  2023-12-28
- official definition: <https://www.newyorkfed.org/markets/reference-rates/sofr>
- official API: <https://markets.newyorkfed.org/static/docs/markets-api.html>
- publication/revision policy:
  <https://www.newyorkfed.org/markets/reference-rates/additional-information-about-reference-rates>

BDRC uses only the median `sofr_percent` at its conservative
`sofr_available_at_utc`. It does not use the percentile or volume summaries,
whose historical vintages require a much later availability clock.

## Frozen H.8 stress state

For H.8 release `t`, using the latest and prior printed weekly levels:

```text
migration_bp[t]
  = 10,000 * (
      log(sa_large_other_deposits_latest / sa_large_other_deposits_prior)
      - log(sa_small_other_deposits_latest / sa_small_other_deposits_prior)
    )

borrowings_bp[t]
  = 10,000 * log(sa_small_borrowings_latest / sa_small_borrowings_prior)

cash_stress_bp[t]
  = -10,000 * log(sa_small_cash_assets_latest / sa_small_cash_assets_prior)
```

Standardize each component independently against exactly the prior 104 H.8
releases. The current release is excluded.

```text
z(x[t]) = (x[t] - median(x[t-104:t]))
          / (1.4826 * MAD(x[t-104:t]))
```

Zero MAD, a non-finite value, or an incomplete 104-release history makes the
row unavailable. Then:

```text
h8_score[t] = mean(z_migration[t], z_borrowings[t], z_cash_stress[t])
h8_sign[t]  = sign(h8_score[t])
```

The H.8 state is valid only when `h8_score != 0` and at least two of the three
component z-score signs equal `h8_sign`. There is no H.8 magnitude threshold,
tail grid, or outcome-fitted weight.

The primary excludes exactly these previously documented methodology breaks:

- 2020-10-02;
- 2023-03-31; and
- 2023-06-30.

Only Thursday and Friday release dates are eligible. Other archived release
weekdays remain source provenance but cannot enter the frozen execution clock.

## Frozen secured-repo confirmation

Convert every admitted `sofr_percent` string to an exact integer basis point:

```text
sofr_bp[s] = Decimal(sofr_percent[s]) * 100
```

Reject rather than round if the result is not integral. For an H.8 decision,
select the greatest SOFR row index `s` whose
`sofr_available_at_utc <= decision_time`. Require that its availability is no
more than 36 hours old and that source rows `s-5` through `s` exist in exact
effective-date order. Calendar interpolation and weekend rows are forbidden.

```text
repo5_bp[t] = sofr_bp[s] - sofr_bp[s-5]
repo_sign[t] = sign(repo5_bp[t])
```

A zero five-observation change is flat and cannot create an event. The lag is
five **published SOFR observations**, not five calendar days.

## Frozen primary event and execution clock

For the H.8 release date, define:

```text
decision_time = 17:00 America/New_York
```

The decision is invalid unless it is strictly later than the archived H.8
`release_time_utc`. The fixed 17:00 local decision consumes 45 minutes after
the generally scheduled 16:15 H.8 release and is after the conservative 15:00
local SOFR availability used by the source panel.

The primary event exists only when:

```text
h8_sign[t] == repo_sign[t] != 0
```

Direction is fixed before incidence and outcomes:

```text
h8_sign = +1  # bank stress and repo tightening
side = -1     # SHORT BTC

h8_sign = -1  # bank relief and repo easing
side = +1     # LONG BTC
```

Execution is:

- entry: 17:05 America/New_York, after one complete five-minute computation
  and transmission bar;
- scheduled exit: exactly 864 five-minute bars / 72 hours after entry;
- exposure: fixed 0.5x account notional;
- no stop, take-profit, dynamic exit, leverage grid, or position overlap;
- chronological reservation; accept a new entry only at or after the previous
  scheduled exit; and
- entry and exit must both be contained in the declared split.

## Outcome-blind support gate

No market or funding value may be loaded. Reject BDRC-864 without an evaluator
unless every condition passes:

| Window | Minimum total | Calendar dispersion | Side floor |
| --- | ---: | --- | ---: |
| 2020-2022 train | 45 | at least 10 in each year | at least 15 long and 15 short |
| 2023 selection | 12 | at least 5 in each half | at least 4 long and 4 short |

The largest entry-month share must be at most 25% separately in train and
selection. Every admitted row must satisfy the exact H.8 history, structural
exclusion, SOFR availability, five-observation path, 36-hour freshness, side,
latency, non-overlap, and split-containment contracts.

The support stage must also publish one-to-one tolerant entry overlap within
plus/minus six hours and signed five-minute occupied-exposure correlation
against the frozen H8DM-1, SFRD-1, and FLCC-1 clocks. These are diagnostics,
not artificial independence gates: BDRC intentionally conditions an H.8
release clock. A later standalone performance pass is required before any
portfolio orthogonality claim.

## Frozen controls

All controls are specified before source incidence. Each independently formed
clock uses the same decision, latency, 72-hour hold, split containment, and
non-overlap rules unless explicitly stated.

1. **H8-only:** every valid nonzero H.8 state, `side=-h8_sign`.
2. **SOFR-only on H.8 schedule:** every fresh nonzero `repo_sign`,
   `side=-repo_sign`.
3. **Discordant state:** `h8_sign == -repo_sign != 0`, with
   `side=-h8_sign`.
4. **NSA H.8:** recompute the three H.8 components from not-seasonally-adjusted
   levels, then require concordance with the same current SOFR state.
5. **One-H.8-release stale:** pair the previous valid H.8 sign with the current
   H.8 decision and current SOFR state.
6. **One-SOFR-observation stale:** replace `(s,s-5)` with `(s-1,s-6)` at the
   current H.8 decision.
7. **Direction flip:** exact primary entries with `side=-primary_side`.
8. **Deterministic random side:** exact primary entries, ordered by entry time;
   `SHA256("BDRC-864-random-side-20260720|" + entry_time)` with first digest
   byte below 128 long and otherwise short.

The source-only support artifact must report every control's event count,
direction balance, calendar concentration, and overlap with primary. Controls
cannot replace primary after outcomes open.

## Later strict evaluator contract

Only a fully passing source-only artifact may authorize a separately written,
tested, committed, and hash-frozen strict evaluator. The later evaluator must
open outcomes sequentially:

1. train: calendar 2020 through 2022;
2. selection: calendar 2023, including H1 and H2;
3. 2024 test only after an unchanged train and selection pass;
4. 2025 eval only after the preceding pass; and
5. recent 2026 only after every preceding pass.

The evaluator must use 6 bp/notional/side base cost, 10 bp stress cost, exact
entry-inclusive/exit-exclusive funding with conservative boundary handling,
full-calendar CAGR including idle cash, and global/pre-entry-high-water strict
MDD over every held five-minute OHLC/funding path with favorable observations
before adverse observations and virtual adverse liquidation cost.

Train and selection must each have positive absolute return,
`CAGR / strict MDD >= 3.0`, strict MDD at most 15%, weekly-cluster one-sided
sign-flip `p <= 0.10`, mean gross underlying move at least 30 bp, positive
10-bp stress return, and positive one-bar-delayed execution. Calendar 2020,
2021, 2022, 2023-H1, and 2023-H2 must each be positive. Long and short sleeves
must each be positive in train and selection.

The primary's minimum train/selection CAGR-to-strict-MDD must exceed every
finite H8-only, SOFR-only, discordant, NSA, H.8-stale, and SOFR-stale control by
at least 0.25. A qualifying component, stale, flipped, or random control
rejects the claimed concordance mechanism; it cannot be promoted instead.

## Stop and live boundary

Any source-support, train, selection, test, eval, or forward failure retires
BDRC-864 without changing its sign, component weights, five-observation lag,
freshness, release timing, hold, cost, threshold, or calendar. Parent failures
cannot be mined to repair this candidate.

The frozen historical sources stop at 2023. A pre-2024 pass would authorize a
new point-in-time 2024 source build, not an API current-vintage shortcut. Live
promotion additionally requires dated H.8 archive ingestion, persisted SOFR
publication values and retrieval timestamps, revision detection, exact DST
handling, and at least 90 days of source/value/clock shadow parity before any
order is enabled.
