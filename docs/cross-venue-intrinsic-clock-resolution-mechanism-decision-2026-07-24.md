# CVICR-72 mechanism decision — cross-venue intrinsic-clock resolution

## Decision and evidence boundary

Freeze one standalone BTC candidate:
**CVICR-72 — Cross-Venue Intrinsic-Clock Resolution Relay**.

CVICR observes one ordered state per eligible UTC day. Binance Spot and USD-M
each accumulate quote notional toward their own causal daily target. The first
venue to reach its target becomes the clock leader. A candidate exists only
when:

1. the two venue anchors are materially separated in intrinsic time;
2. cumulative Spot and USD-M taker flow point in opposite directions when the
   leader reaches its target; and
3. when the lagging venue later reaches its own target, both cumulative flows
   point in the leader's original direction.

The trade follows that original leader-flow side in BTCUSDT USD-M after a full
five-minute computation buffer and holds exactly six hours.

This document freezes the data contract, daily clocks, transition, direction,
latency, hold, controls, source-support gates, comparator cohort, economic
sequence, and RLLM boundary before any CVICR anchor, gap, conflict, resolution,
candidate timestamp, or post-entry outcome is decoded.

The source family and broad predecessor outcomes are already known. CVICR is a
candidate-level frozen test, not a globally pristine market-history discovery.

## Economic mechanism

Calendar time does not say whether cash or leveraged trading consumed its
normal activity budget first. CVICR gives each venue its own notional clock.
The earlier anchor identifies the venue whose current-day activity reached the
same fraction of its causal normal-day scale first.

At that moment, opposite cumulative taker-flow signs indicate unresolved
cross-venue inventory pressure rather than simultaneous consensus. If the
lagging venue later consumes its own notional target and both venues then point
in the original leader direction, the observable transition is:

```text
clock leadership -> cross-venue flow conflict -> laggard resolution
```

The fixed continuation side treats resolution as a completed transfer of
directional sponsorship. The six-hour horizon targets a larger residual
repricing object than the rejected dense one- to four-hour CATCH, CLASP, LURI,
and CVTT policies. This interpretation is falsifiable. The source does not
identify participants, resting liquidity, inventory accounts, or causal price
discovery.

## Why this is not a predecessor repair

- **CATCH/CLASP/CVTT** use minute ordering or timing centroids inside each
  completed five-minute bar. CVICR does not read timing centroids, price
  response, basis, or within-bar lag arrows. It uses a pair of forward daily
  notional first passages.
- **LURI** infers a fixed three-hour USD-M inventory state and trades a
  price/basis release. CVICR has no fixed formation window, price, or basis
  gate.
- **dual intrinsic clocks** race price and flow directional-change event
  counts on one market. CVICR races independent Spot and USD-M notional
  budgets and then requires a cross-venue sign transition.
- **IVLIR/IVFHR/IVPLH** use one USD-M daily anchor or consecutive-day USD-M
  anchors. CVICR requires two same-day venue-specific anchors; neither venue
  alone can create the event.
- **AESS/NETF/TAAR/CARTA** consume aggregate-trade event topology. CVICR reads
  no aggTrade event-size, HHI, run, arrival, or topology feature.

No predecessor threshold, direction, or hold is changed and renamed.

## Immutable source contract

### Primary source

```text
data/binance_cross_venue_minute_leadership_btc_2020_2023/
BTCUSDT_cross_venue_minute_leadership_5m_2020-01_2023-12.csv.gz
SHA256 00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc
```

Manifest:

```text
data/binance_cross_venue_minute_leadership_btc_2020_2023/build_manifest.json
SHA256 544c2945a2b56be478a1edc4abbb93b762bda5afc32cbd0658dd6822ff6b70fa
```

Independent audit:

```text
results/binance_cross_venue_minute_leadership_audit_2026-07-14.json
SHA256 ffe0124ac9c5c0c3f1d1c284b672618cf910dc16cae36e65c1efe79710f039af
```

The source contains 420,768 exact UTC five-minute rows on
`[2020-01-01, 2024-01-01)`. It was built from checksum-verified official
Binance Spot and USD-M one-minute kline archives and is hard-sealed before
2024.

### Source-support allowlist

The source-support builder may decode exactly:

1. `date`;
2. `feature_available_time_utc`;
3. `trade_earliest_time_utc`;
4. `spot_quote_notional`;
5. `um_quote_notional`;
6. `spot_signed_quote_notional`;
7. `um_signed_quote_notional`; and
8. `source_complete`.

It may parse no price, return, path, basis, response, timing-centroid,
leadership, future, target, label, action, reward, PnL, funding, or portfolio
field. The loader must use a column allowlist rather than load-and-drop.

Required source invariants:

- exact SHA-256 identities above;
- exactly 420,768 unique monotonic rows from `2020-01-01 00:00:00` through
  `2023-12-31 23:55:00`;
- exact contiguous five-minute grid;
- both availability fields equal `date + 5m`;
- finite positive Spot and USD-M quote notional on every accepted row;
- finite signed notionals with absolute value no greater than quote notional;
- no 2024-or-later row;
- no interpolation, forward fill, partial-day salvage, or gap skipping.

## Reference-day and live-prefix contract

A historical UTC reference day is complete only when all 288 expected rows:

- exist exactly once;
- have `source_complete=true`;
- satisfy every allowlisted numeric invariant; and
- retain the exact availability relation.

An incomplete historical day:

- cannot enter either venue's volume reference;
- remains a missing observation at its exact calendar position; and
- is never imputed or replaced by an older day.

For complete day `D` and venue `v`, let `V[v,d]` be total quote notional on
complete day `d`.

The causal normal-day scale is:

```text
expected_volume[v,D] =
    median(V[v,d] for complete d in the 28 immediately preceding calendar days)
```

At least 21 complete observations inside those exact 28 calendar positions are
required. The current day is excluded. An incomplete position remains missing
and does not widen the window. The median is NumPy/Pandas float64 linear
median; no clipping, winsorization, session adjustment, weekday adjustment,
or later-day substitution is allowed.

The intrinsic target is fixed:

```text
target[v,D] = 0.50 * expected_volume[v,D]
```

## Exact paired daily anchors

For each venue independently, accumulate completed five-minute quote notional
from `00:00 UTC`:

```text
cumulative_volume[v,t] = sum(quote_notional[v,s], 00:00 <= s <= t)
```

The venue anchor `A[v,D]` is the first completed bar start `t` satisfying:

```text
cumulative_volume[v,t] >= target[v,D]
```

Both anchors must exist no later than the completed bar starting
`17:50 UTC`. This cutoff, together with the frozen computation buffer, permits
the full six-hour position to exit no later than the next UTC-day boundary.
An anchor at `17:55` or later is rejected.

Current-day causality is prefix-only. Every row from `00:00` through `A_late`
must exist, be `source_complete`, and satisfy the allowlisted invariants. The
five-minute computation-buffer row immediately after `A_late` must also be
complete before entry. A missing or invalid row in that causal prefix cancels
the day. A defect after entry is irrelevant to the already frozen position and
cannot be used to remove it. The evaluator may never require the future
remainder of the UTC day to be complete.

Define:

```text
A_early = min(A[spot,D], A[um,D])
A_late  = max(A[spot,D], A[um,D])
leader  = venue at A_early
laggard = the other venue
gap_bars = (A_late - A_early) / 5m
```

An exact tie is ineligible. `gap_bars` is a positive integer.

## Strictly prior clock-gap normalization

For each current paired-anchor day, form a reference from the preceding 180
valid paired-anchor days, excluding the current day and preserving actual
calendar order. At least 90 prior pairs are required.

The current day passes the frozen dislocation threshold only when:

```text
gap_bars >= prior-pair linear q60(gap_bars)
```

The reference is pooled across leader venues. Leader-specific thresholds,
calendar-time bins, year/month tokens, volatility conditioning, and
outcome-selected gap tails are forbidden.

## Exact conflict-to-resolution transition

For venue `v`, define cumulative flow fraction through completed bar `t`:

```text
flow[v,t] =
    sum(signed_quote_notional[v,s], 00:00 <= s <= t)
    / sum(quote_notional[v,s], 00:00 <= s <= t)
```

The denominator must be finite and strictly positive. There is no epsilon,
rounding, clipping, or deadband. Exact zero has sign zero.

At `A_early`, freeze:

```text
d = sign(flow[leader,A_early])
early_laggard_sign = sign(flow[laggard,A_early])
```

The initial conflict requires:

```text
d != 0
early_laggard_sign == -d
```

At `A_late`, the resolution requires:

```text
sign(flow[leader,A_late]) == d
sign(flow[laggard,A_late]) == d
```

The first equality is leader persistence; the second is laggard resolution.
The side is fixed to `d`. No price or later flow confirms the action.

## Availability, execution, and scheduling

- causal origin: `A_early`;
- resolution bar start: `A_late`;
- signal available time: `A_late + 5m`;
- computation/finality buffer: one complete five-minute bar;
- decision/order time: `A_late + 10m`, after the buffer row is complete;
- entry: USD-M BTCUSDT open at `A_late + 10m`;
- exit: scheduled USD-M open exactly 72 five-minute bars after entry;
- side: fixed `d`;
- exposure: fixed `0.5x` account gross;
- stop/take-profit/trailing exit: none;
- candidate cap: at most one primary candidate per UTC source day;
- overlap: suppress, never queue or replace, a candidate whose entry is before
  the previous accepted exit;
- split containment: early anchor, late anchor, availability, buffer, entry,
  every held bar, and exit must remain in one half-open evaluation split.

A raw clock records causal origin, resolution-bar start, signal-available time,
decision/order time, entry, exit, side, and leader venue. Comparator matching
uses `entry`; no downstream artifact may relabel signal availability as the
post-buffer decision.

The raw clock is built once across the complete pre-2024 source, sorted by entry,
and globally non-overlap-reserved before any split is inspected. Split
containment is applied only afterward. A globally reserved candidate that
crosses a split boundary is omitted from reported economics but is not erased
to free a later candidate. Every independent-clock control repeats this exact
order on its own clock.

A later source defect does not retroactively cancel an entered fixed-hold
position. Live operation must nevertheless fail flat on source divergence
before the next decision.

## Frozen source-only controls

Every independent-clock control gets its own chronological non-overlap
schedule. Same-clock side controls reuse primary timestamps.

1. **Gap only** — retain paired anchors and q60 gap; side is leader flow at
   `A_early`; remove both conflict and resolution.
2. **Initial conflict only** — retain gap and opposite early signs; side `d`;
   remove both late resolution equalities.
3. **Late alignment only** — retain gap and final two-venue alignment to `d`;
   remove the early laggard conflict requirement.
4. **No leader persistence** — retain gap, early conflict, and laggard
   resolution; do not require the leader to remain on `d` at `A_late`.
5. **No gap tail** — retain the exact conflict and resolution, but admit every
   non-tied paired anchor.
6. **Fixed expected-time clocks** — for each venue, replace the current
   first-passage anchor with the lower median of that venue's prior 28 valid
   anchor start-minute indices; evaluate the same ordering and flow transition
   at those two strictly prior fixed times. At least 21 prior anchor times are
   required, current day is excluded, and both fixed times use the current
   day's causal flow prefix. The fixed-time gap is compared with the current
   day's already causal primary paired-anchor q60 threshold; the control does
   not fit a second gap distribution.
7. **Stale laggard flow 24h** — retain the current paired anchors, gap, leader,
   and current leader flows, but replace the laggard cumulative flows at
   `A_early` and `A_late` with the immediately preceding complete UTC day's
   laggard cumulative flows at the same two clock times. The control emits no
   earlier than the current primary decision time.
8. **Exact direction flip** — primary timestamps with side `-d`.
9. **Deterministic random side** — primary timestamps; SHA-256 of
   `CVICR-72|entry_time_utc` assigns LONG when the first byte is below 128 and
   SHORT otherwise.
10. **One-bar execution delay** — primary origin and side; entry and exit each
    delayed exactly five minutes, with containment and overlap recomputed.
11. **One-hour execution delay** — primary origin and side; entry and exit each
    delayed exactly twelve bars, with containment and overlap recomputed.

Every trade-emitting control:

- requires a non-tied pair of control anchor times;
- requires `d != 0` and emits only side `-1` or `+1`;
- drops a row rather than inventing a side when a required flow is zero,
  missing, non-finite, or has a non-positive denominator;
- treats a fixed-time tie as ineligible; and
- treats a missing or incomplete prior-day stale prefix as ineligible.

Controls may falsify CVICR. They may not replace a failed primary or select a
different direction, target, threshold, or hold.

## Source-support gates

The outcome-blind support evaluator uses:

- warm-up/source only: `2020-01-01` until references are ready;
- train: `[2020-01-01, 2023-01-01)`;
- selection: `[2023-01-01, 2024-01-01)`;
- 2024 and later: unopened.

The primary must pass every gate.

### Train

- at least 75 accepted candidates;
- at least 20 in each of 2020, 2021, and 2022;
- at least 24 active calendar months;
- LONG and SHORT each at least 20%;
- Spot-led and USD-M-led each at least 15%;
- maximum single-month share at most 15%;
- maximum single-quarter share at most 30%;
- maximum calendar gap between entries at most 90 days;
- maximum same-side run at most 10; and
- maximum same-leader run at most 12.

### Selection

- at least 24 accepted candidates;
- at least 10 in each half-year;
- at least 4 in every quarter;
- at least 8 active calendar months;
- LONG and SHORT each at least 20%;
- Spot-led and USD-M-led each at least 15%;
- maximum single-month share at most 20%;
- maximum calendar gap between entries at most 75 days;
- maximum same-side run at most 8; and
- maximum same-leader run at most 10.

### Mechanism selectivity

Across train and selection separately:

- primary count divided by `gap_only` count must be at most 0.40;
- primary count divided by `initial_conflict_only` count must be at most 0.70;
- primary count divided by `late_alignment_only` count must be at most 0.70;
- primary count divided by `no_gap_tail` count must be at most 0.70;
- exact primary entry Jaccard with `fixed_expected_time_clocks` must be at most
  0.10; and
- exact primary entry Jaccard with `stale_laggard_flow_24h` must be at most
  0.05.

All count ratios and Jaccards above use each clock's own globally reserved,
split-contained accepted entries. Raw pre-reservation incidence and retention
are reported as diagnostics but cannot substitute for the frozen gates.

Undefined denominators, empty required controls, or any failed gate retire
CVICR unchanged before economic evaluation.

## Frozen predecessor comparator cohort

The source-support evaluator must hash-bind and compare the primary CVICR pure
clock against these already-opened predecessor clocks:

| Comparator | Artifact | SHA-256 |
|---|---|---|
| CATCH-12 | `results/cash_auction_transfer_catchup_handoff_clock_2026-07-14.csv` | `066bf8e08267a043cc191eb436f0aa33105ab948de9f9f1edfde4d9c30de46d1` |
| CLASP-24 | `results/cash_late_arrival_spillover_propagation_clock_2026-07-14.csv` | `e166f4bd24afd5a2f129bcc26393ad4293ad0bc5792686b3b0fc4a805d53f9d5` |
| LURI-48 | `results/leveraged_um_inventory_release_handoff_clock_2026-07-14.csv` | `50765cfed0c3ec6a0d1df18857c4e0a3e574d1aa449538c9b89cfac1fff67095` |
| CVTT V01–V04 | `data/cross_venue_temporal_torsion_v2_support_clocks_2020_2022.csv.gz` | `8f933b9d387fbcb764645a7002a5eefa9ee159c9c1ce7e007dca0dc4c16ebe33` |
| IVLIR primary | `data/intrinsic_volume_latent_impact_relay_clocks_2020_2023.csv.gz` | `523f24a0d955fe99cfb86c62078532c5fc9091234e6669ab9acff2a8f3367788` |
| IVFHR primary and `any_handoff` | `data/intrinsic_volume_flow_handoff_relay_clocks_2020_2023.csv.gz` | `ab12762dec9a93d41c293766e46dfc80ade81914fb32753a5923faa6437c338e` |
| IVPLH primary | `data/intrinsic_volume_price_lag_handoff_clocks_2020_2023.csv.gz` | `2efca3b44b0512a9423da90171f43babcadec2316dc6148796f3e61f98138e80` |

For every required nonempty comparator over common coverage:

- exact entry-time Jaccard must be at most 0.10;
- one-bar tolerant Jaccard must be at most 0.20;
- twelve-bar/one-hour tolerant Jaccard must be at most 0.35;
- absolute Pearson correlation of signed occupied exposure on the complete
  common five-minute grid must be at most 0.40; and
- position-time Jaccard is reported but is not a gate because dense
  predecessor clocks mechanically overlap sparse daily clocks.

For IVLIR primary, IVFHR primary, IVFHR `any_handoff`, and IVPLH primary, also
compute six-hour tolerant Jaccard and require it to be at most 0.60. Six-hour
matching against the dense CATCH, CLASP, LURI, and CVTT clocks is reported but
is not a gate.

For tolerance `w`, sort both entry sets and compute maximum-cardinality
one-to-one matching with two indices `i,j`: if `a[i] < b[j]-w`, advance `i`;
if `b[j] < a[i]-w`, advance `j`; otherwise match the pair and advance both.
The tolerant Jaccard is:

```text
matched_pairs / (count_CVICR + count_comparator - matched_pairs)
```

CVTT policies are compared separately and only over their frozen
`[2020-01-01, 2023-01-01)` common coverage. Absence of CVTT selection-year
rows is not a missing-coverage failure. Every other comparator uses the
intersection of its declared artifact coverage and the CVICR source.

Hash mismatch, empty required extraction inside declared common coverage,
undefined signed-exposure correlation, or a failed metric retires CVICR before
any outcome.

## Sequential economic protocol

Only a complete source-support, selectivity, and novelty pass authorizes a
separate strict evaluator. That evaluator must be implemented, tested,
committed, and hash-frozen before loading execution OHLC or funding.

Open stages strictly in order:

1. train 2020–2022;
2. selection 2023 only after train passes;
3. test 2024 only after selection passes;
4. eval 2025 only after test passes;
5. 2026 YTD as report-only forward evidence after every prior pass.

The pre-2024 stages use the existing sealed source. A post-2023 source
extension must reproduce the exact live feature contract and be separately
audited before 2024 is opened.

### Execution accounting

- leverage: `0.5x`;
- fee: 5 bp/notional/side;
- slippage: 1 bp/notional/side;
- base account cost per side: `0.5 * 0.0006 = 0.0003`;
- stress cost replaces base fee/slippage with 10 bp/notional/side, so stress
  account cost per side is `0.5 * 0.0010 = 0.0005`; it is not added to 6 bp;
- exact realized BTCUSDT funding over `[entry, exit)`;
- full-calendar CAGR including warm-up and idle cash;
- strict MDD from global/pre-entry high-water, entry cost, every held-bar
  favorable-then-adverse OHLC path, funding debit ordering, virtual adverse
  liquidation cost, scheduled-open exit, and exit cost; the scheduled exit
  bar's later high/low is excluded; and
- weekly entry-cluster sign-flip test with 100,000 draws and frozen seed
  `20260724`.

The cluster test is exact:

1. assign each trade's net account return after base costs and funding to the
   UTC ISO week of its entry, Monday `00:00`;
2. omit empty weeks, retain zero-return trades, and sum returns inside each
   nonempty week;
3. use the trade-weighted observed mean `sum(trade_returns) / N`;
4. initialize `numpy.random.default_rng(20260724)` independently for each
   reported split and control;
5. draw one independent Rademacher sign per nonempty week for each of 100,000
   permutations;
6. compute each null statistic as
   `sum(sign[week] * weekly_sum[week]) / N`; and
7. report the one-sided positive-mean p-value
   `(1 + count(null >= observed)) / 100001`, returning `1.0` for an empty
   trade set or no nonempty cluster.

### Qualification gates

Train and selection must each satisfy:

- positive absolute return;
- `CAGR / strict MDD >= 3.0`;
- strict MDD at most 15%;
- mean gross underlying move at least 30 bp/trade;
- positive 10 bp/notional/side stress return;
- positive one-bar-delayed return;
- weekly-cluster one-sided `p <= 0.10`;
- positive LONG and SHORT sleeve returns; and
- primary CAGR/strict-MDD at least 0.50 above every score-bearing mechanism
  control evaluated on its own frozen clock.

The score-bearing mechanism controls are exactly:

1. `gap_only`;
2. `initial_conflict_only`;
3. `late_alignment_only`;
4. `no_leader_persistence`;
5. `no_gap_tail`;
6. `fixed_expected_time_clocks`; and
7. `stale_laggard_flow_24h`.

`exact_direction_flip`, `deterministic_random_side`, `one_bar_execution_delay`,
and `one_hour_execution_delay` are falsification/stability controls rather than
score-bearing alternatives. None may independently satisfy the complete
stage qualification gate. The primary itself must additionally retain the
already frozen positive one-bar-delay requirement.

Every contained train calendar year and both selection half-years must have
positive absolute return. Selection must retain at least 24 trades and at
least 10 per half after execution containment.

Test 2024 and eval 2025 must independently satisfy positive return,
`CAGR/strict-MDD >= 3`, strict MDD at most 15%, at least 20 trades, positive
stress and one-bar-delay returns, and weekly-cluster `p <= 0.10`. Combined
2024–2025 requires `p <= 0.05`. No later stage may repair an earlier failure.

## Live parity

Live CVICR must:

- persist official Binance Spot and USD-M BTCUSDT one-minute klines;
- verify exact UTC minute boundaries and positive quote activity;
- finalize exact five-minute buckets before accumulation;
- preserve source receipt/finalization timestamps;
- maintain separate venue quote and signed-taker-flow sums;
- preserve the exact 28-calendar-day reference window with at least 21 complete
  days, while canceling any current day whose causal prefix is defective;
- wait the same full computation buffer;
- fail flat on missing/reordered/duplicated bars, clock drift, late data,
  source divergence, or an anchor computed before finalization.

REST repair may restore storage for later reference windows. It may not
backdate a missed live decision or convert an already canceled current-day
prefix into a historical trade.

## RLLM boundary

The deterministic candidate owns:

- opportunity creation;
- fixed side;
- entry;
- hold;
- leverage; and
- cost/risk accounting.

Only after unchanged train, selection, and untouched OOS success may one small
RLLM receive compact causal tokens:

- leader venue;
- gap-rank bucket;
- early conflict relation;
- leader-persistence relation;
- laggard-resolution relation;
- strictly causal flow-strength buckets;
- source-validity state; and
- current position.

Its action space is exactly:

```text
TRADE_FIXED_SIDE
ABSTAIN
```

No timestamp, row identifier, raw price, future label, sealed reward, side
choice, hold choice, or leverage increase may enter the model.

## Stop condition

The next unit must freeze a write-once preregistration builder and immutable
artifact reproducing this document. It may verify source and comparator
hashes, schemas, and headers, but may not decode a CVICR data row or comparator
outcome. After that, a separate tested source-support evaluator may open only
the allowlisted source fields.

Any source, support, selectivity, or novelty failure retires CVICR-72 without
changing the target, reference, gap tail, conflict, resolution, direction,
latency, hold, controls, gates, or comparator cohort.
