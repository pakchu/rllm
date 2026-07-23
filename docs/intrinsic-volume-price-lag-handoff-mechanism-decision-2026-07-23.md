# IVPLH-72 mechanism decision — intrinsic-volume price-lag handoff

## Decision

Freeze one source-seen singleton:
**IVPLH-72 — Intrinsic-Volume Price-Lag Handoff**.

IVPLH asks whether aggressive BTC perpetual flow changing sign at a comparable
daily intrinsic-volume coordinate, while price has not yet moved with the new
side, creates a delayed six-hour catch-up. It follows the new flow side:

- cumulative aggressive flow changes from SHORT to LONG while price is flat or
  down from the UTC-day open: **LONG BTCUSDT**;
- cumulative aggressive flow changes from LONG to SHORT while price is flat or
  up from the UTC-day open: **SHORT BTCUSDT**.

The economic claim is limited to a public-tape inventory-control handoff. The
source does not identify parent orders, resting liquidity, liquidation,
ownership, informed traders, or exchange inventory.

This document freezes the exact source transformation, state, direction,
latency, hold, controls, support gates, novelty cohort, staged economics, and
LLM/RL boundary before IVPLH split counts, exact timestamps, comparator overlap,
or post-entry outcomes are decoded.

## Research-history and contamination boundary

IVPLH is not source-incidence pristine. The predecessor IVFHR-72 report already
published aggregate `any_handoff` support over 2020–2023: 66 events, 29 LONG,
37 SHORT, 37 active months, maximum month share `4/66`, maximum quarter share
`8/66`, a roughly 90.70-day maximum gap, and a seven-event maximum side run.
Those facts are bound disclosures, not selectable parameters.

The exact predecessor control rows, split counts, yearly/half-year incidence,
timestamps, and comparator overlaps have not been decoded for this candidate.
No IVPLH-specific post-entry market value, funding cash flow, return, PnL,
absolute return, CAGR, strict MDD, reward, label, or post-2023 source value has
been opened.

Bound boundary document:

- `docs/intrinsic-volume-price-lag-handoff-boundary-2026-07-23.md`;
- SHA-256:
  `e1f12a7ccf693f2aafecd3b14e74090e2c1560a39a89ca59f2da3356c4cf244d`.

The source-seen lineage is bound to exactly these immutable predecessor
artifacts. Whole-file hashes must pass before any predecessor clock row is
decoded:

| Artifact | SHA-256 |
|---|---|
| `results/intrinsic_volume_flow_handoff_relay_preregistration_2026-07-23.json` | `e01e7f5af034adf98c0eef1e086ed1265c02998641f39d8cddd5137089f4153e` |
| `results/intrinsic_volume_flow_handoff_relay_support_2026-07-23.json` | `ed2a82e875d650f2e6f3197df1d34e39617d07640b5e13a3cc7ccc4bb09661d4` |
| `data/intrinsic_volume_flow_handoff_relay_clocks_2020_2023.csv.gz` | `ab12762dec9a93d41c293766e46dfc80ade81914fb32753a5923faa6437c338e` |

The predecessor clock header must be exactly
`clock_name,source_day,decision_time,entry_time,exit_time,side\n`, SHA-256
`0ad7d7a39f7d772de30d2c47056efd3c9b7740561eea9a1b69007b4870d5d495`.
Only `clock_name=any_handoff` may enter lineage identity; other known IVFHR
controls—`primary`, `no_price_lag`, `no_flow_strength`, `persistence_level`,
`fixed_noon_handoff`, `exact_side_flip`, and `deterministic_random_side`—are
skipped without decoding their non-selector fields, and an unknown clock name
fails closed. These are the only predecessor artifacts that may be opened after
the mechanism, preregistration, and tested source-support evaluator are all
committed.

## Frozen source

Source artifacts:

- panel:
  `data/binance_um_kline_reference_btc_2020_2023/BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz`;
- panel SHA-256:
  `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`;
- manifest:
  `data/binance_um_kline_reference_btc_2020_2023/build_manifest.json`;
- manifest SHA-256:
  `c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e`;
- exact grid: 420,768 completed five-minute rows in
  `[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`.

Only these columns may enter source construction:

```text
date, open, high, low, close,
quote_asset_volume, taker_buy_quote
```

`high` and `low` are validated against the frozen source schema but are not
used by IVPLH state or side. Timestamps must equal the complete five-minute UTC
grid exactly and every UTC day must contain exactly 288 rows. OHLC must satisfy
`high >= max(open,close,low)` and `low <= min(open,close,high)`. Quote volume
must be nonnegative. Taker-buy quote must lie in `[0, quote_volume]` within
tolerance `max(1e-8, abs(quote_volume)*1e-10)`. Any missing, duplicate,
non-finite, non-positive-price, accounting-invalid, or out-of-order row fails
closed. There is no interpolation or forward fill.

Historical bars are treated as available at bar open time plus exactly five
elapsed minutes. Live promotion requires final closed-klines from the official
Binance USD-M interface, local receipt timestamps, and at least 90 days of
forward parity. Archive publication time is not a live decision timestamp.

## Exact intrinsic-volume anchor

Let source day `D` be one UTC calendar day. Define total quote volume for each
complete prior day and:

```text
expected_volume[D]
  = median(total_quote_volume[D-28], ..., total_quote_volume[D-1])
```

At least 21 complete prior days are required; current day is excluded. The
target is exactly:

```text
target[D] = 0.50 * expected_volume[D]
```

The anchor is the first completed five-minute bar whose cumulative
`quote_asset_volume` from `D 00:00` through that bar reaches or exceeds the
target. A bar stamped `t` covers `[t,t+5m)`; its close becomes available at
`t+5m`. Anchors with bar-open timestamp later than `17:55 UTC`, target misses,
or zero/non-finite cumulative volume are invalid.

At the anchor:

```text
signed_quote_i = 2 * taker_buy_quote_i - quote_asset_volume_i

cumulative_flow[D]
  = sum(signed_quote_i from day start through anchor)
    / sum(quote_asset_volume_i over the same bars)

anchor_return[D] = log(anchor_close / day_first_bar_open)
```

Exactly zero or non-finite cumulative flow creates no eligible anchor. Define:

```text
flow_side[D] = +1 when cumulative_flow[D] > 0
             = -1 when cumulative_flow[D] < 0

directional_return[D] = flow_side[D] * anchor_return[D]
```

No magnitude threshold, q60 flow gate, impact ratio, range-position gate, or
future bar enters IVPLH.

## Frozen handoff state

Eligible anchors are ordered by `source_day`. Let `prior_anchors(D)` be all
eligible anchors with source day before `D`, and let `prior_window(D)` be its
last at most 180 rows. `reference_ready[D]` is true exactly when
`len(prior_window(D)) >= 90`. This retained warmup binds the source-seen
predecessor identity; no value from the window sets a threshold.

Let `prev` be the latest row of `prior_anchors(D)`. It is admissible only when
`prev.source_day` is exactly calendar day `D-1`. Any missing or invalid calendar
day resets the state. A candidate exists exactly when:

```text
reference_ready[D]
and prev exists
and prev.source_day == D-1 calendar day
and flow_side[D] == -prev.flow_side
and directional_return[D] <= 0
```

Equality at zero is admitted exactly. The candidate side is `flow_side[D]`.
There is no minimum previous run, current-flow strength, cooldown, price
threshold, regime gate, stop, take-profit, dynamic exit, or side override.

## Decision, execution, reservation, and identity

For anchor bar open `t`:

```text
feature_available_time = t + 5 minutes
decision_time          = t + 5 minutes
entry_time             = t + 10 minutes
exit_time              = entry_time + 72 * 5 minutes
```

The extra complete five-minute latency bar is frozen for live computability.
It is not the predecessor's zero-compute-time next-open assumption. Candidate
identity is nevertheless bound to the predecessor by `source_day`, handoff
side, and `decision_time`; latency may not manufacture novelty.

Raw candidates are sorted by `(entry_time, signal_id)`. UTC timestamps use
exact `YYYY-MM-DDTHH:MM:SSZ` text and source days use exact `YYYY-MM-DD` text.
Identity JSON is Python-compatible `json.dumps` with `sort_keys=True`,
`separators=(",", ":")`, `ensure_ascii=True`, and `allow_nan=False`, encoded
as UTF-8 with no trailing newline; non-finite values are forbidden.
`signal_id` is SHA-256 of exactly:

```json
{"control":"...","decision_time":"...","policy_id":"IVPLH-72","side":"LONG|SHORT","source_day":"...","source_panel_sha256":"..."}
```

Sides are exactly `LONG` or `SHORT`.
Duplicate identity payloads or SHA sort-key collisions fail closed.

Before reservation, retain a raw candidate for split `[S,E)` only when its
source day lies in the split's UTC date range, `S <= decision_time`,
`S <= entry_time`, and `exit_time <= E`. A crossing row is discarded and does
not advance reservation. Each control then reserves independently. Accept a
retained candidate only when `entry_time >= prior_accepted_exit_time`;
suppressed rows are never queued.

Reservation and split containment are independent for:

- lineage/calibration: `[2020-01-01, 2023-01-01)`;
- selection: `[2023-01-01, 2024-01-01)`.

Source-support train statistics use the already reserved subset in
`[2021-01-01, 2023-01-01)`; 2020 is warmup/calibration and never a train
economic result. Equality `exit_time == E` is contained; equality
`entry_time == E` is not.

The recomputed primary must exactly reproduce all 66 predecessor
`any_handoff` rows by `(source_day, side, decision_time)`, where IVPLH
`decision_time` equals predecessor `entry_time`. Every IVPLH entry and exit must
equal predecessor entry and exit plus five minutes. Any mismatch is an identity
failure, not a tunable support result.

## Frozen source-only controls

All controls reuse source integrity, warmup, latency, hold, split containment,
and independent non-overlap scheduling.

1. `primary` — exact IVPLH handoff plus price lag.
2. `handoff_without_price_lag` — retain reference-ready consecutive sign
   handoff, remove only `directional_return <= 0`.
3. `price_lag_without_handoff` — retain reference-ready price lag, remove only
   sign handoff.
4. `fixed_noon` — replace first-passage anchor by the completed 11:55 UTC bar;
   require cumulative volume by that bar to reach the same prior-day target,
   then apply the exact handoff/price-lag state.
5. `stale_24h` — shift primary decision, entry, and exit by exactly 24 elapsed
   hours, then reapply containment and reservation.
6. `direction_flip` — primary timestamps with opposite side.
7. `anchor_side_year_permutation` — for every eligible anchor within each UTC
   source-day year, sort the donor list by ascending hexadecimal SHA-256 of
   `IVPLH-72|anchor_side_year_permutation|donor|year|source_day` and the same
   destination population by the corresponding `destination` key. Assign the
   side of donor position `j` to destination position `j`, then recompute
   reference readiness, calendar-consecutive handoff, and price lag using the
   destination's contemporaneous anchor return.
8. `anchor_return_year_permutation` — use the identical donor/destination
   construction with control name `anchor_return_year_permutation`, assign the
   donor anchor return at position `j` to destination `j`, retain each
   destination's flow side, then recompute reference readiness, handoff, and
   price lag.
9. `deterministic_random_side` — primary timestamps; build a side-free key from
   canonical JSON with exact keys `policy_id`, `control`, `source_day`,
   `decision_time`, `primary_entry_time`, and `source_panel_sha256`. Side is
   LONG when the first byte of its SHA-256 is below 128, otherwise SHORT.
   Compute the control `signal_id` only after this side has been assigned.

Permutation mappings are bijections, preserve yearly populations, and may not
use NumPy/Python RNG state. Duplicate source days, duplicate signal IDs, or
non-bijective mappings fail closed.

## Frozen source-support gates

Before opening any comparator data row or market outcome, primary must pass:

- train `[2021,2023)`: at least 24 events;
- each of 2021 and 2022: at least 10 events;
- each train half-year: at least 3 events;
- train: at least 6 LONG and 6 SHORT and at least 20% each side;
- selection `[2023,2024)`: at least 12 events;
- each 2023 half-year: at least 4 events;
- selection: at least 3 LONG and 3 SHORT and at least 20% each side;
- no split UTC month above 20% and no split quarter above 40%;
- no accepted-entry gap above 120 elapsed days in either split;
- no same-side run above 10 in either split; and
- every candidate has exact causal timing, split containment, unique identity,
  and non-overlap.

For each of `anchor_side_year_permutation` and
`anchor_return_year_permutation`, in train and selection independently:

- exact entry-time Jaccard to primary must be at most `7/20`; and
- exact same-side reproduction divided by primary entries must be at most
  `3/5`.

Empty denominators evaluate to one and fail. These floors were chosen as broad
minimums for a two-sided daily event over two train years and one selection
year. They are not calibrated to the undisclosed split incidence. Any failure
retires IVPLH-72 unchanged.

Support subperiod membership requires the source day in the subperiod's UTC
date range, decision and entry at or after the named start, and scheduled exit
at or before the named end. Month and quarter concentration use accepted entry
UTC calendar labels and divide by all accepted primary entries in the
containing train or selection split, never by the global disclosed count. Gap
means maximum elapsed time between consecutive accepted entries within one
split, excluding source and warmup endpoints. Same-side run is computed on
accepted entries in chronological `(entry_time, signal_id)` order within one
split. Because aggregate global incidence was already known, a support pass is
an operational adequacy check, not independent discovery evidence.

## Frozen comparator cohort and novelty

Comparator files are source-clock artifacts only. Whole-file and header hashes
must all pass before the first comparator data row is decoded. Only the named
timing, side, and selector fields may be read; any return, future, label, PnL,
CAGR, MDD, price, OHLC, funding, or reward header fails closed.

This candidate adopts the prospective common-window policy:

- `docs/novelty-comparator-common-window-policy-2026-07-23.md`;
- SHA-256
  `928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580`.

A previously observed valid comparator interval crossing the 2023/2024
boundary motivated that repository-wide policy. This timing fact is disclosed
contamination, not candidate evidence.

| Comparator | File SHA-256 | Header SHA-256 | Selected group | Hold |
|---|---|---|---|---:|
| IVLIR-72 | `523f24a0d955fe99cfb86c62078532c5fc9091234e6669ab9acff2a8f3367788` | `0ad7d7a39f7d772de30d2c47056efd3c9b7740561eea9a1b69007b4870d5d495` | `clock_name=primary` | 72 bars |
| BAFR-24F | `f3b816a76decce31136ed23d22f043eb8e80ef1b8697b869241b060062f01747` | `437d41a791ba1084c3f38903ed6352f61462c3e3f3c5bb8fa065519a11b13852` | singleton/all rows | 24 bars |
| AFCS-144 | `bf1611554604c1930ba2212e674ea434f7c9793377b3f33ef531b3b4e0381688` | `fbe1d4fc7a2981a9fec253c5e6e04874035899626c9a19d96a97774e2b2d1999` | `branch=afcs_144` | 144 bars |
| LVRT-R0 | `ed9dd6391df2118ac09d147a4e57c3cb3f6e105a13f6c0d973ee424cfedd54d2` | `53ede9e934bd3c0612944e9ad678cb81e1400c5e0c3d64a10ed3401157a900e0` | `branch=lvrt_r0` | 12 bars |
| SMCC-144 | `3b255b224ab510afc30edb265d62428db9fdf07d90610499df62efff9ffa410d` | `56bc773a89f31d3c29c6ab5177451df6fe40518c0d879ec97248696e1ecb2b9c` | singleton/all rows | 144 bars |
| QLCD-288 | `ed882ac8a28f1f0b2b7ad7bf3d2de1f37b175cde63b20d4d1c7a290f3eb89bec` | `56bc773a89f31d3c29c6ab5177451df6fe40518c0d879ec97248696e1ecb2b9c` | singleton/all rows | 288 bars |

Paths, in table order:

1. `data/intrinsic_volume_latent_impact_relay_clocks_2020_2023.csv.gz`;
2. `results/binance_aggressor_frustration_clock_2026-07-20.csv`;
3. `results/aggregate_fill_compression_sweep_clock_2026-07-17.csv`;
4. `results/liquidity_vacuum_replenishment_clock_2026-07-17.csv`;
5. `data/same_millisecond_cascade_clock_2020_2023.csv.gz`;
6. `data/quantity_lattice_cohort_disagreement_clock_2020_2023.csv.gz`.

Header hashes above are SHA-256 over the exact UTF-8 CSV header including its
single trailing `\n`. After whole-file/header verification, parse every row in
every raw comparator group. The only fields that may be decoded are:

| Comparator | Decoded fields |
|---|---|
| IVLIR-72 | `clock_name,source_day,decision_time,entry_time,exit_time,side` |
| BAFR-24F | `entry_date,exit_date,side,hold_bars` |
| AFCS-144 | `entry_date,exit_date,side,branch,hold_bars` |
| LVRT-R0 | `entry_date,exit_date,side,branch,hold_bars` |
| SMCC-144 | `entry_time,exit_time,side` |
| QLCD-288 | `entry_time,exit_time,side` |

The other hash-bound columns may be present but are never decoded. The exact
IVLIR group universe is `primary`, `flow_only`, `no_under_response`,
`no_headroom`, `fixed_noon`, `exact_side_flip`, and
`deterministic_random_side`; an unknown clock name fails closed. AFCS and LVRT
require every raw row to carry exactly their selected branch value. BAFR,
SMCC, and QLCD each have one implicit group. In every raw group, entry and exit
must be timezone-aware UTC and five-minute aligned, side must be `LONG` or
`SHORT`, exit must equal entry plus the table hold, entries must be unique and
sorted, and intervals must not overlap. Any malformed row or empty required
artifact/group fails before window filtering.

The common window is `[2021-01-01T00:00:00Z,
2024-01-01T00:00:00Z)`. After complete raw-group validation, use an interval
only when `entry_time >= W0 and exit_time <= W1`; before-window, after-window,
and boundary-crossing intervals are excluded whole and never clipped. The
report records raw total, fully contained, before, after, and crossing counts
for every group. Candidate intervals use the same full-containment rule.
Changing eligibility after incidence or overlap is opened is forbidden.

Every selected comparator group must have at least ten fully contained entries;
otherwise novelty fails. For each:

- exact entry-time Jaccard must be at most `1/10`; and
- absolute Pearson correlation of signed five-minute occupied exposure over the
  exact common grid must be at most `7/20`; zero-variance exposure fails as one.

Entry Jaccard uses distinct UTC entry-time sets. Signed occupied exposure is a
vector on every five-minute cell in `[2021,2024)`: `+1` for LONG and `-1` for
SHORT from entry inclusive to exit exclusive, zero otherwise. Overlapping rows
within one comparator are invalid rather than summed. Pearson is computed over
the complete grid, including zero cells; a non-finite result fails as one.

For IVLIR, AFCS, LVRT, SMCC, and QLCD, greedy one-to-one entry matching within
plus/minus one elapsed hour must additionally have maximum bidirectional
containment at most `2/5`. BAFR is excluded only from this
tolerant-containment gate because its frozen clock is deliberately dense; it
remains subject to Jaccard and signed-exposure correlation.

The matcher builds every candidate/comparator pair within 60 elapsed minutes,
sorts pairs by `(absolute_delta, candidate_entry_time, comparator_entry_time)`,
and greedily accepts a pair only when both rows remain unmatched.
`maximum bidirectional containment` is exactly
`max(matches / primary_contained_count, matches /
comparator_contained_count)`; an empty denominator evaluates to one and fails.

IVLIR additionally uses exact `source_day` Jaccard at most `1/4`. The five-minute
latency change cannot make a same-day predecessor event appear novel.

Comparator rows remain unopened unless every source-support and permutation
gate passes. Novelty failure retires IVPLH before outcomes.

## Strict economic sequence

Only a complete source-support and novelty pass may authorize a new evaluator.
That evaluator and tests must be committed and hash-bound before any IVPLH
post-entry price or funding row is read.

The fixed economics are:

- BTCUSDT USD-M perpetual, side fixed by IVPLH;
- 0.5x notional exposure;
- base cost 6 bp/notional/side;
- stress cost 10 bp/notional/side;
- realized funding when `entry_time <= funding_time < exit_time`;
- full declared calendar CAGR including idle cash; and
- strict MDD with pre-entry/global HWM, entry cost, every held-bar
  favorable-then-adverse ordering, adverse virtual exit cost, realized funding,
  scheduled-open exit, and exit cost.

At each non-overlapping entry, quantity is
`0.5 * pre_entry_equity / entry_open` and remains fixed through exit. Linear
price PnL is `side_sign * quantity * (mark_price - entry_open)`. Entry and exit
fees are the named cost rate times `quantity * execution_open`. Funding cash is
`-side_sign * quantity * funding_rate * settlement_mark` at each included
settlement. Mean gross underlying bp is the arithmetic mean of
`side_sign * (exit_open / entry_open - 1) * 10_000`, before leverage, funding,
or cost.

Strict MDD starts with equity and global HWM equal to one at the declared stage
start and carries idle equity across the complete calendar. It marks entry fee
first. Within every held bar it updates HWM at the side-favorable extreme,
crediting positive same-bar funding, then measures drawdown at the side-adverse
extreme, debiting negative same-bar funding and a hypothetical adverse-price
exit fee. It then records the close mark and, at scheduled exit open, realizes
price PnL, all funding, and exit fee. Non-finite or non-positive equity fails.
Calendar years equal elapsed seconds divided by `365.25*86400`; CAGR is
`ending_equity ** (1/calendar_years) - 1`, and the ratio divides decimal CAGR by
decimal strict MDD with denominator floor `1e-9`.

The base-cost one-extra-bar-delay control shifts both entry and exit by exactly
five elapsed minutes, keeps the same 72-bar hold, side, event set, and order,
and does not reschedule suppressed events. Any shifted event no longer fully
contained in its stage makes that delay check fail.

Weekly significance aggregates base-cost primary net returns by ISO UTC
entry-week. It performs exactly 20,000 one-sided cluster sign flips. For draw
`j` and sorted week key `w`, the sign is positive iff the first byte of
SHA-256(`IVPLH-72|weekly_signflip|stage|j|w`) is below 128. The p-value is
`(1 + count(null_mean >= observed_mean - 1e-15)) / 20001`, where means divide
by the unchanged trade count. Empty trades or clusters return one.

Stages stop at first failure:

1. train `[2021,2023)`;
2. selection `[2023,2024)` only if train passes;
3. separately source-frozen 2024 test;
4. separately source-frozen 2025 eval;
5. separately source-frozen 2026H1 final.

Every opened stage must have positive absolute return, CAGR/strict-MDD at least
3.0, strict MDD at most 15%, positive base- and stress-cost results, positive
one-extra-bar-delay result, mean gross underlying edge at least 15 bp, and
one-sided UTC-week cluster sign-flip `p <= 0.10`. Train additionally requires
positive absolute return in 2021 and 2022. Selection requires positive absolute
return in both halves. Combined 2024–2025 requires positive absolute return,
CAGR/strict-MDD at least 3.0, and cluster `p <= 0.05`.

The primary must exceed the best same-stage mean gross underlying bp among
`handoff_without_price_lag`, `price_lag_without_handoff`, `fixed_noon`, and
`stale_24h` by at least 5 bp. Direction flip and random side are falsification
controls and may not rescue primary.

No support, comparator, train, selection, test, eval, or final result may change
the anchor fraction, cutoff, handoff, price-lag inequality, side, latency, hold,
cost, leverage, stage, or gate under the IVPLH-72 identity.

## LLM/RL boundary

The deterministic base clock must first pass standalone train and selection.
Only then may a separately preregistered compact LLM/RL policy become active
and choose between:

```text
TRADE_FIXED_SIDE
ABSTAIN
```

The prompt/observation may contain symbolic, strict-prior buckets for new side,
previous side, previous run length, anchor-time bucket, flow-magnitude rank,
price-lag rank, volume-target overshoot rank, time since prior handoff, source
freshness, and current portfolio position. It may not contain raw timestamps,
raw prices, future labels, reward leakage, exact split identity, or a side/hold
alternative. Skipped trades do not release the deterministic base reservation.

A Gemma-family adapter, exact symbolic bucket edges, prompt serialization,
reward function, train-label interval, optimizer, update cadence, seed, and
model size must be frozen after deterministic train passes but before any
selection outcome or selection-derived model label is opened. Model fitting may
use train labels only. Selection is a single untouched activation gate; no
selection reward, threshold, checkpoint choice, or prompt repair may feed back
into training. The later model protocol must itself be committed before any LLM
label/reward dataset is materialized. The LLM may improve abstention only after
gross base edge exists; it cannot size, reverse, retime, manufacture the alpha,
or repair a failed deterministic stage.

## Frozen stopping rule

The first failed identity, integrity, support, permutation, novelty, or economic
gate retires IVPLH-72 unchanged. Absolute return, CAGR, MDD, and other metrics
remain N/A until their sequential stage is explicitly authorized and opened.
