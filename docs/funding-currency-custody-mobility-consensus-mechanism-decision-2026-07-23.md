# FCCM-72 funding-currency custody-mobility consensus — mechanism decision

## Decision

The next outcome-blind BTC candidate is **FCCM-72 — Funding-Currency
Custody-Mobility Consensus**, with a fixed 72-hour hold.

FCCM combines four weak observations:

1. relative `fUSD` versus `fBTC` funding utilization;
2. relative 24-hour funding-warehouse draw;
3. relative funding tenor;
4. signless finalized WBTC custody turnover, breadth, and concentration.

The first three Bitfinex components determine direction by a fixed deductive
consensus. WBTC never supplies direction. It only determines whether collateral
mobility is sufficiently material and distributed to sponsor a Bitfinex state
transition. This is not another WBTC/stablecoin rule, not a mint-long/burn-short
rule, and not a threshold repair of BFMWD, WCDR, WTSL, or WSCF.

This document freezes the mechanism before an FCCM source value, feature,
transition, candidate, comparator row, BTC bar, funding row, future return, PnL,
CAGR, or MDD is opened. Existing source manifests, schemas, hashes, aggregate
row counts, and prior terminal rejection documents were admissible for source
selection. Prior comparator rows and performance values are not inputs to this
decision.

## Research-boundary and prior-family exposure audit

FCCM is **candidate-specific outcome-blind, but not globally clean-room**. The
repository and long-running human research thread have broad prior alpha-family
exposure that cannot be undone.

Opened during this FCCM selection pass:

- BFMWD source decision and preregistration documents plus the Bitfinex source
  manifest and file hashes;
- WBTC source decision/audit documents plus source manifest and file hashes;
- WCDR, WTSL, and WSCF mechanism/rejection documents and aggregate source-
  support fields; and
- WSCF's already-published aggregate comparator-overlap summary. No raw
  comparator row was decoded.

Not opened during this FCCM selection pass:

- the BFMWD train-result artifact or any BFMWD trade return/PnL value;
- any FCCM Bitfinex or WBTC source value row, feature, state, transition, or
  candidate incidence;
- any raw comparator clock row;
- any BTC price/bar, realized funding, future return, PnL, CAGR, or MDD value;
  and
- any post-2023 Bitfinex or WBTC source value.

The wider historical session may have exposed aggregate outcomes from adjacent
families, so FCCM cannot claim a pristine human holdout. Its defensible claim is
limited to a new, preregistered candidate identity with no FCCM-specific
feature/incidence/outcome access before freeze. FCCM differs operationally from
BFMWD: BFMWD required four within-symbol warehouse/deployment robust-z tails;
FCCM uses three exact pairwise `fUSD-fBTC` relative ranks plus an independently
finalized, signless WBTC mobility sponsor. Strong family-specific novelty gates
and the mandatory Bitfinex-only economic comparison below quarantine, rather
than erase, that prior exposure.

This declaration is separately hash-bound in
`results/fccm_mechanism_boundary_2026-07-23.json` (file SHA-256
`08eced75e484d5e0cc18882fef2672d24928f81995cb75ad0191131034c05184`,
manifest hash
`571554f181747fb61dd02612d36eacd16af04f81c90643c7266e12b8a0753dec`).
The ledger lists the direct artifacts and headers seen plus all zero-access
counters. It is explicitly self-attested because no independent OS-level file
access trace exists; it strengthens auditability but does not convert FCCM into
a globally pristine clean-room experiment.

## Economic claim and falsification boundary

Bitfinex publishes separate financing balance sheets for `fUSD` and `fBTC`.
Relative quote-currency utilization, warehouse draw, and tenor can indicate
whether financed-spot demand is rotating toward borrowing dollars or borrowing
BTC:

- stronger `fUSD` deployment than `fBTC` tentatively maps **LONG BTC**;
- stronger `fBTC` deployment than `fUSD` tentatively maps **SHORT BTC**.

Finalized WBTC gross turnover measures realized movement across the native and
wrapped collateral boundary. It does not identify intent and cannot determine
side. High turnover distributed across actors only sponsors a transition that
the independent Bitfinex components already made directional.

The hypothesis is falsifiable. Bitfinex funding can finance assets other than
BTC; WBTC activity can be merchant inventory maintenance; utilization and
tenor can move for rate or operational reasons. The controls below must show
that neither the Bitfinex clock alone nor a broad/stale WBTC state explains the
candidate.

## Frozen sources

### Bitfinex funding statistics

- canonical source:
  `data/bitfinex_margin_funding_stats_2020_2023.csv.gz`;
- SHA-256:
  `71635b9f3a38efa7422a6fcf616859e6a41636bbb79ff0f85e160ef395b0d53c`;
- source manifest:
  `results/bitfinex_margin_funding_stats_source_manifest_2026-07-20.json`;
- manifest file SHA-256:
  `9d7c13d56983d7d33fec1c17e24f1794baca64fcfc666599b798d5d5b49cf9b9`;
- source builder:
  `training/download_bitfinex_margin_funding_stats_v2.py`;
- builder SHA-256:
  `b3bb9434dec618c8724ad584caa2fb66cd705d210dd66889b32ab80fd8f480ca`;
- symbols: exactly `fUSD` and `fBTC`;
- permitted columns: `symbol`, `observation_time`, `available_at`,
  `timestamp_ms`, `average_period_days`, `funding_amount`, and
  `funding_amount_used`;
- forbidden columns for FCCM: `frr` and `funding_below_threshold`.

The source contains 70,116 physical rows over 2020–2023 according to its
committed manifest. FCCM uses the frozen conservative `available_at`, never a
request time or a later revision.

### Finalized WBTC custody flow

- canonical source:
  `data/wbtc_custody_bridge_flow_2020_2023/wbtc_mint_burn_2020_2023.csv.gz`;
- SHA-256:
  `bfcc6ebc2ded0cd8a57e5cda83a77daafe4de325adf606b23ba43ecf486b3b4e`;
- source manifest:
  `results/wbtc_custody_bridge_flow_source_manifest_2026-07-23.json`;
- manifest file SHA-256:
  `e95267d55f390a35bf609580e014c67d44adabe67526022d5e80d555964274e8`;
- source manifest hash:
  `4e4344a7f2841803dc8da625ee1320f79e1821d54cb2366a5464728507b4bcab`;
- source builder:
  `training/build_wbtc_custody_bridge_flow_source.py`;
- builder SHA-256:
  `70816fbcc94d5ecd11f99e3b1ebc3087e396c8f6972adcefd7c3a308f7c6fdbf`;
- permitted columns: `event`, `amount_raw`, `actor_address`,
  `block_number`, `block_hash`, `transaction_hash`, `transaction_index`,
  `semantic_log_index`, and `available_at`; `block_hash` and
  `transaction_hash` are identity/integrity fields only;
- forbidden for features and side: `event_sign`, `block_timestamp`, mint/burn
  direction, REST record metadata, and address labels.

The source has 993 dual-replayed, receipt-paired events according to its
committed manifest. Only `available_at` from canonical block `N+64` is causal.
Live operation must additionally wait for finalized-head coverage.

## Frozen exact source alignment

All decimals, totals, ratios, ranks, thresholds, and comparisons use Python
`Fraction`-equivalent exact rational arithmetic. Binary floating point, log,
logit, clipping, epsilon, interpolation, forward fill, and backward fill are
forbidden.

For each UTC hour `H`, assign a Bitfinex row to `H` by
`floor(observation_time, 1 hour)`. A paired anchor is valid only when exactly one
`fUSD` and one `fBTC` row exist for `H`, both rows satisfy
`0 <= funding_amount_used <= funding_amount`, both totals are positive, and all
permitted numeric fields are finite. Its causal availability is the maximum of
the two source `available_at` values.

The 24-hour lag must be the exact paired anchor `H-24h`. A missing current row
is known conservatively only at `H+1h+15m`. A partially
present pair is invalid at the maximum available time of its present row and
`H+1h+15m`. A duplicate or invalid pair is invalid at the maximum source
availability among its rows. A missing, duplicate, invalid, or noncausal
current/lag pair invalidates `H`, resets the directional state to neutral, and
prevents the first later valid anchor from triggering; one later valid anchor
must first re-establish the prior state.

Equal-availability anchors form one causal batch. Their features and ranks are
computed against history available strictly before the batch. Only the greatest
hour in the batch may change state or emit a candidate.

Batch/state processing is complete and ordered as follows:

1. Compute validity and, for valid anchors, features/ranks against strictly
   pre-batch history.
2. If any anchor in the batch is invalid, the whole batch emits nothing and
   resets state. Valid feature rows from that batch may enter later rank history
   only after the reset; none establishes a state.
3. Otherwise, every valid feature row enters later history after all ranks are
   fixed, but only the greatest `H` is state-eligible.
4. A greatest neutral anchor establishes state `0`. A greatest directional
   anchor after a reset establishes its state but cannot trigger.
5. A later greatest directional anchor that transitions into `p` updates state
   to `p` whether or not WBTC sponsorship is active. If WBTC is inactive it is
   suppressed permanently and cannot trigger later while state remains `p`.
6. Non-greatest directional anchors never establish, transition, queue, or emit.

## Frozen Bitfinex components

For symbol `s` and paired hour `H`:

```text
total_s[H]  = funding_amount_s[H]
used_s[H]   = funding_amount_used_s[H]
unused_s[H] = total_s[H] - used_s[H]
util_s[H]   = used_s[H] / total_s[H]

draw_s[H] = (unused_s[H-24h] - unused_s[H]) / total_s[H-24h]

util_rotation[H]  = util_fUSD[H] - util_fBTC[H]
draw_rotation[H]  = draw_fUSD[H] - draw_fBTC[H]
tenor_rotation[H] = average_period_days_fUSD[H]
                    - average_period_days_fBTC[H]
```

Each component is independently transformed to an exact signed midrank unit
against the latest 720 valid paired anchors whose causal availability is
strictly earlier than the current batch. The current batch is excluded and all
720 prior values are required:

```text
u(x) = (2 * count(prior < x) + count(prior == x) - 720) / 720
```

For each component, vote `+1` when `u >= 1/4`, vote `-1` when `u <= -1/4`, and
vote `0` otherwise. Let `score` be the exact mean of the three midrank units.

Directional state is fixed:

```text
LONG state  (+1): at least two +1 votes and score >= 1/3
SHORT state (-1): at least two -1 votes and score <= -1/3
neutral      (0): otherwise
```

No component, vote, threshold, rank history, or tie policy may change after
FCCM incidence is observed.

## Frozen signless WBTC mobility sponsorship

Define daily anchor `D` at `00:00:00 UTC`. Membership is exclusively by causal
WBTC `available_at`, never `block_timestamp`:

```text
window(D)       = {e : D-14d < e.available_at <= D}
gross_raw[D]    = sum(amount_raw)
actor_gross[a]  = sum(amount_raw for actor a)
actor_count[D]  = count(distinct nonzero actor_address)
top_share[D]    = max(actor_gross) / gross_raw[D]
```

The canonical source has already rejected malformed/zero actors and nonpositive
event amounts; their later appearance is a source-integrity failure, not a
skipped window. For rank history, every UTC daily anchor has an exact gross,
including zero when its window has no events. Compute:

```text
u_gross[D]
  = (2 * count(prior_gross < gross_raw[D])
       + count(prior_gross == gross_raw[D]) - 180) / 180
```

The prior set is exactly the 180 daily anchors `D-180d .. D-1d`, each using its
own `available_at`-based 14-day window. No prior anchor is skipped. Missing
source coverage, malformed identity, or a source-integrity failure invalidates
the current and all dependent anchors rather than shortening the rank history.
The current anchor is inactive when its gross is zero or fewer than two actors
are present; `top_share` is then recorded as null and never compared.

The WBTC mobility state is active only when all hold:

- gross midrank unit `u_gross >= 1/5` (the 60th percentile or higher);
- at least two distinct actors; and
- exact `top_share <= 4/5`.

Mint/burn sign, net flow, current BTC price, stablecoin flow, address labels,
and post-2023 events are forbidden. The latest daily WBTC state with `D` no
later than the Bitfinex signal availability is used; by construction it is less
than 24 elapsed hours old.

## Frozen candidate state machine

Process eligible Bitfinex anchors in causal batch order.

1. A candidate opportunity exists only on a transition from a previously
   established state different from `p` into directional state `p`.
2. The latest causal WBTC mobility state must be active at that same Bitfinex
   availability. An inactive WBTC state suppresses the opportunity without a
   queue.
3. Remaining in state `p` cannot trigger later merely because WBTC becomes
   active. FCCM requires a later exit from and re-entry into `p`.
4. An invalid Bitfinex anchor resets state and cannot trigger. The first later
   valid state only establishes a baseline.
5. Same-batch suppressed anchors are not queued. Only the greatest eligible
   hour may transition.

Side is deterministic: `p=+1` is LONG and `p=-1` is SHORT.

Record the three units/votes, score, WBTC anchor, gross-rank unit, actor count,
top share, Bitfinex source identities, and WBTC source identity hash for every
accepted clock row.

## Frozen execution

- signal: paired Bitfinex anchor's conservative causal availability;
- entry: `ceil_to_5m(signal) + 5 elapsed minutes`, including exact-grid signals;
- exit: exactly 72 elapsed hours / 864 five-minute bars later;
- fixed BTCUSDT perpetual notional exposure: `0.5x`;
- one global chronological reservation on `[entry, exit)`;
- accept only when entry is at or after the prior accepted exit;
- suppressed candidates are never queued or replaced;
- entry and exit must be wholly contained in one declared split; and
- no TP, SL, trailing exit, dynamic size, price gate, stablecoin gate, REX gate,
  external regime gate, direction override, model-selected side, or alternate
  hold.

## Frozen windows and source-support gates

- source warm-up: calendar 2020;
- train: `[2021-01-01, 2023-01-01)` by entry time;
- selection: `[2023-01-01, 2024-01-01)` by entry time;
- sealed from: `2024-01-01T00:00:00Z`.

Before any comparator row or market outcome is opened, primary FCCM clocks must
satisfy every gate:

### Train 2021–2022

- at least 60 accepted entries total;
- at least 24 in each year;
- at least 10 in every half-year;
- at least 15 LONG and 15 SHORT;
- every quarter active;
- maximum UTC-month share at most 15%;
- maximum accepted-entry gap at most 45 elapsed days;
- maximum consecutive same-side run at most eight; and
- at least ten distinct WBTC actors across accepted sponsorship windows.

### Selection 2023

- at least 24 accepted entries total;
- at least 10 in each half-year;
- at least 6 LONG and 6 SHORT;
- every quarter active;
- maximum UTC-month share at most 20%;
- maximum accepted-entry gap at most 60 elapsed days;
- maximum consecutive same-side run at most eight; and
- at least five distinct WBTC actors across accepted sponsorship windows.

### Mechanism contribution and selectivity

- among raw directional Bitfinex transitions before non-overlap, WBTC
  sponsorship must be active for at least 20% and at most 70% in train and
  selection separately;
- each of utilization, draw, and tenor must vote with the accepted side in at
  least 35% of train and 35% of selection entries;
- at least three distinct directional vote patterns must appear in train and at
  least two in selection;
- no duplicate Bitfinex anchor, WBTC anchor, candidate identity, entry, or
  occupied interval;
- exact source hashes, arithmetic, ranks, state transitions, timing, split
  containment, and global non-overlap must pass; and
- zero post-2023 source value rows.

Any failure retires FCCM-72 unchanged before novelty and outcomes. Observed
incidence may not change a source, field, formula, lookback, threshold, vote,
direction, hold, support floor, or scheduler.

## Frozen source controls and falsifications

Every causal control runs its own chronological non-overlap scheduler:

1. `bitfinex_consensus_only`: exact directional transitions without WBTC;
2. `utilization_only`: utilization vote transitions with active WBTC;
3. `draw_only`: draw vote transitions with active WBTC;
4. `tenor_only`: tenor vote transitions with active WBTC;
5. `majority_without_score`: two-of-three vote majority without the mean-score
   requirement, with active WBTC;
6. `wbtc_stale_7d`: at Bitfinex signal time using latest daily anchor `D`, use
   the already-computed WBTC state at exactly `D-7d`; schedule at the unchanged
   Bitfinex signal/entry and never shift a WBTC anchor forward;
7. `bitfinex_stale_24h`: at hour `H`, use the directional state series computed
   from exact hour `H-24h`, shift that state/transition forward to `H`, use the
   current signal-time WBTC lookup, and schedule at `H`'s unchanged causal
   availability;
8. `exact_direction_flip`: exact primary entries with both sides reversed;
9. `deterministic_random_side`: exact primary entries. Encode the UTC entry as
   whole-second `YYYY-MM-DDTHH:MM:SSZ`, hash UTF-8
   `FCCM-72|random-side|<entry>`, and assign LONG when digest byte zero is below
   128, otherwise SHORT; and
10. `one_bar_delay`: primary entry and exit shifted five minutes, dropping any
   row that leaves its original split.

Two noncausal source placebos are incidence diagnostics only and may never emit
an execution clock or enter economics. Within each `available_at` calendar year,
sort source rows by the canonical WBTC identity defined below. For placebo
field `F`, sort the original `F` values by the key
`(SHA256(UTF-8("FCCM-72|placebo|F|year|source_identity")), source_identity)`
and assign that ordered value sequence to destination rows in canonical identity
order. This preserves the exact within-year multiset, is collision-stable, uses
no RNG/library-dependent shuffle, and has no tunable seed:

- within-year WBTC `amount_raw` permutation; and
- within-year WBTC `actor_address` permutation.

## Frozen novelty sequence

Only a complete source-support pass authorizes a separately committed novelty
evaluator. The cohort is fixed now; there is no later registry scan:

| Comparator artifact | SHA-256 |
|---|---|
| `data/bitfinex_margin_warehouse_deployment_clocks_2021_2023.csv.gz` | `02b4fcc462a5a48be7673649f4cf4b2f9bb210baca4294eed1696d479820cccc` |
| `data/wrapped_collateral_dollar_liquidity_rotation_2021_2023/wcdr2016_support_clocks_2021_2023.csv.gz` | `241d96a64a654ba2faeda2d4a8460131269acf21d0bbbf31177d35d1ecd63b3c` |
| `data/wbtc_turnover_stablecoin_liquidity_2021_2023/wtsl168_support_clocks_2021_2023.csv.gz` | `df8cb085d439c9ee9e89334cb891b9e3b04f54c2a8e70bd4f552a90648ea8b6d` |
| `data/wbtc_stablecoin_finalized_confirmation_relay_2021_2023/wscf72_support_clocks_2021_2023.csv.gz` | `86565774ae97a1024c5a66b4d59a1f5413bf4608398623359dd3ee24572f0ef3` |
| `results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz` | `73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08` |

Every nonempty comparator is gate-eligible regardless of event count; an empty
overlap interval is reported and contributes no ratio. On the common
`[2021-01-01, 2024-01-01)` interval require signless exact-entry Jaccard
`<= 0.10`, one-to-one signless matching within plus/minus six hours covering at
most 35% in **both** FCCM-to-comparator and comparator-to-FCCM directions, and
signed occupied-exposure correlation `<= 0.40` when side and exit exist. Report
same-side overlaps additionally but do not use them to relax signless gates. No
artifact or comparator group may be removed after FCCM incidence is observed.

Comparator parsing/grouping is fixed:

- BFMWD: require its committed header, filter `control == "primary"`, and
  compare each `variant_id` separately for exactly
  `bfmwd_w12_d3_z10_h12`, `bfmwd_w24_d3_z10_h12`,
  `bfmwd_w12_d6_z10_h12`, and `bfmwd_w24_d6_z10_h12`;
- WCDR: filter `candidate == "WCDR-2016"` and `control == "primary"`;
- WTSL: filter `candidate == "WTSL-168-SOURCE-SEEN"` and
  `control == "primary"`;
- WSCF: filter `candidate == "WSCF-72-SOURCE-FAMILY-SEEN"` and
  `control == "primary"`; and
- live bundle: group separately by exactly `live:cand_rex_veto_7`,
  `live:new_long_minimal_funding_premium`, and
  `live:oi_upbit_ratio288_low`.

Every parser requires the exact committed full header and canonical UTC
`entry_time`; correlation additionally requires canonical `exit_time` and side
encoded as base-10 `1` or `-1`. A duplicate entry or overlapping occupied
interval inside one comparator group fails closed. Comparator `row_identity` is
SHA-256 of
`FCCM-72|comparator|artifact_sha256|group_name|entry_time|exit_time|side`.

For directional one-to-one near matching, sort left rows by
`(entry_time, row_identity)`. For each left row, choose among unmatched right
rows within six elapsed hours the row minimizing
`(absolute_time_difference, right_entry_time, right_row_identity)`. Match it and
continue; unmatched rows remain unmatched. Run this algorithm independently in
both directions. Signless matching ignores side; same-side reporting first
partitions both inputs by side and applies the same algorithm.

Exposure correlation uses every five-minute interval start `t` on the exact UTC
grid `[2021-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`. A group's exposure is
`side * 1/2` when `entry_time <= t < exit_time`, otherwise zero. Compute Pearson
correlation on the full grid, including joint-zero intervals, and gate its
absolute value. Zero variance in either nonempty group fails closed; an empty
group was already handled as ineligible-empty and has no correlation.

## Frozen canonical identities and serialization

All hashes are lowercase SHA-256 hex over UTF-8 with no trailing whitespace.
All timestamps use whole-second UTC `YYYY-MM-DDTHH:MM:SSZ`; integers use base-10
without signs or padding unless negative values are explicitly permitted.

- Bitfinex row identity: `symbol|timestamp_ms`.
- Paired-hour identity: SHA-256 of
  `FCCM-72|bitfinex-pair|H|fUSD|fUSD_timestamp_ms|fBTC|fBTC_timestamp_ms`.
- WBTC event identity: lowercase
  `block_hash|transaction_hash|semantic_log_index`; `block_hash` and
  `transaction_hash` are permitted for identity/integrity only, never features.
- WBTC-window identity: SHA-256 of
  `FCCM-72|wbtc-window|D\n` followed by the sorted event identities joined by
  newline; the empty window hashes the prefix alone.
- Primary candidate identity: SHA-256 of
  `FCCM-72|candidate|paired_hour_hash|wbtc_window_hash|p|entry|exit`.
- Control-row identity: SHA-256 of
  `FCCM-72|control-row|control|paired_hour_hash|wbtc_window_hash|side|entry|exit`.

Primary-candidate uniqueness is scoped to primary rows. Control-row uniqueness
is scoped within each named control; equal underlying clocks across different
controls are allowed. Clock CSV rows sort by `(entry, row_identity, control)`.
JSON uses sorted
keys, compact separators, UTF-8, and rejects NaN/Infinity. Gzip uses `mtime=0`
and no original filename. Duplicate source, pair, window, primary-candidate,
primary-entry, or within-control row identities fail closed.

## Frozen economic contract if source and novelty pass

Only after source support and novelty pass may a separate evaluator be written,
tested, reviewed, committed, and hash-bound before outcomes:

- BTCUSDT USD-M five-minute next-open execution;
- exact realized funding;
- 6 bp/notional/side base cost and 10 bp/notional/side stress;
- full-declared-calendar absolute return and CAGR, including cash time;
- strict path MDD from the global HWM, including entry/exit costs and every held
  favorable-then-adverse bar extreme;
- deterministic weekly-cluster inference; and
- source controls evaluated under the same clock/economic contract.

Train and selection must independently have positive absolute return,
`CAGR / strict MDD >= 3`, strict MDD `<= 15%`, positive contained halves,
positive LONG and SHORT contributions, mean gross side-adjusted move at least
30 bp, positive one-bar-delay result, weekly-cluster `p <= 0.10`, and
positive 10 bp stress with `CAGR / strict MDD >= 2.5`.

For inference, allocate every five-minute net strategy return to the UTC week
starting Monday `00:00`. Require at least 20 nonzero train weeks and 10 nonzero
selection weeks. The one-sided statistic is the arithmetic mean weekly net
return, recomputed after every draw with fixed Decimal precision of 50 digits.
Generate exactly 100,000 synchronized wild-sign draws indexed by base-10
`j = 0..99,999`. For draw `j` and week start `W`, multiply every policy's return for that same
week by `+1` when the first byte of
`SHA256(UTF-8("FCCM-72|weekly-sign|20260723|j|W"))` is below 128 and by `-1`
otherwise. The p-value is
`(1 + count(T_draw >= T_observed)) / 100001`. The primary's one-sided p-value
must be at most 0.10 independently in train and selection. No library RNG,
optional stopping, alternate cluster, or alternate statistic is allowed.

The primary must beat `bitfinex_consensus_only`, `bitfinex_stale_24h`, and every
single-component control by at least 5 bp mean gross move and by at least 0.25
in CAGR/MDD; otherwise the full mechanism added no economic value. A stale
Bitfinex control independently satisfying the primary's full economic battery
rejects FCCM even if the margin comparison barely passes.

Train opens before selection. Selection remains sealed on train failure. Every
2024+ source value and outcome remains sealed until both pre-2024 stages pass
and a later sequential policy is separately frozen.

No sign inversion, source extension, threshold repair, hold repair, WBTC
direction, stablecoin confirmation, LLM rescue, RL reward search, or portfolio
gate may reuse the FCCM-72 identity after failure. An LLM/RL layer may later
allocate among independently validated alphas, but it may not rewrite FCCM's
features, side, clock, or economics.

## Live-parity boundary

Live promotion requires all of:

- Bitfinex public funding-stat collection with the same symbol/field identity,
  conservative availability, gap rules, and exact hourly pairing;
- canonical Ethereum WBTC logs with receipt companion verification,
  finalized-head coverage, and no event-time backdating;
- persisted raw identities, source revisions, node/provider health, and reorg
  invalidation;
- at least 90 shadow days with field-by-field historical/live parity; and
- deterministic replay producing identical feature, state, and order intent.

FCCM cannot authorize production capital from historical results alone.
