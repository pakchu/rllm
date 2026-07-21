# CCHR-288 — Cross-Collateral Cohort Handoff Relay mechanism decision

## Decision

The next outcome-blind BTC candidate is **CCHR-288 — Cross-Collateral Cohort
Handoff Relay, 24-hour hold**.

CCHR does not trade a static positioning tail or a contemporaneous
cross-collateral disagreement.  It waits for one completed state transition:

1. Binance USD-M global accounts enter an extreme directional crowd state;
2. USD-M aggressive flow initially supports that state relative to COIN-M;
3. the crowd state survives long enough to represent held inventory; and
4. the first material crowd contraction accompanied by a reversal of
   USD-M-versus-COIN-M aggressive-flow leadership emits an unwind trade.

This document opens no CCHR source values, event incidence, BTC execution
price, funding value, future return, PnL, equity, CAGR, MDD, comparator outcome,
or 2024-or-later row.

## Why this route follows the failure evidence

The recent source campaign repeatedly failed because strict relay rules became
too sparse, one-sided, or calendar-concentrated before outcomes were opened.
Dense five-minute flow tails then failed for the opposite reason: their gross
move was too small relative to execution cost.  CCHR therefore uses an hourly
state machine over a continuously observed exchange source and targets a
24-hour inventory-resolution consequence rather than another immediate bar.

The source axis is not assumed profitable.  Its advantages are narrower:

- public, checksum-bound historical data already exists through 2023;
- the same metric families have live REST counterparts;
- the side is structurally symmetric;
- the setup, persistence, first handoff, and execution timestamps are causal;
- the event can be rejected on support and clock novelty before any BTC
  outcome is loaded.

## Exact source boundary

The only primary feature source is
`data/binance_cross_collateral_metrics_btc_2021_2023/BTC_cross_collateral_metrics_5m_2021-07-08_2023-12-31.csv.gz`.

- file SHA-256:
  `ab9f18ba7745f21b17ac1124c45bb755245d404d66100c595bb77631f4bc1757`;
- source audit:
  `docs/binance-cross-collateral-positioning-metrics-source-audit-2026-07-17.md`,
  SHA-256
  `2e72881dac5aae71b8a8a078ea0748fcce015e3c52c8bfb985dbcbe04a8e13a2`;
- source manifest:
  `results/binance_cross_collateral_metrics_btc_2021_2023_manifest.json`,
  SHA-256
  `c0732ca47451209a9bb519545b0e349550994d870d476ee66ecbae81588fb159`.

The primary may materialize exactly these fields:

1. `date`;
2. `um_count_long_short_ratio`;
3. `um_sum_taker_long_short_vol_ratio`;
4. `cm_sum_taker_long_short_vol_ratio`.

The panel's existing `source_complete` column is explicitly excluded because
its builder also requires open-interest fields that CCHR does not use.  CCHR
derives its own local validity solely from exact row presence and the four
allowlisted fields above; that local flag is not a new external source.

It must not request or materialize price, return, funding, premium, PnL, or
future-label fields.  It also must not use any COIN-M account or top-trader
long/short ratio: all three COIN-M fields are empty throughout the frozen
pre-2024 panel.  USD-M open interest and all top-trader fields are excluded as
well.  This prevents CCHR from silently becoming a CCPR, DTV, or sparse
top-trader-lifecycle repair.

## Causal hourly observations

Only exact UTC `:55` source rows can be hourly anchors.  `G`, `U`, `M`, and `H`
share one fail-closed validity chain.  Anchor `t` is combined-valid only when:

1. the USD-M global-account ratio at `t` is positive and finite;
2. rows `t-55m, t-50m, ..., t` are all present at exact five-minute spacing;
3. the panel has unique, monotonically increasing timestamps; and
4. all twelve USD-M and COIN-M taker ratios are positive and finite.

Any failed condition creates an invalid hourly anchor.  An invalid anchor
clears both rank histories, cancels the active episode, and leaves the episode
gate unarmed.  It does not erase or truncate a trade that was already accepted:
the trade reservation, exit, and later outcome accounting remain independent
of feature-state validity.  A new setup is impossible until the complete
168-anchor causal history has rebuilt and the rebuilt crowd rank subsequently
enters the neutral rearm band.  Thus the causal-history rebuild itself is the
post-gap quarantine; there is no shorter calendar override.

No interpolation, forward fill, skip-over anchor, partial-hour salvage, or
independent `G` history is allowed.  This deliberately treats a taker outage as
missing evidence for the complete cohort-handoff mechanism rather than letting
the crowd state bridge the outage.

For valid anchor `t`, define:

```text
G[t] = log(um_count_long_short_ratio[t])

U[t] = median(log(um_sum_taker_long_short_vol_ratio))
       over the exact 12 rows ending at t

M[t] = median(log(cm_sum_taker_long_short_vol_ratio))
       over the exact 12 rows ending at t

H[t] = U[t] - M[t]
```

`G` is a USD-M global-account crowd coordinate.  `H` is a dimensionless
aggressive-flow leadership coordinate; it does not compare incompatible raw
notionals.

The archive timestamp is not treated as proof of instantaneous availability.
The completed `:55` anchor becomes decision-available only at `t+5m`.  The
earliest execution is the following five-minute open at `t+10m`.

## Strict-prior ranks

Both ranks use exactly the 168 immediately preceding **combined-valid** hourly
anchors, excluding `t`.  Those anchors must be contiguous hourly anchors with
no invalid anchor between them and `t`.  Any gap resets both warm-ups.

### Crowd state

Rank `G[t]` against the 168 strict-prior combined-valid anchors:

```text
R_G[t] = (
    count(prior_G < G[t]) + 0.5 * count(prior_G == G[t])
) / 168

C[t] = +1 if R_G[t] >= 0.90
       -1 if R_G[t] <= 0.10
        0 otherwise

S[t] = +1 if R_G[t] > 0.50
       -1 if R_G[t] < 0.50
        0 otherwise

E[t] = abs(2 * R_G[t] - 1)
```

`C` is the extreme-tail setup state.  `S` preserves the crowd orientation while
an episode contracts away from the tail.  Keeping these states separate is
essential: requiring `C` to remain extreme would make a material contraction
nearly impossible by construction.

Ties use exact parsed floating values.  No epsilon, winsorization, or current
row inclusion is allowed.

### Handoff strength

Rank `abs(H[t])` against the same 168 strict-prior combined-valid hourly
anchors, with the same exact midrank formula.  Call this `R_H[t]`.

## Singleton state machine

The state machine has no threshold or hold grid.

At every hourly anchor, processing order is fixed:

1. validate the combined source chain and compute ranks;
2. if invalid, reset histories/state and perform no other transition;
3. if an episode is active, update its peak, then test cancellation, handoff,
   and age-72 expiry in that order;
4. mark any cancellation, handoff, or expiry as a termination on this anchor;
5. only when no episode was active and no termination occurred on this anchor,
   update the neutral rearm gate; and
6. evaluate setup only when the gate was armed on a strictly earlier anchor.

A termination anchor can never rearm or start another episode, even if its
crowd rank is neutral.  The neutral arming anchor can never also be a setup
anchor.

### Setup

The episode gate starts unarmed at panel start and after every source gap or
episode termination.  While no episode is active, the first combined-valid
anchor with `0.25 < R_G[t] < 0.75` arms it.  An armed gate starts an episode on
the first later anchor where all conditions are true:

1. `C[t] != 0`;
2. `sign(H[t]) == C[t]`;
3. `R_H[t] >= 0.60`.

Freeze `episode_side = C[t]`, `episode_start = t`, and
`peak_extremity = E[t]`, then disarm the gate.

### Persistence and cancellation

While active:

- update `peak_extremity = max(peak_extremity, E[t])` when `S[t]` remains the
  episode side;
- cancel immediately if the source chain breaks or `S[t]` ceases to equal the
  episode side;
- evaluate the handoff at elapsed ages from 12 through 72 hours inclusive,
  then expire without a signal after the age-72 evaluation; and
- after cancellation, expiry, or handoff, do not arm another episode until the
  crowd rank first enters the fixed neutral band `0.25 < R_G[t] < 0.75`.

### First handoff

After at least 12 elapsed hours and no more than 72 hours, emit one candidate at
the first anchor satisfying all conditions:

1. `S[t] == episode_side`;
2. `E[t] <= peak_extremity - 0.20`;
3. `sign(H[t]) == -episode_side`;
4. `R_H[t] >= 0.75`; and
5. at the immediately preceding combined-valid hourly anchor,
   `sign(H[t-1h]) != -episode_side`.

The fixed trade side is `-episode_side`.  A high USD-M long-account tail that
contracts while aggressive leadership moves toward COIN-M therefore emits a
short; the mirror state emits a long.  This is an unwind-continuation
hypothesis, not a claim about trader identity or observed liquidation.

The episode terminates whether the candidate is later admitted or suppressed
by global non-overlap.  It cannot retry a later handoff.

## Execution contract

- decision time: handoff anchor `t+5m`;
- entry: `t+10m` five-minute open;
- side: frozen `-episode_side`;
- exit: entry plus exactly 288 five-minute bars / 24 hours;
- exposure: fixed `0.5x` account notional;
- global chronological non-overlap: suppress, never queue, a candidate whose
  entry is earlier than the prior accepted exit;
- no stop, take-profit, trailing exit, score priority, leverage search, regime
  gate, model, or LLM in the deterministic candidate;
- any event crossing a declared split boundary is dropped, never truncated.

The 12-hour minimum represents held crowd inventory rather than an intrabar
shock.  The 72-hour expiry prevents stale episodes.  The 24-hour hold targets
the consequence of a collateral-cohort handoff and leaves a large required
gross-edge margin over round-trip cost.

### Whole-panel construction, containment, and scheduling

For the primary and each control independently:

1. construct combined-valid features and raw state transitions once, in UTC
   chronological order, over the complete frozen pre-2024 source panel;
2. assign a raw candidate to a research split only when its `episode_start`,
   handoff anchor, decision time, entry time, every five-minute held bar, and
   scheduled exit all lie inside that same half-open split;
3. drop a boundary-crossing candidate before scheduling; it reserves no time;
4. sort the remaining candidates by entry time and apply the policy's own
   global non-overlap schedule across the whole panel; and
5. accept an entry only when `entry_time >= prior_accepted_exit`, suppressing
   rather than queueing every earlier entry.

The handoff-only control uses its frozen reference-sign anchor in place of
`episode_start`.  Exact-clock controls inherit the primary episode start and
handoff anchor.  A feature gap resets feature/state construction but never
clears an already accepted trade's reservation.  Statistics may group accepted
events by year, half, or quarter only after this single chronological schedule;
they may not rebuild a more favorable per-bucket schedule.

## Research windows

- source warm-up only: `[2021-07-08, 2021-08-08)`;
- train: `[2021-08-08, 2023-01-01)`;
- selection: `[2023-01-01, 2024-01-01)`;
- 2024, 2025, and 2026 remain sealed until every earlier gate passes.

The train interval deliberately contains the partial 2021 source era and full
2022.  The 2022 USD-M taker-ratio outage pattern is not filled; it must reset
the state and can cause a source-support rejection.

## Source-only controls

Every control uses the same combined-valid source chain, exact-midrank tie
rule, whole-panel split containment, zero-sign rejection, 24-hour exit, and
independent global non-overlap schedule unless an exact-clock rule below says
otherwise.  A gap clears its feature history and state but not an accepted
trade reservation.  No control may replace a failed primary.

1. `crowd_resolution_only`
   - The neutral rearm and crowd episode are identical to the primary, except
     setup is the first armed `C[t] != 0` anchor and has no `H` predicate.
   - Freeze `episode_side`, `episode_start`, and `peak_extremity` identically.
   - At ages 12 through 72 hours inclusive, emit at the first anchor with
     `S[t] == episode_side` and
     `E[t] <= peak_extremity - 0.20`; side is `-episode_side`.
   - Crowd cancellation, expiry, rearm, latency, and scheduling are otherwise
     identical.  Neither `sign(H)` nor `R_H` participates.
2. `handoff_only`
   - There is no crowd episode, age, contraction, or neutral rearm.
   - At a combined-valid anchor with `sign(H[t]) != 0` and `R_H[t] >= 0.75`,
     emit only when the immediately preceding combined-valid hourly anchor has
     nonzero `sign(H[t-1h])` and the two signs differ.
   - The preceding anchor is the control's causal origin, side is
     `sign(H[t])`, and normal `t+5m` decision / `t+10m` entry applies.
   - Because emission requires a sign-change onset, no further raw event can
     occur until `H` changes sign again; accepted events additionally obey the
     independent 24-hour non-overlap schedule.
3. `no_age`
   - Exact primary state machine with minimum age fixed to zero; all other
     setup, contraction, handoff, expiry, rearm, side, and timing rules remain.
4. `um_taker_only`
   - Define `R_U[t]` as the strict-prior midrank of `abs(U[t])` against the same
     168 combined-valid anchors.
   - Replace `H`, `sign(H)`, and `R_H` everywhere in primary setup/handoff with
     `U`, `sign(U)`, and `R_U`; every other rule remains exact.
5. `cm_stale_1h`
   - Require the immediately prior combined-valid anchor at exactly `t-1h` and
     define `H_stale[t] = U[t] - M[t-1h]`.
   - Rank `abs(H_stale)` against its own exact 168 prior consecutive transformed
     anchors.  The first transformed anchor after a reset is invalid, so its
     rank cannot silently shorten the warm-up.
   - Replace `H` and `R_H` everywhere in primary setup/handoff with
     `H_stale` and `R_H_stale`; every other rule remains exact.
6. `one_hour_execution_delay`
   - Copy each raw primary episode, handoff, and side exactly; set decision,
     entry, and exit to the primary times plus exactly twelve five-minute bars.
   - Retain it only when the original `episode_start`, original handoff anchor,
     shifted decision, shifted entry, every shifted held five-minute bar, and
     shifted exit all lie in the same half-open research split.
   - Drop failures before building the independent chronological non-overlap
     schedule.
7. `direction_flip`
   - Copy the exact primary raw clock, containment, and accepted schedule; use
     the opposite side and change nothing else.
8. `deterministic_random_side`
   - Copy the exact primary raw clock, containment, and accepted schedule.
   - Side is `+1` when the first byte of
     `SHA256("CCHR-288|20260721|" + entry_time_utc_iso_z)` is even and `-1`
     when it is odd.  `entry_time_utc_iso_z` is the exact ASCII byte sequence
     produced by UTC `strftime("%Y-%m-%dT%H:%M:%SZ")`; fractional seconds,
     offsets, spaces, and alternative zero-zone spellings are forbidden.

`sign(0)` is zero everywhere.  A zero sign never starts an episode, satisfies
a handoff, or serves as the prior nonzero sign in `handoff_only`.

## Source-only admission and novelty gates

The machine-readable preregistration must bind the source, exact formulas,
controls, comparator identities, and these floors before real CCHR incidence is
constructed.

Primary support must satisfy all of:

- train at least 60 accepted events, at least 18 in 2021 partial, at least 36
  in 2022, and at least 12 in each train half-year bucket that contains at
  least 90 source days;
- selection at least 40 accepted events, at least 18 in each half and 7 in each
  quarter;
- each side at least 25% and at least 15 events in train, and at least 12 events
  in selection;
- maximum UTC month share at most 20% and maximum UTC weekday share at most 25%
  in each split; and
- no incomplete source chain, warm-up row, stale episode, or split-crossing
  event contributes to any clock.

### Exact external comparator registry

Novelty is tested against every exact comparator member below, never against a
post-hoc family label or a best member selected from outcomes.

1. **CCPR**
   - preregistration:
     `training/preregister_cross_collateral_positioning_recoil.py`, SHA-256
     `6d1224c3a0d24686bf3b997f424b10f65d004c211269916f6007de8b8464a0a5`;
   - source-only clock:
     `results/cross_collateral_positioning_recoil_clocks_2026-07-17.csv`,
     SHA-256
     `2a864ec2b616a3118bf9ffa44f99f96fbe19e79d82870f21a0d7d9010d5c993a`;
   - six members: `control=primary`, each `q in {0.80,0.85,0.90}`, crossed
     with native holds `CCPR-H4=48` and `CCPR-H8=96` five-minute bars.
2. **Positioning-disagreement lifecycle hazard (PDLH)**
   - source implementation:
     `training/search_positioning_lifecycle_hazard_alpha.py`, SHA-256
     `86d3f8dee6f3ce72ba2bc7f75daae559cd1e19d9260bc0ad535a53f6652e73f3`;
   - provenance result, hash only and never readable by the novelty stage:
     `results/positioning_lifecycle_hazard_alpha_scan_2026-07-13.json`,
     SHA-256
     `f72029be60dc63e2de78d30565acb6ca4d4879478e79167fc4f70168efcec0af`;
   - all sixteen members from
     `disagreement in {top_position_minus_global,top_account_minus_global}`,
     `min_age in {144,432}`, `trigger in {contraction,zero_cross}`, and
     `hold in {72,216}` five-minute bars.
3. **Debt-transfer velocity (DTV)**
   - source implementation:
     `training/search_debt_transfer_velocity_alpha.py`, SHA-256
     `babe8479d55853e2b4ab9263b44d835ec058455c17615c39275c3e85f87d1880`;
   - provenance result, hash only:
     `results/debt_transfer_velocity_alpha_scan_2026-07-13.json`, SHA-256
     `81a89f1c77d7d238e03faa842410378a19ebf3928443fc35e2b358c917589d55`;
   - all twenty-four members from `memory in {72,288}`,
     `acceptance_horizon in {72,288}`, `q in {0.90,0.95}`, and
     `hold in {72,144,288}` five-minute bars.
4. **Funding-age rollover transfer (FAR)**
   - source implementation:
     `training/search_funding_age_rollover_transfer_alpha.py`, SHA-256
     `5e28f645f6368bf13879443e1f866feacc3e8296bd2591ef4e8e31d3d6d5062d`;
   - provenance result, hash only:
     `results/funding_age_rollover_transfer_alpha_scan_2026-07-13.json`,
     SHA-256
     `fef5a6c761e5393a6795ff5e91c8777ebc62faed204848a597b1f780b5ea0c79`;
   - all twelve members from `min_age_settlements in {1,3,6}`,
     `half_life_bars in {288,864}`, fixed `q=0.90`, and
     `hold in {72,144}` five-minute bars.
5. **DLPD-12**
   - clock artifact:
     `data/btcdom_leverage_polarity_decomposition_evaluation_clocks_2022_2023.csv.gz`,
     SHA-256
     `38ccc18df700d24462d0cae91e34733856ed053dc400c584a3eedaf3f9ed60f1`;
   - one member: `candidate=DLPD-12, control=primary`, preserving its artifact
     decision, entry, exit, and side.
6. **Frozen executable live portfolio**
   - portfolio config:
     `configs/live/portfolio_gross385_trainmdd40_2026-07-12.json`, SHA-256
     `86f255ca3967245b8b0676b00025b955d7f33668ab1ef9d813623191b4ecd1e7`;
   - exact weighted union:
     `oi_upbit_ratio288_low=0.65`, `new_long_minimal_funding_premium=1.75`,
     and `cand_rex_veto_7=1.45`;
   - component configs and SHA-256 values are respectively
     `configs/live/oi_upbit_ratio288_low_candidate.json` /
     `659239373e1f51fc2df9615f5387686fd9252a56e1c366b45421bf39d3d6223f`,
     `configs/live/new_long_minimal_funding_premium_candidate.json` /
     `f0848c5fea1fcc7823ed15b6e4b865a8dc2731c2d2bfd2ba21b0f92c534f0f03`,
     and `configs/live/rex_veto_7_candidate.json` /
     `36df47c4737eb99f4ca5e2b257d9bd2fbf130df9d731b9ac02fcfe5192acd4db`.

Comparator IDs are byte-exact and parameter-complete:

- CCPR: `ccpr:q={q:.2f}:hold={hold}`;
- PDLH:
  `pdlh:{disagreement}:age={min_age}:trigger={trigger}:hold={hold}`;
- DTV: `dtv:memory={memory}:accept={acceptance_horizon}:q={q:.2f}:hold={hold}`;
- FAR: `far:age={min_age_settlements}:half_life={half_life_bars}:q=0.90:hold={hold}`;
- DLPD: `dlpd:DLPD-12:primary`; and
- live components: `live:oi_upbit_ratio288_low`,
  `live:new_long_minimal_funding_premium`, and `live:cand_rex_veto_7`.

Every export manifest must contain the exact sorted
`candidate_id -> {family, parameters, hold_bars, component_weight}` mapping.
Serialize that map as UTF-8 canonical JSON with sorted keys and separators
`(',', ':')`, then bind its SHA-256 in the CCHR preregistration.  A missing,
extra, renamed, duplicated, or parameter/weight-remapped ID fails closed even
when aggregate member count is unchanged.

PDLH, DTV, FAR, and the live portfolio do not yet have frozen pure-clock
artifacts.  They are mandatory prerequisites, not permission for the CCHR
evaluator to reconstruct a clock from a legacy search:

- each family must first receive a separate outcome-blind clock-export
  preregistration that freezes generator path/hash, every raw input path/hash,
  exact input-column allowlists, feature timing, threshold fit, raw onset,
  split containment, hold, and scheduling;
- the exporter must emit one checksum-bound artifact with exactly
  `candidate_id`, `split`, `decision_time`, `entry_time`, `exit_time`, and
  `side`;
- it constructs raw signals over its frozen pre-2024 source panel, drops any
  event whose causal origin, decision, entry, every held bar, and exit are not
  wholly contained by one CCHR research split, then applies chronological
  non-overlap per `candidate_id` over the whole panel before writing accepted
  clocks;
- the CCHR preregistration must bind the exporter hash, exporter manifest hash,
  pure-clock artifact hash, coverage interval, exact ID-map hash, and member
  count before any real CCHR incidence is constructed; and
- the CCHR process may read only that six-column clock projection.  It may not
  import or execute a legacy search, read a raw source, or read OHLC, close,
  funding, return, future extrema, PnL, equity, or either provenance JSON.

The existing CCPR reader may request only `signal_time`, `entry_time`, `q`,
`control`, and `side`.  For each `control=primary` row it derives decision as
`signal_time+5m`, crosses each `q` independently with hold 48 and 96, derives
exit from entry plus that many five-minute bars, and assigns the exact CCPR ID
above.  For each of the six IDs independently, it drops rows unless signal,
decision, entry, every held bar, and exit are wholly contained by one CCHR
split, sorts the survivors by entry, and accepts only entries at or after that
member's prior accepted exit.  Thus the source ledger is treated as raw onset
input, never as an already scheduled clock.

The DLPD reader may request only `candidate`, `control`, `split`,
`source_hour_start`, `decision_time`, `entry_time`, `exit_time`, and `side`.
It keeps `candidate=DLPD-12, control=primary`, assigns the exact DLPD ID above,
and recomputes the CCHR split from source-hour start, decision, entry, every
held bar, and exit.  A boundary-crossing row is dropped.  For every retained
row, artifact label `2022` must normalize to CCHR `train` and `2023` to CCHR
`selection`; any other label or mismatch fails the entire comparator.  It then
sorts and reapplies chronological non-overlap for that ID over the whole panel.

No other column from either wider frozen artifact may be materialized.  Every
generated pure-clock reader accepts only the exact six-column schema above.
Missing paths/hashes, extra or missing columns, duplicate member-entry pairs,
invalid sides, non-five-minute times, or a mismatch between declared and
observed coverage/member counts/ID map blocks the source-only evaluator.  The
outcome-bearing provenance JSON files may never substitute for clocks.

For the live comparator, each component keeps its own native accepted schedule
and exact ID.  Duplicate `(candidate_id, entry_time)` rows fail; simultaneous
rows from different components are retained.  The portfolio entry set is the
set union of component entry times.  On each five-minute interval its signed
exposure is the unmodified sum `weight_i * side_i` across active components;
there is no clipping, netting rule beyond arithmetic summation,
renormalization, or replacement of a missing component.

For each member independently, compare accepted clocks on the intersection of
its declared coverage and the CCHR five-minute grid in each split.  The CCHR
exposure is `0.5 * side` from entry through the bar immediately before exit; a
comparator uses its frozen native signed exposure (the live member uses its
exact weighted sum).  Reject CCHR before outcomes if any member exceeds:

```text
exact entry Jaccard                         > 0.10
fraction of CCHR entries within +/-6 hours > 0.35
absolute signed occupied-exposure Pearson  > 0.40
```

Missing, ambiguous, empty-required, outcome-bearing, or hash-drifting
comparators, a zero-variance exposure vector, or a comparator exporter/reader
that requests any non-allowlisted input or output column fail closed.

All internal controls are still constructed and reported in the source-only
artifact, including exact-entry overlap, `+/-6h` proximity, signed exposure
correlation, event counts, sides, and calendar concentration.  Those
diagnostics do not create an additional source-support admission threshold and
do not claim mechanism distinctness.  Mechanism-removal dominance is decided
only by the frozen economic rule below; direction-flip and random-side outcomes
remain interpretation falsifiers.

Any support or novelty failure retires **CCHR-288** without changing its rank
tails, age, contraction, handoff, side, latency, hold, or floors.

## Conditional economic protocol

Only an unchanged source-support and novelty pass may authorize a separately
implemented, committed, and hash-frozen strict evaluator.  That evaluator must
open train first and stop before selection on any failure.

Every opened table must report absolute return, full-calendar CAGR including
warm-up and idle cash, global/pre-entry-HWM strict MDD including every held
five-minute adverse path, CAGR/strict-MDD, trade count, long/short count, mean
gross underlying move, exact realized funding, and weekly-cluster sign-flip
inference.

Train and selection independently require:

- positive absolute return;
- CAGR / strict MDD at least 3.0;
- strict MDD at most 15%;
- every declared half positive;
- positive result at 10 bp/notional/side stress cost;
- positive result with one extra five-minute execution bar;
- mean gross underlying move at least 35 bp; and
- weekly-cluster one-sided `p <= 0.10`.

The primary must exceed every mechanism-removal control in CAGR/strict-MDD by
at least 0.25.  A direction flip or deterministic-random control that passes
the complete primary battery rejects the interpretation.

## RLLM boundary

CCHR is intentionally a deterministic alpha test first.  Gemma/RLLM cannot
create the side, alter the first-handoff clock, search age/expiry/hold, or rescue
a failed train or selection result.

Only after an unchanged deterministic train and selection pass may a single
small Gemma4-class policy receive symbolic state cards containing crowd age,
peak contraction, handoff-strength rank, source freshness, current position,
and portfolio risk.  It may choose only `TRADE` or `ABSTAIN`, or reduce size;
the frozen side and maximum hold remain unchanged.  Any RL reward must penalize
strict drawdown and turnover and must release labels only after the scheduled
exit.

## Next admissible action

Implement and test one machine-readable CCHR preregistration artifact.  Freeze
its source and comparator hashes before constructing real event incidence.  Do
not load BTC execution outcomes during that work unit.
