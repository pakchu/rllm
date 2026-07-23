# RCRE-72 mechanism decision — 2026-07-23

## Decision

Preregister one source-reuse, signed-feature-incidence-blind singleton:
**RCRE-72 — Repo Collateral Routing Efficiency**.

RCRE measures whether agency collateral is relatively concentrated in the repo
venue where agency financing is relatively more expensive or cheaper than
Treasury financing. It multiplies a signed cross-venue quantity gap by a signed
cross-venue relative-rate gap:

- positive product: agency collateral is more concentrated in the venue with
  the higher agency-versus-Treasury rate premium, indicating routing friction
  or constrained balance-sheet capacity;
- negative product: agency collateral is more concentrated in the venue with
  the lower relative financing premium, indicating efficient routing toward
  cheaper capacity.

The product is invariant to swapping the names GCF and TRIV1 because both
signed differences reverse together. This avoids assigning an arbitrary
direction to either venue.

RCRE is not an RVFC threshold repair and not an RMSR absorption-only variant.
RVFC discarded signs through absolute disagreement; RMSR modeled a temporal
race between absolute mix and rate states. RCRE instead freezes one new signed
price-by-quantity interaction and trades only its strict-prior extreme-state
transitions.

This document freezes source fields, arithmetic, feature sign, prior-only
normalization, event direction, execution, controls, support gates, novelty,
economic sequence, and no-repair rule before the signed gaps, product, RCRE
incidence, comparator rows, or any RCRE market outcome is computed.

## Evidence and contamination boundary

The OFR source values and RVFC/RMSR absolute component incidence through 2023
have already been opened. RMSR established source-only that most absolute
collateral-mix extremes normalize before same-polarity absolute-rate
confirmation. No BTC, funding, return, PnL, CAGR, or MDD was opened for either
OFR candidate.

During RMSR verification, the frozen comparator cohort was parsed until a
valid comparator interval crossing from 2023 into 2024 triggered its old
fail-closed rule. Comparator timing rows are therefore not pristine, although
RCRE overlap metrics remain unopened. RCRE fixes common-window handling before
its own incidence is computed: raw intervals are validated but never clipped,
and only fully contained 2021–2023 intervals enter novelty metrics.

The candidate-independent prospective rule was committed first as
`docs/novelty-comparator-common-window-policy-2026-07-23.md`, SHA-256
`928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580`.
RCRE must hash-bind and implement that document exactly.

Consequently:

- RCRE is not source-value-blind;
- exact signed quantity gaps, signed rate gaps, their product, RCRE state and
  event incidence, comparator overlap, and market outcomes remain unopened;
- prior comparator timing rows were partially opened for validation, but no
  RCRE clock existed and no RCRE comparison was possible;
- no RVFC/RMSR count, gap, side, terminal, or control can select an RCRE
  threshold or direction; and
- RCRE makes no pristine-source or broad-liquidity claim.

Any source-support, novelty, train, or selection failure retires RCRE unchanged.

## Frozen source and causal availability

Source artifacts:

- `data/ofr_repo_preliminary_2019_2023/ofr_repo_preliminary_observations_2019_2023.csv.gz`;
- `data/ofr_repo_preliminary_2019_2023/ofr_repo_preliminary_metadata_2019_2023.json.gz`;
- `data/ofr_repo_preliminary_2019_2023/build_manifest.json`.

Only preliminary (`-P`) rows may be used. Final values, disclosure-edit flag
`1`, nulls, interpolation, forward filling, and post-2023 observations are
forbidden. A date is complete only when each required mnemonic has exactly one
finite non-null row for the same `observation_date` and disclosure-edit flag
`0`.

Required series:

- `REPO-GCF_AR_AG-P`, `REPO-GCF_AR_T-P`;
- `REPO-TRIV1_AR_AG-P`, `REPO-TRIV1_AR_T-P`;
- `REPO-GCF_TV_AG-P`, `REPO-GCF_TV_T-P`;
- `REPO-TRIV1_TV_AG-P`, `REPO-TRIV1_TV_T-P`.

`TRIV1` excluding Federal Reserve transactions is mandatory. `TRI`, DVP,
venue-total series, final series, and sparse tenor buckets are forbidden.

For GCF and TRIV1 separately, `AG + T` transaction volume must be strictly
positive, and agency and Treasury collateral must each be at least 5% of that
venue's `AG + T` volume. Any failure invalidates the date and breaks state
continuity.

Every required row remains subject to:

```text
max(observation_date + 8 elapsed calendar days,
    2020-09-10 00:00:00 UTC)
```

Vector availability is the maximum required-row availability. Rows sharing an
availability timestamp form one causal batch. Every row in a batch is ranked
only against history available before that batch; after ranking, all complete
batch rows may enter later history. Only the greatest complete
`observation_date` in the batch may create a state or event.

## Frozen exact signed interaction

All decimals, shares, differences, products, midranks, and ties use exact
rational arithmetic. Binary floating point is forbidden.

```text
gcf_agency_share = GCF_TV_AG / (GCF_TV_AG + GCF_TV_T)
tri_agency_share = TRIV1_TV_AG / (TRIV1_TV_AG + TRIV1_TV_T)

quantity_gap = gcf_agency_share - tri_agency_share

gcf_relative_rate = GCF_AR_AG - GCF_AR_T
tri_relative_rate = TRIV1_AR_AG - TRIV1_AR_T
rate_gap = gcf_relative_rate - tri_relative_rate

routing_pressure = quantity_gap * rate_gap
```

Swapping every GCF and TRIV1 input must negate `quantity_gap`, negate
`rate_gap`, and leave `routing_pressure` exactly unchanged. Failure of this
identity on any complete date rejects the source build.

Neither `quantity_gap` nor `rate_gap` alone has a frozen BTC direction. The
economic sign exists only in their venue-label-invariant product.

## Strict-prior normalization and state

For `routing_pressure`, use exactly the previous 252 complete source dates in
observation-date order. The current row and every row sharing its availability
batch are excluded. No expanding fallback, calendar interpolation, fitted
transform, alternate window, or winsorization is allowed.

```text
midrank[t] = (count(prior < current)
              + 0.5 * count(prior == current)) / 252
u_pressure[t] = 2 * midrank[t] - 1

FRICTION = +1 when routing_pressure > 0 and u_pressure >= +0.50
EFFICIENCY = -1 when routing_pressure < 0 and u_pressure <= -0.50
NEUTRAL = 0 otherwise
```

Zero product is always neutral. A candidate occurs only when current state is
`FRICTION` or `EFFICIENCY` and differs from the immediately prior continuous
decision state. Persistence does not retrade. A direct `+1 ↔ -1` reversal
creates one event at the new state.

A missing or invalid date breaks continuity. The first complete rank-ready
decision row after a break establishes state but cannot trigger an event.

Frozen side:

- `FRICTION`: **SHORT BTC**;
- `EFFICIENCY`: **LONG BTC**.

## Frozen execution

- signal: current vector's conservative `available_at_utc`;
- entry: `ceil_to_5m(signal) + 5 elapsed minutes`, including exact-grid times;
- exit: exactly 72 elapsed hours / 864 five-minute bars later;
- fixed BTCUSDT perpetual notional exposure: 0.5x;
- one global chronological reservation on `[entry, exit)`;
- accept only when entry is at or after the previous accepted exit;
- suppressed events are not queued;
- entry and exit must remain in one declared split;
- no stop, take-profit, trailing exit, dynamic size, price gate, external
  regime gate, direction override, or leverage search.

## Frozen windows and source-support gates

- warmup/source history: 2019–2020;
- train clock: `[2021-01-01, 2023-01-01)` by entry time;
- selection clock: `[2023-01-01, 2024-01-01)` by entry time;
- sealed from: `2024-01-01T00:00:00Z`.

Before comparator rows or market outcomes are read, the primary clock must
satisfy all of:

- train total at least 45;
- each train year at least 15;
- each train half-year at least 6;
- train at least 10 LONG and 10 SHORT;
- selection total at least 20;
- each selection half-year at least 7;
- selection at least 5 LONG and 5 SHORT;
- every train and selection quarter active;
- maximum UTC-month share at most 20% train and 25% selection;
- maximum accepted-entry gap at most 90 elapsed days;
- both raw product signs each represent at least 20% of train and 15% of
  selection accepted events;
- each of the four exact sign quadrants `(q+,r+)`, `(q-,r-)`, `(q+,r-)`, and
  `(q-,r+)` represents at least 10% of train and 5% of selection accepted
  events;
- no one sign quadrant exceeds 50% of train or 60% of selection accepted
  events;
- exact venue-swap invariance on every complete source date;
- every accepted event has nonzero exact-rational quantity gap, rate gap, and
  product, valid strict-prior rank, exact timing, unique entry, split
  containment, and global non-overlap; and
- zero post-2023 source rows.

Any failure rejects RCRE before novelty and outcomes. Observed incidence may
not change 252, 0.50, sign requirements, source fields, support floors, side,
hold, or execution.

## Frozen source controls and falsifications

Every source control uses the same availability, batch history, transition,
entry latency, split containment, 72-hour hold, and non-overlap unless its
stated feature differs:

1. `quantity_gap_label_pair`: build original-label and GCF/TRIV1-swapped signed
   quantity-gap transition clocks. Their entry times must match and directions
   must be exact flips. They are label-sensitivity diagnostics with no alpha
   direction, cannot replace or economically falsify the invariant primary,
   and cannot enter a portfolio;
2. `rate_gap_label_pair`: the analogous original/swapped signed rate-gap
   clocks, under the same non-falsifying restriction;
3. `absolute_pressure`: rank `abs(quantity_gap) * abs(rate_gap)`; upper extreme
   SHORT, lower extreme LONG, without signed routing information;
4. `both_legs_extreme`: independently rank `abs(quantity_gap)` and
   `abs(rate_gap)` against 252 strict-prior complete dates. Require both unit
   ranks at least `+0.50`; side is SHORT for positive product and LONG for
   negative product. This is label invariant and ignores product magnitude;
5. `absolute_rank_additive`: equal mean of the two independently normalized
   absolute-gap ranks. Require mean at least `+0.50`; side is SHORT for positive
   product and LONG for negative product. This is label invariant;
6. `sign_without_magnitude`: positive product state SHORT and negative product
   state LONG, transition-only, without a rank threshold;
7. `one_complete_date_stale`: one complete decision row's RCRE state applied at
   current availability;
8. `five_complete_date_stale`: the same with five complete decision rows;
9. `year_rate_gap_permutation`: within each observation year, assign rate-gap
   values by deterministic source order
   `SHA256("RCRE-72|year_rate_gap_permutation|<year>|<observation_date>")`
   to chronological destination dates while preserving quantity gaps and
   current availability; recompute product and strict-prior ranks; and
10. `year_product_permutation`: permute complete routing-pressure values within
   year by
   `SHA256("RCRE-72|year_product_permutation|<year>|<observation_date>")`,
   preserving current availability.

The artifact must also prove exact venue-swap invariance rather than outputting
an identical redundant control.

Economic side controls reuse exact primary accepted entries and exits:

- exact direction flip;
- deterministic random side from the first byte of
  `SHA256("RCRE-72|deterministic_random_side|<entry_time_utc_iso>")`;
- constant LONG; and
- constant SHORT.

No control may replace the primary. The primary must beat the label-invariant
unsigned-magnitude, both-legs-extreme, absolute-rank-additive, sign-only,
stale, and permutation controls on frozen train and selection economics;
otherwise the multiplicative routing claim is rejected. Original/swapped
single-gap pairs remain non-falsifying diagnostics because assigning them an
economic direction would violate venue-label invariance.

## Frozen novelty gate

Only a complete source-support pass may open comparator rows. The exact cohort
is frozen here; no transitive lookup or later registry resolution is allowed:

| Comparator artifact | SHA-256 | Required groups |
|---|---|---|
| `results/overnight_rrp_flow_release_clocks_2026-07-17.csv.gz` | `7242d9870627dfc0cf067ff87d9664a1576dd374cb8985e927b40f15d1e3d480` | every `control` |
| `results/overnight_rrp_participant_breadth_support_clocks_2026-07-21.csv.gz` | `ef21323229801f11557e0c2d9d4465f7d58b13569552d656d64fdb7d440622ed` | every `control` |
| `results/federal_liquidity_component_concordance_preregistered_clock_2026-07-17.csv.gz` | `7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c` | every `candidate_id × clock_name` |
| `results/daily_treasury_fiscal_flow_breadth_primary_clock_2026-07-21.csv.gz` | `df53e1a27fcbc6ea2c4bc3f462a557a75c76a98db3c362944dad0b4d74382978` | every `policy_id × clock` |
| `results/daily_treasury_fiscal_flow_breadth_control_clocks_2026-07-21.csv.gz` | `416fc8663b292fcee069e4aca53b83e99a05b594a96940ab2c557e6e0d05e312` | every `policy_id × clock` |
| `results/sofr_rate_dislocation_preregistered_clock_2026-07-17.csv.gz` | `391c42dd2b0d5b87ffcd73058dd9fa0c4d18fd2f535597effff5a4c8edea2e69` | `SFRD-1|primary` |
| `results/bank_deposit_secured_repo_concordance_clocks_2026-07-20.csv.gz` | `1ff3a6075e3ceff928e1dd19d05880dbe9dbab0e07d79b853146d7b4c8f6cabc` | every `clock_name` |
| `results/fed_h8_deposit_migration_preregistered_clock_2026-07-18.csv.gz` | `20405f79b86861adcc784c81223baae1c40fdf3c73edda339578471a6a6d1b40` | `clock_mode == primary` |
| `results/soma_lending_collateral_scarcity_clocks_2026-07-23.csv.gz` | `b3fe0dc8c895f9a8974cdf08b5bed9d58ff693b8aca7ed59c224627de930a948` | `control == primary` |
| `results/cross_domain_liquidity_transmission_relay_support_clock_2026-07-21.csv.gz` | `aa2bcafd0f62ebe585f93cbd357d29c37ae526a95a90b8a6c0bd7c068cd6e5a1` | every `clock` |
| `results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz` | `73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08` | every `candidate_id` |
| `results/ofr_repo_venue_fragmentation_consensus_clocks_2026-07-23.csv.gz` | `b2d251b147feecc98def94610834c7b11f078f35e580f6e7a950fc55bfe0724e` | `control == primary` |
| `results/ofr_repo_mix_shock_resolution_race_clocks_2026-07-23.csv.gz` | `bed2f522bd68fd389cb0af3d655980999324c0e6adb4f1dc027c6b6ea2cc5ae6` | `control == primary` |

Constituent RCRE controls are mechanism specificity controls, not novelty
comparators. Over common 2021–2023 coverage, every comparator group with at
least ten entries must satisfy:

- exact-entry Jaccard at most 0.10;
- one-to-one RCRE containment within ±24 elapsed hours at most 0.35; and
- absolute signed occupied-exposure correlation at most 0.35.

Every raw comparator interval must satisfy the separately committed prospective
common-window policy. Novelty uses only intervals fully contained in
`[2021-01-01T00:00:00Z, 2024-01-01T00:00:00Z)`. Intervals ending before the
window, starting after it, or crossing either boundary are excluded whole,
never clipped, and their counts are reported. Missing, hash-mismatched,
malformed, overlapping, or empty required in-window groups fail closed and
produce a deterministic pre-outcome rejection artifact. These common-window
rules are frozen now and cannot change after RCRE incidence is opened.

## Strict economic sequence

Only source-support and novelty pass may authorize a separate evaluator freeze:

1. train 2021–2022;
2. selection 2023 only after exact train pass;
3. immutable source extension and test 2024 only after pre-2024 pass;
4. eval 2025 only after test pass;
5. recent 2026 only after eval pass.

Each economic split requires positive absolute return, full-calendar CAGR /
strict intratrade MDD at least 3.0, strict MDD at most 15%, realized funding,
6 bps notional cost per side, positive return under 10 bps stress, frozen trade
and side floors, positive required subperiods, and calendar-month cluster
sign-flip p-value at most 0.10. Inactive time remains in CAGR.

## RLLM boundary

RLLM remains unauthorized until deterministic source support, novelty, train,
and selection all pass. A later compact model may only choose
`TRADE_FIXED_SIDE` or `ABSTAIN` from causal bucketed RCRE text plus current
position/risk context. It cannot create events, reverse side, alter size or
hold, use future market outcomes as inputs, or bypass a source gate.

## Stopping rule

Any provenance, causality, source-support, novelty, train, or selection failure
retires `RCRE-72-SOURCE-REUSE` unchanged. A successor requires a new mechanism,
new ID, and fresh preregistration; no threshold, sign, source, support-floor,
hold, or side repair is permitted.
