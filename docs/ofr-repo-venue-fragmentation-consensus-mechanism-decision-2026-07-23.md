# RVFC-72 mechanism decision — 2026-07-23

## Decision

Preregister one new-source, candidate-incidence-blind singleton:
**RVFC-72 — Repo Venue Fragmentation Consensus**.

RVFC combines four weak, independently interpretable observables from the OFR
preliminary U.S. Repo Markets release. No component is treated as a standalone
alpha. The hypothesis is that agreement among venue-rate dispersion, venue
volume concentration, collateral-rate disagreement, and collateral-mix
disagreement identifies temporary fragmentation or normalization in secured
dollar funding that can transmit to BTC risk appetite over 72 elapsed hours.

This document freezes source fields, exact arithmetic, prior-only
normalization, direction, event construction, execution, controls, support
gates, novelty tests, staged economic gates, and the no-repair rule before any
RVFC component, rank, state, candidate incidence, comparator row, or BTC
outcome is computed.

## Evidence and contamination boundary

The OFR source audit opened only metadata, row/date counts, missingness,
disclosure-edit structure, and deterministic hashes. It did not compute a
cross-series spread, share, concentration, rank, state, event, side, market
return, or PnL. This is the first candidate on the OFR repo-segmentation source
axis.

Broader USD-liquidity research is not pristine. This repository has already
opened outcomes for H.4.1, overnight RRP, SOFR, H.8, Treasury, and NY Fed SOMA
securities-lending candidates. RVFC therefore makes no claim to be independent
merely because its exact OFR source is new. It must pass the frozen clock and
occupied-exposure novelty battery against those families and the live
portfolio before any market outcome is opened.

## Frozen source and causal availability

Source artifacts:

- normalized preliminary observations:
  `data/ofr_repo_preliminary_2019_2023/ofr_repo_preliminary_observations_2019_2023.csv.gz`;
- metadata definitions:
  `data/ofr_repo_preliminary_2019_2023/ofr_repo_preliminary_metadata_2019_2023.json.gz`;
- source manifest:
  `data/ofr_repo_preliminary_2019_2023/build_manifest.json`.

Official references:

- <https://www.financialresearch.gov/short-term-funding-monitor/datasets/repo/>;
- <https://www.financialresearch.gov/short-term-funding-monitor/documentation/>;
- <https://www.financialresearch.gov/short-term-funding-monitor/api/>.

Only preliminary (`-P`) aggregation rows may be used. Final values,
out-of-window disclosure markers, null values, interpolation, forward filling,
and post-2023 observations are forbidden. A source date is complete only when
all fourteen required series below have one finite non-null row for that exact
`observation_date`; required volume denominators must be strictly positive.
In addition, agency and Treasury collateral must each represent at least 5% of
the `AG + T` transaction volume inside both GCF and TRIV1. A failed 5% materiality
gate invalidates the whole date and breaks continuity; it cannot be relaxed
after source incidence is seen.

The vector availability is the maximum `available_at_utc` across its required
rows. Every required row is already subject to the frozen source clock:

```text
max(observation_date + 8 elapsed calendar days,
    2020-09-10 00:00:00 UTC)
```

Several historical dates share the publication floor. Equal-availability rows
form one causal batch: only the greatest complete `observation_date` in that
batch may become a decision row. Earlier rows in the batch may enter the
strict-prior history because the whole batch is known, but they may not create
sequential same-timestamp events.

## Required series

### Venue totals

- `REPO-DVP_AR_TOT-P`, `REPO-GCF_AR_TOT-P`, `REPO-TRIV1_AR_TOT-P`;
- `REPO-DVP_TV_TOT-P`, `REPO-GCF_TV_TOT-P`, `REPO-TRIV1_TV_TOT-P`.

`TRIV1` is the OFR definition explicitly excluding Federal Reserve
transactions. The corresponding `TRI` series is forbidden so the primary does
not mechanically encode Federal Reserve tri-party activity already studied in
overnight-RRP research.

### Collateral subdivisions

- `REPO-GCF_AR_AG-P`, `REPO-GCF_AR_T-P`;
- `REPO-TRIV1_AR_AG-P`, `REPO-TRIV1_AR_T-P`;
- `REPO-GCF_TV_AG-P`, `REPO-GCF_TV_T-P`;
- `REPO-TRIV1_TV_AG-P`, `REPO-TRIV1_TV_T-P`.

`AG` is Federal Agency/GSE collateral and `T` is U.S. Treasury collateral.
The source-audit-rejected sparse GCF tenor buckets `G30`, `LE30`, and `OO` are
not used.

## Four frozen weak-signal components

Parse every published decimal as an exact rational number. Feature comparison,
midrank ties, ratios, and HHI use exact rational arithmetic; binary floating
point is forbidden.

For complete source date `t`, define:

### 1. Venue total-rate dispersion

```text
rate_dispersion[t]
  = max(DVP_AR_TOT, GCF_AR_TOT, TRIV1_AR_TOT)
    - min(DVP_AR_TOT, GCF_AR_TOT, TRIV1_AR_TOT)
```

This removes the common rate level and retains disagreement across clearing
and settlement venues.

### 2. Venue transaction-volume concentration

```text
V = DVP_TV_TOT + GCF_TV_TOT + TRIV1_TV_TOT
venue_hhi[t] = (DVP_TV_TOT / V)^2
             + (GCF_TV_TOT / V)^2
             + (TRIV1_TV_TOT / V)^2
```

`V` must be strictly positive. HHI measures migration toward one venue without
using the sign of any rate move.

### 3. Collateral-rate disagreement

```text
collateral_rate_disagreement[t]
  = (abs(GCF_AR_AG - GCF_AR_T)
     + abs(TRIV1_AR_AG - TRIV1_AR_T)) / 2
```

This asks whether Treasury and agency collateral are priced differently across
two repo venues, rather than whether the whole secured-rate level rose.

### 4. Collateral-mix disagreement

```text
gcf_agency_share = GCF_TV_AG / (GCF_TV_AG + GCF_TV_T)
tri_agency_share = TRIV1_TV_AG / (TRIV1_TV_AG + TRIV1_TV_T)

collateral_mix_disagreement[t]
  = abs(gcf_agency_share - tri_agency_share)
```

Both share denominators must be strictly positive. The component measures
cross-venue disagreement in collateral composition, not total activity.

## Strict-prior normalization

For each component independently, use exactly the previous 252 complete source
dates in `observation_date` order. The current date is excluded. No expanding
fallback, calendar interpolation, fitted mean/variance, or outcome-selected
threshold is allowed.

```text
midrank[t] = (count(prior < current)
              + 0.5 * count(prior == current)) / 252
u[t] = 2 * midrank[t] - 1
```

Ties use exact rational equality. A missing required row breaks state
continuity. The next complete rank-ready row may establish a state but cannot
trigger an event.

## Frozen state and direction

```text
positive_votes = count(component u > 0)
negative_votes = count(component u < 0)
score = mean(the four component u values)

HIGH = positive_votes >= 3 and score >= +0.50
LOW  = negative_votes >= 3 and score <= -0.50
NEUTRAL = otherwise
```

Zero components do not vote. A candidate occurs only when the current state is
`HIGH` or `LOW` and differs from the immediately prior complete rank-ready
state. Persistence inside the same state does not retrade.

Direction is frozen:

- `HIGH`: broad secured-funding fragmentation/concentration, **SHORT BTC**;
- `LOW`: broad normalization/distribution, **LONG BTC**.

No component can reverse or veto the fixed state direction after outcomes are
opened.

## Frozen execution

- signal: the current vector's conservative `available_at_utc`;
- entry: `ceil_to_5m(signal) + 5 elapsed minutes`, including exact-grid signals;
- exit: exactly 72 elapsed hours / 864 five-minute bars later;
- fixed BTCUSDT perpetual notional exposure: 0.5x;
- one global chronological reservation on `[entry, exit)`;
- accept only when entry is at or after the previous accepted exit;
- suppressed candidates are not queued;
- entry and exit must remain in the same declared split;
- no stop, take-profit, trailing exit, dynamic size, price gate, regime gate,
  direction override, or leverage search.

## Frozen windows and source-support gates

- warmup/source history: 2019–2020;
- train clock: `[2021-01-01, 2023-01-01)` by entry time;
- selection clock: `[2023-01-01, 2024-01-01)` by entry time;
- sealed from: `2024-01-01T00:00:00Z`.

Before any comparator row or market outcome is loaded, the accepted primary
clock must satisfy all of:

- train total at least 60;
- each of 2021 and 2022 at least 20;
- each train half-year at least 8;
- train at least 15 long and 15 short;
- selection total at least 20;
- each 2023 half-year at least 7;
- selection at least 5 long and 5 short;
- every train and selection quarter active;
- maximum UTC-month share at most 15% in train and 20% in selection;
- maximum accepted-entry gap at most 45 elapsed days;
- exact timing, uniqueness, split containment, and non-overlap; and
- no required component or denominator failure on an accepted event.

Any failure rejects RVFC-72 before comparator access and outcomes. Observed
incidence may not lower a floor, change 252 or 0.50, add persistence entries,
drop a component, replace `TRIV1` with `TRI`, or alter the hold.

## Frozen controls and falsifications

Source/component controls use the same latency, transition construction,
72-hour hold, split containment, and global non-overlap:

1. **component-only clocks:** for each component, `HIGH` at `u >= +0.50`,
   `LOW` at `u <= -0.50`, and `NEUTRAL` otherwise; use the exact primary
   transition rule;
2. **mean_without_consensus:** `HIGH` when the four-component mean is at least
   `+0.50`, `LOW` when it is at most `-0.50`, neutral otherwise; no vote gate;
3. **same_sign_without_magnitude:** `HIGH` on at least three strictly positive
   components and `LOW` on at least three strictly negative components; ignore
   score magnitude;
4. **rate_family_only:** use components 1 and 3; `HIGH` only when both are at
   least `+0.50`, `LOW` only when both are at most `-0.50`;
5. **volume_family_only:** the same rule on components 2 and 4;
6. **four leave-one-component clocks:** on each remaining three-component
   vector, require at least two same-sign votes and remaining-component mean
   at least `+0.50` or at most `-0.50`;
7. **one_complete_day_stale / five_complete_day_stale:** replace the current
   rank vector by the vector one or five prior complete rank-ready source dates
   old, retain the current availability as the signal, and apply the exact
   primary state/transition rule to the stale sequence;
8. **year_component_permutation:** independently for each component and
   observation year, order source rows by
   `SHA256("RVFC-72|year_component_permutation|<year>|<component>|<observation_date>")`;
   assign component values in that hashed source order to destination dates in
   chronological order, then apply the primary state/transition rule.

Economic controls reuse the exact accepted primary entries/exits:

- `exact_direction_flip`: multiply every primary side by `-1`;
- `deterministic_random_side`: LONG iff the first byte of
  `SHA256("RVFC-72|deterministic_random_side|<entry_time_utc_iso>")` is below
  128, otherwise SHORT;
- `constant_long`: side is always LONG; and
- `constant_short`: side is always SHORT.

No control may replace the primary after outcomes open. The primary must beat
both family controls and all four leave-one-component controls on train and
selection risk-adjusted performance; otherwise the four-weak-signal interaction
claim is rejected.

The source-support artifact must also report, without changing eligibility or
side, the dominant total-rate venue, dominant total-volume venue, dominant
collateral-rate-spread venue, and their event/year shares. These diagnostics
are frozen falsifications for a one-venue explanation, never a post hoc gate or
direction repair.

## Novelty gate

Only after every source-support gate passes may the builder read frozen
comparator clocks. The cohort is exactly the following hash-bound files; no
later passed/failed status may add, remove, or select a group:

| Frozen comparator artifact | SHA-256 | Required groups |
|---|---|---|
| `results/overnight_rrp_flow_release_clocks_2026-07-17.csv.gz` | `7242d9870627dfc0cf067ff87d9664a1576dd374cb8985e927b40f15d1e3d480` | every `control` |
| `results/overnight_rrp_participant_breadth_support_clocks_2026-07-21.csv.gz` | `ef21323229801f11557e0c2d9d4465f7d58b13569552d656d64fdb7d440622ed` | every `control` |
| `results/federal_liquidity_component_concordance_preregistered_clock_2026-07-17.csv.gz` | `7ebb0450422d9265e46c596e0b6415b6a8816c66f5e0cbb9ccda14ca6cb4c67c` | every `candidate_id × clock_name` |
| `results/daily_treasury_fiscal_flow_breadth_primary_clock_2026-07-21.csv.gz` | `df53e1a27fcbc6ea2c4bc3f462a557a75c76a98db3c362944dad0b4d74382978` | every `policy_id × clock` |
| `results/daily_treasury_fiscal_flow_breadth_control_clocks_2026-07-21.csv.gz` | `416fc8663b292fcee069e4aca53b83e99a05b594a96940ab2c557e6e0d05e312` | every `policy_id × clock` |
| `results/sofr_rate_dislocation_preregistered_clock_2026-07-17.csv.gz` | `391c42dd2b0d5b87ffcd73058dd9fa0c4d18fd2f535597effff5a4c8edea2e69` | fixed `SFRD-1|primary` |
| `results/bank_deposit_secured_repo_concordance_clocks_2026-07-20.csv.gz` | `1ff3a6075e3ceff928e1dd19d05880dbe9dbab0e07d79b853146d7b4c8f6cabc` | every `clock_name` |
| `results/fed_h8_deposit_migration_preregistered_clock_2026-07-18.csv.gz` | `20405f79b86861adcc784c81223baae1c40fdf3c73edda339578471a6a6d1b40` | `clock_mode == primary` |
| `results/soma_lending_collateral_scarcity_clocks_2026-07-23.csv.gz` | `b3fe0dc8c895f9a8974cdf08b5bed9d58ff693b8aca7ed59c224627de930a948` | `control == primary` |
| `results/cross_domain_liquidity_transmission_relay_support_clock_2026-07-21.csv.gz` | `aa2bcafd0f62ebe585f93cbd357d29c37ae526a95a90b8a6c0bd7c068cd6e5a1` | every `clock` |
| `results/cchr_live_portfolio_pure_clocks_2026-07-21.csv.gz` | `73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08` | every `candidate_id` |

This includes prior H.4.1/H.8, overnight-RRP, SOFR, Treasury-fiscal,
bank-deposit/repo, rejected SLCS, and live clocks regardless of whether those
families later passed economics.

Over common 2021–2023 coverage, every comparator group with at least ten
entries must satisfy:

- exact-entry Jaccard at most 0.10;
- one-to-one RVFC containment within ±24 elapsed hours at most 0.35; and
- absolute signed occupied-exposure correlation at most 0.35.

Missing, hash-mismatched, malformed, empty required, overlapping, or post-2023
comparator clocks fail closed. Controls are specificity checks, not novelty
substitutes.

## Strict economic sequence

Only an unchanged source-support and novelty pass may authorize a separately
tested and committed evaluator. It opens train 2021–2022 first and selection
2023 only after the full train pass. Immutable source extension and 2024 test,
2025 eval, and 2026 recent evaluation remain sequentially sealed.

Every opened primary split requires positive full-calendar absolute return,
full-calendar `CAGR / strict MDD >= 3.0`, strict MDD at most 15%, positive return
under 10 bp/notional/side stress, exact funding, sufficient gross edge, at
least 20 executed trades, at least 5 per side, positive required subperiods,
calendar-month clustered sign-flip `p <= 0.10`, primary CAGR/MDD strictly above
both family-only and every leave-one-component control, and failure of the
component, stale, permutation, flipped, random, and constant-side explanations.

Base cost is 6 bp/notional/side. Strict MDD includes pre-entry high water and
intratrade adverse marks, with favorable observations ordered before adverse
observations and virtual adverse exit cost.

## RLLM boundary

RLLM is disabled until the unchanged deterministic policy passes source,
novelty, train, and selection. A later compact model may receive only causal
bucketed component ranks, state-transition reasons, current position,
time-in-position, and risk budget. It may choose only
`TRADE_FIXED_SIDE` or `ABSTAIN`; it cannot create events, reverse direction,
change hold/leverage, inspect future values, or repair a failed gate.

## Stopping rule

Any provenance, causality, source-support, specificity, novelty, train, or
selection failure retires `RVFC-72-NEW-SOURCE` unchanged. Any feature, source,
direction, threshold, prior length, clock, hold, support floor, or comparator
change requires a new candidate identity committed before access.
