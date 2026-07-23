# RMSR-72 mechanism decision — 2026-07-23

## Decision

Preregister one source-reuse, exact-race-incidence-blind singleton:
**RMSR-72 — Repo Mix Shock Resolution Race**.

RMSR does not average weak signals and does not repair RVFC-72. It treats a
cross-venue collateral-composition extreme as an unpriced quantity shock and
observes which of two strictly later events happens first:

1. collateral-rate disagreement enters the same-polarity extreme, confirming
   that the quantity shock propagated into secured-funding prices; or
2. collateral-mix disagreement exits its extreme, showing that the quantity
   shock was absorbed before same-polarity repricing.

The first terminal event fixes direction. A confirmed high-fragmentation shock
is SHORT and a confirmed low-fragmentation normalization is LONG. An absorbed
shock takes the opposite side of that failed propagation. The economic object
is therefore a signed first-passage race, not a simultaneous consensus,
single-component threshold, persistence trade, or threshold relaxation.

This document freezes source fields, exact arithmetic, normalization, state
machine, direction, timeout, execution, controls, support gates, novelty tests,
economic sequence, and no-repair rule before RMSR race incidence, comparator
rows, or any BTC outcome is computed.

## Evidence and contamination boundary

The earlier OFR audit and RVFC source-support run opened the source values,
four RVFC component clocks, and their source-only incidence through 2023. It
also established that individual collateral-mix and collateral-rate transition
clocks are materially denser than RVFC's simultaneous consensus. Consequently:

- RMSR is **not** source-value-blind;
- the exact RMSR precursor/terminal pairing, race incidence, side mix, timeout
  rate, comparator overlap, and market outcomes remain unopened;
- the two unused RVFC components cannot be introduced later as gates; and
- RMSR makes no pristine-source or pristine-broad-liquidity claim.

The RVFC run read zero BTC bars, funding rows, future returns, PnL, CAGR, or MDD.
RMSR must independently pass source support and frozen clock novelty before an
evaluator may be frozen. Any failure retires this identity unchanged.

## Frozen source and causal availability

Source artifacts:

- `data/ofr_repo_preliminary_2019_2023/ofr_repo_preliminary_observations_2019_2023.csv.gz`;
- `data/ofr_repo_preliminary_2019_2023/ofr_repo_preliminary_metadata_2019_2023.json.gz`;
- `data/ofr_repo_preliminary_2019_2023/build_manifest.json`.

Official references:

- <https://www.financialresearch.gov/short-term-funding-monitor/datasets/repo/>;
- <https://www.financialresearch.gov/short-term-funding-monitor/documentation/>;
- <https://www.financialresearch.gov/short-term-funding-monitor/api/>.

Only preliminary (`-P`) rows may be used. Final values, disclosure markers,
nulls, interpolation, forward filling, and post-2023 observations are
forbidden. A date is complete only when every required series has exactly one
finite non-null row for the same `observation_date`.

The vector availability is the maximum `available_at_utc` across all required
rows. Every row remains subject to the conservative source clock:

```text
max(observation_date + 8 elapsed calendar days,
    2020-09-10 00:00:00 UTC)
```

Rows sharing an availability timestamp form one causal batch. All complete
rows in the batch may enter later strict-prior history, but only the greatest
complete `observation_date` may be a decision row. No two same-availability
dates can form separate race legs.

## Required series and materiality

RMSR uses exactly eight preliminary series:

- `REPO-GCF_AR_AG-P`, `REPO-GCF_AR_T-P`;
- `REPO-TRIV1_AR_AG-P`, `REPO-TRIV1_AR_T-P`;
- `REPO-GCF_TV_AG-P`, `REPO-GCF_TV_T-P`;
- `REPO-TRIV1_TV_AG-P`, `REPO-TRIV1_TV_T-P`.

`TRIV1` excludes Federal Reserve transactions. `TRI`, DVP, venue-total rows,
and sparse GCF tenor buckets are forbidden. They may not become gates,
features, direction overrides, or tie breakers after incidence is opened.

For both GCF and TRIV1, `AG + T` transaction volume must be strictly positive,
and agency and Treasury collateral must each be at least 5% of that venue's
`AG + T` volume. Failure invalidates the whole date and breaks continuity.

## Frozen exact features

Every source decimal is parsed as an exact rational. Ratios, absolute
differences, ranks, ties, and comparisons use exact rational arithmetic;
binary floating point is forbidden.

```text
gcf_agency_share = GCF_TV_AG / (GCF_TV_AG + GCF_TV_T)
tri_agency_share = TRIV1_TV_AG / (TRIV1_TV_AG + TRIV1_TV_T)

mix_disagreement
  = abs(gcf_agency_share - tri_agency_share)

rate_disagreement
  = (abs(GCF_AR_AG - GCF_AR_T)
     + abs(TRIV1_AR_AG - TRIV1_AR_T)) / 2
```

The source-support artifact must report the venue contributing the larger
absolute collateral-rate spread at each accepted terminal. Exact ties are
reported as ties. This is a frozen diagnostic and source-support
concentration gate, never a side override.

## Strict-prior normalization and states

Each feature independently uses exactly the previous 252 complete source dates
in `observation_date` order. The current date is excluded. There is no
expanding fallback, calendar interpolation, fitted transform, or alternate
window.

```text
midrank[t] = (count(prior < current)
              + 0.5 * count(prior == current)) / 252
u[t] = 2 * midrank[t] - 1

state(u) = +1 when u >= +0.50
         = -1 when u <= -0.50
         =  0 otherwise
```

Ties use exact equality. A precursor transition into polarity `p` occurs only
when current `mix_state == p` and immediately prior continuous
`mix_state != p`, where `p` is `+1` or `-1`.

A missing or invalid date breaks continuity and cancels an active race. The
first complete rank-ready row after a break establishes states but cannot arm
or terminate a race.

## Frozen first-passage race

### Arming

When no race is active, a mix transition into polarity `p` arms a race only if
current `rate_state != p`. If the rate state is already `p`, the precursor is
classified `already_priced` and discarded permanently.

The precursor date cannot itself terminate the race. The window consists of
the next **20 complete rank-ready decision dates**, indexed 1 through 20.

### Terminal events

On each strictly later in-window date, evaluate both terminals from states
available on that date:

```text
rate_confirmation = current rate_state == p
                    and prior rate_state != p
mix_exit = current mix_state != p
           and prior mix_state == p
```

- If `rate_confirmation` is true and `mix_exit` is false, emit
  `PRICE_CONFIRMATION` and set `side = -p`.
- If `mix_exit` is true and `rate_confirmation` is false, emit
  `QUANTITY_ABSORPTION` and set `side = +p`.
- If both are true on the same date, classify `AMBIGUOUS_SAME_DATE`, cancel the
  race, emit no candidate, and do not re-arm the new mix state that day.
- If neither occurs through date 20, classify `TIMEOUT` and emit no candidate.

Thus the four frozen economic paths are:

| Precursor | First terminal | Side | Interpretation |
|---|---|---:|---|
| high mix (`p=+1`) | same-high rate confirmation | SHORT | fragmentation repriced |
| high mix (`p=+1`) | mix exits first | LONG | fragmentation absorbed |
| low mix (`p=-1`) | same-low rate confirmation | LONG | normalization repriced |
| low mix (`p=-1`) | mix exits first | SHORT | normalization rejected |

Every precursor is consumed by confirmation, absorption, ambiguity, timeout,
or continuity break. Suppressed and expired races are never queued. A terminal
date cannot simultaneously become a new precursor.

## Frozen execution

- signal: terminal vector's conservative `available_at_utc`;
- entry: `ceil_to_5m(signal) + 5 elapsed minutes`, including exact-grid times;
- exit: exactly 72 elapsed hours / 864 five-minute bars later;
- fixed BTCUSDT perpetual notional exposure: 0.5x;
- one global chronological reservation on `[entry, exit)`;
- accept only when entry is at or after the previous accepted exit;
- suppressed candidates are not queued;
- entry and exit must remain inside one declared split;
- no stop, take-profit, trailing exit, dynamic size, market-price gate,
  external regime gate, direction override, or leverage search.

## Frozen windows and source-support gates

- warmup/source history: 2019–2020;
- train clock: `[2021-01-01, 2023-01-01)` by entry time;
- selection clock: `[2023-01-01, 2024-01-01)` by entry time;
- sealed from: `2024-01-01T00:00:00Z`.

Before any comparator row or market outcome is read, the accepted primary clock
must satisfy all of:

- train total at least 35;
- each of 2021 and 2022 at least 12;
- each train half-year at least 5;
- train at least 8 LONG and 8 SHORT;
- selection total at least 14;
- each selection half-year at least 4;
- selection at least 3 LONG and 3 SHORT;
- every train and selection quarter active;
- maximum UTC-month share at most 20% in train and 25% in selection;
- maximum accepted-entry gap at most 90 elapsed days;
- each terminal type at least 20% of train and 15% of selection events;
- no non-tie dominant collateral-rate-spread venue above 85% in train or
  selection;
- exact timing, uniqueness, split containment, and non-overlap;
- every accepted terminal has a strictly earlier eligible precursor, age 1–20,
  no ambiguity, and complete exact-rational features; and
- no post-2023 source row is read.

Any failure rejects RMSR-72 before comparator access and outcomes. Observed
incidence may not change 252, 0.50, 20 dates, the race priority, terminal side,
support floors, required source, 72-hour hold, or execution.

## Frozen controls and falsifications

Every source control uses the same causal availability, exact state
definitions, entry latency, split containment, 72-hour hold, and global
non-overlap unless its stated purpose requires a different race construction:

1. `mix_transition_only`: every eligible mix precursor, `side = -p`;
2. `rate_transition_only`: every rate transition into `p`, `side = -p`;
3. `price_confirmation_only`: primary confirmation terminals only;
4. `quantity_absorption_only`: primary absorption terminals only;
5. `reverse_race`: rate transition into `p` is the unpriced precursor and mix
   confirmation versus rate exit is the terminal race, with analogous sides;
6. `five_date_window`: exact primary with a five-date timeout;
7. `forty_date_window`: exact primary with a forty-date timeout;
8. `one_complete_date_stale`: evaluate the two state sequences one complete
   decision date stale at the current availability;
9. `five_complete_date_stale`: the same with five complete dates stale;
10. `year_rate_permutation`: within each observation year, assign rate-state
    vectors in deterministic order using
    `SHA256("RMSR-72|year_rate_permutation|<year>|<observation_date>")` while
    preserving mix states and current availability; and
11. `same_date_alignment`: emit only when a mix precursor and same-polarity
    rate transition occur on the same decision date, `side = -p`.

Economic side controls reuse the exact accepted primary entries and exits:

- `exact_direction_flip`;
- deterministic random side from the first byte of
  `SHA256("RMSR-72|deterministic_random_side|<entry_time_utc_iso>")`;
- `constant_long`; and
- `constant_short`.

No control may replace the primary. The primary must beat both component-only,
reverse-race, adjacent-window, stale, and same-date controls under the frozen
train and selection economic gates. Confirmation-only and absorption-only
must each have positive train and selection returns; otherwise the claimed
two-terminal resolution mechanism is rejected.

## Frozen novelty gate

Only after every source-support gate passes may comparator rows be opened. The
comparator cohort is the exact hash-bound RVFC cohort plus the rejected RVFC
primary clock. It includes overnight RRP, Federal liquidity, Treasury fiscal,
SOFR, H.8/deposit-repo, SOMA lending, cross-domain liquidity, live portfolio,
and `RVFC-72|primary` clocks regardless of prior economic status.

Over common 2021–2023 coverage, every comparator group with at least ten
entries must satisfy:

- exact-entry Jaccard at most 0.10;
- one-to-one RMSR containment within ±24 elapsed hours at most 0.35; and
- absolute signed occupied-exposure correlation at most 0.35.

Missing, hash-mismatched, malformed, empty required, overlapping, or post-2023
comparator clocks fail closed. RMSR's constituent component controls are
specificity controls rather than novelty comparators because a terminal is
structurally tied to one constituent transition.

## Strict economic sequence

Only a complete source-support and novelty pass may authorize a separate,
hash-bound evaluator freeze. The sequence is then:

1. train 2021–2022 only;
2. selection 2023 only after exact train pass;
3. immutable source extension and test 2024 only after pre-2024 pass;
4. eval 2025 only after test pass; and
5. recent 2026 only after eval pass.

Each opened economic split requires positive absolute return, full-calendar
CAGR / strict intratrade MDD at least 3.0, strict MDD at most 15%, realized
funding, 6 bps notional cost per side, positive return under 10 bps stress,
the frozen minimum trade and side counts, positive required subperiods, and a
calendar-month cluster sign-flip p-value at most 0.10. CAGR always includes
inactive calendar time.

The primary must also beat the frozen mechanism controls. No train or
selection failure can be repaired with a different timeout, rank window,
threshold, source subset, side mapping, hold, gate, or control.

## RLLM boundary

RLLM is unauthorized before deterministic source support, novelty, train, and
selection all pass. A later compact model may choose only
`TRADE_FIXED_SIDE` or `ABSTAIN` from causal, bucketed race-state text and
current position/risk context. It may not create an event, reverse side, alter
size or hold, use market outcomes as prompt features, or bypass a source gate.

## Stopping rule

Any provenance, causality, source-support, novelty, train, or selection failure
retires `RMSR-72-SOURCE-REUSE` unchanged. A successor requires a new economic
mechanism, new ID, and fresh preregistration; it cannot be a parameter repair.
