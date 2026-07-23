# DMSH-168 mechanism decision — 2026-07-23

## Decision

Preregister one source-reuse, candidate-incidence-blind singleton:
**DMSH-168 — DVP Maturity Stock-Flow Handoff**.

DMSH asks whether a change in the maturity composition of newly initiated DVP
repo transactions is subsequently acknowledged by the DVP term-versus-overnight
rate curve. It uses the difference between the overnight share of current
transaction flow and the overnight share of outstanding inventory as a quantity
precursor. A trade exists only when the rate curve moves into the same strict-prior
extreme on a later source date. An unconfirmed or contradicted precursor never
trades.

The causal claim is a delayed maturity handoff:

- unusually overnight-heavy new flow relative to the outstanding book, followed
  by an unusually expensive term curve, is secured-funding rollover pressure and
  is **SHORT BTC**;
- unusually term-heavy new flow relative to the outstanding book, followed by an
  unusually cheap term curve, is maturity extension and is **LONG BTC**.

This is not another simultaneous OFR fragmentation consensus, not the RMSR
two-terminal absorption race, and not the RCRE cross-venue collateral product.
It uses one venue, signed maturity stock-versus-flow composition, strict temporal
ordering, and only the price-confirmation terminal. No absorption, timeout, or
direction-flip terminal creates a trade.

This document freezes source fields, exact arithmetic, causal availability,
normalization, state machine, direction, execution, controls, support gates,
novelty gates, staged economic sequence, and the no-repair rule before any DMSH
feature, state, candidate incidence, comparator overlap, or BTC outcome is
computed.

## Evidence and contamination boundary

The OFR preliminary source has already been normalized and source-audited. Source
rows and prior OFR candidate incidence are therefore not pristine. RVFC opened
venue-total dispersion/concentration and absolute collateral disagreement; RMSR
opened its collateral-mix/rate race; RCRE opened signed GCF-versus-TRIV1
collateral routing. None opened the DMSH maturity-flow gap, DVP curve gap, DMSH
state, or DMSH event incidence.

During selection of this successor, the repository inspected OFR metadata and a
20-row normalized head sample that included DVP rows. It also counted valid DVP
series rows as a source-coverage check. It did not compute a component ratio,
rank, state, event, comparator statistic, BTC return, funding return, PnL, CAGR,
or MDD. The exact DMSH candidate incidence remains unopened at this decision.

Broader USD-liquidity and BTC outcomes have been opened elsewhere in this
repository. DMSH makes no pristine-discovery claim and must pass a frozen novelty
battery against those clocks before any market outcome is accessed.

## Frozen source and causal availability

Source artifacts:

- `data/ofr_repo_preliminary_2019_2023/ofr_repo_preliminary_observations_2019_2023.csv.gz`;
- `data/ofr_repo_preliminary_2019_2023/ofr_repo_preliminary_metadata_2019_2023.json.gz`;
- `data/ofr_repo_preliminary_2019_2023/build_manifest.json`.

Governing source audit:

- `docs/ofr-repo-preliminary-source-audit-2026-07-23.md`.

Official references:

- <https://www.financialresearch.gov/short-term-funding-monitor/datasets/repo/>;
- <https://www.financialresearch.gov/short-term-funding-monitor/documentation/>;
- <https://www.financialresearch.gov/short-term-funding-monitor/api/>.

Only preliminary (`-P`) aggregation rows may be used. Final values,
disclosure-edit flag `1`, null values, interpolation, forward filling, and
post-2023 observations are forbidden. A complete date requires exactly one
finite, non-null, disclosure-edit-zero row for every required mnemonic:

- outstanding volume: `REPO-DVP_OV_OO-P`, `REPO-DVP_OV_LE30-P`,
  `REPO-DVP_OV_G30-P`;
- transaction volume: `REPO-DVP_TV_OO-P`, `REPO-DVP_TV_LE30-P`,
  `REPO-DVP_TV_G30-P`;
- average rate: `REPO-DVP_AR_OO-P`, `REPO-DVP_AR_LE30-P`,
  `REPO-DVP_AR_G30-P`.

`OO` is overnight/open, `LE30` is term at most 30 days, and `G30` is term over
30 days under the frozen OFR metadata. The DVP total series, the finer `B27` and
`B830` descendants, GCF, TRI, and TRIV1 are forbidden. Component-sum denominators
are used directly; no possibly rounded total is used to repair or validate a
date.

Every required volume must be nonnegative. The following denominators must be
strictly positive:

```text
OV_OO + OV_LE30 + OV_G30
TV_OO + TV_LE30 + TV_G30
TV_LE30 + TV_G30
```

Every required row remains subject to the source-audit clock:

```text
max(observation_date + 8 elapsed calendar days,
    2020-09-10 00:00:00 UTC)
```

Vector availability is the maximum availability of its nine rows. Rows sharing
one availability timestamp form one causal batch. Each batch is ranked only
against rows available before that batch. After all ranks are computed, complete
batch rows may enter later history. Only the greatest complete observation date
in the batch may establish a state, arm a precursor, confirm a handoff, or cancel
one.

An invalid source date is causally known at the maximum `available_at_utc` among
the required rows that are present for that date. A required row that is present
with a null value or disclosure-edit flag `1` therefore invalidates the vector at
that timestamp. If at least one required mnemonic has a row for a date but
another required mnemonic has no row, the vector is missing and uses the maximum
availability of the present required rows as its invalidation timestamp. A date
with no row for any of the nine required mnemonics is not an OFR source date and
does not manufacture a break. An invalid vector cancels a pending precursor at
its defined invalidation timestamp and creates no state, precursor, confirmation,
or candidate. No future complete row may retroactively move that cancellation.

## Frozen exact components

All source decimals, additions, products, ratios, comparisons, and midrank ties
use exact rational arithmetic. Binary floating point is forbidden.

For complete date `t`:

```text
stock_overnight_share[t]
  = OV_OO / (OV_OO + OV_LE30 + OV_G30)

flow_overnight_share[t]
  = TV_OO / (TV_OO + TV_LE30 + TV_G30)

maturity_flow_gap[t]
  = flow_overnight_share[t] - stock_overnight_share[t]

term_rate[t]
  = (AR_LE30 * TV_LE30 + AR_G30 * TV_G30)
    / (TV_LE30 + TV_G30)

curve_gap[t] = term_rate[t] - AR_OO
```

Positive `maturity_flow_gap` means new DVP activity is more overnight-heavy than
the outstanding book. Negative means current flow is extending maturity relative
to stock. Positive `curve_gap` means term repo is more expensive than
overnight/open on the same source date.

## Strict-prior normalization

Normalize the two components independently against exactly the previous 252
complete source dates in observation-date order. The current row and all rows in
its equal-availability batch are excluded. There is no expanding fallback,
calendar interpolation, winsorization, fitted transform, or alternate lookback.

```text
midrank_x[t]
  = (count(prior_x < x[t]) + 0.5 * count(prior_x == x[t])) / 252

u_x[t] = 2 * midrank_x[t] - 1
```

Ties use exact rational equality. The component states are:

```text
FLOW_COMPRESSION = +1 when u_maturity_flow_gap >= +0.50
FLOW_EXTENSION   = -1 when u_maturity_flow_gap <= -0.50
FLOW_NEUTRAL      = 0 otherwise

CURVE_STRESS = +1 when u_curve_gap >= +0.50
CURVE_EASE   = -1 when u_curve_gap <= -0.50
CURVE_NEUTRAL = 0 otherwise
```

## Frozen ordered-handoff state machine

One global state machine operates on causal decision rows. For each valid row,
state transitions are computed first and the following priority is then applied
exactly once:

1. A precursor may arm only when the flow state transitions into `p ∈ {+1,-1}`
   from a different immediately prior continuous flow state.
2. The curve state on the precursor row must not already equal `p`; otherwise the
   quantity change is treated as already priced and no precursor arms.
3. A pending precursor observes the next ten complete causal decision rows. The
   precursor row itself can never confirm. The first later valid row has age one.
4. On each observed row, a flow transition into `-p` or a curve transition into
   `-p` is processed first and cancels the precursor without a trade. This
   contradiction priority applies even if the curve also transitions into `p`
   on the same row through an implementation or data inconsistency; such a row
   must fail closed rather than confirm.
5. Only when no contradiction occurred, the first strictly later transition of
   the curve state into the same polarity `p` confirms the handoff and creates
   one candidate. Confirmation at age ten is allowed.
6. If age ten has been processed without contradiction or confirmation, the
   precursor expires after that row without a trade.
7. A neutral flow or curve row neither confirms nor cancels. A same-polarity flow
   retrigger cannot replace, extend, or refresh a pending precursor.
8. A row that confirms, contradicts, expires, or carries an invalid vector cannot
   also arm a new precursor. After termination, a new precursor requires an
   eligible flow-state transition on a later source row; suppressed transitions
   are not queued.

Candidate side is fixed:

- confirmed `p=+1`: **SHORT BTC**;
- confirmed `p=-1`: **LONG BTC**.

The accepted clock must record precursor date/availability, confirmation
date/availability, polarity, age in complete decision rows, entry, exit, and
side. Every confirmation must be strictly later than its precursor and age must
be in `[1,10]`.

## Frozen execution

- signal: confirmation vector's conservative `available_at_utc`;
- entry: `ceil_to_5m(signal) + 5 elapsed minutes`, including exact-grid signals;
- exit: exactly 168 elapsed hours / 2,016 five-minute bars later;
- fixed BTCUSDT perpetual notional exposure: 0.5x;
- one global chronological reservation on `[entry, exit)`;
- accept only when entry is at or after the prior accepted exit;
- suppressed candidates are never queued;
- entry and exit must be contained in one declared split;
- no stop, take-profit, trailing exit, dynamic size, price gate, external regime
  gate, direction override, leverage search, or alternate hold.

The seven-day hold is frozen because the observable is a delayed daily secured
funding inventory adjustment, not an intraday market microstructure shock.

## Frozen windows and source-support gates

- warmup/source history: 2019–2020;
- train clock: `[2021-01-01, 2023-01-01)` by entry time;
- selection clock: `[2023-01-01, 2024-01-01)` by entry time;
- sealed from: `2024-01-01T00:00:00Z`.

Before any comparator row or market outcome is opened, the accepted primary
clock must satisfy all of:

- train total at least 40;
- each train year at least 16;
- each train half-year at least 7;
- train at least 10 LONG and 10 SHORT;
- selection total at least 18;
- each selection half-year at least 6;
- selection at least 4 LONG and 4 SHORT;
- every train and selection quarter active;
- maximum UTC-month share at most 20% in train and 25% in selection;
- maximum accepted-entry gap at most 120 elapsed days;
- each precursor polarity at least 20% of train and 15% of selection events;
- confirmation ages `1–3`, `4–6`, and `7–10` each represented by at least one
  train event, preventing one exact lag from defining the whole mechanism;
- at most 85% of train or selection confirmations from one rate term bucket as
  measured by the larger absolute weighted contribution
  `|AR_LE30*TV_LE30|` versus `|AR_G30*TV_G30|`;
- exact timing, uniqueness, split containment, global non-overlap, and state
  chronology;
- no accepted row with a missing/non-finite field, nonpositive denominator,
  equal precursor/confirmation date, or invalid confirmation age; and
- zero post-2023 source rows.

Any failure rejects DMSH-168 before novelty and outcomes. Observed incidence may
not change 252, ±0.50, ten rows, source fields, state priority, support floors,
direction, 168-hour hold, or execution.

## Frozen controls and falsifications

Scheduled source controls 1–8 use the same source clock, exact arithmetic, split
containment, entry latency, 168-hour hold, and global non-overlap unless their
stated temporal object differs:

1. `flow_transition_only`: every eligible flow transition, side `SHORT` for
   `+1` and `LONG` for `-1`;
2. `curve_transition_only`: every eligible curve transition with the same side
   mapping;
3. `same_date_conjunction`: flow and curve enter the same polarity on one row;
4. `reverse_order_handoff`: curve transition is precursor and the strictly later
   same-polarity flow transition is confirmation;
5. `five_date_window`: exact primary with a five-row confirmation window;
6. `twenty_date_window`: exact primary with a twenty-row confirmation window;
7. `one_complete_date_stale`: both state sequences shifted one complete decision
   row old at current availability;
8. `five_complete_date_stale`: the same with five complete rows;
Noncausal source-placebo controls 9–10 deliberately destroy chronology and can
never emit an execution clock, enter novelty, open a market outcome, receive an
economic metric, or participate in a superiority gate:

9. `year_curve_permutation_placebo`: within each observation year, sort source rows by
   `(SHA256("DMSH-168|year_curve_permutation|<year>|<observation_date>"),
   observation_date)` ascending, read curve-gap values in that order, sort that
   year's destination rows by `observation_date` ascending, zip the value vector
   to those destinations, and recompute ranks/states solely to report null event
   incidence; destination timestamps are labels, not causal availability; and
10. `year_flow_permutation_placebo`: apply the identical source-sort/destination-zip rule
    with seed prefix `DMSH-168|year_flow_permutation`, permuting maturity-flow-gap
    values while preserving destination curve gaps, solely to report null event
    incidence.

The artifact must label every placebo row `causal=false` and
`economic_evaluation_forbidden=true`. Treating either placebo as causal is a
hard validation failure, not a weak-control result.

Economic side controls reuse exact accepted primary entries and exits:

- exact direction flip;
- deterministic random side from the first byte of
  UTF-8 `SHA256("DMSH-168|deterministic_random_side|<entry_time_utc_iso>")`,
  where the entry is rendered exactly as UTC `YYYY-MM-DDTHH:MM:SSZ`; first byte
  below `128` is LONG (`+1`), otherwise SHORT (`-1`);
- constant LONG; and
- constant SHORT.

No control may replace the primary. The exact economic superiority rule is
frozen below: in both train and selection, primary base CAGR/strict-MDD must
exceed every finite scheduled source-control value among controls 1–8 by at
least `0.25`, primary base mean
gross underlying return must exceed each of `flow_transition_only`,
`curve_transition_only`, `same_date_conjunction`, and `reverse_order_handoff`
by at least `5 bp`, and none of those four direct controls may independently pass
all primary economic gates. Failure rejects the ordered maturity-handoff claim.

## Frozen novelty gate

Only a complete source-support pass may open comparator rows. Before any DMSH
feature or incidence is computed, the preregistration artifact must hash-bind
the candidate-independent common-window policy
`docs/novelty-comparator-common-window-policy-2026-07-23.md` at SHA-256
`928bce6e04fb34001478b4b4ea84156580b661c88a0f0338065a891c009bd580`
and this exact cohort:

| Comparator artifact | SHA-256 | Required groups |
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
| `results/ofr_repo_venue_fragmentation_consensus_clocks_2026-07-23.csv.gz` | `b2d251b147feecc98def94610834c7b11f078f35e580f6e7a950fc55bfe0724e` | `control == primary` |
| `results/ofr_repo_mix_shock_resolution_race_clocks_2026-07-23.csv.gz` | `bed2f522bd68fd389cb0af3d655980999324c0e6adb4f1dc027c6b6ea2cc5ae6` | `control == primary` |
| `results/ofr_repo_collateral_routing_efficiency_clocks_2026-07-23.csv.gz` | `cbe4e5f6fc52b66062abbf931e46ea4aa0d1f3c0157ffd365d0638aa573c2826` | `control == primary` |

Over common 2021–2023 coverage, every required comparator group with at least
ten entries must satisfy:

- exact-entry Jaccard at most 0.10;
- one-to-one DMSH containment within ±24 elapsed hours at most 0.35; and
- absolute signed occupied-exposure correlation at most 0.35.

Missing, hash-mismatched, malformed, empty-required, overlapping, or
common-window-invalid comparator input fails closed. DMSH's own controls are
mechanism-specificity controls and are not novelty comparators.

## Strict economic sequence if source and novelty pass

The economic contract is frozen now, but no outcome parser or simulation is
authorized by this decision. A separate hash-bound strict evaluator may be
implemented only as a mechanical realization of this contract after source and
novelty pass, and it must be committed before any BTC market or funding row is
read. It may not add, remove, or reinterpret a gate. The sequence is:

1. train 2021–2022 only;
2. selection 2023 only after exact train pass;
3. immutable OFR source extension and test 2024 only after the pre-2024 pass;
4. eval 2025 only after test pass; and
5. recent 2026 only after eval pass.

Accounting is fixed:

- initial equity `1.0`, fixed `0.5x` notional exposure, quantity fixed at
  `entry_equity * 0.5 / entry_open`;
- numeric side is `+1` for LONG and `-1` for SHORT; confirmed precursor polarity
  `p=+1` maps to side `-1`, while `p=-1` maps to side `+1`;
- entry and exit at the frozen five-minute opens;
- base cost `6 bp` of notional per side and stress cost `10 bp` per side, with
  exit cost based on exit notional;
- realized funding cash
  `-side * quantity * settlement_mark_price * funding_rate` for settlements in
  `[entry_time, exit_time]`; exact-boundary debits are retained and exact-boundary
  credits discarded and reported;
- idle cash counts throughout each declared full wall-clock interval;
- strict MDD carries the global/pre-entry high-water through idle periods and,
  for each position, orders entry cost, all favorable held five-minute OHLC and
  admissible funding credits, then all adverse held OHLC, funding debits, and a
  hypothetical adverse-price exit cost, followed by realized exit and exit cost;
- bankruptcy floors equity at zero and yields 100% strict MDD; and
- one-extra-bar delay shifts both entry and exit exactly five minutes, preserving
  side, hold length, and event count.

For each trade, gross underlying move in basis points is exactly
`side * (exit_open / entry_open - 1) * 10,000`, before leverage, costs, and
funding. Mean gross underlying move is its arithmetic mean over all contained
trades. LONG-only and SHORT-only contribution are the arithmetic sums of the
already-sized primary simulation's realized net equity changes, including entry
and exit costs and admissible funding, grouped by numeric side and divided by
initial equity `1.0`; both sums must be strictly positive. No side-specific
resizing or independent compounding is allowed.

The primary must independently satisfy in train and selection:

- positive full-calendar absolute return;
- full-calendar `CAGR / strict MDD >= 3.00`;
- strict MDD at most `15.00%`;
- positive 10-bp/side stress absolute return and stress CAGR/strict-MDD at least
  `2.50`;
- positive one-extra-bar-delay absolute return;
- mean gross underlying move at least `30 bp`;
- positive LONG-only and SHORT-only contributions;
- positive 2021 and 2022 train returns, and positive 2023 H1 and H2 selection
  returns;
- at least 20 nonempty UTC `W-SUN` entry-week clusters in train and 8 in
  selection; and
- UTC-week clustered one-sided sign-flip `p <= 0.10`. Group each trade's realized
  net equity change from the primary simulation by its entry week's UTC `W-SUN`
  period and sum within week. The observed statistic is the sum of all weekly
  cluster sums. For at most 18 nonempty clusters, enumerate all `2^k` sign vectors
  in binary integer order and set `p = count(T_null >= T_observed) / 2^k`. For
  more than 18 clusters, use exactly 100,000 draws indexed `0..99,999`; for each
  sorted week-start UTC `YYYY-MM-DDT00:00:00Z`, the sign is `+1` iff the first
  byte of UTF-8
  `SHA256("DMSH-168|weekly_signflip|20260723|<draw_index>|<week_start_iso>")`
  is below `128`, otherwise `-1`. Set Monte Carlo
  `p = (1 + count(T_null >= T_observed)) / 100001`. Comparisons use exact decimal
  equity deltas before final display rounding.

Base and stress runs must contain identical trades, and no ratio may qualify via
a zero-MDD cap. The source-control superiority and independent-pass rejection
rules frozen above are mandatory. Exact direction flip, deterministic random
side, constant LONG, and constant SHORT remain report-only diagnostics and can
never replace the primary. Later windows apply the same accounting and gates,
may veto, and may never rerank, repair, or select another DMSH variant.

## No-repair and live boundary

DMSH-168 is one singleton. The mechanism, source/support contract, comparator
cohort, and economic contract are frozen at this decision. The preregistration
and later evaluator may only bind hashes and implement these exact rules. After
any DMSH feature, incidence, comparator, or economic evidence is opened, none of
its source fields, arithmetic, thresholds, state priority, direction, latency,
confirmation window, hold, cost, split, support gate, comparator cohort,
superiority margin, significance rule, or control may change. A failed stage is
terminal under this identity.

No result from this candidate authorizes testnet, shadow, or live orders. Live
use would additionally require immutable source extension, historical/live
feature parity, release-latency monitoring, complete trade-lifecycle parity,
forward shadow evidence, and a separately committed deployment decision.
