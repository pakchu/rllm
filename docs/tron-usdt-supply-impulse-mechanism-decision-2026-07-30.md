# TUSI-168 — TRON USDt Supply Impulse mechanism decision

## Decision

Freeze one outcome-blind, source-new BTC candidate:

```text
TUSI-168 — TRON USDt Supply Impulse, seven-day hold
```

TUSI tests whether finalized primary-market USDt supply changes on TRON
transmit to BTC over the following week:

- positive net `Issue` supply in one causal execution bucket -> `LONG BTC`;
- negative net `Redeem` supply in one causal execution bucket -> `SHORT BTC`;
- zero net supply -> abstain.

The mapping is falsifiable, not an issuer claim. Issuance can be treasury
inventory, cross-chain migration, or administrative preparation rather than
immediate exchange buying power. Redemption can follow earlier customer flow
and need not create immediate BTC selling. `DestroyedBlackFunds` is
confiscation and is excluded from the primary.

This document is frozen before the complete TRON source replay, pre-cutoff
TUSI incidence, comparator rows, Gross9 reconstruction, BTC market/funding
rows, return, PnL, CAGR, or strict MDD.

## Source and policy separation

Canonical source semantics are frozen separately in
`docs/tron-usdt-supply-events-source-axis-decision-2026-07-30.md`.

TUSI may read only the promoted normalized source fields from that contract.
It may not read explorer labels, Tether customer identity, treasury balances,
exchange addresses, ordinary transfers, current total-supply snapshots,
TRON fees, BTC price, funding, premium, open interest, Gross9 state, or a prior
alpha state to construct its clock.

The source build and this policy remain separate artifacts. Source provenance
does not prove the policy, and policy failure does not invalidate correctly
replayed source rows.

## Exact causal bucket

For each canonical `Issue` or `Redeem` row, take its exact
`available_at_utc`, which is the timestamp of event block `N+64`, and define:

```text
candidate_entry_open = ceil_to_5m(available_at_utc) + 5 elapsed minutes
```

If availability is already on a five-minute boundary, the entry is still five
minutes later.

Group all eligible semantic rows with the same `candidate_entry_open`.
Within a group:

```text
net_supply_raw =
    sum(Issue.amount_raw) - sum(Redeem.amount_raw)
```

- `net_supply_raw > 0` -> `LONG`;
- `net_supply_raw < 0` -> `SHORT`;
- `net_supply_raw == 0` -> no candidate.

Amounts are exact six-decimal integers. There is no logarithm, normalization,
rank, quantile, clipping, winsorization, minimum amount, rolling lookback,
event-onset search, side search, or provider timestamp.

The source identity is the SHA-256 of canonical constituent bytes. Sort
constituents lexicographically by
`(block_number, transaction_index, log_index, transaction_hash, event_type,
amount_raw)`, preserving numbers as JSON integers, the lowercase `0x` hash and
the exact event string. Encode the resulting list of six-element arrays as
UTF-8 compact JSON with sorted object keys, separators `(',', ':')`, ASCII
escaping, finite values only, and no newline. It breaks no ties and cannot
encode an outcome.

## Execution

For each nonzero causal bucket:

```text
decision_time = maximum constituent available_at_utc
entry_time    = candidate_entry_open
exit_time     = entry_time + 168 elapsed hours
```

- fixed standalone exposure: `0.5x` account notional;
- one global position in the single frozen accepted clock;
- reservation interval: `[entry_time, exit_time)`;
- raw candidates sort by
  `(entry_time, decision_time, source_identity, side)`;
- accept only when `entry_time >= previous accepted exit_time`;
- suppressed candidates are not queued;
- no pyramiding, stop, take-profit, trailing exit, or early close;
- containment and scheduling follow the exact construction below.

The seven-day hold is fixed before incidence. It represents a slow
cross-market liquidity-transmission hypothesis and is not selected from a
hold grid.

## Frozen calendars

All periods are half-open and require both entry and exit containment:

```text
full       [2023-06-01, 2026-06-01)
selection  [2023-06-01, 2025-01-01)
future25   [2025-01-01, 2026-01-01)
future26   [2026-01-01, 2026-06-01)
```

Selection diagnostics are:

```text
2023H2 [2023-06-01, 2024-01-01)
2024H1 [2024-01-01, 2024-07-01)
2024H2 [2024-07-01, 2025-01-01)
```

Future25 diagnostics also split at `2025-07-01`. The complete full-calendar
CAGR uses the exact three-year wall-clock interval, including idle cash.

The scheduler is run exactly once per independent construction, not once per
report:

1. build every raw candidate from the complete eligible source;
2. assign it to exactly one of the three disjoint main windows in order
   `selection`, `future25`, `future26` only if both entry and exit are
   contained; discard a main-boundary crosser before reservation;
3. sort the union of assigned rows by the frozen raw-candidate order and run
   one global nonoverlap reservation;
4. define the `full` accepted clock as that accepted union;
5. derive `2023H2`, calendar `2024`, `2024H1`, `2024H2`, `2025H1`, and
   `2025H2` only by projecting the accepted clock to rows whose entry and exit
   are both contained. Never rerun nonoverlap for a diagnostic or economic
   period.

Thus a selection trade crossing a diagnostic boundary remains part of the
selection reservation clock but is skipped from that diagnostic report; it is
never truncated or replaced.

## Source-support gates

Before any comparator or outcome row is opened, the committed evaluator must
require:

```text
dual raw-log replay differences                 = 0
chunk gaps, overlaps, missing response IDs       = 0
receipt/header differences                       = 0
Issue/mint-transfer pair differences             = 0
Redeem/burn-transfer pair differences            = 0
Deprecate events                                  = 0
future-append differences in selection           = 0

selection accepted trades                        >= 8
2023H2 accepted trades                           >= 2
2024H1 accepted trades                           >= 2
2024H2 accepted trades                           >= 2
future25 accepted trades                         >= 4
2025H1 accepted trades                           >= 1
2025H2 accepted trades                           >= 1
future26 accepted trades                         >= 2
maximum UTC-month share in each main period      <= 0.50
maximum accepted-entry gap inside full           <= 240 days
```

The three main periods for the month-share gate are exactly `selection`,
`future25`, and `future26`. For each one, the numerator is the largest count
of accepted primary entries sharing the same UTC `entry_time` calendar month;
the denominator is all accepted primary entries in that same period. The
comparison is the exact inclusive rational `largest_month_count / period_count
<= 1/2`; a zero denominator fails rather than becoming zero.

The gap gate sorts accepted primary entries contained in `full` and takes only
differences between consecutive entry timestamps. Boundary-to-first and
last-to-boundary gaps are excluded. Fewer than two full-period accepted
entries fails; otherwise the exact maximum difference must be at most
`240 * 86,400` seconds.

Future-append invariance performs two fresh constructions. The prefix
construction uses only source rows with
`available_at_utc < 2025-01-01T00:00:00Z`; the full construction uses the
complete source artifact. For primary and every independent control, compare
canonical compact-JSON rows of both (a) every raw candidate assigned to
`selection` and (b) every accepted `selection` clock row. Compared fields are
control, sorted constituent identities, source identity, constituent count,
exact signed bucket amount or count, decision time, entry, exit, and side.
Same-primary-parent controls are then regenerated and compared from the two
accepted primary views. Any row, order, field, or SHA-256 difference fails.

LONG and SHORT counts are reported but not separately floored. A genuinely
one-sided realized source is allowed to fail economically rather than being
repaired by manufacturing a side. Every identity, bucket membership, integer
net amount, side, entry, exit, and source hash must be unique and reproducible.

These are rare-event support floors, not economic evidence. Any failed source,
incidence, dispersion, append, or containment gate retires TUSI-168 without
changing a source, event type, grouping rule, side, latency, hold, support
floor, or calendar.

## Frozen controls

Controls cannot replace a failed primary. The following are independent
rebuilds: they construct their own exact causal buckets, sort their own raw
candidates, and run their own global nonoverlap scheduler:

1. `issue_only`: use only `Issue`, always `LONG`;
2. `redeem_only`: use only `Redeem`, always `SHORT`;
3. `include_destroyed_black_funds`: add
   `DestroyedBlackFunds.amount_raw` as negative supply solely as a
   contamination diagnostic;
4. `count_net_side`: use the sign of `Issue` row count minus `Redeem` row
   count in each exact bucket, ignoring amounts.

The following controls inherit the exact accepted primary parent set and do
not regroup source rows or rerun nonoverlap:

5. `exact_direction_flip`: reverse every primary side;
6. `deterministic_random_side`: SHA-256 encode the UTF-8 string
   `source_identity|TUSI-168|RANDOM_SIDE` and choose `LONG` iff the first byte
   is below 128;
7. `constant_long` and `constant_short`: replace every primary side;
8. `one_bar_delayed_entry`: shift both entry and scheduled exit exactly 300
   seconds while retaining the primary identity and parent membership.

The delayed control keeps every shifted parent row in its support artifact and
retains the parent's main-window label only as provenance. Every diagnostic
and economic projection rechecks both shifted timestamps against its own
half-open period. A shifted boundary crosser is dropped before any market or
funding lookup, and its count plus source identity is reported separately; it
cannot suppress, replace, or cause a reschedule of another parent.

The support stage reports control incidence and exact overlap without price.
Economic superiority is tested only after novelty passes.

## Downstream novelty

Only a complete source-support pass may open comparators. The exhaustive prior
registry is imported without deletion or reinterpretation from:

```text
path
  results/ethereum_settlement_demand_impulse_preregistration_2026-07-30.json
file SHA-256
  2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba
ESDI manifest_hash
  d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a
JSON subtree
  novelty.frozen_comparator_artifacts
subtree item count
  18
canonical sorted compact-JSON subtree SHA-256
  0d13c9de1e098446aaaa78b9a24c7d05c7ec375df05d79c9f8969792546bd4a3
```

Every recorded path, file hash, header hash, filter, group, capability,
required column, comparison domain, and metric requirement in that subtree is
immutable. Each group marked as a separate comparator is evaluated
separately. Every positive-weight Gross9 sleeve is an additional comparator.

For each Gross9 sleeve:

```text
exact-entry Jaccard                  <= 0.10
candidate +/-6h containment          <= 0.35
occupied-bar Jaccard                 <= 0.25
absolute signed-exposure Pearson     <= 0.35
```

For each prior source-family clock with at least ten common-domain entries:

```text
exact-entry Jaccard                  <= 0.20
candidate +/-24h containment         <= 0.50
absolute signed-exposure Pearson     <= 0.40
```

The prior inventory explicitly includes Ethereum stablecoin, authorized
minter, USDC, WBTC, blockspace/network, quote-flow, denominator, collateral,
and custody-liquidity clocks. Missing, malformed, hash-drifted, or
capability-incompatible comparators fail closed. No comparator may be removed
after overlap is seen.

All entries are unique strictly increasing integer UTC seconds after both
clocks are filtered to the registry's identical half-open common domain.
Metrics and undefined behavior are exact:

- entry Jaccard is `|A intersect B| / |A union B|`;
- containment is the maximum of the two directional fractions having an
  opposite entry at inclusive distance at most the frozen window;
- signed exposure is exact `{-1,0,1}` on every 300-second bar open, from
  sorted, contained, nonoverlapping `[entry, exit)` intervals;
- occupied Jaccard uses indexes with nonzero signed exposure;
- signed Pearson is evaluated as the exact squared rational
  `covariance_numerator^2 / (left_variance_numerator *
  right_variance_numerator)`;
- every inclusive threshold is compared by integer cross multiplication;
- duplicate/unsorted timestamps, empty metric denominators, unequal vectors,
  or zero Pearson variance are terminal, never zero-filled;
- the ten-entry prior-family minimum is applied after common-domain filtering;
  below ten is reported as not gated, never silently discarded.

## Strict economics

The evaluator must be implemented, tested, committed, and hash-bound before
opening any BTC market or funding row.

Common accounting:

- standalone leverage `0.5x`;
- base cost `6 bp` of notional per side;
- stress cost `10 bp` per side;
- exact realized funding for
  `entry_time <= funding_time < exit_time`;
- next-open entry and exact scheduled-open exit;
- full-calendar CAGR;
- global/pre-entry high-water strict MDD;
- intratrade favorable and adverse OHLC plus funding in chronological
  high-water/trough order.

The accounting code authority, including the named
`calendar_month_clustered_signflip` helper, is
`training/evaluate_ethereum_settlement_demand_impulse_economics.py` at
SHA-256
`fba7de6a26ede945edfe63c32dd4a0c88760c6459ac0d4f079dd12d546580235`.
TUSI imports its strict pure accounting helpers; the replay claim binds this
file and any hash drift is terminal.

For a sleeve weight `w`, entry open `O`, post-exit equity `E`, and side
`s in {-1,+1}`:

```text
allocated_equity = E * w
quantity         = allocated_equity * 0.5 / O
entry_cost       = abs(quantity * O) * cost_rate
funding_cash     = -s * quantity * funding_rate * settlement_mark
exit_price_cash  = s * quantity * (exit_open - O)
exit_cost        = abs(quantity * exit_open) * cost_rate
```

Entry cost is removed from cash once at entry. At exit, price cash minus exit
cost is added once and the position is removed. Funding cash is applied only
for `entry_time <= funding_time < exit_time`. Marked equity at a bar open is
cash plus `sum(s * quantity * (open - entry_open))`.

On each full-calendar five-minute bar, the frozen strict path order is:

1. mark pre-cost equity at the bar open against the global HWM;
2. execute scheduled exits at that open and charge exit cost, then execute
   entries at the same open and charge entry cost;
3. apply realized funding credits and debits whose timestamp equals the bar;
4. update HWM with the greater of pre-cost open equity and post-entry-cost
   equity plus funding credits plus favorable long-high/short-low excursion;
5. update the trough from the pre-cost open-equity baseline, subtracting each
   aggregate entry/exit cost exactly once, with funding credits, funding
   debits, adverse long-low/short-high excursion, and hypothetical liquidation
   cost at the adverse mark.

With `E_pre` the pre-event open mark, aggregate entry and exit costs `C_in` and
`C_out`, funding credit `F+ >= 0`, funding debit `F- <= 0`, favorable excursion
`A+ >= 0`, adverse excursion `A- <= 0`, gross open quantity `Q`, and adverse
mark `P_bad`:

```text
upper_t = max(E_pre, E_pre - C_in + F+ + A+)
HWM_t   = max(HWM_(t-1), upper_t)
lower_t = min(E_pre,
              E_pre - C_in - C_out + F+ + F- + A-
                    - Q * P_bad * cost_rate)
MDD     = max_t(1 - lower_t / HWM_t)
```

`E_pre` already marks an exiting position at the exit open, so `C_out` in the
lower envelope is the same single exit-cost event charged to cash, not a
second charge. Favorable/adverse excursions use the net signed quantity after
that bar's exits and entries; liquidation cost uses gross quantity.

The HWM begins at `1.0` at the calendar start and therefore includes idle
pre-entry time and every earlier sleeve/trade. No per-trade HWM reset,
close-only drawdown, truncated calendar, or favorable/adverse reordering is
allowed. A nonpositive liquidation-envelope equity is terminal.

For period `[start,end)`, calendar years are exactly `3.0` for `full` and
otherwise `(end-start).total_seconds() / (365.25 * 86,400)`.
`absolute_return = final_equity - 1` and
`CAGR = exp(log(final_equity)/years) - 1`. If CAGR is positive,
`CAGR/strict_MDD = CAGR / max(MDD, 1e-15)`; otherwise it is zero. The mean
gross underlying move is the arithmetic mean of
`side * (exit_open / entry_open - 1) * 10,000` over contained trades.

For the calendar-month clustered sign-flip test, invoke that named helper on
trade records in ascending `(entry_time, exit_time, source_identity)` order.
It accumulates Python `float` net return on allocated equity by UTC entry
month, then forms a NumPy `float64` vector in ascending `YYYY-MM` order,
discarding only clusters with absolute sum at most `1e-15`. If the observed
left-to-right vector sum is nonpositive, `p=1`. With at most 20 nonzero months,
enumerate `itertools.product((-1.0, 1.0), repeat=m)` in its native order and
count `np.dot(np.asarray(signs), ordered) >= observed - 1e-15`.

Above 20 months, construct `rng = np.random.default_rng(20260730)` and consume
exactly 10,000 rows in batches `4096`, `4096`, and `1808` using
`rng.integers(0, 2, size=(batch, m), dtype=np.int8)`. Convert each batch with
`signs.astype(np.float64) * 2.0 - 1.0`, evaluate row-major
`signs @ ordered`, count values at least `observed - 1e-15`, and report
`p=(exceed+1)/10,001`.

Every opened standalone period must pass under base and stress:

```text
absolute return                         > 0
full-calendar CAGR / strict MDD          >= 3.0
strict MDD                              <= 0.15
mean gross underlying move              >= 20 bp
calendar-month clustered sign-flip p    <= 0.10
```

For each opened standalone period and for both base and stress costs, the
primary's exact `full_calendar_CAGR / strict_MDD` must be strictly greater than
each independent-rebuild control having at least one contained accepted trade:
`issue_only`, `redeem_only`, `include_destroyed_black_funds`, and
`count_net_side`. A zero-trade control is reported and not superiority-gated;
an undefined nonzero-support metric is terminal. The primary still must pass
all absolute standalone gates independently.

A same-primary-parent control `exact_direction_flip`,
`deterministic_random_side`, `constant_long`, or `constant_short` completely
qualifies only if it passes all five standalone gates under both base and
stress in every standalone period opened before the stop. None may completely
qualify. `one_bar_delayed_entry` is reported as a timing sensitivity and
cannot replace or select the primary.

Open periods in order and stop permanently at the first failure:

1. `2023H2`;
2. calendar `2024`;
3. combined selection;
4. same-gross Gross9 selection and frozen candidate weight;
5. `future25`;
6. `future26`;
7. exact stitched full report.

Later periods only veto. They cannot invert, rerank, repair, or select rank
two.

## Same-configured-gross Gross9

The authoritative portfolio remains:

```text
cand_rex_veto_7                 1.6
fresh_kimchi_fx                 2.0
frozen_annual_rank7             3.0
markov_transition_long          2.0
rex_taker_low_range_position    0.4
gross                           9.0
```

Its complete authority is frozen by the same ESDI artifact above:

```text
gross9 subtree canonical SHA-256
  d79c79789ed48c7c2a94bac4474583798c2306bd320abb2617c354878c3578fe
gross9.authority subtree canonical SHA-256
  b3490c484d3fda1d5b649498e0d84325e203cd2664086e68cebd76509a54957e
runtime_code_closure subtree canonical SHA-256
  ffffb68c0900836ba06b573398c4825bd9d15161a9e36818aeb68fc33a86d84a
```

The exact base and shadow portfolio configs, five sleeve configs/model bundle,
transitive source manifest, three runtime roots, complete AST import closure,
`pyproject.toml`, `uv.lock`, Python/platform ABI, and all-distribution
inventory recorded in that authority must hash-match before clock
reconstruction. Missing or changed closure/environment evidence is terminal;
no cached substitute clock is allowed.

Candidate weights are frozen at `[0.25, 0.50, 0.75, 1.00]`.
At candidate weight `w`, scale every Gross9 sleeve by `(9-w)/9` and add TUSI
at weight `w`. Treatment and the unscaled Gross9 baseline therefore both have
configured gross exactly `9.0`.

For every candidate weight, both `2023H2` and 2024 and both cost settings must:

- improve base and stress CAGR/strict-MDD by at least `0.05` versus unscaled
  same-gross Gross9;
- retain at least `97%` of Gross9 absolute return;
- have positive base and stress return; and
- reduce strict MDD in at least one of the four period/cost cells.

Rank by the minimum base/stress improvement across the two periods and
tie-break by lower weight. Freeze rank one. For each of `future25` and
`future26` independently, use only that frozen weight and, under both base and
stress, require CAGR/strict-MDD improvement at least `0.05`, at least `97%`
baseline absolute-return retention, positive treatment return, and
liquidation safety. Treatment strict MDD must be lower than baseline in at
least one of the two cost settings for that future period. No future weight
grid, rerank, or alternate is opened.

## Write-once artifacts and hashes

Exact paths are:

```text
preregistration
  results/tron_usdt_supply_impulse_preregistration_2026-07-30.json
source replay claim
  results/tron_usdt_supply_events_source_replay_claim_2026-07-30.json
primary support clock
  results/tron_usdt_supply_impulse_primary_clock_2026-07-30.csv.gz
control support clocks
  results/tron_usdt_supply_impulse_control_clocks_2026-07-30.csv.gz
source-support report
  results/tron_usdt_supply_impulse_source_support_2026-07-30.json
```

Canonical compact JSON is the definition in the source decision. The
preregistration and each JSON report are sorted-key, two-space-indented ASCII
JSON plus one LF; each internal `manifest_hash` is SHA-256 of compact
sorted-key JSON with that field excluded. Every gzip CSV uses UTF-8 LF records,
the frozen column order, compression level 9, empty filename, and `mtime=0`.

Both support-clock CSVs have exactly this header order:

```text
policy_id
control
window
constituent_identities_json
source_identity
constituent_count
bucket_amount_raw
decision_time_utc
entry_time_utc
exit_time_utc
side
```

`policy_id` is `TUSI-168`; `control` is one frozen control string; `window` is
one main-window string; `constituent_identities_json` is the exact compact JSON
used for source identity; `source_identity` is 64 lowercase hex characters;
counts and signed amount/count are canonical base-10 integers; times are whole
second `YYYY-MM-DDTHH:MM:SSZ`; and side is `LONG` or `SHORT`. No null or extra
column is allowed. The primary file contains only `control=primary`. The
control file concatenates controls in the frozen order listed above excluding
primary, with each control internally sorted by the raw-candidate order.
Future-append views use these same semantic fields plus an explicit
`accepted` view label; they are report objects, not a third CSV schema.

The preregistration JSON has exactly these top-level keys and types:

```text
protocol_version:str, policy_id:str, status:str, singleton:bool,
frozen_preregistration:object, source:object, feature_and_signal:object,
execution:object, calendars:object, support_gates:object, controls:object,
novelty:object, economic_contract:object, gross9:object,
strict_sequence:array[str], producer_effects:object,
precutoff_source_rows_opened:bool, source_incidence_opened:bool,
candidate_incidence_opened:bool, comparator_rows_opened:bool,
gross9_rows_opened:bool, btc_market_rows_opened:bool,
funding_rows_opened:bool, returns_opened:bool, pnl_opened:bool,
cagr_opened:bool, strict_mdd_opened:bool, outcomes_opened:bool,
manifest_hash:str
```

The nested objects are exactly the versioned fields specified in the preceding
contract sections and emitted by the committed producer; unknown or missing
keys fail. Boundary arrays are chronological, transports are
`[primary, verification]`, controls use the frozen order, strict sequence uses
the displayed order, and closure paths use the ESDI authority order. JSON
object insertion order has no semantics because byte serialization sorts keys;
all arrays retain their prescribed order.

The source-support report has exactly these top-level keys and types:

```text
protocol_version:str, policy_id:str, status:str, terminal:bool,
artifact_eligible:bool, support_passed:bool, decision:str,
registration:object, source_contract:object,
raw_candidate_counts:object[int], accepted_clock_counts:object[int],
period_diagnostics:object, support_audit:object,
support_checks:object[bool], future_append_selection_invariance:object,
control_overlap:object, clock_artifacts:object[str],
evidence_boundary:object, source_support_precedes_novelty:bool,
novelty_comparator_market_or_outcome_artifacts_opened:bool,
manifest_hash:str
```

Count objects have exactly the frozen control keys; period diagnostics have
exactly `selection`, `2023H2`, `2024`, `2024H1`, `2024H2`, `future25`,
`2025H1`, `2025H2`, `future26`, and `full`; `clock_artifacts` has exactly
`primary_sha256` and `controls_sha256`. Future-append contains the prefix rule,
total difference count, and, in frozen control order, raw/accepted row counts
and SHA-256 pairs. Evidence-boundary keys are booleans or integer opened-row
counts and unknown/missing nested keys fail.

The preregistration binds only its two decisions, producer/test, the exact
ESDI preregistration/metric authority, and the complete Gross9 closure and
environment. It lists later TUSI protocol files as expected metadata but does
not hash not-yet-final evaluators. After its artifact is committed, every
later evaluator hard-codes that artifact's file SHA-256. The separately
committed replay claim then binds the preregistration artifact plus all source,
support, novelty, economics, imported helper, and test blobs. This ordering
removes self-reference and future-file circularity.

## Stopping rule

The only admissible sequence is:

1. commit the source-axis and mechanism decisions;
2. commit a write-once preregistration producer and tests;
3. create and commit its write-once artifact;
4. commit the source builder, source-support evaluator, novelty evaluator,
   strict economics, and synthetic tests bound to that artifact;
5. commit a source-replay claim before the first pre-cutoff event/log/receipt
   replay RPC and before any source incidence is opened;
6. execute the complete source replay once;
7. commit the write-once source CSV and manifest without changing any
   evaluator;
8. run source support, novelty, and economics strictly in order, stopping at
   the first failed gate.

No observed source incidence, comparator overlap, or economic result may
change TUSI-168. Once the source replay claim is committed, evaluator,
threshold, dependency, environment, and parsing bytes are immutable; a replay
or gate failure cannot be repaired under this identity. A failure permanently
retires TUSI-168.
