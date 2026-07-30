# CRSB-336 mechanism decision — 2026-07-30

## Decision

Freeze exactly one source-unseen, outcome-unseen candidate:

**CRSB-336 — Circle Reserve Schema Bridge, 336-hour hold.**

CRSB-336 combines four equal, discrete votes from one monthly Circle Reserve
Fund N-MFP filing:

1. where daily liquidity finishes inside its disclosed monthly range;
2. where weekly liquidity finishes inside its disclosed monthly range;
3. whether WAM compresses or extends from the prior filing; and
4. whether WAL compresses or extends from the prior filing.

A positive majority is `LONG`; a negative majority is `SHORT`; a tie emits no
signal. Entry waits one complete five-minute bar after the conservative SEC
source-availability clock. The hold is 336 elapsed hours at 0.5x standalone
gross exposure.

This file freezes the complete policy, support, novelty, accounting, Gross9,
selection, future-veto, and no-repair contract before the production SEC daily
indexes, Feed archives, Circle liquidity rows, candidate incidence,
comparator rows, Gross9 rows, BTC bars, funding, returns, PnL, CAGR, or strict
MDD are opened.

## Mechanism

Circle states that the majority of USDC reserves is invested in the Circle
Reserve Fund and that the reserve is intended to remain highly liquid and
redeemable. Form N-MFP requires the fund to disclose daily and weekly liquid
assets plus WAM and WAL.

The falsifiable mechanism is:

- ending a reporting month near the high of both disclosed liquidity paths,
  together with shorter reset and legal-life maturity, indicates stronger
  immediately available redemption capacity and votes `LONG` BTC;
- ending near the low of those paths, together with duration extension,
  indicates weaker immediate reserve flexibility and votes `SHORT` BTC; and
- disagreement is resolved only by the equal vote, not by a fitted weight,
  magnitude threshold, market regime, or later outcome.

This is not a claim by Circle, BlackRock, or the SEC. It is a fixed research
hypothesis. CRSB does not use net assets, shareholder flows, USDC supply,
holdings, yields, NAV, or any on-chain mint/redeem event.

## Frozen source authority

The source contract is:

- `docs/circle-reserve-schema-bridge-source-axis-decision-2026-07-30.md`;
- source-axis commit `48672fd58e1da01146a0f31e62bdd6f3736f3944`;
- source-axis file SHA-256
  `e8072718ecd7cc928e8feac51d3afae53517c0669cf4ea086281c0b86b2694e0`;
- source-axis Git blob `9286274301ae021cd97565e4164b8b8fd3d46563`;
- source identity `CRF-NMFP-SB`;
- expected original report months `2022-11` through `2026-04`;
- first-dissemination SEC daily-index and Feed bytes;
- exact current-archive parity; and
- conservative availability at `12:00:00Z` on the fifth calendar day after
  the EDGAR acceptance local date.

The later preregistration must bind the complete source decision blob and every
source/parser/evaluator/test blob by Git object ID and SHA-256 before source
access.

## Exact source row

One valid original filing yields:

```text
accession
form
report_date
acceptance_datetime_et
source_available_at_utc
registrant_cik
registrant_lei
series_id
schema_path_kind
wam_days
wal_days
liquidity_path_json
```

`liquidity_path_json` is canonical compact JSON:

```json
[
  {
    "position": 1,
    "source_label": "friday1 or YYYY-MM-DD",
    "daily_pct": "plain exact decimal",
    "weekly_pct": "plain exact decimal"
  }
]
```

`schema_path_kind` is exactly `nmfp2_friday_slots` or
`nmfp3_dated_details`. N-MFP2 paths contain exactly four or five observations
whose labels are `friday1` through `friday4|5`; N-MFP3 paths contain exactly
15–31 observations whose labels are their exact source dates. Position is
one-based and contiguous in retained source order. The mechanism never
fabricates N-MFP2 dates, resamples either path, or equalizes path lengths.

Rows are ordered by `(source_available_at_utc, accession)`. Report months,
report dates, source availability, and accessions must be strictly unique.
Every row after the first must be the immediately following calendar report
month. Source validation happens before any vote is computed.

## Exact source and clock arithmetic

All source decimals are parsed to exact base-10 rational numbers. Binary
floating point is forbidden for source parsing, votes, comparisons,
scheduling, support, overlap, or novelty-gate decisions. The separately
authenticated ESDI economic accounting intentionally uses its frozen
NumPy/Python `float64` implementation and exact tolerances as specified below.

For a path `P=(p_1,...,p_n)`:

```text
lo(P)      = min_i p_i
hi(P)      = max_i p_i
end(P)     = p_n
balance(P) = 2*end(P) - lo(P) - hi(P)
```

The path vote is:

```text
+1 if balance(P) > 0
-1 if balance(P) < 0
 0 if balance(P) = 0
```

A constant path therefore votes zero. No epsilon, rounding, rank, clip,
normalization, or tolerance is allowed.

The identical path formula, sign interpretation, component weight, tie rule,
and side projection apply to both `schema_path_kind` values. Form family,
path length, source label kind, and the N-MFP2-to-N-MFP3 transition cannot
flip, scale, suppress, or otherwise alter a vote. The first N-MFP3 report
compares WAM and WAL with the immediately prior valid N-MFP2 report; there is
no transition reset.

For report `t>0`, maturity votes compare against the immediately prior report:

```text
wam_vote_t =
  +1 if WAM_t < WAM_(t-1)
  -1 if WAM_t > WAM_(t-1)
   0 otherwise

wal_vote_t =
  +1 if WAL_t < WAL_(t-1)
  -1 if WAL_t > WAL_(t-1)
   0 otherwise
```

The first source report is warm-up and cannot emit any clock.

## Primary vote and side

For each later report:

```text
daily_vote   = vote(daily_pct path)
weekly_vote  = vote(weekly_pct path)
wam_vote     = prior-to-current WAM compression vote
wal_vote     = prior-to-current WAL compression vote
vote_sum     = daily_vote + weekly_vote + wam_vote + wal_vote
```

The primary raw side is:

```text
LONG  if vote_sum > 0
SHORT if vote_sum < 0
NONE  if vote_sum = 0
```

Vote magnitudes are exactly one. A two-two split is no signal. A zero component
does not transfer its vote to another component. A later missing or invalid
source month is impossible after source support; if encountered, it is
terminal rather than a segment reset.

Every timestamp serialized into an identity is exactly 20 ASCII bytes in
whole-second UTC grammar `YYYY-MM-DDTHH:MM:SSZ`; offsets, fractional seconds,
spaces, lower-case `z`, and alternate ISO renderings are forbidden.
`accession` is exactly the SEC dashed accession grammar
`##########-##-######`. `vote_sum` is canonical base-10 ASCII with one leading
minus only when negative, no leading plus, and no leading zero.

The primary source identity is the exact UTF-8 string:

```text
CRSB-336|primary|<accession>|<source_available_at_utc>|<vote_sum>
```

The signal ID is lowercase hexadecimal SHA-256 of that exact UTF-8 string.

## Scheduling and overlap

For every raw signal:

```text
decision_time = source_available_at_utc
entry_time    = decision_time + 5 elapsed minutes
exit_time     = entry_time + 336 elapsed hours
```

The source clock is already five-minute aligned. No same-bar entry is allowed.
The reserved interval is `[entry_time, exit_time)`.

Raw signals are sorted by `(entry_time, signal_id)`. For each clock
independently, accept a signal only when `entry_time >= previous accepted
exit_time`; equality is allowed. An overlapping signal is suppressed without
queue, replacement, extension, side netting, or later release. Source vote
state still advances on every valid report.

No TP, SL, barrier, dynamic exit, funding gate, price gate, volatility gate,
calendar exception, or stale-bar exception exists.

## Frozen evaluation periods

Intervals are half-open UTC:

```text
2023H2:    [2023-06-01, 2024-01-01)
2024:      [2024-01-01, 2025-01-01)
selection: [2023-06-01, 2025-01-01)
future25:  [2025-01-01, 2026-01-01)
future26:  [2026-01-01, 2026-06-01)
full:      [2023-06-01, 2026-06-01)
```

Global overlap suppression is applied before period filtering. A trade belongs
to a period only when decision, entry, and exit are fully contained. Crossing
trades are excluded whole, never clipped. The full-calendar CAGR denominator
is the complete declared interval, including all idle cash before the first
entry and between trades.

## Source-derived attribution controls

Every control uses the same source rows, causal clock, signal-ID grammar,
five-minute latency, 336-hour hold, non-overlap, and period containment.

Single-component controls:

1. `daily_path_only`;
2. `weekly_path_only`;
3. `wam_change_only`;
4. `wal_change_only`.

Each emits its component's sign when nonzero.

Pair controls:

5. `path_pair` uses `daily_vote + weekly_vote`;
6. `maturity_pair` uses `wam_vote + wal_vote`.

Each pair emits the sign of its sum and abstains at zero. These are attribution
controls with a shared parent, not external novelty comparators.

For each nonzero source-derived control vote, the control source identity is
the exact UTF-8 string:

```text
CRSB-336|control|<control_name>|<accession>|<source_available_at_utc>|<control_vote_sum>
```

`control_name` is exactly one of the six labels above and
`control_vote_sum` is the signed value before side projection: exactly the
component vote `-1` or `1` for a single-component control, and exactly the
arithmetic pair sum in `{-2,-1,1,2}` for a pair control. It uses the same
canonical signed-integer grammar as the primary. A zero component or pair sum
emits no source identity. Its signal ID is lowercase hexadecimal SHA-256 of
that exact string.

## Same-parent and timing controls

Using each accepted primary interval:

- `exact_direction_flip`: opposite primary side;
- `deterministic_random_side`: `LONG` iff the first byte of
  `SHA256("CRSB-336|<primary_signal_id>|RANDOM_SIDE")` is below 128;
- `constant_long`;
- `constant_short`; and
- `one_bar_delayed_entry`: primary side, entry and exit each shifted exactly
  five elapsed minutes.

No control can change source membership or primary non-overlap.

For `deterministic_random_side`, the hash input is the exact UTF-8 encoding of
the displayed string after inserting the 64-character lowercase hexadecimal
primary signal ID. “First byte” means `digest()[0]` of the raw 32-byte SHA-256
digest, not the first ASCII byte of its hexadecimal rendering. Values
`0..127` produce `LONG`; values `128..255` produce `SHORT`.

Each same-parent control has the exact UTF-8 source identity:

```text
CRSB-336|control|<control_name>|<primary_signal_id>|<entry_time_utc>|<exit_time_utc>|<side>
```

Here `control_name` is exactly one of `exact_direction_flip`,
`deterministic_random_side`, `constant_long`, `constant_short`, or
`one_bar_delayed_entry`; `primary_signal_id` is exactly 64 lowercase
hexadecimal ASCII characters; timestamps use the canonical UTC grammar above;
and `side` is exactly `LONG` or `SHORT`. The control signal ID is lowercase
hexadecimal SHA-256 of this exact UTF-8 string. The delayed control serializes
its shifted timestamps; the other four serialize the accepted primary
timestamps.

## Source-support gates

Source support opens only sealed SEC source bytes and computes no external
clock or outcome.

### Source integrity

All source-axis gates must pass exactly, including:

- all 630 daily-index receipts;
- exact Feed membership and first-dissemination extraction;
- exact current-archive parity;
- exactly 42 originals, one for every report month 2022-11 through 2026-04;
- exact N-MFP2/N-MFP3 transition;
- exact identity, schema, path, WAM, WAL, and causal clocks;
- deterministic byte-identical rebuild; and
- zero forbidden/source-outcome access.

### Primary support

Accepted primary signals must satisfy:

| Window | Minimum signals |
|---|---:|
| 2023H2 | 4 |
| 2024 | 8 |
| selection | 12 |
| future25 | 8 |
| future26 | 2 |
| full | 25 |

Additional exact gates:

- at least three `LONG` and three `SHORT` signals in selection;
- at least six `LONG` and six `SHORT` signals in full;
- at least three signals of each side in combined future25+future26;
- no duplicate signal ID, decision, entry, or interval;
- all intervals are sorted, contained, and non-overlapping; and
- suppressed overlaps are exactly zero.

### Vote diversity

The `full-window candidate report population` is every valid original after
the first warm-up whose `decision_time`, hypothetical `entry_time`, and
hypothetical `exit_time` are all contained in
`[2023-06-01T00:00:00Z, 2026-06-01T00:00:00Z)`. It is formed before primary
zero-vote filtering and before non-overlap. The denominator for the zero-vote
and mixed-sign fractions is exactly this population.

Within that population, `primary acts` means `vote_sum != 0` before overlap.
For a report on which primary acts, a component disagrees when its nonzero sign
is opposite the primary sign; a zero component is an abstention. The external
period signal counts above use accepted nonoverlapping intervals after the
primary vote filter.

Across this exact population:

- every component vote must be positive at least four times and negative at
  least four times;
- at least 25% of reports must contain both a positive and a negative
  component vote;
- primary zero-vote reports may be at most 30%;
- every single-component control must disagree with the primary side or
  abstain while primary acts on at least four reports; and
- both pair controls must disagree with the primary side or abstain while
  primary acts on at least two reports.

The same population is also partitioned, without overlap, by
`schema_path_kind`. For each of `nmfp2_friday_slots` and
`nmfp3_dated_details`, independently:

- at least eight candidate-population reports must exist;
- at least four accepted primary signals must exist in the full window;
- at least one accepted primary signal must be `LONG` and at least one must be
  `SHORT`; and
- each of the four component votes must be positive at least once and negative
  at least once.

These form-family gates are diagnostics of support for the single frozen
direction rule, not permission to estimate a family-specific rule. The
support artifact must publish candidate-report, accepted-signal, side, and
component-sign counts separately for both schema kinds. The source transition
itself is not a signal, a weight, a regime, or an allowed segmentation
variable.

All fractions use exact integer cross multiplication. These gates validate that
the majority is not a disguised copy of one component; they do not optimize a
threshold.

Failure of any ordered source gate retires CRSB-336 before external novelty.

## Novelty before economics

Only a passed source-support artifact may authorize novelty. Novelty opens
clock artifacts only: no BTC price, funding, return, PnL, CAGR, MDD, or model.

The preregistration imports without deletion or reinterpretation:

- all 18 frozen prior-source-family comparators from
  `results/ethereum_settlement_demand_impulse_preregistration_2026-07-30.json`;
- every positive-weight Gross9 sleeve;
- the exact executable overlap functions and runtime code closure from the
  ESDI authority; and
- every comparator path, schema, coverage rule, Git blob, and SHA-256.

The imported ESDI authority is uniquely frozen as:

```text
latest authority-path commit
  f3de120a288f17e562e3f5cf7952ee77f6511fa7

preregistration path
  results/ethereum_settlement_demand_impulse_preregistration_2026-07-30.json
file SHA-256
  2a481fc60044d3d468340457d50f92a91f2a52184a464e1a91badfb418bbcaba
Git blob
  1b3d8b244426c0876d2995ce4a23159961d3cfa6
manifest_hash
  d5279f95cc7b92757aa77ecbbc5835d8b1cc4ce34f5a81d6f279abdcf2fcfe8a
novelty.frozen_comparator_artifacts canonical SHA-256
  0d13c9de1e098446aaaa78b9a24c7d05c7ec375df05d79c9f8969792546bd4a3
gross9 canonical SHA-256
  d79c79789ed48c7c2a94bac4474583798c2306bd320abb2617c354878c3578fe
gross9.authority canonical SHA-256
  b3490c484d3fda1d5b649498e0d84325e203cd2664086e68cebd76509a54957e
gross9.authority.runtime_code_closure canonical SHA-256
  ffffb68c0900836ba06b573398c4825bd9d15161a9e36818aeb68fc33a86d84a

novelty helper path
  training/preregister_ethereum_settlement_demand_impulse.py
file SHA-256
  1c7d7c822f16818ce0bc8fa0be99db0fe156882dbb76bf804ae19232f2a53b26
Git blob
  26c99dd77083cb0432160292c723da5ac2c019c3

economic helper path
  training/evaluate_ethereum_settlement_demand_impulse_economics.py
file SHA-256
  fba7de6a26ede945edfe63c32dd4a0c88760c6459ac0d4f079dd12d546580235
Git blob
  ff305a740636bdf90a2bfd53ae13d93cd9f994b9
```

Novelty must call the exact bound functions
`entries_in_domain`, `exact_entry_jaccard`,
`bidirectional_entry_containment`, `fraction_at_most`,
`signed_exposure_5m`, `occupied_bar_jaccard`, and
`squared_signed_exposure_pearson`. Economics must use the exact bound ESDI
accounting helper above. A path, file, Git object, manifest, subtree, function,
runtime closure, or environment mismatch is terminal.

The novelty attempt claim must be committed before any comparator or Gross9
clock row is opened.

### Prior-source-family thresholds

For every comparator with at least ten entries after identical half-open common
domain filtering:

```text
exact-entry Jaccard                    <= 0.20
candidate containment within +/-24h   <= 0.50
absolute signed-exposure correlation  <= 0.40
```

Below-minimum comparators are reported and do not gate. They are never silently
removed.

### Gross9 sleeve thresholds

For every positive-weight sleeve:

```text
exact-entry Jaccard                    <= 0.10
candidate containment within +/-6h    <= 0.35
occupied-bar Jaccard                   <= 0.25
absolute signed-exposure correlation  <= 0.35
```

Signed exposure is exact `{-1,0,1}` on the shared five-minute grid. Pearson is
implemented as exact squared correlation with an explicit sign check and
rational threshold. Missing, malformed, hash-drifted, coverage-empty, or
zero-variance required inputs fail closed.

One failed novelty comparison permanently retires CRSB-336 before economics.

## Standalone accounting

Only a passed novelty artifact may authorize economics.

Frozen standalone terms:

```text
instrument:                  Binance USD-M BTCUSDT perpetual
standalone leverage:         0.5x gross
entry/exit:                  next available five-minute open
normal cost:                 6 bp per notional side
stress cost:                 10 bp per notional side
funding:                     exact realized causal funding cash flow
liquidation maintenance:     0.5%
full-calendar CAGR:          mandatory
strict MDD:                  global/pre-entry HWM plus intratrade path
```

Market bars and funding must use the exact ESDI accounting authority and
hash-bound artifacts. Missing bars, a noncausal funding row, duplicate
timestamps, impossible entry/exit, or lifecycle mismatch is terminal.

On each complete five-minute bar, accounting follows the ESDI authoritative
order:

1. start-of-bar HWM and equity;
2. entry cost for entries at the open;
3. adverse/favorable same-BTC intrabar envelope under favorable-before-adverse
   strict-MDD ordering;
4. liquidation check using global equity and gross quantity;
5. realized funding debit or credit at its causal timestamp;
6. exits and exit cost;
7. end-of-bar equity and HWM; and
8. next-bar state.

Strict MDD includes idle pre-entry HWM, entry and exit costs, funding, and the
full intratrade low/high path. Absolute return is always reported beside CAGR.

Source, vote, scheduling, support, and novelty arithmetic remains exact
rational/integer arithmetic. Economic simulation and economic gates instead
use the authenticated ESDI `float64` implementation without rounding before
comparison:

```text
years(full)       = 3.0 exactly
years(other)      = (end-start).total_seconds() / (365.25*86400)
CAGR              = exp(log(final_equity)/years) - 1
ratio, CAGR > 0   = min(float64_max, CAGR/max(strict_MDD, 1e-15))
ratio, CAGR <= 0  = 0
mean move bp      = arithmetic mean of
                    side*(exit_open/entry_open-1)*10000
```

The mean includes completed contained primary trades only. Nonfinite or
nonpositive final equity is terminal. All gate comparisons use unrounded
Python `float64` values and the exact ESDI comparison tolerances. Display
serialization may expose decimal renderings of those binary-derived values,
but a displayed or rounded value can never be used for a gate.

The calendar-month clustered sign-flip test groups primary trade
`net_return_on_allocated_equity` by UTC entry month, sums within month, drops
clusters with absolute value at most `1e-15`, and uses a one-sided exceedance
against the observed sum with the ESDI `1e-15` comparison tolerance. At most
20 nonzero months use exact enumeration; more than 20 use exactly 10,000
NumPy `default_rng(20260730)` draws and the ESDI plus-one Monte Carlo
correction. A nonpositive observed sum has p-value one.

## Standalone gates

The economic evaluator opens periods strictly in this order:

```text
2023H2 -> 2024 -> selection -> same-gross selection
       -> future25 -> future26 -> combined future -> stitched full
```

It stops permanently at the first failure.

Normal and stress cost must each pass:

```text
absolute return                         > 0
full-calendar CAGR / strict MDD        >= 3.0
strict MDD                             <= 0.15
mean gross underlying move             >= 20 bp
no liquidation
```

Minimum completed primary trades:

```text
2023H2 >= 4
2024   >= 8
selection >= 12
future25 >= 8
future26 >= 2
```

The 2023H2 and 2024 calendar-month clustered sign-flip statistics are
report-only. In combined selection, normal and stress cost each require:

```text
calendar-month clustered sign-flip p   <= 0.20
```

## Control superiority

In combined selection under both normal and stress cost:

- primary CAGR/strict-MDD must be strictly greater than
  `exact_direction_flip`, `deterministic_random_side`, `constant_long`, and
  `constant_short`;
- primary CAGR/strict-MDD must be strictly greater than at least four of the
  six source-derived attribution controls;
- primary absolute return must be greater than the maximum of the exact
  direction-flip and deterministic-random controls; and
- every primary/control result must be computed from the same market and
  funding arrays in one authenticated run.

No control can become an alternate candidate.

## Same-configured-gross Gross9 selection

The exact Gross9 authority, five sleeves, baseline weights, configured gross
`9.0`, market/funding arrays, accounting code, and source clocks are imported
from the ESDI preregistration and runtime closure without reinterpretation.

Only these CRSB candidate weights are tested:

```text
0.25, 0.50, 0.75, 1.00
```

For candidate weight `c`, scale every Gross9 sleeve by `(9-c)/9` and add CRSB
at configured weight `c`. The treatment and the unscaled Gross9 baseline both
therefore retain configured gross exactly `9.0`; no `9+c` baseline or
treatment exists.

Configured weight is allocated-equity weight under the ESDI authority:

```text
allocated equity = current equity * configured weight
quantity         = allocated equity * 0.5 / entry open
```

The common global leverage is `0.5`, so CRSB contributes actual entry notional
gross `0.5*c`. For each exact period/cost cell:

```text
ratio improvement = treatment ratio - unscaled Gross9 ratio
MDD reduction     = unscaled Gross9 MDD - treatment MDD
return retention  = treatment absolute return / Gross9 absolute return
```

If baseline absolute return is not positive, retention is `-inf` and the cell
fails.

Fresh selection evaluation covers exactly `2023H2` and calendar `2024`, each
under normal and stress cost. Every one of the four cells must require:

```text
treatment absolute return >= 97% of unscaled Gross9 absolute return
treatment CAGR/strict-MDD >= unscaled Gross9 ratio + 0.05
treatment absolute return > 0
no treatment liquidation
```

A candidate weight also must strictly reduce MDD versus unscaled Gross9 in at
least one of the four cells. Derive each weight's pass/fail flag, then rank all
four weights, including failed weights, by the raw unrounded `float64` minimum
ratio improvement across the exact four cells, descending, tie-breaking by the
raw numeric candidate weight ascending. Ranking uses Python tuple ordering
equivalent to:

```text
sorted(rows, key=(-minimum_improvement, candidate_weight))
```

There is no epsilon, `isclose`, displayed-decimal comparison, or stable-input
order tie-break. Candidate weights are unique, so an exact score tie is
resolved by lower weight. Rank one is frozen only if its already-derived pass
flag is true. If raw rank one fails, CRSB-336 is retired even when a lower rank
passes; rank two can never substitute.

The exact top one is frozen before future arrays are opened. If no weight
passes, CRSB-336 is retired. Future cannot rerank, open rank two, change a
weight, or repair selection.

## Future veto

The frozen top one opens `future25`, then `future26`, then `combined_future`
only. No stage reranks or opens another weight.

For each future subperiod and each cost:

- standalone CRSB must pass the ordinary return, ratio, MDD, move, trade-count,
  and liquidation gates;
- treatment CAGR/strict-MDD must exceed its same-gross baseline by at least
  `0.05`;
- treatment must retain at least `97%` of positive unscaled Gross9 absolute
  return;
- treatment absolute return must remain positive; and
- treatment must be liquidation safe.

Within each future subperiod, treatment strict MDD must be strictly lower than
unscaled Gross9 in at least one of the two cost settings.

`combined_future` is a fresh, non-stitched evaluation over
`[2025-01-01T00:00:00Z, 2026-06-01T00:00:00Z)` using only the already frozen
weight. Under both costs it requires:

```text
treatment absolute return >= 97% of unscaled Gross9 absolute return
treatment CAGR/strict-MDD >= same-gross baseline ratio + 0.05
treatment absolute return > 0
candidate completed trades >= 10
candidate active calendar months >= 10
calendar-month clustered sign-flip p <= 0.20
no liquidation
```

Combined-future treatment strict MDD must be strictly lower than unscaled
Gross9 in at least one cost setting. Only a complete future pass authorizes
the stitched full report. Full results are confirmation/reporting, not a new
selection surface.

## Reproducibility and publication

Every production stage is write-once and manifest-last:

1. preregistration producer and tests committed;
2. write-once preregistration artifact committed;
3. source builder, support, novelty, economics, and tests committed;
4. a source-access claim committed before the 630 daily-index requests;
5. source and support run once;
6. novelty claim committed, then novelty run once;
7. each economic stage claim committed before its rows open;
8. each passed artifact rebuilt independently from sealed inputs and compared
   byte-for-byte; and
9. final tests, typecheck, compile, diff-check, code review, commit, and push.

All production artifacts bind code blobs, source artifacts, prior-stage
manifest hashes, and evidence counters. A write-once existing artifact is
validated byte-for-byte rather than overwritten.

## Stop and anti-repair rule

The first failed source, support, novelty, standalone, control, same-gross,
future, integrity, or reproduction gate permanently retires `CRSB-336`.

After production source incidence opens, the branch may not change:

- source membership, parser, form transition, availability, or path;
- the four votes, vote weight, tie behavior, side, latency, or hold;
- controls, support floors, novelty cohort, overlap thresholds, or domains;
- leverage, costs, funding, MDD, CAGR, trade count, statistical test, weights,
  ranking, future veto, or Gross9 authority; or
- a failed direction, threshold, source field, period, or candidate identity.

No `CRSB-336B`, direction flip, threshold relaxation, source-field addition,
amendment substitution, factsheet fallback, or later-period repair is allowed
on this alpha search.
