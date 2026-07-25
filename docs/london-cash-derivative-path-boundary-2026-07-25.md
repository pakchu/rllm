# LCDP-D1 boundary — London cash/derivative daily path

Date: 2026-07-25

## Decision

Accept one new candidate for an outcome-blind source/token-support freeze:

**LCDP-D1 — London Cash/Derivative Path, twenty-one-line daily
target-position RLLM.**

LCDP-D1 observes one completed Coinbase BTC-USD and Binance BTCUSDT
perpetual path between consecutive 16:00 `Europe/London` boundaries. At the
next executable boundary, one policy chooses exactly one target:

```text
TARGET_SHORT
TARGET_FLAT
TARGET_LONG
```

The target remains in force until the next daily London boundary. Deterministic
code owns source validation, clocks, transforms, ranks, tokens, execution,
costs, funding, reward, position transitions, and strict drawdown. The model
may see only the frozen categorical sequence and current target.

This boundary authorizes only a preregistration and a source/token-support
implementation. It does not authorize execution prices, funding cash flows,
future returns, rewards, model fitting, checkpoint selection, actions, trades,
PnL, CAGR, MDD, or a profitability claim.

## Why this is a new identity rather than an LCLR repair

The reserve selection is recorded in:

```text
docs/post-tracer-alpha-mechanism-audit-2026-07-25.md
SHA256 7394cd096d92b5469eb625605faaa8f53c49fc486b921269a1b2da0b08afbf9e
```

LCLR-24 used a sparse weekday event inside the one-hour
`[15:00,16:00)` London window. It required a same-direction cash lead plus
two of four votes, selected the Coinbase side deterministically, and held for
two hours. It retired on its frozen outcome-blind support gate.

LCDP-D1 is different:

- it emits one state on every London calendar day, including weekends;
- its source path spans the full interval between consecutive daily London
  boundaries rather than the LCLR one-hour event window;
- it has no event filter, optional votes, rolling median trigger, fixed side,
  cooldown, TP/SL, or two-hour hold;
- invalid or rank-unready days remain explicit safety lines and force
  `TARGET_FLAT`; they are never dropped;
- the model selects a target position rather than approving a preselected
  trade; and
- the sequence contains twenty-one ordered daily relation lines plus the
  current target.

The exact LCLR mask is retained only as a later killer baseline. It may not
filter LCDP states, labels, training rows, or actions.

LCDP intentionally reuses the Coinbase/Binance source family. Earlier
Coinbase-leadership candidates and LCLR results are known. This is disclosed
contamination, not a pristine-family claim. The candidate-specific novelty is
the exact dense full-day relational MDP and its frozen sequence language.

## Frozen source containers

Only the exact pre-2023 containers below are authorized during source support.
The loader must project the listed columns while parsing; loading extra
columns and dropping them later is forbidden.

### Coinbase BTC-USD five-minute candles

```text
path
  data/coinbase_btcusd_5m_2020_2022.csv.gz
file SHA256
  07f7a3bddecbbc3724994645b9ac1cd0f391378e0feed421f2c8caa145aab77b
physical header
  date,open,high,low,close,volume,source_complete
header SHA256 including newline
  056e6938d2dea3e9ef9a9230ca192cbfcf11ea270151115f96cf4e7c94c0de17
```

Exact projected columns and order:

```text
date
open
high
low
close
volume
source_complete
```

### Binance BTCUSDT perpetual five-minute candles

```text
path
  data/coinbase_leadership_binance_5m_2020_2022.csv.gz
file SHA256
  1a06f1f4dbbdafaf885fb03844426eed5d5bad4aa206fa72b88db2cbd98bef94
physical header
  date,open,high,low,close,quote_asset_volume
header SHA256 including newline
  8b70cdf275862b56bcbdd7e10b18c7d82c9cac47dcc3dd2c5ceae78f8f102232
```

Exact projected columns and order:

```text
date
open
high
low
close
quote_asset_volume
```

### Audited source manifest

```text
path
  results/coinbase_spot_leadership_source_manifest_2026-07-16.json
file SHA256
  3af321fdcafd0fe6680c4583341b6508124a979fefbf489f8d3376c7ec78a269
manifest_hash
  243ecba3b9e31548d682084dd5acc2e89c6a24423bce241dd6338a57dd6eefe9
start inclusive
  2020-01-01
end exclusive
  2023-01-01
```

The manifest reports 315,648 expected rows, 315,528 complete Coinbase rows,
120 missing Coinbase rows left unimputed, and a complete Binance grid. It also
states `historical_snapshot_is_point_in_time=false`. LCDP may therefore make a
retrospective candidate-level claim only. Live-equivalence requires a later
prospective reconstruction and parity stage.

The funding file named by that manifest is forbidden during source support.
It can be opened only after a source pass and a separately committed economic
evaluator freeze.

## Frozen daily clock

For London calendar date `D`:

```text
B_D                   = D 16:00:00 Europe/London
source window         = [B_(D-1), B_D)
last source bar       = [B_D-5m, B_D)
state complete        = B_D
inference deadline    = B_D+5m
policy decision       = B_D+5m
latency bar           = [B_D+5m, B_D+10m)
rebalance execution   = Binance open at B_D+10m
target interval       = [B_D+10m, B_(D+1)+10m)
```

Consecutive local boundaries are 23, 24, or 25 elapsed hours across London
DST transitions. A complete source window therefore contains exactly 276,
288, or 300 five-minute slots according to its UTC elapsed duration. Treating
every day as 288 rows, using a fixed UTC boundary, or silently deleting a DST
day changes the candidate identity.

There is one decision every calendar day, including Saturday and Sunday. At a
split start, the current target is `TARGET_FLAT`. At a split end, the
calendar-year terminal line is deterministic `TARGET_FLAT` at `B_D+10m`; the
model is not invoked and no target interval may cross into an unopened year.
Consequently, a calendar-year evaluator opens model actions for January 1
through December 30 and uses December 31 only to flatten the December 30
target at the already scheduled boundary. This rule also applies to every
train, validation, test, and sequential transfer year. No TP/SL,
discretionary early exit, event cooldown, or variable outcome-selected hold is
allowed.

If the current source line is invalid or rank-unready, or if inference misses
the deadline, returns an invalid grammar, raises an error, or fails a live
staleness check, the deterministic action is `TARGET_FLAT`. The model is not
allowed to override a safety flat.

## Frozen source validation

A venue window is valid only when all conditions hold:

1. every expected UTC five-minute timestamp in `[B_(D-1),B_D)` appears
   exactly once;
2. no timestamp outside that half-open interval contributes a value;
3. every selected numeric field is finite;
4. OHLC values are positive and satisfy
   `low <= min(open,close) <= max(open,close) <= high`;
5. volume or quote notional is nonnegative; and
6. every Coinbase row has `source_complete == 1`.

The joint line is source-valid only when both venue windows are valid. Missing,
duplicate, malformed, nonfinite, nonpositive, incomplete, or misaligned rows
produce one explicit `SOURCE_INVALID` line. No fill, interpolation, nearest
join, row deletion, or venue-only substitution is allowed.

The `2020-01-01` London line necessarily begins at `2019-12-31 16:00`
London, before the authorized source starts. It is always emitted as the
distinct safety line `SOURCE_INVALID_START`, counts as source-invalid in the
2020 annual and quarterly gate denominators, and contributes no prior rank
value. Pre-2020 data may not be added later to repair it.

The source-only loader may parse date fields throughout the physical files to
enforce chronology and the `<2023-01-01` cutoff. It may parse non-date fields
only for authorized pre-2023 source rows. At-or-after-2023 non-date fields,
funding, execution bars, and outcomes remain unopened.

## Frozen numeric primitives

For venue `v` and the chronologically ordered source bars:

```text
x_v,0       = log(first open)
x_v,i       = log(close_i), i=1..N
step_v,i    = x_v,i - x_v,i-1
return_v    = sum(step_v)
path_v      = sum(abs(step_v))
eff_v       = abs(return_v) / path_v, or 0 when path_v=0
range_v     = log(max(high) / min(low))
quote_cash  = sum(Coinbase volume_i * close_i)
quote_perp  = sum(Binance quote_asset_volume_i)
cash_share  = quote_cash / (quote_cash + quote_perp)
basis_start = log(Coinbase first open / Binance first open)
basis_end   = log(Coinbase last close / Binance last close)
```

The first and second arcs are the first and last halves of the ordered bar
sequence. All authorized window lengths are even:

```text
return_v,first   = sum(first N/2 step_v)
return_v,second  = sum(last N/2 step_v)
relative_first  = return_cash,first - return_perp,first
relative_second = return_cash,second - return_perp,second
```

Cash participation state uses strictly prior values:

```text
lookback                 = previous 126 emitted daily lines
minimum finite valid rows = 63
current line excluded
low boundary             = prior 1/3 quantile
high boundary            = prior 2/3 quantile
```

Invalid values remain missing and are ignored. If fewer than 63 finite prior
cash-share values exist, the entire current line is `RANK_UNREADY`. No other
rank, z-score, tuned threshold, price level, or future statistic is allowed.

## Frozen categorical line language

Each ready line has the exact ordered fields below.

```text
calendar_context
daily_alignment
daily_leader
relative_basis_path
arc_transfer
path_efficiency
range_relation
participation_state
participation_transition
alignment_transition
leader_transition
```

Allowed vocabularies:

```text
calendar_context
  WEEKDAY | SATURDAY | SUNDAY

daily_alignment
  BOTH_RISE | BOTH_FALL | CASH_RISE_PERP_FALL |
  CASH_FALL_PERP_RISE | RETURN_MIXED_OR_FLAT

daily_leader
  CASH_LEADS_RISE | CASH_LEADS_FALL |
  PERP_LEADS_RISE | PERP_LEADS_FALL | NO_CLEAR_LEADER

relative_basis_path
  CASH_RICHENS | CASH_CHEAPENS | BASIS_ROTATES | BASIS_FLAT

arc_transfer
  CASH_LEAD_EXTENDS | CASH_LEAD_REVERSES |
  PERP_LEAD_EXTENDS | PERP_LEAD_REVERSES | ARC_MIXED

path_efficiency
  CASH_CLEANER | PERP_CLEANER | BOTH_CHOPPY_OR_TIE

range_relation
  CASH_RANGE_DOMINANT | PERP_RANGE_DOMINANT | RANGE_BALANCED

participation_state
  CASH_PARTICIPATION_LOW | CASH_PARTICIPATION_MID |
  CASH_PARTICIPATION_HIGH

participation_transition
  CASH_SHARE_RISING | CASH_SHARE_FALLING |
  CASH_SHARE_STABLE | PARTICIPATION_UNKNOWN

alignment_transition
  ALIGNMENT_PERSISTS | ALIGNMENT_FLIPS |
  ALIGNMENT_DISSIPATES | ALIGNMENT_MIXED

leader_transition
  CASH_LEAD_PERSISTS | PERP_LEAD_PERSISTS |
  LEAD_ROTATES_TO_CASH | LEAD_ROTATES_TO_PERP | LEAD_MIXED
```

Exact mapping:

- return signs use exact comparison with zero; no epsilon is fitted;
- `daily_leader` requires same nonzero return sign and compares absolute
  full-window returns, otherwise it is `NO_CLEAR_LEADER`;
- `BASIS_ROTATES` requires nonzero `basis_start` and `basis_end` with opposite
  signs; otherwise rising basis is `CASH_RICHENS`, falling basis is
  `CASH_CHEAPENS`, and equality is `BASIS_FLAT`;
- `arc_transfer` is determined by the exact signs of `relative_first` and
  `relative_second`; a zero in either arc is `ARC_MIXED`;
- efficiency and range relations use exact greater/less comparisons and the
  named tie category;
- participation low is `cash_share < prior q1/3`, high is
  `cash_share > prior q2/3`, and all finite boundary/equality values are mid;
- participation transition compares the current participation-state ordinal
  with the immediately preceding emitted calendar-day line; a safety
  predecessor is `PARTICIPATION_UNKNOWN`;
- alignment persists only when the current and immediately preceding emitted
  calendar-day lines are ready and are the same `BOTH_RISE` or the same
  `BOTH_FALL`; it flips between those two aligned directions, dissipates from
  an aligned category into another alignment category, and is mixed when
  either line is safety or otherwise; and
- leader transition compares the cash/perp venue implied by the current and
  immediately preceding emitted calendar-day `daily_leader` values.
  Same-venue leadership persists, a venue change rotates to the current venue,
  and any safety or `NO_CLEAR_LEADER` involvement is mixed.

Safety lines use the same schema:

```text
calendar_context = actual WEEKDAY | SATURDAY | SUNDAY
every market field = SOURCE_INVALID
```

or:

```text
calendar_context = actual WEEKDAY | SATURDAY | SUNDAY
every market field = RANK_UNREADY
```

The first line instead uses:

```text
calendar_context = WEEKDAY
every market field = SOURCE_INVALID_START
```

The exact primary field vocabulary is the ready-field vocabulary union:

```text
SOURCE_INVALID | SOURCE_INVALID_START | RANK_UNREADY
```

The exact control-only vocabulary is field-scoped:

```text
calendar_context
  ready calendar value | CALENDAR_MASKED

daily_alignment
  ready primary value |
  CASH_ONLY_RISE | CASH_ONLY_FALL | CASH_ONLY_FLAT |
  PERP_ONLY_RISE | PERP_ONLY_FALL | PERP_ONLY_FLAT |
  ABLATION_MASKED | CONTROL_UNREADY

every other market field
  ready primary value | ABLATION_MASKED | CONTROL_UNREADY
```

These safety and control values are not ready-field categories. Exact
serialization is the ordered `field=value` pairs joined by one ASCII `|`
without omitted fields. Safety/mask values may never be collapsed, omitted,
imputed, or treated as a ready category.

## Frozen sequence and model boundary

One model-eligible state contains the current ready line and the twenty
immediately preceding emitted calendar-day lines, oldest to newest. Safety
history lines remain visible. The current target is appended separately.

The prompt may contain only:

```text
STATE_-20 ... STATE_0
CURRENT_TARGET=TARGET_SHORT|TARGET_FLAT|TARGET_LONG
```

Each state line contains the ordered categorical fields above. It contains no
calendar date, year, raw timestamp, price, return, volume, notional, rank,
quantile, probability, future path, funding, reward, action label, PnL, CAGR,
MDD, or previously evaluated split statistic.

The output grammar is exactly:

```json
{"target":"TARGET_SHORT"}
{"target":"TARGET_FLAT"}
{"target":"TARGET_LONG"}
```

No analyzer/trader cascade, separate gate model, side model, model-generated
size, model-generated exit, or direct exchange order is allowed.

The base model, tokenizer, adaptation method, reward implementation, training
algorithm, seeds, checkpoint rule, and parser must be committed and hash-frozen
after source support passes and before any LCDP execution outcome or reward is
constructed.

## Frozen source/token controls

Every control rebuilds all affected primitives, prior quantiles, transitions,
and sequences:

1. `cash_perp_role_swap`: swap cash and perpetual path roles after converting
   both participation inputs to quote notional;
2. `cash_stale_one_day`: pair current perpetual path with the immediately
   preceding emitted cash path;
3. `perp_stale_one_day`: pair current cash path with the immediately preceding
   emitted perpetual path;
4. `lag_7_calendar_days`: use the complete joint path from exactly seven
   emitted calendar days earlier;
5. `calendar_context_mask`: replace calendar context with `CALENDAR_MASKED`;
6. `cash_only_language`: retain calendar context, map the exact sign of
   `return_cash` to `CASH_ONLY_RISE|CASH_ONLY_FALL|CASH_ONLY_FLAT` in
   `daily_alignment`, and set every other market field to
   `ABLATION_MASKED`;
7. `perp_only_language`: retain calendar context, map the exact sign of
   `return_perp` to `PERP_ONLY_RISE|PERP_ONLY_FALL|PERP_ONLY_FLAT` in
   `daily_alignment`, and set every other market field to
   `ABLATION_MASKED`.

When a stale/lag control lacks its required prior source day, every market
field is `CONTROL_UNREADY`. The control line remains present.

The later LCLR-family controls are frozen now but do not filter the primary:

```text
LCLR preregistration
  docs/london-cash-lead-release-preregistration-2026-07-20.md
  SHA256 fd996475dba37953b1abc0ec29cfe9edbe7d33b91d61d7880f4e0c7ea9330c65
LCLR support rejection
  docs/london-cash-lead-release-support-rejection-2026-07-20.md
  SHA256 462a521079ae55076495885516ffe3e6e5dc870a50de7a6f3d310e3026f6d5c6
```

1. `lclr_exact_policy` independently replays the already frozen LCLR event
   definition, side, 16:05 entry, 18:05 exit, and 0.5x exposure on the same
   full calendar equity timeline; and
2. `lclr_mask_daily_target` maps an exact LCLR event with London window date
   `D` to the LCDP line for the same `D`, chooses the frozen LCLR side at the
   LCDP `B_D+10m` execution, chooses `TARGET_FLAT` on every non-event date,
   and holds each result only to the next LCDP boundary.

Neither control may use an LCDP model action, remove an LCDP row, select an
LCDP checkpoint, or gate the primary policy. The candidate fails the later
economic stage if `lclr_mask_daily_target` earns at least 50% of positive
primary net PnL in the same split or independently passes every primary gate.

Controls are diagnostics only. They can never replace, repair, or select the
primary language after incidence is observed.

## Frozen source-support gates

All gates are conjunctive and evaluated in order. The first failure retires
LCDP-D1 unchanged before outcomes.

1. **Protocol and source integrity**
   - exact source, header, manifest, boundary, and preregistration hashes;
   - exact projected column order and strict types;
   - chronological unique physical timestamps;
   - no non-date field parsed at or after `2023-01-01`.
2. **Calendar and DST integrity**
   - exactly 1,096 primary lines for `2020-01-01` through `2022-12-31`;
   - one line per London calendar day without deletion;
   - exactly one `SOURCE_INVALID_START` line on `2020-01-01`;
   - exact 276/288/300-slot expectation from adjacent local boundaries;
   - only the calendar-implied London UTC offsets.
3. **Source validity**
   - at least 97% source-valid lines in every year;
   - at least 95% source-valid lines in every calendar quarter.
4. **Readiness**
   - 2020Q1 is warm-up/accounting only and is not exempt from source validity;
   - at least 280 model-eligible ready sequences in 2020;
   - at least 350 in each of 2021 and 2022; and
   - at least 80 in every quarter after 2020Q1.
5. **Token diversity**
   - in each year, every non-calendar ready field has at least two non-safety
     categories with at least 3% share each;
   - no non-calendar ready category exceeds 94% of its field-year; and
   - both positive and negative cash and perpetual full-window directions
     occur in every year.
6. **Control distinctness**
   - every frozen control stream exists;
   - `cash_perp_role_swap`, both stale controls, and `lag_7_calendar_days`
     differ from the primary on at least 5% of jointly ready dates; and
   - each mask control differs from the primary wherever its source-specific
     fields are ready.
7. **Append replay**
   - prefix builds ending before `2021-01-01`, `2022-01-01`, and
     `2023-01-01` byte-match the corresponding rows from the full build;
   - no prefix build accesses a later non-date value.
8. **Forbidden-access counters**
   - funding rows opened: zero;
   - execution or post-boundary market rows opened: zero;
   - future-return rows built: zero;
   - reward/model/action/trade rows built: zero;
   - PnL/CAGR/MDD values computed: zero; and
   - at-or-after-2023 non-date source rows parsed: zero.

A pass authorizes only a separate economic/RLLM evaluator freeze. It is not an
alpha, model, return, trade, or deployability result.

## Contingent economic sequence

Only after every source gate passes:

1. Commit and hash-freeze one evaluator, one RLLM training procedure, one
   model family, one tokenizer, one prompt serializer, one parser, one
   transaction-cost model, one funding convention, and one checkpoint rule.
2. Open 2020 outcomes for training and 2021 outcomes for validation. The 2022
   outcome path remains sealed.
3. Freeze the selected algorithm/checkpoint rule from 2020/2021, then evaluate
   calendar 2022 exactly once.
4. Require calendar-2022 positive absolute return, CAGR/strict MDD at least
   3.0, strict MDD at most 15%, positive 10 bp/notional/side stress return,
   positive one-additional-bar-delay return, at least 60 nonflat target-days,
   at least 10 long and 10 short target-days, and weekly-cluster sign-flip
   `p <= 0.10`.
5. Require the primary to exceed the best nontrivial source/token killer
   control by at least 0.50 CAGR/strict-MDD in 2022. Always-flat,
   always-long, always-short, exact action flip, one-day delayed action,
   UTC-midnight clock, weekday-only clock, deterministic matched-count random,
   current-line-only, and exact LCLR-mask policies are mandatory controls.
6. Stop if an ablation/control passes the same primary gate or if
   `lclr_mask_daily_target` earns at least half of positive primary net PnL.
7. Only one unchanged pre-2023 pass may authorize a separately committed
   2023+ Coinbase/Binance/funding source extension and live-parity contract.
8. With the algorithm unchanged, refit annually on all strictly prior
   authorized years and evaluate 2023, 2024, 2025, then 2026 YTD
   sequentially. Stop at the first failed year. No within-year refit or
   failed-year repair is allowed.

Strict MDD must include the split-start high-water mark, every target
transition cost, every held five-minute adverse path, exact realized funding,
a virtual adverse close fee at every path mark, and the forced split-end
flatten. The terminal December 31 flat forbids opening any next-year execution
bar, return, or funding mark before that next year is separately authorized.
CAGR uses the full declared wall-clock split, including warm-up, safety-flat,
and idle time.

## Failure and live parity

The first source, coverage, diversity, control, replay, economic, or transfer
failure retires LCDP-D1 under this identity. Changing its boundary, source
window, rank horizon, token mapping, control, model family, action space,
cost, reward, gate, or chronology after observing a failed result is a new
candidate, not a repair.

Historical Coinbase candles were fetched as a current retrospective snapshot.
Before live capital, a prospective shadow stage must:

- reconstruct Coinbase and Binance five-minute bars from live public streams;
- close each source bar before use and reject late revisions past the frozen
  inference deadline;
- reconcile reconstructed bars with the official REST source without
  replacing the point-in-time policy input;
- prove historical/live token parity on an overlap;
- force flat on missing, stale, malformed, disconnected, or divergent input;
  and
- keep model output behind deterministic position, leverage, margin, and
  kill-switch controls.

Candidate-level historical transfer is not evidence of prospective live
profitability. A live claim requires forward shadow and then tightly capped
execution evidence.
