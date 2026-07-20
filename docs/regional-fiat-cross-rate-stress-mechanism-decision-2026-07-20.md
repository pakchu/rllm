# RFXS-576 mechanism decision — regional-fiat cross-rate stress

## Decision

The next standalone BTC candidate is **RFXS-576 — Regional-Fiat Cross-Rate
Stress**, a fixed 48-hour contrarian policy built from completed Binance Spot
daily closes for `BTCUSDT`, `BTCEUR`, `BTCTRY`, and `BTCBRL`.

RFXS removes the common BTC return from three regional BTC books. The residual
is the daily change in the BTC-implied local-fiat/USDT cross rate. A broad,
unusually positive residual means EUR, TRY, and BRL are weakening against USDT
more than their own recent norms; the frozen hypothesis maps that dollar-like
stress to short BTCUSD risk. Broad easing maps long.

This file freezes the observable, direction, timing, source-support gates,
controls, sequential evaluation contract, and no-repair boundary before any
regional close value, residual, z-score, event count, post-entry return,
funding cash flow, PnL, CAGR, or MDD is opened for this candidate.

## Why this is a new observable

The existing FQPR-3 branch uses the same four Spot symbols but explicitly uses
only base volume, trade count, and taker flow. It excludes every OHLC value.
RFXS uses only completed close-to-close cross-rate residuals and excludes all
flow fields.

SDDR-12 used hourly BTC price disagreement across the USDC, FDUSD, and USDT
denominators during a four-month 2023 prefix. RFXS instead uses three sovereign
fiat regions, one-day changes, per-region normalization, and a 2020-2023 source
prefix. It is not a threshold or direction repair of SDDR.

The feature also differs from the existing single-KRW kimchi/FX family. It
contains no KRW, Upbit, DXY, official FX close, funding, premium index, open
interest, order book, aggregate trade, liquidation, on-chain value, existing
alpha state, or portfolio PnL.

## Official source, entitlement, and live parity

Binance's public-data repository documents that Spot kline archives are derived
from `/api/v3/klines`, that daily/monthly ZIPs have companion checksum files,
that Spot timestamps use microseconds from 2025 onward, and that the repository
is MIT-licensed:

- [Binance public-data repository](https://github.com/binance/binance-public-data)
- [BTCUSDT daily archive](https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1d/)
- [BTCEUR daily archive](https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCEUR/1d/)
- [BTCTRY daily archive](https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCTRY/1d/)
- [BTCBRL daily archive](https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCBRL/1d/)
- [Official Spot REST documentation](https://developers.binance.com/en/docs/products/spot/rest-api)
- [Official Spot stream documentation](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md)

A source-existence probe on 2026-07-20 opened only companion checksum text and
current exchange metadata, not any ZIP or price value. All four symbols had
checksum-backed monthly daily-kline archives for 2021-01, 2023-12, and 2026-06,
and all four were reported `TRADING` with Spot trading enabled. BTCBRL's first
complete common monthly boundary is 2020-10; earlier 2020 monthly probes were
absent. The source build will therefore begin exactly at 2020-10-01.

That probe is non-normative feasibility metadata. It cannot enter a feature,
support gate, outcome gate, symbol fallback, or archive-selection rule. In
particular, seeing that a checksum object exists after 2023 does not authorize
opening its ZIP and does not establish historical availability or performance.
It does disclose a present-day live-tradability/survivorship consideration in
the fixed symbol choice. The normative historical vintage is only the exact
set of ZIP bytes whose official companion hashes, download URLs, and locally
computed SHA-256 values are later bound in the deterministic build manifest.
Documentation and current `exchangeInfo` are informative only; a later edit to
either cannot alter a frozen source artifact.

The production source is the UTC daily kline stream for all four symbols. A
daily row is admitted only after every stream reports a closed (`x=true`)
`1d` candle with matching UTC open/close timestamps. All four closed events
must be received, persisted, and validated no later than
`d+1 00:04:59.999 UTC`; otherwise that source day cannot trade. REST is a
recovery path but must satisfy the same deadline; missing, late, or conflicting
live candles fail closed. The historical archive cannot prove live receipt
latency, so the backtest's fixed five-minute latency is a causal simulation,
not evidence of live parity. Production promotion requires separately logged
receipt-time shadow parity.

The historical builder must verify every ZIP against its official companion
checksum and bind the checksum response hash, archive URL, published archive
hash, and locally computed archive hash in a deterministic manifest. A later
Binance archive replacement is a new source revision and cannot silently alter
a frozen evaluation. Replacement detection must halt before feature or outcome
computation until a new, explicitly named source revision is reviewed.

## Frozen source row

For each UTC source day `d`, retain only:

```text
date
source_available_not_before = d + 1 day 00:00:00 UTC
BTCUSDT_close
BTCEUR_close
BTCTRY_close
BTCBRL_close
source_complete
```

Every close must be finite and strictly positive. Each symbol must have exactly
one row with UTC open at `d 00:00:00` and close at
`d 23:59:59.999` in the timestamp unit documented for that archive. Raw OHLC
must satisfy ordinary kline bounds, and raw base volume and trade count must
both be strictly positive; those fields are used only to reject an empty/stale
source candle and are discarded before the source panel is written. The four
symbol calendars must be identical and gap-free, and every retained day must
pass these checks for all four books. No interpolation, stale carry, forward
fill, timezone-offset candle, or alternate symbol is allowed.

Only the close columns may enter the signal. Volume, quote volume, trade count,
taker fields, high, low, open, and post-close values are prohibited.

## Frozen causal feature

For region `r` in `{EUR, TRY, BRL}`, define the completed daily BTC-canceling
residual:

```text
x_r[d]
  = log(BTCr_close[d] / BTCr_close[d-1])
    - log(BTCUSDT_close[d] / BTCUSDT_close[d-1])
```

Equivalently, `x_r` is the daily log change of
`BTCr_close / BTCUSDT_close`. A positive value means more local-fiat units per
USDT than on the prior day; the label "fiat stress" is a hypothesis about that
cross rate, not an exchange-provided semantic.

Normalize each region independently against exactly 180 strictly prior complete
residual days. The current day is excluded. In the notation below, the
lookback is the ordered set `{d-180, ..., d-1}`, not a slice that includes `d`:

```text
z_r[d]
  = (x_r[d] - median({x_r[d-180], ..., x_r[d-1]}))
    / (1.4826 * MAD({x_r[d-180], ..., x_r[d-1]}))
```

An incomplete history, non-finite value, or zero MAD makes the complete day
unavailable. There is no fallback standard deviation, clipping, winsorization,
rank grid, fitted country weight, or expanding use of the current value.

The common state is:

```text
common_z[d] = median(z_EUR[d], z_TRY[d], z_BRL[d])

state[d] = +1  if common_z[d] >= +1.0
         = -1  if common_z[d] <= -1.0
         =  0  otherwise
```

Because the median must cross the threshold, at least two of three regions
must independently have `z >= +1` or at least two must have `z <= -1`. No
additional coherence or magnitude grid is permitted.

## Frozen event, direction, and execution

A `candidate_event` occurs when the current nonzero state differs from the
immediately preceding valid source day's state. Neutral-to-tail and direct sign
reversals qualify. A continuing tail does not repeatedly signal. An unavailable
day emits no candidate and is skipped only for the preceding-valid-state
comparison; it may not be imputed. A candidate suppressed by an existing
position still consumes its source-state transition.

A `reserved_event` is a candidate whose half-open scheduled interval
`[entry_time, exit_time)` does not intersect an earlier reserved interval.
Reservation uses only deterministic timestamps, not execution rows or returns,
and is chronological and continuous across the master source clock. A
split-crossing reserved event is not scored in either adjacent split but still
reserves its interval, matching a live process that cannot reset positions at a
research boundary. For example, a 2023 source-day candidate whose scheduled
entry/exit fall in 2024 may reserve those timestamps without opening any 2024
source or execution value.

An `accepted_event(split)` is a reserved event whose signal, decision, entry,
all 576 held bars, and exit are contained in that declared split. The master
clock is reserved across 2021-2023 before train and 2023 are sliced; later
stages extend the exact same reserved clock without a split reset. Unless a
control is explicitly described as using the primary clock, it builds and
reserves its own master clock independently. Every support count, side count,
region share, month share, quarter count, Jaccard set, and primary exposure
series below uses accepted events for the named split. Calendar buckets use
`entry_time` in UTC.

Direction is fixed before incidence:

```text
state = +1  # broad local-fiat weakness / USDT strength
side  = -1  # SHORT BTCUSDT perpetual

state = -1  # broad local-fiat strength / USDT easing
side  = +1  # LONG BTCUSDT perpetual
```

Execution is:

- decision boundary: not before `d+1 00:00 UTC`, after all four source candles
  close; the signal is valid only if all four pass validation before the live
  receipt deadline above;
- entry: BTCUSDT USD-M perpetual open at `d+1 00:05 UTC`, after one complete
  five-minute computation/transmission bar;
- exit: exactly 576 five-minute bars / 48 UTC hours after entry;
- fixed exposure: `0.5x` account notional;
- no stop, take-profit, trailing rule, dynamic sizing, leverage grid, or
  position overlap;
- chronological reservation, accepting a new entry only at or after the prior
  scheduled exit; and
- signal, entry, every held bar, and exit must be contained in the declared
  split.

## Outcome-blind source-support gate

The first source build is physically capped at 2023-12-31. It may read the four
completed Spot source closes and the frozen source-only comparator clocks. It
may not read USD-M execution OHLC, funding, future return, trade PnL, equity,
CAGR, MDD, or any 2024+ source value.

RFXS-576 is rejected without an outcome evaluator unless every condition
passes:

| Gate | 2021-2022 train | 2023 selection |
|---|---:|---:|
| accepted events | at least 50 | at least 24 |
| calendar subperiod | 2021 at least 18; 2022 at least 24 | each half at least 10 |
| side support | at least 15 long and 15 short | at least 8 long and 8 short |
| quarter support | at least 4 in 2021Q2-Q4 and every 2022 quarter | at least 4 in every quarter |
| maximum entry-month share | at most 20% | at most 25% |
| each region contributes at threshold | at least 40% of events | at least 40% of events |

A region contributes when, on the accepted event source day, its own z-score is
`>= +1` in a positive event or `<= -1` in a negative event. No prior-day
crossing by that individual region is required. At least two contributors are
therefore guaranteed per event; the 40% floor prevents one region from being
structurally irrelevant over a split.

The source stage must also pass these novelty gates:

- absolute Spearman correlation of `common_z` with the independently
  prior-180 robust-z-scored completed BTCUSDT daily return: at most `0.50`,
  separately in train and 2023;
- against FQPR-3 on the common 2021-2023 horizon: exact-entry Jaccard at most
  `0.20` and signed five-minute occupied-exposure correlation at most `0.40`;
- against SDDR-12 on its common 2023 horizon: exact-entry Jaccard at most `0.10`
  and signed occupied-exposure correlation at most `0.40`.

Spearman uses every source day inside the named split on which both
`common_z` and the independently constructed BTC return z-score are finite,
with average ranks for ties and ordinary Pearson correlation of those ranks.
Fewer than three paired days, a constant ranked vector, or a non-finite result
fails the gate.

Comparator rows are restricted to `clock_name == "primary"` for FQPR and
`candidate == "SDDR-12" AND control == "primary"` for SDDR. FQPR's comparison
horizon is `[2021-01-01 00:00, 2024-01-01 00:00) UTC`; SDDR's is
`[2023-09-01 00:00, 2024-01-01 00:00) UTC`. Each strategy keeps only accepted
intervals fully contained in the relevant horizon. Exact-entry Jaccard is the
intersection divided by the union of UTC entry-timestamp sets; an empty union
fails. Signed exposure is `side` on each five-minute timestamp in the
half-open interval `[entry, exit)` and zero otherwise, aligned over every
five-minute timestamp in the complete comparison horizon. Correlation is
ordinary Pearson correlation; a constant series or non-finite result fails.

The comparator clocks are frozen as:

- `results/fiat_quote_participation_rotation_clocks_2026-07-17.csv`, SHA-256
  `54a70cce565d4f1727d095707471235f01345b94179a6c37df9f4c37d1a458a2`;
- `data/stablecoin_denominator_dislocation_clocks_2023.csv.gz`, SHA-256
  `eaf2d6c187af9855e76474d2951fcdc12267174980a72649b73d068982ca8c69`.

Parent H.4.1/FLCC clock overlap may be reported as a macro diagnostic but is not
a source gate. Support failure retires this exact singleton without opening an
outcome.

## Frozen source-only controls

Each control is built independently with the same decision latency, 48-hour
hold, onset logic, split containment, and non-overlap unless stated otherwise:

1. `eur_only`: `+1` when `z_EUR >= +1`, `-1` when `z_EUR <= -1`, and zero
   otherwise, followed by the primary candidate-event onset and contrarian map.
2. `try_only`: the same exact state using only `z_TRY`.
3. `brl_only`: the same exact state using only `z_BRL`.
4. `three_book_sign_only`: `+1` iff all three finite z-scores are strictly
   positive, `-1` iff all three are strictly negative, and zero otherwise;
   apply the primary onset and contrarian map without a magnitude threshold.
5. `btc_return_shadow`: robust-z-score the completed BTCUSDT daily return with
   its own prior 180-day median/MAD, use the same `+/-1` onset and contrarian
   side, and schedule independently.
6. `stale_one_day`: delay the complete primary state and side by exactly one
   source day.
7. `direction_flip`: exact primary entries with the opposite side.
8. `deterministic_random_side`: exact primary entries; side comes from the
   first byte of
   `SHA256("RFXS-576-random-side-20260720|" + entry_time)`, below 128 long and
   otherwise short.

Controls may falsify the mechanism but may not replace primary after source or
outcome results are observed. The source artifact reports each control's raw
candidate count, accepted count, side count, calendar concentration, exact
entry overlap, and signed-exposure correlation with primary; these diagnostics
do not relax any primary support gate.

## Later strict evaluator contract

Only a complete source-support pass may authorize a separate, tested,
committed, hash-frozen strict evaluator. It must open stages sequentially and
stop at the first failure:

1. train: full calendar 2021-01-01 through 2022-12-31, with 2020-10 history
   used only for causal warmup;
2. test: calendar 2023;
3. evaluation: calendar 2024;
4. forward: calendar 2025; and
5. final: 2026-01-01 through 2026-06-30.

The frozen pre-2024 execution sources are:

- official Binance USD-M monthly `BTCUSDT` five-minute klines, materialized as
  `data/binance_um_kline_reference_btc_2020_2023/`
  `BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz`, SHA-256
  `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`,
  bound by `build_manifest.json`, SHA-256
  `c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e`;
- official Binance USD-M funding/mark records materialized as
  `data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz`, SHA-256
  `3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6`,
  bound by
  `results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json`,
  SHA-256
  `a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b`.

The evaluator must fill at the five-minute `open` at `entry_time`, hold the
576 bars whose timestamps are in `[entry_time, exit_time)`, and close at the
`open` at `exit_time`. Base transaction cost is `6 bp/notional/side`; stress
cost is `10 bp/notional/side`. Funding uses each exact recorded settlement
timestamp, is eligible iff `entry_time <= funding_time < exit_time`, and is
assigned to its containing five-minute bar without changing eligibility.
Later-stage source and execution archives must use the same official source and
parser contracts and must be checksum/hash-frozen before that stage simulates;
they cannot be fetched or hashed before the preceding stage passes.

Every opened stage must report absolute return, full-calendar CAGR, strict
held-path MDD, CAGR/MDD, trades, long/short counts, gross mean move, weekly
clustered sign-flip p-value, and base/stress cost results. Full-calendar CAGR
uses the declared split start and exclusive end regardless of first or last
trade. Gross mean move is the arithmetic mean of
`side * (exit_open / entry_open - 1)` before leverage, funding, and costs.

The frozen primary gate is positive absolute return, base CAGR/MDD at least
`3.0`, stress CAGR/MDD at least `2.5`, base and stress strict MDD at most `15%`,
at least 20 accepted trades per full calendar year, mean gross move at least
20 bp, and weekly-cluster one-sided sign-flip `p <= 0.10` using 20,000 draws and
seed `20260720`. The weekly test clusters base-cost net account returns by the
entry timestamp's ISO year/week and independently flips cluster signs; an
insufficient or non-finite test fails.

Stability is also mandatory: calendar 2021 and 2022 and each of their calendar
halves must be positive; each full-year later stage must have positive H1 and
H2; 2026-H1 must have positive Q1 and Q2. Each required train half needs at
least five trades, each later full-year half at least eight, and each 2026
quarter at least five. A one-extra-five-minute-bar entry-delay replay, with the
same 576-bar hold and all costs/funding shifted to its actual interval, must
remain positive and have base CAGR/MDD at least `2.5`.

Strict MDD includes idle/pre-entry equity and its global high-water mark, entry
cost, exact funding, every held five-minute OHLC path ordered
favorable-before-adverse, hypothetical liquidation/virtual exit cost at each
adverse observation, and actual exit cost. It is global within each declared
opened stage and never resets by trade, month, half, year, or a reported stage
subperiod; a subperiod diagnostic inherits the parent stage equity path rather
than manufacturing a fresh high-water mark.

After a primary stage passes its standalone gates, the same stage is evaluated
for every frozen control with identical costs, funding, and path accounting.
The primary stage CAGR/MDD must exceed each finite `eur_only`, `try_only`,
`brl_only`, `three_book_sign_only`, `btc_return_shadow`, and `stale_one_day`
ratio by at least `0.25`. Neither `direction_flip` nor
`deterministic_random_side` may itself satisfy all standalone primary gates.
Failure rejects the claimed regional-concordance mechanism; no control can be
promoted in its place.

No 2024+ ZIP, source row, execution row, or funding value may be opened before
train and 2023 both pass. The earlier checksum-existence probe is explicitly
non-normative metadata, not an opened source archive. No later stage may select
a threshold, side, country, window, hold, cost, or control.

## LLM/RL boundary

RFXS first tests whether the deterministic cross-rate state has economic edge.
An LLM does not calculate these numbers or manufacture a side. Only after a
strict pass through 2023 may a separate train-only RLLM stage encode the three
country states as symbolic text and learn abstention/risk routing while keeping
the frozen direction and execution clock. Failure of the deterministic base is
terminal; Gemma, LoRA, RL, leverage, or prompt tuning may not repair it.

## No-repair rule

Any source-support, train, test, evaluation, forward, or final failure retires
RFXS-576 without changing symbols, sign, normalization window, threshold,
onset, entry delay, hold, leverage, support floors, costs, or performance
gates. A different fiat set, continuation direction, intraday horizon, raw FX
input, or learned country weight is a new preregistered mechanism, not a repair.

The branch has broad prior BTC research exposure, FQPR outcomes, and known
macro-feature history. This sequence can establish only a candidate-level
frozen claim, not a pristine global clean-room claim.
