# CCLH v1 preregistration — 2026-07-14

## Status and claim boundary

- **CCLH returns have not been opened.** This stage may inspect only source
  integrity, causal geometry, state/event incidence, side balance, temporal
  concentration, overlap with the already frozen CLV clock, and executable
  support.
- physically available experiment range: calendar 2023 UTC only
- support artifact:
  `results/cross_collateral_liquidity_hysteresis_support_2026-07-14.json`
- sealed policy windows: full 2024, full 2025, and 2026 YTD
- no profitability claim is made here

CCLH is not a direction or threshold repair of CLV. CLV reacted to a completed
price shock, required taker-flow alignment, compared a one-hour change in
one-percent/five-percent book geometry, and traded every qualifying vacuum
impulse for one hour. CCLH removes price return and taker flow from the signal,
uses all ten cumulative depth levels, confirms a persistent cross-contract
geometry state, emits only on a state transition, and holds for 12 hours.

## Economic hypothesis

Binance USD-M `BTCUSDT` and COIN-M `BTCUSD_PERP` reference BTC but differ in
collateral and payoff structure. Raw depth units are not comparable. CCLH first
reduces each contract to two dimensionless full-depth summaries:

1. **pressure** — average log bid/ask depth across all five cumulative
   distances;
2. **elasticity** — whether cumulative asks grow faster with distance than
   cumulative bids, expressed as a difference of log-log slopes.

When coin-margined pressure and elasticity are both persistently high relative
to USD-M, COIN-M is unusually bid-heavy and its asks are relatively hollow
nearer the book. CCLH interprets that durable collateral-specific state as
bullish. Jointly low values are interpreted as bearish. A 12-bar confirmation
and hysteretic exit are intended to reject transient book noise and reduce
turnover; they do not guarantee that displayed liquidity is firm.

Primary sources define data and book semantics, not profitability:

- Binance public data repository:
  <https://github.com/binance/binance-public-data>
- USD-M `BTCUSDT` `bookDepth` archive:
  <https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/>
- COIN-M `BTCUSD_PERP` `bookDepth` archive:
  <https://data.binance.vision/data/futures/cm/daily/bookDepth/BTCUSD_PERP/>
- Binance COIN-M local-order-book sequencing semantics:
  <https://developers.binance.com/legacy-docs/derivatives/coin-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly>
- Cont, Kukanov, and Stoikov on order-flow imbalance and depth:
  <https://arxiv.org/abs/1011.6402>
- Gould and Bonart on queue imbalance and subsequent price movement:
  <https://arxiv.org/abs/1512.03492>

The papers motivate testing book geometry as state information. They do not
establish CCLH, cross-collateral hysteresis, or a monetizable edge.

## Physically sealed data

| source | range | rows / completeness | SHA256 |
|---|---|---:|---|
| cross-collateral depth panel | 2023-01-01 through 2023-12-31 | 105,120 rows; 101,649 joint-complete | `53e16cf71581f03c7b1cc3da6a13222923ce68aa9e869d89f02078221bb4eee4` |
| Binance BTCUSDT 5m execution market | 2020-01-01 through 2023-12-31 | 420,768 rows | `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d` |

The depth builder is physically bounded to 2023 and checksum-verifies every
available official archive. Each accepted contract bar is the independent
median of at least eight complete snapshots in `[bar open, bar open + 5m)`,
with first snapshot no later than 60 seconds and last no earlier than 240
seconds. Every snapshot must contain all cumulative
`-5..-1,+1..+5` percentage levels.

There is no interpolation, fill, nearest-time join, or compressed event clock.
Missing depth makes the current state unavailable. It may causally weaken and
eventually end an active state, but a source gap after trade entry cannot
cancel that already scheduled trade. Hash, schema, timestamp, and exact-grid
checks fail closed.

## Frozen full-depth geometry

For completed bar `t`, contract `v in {um,cm}`, and distance
`k in {1,2,3,4,5}`, let `B_v,k,t` and `A_v,k,t` be native cumulative bid and ask
depth at minus/plus `k` percent. Define within each contract:

`P_v,t = (1/5) × sum_k log(B_v,k,t / A_v,k,t)`

Let `betaB_v,t` be the OLS slope of `log(B_v,k,t)` on `log(k)`, and
`betaA_v,t` the corresponding ask slope. Define:

`E_v,t = betaA_v,t - betaB_v,t`.

`E>0` means cumulative ask depth grows faster with distance than bid depth, so
asks are relatively less concentrated near the book than bids. Only after
forming these dimensionless within-contract quantities does CCLH compare:

`GP_t = P_cm,t - P_um,t`

`GE_t = E_cm,t - E_um,t`.

Each cross-contract series is independently standardized without the current
observation:

1. shift the raw series one five-minute bar;
2. compute rolling median over 8,640 calendar bars, requiring 2,016 prior
   joint-complete values;
3. compute recursive rolling median of prior absolute deviations from their
   corresponding lagged centers;
4. divide current deviation by `1.4826 × MAD`;
5. treat zero scale as unavailable and clip finite values to `[-12,12]`.

Call the causal scores `ZP_t` and `ZE_t`.

## Frozen hysteresis and event clock

The fixed provisional state is:

- `q_t=+1` when `ZP_t>=+0.50` and `ZE_t>=+0.50`;
- `q_t=-1` when `ZP_t<=-0.50` and `ZE_t<=-0.50`;
- `q_t=0` otherwise or when the source is unavailable.

When flat, 12 consecutive identical nonzero `q` bars activate that state and
emit one event. While active, no refresh event is emitted. The state exits
after 12 consecutive bars where either score has `abs(Z)<0.25` or is
unavailable. Twelve consecutive opposite provisional bars instead flip the
active state and emit one opposite event. Entry/exit thresholds, confirmation
lengths, and state-machine ordering are fixed; no threshold grid is searched.

Action and execution clock:

- bullish transition: side `+1`;
- bearish transition: side `-1`;
- signal becomes available only after bar `t` completes;
- enter at the open of bar `t+1`;
- exit at the open 144 bars later (12 hours);
- positions do not overlap; a new one may enter at the prior scheduled exit;
- signal, entry, full held path, and scheduled exit must remain in the split.

A boundary-crossing trade is skipped, never force-closed. A failed result may
not be repaired under CCLH v1 by changing state direction, thresholds,
confirmation, hold, depth distances, adding price/flow gates, or adding a stop.

## Support-only qualification

There is no support-parameter search. Before any return evaluator is written,
CCLH must have:

1. at least 120 non-overlapping 2023 events;
2. at least 45 in each half;
3. at least 20 in each quarter;
4. long and short each at least 35%;
5. no quarter contributing more than 40% of events;
6. tolerant event Jaccard against frozen CLV no greater than 0.35.

For the independence diagnostic, CCLH and CLV non-overlapping signal positions
are sorted and greedily matched one-to-one within plus/minus 12 five-minute
bars. Jaccard is `matches/(N_CCLH + N_CLV - matches)`. It measures clock
similarity only; it does not inspect a price or outcome.

No return, future high/low, PnL, CAGR, MDD, win rate, or label may be inspected.
Failure rejects CCLH v1 before returns and permits no support repair.

Outcome-blind support passes without a parameter grid:

| measure | frozen value | floor / ceiling |
|---|---:|---:|
| raw state transitions | 201 | diagnostic |
| non-overlapping events | 167 | at least 120 |
| 2023 H1 / H2 | 71 / 96 | at least 45 each |
| Q1 / Q2 / Q3 / Q4 | 33 / 38 / 48 / 48 | at least 20 each |
| long / short | 88 / 79 | at least 35% each |
| largest quarter share | 28.74% | at most 40% |
| CLV matches within 12 bars | 34 of 167 vs 521 | diagnostic |
| tolerant CLV Jaccard | 0.0520 | at most 0.35 |

There are 11,380 provisional bullish and 9,721 provisional bearish rows before
confirmation. The executable clock is balanced, distributed across every
quarter, and has low temporal overlap with CLV. No return or future path was
opened. All parameters are now frozen and no support repair is allowed.

## Frozen 2023 research evaluation

Calendar 2023 is not a repository-wide clean room: prior work has inspected
BTC returns and the rejected CLV clock in this year. CCLH's 2023 evaluation is
therefore a development qualification, not final OOS evidence. Only a passing
frozen mechanism may cause a new 2024 depth file to be built and a true 2024
test to be opened.

The evaluator must be committed and hash-frozen separately before any CCLH
return is opened.

### Windows

1. train: 2023-01-01 through 2023-06-30;
2. select: 2023-07-01 through 2023-12-31;
3. stability: Q1, Q2, Q3, and Q4;
4. sealed test: full 2024;
5. sealed eval: full 2025;
6. untouched forward report: 2026 YTD with explicit end time.

### Exact execution and strict risk

- leverage `L=0.5`;
- fee `5 bp` plus slippage `1 bp` per notional side;
- per-side account cost `c=L×0.0006=0.0003`;
- signed underlying return `r=side×(exit_open/entry_open-1)`;
- exact multiplier `(1-c)×(1+Lr)×(1-c)`;
- full-clock CAGR counts every idle interval;
- strict MDD applies entry cost, favorable held extreme first, adverse held
  extreme second, scheduled-open return, then exit cost; the exit bar's later
  high/low is excluded;
- every table reports **absolute return, CAGR, strict MDD, CAGR/strict-MDD,
  and trades**, plus side counts.

### Frozen controls

The base opportunity clock is reserved before action controls change a side;
abstention cannot release a later candidate:

1. frozen CCLH side;
2. exact direction reversal;
3. always long;
4. always short;
5. sign permutation with `numpy.random.default_rng(20260714)`;
6. completed one-hour price-momentum side on the same event clock.

Additional diagnostics, not replacement candidates, are CCLH after abstaining
on events matched to CLV within plus/minus 12 bars, and causal USD-M-only,
COIN-M-only, pressure-only, and elasticity-only state variants. Different-clock
structural diagnostics cannot win selection or replace CCLH v1.

### Statistical test and qualification

Net account trade returns are clustered by UTC entry week. Each split uses
100,000 independent weekly Rademacher sign flips with seed `20260714`; the
one-sided p-value is `(1 + count(permuted >= observed))/100001`. Empty results
return `p=1.0`.

CCLH advances to a 2024 physical data build only if:

- train and select each have positive absolute return;
- train and select each have CAGR/strict-MDD at least `3.0` and strict MDD at
  most `15%`;
- each quarter has positive absolute return and at least 20 trades;
- train and select each have at least 45 trades and weekly p-value below
  `0.10`;
- base CCLH beats reverse, always-long, always-short, sign-permutation, and
  price-momentum controls on `min(train CAGR/MDD, select CAGR/MDD)`.

Any failure rejects CCLH v1 without repair and leaves 2024+ closed. If every
gate passes, the exact data validation and state machine are extended to a
physically frozen 2024 panel before its returns are opened. Test failure cannot
be repaired on 2024; 2025 and 2026 remain sealed.

## RLLM boundary

No Gemma/RL policy is attached during 2023 mechanism development or the 2024
deterministic test. Only after deterministic CCLH passes both may one compact
Gemma policy train on causal 2023 symbolic state and use 2024 for selection,
leaving 2025 as untouched evaluation. Inputs may include pressure/elasticity
score buckets, active-state age, source freshness, current position, and time
to exit. The policy may abstain or downsize; it may not reverse the base side,
move the event clock, change the hold, receive identifiers/future availability,
or redesign the alpha from sealed outcomes.
