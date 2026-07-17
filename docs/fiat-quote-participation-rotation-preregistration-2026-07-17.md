# FQPR-3 preregistration — Fiat-Quote Participation Rotation

## Claim boundary

**No FQPR-3 post-entry outcome has been opened.** This document freezes one
candidate, one direction, one support-only threshold grid, one execution clock,
and all later rejection gates before any entry-to-exit return is calculated.

- policy: `FQPR-3`
- source: official Binance Spot daily flows for `BTCUSDT`, `BTCEUR`, `BTCTRY`,
  and `BTCBRL`
- action: fixed long BTCUSDT USD-M perpetual
- entry: 00:05 UTC after the completed fiat-quote-flow source day
- hold: 864 five-minute bars / 72 hours
- leverage: 0.5x
- base cost: 6 bp/notional/side
- stress cost: 10 bp/notional/side
- 2024, 2025, and 2026 YTD remain sealed

Historical BTC periods have been seen by unrelated repository research, so
pre-2024 is not a global clean room. The exact FQPR clock and its post-entry
returns remain unopened. A source-only density preflight did see event counts:
it retired a stricter draft that produced only 12 train clocks at `Q=0.65` and
then inspected source-only counts, composition, and control overlap for the
revised definition through 2023. Therefore 2023 is **support-seen but
outcome-sealed**; it may reject the frozen train-selected `Q`, but cannot
reselect or repair it. No preflight read OHLC, returns, funding, CAGR, or
drawdown. The source panel is frozen in commit
`9b21dedb51818d649fc4bdd68207190f836bd176`.

## Economic mechanism and orthogonal axis

FQPR asks whether **simultaneous BTC-denominated participation-share growth in
multiple fiat-quote books**, accompanied by relative aggressive buying, is a
stronger demand-rotation signal than activity in BTCUSDT alone. Pair activity
does not identify trader geography or prove an external fiat deposit, so the
claim is deliberately limited to Binance fiat-quote books. It does not use BTC
price direction to decide that demand exists.

For each of EUR, TRY, and BRL, the same-day BTC base volume and trade count are
scaled by BTCUSDT's same-day values. This removes quote-currency units without
using FX rates or quote prices. Each book's volume-share and ticket-share ranks
are averaged into one weak participation score. At least two scores must be
elevated, and the median fiat-book taker-buy-odds gap versus BTCUSDT must be
positive.

This differs from Kimchi/FX signals, which use one KRW price premium and FX
state, and from BTCUSDT spot/perp flow signals, which observe one global market.
FQPR excludes KRW, all prices, FX, derivatives, REX, and existing-alpha states.

## Frozen causal features

For completed source day `d`, fiat-quote book `r`, and BTCUSDT reference `u`:

- `V_r[d] = log(base_volume_btc_r[d] / base_volume_btc_u[d])`
- `N_r[d] = log(trade_count_r[d] / trade_count_u[d])`
- `O_r[d] = log(taker_buy_base_r[d] / taker_sell_base_r[d])`
- `O_u[d] = log(taker_buy_base_u[d] / taker_sell_base_u[d])`

`R_V_r[d]` and `R_N_r[d]` are mid-rank empirical CDFs of the current value
against **exactly** the prior 180 complete source days `d-180..d-1`. The current
day is excluded. There is no expanding fallback, imputation, or stale carry.

The exact mid-rank is
`(count(prior < current) + 0.5 * count(prior == current)) / 180`.
Zero taker-buy or taker-sell volume is unavailable; no pseudocount is allowed.

For the only support-varying parameter `Q`:

- `P_r[d] = (R_V_r[d] + R_N_r[d]) / 2`;
- `E_r[d] = O_r[d] - O_u[d]`;
- `F[d] = median(E_EUR[d], E_TRY[d], E_BRL[d])`;
- `B[d] = sum_r 1[P_r[d] >= Q]`.

The fixed grid is `{0.50, 0.55, 0.60, 0.65, 0.70}`. A setup occurs when
`B[d] >= 2 AND F[d] > 0` first transitions from false to true. Continuous true
runs produce one episode.

The day-`d` aggregate is available only after `23:59:59.999 UTC`. FQPR waits
through the complete `00:00-00:05` availability bucket and enters at the
`d+1 00:05 UTC` open. Any use of day `d` during day `d`, or any rank including
day `d` itself, is leakage.

## Outcome-blind support gate

Using only 2021-2022 source values and no executable OHLC, choose the **highest**
`Q` satisfying all support, composition, and novelty gates. Then expose 2023
support for that one frozen `Q`. The 2023 support check can pass or reject it;
it cannot choose a fallback.

Required non-overlapping entries:

- train: at least 40;
- 2021 after the 180-day warm-up: at least 20;
- 2022: at least 18;
- 2023: at least 20;
- each 2023 half: at least 8.

Separately in train and 2023:

- no UTC month may exceed 25% of entries;
- each fiat-quote book must participate in at least 30% of entries;
- no one exact participating-book set may exceed 80% of entries.

Entry-clock Jaccard must not exceed the machine-readable preregistration's
limits against the frozen no-ticket, no-taker, volume-only, flow-only,
single-book, BTCUSDT-only, BTCUSDT-suppression, absolute-book-participation, and
one-day-delay controls. Every control reserves its own globally non-overlapping
clock before split slicing.

Clock reservation is global over pre-2024 history. A split keeps only trades
whose signal, entry, and exit are inside that split. Strictly past rank history
may precede the split start (for example 2022 history supporting a January 2023
signal); it may never precede the source start or include the signal day.

Any support failure rejects FQPR-3 without loading strategy returns.

## Frozen falsification controls

1. exact short flip on the primary clock;
2. no-ticket breadth, retaining volume rank and median relative taker pressure;
3. no-taker breadth, retaining the averaged volume/ticket participation score;
4. volume-rank breadth alone;
5. taker-flow breadth alone;
6. one clock for each single fiat-quote book;
7. BTCUSDT's own raw volume/trade rank plus positive taker buy odds;
8. a low-BTCUSDT-participation denominator-suppression clock;
9. absolute fiat-book volume/trade participation breadth;
10. primary signal delayed one UTC day;
11. fixed-seed random side on the primary clock.

A control can falsify the mechanism but cannot replace FQPR after outcomes are
opened.

## Strict staged evaluation

Stage 1 may parse execution OHLC and funding only through 2022-12-31. Stage 2
may open 2023 only if unchanged Stage 1 passes every gate. Every result table
must show **absolute return, full-clock CAGR, strict MDD, CAGR/strict-MDD, and
trade count**.

The primary candidate must, in both train and 2023:

- have positive absolute return;
- achieve CAGR/strict-MDD at least 3;
- keep strict MDD at or below 15%;
- pass weekly-cluster one-sided sign-flip `p <= 0.10` using 20,000 draws and
  seed `20260717`;
- retain positive absolute return at 10 bp/notional/side;
- average at least 35 bp gross underlying move per trade.

Train requires at least 40 executable trades and 2023 at least 20. The frozen
2021/2022 subperiods and both 2023 halves must each be positive with their
same support minimum counts. On Stage 1, the primary train CAGR/MDD must be
strictly greater than each frozen mechanism control. After Stage 2, the minimum
of the primary's train and 2023 CAGR/MDD must be strictly greater than the same
minimum for every control; ties reject. A one-day-delay or random-side control
that fully qualifies rejects the claim. Controls can never replace the
singleton candidate.

Strict MDD uses the global/pre-entry high-water, entry and hypothetical
liquidation costs, favorable-before-adverse held OHLC, exact realized funding,
and scheduled exit cost. CAGR uses the full wall-clock split, including warm-up
and idle periods.

## Orthogonality and RLLM boundary

Only after standalone passage may FQPR be compared with existing alphas. The
comparator universe is frozen to the hashed 381-sleeve deduplication audit, its
family-capped portfolio, the 2026-07-16 added-alpha shadow portfolio, and the
current live configuration listed in the machine-readable preregistration.
Exact PnL duplicates are collapsed; FQPR must pass the entry/position/PnL limits
against every canonical family representative and synchronized selected
portfolio, then improve a synchronized portfolio marginally. Source novelty
alone is not sufficient.

Only after deterministic standalone and orthogonality passage may a compact
RLLM receive symbolic fiat-quote ranks, taker odds, breadth, participating-book set,
current position, and time to exit. It may abstain or size the fixed long. It
may not reverse the side or redesign the base event.

The exact candidate is retired on any support or staged-performance failure.
Direction, thresholds, symbols, features, delay, hold, and gates cannot be
repaired after outcomes open.
