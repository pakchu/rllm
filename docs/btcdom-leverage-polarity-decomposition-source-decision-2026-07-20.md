# BTCDOM leverage-polarity decomposition source decision — 2026-07-20

## Decision

Proceed with one source-only candidate, **DLPD-12 (Dominance Leverage
Polarity Decomposition, twelve-hour hold)**.  The candidate does **not** use
the BTCDOM index level or return.  Those fields would mostly compress the
BTC-versus-alt relative-return and breadth information already studied by
CLD-72 and the cross-asset families.

DLPD instead reads the completed-hour premium-index closes of two USD-M
perpetuals:

- `BTCUSDT`, an absolute BTC contract; and
- `BTCDOMUSDT`, a synthetic relative BTC-versus-alt contract.

The hypothesis is that opposite extreme premium pressure separates a common
crypto risk move from relative alt allocation:

- unusually rich `BTCUSDT` and unusually cheap `BTCDOMUSDT` means broad
  risk-on demand in which BTC participates but alts are preferred: tentatively
  `LONG BTCUSDT`;
- unusually cheap `BTCUSDT` and unusually rich `BTCDOMUSDT` means broad
  risk-off supply in which BTC is preferred only relative to alts: tentatively
  `SHORT BTCUSDT`.

This is a hypothesis about **leveraged demand decomposition**, not a claim that
BTCDOM direction identifies BTC direction.  BTC direction comes from the
absolute BTC premium leg; the dominance premium is an incremental relative
state requirement.

No BTC execution price, return, excursion, funding cash flow, PnL, equity,
CAGR, MDD, label, or post-2023 source row was opened in making this decision.

## Why this is not the rejected BTCDOM price shortcut

Binance describes BTCDOM as an uncapped BTC-dominance proxy built from BTC
priced in a changing basket of major non-stablecoin cryptocurrencies.  Its
level can rise while BTCUSDT falls and fall while BTCUSDT rises.  Therefore:

- BTCDOM level/return alone is underidentified for absolute BTC direction and
  is rejected before outcomes;
- the candidate retains only normalized perpetual premium pressure, not the
  index level, constituent prices, mark price, or contract return;
- both premium legs must be extreme in opposite directions, so neither a
  BTC-premium tail nor a dominance-premium tail can activate the primary clock
  alone.

The source is still adjacent to earlier BTC premium-path and six-alt derivative
crowding work.  Source-clock novelty against those families is therefore a
mandatory gate, not an assumed property.

## Official source and live parity

BTCDOMUSDT launched in 2021 and was `TRADING` as a USD-M `PERPETUAL` with
`underlyingType=INDEX` when checked on 2026-07-20.  The official composite-index
endpoint returned twenty current BTC/alt components.  Official live endpoints
returned finalizable `BTCUSDT` and `BTCDOMUSDT` premium-index klines, while the
official monthly archive exposes adjacent checksum files for both symbols.

Sources:

- BTCDOM launch announcement:
  <https://www.binance.com/en/support/announcement/detail/57333a98450342cd99a5552a1865af65>
- BTCDOM FAQ:
  <https://www.binance.com/en/support/faq/detail/e3b1ab97a3e24df4b0e41a469ccf7a21>
- USD-M composite-index and premium-kline API documentation:
  <https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data>
- live composite index metadata:
  <https://fapi.binance.com/fapi/v1/indexInfo?symbol=BTCDOMUSDT>
- live premium-index klines:
  <https://fapi.binance.com/fapi/v1/premiumIndexKlines?symbol=BTCDOMUSDT&interval=1h&limit=1>
- official monthly premium-index archive:
  <https://data.binance.vision/?prefix=data%2Ffutures%2Fum%2Fmonthly%2FpremiumIndexKlines%2F>

Historical archive existence was checked for both legs from July 2021 through
December 2023 and for BTCDOMUSDT through June 2026.  The initial physical build
will remain end-exclusive at `2024-01-01`; later predictor rows stay sealed.

## Frozen source contract

- Symbols: exactly `BTCUSDT`, `BTCDOMUSDT`.
- Source: official Binance Vision USD-M monthly `premiumIndexKlines`.
- Interval: one completed UTC hour.
- Physical prefix: `[2021-07-02 00:00, 2024-01-01 00:00)` UTC.
- Every ZIP must match its adjacent published `.CHECKSUM`, and the published
  hashes must first be frozen in a committed checksum inventory.
- Millisecond and later microsecond timestamps are normalized only after mixed
  or non-aligned units are rejected.
- Each row is available at `close_time + 1.001 seconds`; missing rows remain on
  the exact hourly grid with an explicit validity flag and missing values.
- The combined row is valid only when both symbols have the exact same hour and
  both source rows are valid.
- Retained values are limited to timestamps, validity, and the two premium
  closes.  Premium OHLC paths, volume/count placeholders, contract/index
  prices, funding and all market outcomes are discarded.
- Current BTCDOM constituent weights are audit metadata only.  They are not
  backfilled into history or treated as if they were historical weights.

## Frozen DLPD-12 source clock

For each valid leg independently:

1. estimate the strictly-prior rolling median and IQR over the latest 720
   valid completed hours, requiring at least 672 observations;
2. set robust scale to `IQR / 1.349`; zero/non-finite scale disables the leg;
3. standardize the current premium close against those prior-only statistics;
4. activate only when the z-scores have opposite signs and both absolute
   z-scores are at least `1.0`;
5. set side to `sign(BTCUSDT premium z-score)`;
6. signal only on a false-to-true onset;
7. decide at the exact hour boundary after the source closes, leave the next
   five-minute bucket empty, enter at `hour + 5m`, and hold twelve hours;
8. enforce global non-overlap and keep events inside the declared calendar
   split.

There is no threshold, sign, smoothing, hold, latency, stop, take-profit,
regime, price, model, or LLM grid after this decision.

## Disclosed source-only feasibility inspection

Before freezing the singleton, an outcome-blind scratch probe inspected only
the two hourly premium closes.  With the exact rule above and a twelve-hour
non-overlap contract it found 237 events in 2022 and 183 in 2023 (122 long / 61
short in 2023).  This count inspection selected the singleton from a bounded
source-only check of raw/4h/8h premium averaging and absolute z thresholds
`{0.75, 1.0, 1.25, 1.5}`.  No market or funding outcome was loaded.  The probe
is disclosed so the threshold is not misrepresented as theory-only.

## Source-only controls and gates

Controls are frozen before the production source is built:

- `btc_only_tail`: the BTC premium threshold without BTCDOM confirmation;
- `dom_only_mirror`: the dominance premium threshold, mapped to the opposite
  side, without BTC confirmation;
- `same_sign`: both legs extreme in the same direction;
- `stale_btc_1h`: prior-hour BTC z-score with current dominance z-score;
- `stale_dom_1h`: current BTC z-score with prior-hour dominance z-score.

Primary support must pass all of the following in both 2022 and 2023:

- at least 120 non-overlapping events per year;
- each side at least 25% of annual events;
- every calendar quarter at least 20 events;
- no month above 20% of annual events.

The 2023 primary entry clock must also have exact Jaccard at most `0.10` and
maximum bidirectional containment within one hour at most `0.35` against the
frozen primary clocks of PSR-30/6, PCBR-12, OPDR-24, CLD-72 and FCIR-12.  A
failure rejects DLPD before any outcome is opened; the disclosed threshold or
gate may not be repaired.

## Conditional outcome protocol

Only a complete source/support pass may authorize a separately committed strict
evaluator.  The intended sequential windows are train 2022, test 2023, eval
2024-2025 and final 2026H1.  Each later source/outcome window remains sealed
until every earlier gate passes.  The evaluator must use next-open execution,
full-calendar CAGR including idle cash, exact funding, implementation-cost
stress, global/pre-entry high-water strict MDD, held OHLC paths, contained
subperiods, direction/leg/staleness controls, and clustered significance.

Source support is not profitability evidence.  The candidate is retired at the
first failed stage without direction flip, threshold repair, or hold search.

## Known risks

- BTCDOM constituent membership and weights change; the archive is an
  exchange-defined evolving index, not a constant basket.
- Premium pressure can reflect arbitrage and funding mechanics rather than a
  directional belief.  Both singleton and component controls are required.
- BTCDOM live collection is not yet wired into the repository DB.  Historical
  success cannot promote the strategy until a fail-closed final-kline collector
  reproduces this exact hourly contract.
- Source-clock novelty does not prove PnL independence; portfolio correlation
  and marginal contribution are later promotion gates if strict outcomes pass.
