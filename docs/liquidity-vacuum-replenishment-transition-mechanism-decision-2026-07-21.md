# LVRT-72 — Liquidity Vacuum Replenishment Transition mechanism decision

## Decision and evidence boundary

The next standalone BTC candidate is **LVRT-72**, a rare aggregate-trade
liquidity-vacuum-to-replenishment transition with a fixed six-hour hold.

LVRT does not trade a static flow tail. It waits for an unusually bursty,
concentrated, low-effective-count, one-sided aggregate-trade state and then for
the first later bar in which event participation refills, concentration falls,
and directional event flow reverses. The hypothesis is that the first state is
a one-sided liquidity vacuum and the second marks metaorder exhaustion plus
opposite-side replenishment.

This document was written after MFIC, AFCS, and BAFR outcomes were already known.
LVRT is therefore a successor generated from failure evidence, not a pristine
clean-room discovery. At decision time the repository inspected only the
aggregate-trade source schema, immutable source identity, source-gap metadata,
and prior pure-clock availability. It did not compute an LVRT feature rank,
event incidence, BTC post-entry return, funding cash flow, PnL, equity path,
CAGR, MDD, or 2024-or-later value.

The known failures may justify the new sequential object, but they may not be
used to tune LVRT after this freeze.

## Why this is a new mechanism

- **MFIC** combined same-window hidden-metaorder persistence with price-impact
  curvature and held for 15–60 minutes. Its best stable subgroup still averaged
  only about 2.6 bp gross at account level. LVRT uses no price, response,
  curvature, extension, continuation/fade branch, or MFIC threshold. It trades
  only after a later replenishment transition and holds six hours.
- **BAFR** was a dense five-minute q90 aggressor/tick-frustration reversal with
  8,220 train trades and a roughly -1.9 bp mean gross move. LVRT uses no tick
  direction, carried zero tick, same-bar price response, or q90 flow-vs-price
  conjunction. Its source support has an explicit upper event-count bound.
- **AFCS** traded a contemporaneous compression/sweep condition for 12 hours.
  LVRT requires an ordered vacuum then refill, uses event-time burstiness and
  run structure, and does not reuse AFCS's fill-compression branch.

The mechanism can still be false. Aggregate-trade packing is exchange-specific;
HFT bursts, liquidation cascades, API batching, event aggregation, or regime
changes can imitate a vacuum. Replenishment may confirm continuation rather
than reversal. The fixed primary must fail rather than switch to one of those
stories after outcomes are observed.

## Frozen source and comparator availability

Primary source:

- file:
  `data/binance_um_aggtrade_microstructure_btc_2020_2023/BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz`;
- file SHA-256:
  `c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf`;
- manifest:
  `data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json`;
- manifest SHA-256:
  `6eec40460a6146c58994e52f1af9ace4eecc0c085887d97af5ef17c30b9f7e73`;
- declared rows: 420,732 on `[2020-01-01, 2024-01-01)`; and
- source protocol: official Binance USD-M Futures daily `aggTrades` archives,
  published checksums verified, UTC-floor five-minute aggregation,
  `outcomes_opened=false`.

Required prior comparator clocks are available before LVRT incidence:

1. AFCS-144:
   `results/aggregate_fill_compression_sweep_clock_2026-07-17.csv`,
   SHA-256
   `bf1611554604c1930ba2212e674ea434f7c9793377b3f33ef531b3b4e0381688`,
   573 rows;
2. BAFR-24F:
   `results/binance_aggressor_frustration_clock_2026-07-20.csv`,
   SHA-256
   `f3b816a76decce31136ed23d22f043eb8e80ef1b8697b869241b060062f01747`,
   11,248 rows;
3. MFIC fast/slow reconstruction source:
   `training/preregister_metaorder_fragmentation_impact_curvature.py`,
   SHA-256
   `51e99dbdc5ba13e6b4ac15e3915ec5b30e36dff89c1e5b31a5f3f7f272f01a59`;
4. MFIC support binding:
   `results/metaorder_fragmentation_impact_curvature_support_2026-07-14.json`,
   SHA-256
   `03bc5b2f67f974efa04715511920701c0db875b8bb4251f2e4c734a591aa80c8`,
   declaring 1,566 fast and 1,635 slow nonoverlap events; and
5. the frozen three-member live pure-clock bundle already used by TSDR/CCHR.

If any required comparator cannot produce a nonempty pure clock over the LVRT
research interval, LVRT retires before its own real incidence is published.

## Allowlisted source fields

LVRT may read exactly:

1. `date`;
2. `agg_trade_count`;
3. `event_notional_hhi`;
4. `normalized_effective_event_count`;
5. `signed_event_imbalance`;
6. `max_same_sign_run_share`; and
7. `interarrival_burstiness`.

Every price/notional field is forbidden, including `first_price`, `last_price`,
`micro_log_return`, `signed_price_response`, `quote_notional`, all signed/buy/
sell notionals, event-notional levels/quantiles/max, base volume, and
buy/sell event-size ratio. Fill counts, flow coherence, sign-flip rate,
mean run length, and interarrival mean/std are also excluded. This prevents an
unregistered MFIC/BAFR/AFCS repair and keeps the source gate free of BTC price.

## Exact grid, gaps, and availability

Materialize the complete UTC five-minute grid on
`[2020-01-01, 2024-01-01)`. A row is locally valid only when:

- its timestamp is unique and exactly on the grid;
- all six numeric feature fields are finite;
- `agg_trade_count` is a positive integer;
- HHI and normalized effective count are in `(0, 1]`;
- signed event imbalance and burstiness are in `[-1, 1]`; and
- max same-sign run share is in `(0, 1]`.

The source manifest's aggregate-trade ID audit identifies these gap days:

```text
2020-04-15, 2021-02-09, 2021-02-24, 2021-05-19, 2022-09-06
```

Every bar on a gap day is invalid even if a five-minute aggregate exists. A
missing/duplicate/malformed bar or gap-day bar clears all rank histories,
cancels the active episode, and leaves the state idle. There is no interpolation,
forward fill, partial-day salvage, skip-over rank history, or shorter quarantine.
The required 2,016-bar history must rebuild contiguously after the gap.

For a source row starting at `t`, the observation interval is `[t, t+5m)`.
Historical publication timestamps are not available. Freeze a conservative
availability time of `t+10m`: one complete five-minute finality buffer after
the interval closes. A confirming event enters only at `t+15m`, the following
five-minute open. Live operation must use the later of this synthetic clock and
the locally persisted final classification time.

## Strict-prior empirical ranks

At every locally valid bar, compute empirical midranks against exactly the
2,016 immediately preceding valid **contiguous** bars, excluding the current
bar. If the full seven-day history is absent, every rank is unavailable.

```text
midrank(x) = (count(prior < x) + 0.5 * count(prior == x)) / 2016
```

Define:

```text
R_burst = rank(interarrival_burstiness)
R_hhi   = rank(event_notional_hhi)
R_neff  = rank(normalized_effective_event_count)
R_run   = rank(max_same_sign_run_share)
R_flow  = rank(abs(signed_event_imbalance))
R_count = rank(agg_trade_count)

vacuum_score = mean(R_burst, R_hhi, 1 - R_neff, R_run)
flow_sign    = sign(signed_event_imbalance)
```

Ties use exact parsed binary64 values. There is no epsilon, clipping, winsor,
cross-sectional normalization, day/time adjustment, or current-row inclusion.

## Frozen first-replenishment state machine

Process bars once in UTC order.

### Setup

When idle, start an episode at the first ranked bar satisfying all conditions:

```text
vacuum_score >= 0.90
R_flow       >= 0.90
R_count      <= 0.25
flow_sign    != 0
```

Freeze `setup_sign`, `setup_time`, and `deadline = setup_time + 12 bars`.

### Armed episode

Inspect the next 12 source bars, ages 1 through 12 inclusive. Ignore every
later setup while armed. The first bar satisfying all conditions confirms:

```text
flow_sign == -setup_sign
R_flow    >= 0.60
R_burst   <= 0.50
R_hhi     <= 0.50
R_neff    >= 0.50
R_run     <= 0.50
R_count   >= 0.50
```

The fixed trade side is `-setup_sign`, equal to the confirming flow sign. If no
confirmation appears by age 12, expire. A source invalidity cancels immediately.
A confirmation, cancellation, or expiry terminates the episode; its bar cannot
also start another episode. The episode cannot refresh, replace its onset, wait
for a stronger confirmation, or retry after an overlap suppression.

## Frozen execution

- causal origin: setup bar start;
- decision event: confirming bar start;
- decision availability: confirming bar start plus 10 minutes;
- entry: confirming bar start plus 15 minutes;
- exit: entry plus exactly 72 five-minute bars / six hours;
- side: fixed `-setup_sign`;
- exposure: fixed `0.5x` account notional;
- chronological non-overlap: suppress, never queue or replace, a candidate
  with `entry < prior_accepted_exit`; and
- split containment: setup, every armed bar, confirmation, availability,
  entry, complete held path, and exit must remain in one half-open split.

No stop, take-profit, trailing exit, price filter, funding filter, OI gate,
regime gate, score priority, leverage search, LLM, or RL is allowed in the
deterministic candidate.

## Research windows and source-only support

- rank warm-up only: `[2020-01-01, 2020-02-01)`;
- train: `[2020-02-01, 2023-01-01)`;
- selection: `[2023-01-01, 2024-01-01)`; and
- sealed forward evaluation: 2024 and later.

The primary must pass every source-only check.

### Train

- 100 through 360 accepted events inclusive;
- at least 25 and at most 160 in each of 2020, 2021, and 2022;
- at least 12 in every contained half-year;
- at least 35 LONG and 35 SHORT;
- at least 60 active UTC weeks;
- maximum calendar-month share at most 15%; and
- maximum UTC-entry-weekday share at most 22%.

### Selection

- 45 through 180 accepted events inclusive;
- at least 18 in each half-year;
- at least 8 in every quarter;
- at least 15 LONG and 15 SHORT;
- at least 25 active UTC weeks;
- maximum calendar-month share at most 20%; and
- maximum UTC-entry-weekday share at most 25%.

Every source/schema/range/gap/rank/state/latency/hold/non-overlap/containment
assertion and every required-comparator availability check must pass. Too many
events fails because LVRT is specifically a rare-transition hypothesis.

## Frozen controls

Each non-exact control has its own state machine, split containment, and
non-overlap scheduler. None may replace a failed primary.

1. **Vacuum only** — every primary setup emits after the same 15-minute latency,
   side `-setup_sign`, six-hour hold; no replenishment wait.
2. **Replenishment only** — every bar meeting the exact confirmation geometry
   and nonzero `flow_sign` emits in `flow_sign`; no prior vacuum.
3. **No flow flip** — exact primary setup; first later bar within 12 bars meeting
   every replenishment condition except the sign-flip predicate confirms; side
   remains `-setup_sign`.
4. **Reverse-order relay** — a replenishment-geometry bar arms; the first
   later vacuum setup within 12 bars confirms, side `-vacuum_sign`.
5. **Exact direction flip** — exact primary clocks, `side=-primary_side`.
6. **Deterministic random side** — exact primary clocks; SHA-256 of
   `"LVRT-72-random-side-20260721|" + entry_time` assigns LONG when its first
   byte is below 128 and SHORT otherwise.
7. **One-bar execution delay** — exact primary origin, confirmation, and side;
   shift entry and exit by exactly five minutes, dropping boundary/overlap
   failures without replacement.

## Frozen novelty gate

Before any LVRT market outcome is opened, compare its pure clock against AFCS,
BAFR, MFIC fast, MFIC slow, and all three frozen live sleeves. For every
required nonempty comparator over common coverage require:

- exact entry-time Jaccard `<= 0.20`;
- maximum one-to-one matching within plus/minus six hours covers at most 35%
  of LVRT entries; and
- absolute signed occupied-exposure correlation on the complete common
  five-minute grid `<= 0.40`.

Publish position-bar Jaccard as a diagnostic. Missing comparator coverage,
empty required members, or a failed metric retires LVRT before economic
evaluation. A comparator cannot be dropped after LVRT incidence is known.

## Sequential economic gate

Only an exact support and novelty pass authorizes a separately implemented,
tested, committed, and hash-frozen evaluator. Open 2020-02 through 2022 train
first. Open 2023 selection only after an exact train pass. Open 2024, 2025, and
recent 2026 sequentially only after every prior gate passes unchanged.

Every opened window must report absolute return, full-calendar CAGR including
idle cash, strict intratrade/pre-entry-HWM MDD, CAGR/strict-MDD, trades, both
sides, exact funding, base and 10 bp/notional/side stress costs, extra latency,
weekly-cluster significance, and every mechanism control.

Primary qualification requires:

- positive absolute return;
- `CAGR / strict MDD >= 3.0`;
- strict MDD `<= 15%`;
- mean gross underlying move at least 30 bp/trade;
- weekly-cluster one-sided sign-flip `p <= 0.10`;
- positive 10 bp/notional/side stress return;
- positive one-bar-delayed return;
- positive return in every contained train year and selection half;
- positive LONG and SHORT sleeves in train and selection; and
- primary CAGR/strict-MDD at least 0.50 above every mechanism control.

Because LVRT was generated after related 2020–2023 outcomes were known, train
and selection success cannot authorize production. At least one untouched
12-month post-2023 evaluation plus forward paper trading is mandatory.

## Live parity

Historical rows came from official Binance USD-M daily `aggTrades` archives.
Live promotion must consume the official BTCUSDT aggregate-trade WebSocket,
persist raw events and local receipt times, reproduce buyer-maker aggressor
signs and aggregate-event/fill IDs, close exact UTC five-minute bars, wait the
same finality buffer, and fail flat on dropped/reordered/duplicated IDs,
WebSocket gaps, late events, clock drift, or feature divergence. REST backfill
may repair storage only before the rank history rebuilds; it cannot backdate a
missed live signal.

## Stop condition

The next work unit is a comparator-availability freeze and deterministic
source-only support builder. It may read only the allowlisted aggregate-trade
fields and pure comparator clocks. If support or comparator availability fails,
LVRT-72 retires. Market OHLC, funding, returns, and economic simulation remain
forbidden until support and novelty both pass.
