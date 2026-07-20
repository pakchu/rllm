# Deribit Funding-Impulse Absorption mechanism decision — 2026-07-20

## Decision

The next standalone BTC candidate is **DFIA-72 — Deribit Funding-Impulse
Absorption, six-hour hold**. It will use the previously unused hourly
`interest_1h`, `interest_8h`, `index_price`, and `prev_index_price` fields from
Deribit's BTC perpetual funding-history endpoint.

DFIA asks whether a sharp change in the most recent hour of continuously paid
perpetual funding, relative to its eight-hour memory, fails to move the Deribit
BTC index in the crowded direction. That failure is interpreted as absorption
and tentatively traded against the newly crowded side.

This file freezes the source axis, algebra, economic direction, latency, and
hold before the complete source prefix or signal incidence is opened. It reads
no Binance execution row, Binance funding row, post-entry return, held path,
PnL, or 2024+ source row.

## Why this is a new observable axis

Deribit documents `public/get_funding_rate_history` as hourly historical
funding data for a perpetual instrument. Each result row contains:

- `timestamp`;
- `interest_1h`, the one-hour interest rate;
- `interest_8h`, the eight-hour interest rate;
- `index_price`; and
- `prev_index_price`.

Deribit's inverse-perpetual specification says funding is measured and paid
continuously, is displayed as an eight-hour rate, and is derived from the
mark-price premium over the Deribit index after a damper and cap. Positive
funding transfers value from longs to shorts; negative funding transfers value
from shorts to longs.

Official references:

- [Hourly funding-rate history](https://docs.deribit.com/api-reference/market-data/public-get_funding_rate_history)
- [BTC inverse-perpetual funding specification](https://support.deribit.com/hc/en-us/articles/31424954847133-Inverse-Perpetual)
- [API usage policy](https://docs.deribit.com/articles/api-usage-policy)
- [API rate limits](https://docs.deribit.com/articles/rate-limits)

Repository-wide search found no prior use of this endpoint or any
`interest_1h`/`interest_8h` field. The rejected CFCF experiment compared
scheduled Binance and Bybit funding/premium differences at common eight-hour
boundaries. DFIA instead observes **within-venue continuous funding memory at
an hourly cadence**. It does not repair CFCF's venue-spread clock or reuse its
side map.

The source is public market data under Deribit's API usage and rate-limit
policies. No open redistribution grant was found. Raw responses and the local
hourly source file will therefore remain ignored; committed artifacts may
contain URLs, code, hashes, source-quality summaries, a derived candidate
clock, and aggregate results only. Production polling is one request per hour
and remains far below the documented public-request limits.

## Mechanism

For completed hourly row `t`, convert the reported eight-hour interest to its
hourly mean and compare the newest hour with that memory:

```text
trailing_hourly_mean_t = interest_8h_t / 8
funding_impulse_t      = interest_1h_t - trailing_hourly_mean_t
index_return_1h_t      = log(index_price_t / prev_index_price_t)
```

Dividing the eight-hour interest by eight is a research normalization that
puts both terms on a one-hour scale. It is sign-equivalent to
`8 * interest_1h - interest_8h`. Deribit documents the two source fields but
does not assert that this difference predicts price.

DFIA will standardize the funding impulse and completed index return against
strictly earlier hourly observations and require an opposing dislocation:

- unusually positive funding impulse with nonpositive/weak index response is
  tentatively **short**: long-side funding pressure accelerated but failed to
  lift the index;
- unusually negative funding impulse with nonnegative/strong index response is
  tentatively **long**: short-side funding pressure accelerated but failed to
  depress the index.

The terms “crowding” and “absorption” are falsifiable interpretations. Funding
is a mark-index premium transform, not a direct participant census; the
Deribit index is not the executable Binance price; and one-hour versus
eight-hour memory is mechanically related rather than independent evidence.
Mandatory component, stale, side, and timing controls must reject the candidate
if the interaction adds no information.

Thresholds, strictly prior reference length, transition/re-entry scheduler,
support minima, and exact controls must be committed before downloading the
complete 2019–2023 prefix. There will be no sign, threshold, reference, latency,
or hold grid. The hold is fixed at 72 five-minute bars (six hours): shorter
than the eight-hour memory so the trade tests the unresolved impulse rather
than simply waiting for another full memory cycle.

## Source coverage and bounded schema probe

The production endpoint is:

```text
GET https://www.deribit.com/api/v2/public/get_funding_rate_history
    ?instrument_name=BTC-PERPETUAL
    &start_timestamp=<epoch milliseconds>
    &end_timestamp=<epoch milliseconds>
```

Bounded source-only probes on 2026-07-20 established:

- the earliest observed row is 2019-04-30 10:00 UTC;
- hourly rows are present in 2021, 2023, and 2026;
- rows have the five exact result fields listed above, positive index prices,
  finite interest rates, and whole-hour UTC timestamps; and
- the REST response also contains JSON-RPC/server-timing metadata, which is not
  a signal field.

Across three bounded 32-hour probes in 2021, 2023, and 2026, reported
`interest_8h` was numerically close to the rolling sum of the latest eight
`interest_1h` observations (median relative discrepancy below 0.5% in each
probe). This supports the memory interpretation but is an observed source
invariant, not a formula promised by the endpoint documentation. The complete
source audit must report the discrepancy and fail closed on material semantic
drift.

The endpoint silently returned at most 744 rows for a wide request in the
probe. The source loader must therefore issue deterministic windows no wider
than 28 days, overlap request boundaries by one hour because
`start_timestamp` behaves exclusively, deduplicate exact repeated boundary
rows, and reject conflicts, reversals, scope drift, or a response at the
window cap.

These probes opened no complete prefix, funding impulse, reference statistic,
event count, market outcome, or performance metric.

## Causal availability and live parity

The official endpoint calls the data hourly but does not publish a historical
first-seen timestamp or immutable-vintage archive. Probe behavior is consistent
with a row timestamp marking the completed hour: a request starting exactly at
an hourly timestamp returns the following hour first.

The frozen conservative clock is:

1. source row timestamp `T`;
2. historical synthetic availability no earlier than `T + 5 minutes`;
3. earliest observable five-minute open at that availability boundary;
4. one additional completed five-minute latency bar; and
5. earliest executable entry at `T + 10 minutes`.

Live/shadow collection uses the actual first successful observation when it is
later. A late or missing row delays or cancels the signal and can never be
backdated. `prev_index_price` must equal the preceding contiguous row's
`index_price`; a gap breaks the feature chain rather than being filled.

Historical values may be restated by the exchange. The deterministic source
hash freezes one downloaded vintage but cannot recreate an archive of every
historical publication. Live promotion therefore requires a forward
vintage-parity audit.

## Frozen research sequence

1. Implement and test a source-only downloader/parser for
   `[2019-04-30T10:00:00Z, 2024-01-01T00:00:00Z)` without running the complete
   download. It must preserve exact JSON decimals, bind deterministic response
   result hashes, and fail closed on schema, window, duplicate, continuity,
   price-chain, and server-environment drift.
2. Commit one exact source-support preregistration before the complete download.
   It must freeze feature/reference transforms, thresholds, scheduler, support
   periods, coverage/dispersion gates, side balance, and controls.
3. Open only the ignored hourly source and outcome-blind support. Reject without
   repair if source coverage, both directions, calendar dispersion, or
   split-contained non-overlapping event counts fail. No Binance market or
   funding source may be opened for a support failure.
4. If support passes, hash-freeze a strict evaluator before loading any
   post-entry path. Open train first and calendar 2023 only after train passes.
5. Open 2024, 2025, and 2026 YTD sequentially, stopping at the first failed
   sealed gate.

The later evaluator must use full-calendar CAGR, global/pre-entry-HWM strict
MDD, every held five-minute path, entry cost, exact Binance funding boundaries,
virtual adverse exit cost, actual exit, base/stress costs, and split-contained
nonoverlap. Controls must include exact side flip, funding-impulse-only,
index-response-only, eight-hour stale source, one extra execution bar, constant
long/short sides on the exact clock, and a deterministic within-year paired
source permutation.

The target remains positive absolute return, full-calendar CAGR divided by
strict MDD of at least `3`, strict MDD no greater than `15%`, stress-cost
survival, statistically meaningful trades and independent time clusters,
balanced directions, positive contained subperiods, and a precommitted margin
over the best component/stale control. The branch has broad prior BTC exposure,
so the result can establish only a candidate-level frozen claim, not a pristine
global holdout.
