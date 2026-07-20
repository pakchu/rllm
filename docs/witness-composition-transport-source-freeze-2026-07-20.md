# WCTR-288 source freeze — 2026-07-20

## Frozen response

The source-only WCTR-288 loader committed at `eb35b8a` retrieved the frozen
Mempool endpoint once on `2026-07-20T10:27:31.250986Z`:

- endpoint: <https://mempool.space/api/v1/mining/blocks/sizes-weights/4y>;
- upstream implementation commit:
  [`e9d6cf8c042f946be53e372bb36530cd7b7851a4`](https://github.com/mempool/mempool/commit/e9d6cf8c042f946be53e372bb36530cd7b7851a4);
- HTTP status: `200`;
- response `Date`: `Mon, 20 Jul 2026 10:27:29 GMT`;
- response ETag: `W/"59e71-wimSunVghY3LTE+daWrtvIOPIaY"`;
- response `Last-Modified`: absent; and
- response content type: `application/json; charset=utf-8`.

Earlier bounded schema/coverage probes were not reused. The exact response
below is the only payload bound to WCTR-288.

## Artifact identities

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Raw HTTP body, deterministic gzip | 73,145 | `ddd3615294d501ed3b24c5d43e2fc16319bd87f1add3c14dde2362c4b789c4c1` |
| Decompressed exact HTTP body | 368,241 | `6fdb0db77ae56c5b348918bc966384c1a88032655b9e061c510bfcc3df642e94` |
| Normalized deterministic CSV gzip | 69,318 | `ee761e813085dfdee675ca9d420516f814c4c2824f3f5cef604acc3871d46c61` |

The machine-readable manifest is
`results/mempool_witness_composition_source_manifest_2026-07-20.json`. Its
canonical manifest hash is
`55914b3ec31fe8fb66d8a8dc31acb3784a10b256625073a5aeff1d317660ea8d`.
The loader SHA-256 recorded by that manifest is
`d5cd3f2cab5e501d5484539f1ea3c5aac5a96916dd65aa1060bb561fa639d721`.

## Coverage and clock audit

- Paired response rows: 2,923 size rows and 2,923 weight rows.
- Deliberately dropped edge pairs: 2, the first and last response buckets.
- Retained paired rows: 2,921.
- First retained bucket start: `2022-07-20T12:00:00Z`.
- Last retained bucket start: `2026-07-19T12:00:00Z`.
- Missing 12-hour buckets: 0.
- Maximum adjacent bucket gap: 43,200 seconds.
- Pair height and timestamp mismatches: 0.
- Average block heights, average timestamps, and bucket IDs are strictly
  increasing and unique.
- BIP 141 average-size/weight bound violations: 0.
- Rows requiring the explicit four-byte integer-average tolerance: 0.

Every retained source row is marked available only at its fixed 12-hour bucket
end plus 48 hours. A later evaluator must then wait for one complete five-minute
execution bar. Any missing source bucket is a hard failure; it may not be
forward-filled, backfilled into a signal clock, or treated as an adjacent row.

## Source-only boundary

The normalized artifact contains exactly:

```text
bucket_start_utc, bucket_end_utc, available_at_utc,
avg_height, avg_timestamp, avg_size, avg_weight
```

It contains no derived witness share, transport, rank, threshold, event,
market price, funding, premium, OI, return, or PnL field. The manifest records
zero rows or fields from every outcome family and binds the decision and loader
hashes.

## Vintage and next gate

This is a rolling present-day source snapshot, not proof that the same
integer-rounded aggregates were historically published at each bucket time.
The 48-hour lag is a conservative causal clock, not a substitute for vintage
evidence. Live promotion still requires a forward retrieval archive with
schema, freshness, hash, and value-stability monitoring.

No WCTR feature incidence or BTC performance statistic was calculated while
freezing this source. The next permitted action is to commit one exact
machine-readable WCTR singleton policy and support gate before opening complete
feature incidence.
