# BFRT-288 source freeze — 2026-07-20

## Frozen response

The source-only BFRT-288 loader committed at `a4425d3` retrieved the frozen
Mempool endpoint once on `2026-07-20T09:02:30.628066Z`:

- endpoint: <https://mempool.space/api/v1/mining/blocks/fee-rates/3y>;
- upstream implementation commit:
  [`e9d6cf8c042f946be53e372bb36530cd7b7851a4`](https://github.com/mempool/mempool/commit/e9d6cf8c042f946be53e372bb36530cd7b7851a4);
- HTTP status: `200`;
- response `Date`: `Mon, 20 Jul 2026 09:02:28 GMT`;
- response ETag: `W/"4e564-RJvYwNUJqLGdQWvvCgzgfWMQELc"`;
- response `Last-Modified`: absent; and
- response content type: `application/json; charset=utf-8`.

The earlier bounded coverage probe was not reused. The exact response below is
the only payload bound to BFRT-288.

## Artifact identities

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Raw HTTP body, deterministic gzip | 35,671 | `4309dfbbdb08b89cd9cc92a341bd6186146b1e67adc2c3f926c8154ddabc4898` |
| Decompressed exact HTTP body | 320,868 | `480a99c3ebfd49f98511f94fe05d9f8d76a2e28ebef7cde937768b0d4321e008` |
| Normalized deterministic CSV gzip | 47,095 | `007d13ba756fd29faae1ae87caa11554438b54bb5028f24b2f0c21ddf3a0e55d` |

The machine-readable manifest is
`results/mempool_block_feerates_source_manifest_2026-07-20.json`. Its canonical
manifest hash is
`fe616bcf294e8b3b2abc6dec124e922f77df4bca47a86249fc270f2af6b46f21`.
The loader SHA-256 recorded by that manifest is
`ebd30dd109a92c4dc5a2a6a444a5d5760fa4360c7fd848b02923f0670e4a2910`.

## Coverage and clock audit

- Response rows: 2,193.
- Deliberately dropped edge rows: 2, the first and last response buckets.
- Retained rows: 2,191.
- First retained bucket start: `2023-07-20T12:00:00Z`.
- Last retained bucket start: `2026-07-19T12:00:00Z`.
- Missing 12-hour buckets: 0.
- Maximum adjacent bucket gap: 43,200 seconds.
- Average block heights, average timestamps, and derived bucket IDs are all
  strictly increasing and unique.
- Every row satisfies non-negative, monotone percentile ordering.

Each retained source row is marked available only at its fixed 12-hour bucket
end plus 48 hours. A later evaluator must then wait for one complete five-minute
execution bar. Missing rows may never be forward-filled or backdated.

## Independent replay check

After collection, a fresh verifier:

1. recomputed the decision, loader, compressed artifact, decompressed payload,
   normalized artifact, and canonical manifest hashes;
2. decoded the archived HTTP bytes with duplicate-key and non-standard-number
   rejection;
3. rebuilt every normalized CSV row from the raw archive; and
4. confirmed exact row-for-row equality with the frozen CSV.

All checks passed. The source transport loaded zero BTC market, funding,
premium, OI, return, PnL, BFRT feature, or signal-incidence rows.

## Boundary and next gate

This is a rolling present-day source snapshot, not proof that the same values
were historically published at each backtest timestamp. The 48-hour lag is a
conservative research clock, not a substitute for vintage evidence. Live
promotion therefore still requires at least 90 forward shadow days with
fail-closed schema, freshness, and value-stability checks.

No fee-rate value, feature incidence, BTC outcome, or performance statistic was
summarized while freezing this source. The next permitted action is to commit
one exact BFRT singleton policy and support gate before opening full feature
incidence. Test and eval remain sealed.
