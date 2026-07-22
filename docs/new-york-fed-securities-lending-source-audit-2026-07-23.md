# New York Fed SOMA securities-lending source audit — 2026-07-23

## Decision

**PASS as a candidate-blind source axis.** The official New York Fed API
produced a complete, hash-frozen 2019–2023 daily securities-lending operation
panel and CUSIP-detail panel. No candidate feature, event count, side, hold,
BTC row, funding value, forward return, PnL, CAGR, or MDD was computed.

The next authorized step is to commit exactly one collateral-scarcity mechanism
before combining these source values into candidate features.

## Official source

- program purpose and daily noon auction:
  <https://www.newyorkfed.org/markets/domestic-market-operations/monetary-policy-implementation/securities-lending>;
- field definitions and historical search:
  <https://www.newyorkfed.org/markets/desk-operations/securities-lending>;
- API documentation:
  <https://markets.newyorkfed.org/static/docs/markets-api.html>;
- frozen OpenAPI bytes:
  <https://markets.newyorkfed.org/static/docs/markets-api.yml>.

The source was retrieved in six exact requests: one OpenAPI document and one
inclusive detailed-operation response for each calendar year from 2019 through
2023. Raw response bytes, URLs, retrieval timestamps, content types, sizes, and
SHA-256 values are retained under
`data/new_york_fed_securities_lending_2019_2023/raw/`.

## Frozen coverage

| Item | Result |
|---|---:|
| Operations | 1,259 |
| 2019 / 2020 / 2021 / 2022 / 2023 operations | 252 / 253 / 252 / 251 / 251 |
| First / last operation date | 2019-01-02 / 2023-12-29 |
| Unique operation/CUSIP detail rows | 182,616 |
| 2019 / 2020 / 2021 / 2022 / 2023 detail rows | 27,040 / 32,428 / 33,465 / 47,614 / 42,069 |
| Detail rows with unavailable weighted rate | 744 |
| Other normalized-detail null values | 0 |
| Nonempty operation notes | 0 |
| Nonempty extension totals | 0 |

Every unavailable weighted rate belongs to a zero-award detail. The exact
official `"N/A"` marker is normalized to null, never zero. Submitted and
accepted detail amounts are complete, and their sums reconcile exactly to
every operation total.

## Timing audit

The ordinary operation is released at 12:00 Eastern and closes at 12:15, but
the official history contains a small number of later or irregular auction
times. The builder preserves each printed time and does not force a normal
schedule.

Because the API does not promise a historical result-publication minute, every
operation is unavailable until the later of:

1. the next UTC midnight after `operationDate`; and
2. `lastUpdated`, interpreted in `America/New_York`.

Ambiguous or nonexistent New York local timestamps reject the build. All 1,259
rows passed and are unavailable no earlier than the next UTC midnight. A later
candidate must additionally wait one complete five-minute bar.

## Structural validation

- exact official host, path, query string, HTTP 200, and JSON content type;
- exact six-entry fetch ledger with cache hash/byte/URL replay;
- unknown operation or detail fields reject normalization;
- exact operation ID and operation/CUSIP uniqueness;
- exact frozen year membership, with every annual response nonempty;
- finite nonnegative exact-decimal values;
- accepted amount never above submitted amount;
- nonzero awards require a numeric weighted-average lending rate;
- exact operation/detail submitted and accepted total reconciliation;
- deterministic gzip CSV outputs and canonical JSON manifest;
- cache refresh refused after the initial frozen retrieval.

The first full build exposed one implementation error: a correctly false
`unknown_fields_accepted` flag had been included in an `all(...)` success test.
The run failed before writing a manifest. Commit `c051de5` renamed the positive
invariant to `unknown_fields_rejected`, tightened URL/cache/annual completeness,
preserved nullable non-reconciliation fields, and added DST rejection tests.
The cached official bytes were not refreshed. The corrected cache replay then
passed.

## Frozen identities

- source-axis decision SHA-256:
  `be474218fc19fa55023b2712b187fc53fbc1a1079bb8fe15f8340831bd30a795`;
- builder SHA-256:
  `2f0b5b3daca253ca015c7f691faf0ab75d11c200c11f5bc1c47b34ed1b85ef45`;
- operation panel SHA-256:
  `99eb8c37c05417789dfad7452c7b2ddc5b6b640078b87451f1c945158af77906`;
- detail panel SHA-256:
  `27178d8738cb50c4e6c13f1e5940fcfdf4009e6979b006c42fb86fb399d0716d`;
- build-manifest file SHA-256:
  `58b9eb56728065d919978b8969e9bbb4bcb291f723a290d22045fe2ca3da2019`;
- canonical manifest hash:
  `748db33b3ea40eb48d126d0e9882b05e1994741bf851a8f3c7b89d5166db969c`;
- fetch-ledger file SHA-256:
  `a94ffd5ec5122af115bb29766508c013771f48d83b72ec24690b3e3db8d50bac`.

The complete frozen source directory is 7.6 MiB compressed.

## Revision limitation

This is a current official historical API snapshot, not a captured copy of
every original daily publication. `lastUpdated` is honored as a conservative
availability bound, but the API documentation does not prove that every old
revision timestamp is immutable or exhaustive. This limitation must remain in
every later candidate and live decision. Live promotion requires forward raw
response capture, retrieval timestamps, revision alarms, and schema parity.

## Evidence boundary

The build manifest records:

- `candidate_features_computed=[]`;
- `candidate_incidence_opened=false`;
- BTC market rows read: `0`;
- funding rows read: `0`;
- return rows read: `0`; and
- PnL/CAGR/MDD opened: `false`.

No source ratio, breadth, rank, z-score, transition, direction, or event clock
has yet been calculated.
