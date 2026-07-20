# BATE-288 persistent transport freeze — 2026-07-20

## Decision

The frozen Mempool Esplora backfill may be resumed through
`training/run_bate_esplora_backfill.py`, SHA-256
`028f3d3ff128d17dbda4ac2dc4898267e0c91db98a639b570531a8a7fe0e2a74`.

This wrapper changes only HTTP connection reuse and retry transport. The
already frozen `download_bitcoin_block_summaries.py` continues to own:

- exact host/path construction;
- ten-row page and schema validation;
- height and hash-chain continuity;
- pre-2024 containment;
- atomic SQLite checkpoint commits;
- complete-range audit;
- deterministic gzip output; and
- source-manifest construction.

The wrapper exposes only request pause, retry count, and timeout. It refuses
changes to source host, height range, cutoff, output paths, checkpoint path, or
any other research scope.

## Bounded source-only benchmark

Ten distinct Esplora block-summary pages were requested without loading any
market, funding, return, PnL, or candidate-incidence field:

- fresh `urllib` connection per request: 6.34 seconds;
- one persistent standard-library HTTPS connection: 2.44 seconds.

The benchmark motivated connection reuse only. It did not select a feature,
threshold, side, event, or outcome.

## Frozen safety behavior

- Requests must match `https://mempool.space/api/blocks/<canonical integer>`.
- Redirects, alternate hosts, alternate schemes, query strings, fragments,
  leading-zero heights, and `/api/v1` scope are rejected.
- HTTP 429 and 5xx responses reset the connection and use bounded exponential
  or `Retry-After` backoff.
- Other HTTP status codes fail closed.
- A closed, timed-out, or malformed connection is reset before retry.
- JSON decoding still uses the frozen loader's exact-decimal and non-standard
  constant rejection.
- Existing pages remain valid because the checkpoint contract already pins the
  same host, range, schema protocol, and source decision.

The completed support run must verify the wrapper hash in addition to the
frozen source loader, source-host decision, manifest, block anchors, and source
file. Production remains self-hosted Bitcoin Core; persistent access to the
public host is only a bounded private research transport.
