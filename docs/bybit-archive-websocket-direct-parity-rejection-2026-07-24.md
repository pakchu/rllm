# BAWDP-v1 direct-parity rejection — 2026-07-24

## Decision

**Retire BAWDP-v1 unchanged before any candidate mechanism, comparator, or
market outcome.**

The committed verifier began the one authorized stream of the official
2026-07-23 Bybit `BTCUSDT` public-trade archive. The archive exposed a
timestamp that does not map exactly to an integer millisecond under the frozen
rule:

```text
Decimal(timestamp) * 1000 must be an exact integer
```

Rounding, truncating, tolerance matching, composite-key fallback, changing the
comparison interval, or rerunning against another day is forbidden. Direct
archive↔WebSocket parity therefore cannot be established under BAWDP-v1.

## Immutable evidence

Executed protocol commit:

```text
992a7052b6c610172c47e55f60c71090420f793f
```

Rejection artifact:

```text
results/bybit_archive_websocket_direct_parity_2026-07-24.json
SHA256 8e380fb2f53272a6b266900da5fd2f98667ecd76aa14c16dc8f7311f66768235
manifest_hash 81d47da5de96108343fcf3a8a848ad0f69a55c02285ed43f6080f058d080102f
```

Frozen capture inputs remained hash-identical:

```text
capture manifest
38027f767122c9f7d5a57ae5a5a0f1445525e63bb05fcd75981de42677f68e16

WebSocket NDJSON gzip
b63ec8ee4f8f2c260631e09ce02d8dec71b8f17827a8e3592d792e4b2f5b6937
```

The archive body was streamed only until the terminal schema/time-identity
failure. No raw or decompressed archive was persisted. Because the stream did
not complete, no complete compressed archive hash is claimed.

## Outcome boundary

The rejection artifact records all of the following as false:

- BSEA reopened or BSEA-family mechanism authorized;
- candidate definition or candidate incidence opened;
- Binance comparator, market, or funding rows opened;
- future returns or labels opened;
- model training or inference opened; and
- PnL, CAGR, or strict MDD opened.

No interval source artifact was written.

## CLI reporting defect after durable rejection

The verifier atomically wrote the valid terminal rejection artifact and then
its console-summary path raised a `KeyError` because a terminal report has no
complete archive SHA. This happened after the source decision was already
durably written and does not change it.

Commit `747d878` makes the console summary handle a missing complete archive
hash and adds a regression test. The source verifier was **not rerun**, because
the frozen protocol permits no rerun after a partial-body terminal failure.

## No-repair consequence

BAWDP-v1 and BSEA-24 are both closed. This result may not authorize:

- coercing archive timestamps to WebSocket milliseconds;
- matching by nearest millisecond or a tolerance window;
- dropping archive rows with sub-millisecond timestamps;
- using price/size/side as a composite identity;
- treating REST overflow or archive timestamp drift as harmless; or
- reviving taker-side persistence, entropy, continuation/reversal, or
  cross-venue nonconfirmation under a renamed Bybit candidate.

Any future Bybit public-trade work would require a separately committed,
orthogonal source identity and mechanism. It is not the next priority. The
alpha search returns to a different high-density source family with an already
proven historical/live representation.

