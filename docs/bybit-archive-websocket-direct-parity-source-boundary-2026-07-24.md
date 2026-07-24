# Bybit archive↔WebSocket direct-parity source boundary — 2026-07-24

## Decision

Open one new, source-seen but candidate-incidence-unseen transport identity:
**BAWDP-v1 — Bybit Archive↔WebSocket Direct Parity**.

BAWDP asks only whether the official next-day `BTCUSDT` public-trade archive
is an exact ledger for the already captured public WebSocket stream over one
prospectively fixed interior interval. It does not define BSEA-24, a successor
alpha, a feature, a side, a hold, a model, or a profitability claim.

The direct comparison is authorized because the official archive for the
capture day is now published:

```text
https://public.bybit.com/trading/BTCUSDT/BTCUSDT2026-07-23.csv.gz
```

An HTTP metadata request on 2026-07-24 returned status `200`,
`Content-Length: 51913751`, and
`ETag: "b4cd5a78805f5456092fe04e83913178-7"`. No archive body or CSV row was
opened while writing this boundary.

## Why this is not a BSEA repair

`BSEA-24` remains permanently rejected under its frozen three-surface
transport. Its one-second REST `limit=1000` windows skipped 5,816 observed
WebSocket IDs during two bursts even though the scheduler and host clock
passed. The missing REST rows may not be ignored, sampled, imputed, or repaired
by a faster polling interval.

BAWDP:

- discards REST as an evidentiary surface rather than weakening its gate;
- compares the immutable next-day archive directly with the original
  WebSocket capture;
- creates a new transport identity before opening the archive body;
- opens no BSEA sequence feature or candidate incidence; and
- must fail unchanged if direct completeness is not exact.

The prior source-axis decision explicitly required a separately preregistered
source axis for an endpoint change. BAWDP is that separate source axis. Passing
BAWDP would authorize only a new mechanism decision under a new candidate ID;
it would not revive BSEA-24.

**A BAWDP pass does not satisfy, replace, waive, or retroactively pass BSEA's
failed REST↔WebSocket↔archive gate. It cannot authorize BSEA-24 or any
BSEA-family mechanism.**

## Frozen inputs

Source-axis decision:

```text
docs/bybit-public-trade-sequence-source-axis-decision-2026-07-23.md
SHA256 fb12c54b8a4a89cb446baa9014f89546bf6c99e46687be2471b51a2bf1989a21
```

Source audit:

```text
docs/bybit-public-trade-sequence-source-audit-2026-07-23.md
SHA256 fe324cccfb0c3f66963c142b9a6c0237489420313750de873622cadb10e8c112
```

Rejected BSEA transport contract and result:

```text
docs/bybit-public-trade-live-parity-capture-contract-2026-07-23.md
SHA256 50cca9c3e103e8978bb260c65b103dd90361615f6e69443181350ae560622b6c

docs/bybit-public-trade-live-parity-v3-rejection-2026-07-23.md
SHA256 f894abd2fc55c02a75e4e82c076b3ffc67ced0e63f1f774e1ce6be0671ed0ed6

results/bybit_public_trade_live_parity_capture_v3_reject_2026-07-23.json
SHA256 493cde97193c3c837cda4ca2101c7d2068cab8972836fdb511a68b2b7b9fc5d5
```

Immutable capture manifest and WebSocket bytes:

```text
data/bybit_public_trade_parity_capture_2026-07-23T07-10-40-429968Z/manifest.json
SHA256 38027f767122c9f7d5a57ae5a5a0f1445525e63bb05fcd75981de42677f68e16

data/bybit_public_trade_parity_capture_2026-07-23T07-10-40-429968Z/
websocket_messages.ndjson.gz
SHA256 b63ec8ee4f8f2c260631e09ce02d8dec71b8f17827a8e3592d792e4b2f5b6937
```

The manifest already proves one WebSocket session, one successful
subscription, no reconnect, a clean close, a nonreversing host UTC clock, and
strictly increasing `CLOCK_MONOTONIC_RAW`. The ignored REST payload is not an
input to BAWDP and must not be decoded by its verifier.

## Exact canonical fields

The WebSocket side may decode only:

```text
message: ts
trade: T, s, S, v, p, i
```

The archive side may decode only:

```text
timestamp, symbol, side, size, price, trdMatchID
```

Canonical mapping:

```text
T           ↔ exact integer milliseconds from Decimal(timestamp) * 1000
s           ↔ symbol
S           ↔ side
v           ↔ size
p           ↔ price
i           ↔ trdMatchID
```

Price and size compare as finite positive arbitrary-precision decimals.
Symbol must be exactly `BTCUSDT`; side must be exactly `Buy` or `Sell`;
timestamps and IDs must be nonempty and valid. Decimal timestamp conversion
must produce an exact integer millisecond; rounding is forbidden.

WebSocket `seq`, tick direction, block flags, RPI flags, message boundaries,
receipt time, and local clock samples are source-integrity metadata only and
cannot enter a future alpha. Archive-only or recent-only fields are ignored
after exact header validation.

## Prospectively fixed comparison interval

Normalize the frozen WebSocket file once. Let:

```text
first_ws_ms = minimum valid WebSocket trade T
last_ws_ms  = maximum valid WebSocket trade T

start_ms = first_ws_ms + 5000
end_ms   = last_ws_ms  - 5000
```

The comparison interval is the half-open server-time interval
`[start_ms, end_ms)`. The five-second edge exclusion is fixed before archive
row access and is never adjusted from observed mismatches.

The interval must span at least 300,000 milliseconds and contain at least
1,000 unique WebSocket IDs. These are inherited operational source floors,
not alpha-support evidence. Failure retires BAWDP.

## Exact direct-parity gate

The verifier passes only if all conditions hold:

1. frozen input hashes and manifest self-hash validate;
2. the repository worktree is exactly the committed verifier revision and
   fully clean before any archive body request;
3. filesystem use is below 300 GiB before the request;
4. the response is HTTP 200 from exact host `public.bybit.com`, with no
   redirect to another host;
5. the decompressed header is exactly:
   `timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,`
   `homeNotional,foreignNotional,RPI`;
6. the gzip stream is read exactly once and its complete compressed SHA-256,
   byte count, ETag, Last-Modified, and response time are recorded;
7. every archive row in `[start_ms, end_ms)` has one unique `trdMatchID`;
8. every WebSocket row in `[start_ms, end_ms)` has one unique `i`;
9. the two interval ID sets are exactly equal;
10. all six canonical fields agree for every ID;
11. no duplicate ID has conflicting canonical fields;
12. records with different millisecond timestamps induce the same
    nondecreasing temporal order; tie order inside one millisecond is not
    treated as documented semantics; and
13. no candidate feature, comparator, market, funding, return, label, model,
    PnL, CAGR, or MDD value is opened.

There is no percentage tolerance, sampled match, composite-key fallback,
missing-row allowance, retry on a different day, or post-run interval change.

## Storage and execution contract

The archive must be streamed and discarded. It may not be persisted in raw or
decompressed form. The only retained outputs are:

- a hash-bound JSON source-parity report;
- an optional deterministic gzip CSV containing canonical rows only for the
  fixed parity interval; and
- a Markdown pass/rejection record.

The verifier and tests must be committed before the first archive body byte is
read. The real verifier may execute once from a HEAD-clean worktree. A network
failure before a response body begins may be retried against the same exact
URL; any partial-body, gzip, schema, hash, identity, or parity failure is a
terminal no-repair rejection.

## Candidate boundary after a pass

Only exact BAWDP passage may authorize one new relational sequence mechanism.
That mechanism must be committed before historical archive values beyond the
fixed parity interval are reduced.

The successor must not be:

- Bybit-only momentum or aggregate taker imbalance;
- volume, trade count, average ticket, HHI, fill dispersion, or a thresholded
  version of those known families;
- generic Bybit-versus-Binance lead/lag or transfer entropy;
- trade-ID, `seq`, UUID, message-boundary, or archive-defect economics;
- a repair or renamed continuation/reversal version of BSEA-24; or
- an analyzer/trader two-model architecture.

BAWDP deliberately does not select a successor design space. In particular,
the BSEA concepts of micro-bucketed taker-side sequence persistence,
continuation/reversal, entropy, and Bybit-versus-Binance nonconfirmation are
closed rather than transferred to a new name. After a pass, a separate
candidate-boundary commit must prove that its economic object is orthogonal to
those concepts and to the repository's prior fill-dispersion, same-millisecond
cascade, aggressor-frustration, quantity-lattice, transfer-entropy, and generic
lead/lag families.

Any later deterministic composer must own opportunity time, fixed side, hold,
and leverage. A small Gemma policy may receive only causal symbolic relation
tokens plus current position/risk state and choose exactly
`TRADE_FIXED_SIDE` or `ABSTAIN`.

## Mandatory sequence

1. commit this BAWDP boundary;
2. implement and test an archive-streaming direct-parity verifier;
3. independently review the verifier for source leakage and failure safety;
4. commit the verifier from a clean worktree;
5. execute the real source-parity run once;
6. commit the immutable pass/rejection artifact;
7. on a pass, freeze one new mechanism and preregistration before historical
   source reduction;
8. run source support and pre-outcome novelty before any economic evaluator;
9. retire unchanged at the first failed gate.
