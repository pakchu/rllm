# Aave V3 Ethereum borrow-rate ledger source boundary — 2026-07-24

## Decision

Select **AV3BRL-v1 — Aave V3 Ethereum Borrow-Rate Ledger** as the next
independent BTC alpha source axis.

AV3BRL asks only whether the finalized Ethereum log stream emitted by the
Aave V3 Ethereum Pool can reproduce a causal, live-readable history of reserve
borrow-rate updates for:

- WBTC;
- WETH;
- USDC; and
- USDT.

This document freezes a source identity, transport, schema, and no-repair
sequence. It does **not** freeze a trading signal, action, side, threshold,
holding period, model, reward, or profitability claim. No Aave event log,
event count, rate value, candidate clock, BTC outcome, funding row, return,
PnL, CAGR, or strict MDD was opened before this boundary.

Only immutable block-header metadata needed to bind the two source-parity
windows was queried before this commit.

## Why the preceding branches are not reopened

### Binance Options `EOHSummary`

`EOHSummary` was already rejected at source coverage on 2026-07-20. The
official BTCUSDT prefix then contained only 147 daily archives from
2023-05-18 through 2023-10-23, about five months:

- `docs/btc-alpha-source-axis-decision-2026-07-20.md`
- <https://data.binance.vision/?prefix=data/option/daily/EOHSummary/BTCUSDT/>

No option-surface rule may repair or relabel that source failure.

### OKX public market data

OKX exposes technically attractive historical trade and borrowing-rate
interfaces, but its current API Agreement grants a limited license for the
user's own internal purposes and own OKX account, and applies market-data
restrictions to public endpoints. This repository's production execution
target is not being changed from Binance to OKX inside an alpha-search source
decision. OKX data is therefore not adopted here.

- <https://www.okx.com/en-us/help/okx-api-agreement>
- <https://www.okx.com/docs-v5/en/>

### Prior microstructure and chain families

The following remain closed:

- Bybit and Deribit public-trade parity identities;
- Binance individual-fill, aggregate-fill, taker-flow, ticket-size,
  sequence, and intrinsic-volume families;
- funding, premium, open-interest, liquidation, and cross-venue handoff
  families;
- Bitcoin block, fee, UTXO, miner, witness, and mempool families; and
- stablecoin issuance/redemption, bridge, custody, and exchange-inventory
  families.

AV3BRL is not permission to reconstruct any of those clocks from a new venue.

## Official contract and deployment evidence

### Address book

The canonical address source is the Aave DAO address book at commit:

```text
21004a8871ca774b8da27114cbbd74931f3a436f
```

Pinned file:

<https://raw.githubusercontent.com/aave-dao/aave-address-book/21004a8871ca774b8da27114cbbd74931f3a436f/src/AaveV3Ethereum.sol>

The exact addresses are:

| identity | address |
|---|---|
| Aave V3 Ethereum Pool proxy | `0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2` |
| WBTC reserve | `0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599` |
| WETH reserve | `0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2` |
| USDC reserve | `0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48` |
| USDT reserve | `0xdAC17F958D2ee523a2206206994597C13D831ec7` |

The pinned address-book file SHA-256 is:

```text
1a862b7389de3d59ee77680a8edb451a29e630a07813a0c5becdce65d730a22a
```

The address-book license at the same commit is MIT:

<https://raw.githubusercontent.com/aave-dao/aave-address-book/21004a8871ca774b8da27114cbbd74931f3a436f/LICENSE>

Pinned license SHA-256:

```text
719dd01a9b4549c3f489ff04420d4e8f010d91a368f57894505fe691e19b6c48
```

### Pool event ABI

The current canonical interface source is Aave V3 Origin at commit:

```text
fd1fbd9150426ca8ace9cee45b4acf912ae84f5b
```

Pinned interface:

<https://raw.githubusercontent.com/aave-dao/aave-v3-origin/fd1fbd9150426ca8ace9cee45b4acf912ae84f5b/src/contracts/interfaces/IPool.sol>

Pinned interface SHA-256:

```text
ee58bacafc0b033adeaf673da3b08130d9f3a9a3eb563f7f19a03240e488eba7
```

The archived Aave V3 Core interface at commit:

```text
782f51917056a53a2c228701058a6c3fb233684a
```

contains the identical event signature:

<https://raw.githubusercontent.com/aave/aave-v3-core/782f51917056a53a2c228701058a6c3fb233684a/contracts/interfaces/IPool.sol>

Pinned archived-interface SHA-256:

```text
d708ba2c3cf29fe81083b7b8127ef61b7d750dc017399ebaa2951d6f61a93dda
```

The verifier must hash and compare both interface specimens. Any disagreement
between the archived and current event declarations rejects the source before
RPC log access.

The exact non-anonymous event is:

```solidity
event ReserveDataUpdated(
  address indexed reserve,
  uint256 liquidityRate,
  uint256 stableBorrowRate,
  uint256 variableBorrowRate,
  uint256 liquidityIndex,
  uint256 variableBorrowIndex
);
```

In each pinned interface, remove Solidity line and block comments, extract the
single declaration from the `event ReserveDataUpdated` token through its first
semicolon, replace every non-empty ASCII whitespace run with one U+0020 space,
and trim both ends. Both normalized declarations must equal exactly:

```text
event ReserveDataUpdated( address indexed reserve, uint256 liquidityRate, uint256 stableBorrowRate, uint256 variableBorrowRate, uint256 liquidityIndex, uint256 variableBorrowIndex );
```

Its exact Keccak-256 topic is:

```text
0x804c9b842b2748a22bb64b345453a3de7ca54a6ca45ce00d415894979e22897a
```

The Aave V3 Ethereum market launched on 2023-01-27. WBTC, WETH, and USDC
were initial assets; USDT was subsequently added through Aave governance.

- <https://governance.aave.com/t/arc-chaos-labs-risk-parameter-updates-aave-v3-ethereum-2023-02-22/12015>
- <https://governance.aave.com/t/arc-aave-ethereum-v3-market-initial-onboarded-assets/11318>
- <https://governance.aave.com/t/arfc-add-usdt-to-ethereum-v3-market/11536>

This gives the source family more than three years of possible finalized
history before the fixed recent window below. The source verifier must still
prove actual event availability; launch documentation cannot substitute for
event support.

## Frozen JSON-RPC transport

The primary read-only Ethereum RPC is the same public mainnet endpoint shown
in the now-archived Aave utilities example:

```text
https://eth-mainnet.public.blastapi.io
```

Pinned archived Aave example:

<https://raw.githubusercontent.com/aave/aave-utilities/2d0db501009fdb34824cbc9a486c3e9fd7191ec7/README.md>

Pinned README SHA-256:

```text
42f2a95d8938619b074e143f31388c02c39c38bfde731dccefea594efd3963fe
```

That repository is explicitly deprecated and was archived in April 2026. It
is evidence only that the endpoint appeared in an Aave example, not a current
support commitment.

The independent parity RPC is:

```text
https://ethereum-rpc.publicnode.com
```

PublicNode's Ethereum endpoint directory:

<https://ethereum.publicnode.com/>

Both endpoints returned chain ID `0x1` before this boundary. Neither endpoint
may be replaced inside the AV3BRL-v1 research verification after source values
are opened.

These keyless endpoints are **research-only transports**. Their appearance in
an Aave example or public-node directory is not a production SLA, archival
guarantee, or legal opinion. AV3BRL-v1 does not authorize a real-money collector
to depend on either host.

Before any live shadow or order admission, a separately committed
`AV3BRL-LIVE-v1` deployment boundary must:

- bind two self-hosted or contractually authorized Ethereum archival RPCs;
- record their rate limits, retention, terms, and operational contacts;
- replay the fixed historical and recent windows against the frozen hashes;
- prove a prospective finalized-head overlap with the research ledger;
- define stale-source, provider-disagreement, restart, and fail-flat behavior;
  and
- preserve the exact chain, Pool, reserve, event, decoding, and candidate-field
  identity frozen here.

That later transport qualification may not alter or rescue the AV3BRL-v1
research result. Until it passes, all AV3BRL actions are research-only and
force live `ABSTAIN`.

Only these methods are authorized:

- `eth_chainId`;
- `eth_getBlockByNumber`; and
- `eth_getLogs`.

The Ethereum JSON-RPC contract for finalized block tags, log fields, contract
addresses, and ordered topics is:

<https://ethereum.org/developers/docs/apis/json-rpc/>

The verifier must:

1. send JSON-RPC 2.0 POST requests with explicit IDs;
2. cap each host at two requests per second;
3. use no API key, cookie, browser session, or private account;
4. reject a non-`0x1` chain ID;
5. reject JSON-RPC errors, a response ID different from the request ID,
   batched responses, partial bodies, malformed
   hex quantities, null finalized fields, or provider disagreement;
6. retry only a pre-body connection failure, HTTP 429, or HTTP 5xx, at most
   twice with deterministic one- and two-second waits; and
7. never retry or repair a decoded-content disagreement.

Every HTTP request has a 10-second connection timeout and a 60-second body
timeout, accepts at most 64 MiB, requires one UTF-8 JSON object with no
duplicate keys, and rejects every non-200 status not covered by the retry
rule. JSON-RPC batches are forbidden.

Every `eth_getBlockByNumber` call passes `false` for full transactions. Every
`eth_getLogs` window is split into non-overlapping inclusive chunks of exactly
512 blocks, starting at the frozen first block; only the final chunk may be
shorter. The exact filter is:

```json
{
  "address": "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2",
  "fromBlock": "<lowercase minimal chunk-start quantity>",
  "toBlock": "<lowercase minimal chunk-end quantity>",
  "topics": [
    "0x804c9b842b2748a22bb64b345453a3de7ca54a6ca45ce00d415894979e22897a",
    [
      "0x0000000000000000000000002260fac5e5542a773aa44fbcfedf7c193bc2c599",
      "0x000000000000000000000000c02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
      "0x000000000000000000000000a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
      "0x000000000000000000000000dac17f958d2ee523a2206206994597c13d831ec7"
    ]
  ]
}
```

Serialize the request object with UTF-8, sorted keys, compact separators, and
no NaN. Request IDs are JSON integers:

```text
eth_chainId                                  1
eth_getBlockByNumber("finalized", false)     2
eth_getBlockByNumber(block, false)           100,000,000 + block
eth_getLogs(512-block parent chunk)           200,000,000 + chunk_ordinal
eth_getLogs(128-block subdivision)            300,000,000 + 4 * chunk_ordinal + subdivision_ordinal
```

For Stage A, historical chunks have ordinals `0..13` and recent chunks have
ordinals `14..28`. For Stage B, full-history chunks have ordinals
`0..16,054`. Within each parent chunk, non-empty subdivisions have ordinals
`0..3` in ascending block order; each contains at most 128 blocks and only the
last may be shorter. Primary and parity calls use the same ID and params. Exact
block requests are deduplicated per provider, so a block number has one
deterministic request ID in a stage.

Every accepted block header must contain non-null `number`, `hash`,
`parentHash`, and `timestamp`. Validate hashes as exact 32-byte strings and
quantities with the same minimal-hex rules as logs. The requested exact block
number must equal the decoded `number`; extra header keys are ignored and
forbidden from downstream use.

Changing chunk size, using overlapping chunks, issuing one request per reserve,
or accepting a provider-truncated response is forbidden.

Every 512-block parent query has deterministic subdivision redundancy. For
each provider independently, canonicalize the parent response and the
concatenated canonical responses from its non-overlapping 128-block
subdivisions. They must be exactly equal. The primary and parity parent
streams, every corresponding subdivision stream, and both reconstructed
subdivision unions must all agree. A missing, duplicate, out-of-range, or
conflicting event in any view rejects the stage. This check is mandatory even
when the parent response is empty and is never repaired by a smaller ad hoc
range.

## Exact log schema

Every accepted log must have:

- `address` exactly equal to the Pool proxy;
- exactly two topics;
- `topics[0]` exactly equal to the frozen event topic;
- `topics[1]` equal to the left-zero-padded 32-byte encoding of one frozen
  reserve address;
- exactly five 32-byte ABI words in `data`;
- non-null `blockNumber`, `blockHash`, `transactionHash`,
  `transactionIndex`, and `logIndex`; and
- `removed == false`.

Decode the five unsigned 256-bit words in this order:

1. `liquidityRate`;
2. `stableBorrowRate`;
3. `variableBorrowRate`;
4. `liquidityIndex`; and
5. `variableBorrowIndex`.

Zero is valid. Negative values, signed reinterpretation, floating-point
decoding, decimal text coercion, truncation, overflow, or provider-specific
field repair are forbidden.

Validate the Pool `address` as exactly 20 bytes, each topic, block hash, and
transaction hash as exactly 32 bytes, and `data` as exactly 160 bytes, all
`0x`-prefixed with two hexadecimal characters per byte. Then lowercase those
fixed-size byte strings for canonical comparison. Validate JSON-RPC quantities
as minimal nonnegative hexadecimal without leading zeroes, parse them as
integers, and re-encode them as lowercase minimal quantities. Canonicalization
may change hexadecimal letter case only; it may not change bytes or numeric
values. Provider-specific extra log keys are ignored after the required fields
pass validation and are forbidden from every hash, report, ledger, or feature.

Canonical ordering is:

```text
(blockNumber, transactionIndex, logIndex)
```

The canonical identity is:

```text
(blockHash, transactionHash, logIndex)
```

Duplicate canonical identities, conflicting duplicates, or any primary/parity
RPC difference rejects AV3BRL-v1. Provider response order is not economic data:
each valid response is sorted into canonical order before parity comparison,
and raw list ordering is neither a pass nor a rejection condition.

For Stage A, after canonical parity, query block headers from both RPCs for the
first and last log of every `(window, reserve)` group and every 256th log in
each complete window stream. Deduplicate that deterministic audit set by block
number. Every audited header's number, hash, parent hash, and timestamp must
agree across providers, and its hash must equal the log's block hash.

Stage B does not sample this proof: it performs the same validation for every
distinct event block before a ledger row can be published.

## Candidate-field boundary

The source verifier must decode and compare all event words, but the first
future AV3BRL mechanism may use only:

- reserve identity;
- finalized block timestamp; and
- `variableBorrowRate`.

The following are parity-only and forbidden from the first candidate:

- `liquidityRate`;
- `stableBorrowRate`;
- `liquidityIndex`;
- `variableBorrowIndex`;
- transaction or log identifiers;
- block spacing, gas, calldata, sender, recipient, or transaction value; and
- Aave `Borrow`, `Repay`, `Supply`, `Withdraw`, liquidation, oracle, balance,
  debt-token, aToken, or governance events.

`blockNumber`, `transactionIndex`, and `logIndex` remain mandatory internal
ordering and tie-break metadata for exact state reconstruction. Block and
transaction hashes remain mandatory integrity metadata. They are forbidden as
features, tokens, thresholds, labels, actions, or model inputs, not forbidden
from the source state machine.

The first reconstructible state starts only after all four reserves have
emitted at least one canonical event inside the frozen full-history range.
Within that range, the latest state before an anchor is the last event in
canonical order whose finalized block is strictly earlier than the anchor's
first included block. No event before the frozen full-history start may seed a
state; an incomplete four-reserve state forces `ABSTAIN`.

This prevents a source pass from silently becoming a flow, utilization,
liquidation, fee, holder, or transaction-identity search.

## Prospectively frozen full-history reconstruction range

The initial source verifier below is deliberately a two-window
schema/transport parity gate. **Its pass does not prove complete history and
does not authorize a backtest.**

Before opening full-history incidence or rate values, an exact AV3BRL
mechanism decision and preregistration must be committed. That preregistration
must bind this already-frozen reconstruction contract without changing it.
Only then may the separate full-history builder execute.

The future full-history inclusive range is:

```text
17,382,266 .. 25,602,012
0x1093b7a .. 0x186a7dc
```

It covers:

```text
[2023-06-01T00:00:00Z, 2026-07-24T10:30:48Z)
```

The first block is the first block at or after the UTC start. The final block
is the fixed finalized block bound below. The range contains exactly:

```text
8,219,747 blocks
16,055 non-overlapping 512-block chunks
```

The full-history builder processes chunks strictly from oldest to newest. For
each chunk:

1. fetch the parent filter and its fixed 128-block subdivisions from both
   providers;
2. validate the JSON-RPC envelope and response ID;
3. decode and canonical-sort every parent and subdivision event set;
4. require parent/subdivision-union equality within each provider and exact
   cross-provider equality of every view, event field, and decoded ABI word;
5. reject any event outside the chunk;
6. reject duplicate identities within or across chunks;
7. compute a per-chunk domain-separated hash; and
8. fetch and compare both providers' exact block header for every distinct
   event block in the chunk before appending compact candidate-ledger rows to a
   deterministic gzip stream.

The per-chunk hash preimage is:

```text
UTF8("AV3BRL-v1\0chunk\0")
+ uint64_be(chunk_start)
+ uint64_be(chunk_end)
+ for each canonical event:
     uint32_be(canonical_json_byte_length)
     + canonical_event_json_bytes
```

`canonical_event_json_bytes` is UTF-8 JSON with sorted keys, compact
separators, no NaN, lowercase fixed-size hex, decimal JSON integers for parsed
quantities and all five ABI words, and exactly these keys:

```text
address
block_hash
block_number
data_words
log_index
reserve
topics
transaction_hash
transaction_index
```

`data_words` is a JSON array of exactly five decimal JSON integers in this
positional ABI order:

```text
[liquidityRate,stableBorrowRate,variableBorrowRate,liquidityIndex,variableBorrowIndex]
```

It is never an object and contains no strings or field labels. `topics` is a
JSON array of exactly the two canonical lowercase 32-byte topic strings.
`reserve` is the canonical lowercase 20-byte address decoded from `topics[1]`.

The complete parity hash is SHA-256 over:

```text
UTF8("AV3BRL-v1\0full-history\0")
+ concatenated 32-byte per-chunk hash digests in chunk order
```

Cross-provider equality is therefore over canonical decoded event sets, not
raw HTTP bytes or provider list ordering.

For each fixed parity window, compute a chunk-boundary-independent stream hash
over only the canonical events inside that exact window:

```text
UTF8("AV3BRL-v1\0window\0" + window_name + "\0")
+ for each canonical event:
    uint32_be(canonical_json_byte_length)
    + canonical_event_json_bytes
```

`window_name` is exactly `historical` or `recent`. Stage B must reproduce both
Stage A window-stream hashes exactly from the full-history stream.

### Compact source ledger

The local ignored gzip contains exactly this UTF-8 CSV header:

```text
block_number,block_timestamp,transaction_index,log_index,block_hash,transaction_hash,reserve,variable_borrow_rate
```

Rows are written in global canonical order with:

- decimal ASCII integers without leading zeroes;
- `block_timestamp` as the event block's nonnegative Unix-seconds integer;
- lowercase fixed-size hashes and reserve addresses;
- RFC 4180-compatible commas and LF line endings; and
- no extra columns, comments, blank rows, index, or BOM.

Use gzip compression level 9, `mtime=0`, and an empty embedded filename. Hash
the final gzip bytes with SHA-256. The file contains source values and remains
ignored and local; only its byte length and SHA-256 may appear in the terminal
report. Ordering identifiers are integrity metadata and may not become model
features.

Every distinct event block header must be fetched from both providers. Its
number, hash, parent hash, and timestamp must agree, and its hash must equal
the log's `blockHash`. A missing, malformed, disagreeing, or non-monotone
header rejects Stage B. The ledger timestamp therefore comes from two-provider
header parity, not provider-local receipt time or an inferred average block
interval.

### Monthly continuity

For every UTC month start from 2023-06-01 through 2026-07-01, find the first
block whose timestamp is greater than or equal to the month start by
deterministic lower-bound binary search over the frozen range. Verify on both
providers that:

- the selected block number, hash, and timestamp agree;
- the selected timestamp is at or after the boundary; and
- its predecessor timestamp is strictly before the boundary.

The complete months June 2023 through June 2026 and the partial July 2026
interval through the frozen final block must each contain at least one
canonical event for every frozen reserve. Any missing reserve-month rejects the
source unchanged.

The full-history pass must also prove that every event assigned to a month lies
within that month's exact block boundaries. This is source availability only;
monthly event counts and values remain hidden.

## Fixed historical parity window

The half-open UTC interval is:

```text
[2023-06-01T00:00:00Z, 2023-06-02T00:00:00Z)
```

Its exact inclusive block range is:

```text
17,382,266 .. 17,389,364
0x1093b7a .. 0x1095734
```

Boundary headers:

| role | block | timestamp UTC | hash |
|---|---:|---|---|
| previous | 17,382,265 | 2023-05-31 23:59:59 | `0x5b54c7cbb1816c36c6b5431e31ca12b1162b8d87a305318702d954e1f2b0b0cc` |
| first included | 17,382,266 | 2023-06-01 00:00:11 | `0xe0ef11cab4909c80599087b4ffb0bf1e92b1affcc72abc3b802f20a9d5d21096` |
| last included | 17,389,364 | 2023-06-01 23:59:59 | `0x34e4ed79dbd53a7fd5e8455b3d01b44ce186d8b79e7b186c8b153ca869d91cfc` |
| first excluded | 17,389,365 | 2023-06-02 00:00:11 | `0x73be93661691df088828824d1bd7ff3cd555426678744f8ff7aa0a2dde15643b` |

Both frozen RPCs agreed on all four header hashes before this boundary.

## Fixed recent parity window

The fixed finalized inclusive range is:

```text
25,594,813 .. 25,602,012
0x1868bbd .. 0x186a7dc
```

Boundary headers:

| role | block | timestamp UTC | hash |
|---|---:|---|---|
| first included | 25,594,813 | 2026-07-23 10:23:35 | `0x048f2cdcafc8b22de300cb6bf9b7c4d60cdba5b7d2ecb99871b152e586ce30a2` |
| last included | 25,602,012 | 2026-07-24 10:30:47 | `0x53d736133c333e9751920c0a45398938533a04c0601599aafe22636c0d43ddad` |

Both frozen RPCs agreed on these fixed block headers before the boundary.
During execution, each RPC's current `finalized` head must be at or beyond the
frozen final block and the frozen final block hash must still match. The source
run uses the exact fixed range, not a moving recent window.

## Two-stage source verification gates

AV3BRL-v1 has two separate one-shot execution identities. Stage A proves only
that the frozen schema and transports agree in two fixed windows. Stage B,
which is forbidden until an exact mechanism and preregistration are committed,
proves the complete frozen-history reconstruction contract. A Stage A pass is
not a full-history pass, source-support pass, backtest authorization, or alpha
claim.

### Stage A — fixed-window source parity

The Stage A paths are:

```text
training/verify_aave_v3_borrow_rate_ledger.py
tests/test_verify_aave_v3_borrow_rate_ledger.py
results/.aave_v3_borrow_rate_ledger_source_parity_2026-07-24.started
results/.aave_v3_borrow_rate_ledger_source_parity_2026-07-24.json.tmp
results/aave_v3_borrow_rate_ledger_source_parity_2026-07-24.json
```

The verifier must, in this order:

1. enforce a committed, HEAD-clean source-boundary and verifier guard;
2. create the Stage A sentinel with exclusive-create semantics;
3. enforce the exact disk guard below;
4. verify every pinned source hash, normalize and compare the archived/current
   event declarations, and recompute the event topic;
5. verify chain ID, the fixed boundary headers, and that each provider's
   current `finalized` head is at or beyond block `25,602,012`;
6. request the 14 historical-window parent chunks and every fixed subdivision
   from both providers;
7. request the 15 recent-window parent chunks and every fixed subdivision from
   both providers;
8. validate, decode, canonical-sort, and require exact cross-provider equality
   plus parent/subdivision-union equality for every returned event field and
   ABI word;
9. require the subdivision-redundancy gate to pass for every parent chunk;
10. require at least one canonical event for each frozen reserve in each fixed
   window;
11. compute the two domain-separated canonical window-stream hashes;
12. execute the deterministic event-block header audit over both complete
    window streams;
13. record every outcome/model/economics access flag as false; and
14. write only the Stage A terminal report.

Stage A writes no ledger, decoded row, event count, event timestamp, or rate
value. Its pass authorizes only the next exact mechanism decision and
preregistration. It does not authorize the Stage B RPC run until those
documents are committed and hash-bound.

### Stage B — preregistered full-history reconstruction

The Stage B paths are:

```text
training/build_aave_v3_borrow_rate_ledger.py
tests/test_build_aave_v3_borrow_rate_ledger.py
data/aave_v3_borrow_rate_ledger_full_history_2026-07-24/
data/aave_v3_borrow_rate_ledger_full_history_2026-07-24/ledger.csv.gz
data/aave_v3_borrow_rate_ledger_full_history_2026-07-24/report.json
data/.aave_v3_borrow_rate_ledger_full_history_2026-07-24.staging/
results/.aave_v3_borrow_rate_ledger_full_history_2026-07-24.started
```

Stage B may execute only when:

- the immutable Stage A report is `PASS` and its exact file SHA-256 is frozen
  in the committed builder;
- one exact mechanism-decision document and one exact preregistration document
  are committed;
- their repository-relative paths and exact file SHA-256 values are frozen in
  the committed builder; and
- neither document has opened an Aave incidence, count, event time, rate value,
  market outcome, funding row, return, PnL, model result, or economics metric.

The builder must, in this order:

1. enforce the committed, HEAD-clean boundary, Stage A report, mechanism,
   preregistration, builder, and test bindings;
2. create the Stage B sentinel with exclusive-create semantics;
3. enforce the exact disk guard;
4. reverify all pins, chain identity, finalized-head floor, and fixed boundary
   headers on both providers;
5. process all 16,055 full-history parent chunks and their fixed subdivisions
   from oldest to newest;
6. validate, decode, canonical-sort, compare, and hash every event exactly,
   requiring subdivision redundancy for every parent;
7. fetch both providers' headers for every distinct event block and require
   exact number, hash, parent-hash, and timestamp parity;
8. prove the frozen monthly-continuity conditions for every reserve;
9. prove both fixed windows and their per-reserve support from the reconstructed
   canonical stream;
10. reproduce the two Stage A window-stream hashes exactly;
11. record every outcome/model/economics access flag as false;
12. write and `fsync` the compact gzip ledger and terminal report inside one
    hidden sibling staging directory;
13. atomically publish that complete Stage B bundle by one directory rename
    followed by parent-directory `fsync`.

Only a Stage B pass authorizes source-support statistics, pre-outcome novelty,
or any later market-data access.

### Independent one-shot state machines

Stage A and Stage B each use a separate state machine and artifact set. The
only states are:

```text
NOT_STARTED
STARTED
TERMINAL_PASS
TERMINAL_REJECT
STARTED_ORPHANED
```

- `NOT_STARTED`: the corresponding stage has no sentinel, report, unpublished
  staging artifact, or, for Stage B, final bundle directory.
- Exclusive creation of the corresponding sentinel changes that stage to
  `STARTED`. The sentinel is immutable and is never removed.
- A complete Stage A pass writes its report and becomes `TERMINAL_PASS`.
- A complete Stage B pass atomically publishes one final directory containing
  exactly `ledger.csv.gz` and `report.json`, then becomes `TERMINAL_PASS`.
- Any caught exception, timeout, retry exhaustion, partial provider success,
  partial canonicalization, hash mismatch, user interrupt, or disk breach after
  sentinel creation writes a rejection report and becomes `TERMINAL_REJECT`.
- Process death or power loss after sentinel creation but before a report is
  `STARTED_ORPHANED`, which is semantically terminal rejection. A later
  invocation must refuse before any source access and may not resume, delete
  the sentinel, reuse partial bytes, or rerun that stage.
- Any inconsistent pre-existing artifact combination also refuses before
  source access.
- Failure before exclusive sentinel creation leaves that stage `NOT_STARTED`
  and opens no source.

After entering `STARTED`, each runner catches `Exception`,
`KeyboardInterrupt`, and `SystemExit`, removes only its own unpublished
temporary bytes, and attempts its terminal rejection report. A report-write
failure leaves `STARTED_ORPHANED`. No code path can make an orphaned or rejected
stage pass.

For Stage B, a caught failure deletes the unpublished staging ledger, writes a
rejection `report.json` into a fresh staging directory, and atomically renames
that report-only directory to the final bundle path. A Stage B pass bundle has
exactly two files; a rejection bundle has exactly one. Source-value bytes can
never appear at the final ledger path without the same atomic publication also
containing its hash-bound pass report. After an uncatchable orphan, a separate
offline cleanup may delete only the unpublished staging directory without RPC
access; the sentinel remains and Stage B stays permanently rejected.

The staging and final bundle directories are siblings on the repository
filesystem. The builder `fsync`s each file, `fsync`s the staging directory,
requires the final path not to exist, calls Linux `renameat2` with
`RENAME_NOREPLACE` for one same-filesystem directory rename, then `fsync`s the
parent directory. Missing `renameat2` support rejects before source access.
Cross-device moves, copy-then-delete publication, per-file publication, and
overwrite semantics are forbidden.

Stage A uses the same `fsync` and `renameat2(RENAME_NOREPLACE)` contract to
publish its exact temporary report file to the final report path. Support for
that syscall and same-filesystem staging is checked before either stage opens a
source.

### Exact disk guard

Let `REPOSITORY_ROOT` be the resolved parent of the committed `training`
directory. At preflight, before every `eth_getLogs` request pair, and before
every Stage B temporary-ledger flush, call:

```python
shutil.disk_usage(REPOSITORY_ROOT)
```

Pass only when:

```text
used < 300 * 1024**3
free >= 1 * 1024**3
```

Use the returned `used` and `free` fields directly. `du`, reserved-block
arithmetic, WSL virtual-disk maximum size, decimal GB, another mount, or
project-directory apparent size may not replace this check.

### Canonical terminal reports

Stage A uses protocol version:

```text
av3brl_source_parity_v1
```

Stage B uses protocol version:

```text
av3brl_full_history_v1
```

Serialize UTF-8 JSON with sorted keys, compact separators,
`ensure_ascii=False`, `allow_nan=False`, and one trailing LF. The top-level
keys are exactly:

```text
protocol_version
status
state
source_boundary_sha256
verifier_commit
started_at_utc
terminal_at_utc
pins
bindings
range
providers
schema_sha256
window_streams
full_history_sha256
ledger_gzip_sha256
ledger_gzip_bytes
gates
failure
forbidden_access
manifest_sha256
```

`status` is `PASS` or `REJECT`; `state` is `TERMINAL_PASS` or
`TERMINAL_REJECT`. `STARTED_ORPHANED` has no terminal report by definition;
it is inferred from a sentinel without a report. `verifier_commit` is the
exact 40-lowercase-hex commit containing the executed Stage A verifier or
Stage B builder and its tests. Timestamps use UTC
`YYYY-MM-DDTHH:MM:SSZ`.

`failure` is null on pass and otherwise contains exactly `stage`,
`exception_type`, and `message`. `stage` is mapped to exactly one of:

```text
preflight
source_read
canonicalization
support
header_validation
publication
terminalization
```

`exception_type` is mapped to exactly one of:

```text
ProtocolError
TransportError
SchemaError
ParityError
SupportError
HeaderError
DiskGuardError
PublicationError
InterruptedError
UnexpectedError
```

`message` is the corresponding generic ASCII phrase:

```text
protocol precondition failed
transport failed
schema validation failed
canonical parity failed
support gate failed
header validation failed
disk guard failed
atomic publication failed
interrupted
unexpected failure
```

Raw exception text is never serialized. Failure metadata may not identify a
provider URL query, reserve, window, month, block, chunk, transaction, log,
event presence, count, timestamp, rate, or source value. `forbidden_access`
contains false booleans for market, funding, outcome, return, PnL, reward,
model, checkpoint, CAGR, and MDD access. `gates` contains only the exact named
booleans for that protocol version; it contains no counts or values.

On any rejection after source access begins, all source-content-derived gates
(`subdivision_redundancy`, `historical_window`, `recent_window`,
`event_header_audit`, `full_history_parity`, and `monthly_continuity` when
present) are serialized as false regardless of partial progress. Reports never
publish which source window, reserve, month, block, or chunk triggered failure.

The nested keys are:

```text
pins:
  address_book_sha256
  address_book_license_sha256
  aave_utilities_readme_sha256
  archived_ipool_sha256
  current_ipool_sha256
  event_topic

bindings:
  source_parity_report_path
  source_parity_report_sha256
  mechanism_document_path
  mechanism_document_sha256
  preregistration_document_path
  preregistration_document_sha256

range:
  chunk_size
  subdivision_size
  historical_window
  recent_window
  full_history

range.<each window>:
  first_block
  first_block_hash
  last_block
  last_block_hash
  chunk_count

providers:
  primary
  parity
  chain_id
  canonical_streams_equal

window_streams:
  historical_sha256
  recent_sha256

forbidden_access:
  market
  funding
  outcome
  return
  pnl
  reward
  model
  checkpoint
  cagr
  mdd
```

Stage A `bindings` values are all JSON null. Stage B `bindings` values are the
committed repository-relative paths and lowercase SHA-256 values described
above. The exact Stage A `gates` keys are:

```text
committed_protocol
clean_head
disk
pinned_sources
chain_identity
boundary_headers
subdivision_redundancy
historical_window
recent_window
event_header_audit
forbidden_access
```

The exact Stage B `gates` keys are:

```text
committed_protocol
clean_head
disk
source_parity_binding
mechanism_preregistration_binding
pinned_sources
chain_identity
boundary_headers
subdivision_redundancy
full_history_parity
monthly_continuity
historical_window
recent_window
event_header_audit
atomic_ledger
forbidden_access
```

The range constants are:

| range | first block | first hash | last block | last hash | chunks |
|---|---:|---|---:|---|---:|
| historical window | 17,382,266 | `0xe0ef11cab4909c80599087b4ffb0bf1e92b1affcc72abc3b802f20a9d5d21096` | 17,389,364 | `0x34e4ed79dbd53a7fd5e8455b3d01b44ce186d8b79e7b186c8b153ca869d91cfc` | 14 |
| recent window | 25,594,813 | `0x048f2cdcafc8b22de300cb6bf9b7c4d60cdba5b7d2ecb99871b152e586ce30a2` | 25,602,012 | `0x53d736133c333e9751920c0a45398938533a04c0601599aafe22636c0d43ddad` | 15 |
| full history | 17,382,266 | `0xe0ef11cab4909c80599087b4ffb0bf1e92b1affcc72abc3b802f20a9d5d21096` | 25,602,012 | `0x53d736133c333e9751920c0a45398938533a04c0601599aafe22636c0d43ddad` | 16,055 |

`chunk_size` is the JSON integer `512`; `subdivision_size` is `128`. Provider
values are the two frozen base URLs without query strings. On `PASS`, every
result SHA field is a lowercase 64-hex string. On `REJECT`, `window_streams`,
`full_history_sha256`, `ledger_gzip_sha256`, and `ledger_gzip_bytes` are all
JSON null, regardless of partial work; no partial source commitment is
published. `providers.chain_id` is the expected JSON integer `1`, while
`providers.canonical_streams_equal` is true only on `PASS` and false on
`REJECT`. Block, chunk, byte, and chain identifiers are JSON integers. All
non-null booleans are JSON booleans.

The schema digest is SHA-256 over these exact UTF-8 bytes:

```text
AV3BRL-v1\0schema\0event_keys=address,block_hash,block_number,data_words,log_index,reserve,topics,transaction_hash,transaction_index\n
data_words=liquidity_rate,stable_borrow_rate,variable_borrow_rate,liquidity_index,variable_borrow_index\n
ledger_header=block_number,block_timestamp,transaction_index,log_index,block_hash,transaction_hash,reserve,variable_borrow_rate\n
```

The displayed `\0` and `\n` are one NUL byte and one LF byte respectively;
there are no Markdown fence bytes or extra spaces in the preimage.
The required `schema_sha256` is:

```text
33e62c18f3e221a646ac3f18e90fe1cf447c6c17bc1b14ca52cb72fcb0775590
```

Stage A sets `full_history_sha256`, `ledger_gzip_sha256`, and
`ledger_gzip_bytes` to JSON null. Stage B requires the lowercase full-history
SHA-256, lowercase gzip SHA-256, and gzip byte count.

Compute `manifest_sha256` over:

```text
UTF8("AV3BRL-v1\0terminal-report\0")
+ canonical_report_bytes_with_manifest_sha256_omitted
```

For that preimage, remove the `manifest_sha256` key and value entirely from a
copy of the top-level object, then serialize the remaining object with the
same sorted-key, compact, UTF-8, no-NaN settings and **no trailing LF**. Do not
set the key to null or a placeholder. Insert the lowercase digest into the
original object and serialize the final report with one trailing LF. Hashes
never include nondeterministic HTTP headers, raw provider bodies, temporary
paths, or wall-clock progress output.

No source count, rate distribution, event time, or decoded rate value may be
printed to the terminal or committed. Stage A progress may contain only the
current fixed-window chunk index, total fixed-window chunk count, elapsed
monotonic time, disk-guard status, and terminal stage. Stage B progress may
contain the same fields for the full-history chunk schedule. Reports may
contain only the frozen metadata, pass/fail booleans, hashes, byte count, and
bounded terminal failure text defined above.

Failure retires the corresponding AV3BRL-v1 stage unchanged. The following are
forbidden repairs:

- changing a reserve set;
- changing either block window or the full-history range;
- relaxing exact provider parity;
- switching RPC hosts;
- admitting unfinalized blocks;
- allowing a different Pool address or event ABI;
- dropping an event word;
- replacing missing events with `eth_call`;
- opening Aave V2, another chain, another lending protocol, or an indexed
  third-party API under the AV3BRL identity; or
- moving directly to rates, outcomes, or a trading mechanism after a failed
  prerequisite stage.

## Live causal reconstruction contract

If and only if Stage B and the separately committed `AV3BRL-LIVE-v1`
transport qualification both pass, a later mechanism may define a daily or
intraday causal state from the latest **finalized** `ReserveDataUpdated` event
strictly available before its decision anchor.

Production collection must:

1. poll the qualified live primary endpoint's finalized head;
2. verify the same finalized block hash on the qualified live parity endpoint;
3. request every not-yet-processed finalized block without gaps;
4. persist the last processed block number and hash atomically;
5. replay from the last durable finalized block after a restart;
6. halt on reorg evidence, provider disagreement, schema drift, or stale
   collection; and
7. never substitute current `eth_call` state into a historical timestamp.

No event may become actionable before its containing block is finalized and
the complete source transform has finished. A later mechanism must separately
freeze the exact decision latency, stale-state policy, opportunity clock,
tokens, side/action space, and execution market before any BTC outcome is read.

## Novelty and RLLM boundary

After a Stage B pass, the preregistered mechanism must remain a
**cross-reserve relational borrow-cost state**, not a thresholded copy of
price, funding, OI, taker flow, stablecoin supply, or prior DeFi clocks.

Before outcomes and after Stage B:

- bind the exact comparator registry already enumerated in the novelty table of
  `docs/paired-intrinsic-venue-orderflow-topology-mechanism-decision-2026-07-24.md`
  whose SHA-256 is
  `7d9cbf6ea3ad3ad938f52c80bf76bd1585d6adc8d55ac6bcb4df888112990d02`;
- include the frozen live-sleeve action clock
  `results/cchr_live_portfolio_pure_clocks_2020_2026.csv.gz` with expected
  SHA-256
  `73d6efbd35b3be64b0fa04fa9c8cb2db25866ef884f19b1ae673949e22a42b08`
  and manifest
  `results/cchr_live_portfolio_pure_clock_manifest_2026-07-21.json` with
  SHA-256
  `6c53ae482cf72bba0f286a47626842bf43070276ff5fe359be718e44864af57d`;
- reject before outcomes if any required non-forbidden comparator is missing
  or has a hash mismatch;
- require exact-entry Jaccard strictly below `0.35` against every comparator;
- under deterministic one-to-one nearest matching within `±6h`, require both
  candidate-to-comparator and comparator-to-candidate matched fractions
  strictly below `0.50`;
- after removing the union of all tolerant matches, retain at least 40
  candidate actions and at least 50% of the frozen candidate action clock;
- include future-append, reserve-order permutation, sign/rank mirror,
  timestamp-delay, stale-state, and deterministic-random controls;
- use strictly prior transforms only; and
- reject if any UTC month holds more than 20% of frozen opportunities;
- require every reserve token to have at least two non-missing levels with each
  level holding at least 5% in train, selection, and sealed evaluation
  separately; and
- reject any mechanism that thresholds an absolute raw rate or whose strongest
  single-reserve or leave-one-reserve-out cheap baseline matches or exceeds the
  full relation baseline on the frozen selection ordering.

Nearest tolerant matching sorts both clocks by UTC entry, advances the earlier
unmatched entry, and matches a pair only when the absolute time difference is
at most six hours. Each row can match once. Comparator clocks are evaluated
separately; the union-removal gate removes a candidate row matched by any
comparator. These definitions may not be changed after source incidence.

Because the mechanism and preregistration are frozen before Stage B, any
post-Stage-B source-support, novelty, control, cheap-baseline, selection, or
sealed-evaluation rejection retires that AV3BRL-v1 mechanism permanently. No
new threshold, tokenization, opportunity clock, side, hold, model target, or
replacement mechanism may be designed from the opened Aave incidence or rate
values. A later attempt requires a genuinely new source identity and a new
pre-source boundary; it may not reuse or inspect the rejected AV3BRL ledger.

A later LLM path may use one compact text model only after a cheap deterministic
baseline proves learnability. The model may receive frozen ordinal relation
tokens plus current position/risk state and choose only:

```text
TRADE_FIXED_SIDE
ABSTAIN
```

The LLM may not generate a side, threshold, hold, feature, rationale, or
free-form strategy. Analyzer/trader dual-model architecture is forbidden.

## Storage and publication

- Raw RPC responses and decoded event rows remain local and ignored.
- No Aave log row or reconstructed rate series is committed or redistributed.
- Only code, protocol documents, tests, hashes, and bounded pass/rejection
  metadata may be committed.
- For Stage B, only
  `data/aave_v3_borrow_rate_ledger_full_history_2026-07-24/report.json` may be
  force-added from the ignored atomic bundle; `ledger.csv.gz` remains ignored
  and local.
- Streaming aggregation must keep actual WSL filesystem use below 300 GiB.
- Rejected checkpoints and temporary responses must be deleted without
  deleting the immutable sentinel or terminal report.

## Mandatory sequence

1. Commit this boundary before the first `eth_getLogs` call.
2. Obtain independent adversarial review of the committed source contract. Any
   correction requires another commit and another review before source access.
3. Implement and test only the Stage A fixed-window parity verifier with mocked
   fixtures.
4. Commit the Stage A verifier and tests while the worktree is clean.
5. Execute Stage A once against the two frozen RPCs.
6. Commit the immutable Stage A pass or rejection metadata.
7. Only a Stage A pass may authorize one exact mechanism decision and one exact
   preregistration, both committed before any full-history incidence or rate
   value is opened.
8. Implement and test the Stage B builder with exact hash bindings to Stage A,
   the mechanism, and the preregistration.
9. Commit the Stage B builder and tests while the worktree is clean.
10. Execute Stage B once against all 16,055 chunks.
11. Commit the immutable Stage B pass or rejection metadata.
12. Only a Stage B pass may authorize source-support statistics and
    pre-outcome novelty.
13. Run cheap causal baselines before any RLLM training.
14. Open sealed evaluation only after every prior gate passes.

A stage pass is only permission to continue. It is not evidence of alpha.
