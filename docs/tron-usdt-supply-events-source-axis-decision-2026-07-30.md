# TRON USDt supply-events source-axis decision — 2026-07-30

## Decision

Proceed to a source-only builder for finalized TRON mainnet USDt `Issue`,
`Redeem`, `DestroyedBlackFunds`, and `Deprecate` logs from
`2023-01-01T00:00:00Z` through the last event that can receive the frozen
64-block confirmation before `2026-06-01T00:00:00Z`.

This decision freezes a canonical source panel, not a trading policy. It
opens no pre-cutoff event row, source incidence, threshold, rank, direction,
hold, BTC market row, funding row, return, PnL, portfolio weight, CAGR, strict
MDD, model, or LLM policy.

## Why this source is new

Repository-wide search before this decision found no alpha source, feature,
clock, or source builder using TRON, TRC-20, TronGrid, TRONSCAN, or the TRON
USDt contract.

The source is related semantically to the promoted Ethereum USDT issuance and
redemption panel, so a later policy cannot call the broad stablecoin family
pristine. TRON nevertheless supplies a distinct chain, confirmation clock,
contract history, event incidence, and primary-market venue. Every later
candidate must compare against the complete frozen Ethereum/stablecoin family
inventory before economics.

The US spot-BTC ETF axis considered in parallel was rejected because issuer
surfaces did not provide an immutable daily historical replay. See
`docs/us-btc-etf-primary-market-source-rejection-2026-07-30.md`.

## Official semantic authority

- [Tether supported protocols](https://tether.to/en/supported-protocols/)
  identifies TRON USDt at
  `TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t`.
- [Tether's operating description](https://tether.to/en/how-it-works/)
  describes issuance and redemption through Tether's verified customer
  process. It does not establish that every issue is immediate BTC buying or
  every redemption is immediate BTC selling.
- [TRON's TRC-20 interface](https://developers.tron.network/docs/trc20-protocol-interface)
  defines the `Transfer(address,address,uint256)` event and zero-address
  mint/burn convention.
- [TRON `eth_getLogs`](https://developers.tron.network/reference/eth_getlogs)
  defines address/topic-filtered raw-log retrieval and a default 5,000-block
  range.
- [TRON JSON-RPC](https://developers.tron.network/reference/json-rpc-api-overview)
  defines hexadecimal addresses, quantities, finalized block tags, batch
  limits, and log-filter limits.
- [TRON confirmation semantics](https://developers.tron.network/docs/tron-protocol-transaction)
  and
  [consensus](https://developers.tron.network/docs/concensus)
  distinguish confirmed/solidified state from an unconfirmed head.

TRON mainnet is content authority. Hosted RPC providers are independent
transports, not authorities.

## Frozen chain and event vocabulary

| Field | Frozen value |
|---|---|
| TRON EVM chain ID | `0x2b6653dc` |
| Base58 contract | `TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t` |
| 20-byte log address | `0xa614f803b6fd780986a42c78ec9c7f77e6ded13c` |
| TRON zero address | `0x0000000000000000000000000000000000000000` |
| `Issue(uint256)` | `0xcb8241adb0c3fdb35b70c24ce35c5eb0c17af7431c99f827d44a445ca624176a` |
| `Redeem(uint256)` | `0x702d5967f45f6513a38ffc42d6ba9bf230bd40e8f53b16363c7eb4fd2deb9a44` |
| `DestroyedBlackFunds(address,uint256)` | `0x61e6e66b0d6339b2980aecc6ccc0039736791f0ccde9ed512e789a7fbdd698c6` |
| `Deprecate(address)` | `0xcc358699805e9a8b7f77b522628c7cb9abd07d9efb86b6fb616af1609036a99e` |
| `Transfer(address,address,uint256)` | `0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef` |
| decimals | `6` |

`Issue` has supply sign `+1`. `Redeem` has supply sign `-1`.
`DestroyedBlackFunds` also reduces contract supply but is retained under its
own label and may not be relabelled as customer redemption. `Deprecate` has
sign zero and terminates source v1 for explicit contract-handoff review.

Raw ABI shape is frozen before replay:

| Event | exact topics | exact data | decoded constraints |
|---|---:|---:|---|
| `Issue` | 1 | 32 bytes | one unsigned `amount_raw`, `0 < amount_raw < 2^256` |
| `Redeem` | 1 | 32 bytes | one unsigned `amount_raw`, `0 < amount_raw < 2^256` |
| `DestroyedBlackFunds` | 1 | 64 bytes | first word is a left-zero-padded 20-byte actor; second word is `amount_raw > 0` |
| `Deprecate` | 1 | 32 bytes | one left-zero-padded 20-byte replacement address |
| `Transfer` | 3 | 32 bytes | indexed `from` and `to` are exact left-zero-padded 20-byte addresses; data is `amount_raw > 0` |

No extra topic, short or long data word, nonzero high address padding, zero
amount, removed log, or noncanonical encoding is admissible.

Every retained transaction receipt must have successful status. Pairing is
bidirectional and one-to-one within the exact receipt:

- each `Issue` has exactly one same-block, same-transaction, same-amount
  `Transfer(zero, recipient, amount)`, and every zero-from transfer has exactly
  one such `Issue`;
- each `Redeem` has exactly one same-block, same-transaction, same-amount
  `Transfer(sender, zero, amount)`, and every zero-to transfer has exactly one
  such `Redeem`;
- the normalized `actor_address` is the mint recipient for `Issue`, the burn
  sender for `Redeem`, and the first data-word address for
  `DestroyedBlackFunds`.

Actor and amount equality are exact. An orphan semantic event, orphan
zero-address transfer, one-to-many or many-to-one pairing, or unsuccessful
receipt is terminal. The companion transfer is integrity evidence, not a
second economic event.

## Frozen UTC and block envelope

Before any event/log/receipt replay or policy construction, two independent
transports performed metadata-only `eth_getBlockByNumber` probes and reproduced
every boundary below, including the immediately preceding header and parent
relation:

| UTC boundary | first block at or after | block hash |
|---|---:|---|
| `2023-01-01T00:00:00Z` | `47,313,358` | `0x0000000002d1f1ce5e430281e5308004cf19dd6e31afd4402b670fc05da5b340` |
| `2023-06-01T00:00:00Z` | `51,652,374` | `0x0000000003142716b7305d5d621414bc745837a849273ce4eab4b200c598af9d` |
| `2024-01-01T00:00:00Z` | `57,811,194` | `0x00000000037220fa937d59050fab5c3740ef10f5f7715b0f45035353878cd98f` |
| `2025-01-01T00:00:00Z` | `68,346,198` | `0x000000000412e156401b47b5e85900fecd1744a7dd70e0ec6d7c9db7b5b7b8fd` |
| `2026-01-01T00:00:00Z` | `78,854,231` | `0x0000000004b338578474ba7a2a5fd3f2e19d303cb79f30d0b8e05ee361607b33` |
| `2026-06-01T00:00:00Z` | `83,201,056` | `0x0000000004f58c20deab323895309dd25eecc6bbbe4cd6c940713da2d78ca67a` |

The source begins at block `47,313,358`. The exclusive UTC end boundary is
block `83,201,056`. With the frozen confirmation convention `N+64`, the last
admissible event block is `83,200,991`; its confirmation block is
`83,201,055`, still before the end boundary.

The raw-log envelope is exactly
`[47,313,358, 83,200,991]`, or `35,887,634` inclusive blocks. It is split
into exactly `7,178` contiguous inclusive chunks: `7,177` chunks of exactly
`5,000` blocks followed by the frozen final chunk
`[83,198,358, 83,200,991]` of exactly `2,634` blocks.
Blocks `83,200,992` through `83,201,055` are confirmation-only and are never
queried as admissible event blocks; block `83,201,056` is the exclusive UTC
boundary header. No gap, overlap, dynamic split, provider-specific range, or
shortening of the exact final chunk may be accepted.

## Dual raw-log replay

Production transport roles are frozen before source access, while request
targets remain credential-bearing runtime configuration:

1. `TRON_PRIMARY_RPC_URL`: HTTPS, hostname `api.trongrid.io`, port `443`,
   maximum JSON-RPC batch `100`;
2. `TRON_VERIFY_RPC_URL`: HTTPS, hostname
   `tron-mainnet.core.chainstack.com`, port `443`, maximum JSON-RPC batch
   `30`.

The builder validates scheme, hostname, port, a nonempty provider path, and
the distinct transport roles. URL userinfo, query, and fragment are forbidden.
The provider path may be credential-bearing and, with the full URL, may exist
only in process memory long enough to issue a request. Neither may appear in
stdout, stderr, exception text, tests, source rows, claims, manifests, hashes,
or commits. Serialized transport identity is exactly
`(role, scheme, hostname, port)`.

For every block chunk, both transports must return all three filters:

1. topic 0 in
   `{Issue,Redeem,DestroyedBlackFunds,Deprecate}`;
2. `Transfer` with indexed `from == zero`; and
3. `Transfer` with indexed `to == zero`.

Canonical raw-log fields are address, block number/hash, transaction
hash/index, log index, topics, data, and `removed`. Quantity and data encodings
must be canonical. The two transports must agree exactly per chunk, per
filter, and globally after canonical sorting. A transport error, JSON-RPC
error, omitted response ID, duplicate ID, malformed result, disagreement, or
shortened range is terminal.

Every semantic event's transaction receipt, event block header, and
confirmation header are fetched through both transports and must agree
exactly. The receipt status must equal successful `0x1`; it must contain the
exact semantic log and required uniquely paired companion transfer. The common
finalized head must cover every retained confirmation block.

There is one request attempt, no retry, no response-dependent backoff, no
provider substitution, no checkpoint, no resume, and no partial publication.
The production inter-batch throttle is exactly `0.25` elapsed seconds for each
transport after its first batch; it is not a CLI parameter and is bound in the
replay claim and source manifest.

## Causal availability

For an event in block `N`:

```text
confirmation_block = N + 64
available_at       = canonical timestamp of confirmation_block
```

The 64-produced-block delay is a conservative historical convention, not a
claim that TRON finality always occurs at exactly 64 blocks. Live operation
must additionally observe solidified/finalized state. Event block time,
provider receipt time, explorer display time, and an unconfirmed stream are
forbidden as the decision clock.

## Excluded feasibility and header-only boundary evidence

All event/log/indexer feasibility probes used only blocks and events after
`2026-06-01T00:00:00Z`. The only pre-cutoff RPCs before this decision were the
twelve metadata-only block-header calls described above. They opened no
contract log, transaction receipt, event type, amount, actor, source
incidence, candidate incidence, market row, or outcome.

- TronGrid, the frozen Chainstack transport, and a third QuickNode demo
  transport returned one identical normalized `Issue` log in a 5,000-block
  range. Canonical SHA-256:
  `c24557cda5ee3f7ba7f132ac4b2fe1116fdab0e3c958f81ec23bf30ee60de0d2`.
- TronGrid and Chainstack returned identical results for thirty contiguous
  5,000-block requests in one batch schedule over all four semantic topics:
  12 logs, canonical SHA-256
  `a4abfeb7e169bec498412c7dafdb2383165c5374c3e09503c19b21205ce1aa17`.
- The TronGrid `Issue` transaction-ID set exactly matched the independently
  indexed TRONSCAN zero-address mint-transfer set over the excluded probe.
  The corresponding set SHA-256 was
  `c601aa9495f0de7daa6bbd800ca80343a8a36392609e926134151eb746863414`.
- Both frozen transports reproduced all twelve boundary headers with
  canonical header-set SHA-256
  `d2513baf86cab444b034ef19079e18515ee8da3d756f4dc041fb1e889707927b`.
  The hash input is compact sorted-key JSON of the twelve normalized objects
  `{number,hash,parentHash,timestamp}` in boundary order, immediately
  preceding header then boundary header.

These probes establish shape, reach, batching, boundary mapping, and
independent agreement only. The committed replay claim is required before the
first pre-cutoff `eth_getLogs` or `eth_getTransactionReceipt` request and before
any event/source incidence is opened. The official builder must refetch the
entire frozen range and may not copy a probe response into the source panel.

## Required source artifact

The exact write-once paths are:

```text
claim
  results/tron_usdt_supply_events_source_replay_claim_2026-07-30.json
source CSV
  data/tron_usdt_supply_events_2023_2026/tron_usdt_supply_events_2023_2026.csv.gz
source manifest
  results/tron_usdt_supply_events_source_manifest_2026-07-30.json
```

The builder must atomically publish a deterministic gzip CSV containing only
canonical source fields:

```text
event_type
supply_direction
actor_address
amount_raw
block_number
block_hash
transaction_hash
transaction_index
log_index
paired_transfer_log_index
event_timestamp_utc
confirmation_block
confirmation_block_hash
available_at_utc
```

Canonical compact JSON means UTF-8 encoded, lexicographically sorted object
keys, separators `(',', ':')`, ASCII escaping, finite values only, and no
trailing whitespace. The claim is canonical compact JSON plus one LF;
`claim_hash` is SHA-256 of the same object with `claim_hash` excluded. It binds:

- the protocol parent commit and the Git blob plus SHA-256 of every decision,
  preregistration, builder, source-support, novelty, economics, imported
  accounting helper, and synthetic test file;
- sanitized transport identities, exact batch maxima, fixed throttle, methods,
  block envelope, and canonical output paths;
- one-shot/no-retry status and zero source/event access at claim creation.

The claim-only commit is made after the protocol parent commit. Replay requires
the current pushed commit to contain a clean canonical claim, have the claimed
protocol commit as its first parent, and retain every claimed protocol
blob/hash unchanged.

The source CSV is RFC-4180-style UTF-8 with the displayed header order and LF
records, then gzip level 9 with empty filename and `mtime=0`. The source
manifest is sorted-key, two-space-indented ASCII JSON plus one LF.
`manifest_hash` is SHA-256 of canonical compact JSON with `manifest_hash`
excluded; the manifest does not attempt to contain its own file-byte hash.
It must explicitly bind `protocol_parent_commit`, `replay_claim_commit`,
`replay_claim_sha256`, `source_csv_sha256`, sanitized transport identities
only, ranges, request schedule, exact replay hashes, receipt/header hashes,
event and year counts, and explicit zero outcome access. It must never
serialize a provider URL, path, query, userinfo, or credential.

Source promotion requires exact dual replay, complete transfer pairing, no
`Deprecate`, successful receipts, no orphan semantic or zero-address transfer,
no duplicate canonical identity, monotone causal availability, complete fixed
chunks, and atomic output. Failure retires this source build without
transport, range, topic, confirmation, or pairing repair.

## Later-policy boundary

A trading candidate, if separately frozen before full source replay, may use
only the normalized fields above. Direction, batching, latency, hold,
non-overlap, support floors, novelty, Gross9 reconstruction, same-gross
portfolio construction, and economic gates belong to a separate usage and
construction contract. None is authorized by this source decision.
