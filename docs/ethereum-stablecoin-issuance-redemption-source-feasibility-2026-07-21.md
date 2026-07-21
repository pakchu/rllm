# Ethereum stablecoin issuance/redemption source feasibility — 2026-07-21

## Decision

Proceed to a source-only builder for finalized Ethereum **USDT issuance and
redemption** plus **USDC mint and burn** logs over 2020–2023.

This is a source-feasibility pass, not alpha evidence.  It opens no complete
history, feature incidence, BTC price, funding, future return, PnL, absolute
return, CAGR, or strict MDD.  No trading direction, threshold, hold, model, or
LLM policy is selected here.

The purpose is to replace the revision-prone latest-snapshot supply history
that made the prior stablecoin-supply hypothesis non-promotable.  Canonical
contract logs are block-height records.  A later source builder must still
prove completeness and independent replay; an RPC provider's response is not
accepted as provenance by itself.

## Canonical contracts and events

| Asset | Ethereum mainnet contract | Included events | Topic 0 |
|---|---|---|---|
| USDT | `0xdAC17F958D2ee523a2206206994597C13D831ec7` | `Issue(uint256)` | `0xcb8241adb0c3fdb35b70c24ce35c5eb0c17af7431c99f827d44a445ca624176a` |
| USDT | same | `Redeem(uint256)` | `0x702d5967f45f6513a38ffc42d6ba9bf230bd40e8f53b16363c7eb4fd2deb9a44` |
| USDT | same | `DestroyedBlackFunds(address,uint256)` | `0x61e6e66b0d6339b2980aecc6ccc0039736791f0ccde9ed512e789a7fbdd698c6` |
| USDT | same | `Deprecate(address)` | `0xcc358699805e9a8b7f77b522628c7cb9abd07d9efb86b6fb616af1609036a99e` |
| USDC | `0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48` | `Mint(address,address,uint256)` | `0xab8530f87dc9b59234c4623bf917212bb2536d647574c8e7e5da92c2ede0c9f8` |
| USDC | same | `Burn(address,uint256)` | `0xcc16f5dbb4873280815c1ee09dbd06736cffcc184412cf7a71a0fdb75d397ca5` |

The contract addresses are frozen from the issuers' current official
integration pages:

- [Tether supported protocols](https://tether.to/en/supported-protocols/)
- [Circle USDC contract addresses](https://developers.circle.com/stablecoins/usdc-contract-addresses)

Circle's official `stablecoin-evm` repository documents the FiatToken proxy,
upgrade model, and on-demand mint/burn roles.  The source builder must filter
logs at the stable proxy address, not at a transient implementation address:

- [Circle stablecoin-evm](https://github.com/circlefin/stablecoin-evm)

The six topic hashes are the Ethereum Keccak-256 hashes of the exact event
signatures above.  Amounts are unsigned 32-byte event data and both contracts
use six decimal places.  `DestroyedBlackFunds` stores an address and amount in
two data words.  `Deprecate` stores the replacement address and is metadata,
not a signed supply delta.  Indexed/data addresses, block number/hash,
transaction hash, and log index are retained only for identity,
deduplication, source audit, and future contract-handoff safety.

USDT `Issue` is a contract-level supply increase.  `Redeem` and
`DestroyedBlackFunds` are contract-level supply decreases, but confiscation is
kept as a separately labelled event and may not be silently treated as an
ordinary customer redemption.  `Deprecate` has sign zero and forces a source
handoff review.  It is **not** assumed that an issue event is already exchange
buying power or that every redemption is BTC selling.  USDC mint/burn events
have the same interpretation boundary.  Any later economic mechanism must
treat issuer inventory management, cross-chain movement, confiscation, and
redemption workflow as explicit confounders.

## Causal availability and reorg policy

Ethereum's JSON-RPC specification exposes `eth_getLogs`, historical block
headers, and the `safe` / `finalized` block tags:

- [Ethereum JSON-RPC API](https://ethereum.org/developers/docs/apis/json-rpc/)

The source clock is deliberately chain-version agnostic across pre-Merge PoW
and post-Merge PoS:

1. identify the canonical event at block `N`;
2. fetch and bind the canonical hash and timestamp of `N`;
3. wait through block `N+64`;
4. set `available_at` to the timestamp of canonical block `N+64`; and
5. live code additionally requires the event block to remain an ancestor of a
   head at least 64 blocks later and not be newer than the reported finalized
   head.

No event may be used at its own block timestamp.  No provider receipt time is
backfilled into history.  A missing header, hash mismatch, removed log,
duplicate `(block_hash, transaction_hash, log_index)`, or provider disagreement
fails closed.  The later alpha may execute only after `available_at` and after
one complete five-minute latency bar.

The fixed 64-block rule is a conservative causal confirmation convention, not
a claim that all Ethereum finality always occurs in exactly 64 blocks.  Live
collection should also record the provider's finalized head for monitoring.

## Bounded source-only probe

The probe used the unauthenticated `https://eth.drpc.org` transport and read
only four bounded historical log ranges near Ethereum block `11,565,000`, plus
block headers.  The initial probe covered issue/redeem and mint/burn topics;
the full builder additionally queries confiscation and deprecation topics.  It
did not download a complete prefix or calculate an alpha clock.

| Contract | Blocks probed | RPC log calls | Events observed | Shape checks |
|---|---:|---:|---:|---|
| USDT | 30,000 | 3 | 1 issue / 0 redeem | topic count 1, data 32 bytes, `removed=false` |
| USDC | 10,000 | 1 | 46 mint / 20 burn | mint topics 3, burn topics 2, data 32 bytes, `removed=false` |

The first observed USDT issue at block `11,568,075` and its 64-confirmation
header at block `11,568,139` were both retrievable.  The transport also
returned current `latest`, `safe`, and `finalized` block headers.  A separate
public endpoint rejected historical log access without a personal token; the
builder therefore cannot assume that every public RPC supports archive logs.

The probe establishes only that the canonical fields and historical reach are
obtainable.  It does not establish complete-provider delivery, long-term free
service, commercial SLA, or alpha value.

## Source-builder requirements

The next work unit may build only a 2020–2023 immutable event panel and source
manifest.  It must:

1. accept an RPC URL from configuration; no credential or provider URL is
   embedded in the artifact identity;
2. locate exact UTC boundary blocks by canonical block timestamps;
3. use bounded `eth_getLogs` ranges and fail on any gap or RPC error;
4. preserve block hash, block number, transaction hash, log index, event type,
   raw integer amount when applicable, indexed/data addresses, event
   timestamp, confirmation block/hash, and `available_at`;
5. keep confiscation separate from redemption and fail closed if a deprecation
   event appears without an explicitly reviewed successor contract;
6. verify contract bytecode is non-empty at the first and last source blocks;
7. sort and deduplicate by canonical log identity without keeping the last of
   conflicting rows;
8. replay the complete range through a second independent RPC transport and
   require an identical canonical event hash before source promotion;
9. write deterministic gzip and a hash-bound manifest; and
10. read zero BTC market, funding, return, label, portfolio, or post-2023 rows.

The complete event counts remain sealed until the builder, tests, and source
contract are committed.  If independent replay cannot be obtained, the source
is rejected before a mechanism or outcome evaluator is opened.

## Relationship to prior stablecoin research

The prior Coin Metrics supply-breadth candidate failed frozen OOS and was also
non-promotable because reviewed `SplyCur` history was a latest snapshot rather
than a point-in-time value-vintage archive.  This source work does not repair
that rejected policy, reuse its threshold, or reopen its consumed outcomes.

Direct finalized logs solve the historical-value vintage problem, but they do
not make the old breadth/price-divergence rule valid.  A later candidate must
use a separately preregistered event-level mechanism and pass novelty against
the rejected supply-breadth clock before any BTC outcome is opened.
