# WCBF source-axis decision — finalized WBTC custody-bridge supply flow

## Decision

Proceed to a source-only builder for finalized Ethereum mainnet WBTC `Mint`
and `Burn` events over 2020–2023.

This decision promotes an immutable **weak-factor source**, not a standalone
trading alpha.  It opens no BTC market row, funding row, return, label, PnL,
CAGR, strict MDD, threshold, direction, hold, model, or LLM policy.  WBTC
issuance and redemption are merchant/custodian workflows with material
operational latency; later work may test their interaction with independently
defined weak signals but may not relabel every mint as immediate BTC demand or
every burn as immediate BTC supply.

## Economic object and interpretation boundary

The official WBTC mechanism states that:

- a mint follows BTC delivery to custody and increases circulating WBTC;
- a burn permanently removes WBTC before the custodian releases BTC; and
- users transact through approved merchants rather than directly with the
  custodian.

Primary references:

- [WBTC contract addresses](https://docs.wbtc.network/resources/contract-addresses)
- [WBTC mint/burn mechanism](https://docs.wbtc.network/how-wbtc-works/mint-burn-mechanism)
- [WBTC proof of reserve](https://docs.wbtc.network/overview/proof-of-reserve-and-transparency-dashboard)
- [WBTC public API](https://docs.wbtc.network/resources/api-reference)
- [Ethereum JSON-RPC](https://ethereum.org/developers/docs/apis/json-rpc/)
- [Ethereum execution API `eth_getLogs`](https://ethereum.github.io/execution-apis/api/methods/eth_getLogs/)

The permitted observable is therefore finalized **realized token supply
change**.  Pending, rejected, or cancelled requests are not inferred from token
logs.  Merchant identity, destination, reserve-wallet movement, exchange flow,
and user intent are outside the source contract.  The current WBTC REST API is
corroborative only: its status and merchant records are mutable current-state
metadata, not an immutable point-in-time history.

## Frozen Ethereum contract and ABI

| Field | Frozen value |
|---|---|
| Chain | Ethereum mainnet, chain ID `1` |
| Contract | `0x2260fac5e5542a773aa44fbcfedf7c193bc2c599` |
| Decimals | `8` |
| Mint event | `Mint(address,uint256)` |
| Mint topic | `0x0f6798a560793a54c3bcfe86a93cde1e73087d944c0ea20544137d4121396885` |
| Burn event | `Burn(address,uint256)` |
| Burn topic | `0xcc16f5dbb4873280815c1ee09dbd06736cffcc184412cf7a71a0fdb75d397ca5` |
| Transfer event | `Transfer(address,address,uint256)` |
| Transfer topic | `0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef` |

The official address page identifies the contract and eight decimals.  Its
verified ABI exposes direct `mint` and `burn` functions plus indexed
`Mint(address,uint256)`, `Burn(address,uint256)`, and ERC-20 `Transfer`
events.  The source builder must parse the semantic Mint/Burn events and then
verify the same transaction receipt contains the matching zero-address
Transfer with identical actor and amount:

```text
Mint(to, amount)   -> Transfer(0x0, to, amount)
Burn(from, amount) -> Transfer(from, 0x0, amount)
```

The zero-address Transfer is an integrity companion, not a second economic
event.  It may not be double counted.

## Fixed source envelope and causal clock

- event interval: `[2020-01-01T00:00:00Z, 2024-01-01T00:00:00Z)` by canonical
  Ethereum block timestamp;
- start boundary already established by the promoted Ethereum source:
  block `9,193,266`, timestamp `2020-01-01T00:00:11Z`;
- exclusive end boundary already established by the promoted Ethereum source:
  block `18,908,895`, timestamp `2024-01-01T00:00:11Z`;
- last admissible event block: `18,908,894`;
- confirmation convention: canonical block `N+64`;
- `available_at`: timestamp of canonical block `N+64`, never the event block
  timestamp, provider receipt time, WBTC dashboard time, or REST record date.

Every live decision must additionally wait for Ethereum's reported finalized
head to cover the event and one complete five-minute execution-latency bar.
The fixed 64-block delay is the historical causal convention; it is not a
claim that protocol finality always occurs at exactly 64 blocks.

## Bounded source-only evidence already opened

The bounded probe read no BTC outcome and used only Ethereum mainnet logs,
receipts, headers, and bytecode.

Two independent transports returned identical WBTC code at both source
boundaries:

| Boundary block | Bytecode bytes | SHA-256 |
|---:|---:|---|
| `9,193,266` | `4,582` | `1377b93c57d7373bf87a742b4783deff88599e7a1bcca58f0a5d6a93e8a2973b` |
| `18,908,894` | `4,582` | `1377b93c57d7373bf87a742b4783deff88599e7a1bcca58f0a5d6a93e8a2973b` |

The same transports returned byte-identical semantic logs in four disjoint
10,000-block probes.  The probes included one 400-WBTC burn at block
`11,505,404`, two 250-WBTC burns at blocks `17,020,684` and `17,035,332`, and
one 299.52-WBTC mint at block `17,045,939`.  Transaction receipts showed the
required same-amount zero-address Transfer companions.  The exact per-range
canonical hashes were:

| Block range | Rows | Canonical SHA-256 |
|---|---:|---|
| `11,500,000–11,509,999` | 1 | `ca21e69844aaee0726eadd52afdf77339d222e7da975ec7ba86ee8631436ec71` |
| `17,020,000–17,029,999` | 1 | `176457c19ec92bf7497dcd029cb2ccfddb56b9d1f907a4a3f3e9a65f487c4785` |
| `17,030,000–17,039,999` | 1 | `ebec75f72a0fa6ef01884eeb1d347d7743257c72a8132ed5c5da6e4be70449d3` |
| `17,040,000–17,049,999` | 1 | `97da208ecebd983fb2fb5511761ab395725e7c241a944cabcdb62a274531e740` |

A separate source-only read of the current official WBTC API exposed 1,317
Ethereum records across 2019–2026, including 395/264/285/157 records labelled
2020/2021/2022/2023 across all statuses.  This was used only to establish that
the source is not structurally empty.  Those mutable REST rows and dates are
forbidden from the canonical panel and from later point-in-time features.

## Required source builder

The next work unit must:

1. query only the frozen contract, topics, and block envelope;
2. bind chain ID, exact boundary blocks/hashes, and boundary bytecode hashes;
3. use bounded `eth_getLogs` ranges and fail on every RPC or range error;
4. replay the complete semantic log set through two independent hostnames and
   require identical canonical hashes;
5. fetch and validate transaction receipts for every semantic event, including
   successful transaction status, canonical block identity, and exactly one
   matching zero-address Transfer companion;
6. reject removed, malformed, zero-amount, duplicate, conflicting, or
   out-of-range logs;
7. bind event and `N+64` headers, require the event block hash to match, and
   set `available_at` only from the confirmation header;
8. require the reported finalized head to cover the last confirmation block;
9. write deterministic gzip plus a hash-bound source manifest; and
10. record zero access to BTC market, funding, labels, returns, PnL, portfolio
    statistics, and post-2023 contract-event rows.

Source promotion requires at least one valid mint and burn in every calendar
year, exact dual replay, receipt-pair parity for every row, unchanged boundary
code, and no duplicate canonical identity.  Failure retires WCBF source v1
without threshold, date, event, provider, or semantic repair.

## Later alpha boundary

Only a promoted source may authorize a separately committed mechanism.  That
mechanism must treat WCBF as one weak factor and predeclare its combinations,
multiplicity control, direction map, latency, hold, and train/selection/test/
eval sequence before opening BTC outcomes.  The source decision does not
authorize a standalone mint-long/burn-short rule, a reversed rule, a grid, or
an LLM-selected direction.
