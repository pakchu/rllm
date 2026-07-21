# Bitcoin coinbase-payout topology transport — rejected before full build

## Decision

Retire the frozen Mempool-summary transport before building the 2020–2023
coinbase-payout topology panel. A 20-block bounded audit reproduced one required
field hole, so the source contract's all-block completeness condition is false.
No fallback, candidate clock, BTC market data, funding, return, PnL, absolute
return, CAGR, or MDD was opened.

- audit artifact:
  `results/bitcoin_coinbase_payout_transport_rejection_2026-07-21.json`;
- artifact SHA-256:
  `c00bf8193a012b01a7843d27afc636cd9a205ddcecc6717367a700c7d8b8f99e`;
- manifest hash:
  `6c749c30996fbc8a3793b4abbba9d1e806665d6ca23c5b2d699e520dd67f5491`;
- auditor SHA-256:
  `a47c5a3eff3413fbbe52e4c6af7e9b7ac90dea8cdb0703dc65bf831b79396029`;
- source reference: 213,095 hash-linked best-chain blocks, SHA-256
  `8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f`.

## Reproduced failure

The audit requested the two overlapping 15-block pages at heights 800,019 and
800,014, covering exactly 800,000–800,019 without opening a block outside that
range. All 20 block identities, parent links, and timestamps matched the bound
best-chain reference.

| Item | Result |
|---|---:|
| unique blocks | 20 |
| complete Mempool summary topology rows | 19 |
| missing summary topology rows | **1** |
| missing height | **800,015** |

At height 800,015, `extras.coinbaseSignature` was `null` and
`extras.coinbaseAddresses` was empty. The block was not stale and its identity
matched the frozen chain. Independent transaction-level requests to both
Blockstream and Mempool returned the same non-empty first-output script and the
same one-address set. Therefore the consensus object exists; the frozen
`v1/blocks/:height` summary enrichment is incomplete.

This distinction is consistent with the original source contract: Bitcoin
Core's verbose `getblock` exposes transaction outputs, while the Mempool
summary field is an index-derived transport convenience. See the
[Bitcoin Core `getblock` reference](https://bitcoincore.org/en/doc/30.0.0/rpc/blockchain/getblock/),
[Bitcoin coinbase transaction reference](https://developer.bitcoin.org/reference/transactions.html#coinbase-input-the-input-of-the-first-transaction-in-a-block),
and [Mempool REST documentation](https://mempool.space/docs/api/rest).

## Boundary and no-repair rule

The committed audit made four source-only calls. Earlier interactive diagnosis
made at least nine source calls and printed one clear coinbase address; this is
disclosed rather than represented as a clean room. It did not inspect any
market or post-entry outcome.

```text
BTC market rows       = 0
funding rows          = 0
future-return rows    = 0
return/PnL fields     = 0
post-2023 source rows = 0
candidate clocks      = 0
```

Using transaction-level fallback only for missing rows would change the frozen
transport after seeing a failure and create heterogeneous availability. It is
not authorized. A future transaction-level panel would require a new source
freeze and an owned Bitcoin Core node or a separately justified complete
transport; it cannot repair this candidate.

The aborted full builder was deleted. Only the minimal reproducible audit is
kept, avoiding dead production code for a retired source path.
