# WBTC custody bridge-flow source audit (2026-07-23)

## Verdict

`PASS_SOURCE`.

The frozen 2020-01-01 through 2023-12-31 Ethereum source contains 993
canonical WBTC custody mint/burn events. Two independent Ethereum transports
returned the same normalized log sequence, every event was reproduced from a
successful transaction receipt, and every semantic event had the required
adjacent equal-amount zero-address `Transfer` companion. No BTC market,
funding, label, future-return, PnL, CAGR, MDD, or mechanism-feature rows were
opened while selecting or building this source.

This is a valid immutable weak-factor source. It is **not** evidence that mint
means BTC-long or burn means BTC-short, and it is not promoted as a standalone
alpha.

## Frozen contract and interval

| Field | Frozen value |
|---|---|
| Chain | Ethereum mainnet, chain ID 1 |
| Contract | `0x2260fac5e5542a773aa44fbcfedf7c193bc2c599` |
| Decimals | 8 |
| Start boundary | block 9,193,266, `2020-01-01T00:00:11Z` |
| End boundary, exclusive | block 18,908,895, `2024-01-01T00:00:11Z` |
| Last source block | 18,908,894 |
| Confirmation delay | 64 canonical blocks |
| Boundary bytecode | 4,582 bytes |
| Boundary bytecode SHA-256 | `1377b93c57d7373bf87a742b4783deff88599e7a1bcca58f0a5d6a93e8a2973b` |

Official references:

- [WBTC contract addresses](https://docs.wbtc.network/resources/contract-addresses)
- [WBTC mint/burn mechanism](https://docs.wbtc.network/how-wbtc-works/mint-burn-mechanism)
- [WBTC proof of reserve and transparency](https://docs.wbtc.network/overview/proof-of-reserve-and-transparency-dashboard)
- [WBTC Open API reference](https://docs.wbtc.network/resources/api-reference)
- [Ethereum JSON-RPC](https://ethereum.org/developers/docs/apis/json-rpc/)
- [Ethereum `eth_getLogs`](https://ethereum.github.io/execution-apis/api/methods/eth_getLogs/)

## Integrity evidence

### Canonical logs

- primary replay rows: 993
- independent verification replay rows: 993
- normalized replay equality: exact
- canonical log hash, both replays:
  `b8bcb672126e2668ef09a706eb68c6fed0e8a1ac19b31cdbc7a68ba40ed1245e`
- unique canonical identities: 993
- duplicate identities: 0

### Receipt and transfer pairing

- successful unique receipts: 993
- semantic events reproduced at the same block, transaction, and log index:
  993
- adjacent zero-address `Transfer` companions: 993
- canonical pair hash:
  `3718a1d1b21ba572f3d58a0f76a318eb059d7e0808e67bd3b627aed9390f5b5c`

For mint, the companion transfer is `0x0 -> recipient`; for burn it is
`burner -> 0x0`. Actor topic and raw amount must exactly match the semantic
event, and the companion log index must equal the semantic index plus one.

### Causal availability

Each row's `available_at` is the timestamp of canonical block `N + 64`, not
the event block timestamp. Event-block hashes were cross-checked against
canonical headers and the Ethereum `finalized` tag covered every required
confirmation block. The exact current finalized-head number is intentionally
excluded from the manifest so repeated builds remain hash-reproducible.

## Source support

| Year | Mint | Burn | Total |
|---:|---:|---:|---:|
| 2020 | 285 | 38 | 323 |
| 2021 | 200 | 50 | 250 |
| 2022 | 101 | 176 | 277 |
| 2023 | 78 | 65 | 143 |
| **Total** | **664** | **329** | **993** |

Both event directions exist in every calendar year. The declining event count
and strong year-to-year imbalance mean the source should enter a later
mechanism as sparse, rolling weak evidence rather than as a raw directional
trigger.

## Report-only Open API reconciliation

The official current-state endpoint
`https://openapi.wbtc.network/public/v1/mint-burn` was queried at
`2026-07-22T18:23:15Z` only after the canonical source had passed. It was not
used to select rows, timestamps, actors, statuses, or amounts.

- API records across all networks/statuses: 1,355
- Ethereum records across all statuses: 1,317
- completed Ethereum records dated 2020-2023 with a transaction hash: 1,041
- exact transaction-hash overlap with canonical semantic events: 326
- among those 326 overlaps, action mismatches: 0
- among those 326 overlaps, 8-decimal raw-amount mismatches: 0
- canonical semantic transactions absent from the current API: 667
- API transactions without a matching canonical semantic Mint/Burn row: 715

The exact overlap corroborates decoding semantics. The large non-overlap also
proves that the mutable workflow/order API is not a complete one-to-one
historical contract-event registry. It therefore remains corroborative only;
the immutable Ethereum log/receipt record is canonical.

## Frozen artifacts

- source:
  `data/wbtc_custody_bridge_flow_2020_2023/wbtc_mint_burn_2020_2023.csv.gz`
- source rows: 993
- source bytes: 150,891
- source SHA-256:
  `bfcc6ebc2ded0cd8a57e5cda83a77daafe4de325adf606b23ba43ecf486b3b4e`
- source manifest:
  `results/wbtc_custody_bridge_flow_source_manifest_2026-07-23.json`
- manifest hash:
  `4e4344a7f2841803dc8da625ee1320f79e1821d54cb2366a5464728507b4bcab`
- source protocol manifest hash:
  `dbf263a3baae3c35833e4b2297880e7c2ec5ea23b6e2065f34ce14106a2a72a9`

## Boundary for the next stage

The next commit must freeze exactly one mechanism and its controls before any
post-2023 WBTC events or BTC outcomes are read. This source audit does not
authorize threshold search, mint-long/burn-short mapping, or outcome-driven
feature selection.
