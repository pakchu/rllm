# Ethereum stablecoin issuance/redemption source audit — 2026-07-21

## Verdict

**Promote the 2020–2023 Ethereum event panel as an immutable research source.**

This verdict establishes source provenance and causal availability only. It is
not evidence that issuance, mint, redeem, burn, or confiscation predicts BTC,
and it opens no BTC price, funding, future return, PnL, absolute return, CAGR,
or strict MDD.

## Reproduced artifact

| Field | Value |
|---|---:|
| Source rows | 266,362 |
| UTC block-time interval | 2020-01-01 through 2023-12-31 |
| CSV gzip bytes | 38,163,040 |
| CSV SHA-256 | `70ba3799ba84dc671051623a8d167b1731f043cf84a686b9878a67fcd52e5901` |
| Canonical event hash | `7266d7ae13a9c439280d5a392a807f7e2c93c7f248fb7276a3a61c9567f9cc72` |
| Canonical log hash | `22cca8a5447ef693cf24e46e4a3f94e08b5fae1155d03540146f09fb9a18c3a1` |
| Manifest hash | `a0c7740db64f7779fade68d76985c629cabe81983bf594e8258cef16a5725a1b` |

Two independent archive transports returned the same complete canonical log
hash. A third, independent header transport supplied event and `N+64` block
headers; every event block hash matched its log, and the finalized head covered
the last required confirmation block. Transport URLs are configuration, not
artifact identity, and are absent from the manifest.

## Event support

| Event | Rows |
|---|---:|
| USDC `Burn` | 166,552 |
| USDC `Mint` | 99,033 |
| USDT `DestroyedBlackFunds` | 642 |
| USDT `Issue` | 132 |
| USDT `Redeem` | **3** |

| Year | Rows |
|---|---:|
| 2020 | 11,323 |
| 2021 | 68,663 |
| 2022 | 106,971 |
| 2023 | 79,405 |

No `Deprecate` row occurred in the interval. The builder would have failed
closed rather than silently following a successor contract.

## Integrity checks

- chain ID is exactly Ethereum mainnet `1`;
- exact UTC boundary blocks are hash-bound;
- contract bytecode is present and hash-bound at both source boundaries;
- every row has a unique `(block_hash, transaction_hash, log_index)` identity;
- malformed ABI, removed logs, duplicate identities, block-hash mismatch, and
  independent replay disagreement fail closed;
- each `available_at` is the canonical timestamp of block `N+64`, not event
  time or provider receipt time;
- deterministic gzip, output columns, row count, event counts, year counts,
  file hash, and manifest hash are regression tested; and
- the manifest records zero BTC market, funding, return, label, or PnL access.

## Material source limitation discovered

USDT `Issue` and especially `Redeem` are sparse compared with USDC
`Mint`/`Burn`. Only three USDT redemptions exist in the complete four-year
panel. This is not a missing-range symptom: both archive replays agree exactly.
It means these contract events cannot be assumed to enumerate all Tether
treasury inventory movement, cross-chain issuance, or customer redemption.

Consequences:

1. a balanced long/short mechanism that requires a rolling history of USDT
   `Redeem` events is structurally unsupported;
2. `DestroyedBlackFunds` may not be relabelled as ordinary redemption to repair
   that support failure;
3. USDC mint/burn frequency does not by itself establish exchange-directed
   buying power; and
4. any later mechanism must pass source-only incidence and novelty gates before
   BTC outcomes are opened.

The already frozen IRH-36 mechanism requires 32 strictly prior USDT redeem
rows for its SHORT tail. With only three total rows, that state is impossible.
IRH-36 therefore must be retired before outcomes rather than retuned.

## Artifacts

- source:
  `data/ethereum_stablecoin_issuance_redemption_2020_2023/ethereum_usdt_usdc_issuance_redemption_2020_2023.csv.gz`
- manifest:
  `results/ethereum_stablecoin_issuance_redemption_source_manifest_2026-07-21.json`
- builder:
  `training/build_ethereum_stablecoin_issuance_redemption.py`
- tests:
  `tests/test_build_ethereum_stablecoin_issuance_redemption.py`
  and `tests/test_ethereum_stablecoin_source_artifact.py`

Official source definitions remain bound in the manifest:

- <https://ethereum.org/developers/docs/apis/json-rpc/>
- <https://tether.to/en/supported-protocols/>
- <https://developers.circle.com/stablecoins/usdc-contract-addresses>
- <https://github.com/circlefin/stablecoin-evm>

