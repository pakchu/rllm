# BATE-288 source-host fallback — 2026-07-20

## Decision

The frozen BATE-288 research backfill will use the Esplora-compatible endpoint
`https://mempool.space/api/blocks/:start_height`. It retains the already
committed source loader, schema, height interval, source-only boundary, and
support preregistration. It changes only the hosted research transport.

Production remains self-hosted Bitcoin Core with actual local first-seen
timestamps. The hosted endpoint is not a live-trading dependency.

## Why the Blockstream public-host run stopped

The first run against `https://blockstream.info/api` atomically committed 6,610
contiguous rows at heights `817176..823785`—3.10% of the frozen prefix—then
terminated with HTTP 429 after exhausting the committed retry policy. It wrote
no final CSV or source manifest and opened no market, funding, return, or PnL
field.

Official Esplora documentation exposes only a single-block endpoint and a
ten-block descending page; no official bulk block-summary archive was found:

- [Esplora API](https://github.com/Blockstream/esplora/blob/master/API.md)
- [Esplora MIT license](https://github.com/Blockstream/esplora/blob/master/LICENSE)

Blockstream documents an authenticated managed Explorer API with higher-limit
plans, but obtaining and operating credentials is a separate external-service
decision and its public help pages do not provide a bulk block-summary export:

- [Create and manage Explorer API keys](https://help.blockstream.com/blockstream-explorer-api/set-up-explorer-api/create-and-manage-your-api-keys.md)
- [Make an authenticated Explorer API request](https://help.blockstream.com/blockstream-explorer-api/use-explorer-api/make-a-rest-api-request-with-your-api-keys.md)

## Fallback source audit

Mempool's official project describes `mempool.space` as an explorer and API
service and recommends self-hosting for sovereignty. Its REST documentation
explicitly states that hosted requests are rate-limited:

- [Mempool project and self-hosting](https://github.com/mempool/mempool)
- [Mempool REST API](https://mempool.space/docs/api/rest)
- [Mempool AGPL software license](https://github.com/mempool/mempool/blob/master/COPYING.md)

A bounded source-only probe of `GET /api/blocks/823785` returned ten rows with
the exact frozen Esplora keys:

`bits,difficulty,height,id,mediantime,merkle_root,nonce,previousblockhash,size,timestamp,tx_count,version,weight`.

Three overlap anchors matched the independently downloaded Blockstream partial
checkpoint exactly:

| height | frozen main-chain block hash |
|---:|---|
| 823785 | `00000000000000000000d0cd9e5661fca08ed8916c8bb4f8ac2a3a34c8d3fa4b` |
| 820000 | `00000000000000000000ba232574c32b4f0cd023e133c05125310625626d6571` |
| 817176 | `000000000000000000028e5ab706bdaa53fcd907cde34d45d2446ce6059069ae` |

No complete prefix, candidate incidence, market value, outcome, or performance
statistic was opened by these probes.

## Frozen fallback rules

1. Delete the incomplete Blockstream checkpoint rather than mixing providers.
2. Start a new checkpoint whose contract records exactly
   `https://mempool.space/api`.
3. Use the committed ten-block loader unchanged and respect hosted rate limits;
   retries may wait, but pages may not be skipped or imputed.
4. The completed source must match all three overlap anchors, the exact
   `610691..823785` range, every hash-chain link, and the pre-2024 timestamp
   boundary.
5. Raw responses remain ephemeral and are not redistributed.
6. A hosted-service SLA or data-use grant is not inferred from the software
   license. This is a private research backfill only. Live production must use
   a self-hosted Bitcoin Core source.

Bitcoin Core remains the clean long-run source because `getblock` supplies
`height`, `size`, `weight`, `time`, `mediantime`, `nTx`, and chain linkage under
an MIT-licensed local runtime:

- [Bitcoin Core 30.0 `getblock`](https://bitcoincore.org/en/doc/30.0.0/rpc/blockchain/getblock/)
- [Bitcoin Core 30.0 `getblockhash`](https://bitcoincore.org/en/doc/30.0.0/rpc/blockchain/getblockhash/)
- [Bitcoin Core MIT license](https://github.com/bitcoin/bitcoin/blob/master/COPYING)

The current machine's under-300-GB usage constraint makes a conventional
600–740-GB historical node sync inappropriate for this bounded research
backfill. That operational constraint does not weaken the future production
requirement.
