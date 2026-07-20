# UFCP confirmed-ledger source freeze — 2026-07-20

## Decision

The complete source-only UFCP prefix passed its frozen acquisition and
integrity contract.  No BTC market value, funding value, event incidence,
future return, PnL, CAGR, or drawdown was loaded.

## Frozen source

- block heights: `610691..823785` inclusive;
- rows: `213095`;
- header-time range: `1577836985..1704066372`;
- physical cutoff: every row is before `1704067200` (`2024-01-01 UTC`);
- hash links checked: `213094`;
- unique canonical block hashes: pass;
- `utxo_set_change == total_outputs - total_inputs`: pass on every row;
- basic fields matched the earlier frozen block source on every row.

Artifacts:

- source CSV:
  `data/bitcoin_utxo_fee_block_stats_2020_2023.csv.gz`;
- source CSV SHA-256:
  `8d5160b59bd104819adb34752f2bd5b01e9c07d7aa108188abcd9e6d323d102f`;
- source CSV bytes: `13991597`;
- source manifest:
  `results/bitcoin_utxo_fee_block_stats_source_manifest_2026-07-20.json`;
- source manifest file SHA-256:
  `ceb096b7bfeaf309729c4cab9f5cd4d2e0d526e261b1e460e9d03cd4f24e8084`;
- source manifest canonical hash:
  `98a84b0bd0338300f62eaa047b87498cc5a8d9505a03f6bd1912d1deb9564e8c`;
- source builder SHA-256:
  `099454feff009a5a4d44a96bd3790ff586d0365eba2e9b72e7b071d34e743633`;
- reference block-source SHA-256:
  `1f8d1c0153717f2c18d8fa6f09428c780a850b2aecc8fb42bc497e16e68e1833`.

The downloader was rerun from its completed SQLite checkpoint with the same
transport configuration.  The source CSV and manifest were byte-identical.
The completed acquisition took `41m43.86s` wall time and used at most
`554500 KiB` RSS.

## Outcome and deployment boundary

The source manifest records zero market rows, funding rows, outcome rows,
return/PnL fields, and post-2023 rows.  It persists no raw Mempool response,
pool tag, expected-fee field, or unrelated metadata.

The public Mempool endpoint is only a private historical research transport.
Its output has no separately documented commercial-data licence in the
reviewed official material.  Production UFCP requires an owned Bitcoin Core
node and field-by-field forward parity before orders are enabled.

Historical `firstSeen` was unavailable, so this source cannot justify an
immediate block-time entry.  The preregistered daily `D+2 00:00 UTC`
availability, six-successor requirement, and one complete five-minute latency
remain mandatory.  Live promotion additionally requires forward node receipt
times and 90 shadow days.

## Next boundary

The exact source output, source manifest, builder, and reference SHA values are
now pinned in the UFCP preregistration generator.  Only a source-manifest-only
preregistration artifact may be generated next.  Real source incidence remains
closed until that artifact is committed.
