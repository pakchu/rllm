# BRCR source rejection — 2026-07-23

## Terminal decision

**BRCR is rejected with `REJECT_NO_REPAIR`.** No BTC market clock, return,
PnL, prior-alpha clock, churn threshold, mechanism parser, or outcome was
opened.

The terminal machine result is:

- `results/bgp_routing_churn_relay_source_support_2026-07-22.json`
- file SHA-256:
  `940323c2b72515c62fc439f50fa2c96eba980007ffa15af14fbbb815c0db2ab6`
- manifest hash:
  `fe7672dbadc589adb6625691b373daf4f266d0079c346f4acf8c7dde61254d8a`
- frozen protocol hash:
  `97dd7fff76fee4e9cefd76a9e8af7e6e2df04a9c2e69e5dc68f6827748ef44e7`
- committed builder SHA-256:
  `243689bc9dcd3f430202c925fa02a223643c87fe45972846899584303e003b3d`

## Failure point

The first 142 expected labels were checkpointed. The next label,
`2020-02-05T12:00:00Z`, failed the frozen publication-metadata rule:

- URL:
  `https://data.ris.ripe.net/rrc00/2020.02/updates.20200205.1200.gz`
- compressed bytes: `5,432,218`
- compressed SHA-256:
  `a03cf19b9e03adb184a1ba536ef0eac08286b82b2431ba520656d921348da6ab`
- HTTP `Last-Modified`: `Wed, 05 Feb 2020 13:16:48 GMT`
- archive-label-to-modification delay: `76 minutes 48 seconds`
- frozen accepted delay: inclusive range `5` through `30` minutes.

The builder fetched the object again for fatal confirmation with the same body
and validation headers. A subsequent rejection audit fetched the same SHA-256,
byte count, `Last-Modified`, and ETag (`"5e3ac040-52e39a"`). The failure is
therefore reproducible rather than a partial transfer.

## Payload integrity

The payload itself passes gzip CRC and exact MRT framing:

- decompressed bytes: `38,678,369`;
- records: `245,151`;
- message records: `244,876`;
- state-change records: `275`;
- embedded timestamps: exactly `12:00:00` through `12:04:59` UTC;
- type/subtype counts:
  - `16:0` = `144`;
  - `16:1` = `1,205`;
  - `16:4` = `243,671`;
  - `16:5` = `131`.

Thus the rejection is not caused by malformed routing records. It is caused by
the point-in-time contract: the currently archived object carries evidence of
publication or republication more than one hour after its event window.

## Why the run cannot resume

Changing availability from label + 15 minutes to HTTP `Last-Modified`, widening
the accepted delay, or ignoring the metadata now would be a post-incidence
repair. It would also admit lookahead in the already frozen live/backtest
alignment because this object could not be shown available at label + 15
minutes.

The BRCR v1 protocol explicitly made metadata delay beyond 30 minutes fatal
and prohibited changing collector, hours, interval, fields, or gates after
full source incidence opened. The partial SQLite checkpoint is therefore not
an admissible basis for another BRCR run. BRCR is retired before mechanism
construction and before all economic evaluation.
