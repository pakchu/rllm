# BRCR source protocol — 2026-07-22

## Status

**Frozen before full 2020–2023 RIPE incidence and before every BTC outcome.**

- source-axis decision:
  `docs/bgp-routing-churn-relay-source-axis-decision-2026-07-22.md`
- executable protocol:
  `training/preregister_bgp_routing_churn_relay.py`
- machine manifest:
  `results/bgp_routing_churn_relay_source_protocol_2026-07-22.json`
- manifest hash:
  `97dd7fff76fee4e9cefd76a9e8af7e6e2df04a9c2e69e5dc68f6827748ef44e7`
- manifest file SHA-256:
  `5f4204f6e630ce690bab3f961d2fce74849b8c48dd68b5ca56a2f9daac09aee1`
- implementation SHA-256:
  `7e7d9cad14fb0d33e17098db1e369d4bbe1a20f59fba41a2402e943cb534d6a7`
- outcomes opened: `false`
- historical source incidence opened: `false`
- mechanism parser opened: `false`

## Frozen source

BRCR streams only RIPE NCC RIS RRC00 update objects:

```text
https://data.ris.ripe.net/rrc00/YYYY.MM/updates.YYYYMMDD.HHmm.gz
```

The source envelope is exactly 00:00, 06:00, 12:00, and 18:00 UTC from
2020-01-01 through 2023-12-31: 5,844 expected five-minute objects. RRC00 is
the only collector. `bview` files, other collectors, RIPEstat, RIPE Atlas,
IODA, and mirrors are forbidden fallbacks.

Every normalized object becomes usable at archive label + 15 minutes. HTTP
`Last-Modified`, `ETag`, and `Content-Length` are validation evidence only;
they never become signal fields. `Last-Modified` must lie from label + 5
minutes through label + 30 minutes, inclusive. The received length must equal
the declared length, and the ETag must retain the strong quoted Apache form
observed in the bounded probe.

## Exact gzip and MRT contract

- gzip magic must be `1f8b` and decompression must pass CRC;
- decompressed size is capped at 1 GiB and compressed size at 128 MiB;
- every record uses the 12-byte RFC 6396 network-order common header `!IHHI`;
- record type is exactly BGP4MP (`16`);
- allowed subtypes are exactly `0`, `1`, `4`, `5`, `6`, and `7`;
- every timestamp lies in `[label, label + 5 minutes)`;
- timestamps are monotone within an object;
- payload lengths terminate exactly at the decompressed stream boundary; and
- every object contains at least one message record.

The source artifact retains only object identity, aggregate record counts,
exact type/subtype counts, embedded timestamp bounds, byte counts, SHA-256,
and transport evidence. Raw payloads, prefixes, AS paths, peers, communities,
and next hops remain forbidden. Two in-memory parse passes must be identical.

## Bounded probe disclosure

Four object bodies were opened at the same `January 1 00:00 UTC` coordinate,
one in each source year. One January 2023 directory listing was opened to
freeze the actual plural `updates.*.gz` filename. All four objects passed gzip
CRC and exact MRT-boundary checks, contained only type 16 subtypes 0/1/4/5,
were timestamp-monotone, and covered exactly 00:00:00 through 00:04:59 UTC.
No full-period incidence, mechanism threshold, market clock, or outcome was
opened.

The exact probe bytes, hashes, record counts, HTTP modification values, and
ETags are embedded in the machine manifest.

## Frozen source gates

Transport must satisfy:

- at least 99.5% expected-object coverage in every year;
- at least 98% expected-object coverage in every month;
- no more than four consecutive missing expected objects;
- SHA-256, length, ETag, metadata-age, gzip CRC, and MRT validation for every
  fetched object;
- zero duplicate compressed SHA-256 values across distinct labels; and
- exact equality between two in-memory parse passes.

Replay must refetch exactly the 00:00 UTC object on the first day of each of
the 48 source months. Every replay body hash and validation header must equal
its first fetch. All coverage denominators use the complete expected-label
calendar, never the fetched or successful subset. Missing/unparseable objects
remain missing; no imputation or forward fill is allowed.

Any failure is `REJECT_NO_REPAIR`. The collector, interval, anchor hours,
fields, and gates cannot change after full incidence opens.

## Deferred mechanism and RLLM boundary

A source pass authorizes only a separately committed mechanism protocol. That
protocol must freeze prior-only churn algebra, baseline, event direction,
execution delay, hold, leverage, support controls, and sparse-clock novelty
comparators before BTC data opens.

An LLM or RLLM may later consume a frozen aggregate churn state or make
train-only abstention and sizing decisions. It may not create, delete, retime,
relabel, impute, or repair source objects; decode forbidden BGP identities
without another pre-outcome protocol; or use eval outcomes to select a
direction or threshold.
