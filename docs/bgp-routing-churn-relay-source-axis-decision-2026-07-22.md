# BRCR source-axis decision — RIPE RRC00 routing churn

## Decision

Advance **BRCR (BGP Routing Churn Relay)** to a source-only
preregistration. BRCR is a new exogenous source axis: raw Border Gateway
Protocol update records received by RIPE NCC's globally scoped RRC00 route
collector. It is not a price, volume, funding, order-flow, on-chain, news,
search, weather, calendar, or Bitcoin-protocol observable.

This decision is frozen before reading full 2020–2023 source incidence and
before opening any BTC clock, price, return, PnL, CAGR, MDD, prior-alpha clock,
or portfolio result.

## Why the mechanism is plausible

The intended causal chain is network-physical rather than a relabelled market
feature:

1. a surge in Internet routing announcements, withdrawals, or peer-state
   changes marks unusually high inter-domain routing churn;
2. routing instability can delay or partition continuously connected systems;
   and
3. Bitcoin mining, block propagation, exchange connectivity, and cross-venue
   arbitrage all depend on Internet paths, so a sufficiently broad routing
   disturbance may create a delayed liquidity or inventory response.

This is only a mechanism hypothesis. It does not claim that generic routing
churn is a Bitcoin-targeted attack or that the effect is economically material.
The 2017 IEEE Symposium on Security and Privacy paper *Hijacking Bitcoin*
demonstrated that BGP manipulation can isolate mining power and delay Bitcoin
block propagation. Bitcoin's developer documentation likewise describes a
peer-to-peer network and miner relay paths. These references motivate the
source axis but do not choose direction, threshold, hold, leverage, or regime.

Primary references:

- peer-reviewed Bitcoin/BGP study and DOI:
  https://nsg.ee.ethz.ch/publications/2017-06-26-hijacking-bitcoin-routing-attacks-on-cryptocurrencies-20-500-11850-192153/
  and https://doi.org/10.1109/SP.2017.29
- Bitcoin P2P developer guide:
  https://developer.bitcoin.org/devguide/p2p_network.html

## Official source and point-in-time boundary

Use only RIPE NCC Routing Information Service raw MRT update files:

```text
https://data.ris.ripe.net/rrc00/YYYY.MM/updates.YYYYMMDD.HHmm.gz
```

RIPE documents that update files contain routing changes for an interval, are
created every five minutes, and are retained in collector/month directories.
RIPE also identifies RRC00 as a multihop collector with global scope. MRT
common headers are governed by RFC 6396.

Official references:

- RIS raw MRT archive contract: https://ris.ripe.net/docs/mrt/
- RIS collector roles and RRC00 scope:
  https://ris.ripe.net/docs/route-collectors/
- MRT format: https://www.rfc-editor.org/rfc/rfc6396

The documented example uses singular `update` as the type label, but the
actual official directory names the five-minute files `updates.*.gz`. BRCR
freezes the observed official filename, not an alias.

Each authorized file is available to a live process only at
`archive label + 15 minutes`. This is deliberately later than the documented
five-minute creation cadence and the bounded probe's HTTP modification times.
HTTP `Last-Modified` and `ETag` are transport evidence only and may never enter
a signal. A file whose `Last-Modified` is before label + 5 minutes or after
label + 30 minutes is rejected rather than retimed.

RIPE has publicly documented a May 2026 incident in which historical **bview**
files were recopied and their modification timestamps changed. BRCR therefore
forbids bview files entirely, never treats HTTP modification time as an event
time, and fails closed if an authorized update file shows comparable metadata
drift. No claim is made that RIPE promises immutable historical bytes.

- RIPE incident notice:
  https://lists-ext-2.ripe.net/archives/list/mat-wg@ripe.net/2026/5/

## Fixed collector and sampling envelope

- collector: exactly `rrc00`;
- physical source interval: `[2020-01-01, 2024-01-01)` UTC;
- archive labels: exactly `00:00`, `06:00`, `12:00`, and `18:00` UTC each day;
- file interval: exactly five minutes beginning at the archive label;
- expected objects: 5,844;
- public availability: archive label + 15 minutes;
- source construction: stream one gzip object, hash it, decompress and parse
  twice in memory, persist only normalized source statistics, then release the
  raw bytes;
- deterministic replay audit: refetch exactly the `00:00` object on the first
  UTC day of each month and require byte-identical SHA-256 and headers;
- disk guard: abort before a download at 300 GiB used.

RRC00 is fixed because RIPE classifies it as globally scoped. No collector may
be added, removed, substituted, or averaged after full incidence is opened.
The four UTC anchors are fixed to bound the source transfer while avoiding a
single regional clock. BRCR makes no claim that they cover every Internet
event.

## Exact source fields

The frozen source artifact may contain only:

- archive label and conservative availability timestamp;
- official URL;
- compressed byte count and SHA-256;
- decompressed byte count;
- total MRT record count;
- minimum and maximum embedded MRT timestamps;
- counts keyed by the exact `(type, subtype)` pair; and
- transport-only `Last-Modified`, `ETag`, and declared content length for
  validation and replay evidence.

Raw BGP payloads, prefixes, AS paths, peer IPs, peer ASNs, communities, next
hops, and HTTP metadata are forbidden from a later signal unless a separate
pre-outcome parser protocol explicitly authorizes them. The first BRCR
mechanism is intentionally limited to aggregate message and peer-state churn.

Every MRT record must have the exact 12-byte network-order common header:
unsigned 32-bit timestamp, unsigned 16-bit type, unsigned 16-bit subtype, and
unsigned 32-bit payload length. Payload length must terminate exactly at the
gzip stream boundary. Records must be timestamp-monotone and lie in
`[label, label + 5 minutes)`. Type must be `BGP4MP` (`16`); accepted subtypes
are frozen to state-change/message forms `0`, `1`, `4`, `5`, `6`, and `7`.
Unknown types or subtypes are fatal, not silently skipped.

## Bounded source-only probe already opened

Exactly four official objects were read, one per source year and all at the
same fixed `January 1 00:00 UTC` coordinate. The January 2023 directory listing
was also read solely to resolve the documented singular `update` label against
the official plural `updates.*.gz` filenames. No other object body, source
incidence, BTC series, or economic outcome was opened.

| year | compressed bytes | decompressed bytes | records | embedded UTC seconds | HTTP modification delay |
|---:|---:|---:|---:|---|---:|
| 2020 | 3,953,511 | 19,967,733 | 138,891 | `00:00:00`–`00:04:59` | 5m15s |
| 2021 | 7,573,375 | 44,790,599 | 295,941 | `00:00:00`–`00:04:59` | 5m44s |
| 2022 | 3,085,355 | 18,301,516 | 114,035 | `00:00:00`–`00:04:59` | 5m08s |
| 2023 | 3,592,286 | 24,584,606 | 138,274 | `00:00:00`–`00:04:59` | 5m13s |

Compressed SHA-256 values, in year order:

- 2020: `84ea79cd40424754a77530bcaf98eff988f6582712ef5199d53cc1ec159b3162`
- 2021: `ed199898df7cb1df6388315990cb0dd69492b3b144046cf1e226c97efe88c929`
- 2022: `6815f3dd52a507ec202f774ee56ecc18ca02c2a0681606517e65d345b7d8d11a`
- 2023: `e7233b1424bfee50dfa5bd3990c03ebf53f9662e120c343d5ed26f3e51d3c3a3`

All four streams passed gzip CRC, ended exactly on an MRT record boundary,
were timestamp-monotone, and contained only type 16 subtypes 0, 1, 4, and 5.
The probe establishes transport and schema only; its record counts may not set
an event threshold.

## Novelty boundary

Repository-wide search found no prior RIPE, BGP, routing-churn, or Internet-
outage alpha implementation. BRCR is therefore source-distinct from existing
exchange microstructure, derivatives, on-chain, macro-FX, calendar, document,
narrative, weather, energy, and price-action families.

Clock-level novelty is not asserted here. If source support passes, BRCR must
freeze its aggregate-churn parser and compare the resulting sparse clock with
canonical prior-alpha clocks before any economic result can promote it.

## Stop and anti-repair rule

The next commit must preregister exact URL generation, gzip/MRT parsing,
timestamp, type/subtype, metadata-age, coverage, replay, and deterministic-
parse gates before downloading the 5,844-object envelope. Failure retires BRCR
without changing collector, anchor hours, interval, fields, or thresholds
after incidence.

Only a source-support pass may authorize a separately committed mechanism
protocol. Signal direction, churn threshold, hold, leverage, and economic gates
remain unopened. An LLM or RLLM may later consume a frozen churn state or make
train-only abstention and sizing decisions; it may not create, delete, retime,
relabel, impute, or repair source objects, and eval outcomes may not select a
direction or threshold.
