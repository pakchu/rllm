# TCRR source-axis decision — signed Tor relay consensus reconfiguration

## Decision

Advance **TCRR (Tor Consensus Relay Reconfiguration)** to a source-only
preregistration. TCRR is an exogenous network-topology axis built from the
hourly, multiply signed Tor network-status consensus. It is not a BTC price,
volume, funding, open-interest, order-flow, on-chain, news, search, calendar,
or prior-alpha observable.

This decision is frozen before reading full 2020–2023 relay incidence and
before opening any BTC clock, bar, return, PnL, CAGR, MDD, prior-alpha clock,
or portfolio result.

## Why the mechanism is plausible

The intended chain is operational rather than semantic:

1. Tor directory authorities publish an hourly consensus describing the live
   relay set, relay flags, and relay-selection information;
2. abrupt changes in reachable relays, entry guards, exits, or consensus
   bandwidth can impair or re-route Tor circuits;
3. Bitcoin Core supports Tor/onion peer connectivity, so sufficiently broad
   Tor reconfiguration can alter connectivity for the privacy-preserving part
   of the Bitcoin node network; and
4. that state may produce a delayed, weak liquidity or risk-transfer response.

This is only a source and mechanism hypothesis. It does not select a direction,
threshold, hold, leverage, or economic rule.

Official references:

- Tor CollecTor source overview:
  https://metrics.torproject.org/collector.html
- Tor consensus format:
  https://spec.torproject.org/dir-spec/consensus-formats.html
- Tor voting timeline:
  https://spec.torproject.org/dir-spec/outline.html
- publication of a majority-signed consensus:
  https://spec.torproject.org/dir-spec/publishing-consensus.html
- signed-document encoding:
  https://spec.torproject.org/dir-spec/netdoc.html
- authority key certificates:
  https://spec.torproject.org/dir-spec/creating-key-certificates.html
- Bitcoin Core Tor support:
  https://bitcoincore.org/en/releases/0.12.0/
- Bitcoin Core Tor-v3 transition:
  https://bitcoincore.org/en/releases/22.0/
- current Tor-specific proxy configuration:
  https://bitcoincore.org/en/releases/30.0/

## Official source and archive boundary

Use the Tor Project CollecTor monthly consensus archives:

```text
https://collector.torproject.org/archive/relay-descriptors/consensuses/
  consensuses-YYYY-MM.tar.xz
```

The machine-readable archive inventory is:

```text
https://collector.torproject.org/index/index.json
```

CollecTor documents network-status consensuses as
`@type network-status-consensus-3 1.0`, published hourly and retained in
compressed archive tarballs. The current official inventory contains every
monthly archive from `2020-01` through `2023-12`; observed compressed sizes are
approximately 17.6–24.6 MiB per month.

No Tor Project statement was found that monthly tarballs are immutable. TCRR
therefore treats HTTP `Last-Modified` and archive-member mtime as advisory only.
The preregistration must freeze the 48 exact archive paths, byte counts, and
SHA-256 values currently reported by `index.json`. Every run must hash the
received compressed stream and fail closed on any mismatch. A future changed
archive is a new source version, never an in-place repair.

## Point-in-time rule

Each retained document must contain exactly one each of:

- `valid-after`;
- `fresh-until`;
- `valid-until`; and
- `voting-delay`.

Tor specifies that the voting period ends at `valid-after` and that a consensus
signed by a majority of authorities is then made available. TCRR assigns the
conservative public-availability time:

```text
valid-after + 15 minutes
```

The document must still be fresh at that time. Therefore the required relation
is:

```text
valid-after < valid-after + 15m < fresh-until < valid-until
```

The monthly archive publication date is deliberately not used as historical
availability. It is a later packaging timestamp, whereas the signed consensus
contains the network validity interval agreed before `valid-after`.

No stale-consensus grace period is allowed. A later market stage may only use a
source state after the frozen availability time and must execute on a strictly
later market bar.

## Signed-integrity boundary

The Tor specification defines a consensus signature over bytes beginning with
`network-status-version` and ending at the space immediately after the first
`directory-signature` keyword. Plain `ns` consensuses use SHA-1 and the Tor
netdoc PKCS#1 v1.5 raw-digest signature form. Only one signature per authority
counts.

TCRR source support must require:

- the `network-status-consensus-3 1.0` CollecTor annotation;
- every signing authority identity is a `dir-source` authority;
- distinct authority identity and signing-key digests;
- at least a strict majority of distinct authority signatures;
- deterministic archive and member identities; and
- a frozen cryptographic audit sample whose authority identities are anchored
  to a checksum-verified official Tor source release.

Full historical certificate collection is available indirectly in the much
larger CollecTor vote archives because votes embed authority certificates. It
is not required for every source row in v1: the 48 consensus tarballs are
already frozen by official SHA-256, and the bounded audit establishes that the
archived signature representation is independently verifiable. Expanding
certificate verification beyond the frozen audit sample is permitted only as
a stricter source-only validator; it may not change incidence or outcomes.

## Fixed source envelope

- source interval: `[2020-01-01, 2024-01-01)` UTC;
- target consensus anchors: exactly `00`, `06`, `12`, and `18` UTC daily;
- expected target documents: `5,844`;
- expected monthly archives: `48`;
- transport: HTTPS, XZ-compressed POSIX tar;
- member selection: exact archive month, calendar day, and
  `YYYY-MM-DD-HH-00-00-consensus` basename;
- processing: one compressed archive stream, hash once, parse selected members
  in memory, retain compact source summaries only;
- tar order is never chronological and must not define state order;
- disk guard: abort before download when used storage is at least 300 GiB;
- no fallback to Onionoo, a third-party Tor mirror, microdescriptor consensus,
  relay server descriptors, votes as the primary incidence source, or current
  API state.

The source stage may retain only signed-consensus observables needed to form a
later mechanism artifact:

- relay RSA identity;
- relay flags;
- consensus bandwidth value and bandwidth-weight modifiers;
- authority identities and signature headers;
- consensus validity times and method;
- source archive/member byte counts and SHA-256 identities.

Relay IP address, nickname, contact, software version, exit policy, geolocation,
and free text are forbidden in TCRR v1. BTC and prior-alpha data remain
forbidden until source support passes.

## Bounded source-only probe already opened

Exactly one consensus member was inspected:

```text
archive:
https://collector.torproject.org/archive/relay-descriptors/consensuses/
  consensuses-2023-01.tar.xz

member:
consensuses-2023-01/13/2023-01-13-09-00-00-consensus
```

Observed source facts:

- member bytes: `2,477,964`;
- member SHA-256:
  `b43dbe5c0297e8d6f2c4344c6c8461eb864aaaf5f117151493072caf3311eba7`;
- `consensus-method 32`;
- `valid-after 2023-01-13 09:00:00`;
- `fresh-until 2023-01-13 10:00:00`;
- `valid-until 2023-01-13 12:00:00`;
- `voting-delay 300 300`;
- 6,307 relay entries and 6,307 flag lines;
- 8 authority entries and 8 distinct consensus signatures.

For this audit only, the corresponding eight vote members were streamed from
`votes-2023-01.tar.xz` to obtain their embedded authority certificates. No vote
incidence was retained. A dependency-free RSA verifier confirmed:

- all 8 certificate identity fingerprints equal SHA-1 of their encoded RSA
  identity keys;
- all 8 authority certificates have valid Tor netdoc self-signatures;
- every certificate covers the complete consensus validity interval;
- all 8 consensus signatures validate under the matching signing key; and
- all 8 authority identities exactly equal the eight `v3ident` trust anchors
  in Tor `0.4.7.13` official source.

The official source tarball was fetched from
`https://archive.torproject.org/tor-package-archive/tor-0.4.7.13.tar.gz` and
matched its published SHA-256
`2079172cce034556f110048e26083ce9bea751f3154b0ad2809751815b11ea9d`.

No other consensus member, relay transition, candidate signal, BTC series, or
economic outcome was inspected.

## Candidate source summaries after a source pass

Only after the source gate passes may a separately frozen mechanism protocol
derive weak, direction-free summaries such as:

- relay-set Jaccard turnover;
- Guard and Exit join/leave breadth;
- total consensus-bandwidth change;
- relay-count change;
- bandwidth concentration or breadth; and
- persistence or reversal of reconfiguration over fixed lags.

These are a bounded mechanism vocabulary, not approved alpha formulas. Signal
direction, interactions, thresholds, and holding periods remain unopened.

## Novelty boundary

Repository-wide search found no prior Tor, CollecTor, onion-relay, or signed
network-consensus alpha implementation. TCRR is source-distinct from existing
exchange microstructure, derivatives, on-chain, macro-FX, calendar, document,
narrative, price-action, weather, and BGP-routing candidates.

Clock-level and trade-level orthogonality are not asserted. If source support
passes, TCRR must compare its frozen event clock with canonical alpha clocks
before any economic promotion.

## Stop and anti-repair rule

The next work unit must preregister exact archive identities, path grammar,
tar-safety rules, consensus schema, authority/signature rules, source coverage,
deterministic replay, and disk guards before opening full incidence.

Any missing frozen archive, archive-hash mismatch, unsafe member, duplicate
target, target/schema failure, invalid validity relation, signing identity not
present in `dir-source`, insufficient distinct authority signatures, or
deterministic-replay failure retires TCRR v1 without changing months, anchors,
interval, availability delay, fields, or thresholds.

Only a source-support pass may authorize a separately committed mechanism
artifact. Only that mechanism pass may authorize BTC evaluation.
