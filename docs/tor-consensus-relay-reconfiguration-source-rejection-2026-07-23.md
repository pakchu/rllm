# TCRR source rejection — 2026-07-23

## Terminal decision

**TCRR v1 is rejected with `REJECT_NO_REPAIR`.** No Tor relay-transition
feature, event direction, threshold, BTC clock, bar, return, PnL, prior-alpha
clock, or portfolio result was opened.

The terminal machine result is:

- `results/tor_consensus_relay_reconfiguration_source_support_2026-07-23.json`
- file SHA-256:
  `abf3b67bee149568f054c37d80247aeeb3041ad6e8619a5acd948dc9e4255505`
- support manifest hash:
  `ffe4ff5ffd635c6d6cb7db39f56a656038bfe888499b5d68523a9f445b803faa`
- frozen protocol file SHA-256:
  `aa1d4670dcfcfeaf5e1d32b58c7e309b0745a05ca964cb0d2178b2ccf174db70`
- frozen protocol manifest hash:
  `454afbd388651932de3c71bcfa5dcd0d76ce2633fda55174c81aee58c4603564`
- committed builder SHA-256:
  `6312b675fe737e1a4eb7e0bf0ff38a928cf2792b9b9e9b0f40749c564d5a5531`

## Failure point

The first full monthly source object was the frozen official archive:

```text
https://collector.torproject.org/archive/relay-descriptors/consensuses/
  consensuses-2020-01.tar.xz
```

Its transport passed the frozen source identity:

- compressed bytes: `23,463,012`;
- compressed SHA-256:
  `16ff174aefea61518243120b2c3ada54d0b3bdb0ccca3e051f6079e46d23ff8e`;
- content type: XZ archive transport.

The XZ and tar streams were readable and target members were parsed, but the
archive did not contain the preregistered target member:

```text
consensuses-2020-01/30/2020-01-30-00-00-00-consensus
```

No unexpected target member replaced it. The builder fetched the same monthly
object again for fatal confirmation and observed the same body identity and
transport metadata before emitting the rejection. The checkpoint contains
zero completed archive rows because a month is atomic.

## Why the run cannot be repaired

TCRR v1 froze all four daily anchors, all `5,844` target documents, 100% target
membership, and a fail-closed rule for any missing target before opening full
source incidence. Dropping the missing timestamp, changing the anchor panel,
allowing a coverage fraction, carrying a stale consensus, or imputing from an
adjacent hour now would be a post-incidence repair.

The missing document does not imply that Tor consensus data are unusable in
general. It means this exact complete-panel TCRR v1 contract is false. A future
Tor-derived candidate would need a genuinely new source version and a newly
motivated missingness model selected before any incidence, not a continuation
of TCRR v1.

TCRR is therefore retired before mechanism construction and before all
economic evaluation.
