# Bitnodes reachable-node topology source-axis rejection — 2026-08-13

## Decision

Reject before preregistration and before opening any historical snapshot body,
source incidence, BTC outcome, Gross9 row, execution price, or funding value.

The proposed independent observable was change in the reachable Bitcoin P2P
node topology (node count, network-type mix, client-version breadth, or ASN
concentration) during a high-volatility BTC state.

## Official transport boundary

The official Bitnodes API documentation states that ordinary API snapshots at
full granularity are retained for only up to 60 days.  Older snapshots are
listed on the Bitnodes Archive page at approximately seven-day intervals, but
the official archive page states that historical download links are available
only to PRO users; academic whitelisting is separately approval-gated.

Official metadata inspected:

- `https://bitnodes.io/api/`
- `https://bitnodes.io/archive/`
- `https://bitnodes.io/tos/`

The current repository environment has no authenticated Bitnodes PRO archive
contract or approved academic whitelist.  Archive-page filenames, timestamps,
sizes, and SHA-256 values establish object identity but do not expose the node
rows needed to reproduce topology features across the complete frozen
2023-07-01 through 2026-08-01 window.

## Why the axis cannot proceed

A recent 60-day API panel cannot satisfy the unchanged train/test/eval/final
periods.  Scraping rendered historical summaries, relying on search-engine
caches, substituting third-party mirrors, or starting a forward collector now
would not reconstruct the point-in-time historical source.  None is an
admissible replacement for the official, hash-listed archive objects.

This is a source-access rejection, not a negative alpha result.  It may be
reconsidered only if the environment receives authenticated archive access
that explicitly covers the complete frozen window; the current candidate may
not be repaired with a shorter sample or another provider.

## Boundary record

- Only official documentation and archive metadata were inspected.
- No snapshot JSON body or node-level observation was downloaded.
- No event count, topology statistic, timestamp panel, side, return, or PnL was
  computed.
- No formula, threshold, clock, side, hold, or universe was selected from
  incidence or outcomes.

