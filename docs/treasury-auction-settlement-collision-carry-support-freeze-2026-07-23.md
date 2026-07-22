# TASCC-72 source-support builder freeze — 2026-07-23

## Frozen boundary

The TASCC source-only support builder is committed before its first derivation
of exact issue-date collision baskets.

- candidate: `TASCC-72-SOURCE-FAMILY-SEEN`;
- preregistration SHA-256:
  `090cc8e07a76ae033db413d0d0a9356f2a43c38fcb865a06c20f06b5f04cad67`;
- preregistration manifest:
  `56be616879365bca531de961c868b3c39ec406d0436094aa0cf27f5440881f4e`;
- preregistration policy:
  `07fa97ad5ce1720d4296cac7c768f03ba5e153795ba046bbc110b535593d85d9`;
- builder:
  `training/build_treasury_auction_settlement_collision_carry_support.py`;
- builder SHA-256:
  `9a875b0f702c6e6147a6414ead58973fd2c6fb02c28615ba1668f5d5cdb469b4`;
- builder tests SHA-256:
  `9947602ea9a2af9b31a6a5250e0730c0568004bf40dc7e502e01f9267de1f3ee`.

Eighteen preregistration and synthetic support tests passed. They cover the
pre-2024 normalized-panel join, raw transport filtering, incomplete-row
exclusion, belly/long collision geometry, late-result veto, component clocks,
deterministic term permutation, exact-grid execution delay, 72-hour
nonoverlap, split crossing, specificity orientation, comparator overlap, and
market/outcome-field exclusion.

## Authorized read

The first real run may read the frozen normalized auction panel, the two frozen
raw TreasuryDirect transport pages, and timestamps from the five hash-bound
comparator artifacts. All raw transport objects are counted; rows outside the
pre-2024 normalized-panel key set may expose only `auctionDate` and `cusip` to
the candidate code path and are discarded before retaining issue date or term.

The run may not read BTC/funding/future-return rows, calculate PnL/CAGR/MDD,
access the network, or run a subprocess. Its terminal decision is
`PASS_SOURCE` or `REJECT_SOURCE`. A rejection retires this identity without
maturity-group, marker, side, or hold repair.
