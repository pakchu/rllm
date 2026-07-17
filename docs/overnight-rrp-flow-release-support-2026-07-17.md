# ORFR-1 source-only support and controls — 2026-07-17

## Decision

**PASS; freeze the strict evaluator.** This work unit opened no BTC price,
funding, return, trade, CAGR, drawdown, or existing-alpha PnL.

| Clock | Stage1 events | Long | Short | Sealed 2023 events | Long | Short |
|---|---:|---:|---:|---:|---:|---:|
| Primary 5-operation residual | 112 | 63 | 49 | 74 | 50 | 24 |
| One-day-delta mechanism control | 99 | 57 | 42 | 83 | 45 | 38 |
| One-release-delay control | 112 | 63 | 49 | 74 | 50 | 24 |

The primary satisfies every frozen source-only support gate: at least 100
Stage1 events, 45 per Stage1 year, 35 per side, 60 events and 15 per side in
sealed 2023, 20 events per 2023 half, and at most 20% single-month
concentration in full Stage1 and full 2023.

## Clock integrity

- source rows: 1,498;
- complete source rows: 1,489;
- later-updated quarantined rows: 9;
- market/funding rows opened: 0;
- all entries: source availability + 5 minutes;
- all exits: next normal ON RRP availability + 5 minutes;
- overlaps: 0;
- control-ledger rows: 1,001;
- control-ledger SHA-256:
  `7242d9870627dfc0cf067ff87d9664a1576dd374cb8985e927b40f15d1e3d480`.

Quarantined rows expose no amount and reset the local baseline. The primary
and one-day controls are independently rebuilt from the frozen official panel.
The delay control moves each primary side by exactly one complete normal ON
RRP operation.

## Source-clock diagnostics

These are timestamp diagnostics only, not performance overlap:

| Control vs primary | Stage1 entry Jaccard | 2023 entry Jaccard |
|---|---:|---:|
| One-day-delta tail | 0.3526 | 0.3894 |
| One-release delay | 0.3827 | 0.3832 |

Direction-flip and deterministic-random-side controls will reuse the exact
primary entries inside the evaluator, so their entry Jaccard is 1 by design.

## Frozen artifacts

- Builder: `training/build_overnight_rrp_flow_release_support.py`
- Tests: `tests/test_build_overnight_rrp_flow_release_support.py`
- Support JSON:
  `results/overnight_rrp_flow_release_support_2026-07-17.json`
- Support manifest hash:
  `ce84c78e94ce213a3c3635511c64579a87498970a13319cf32bf0f50a42f3d0a`
- Support JSON SHA-256:
  `7ac6b888bc993015951f09ee1f0ef3c19b47faa8d92e99864a87b139ec57dd6a`

The next work unit may freeze the evaluator, but still may not open an outcome.
Only the subsequent Stage1 command may physically parse `[2021, 2023)`.
