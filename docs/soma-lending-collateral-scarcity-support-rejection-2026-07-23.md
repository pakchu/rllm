# SLCS-72 source-support rejection — 2026-07-23

## Decision

**Reject `SLCS-72-NEW-SOURCE` before outcomes. Do not repair this identity and
do not open its BTC or funding returns.**

The committed source-only builder reproduced byte-identical artifacts on two
runs:

- report: `results/soma_lending_collateral_scarcity_support_2026-07-23.json`;
- report SHA-256:
  `354f3edb9f1d9bdbac1f609e50882f2e4d1df6ee8cfa555287ca99a15148a738`;
- report manifest:
  `95d6e4b3220645bc63d323b7834286beb6b9b7f02bdf8fe2f6db1f6bfc52ad4b`;
- source/control clock:
  `results/soma_lending_collateral_scarcity_clocks_2026-07-23.csv.gz`;
- clock SHA-256:
  `b3fe0dc8c895f9a8974cdf08b5bed9d58ff693b8aca7ed59c224627de930a948`.

## Source support

All 1,259 operations and 182,616 security-detail rows reconciled. All
operations produced complete four-component features, and 1,007 operations
were rank-ready after the strict 252-operation warmup. Ten pairs of operations
shared an availability timestamp; each pair was ranked as one causal batch, so
neither simultaneous operation entered the other's prior history.

The primary clock had adequate count and side support but violated its frozen
maximum-gap gate:

| Split | Events | Long | Short | Maximum accepted-entry gap |
|---|---:|---:|---:|---:|
| train 2020–2022 | 75 | 39 | 36 | **78 days** |
| selection 2023 | 35 | 13 | 22 | 35 days |

The preregistered maximum was 45 elapsed days. All other frozen source-support
gates passed, so the failure cannot be repaired by changing the threshold.

## Novelty failure

SLCS also narrowly failed the independent novelty gate against the frozen
overnight-RRP clock:

- primary SLCS entries over 2020–2023: 110;
- one-day one-to-one RRP matches: 41;
- SLCS containment: **37.27%**, above the frozen 35% maximum;
- exact-entry intersection: 0;
- signed five-minute occupied-exposure correlation: 0.0458.

The low signed correlation shows that SLCS was not directionally equivalent to
RRP, but its event timing was still too concentrated around an already-studied
Federal Reserve liquidity clock. All other qualifying comparator groups passed.

## Outcome boundary and retained lesson

The run read zero BTC bars, funding rows, return rows, PnL, CAGR, or MDD and ran
zero economic simulations. The candidate therefore ends as a clean source-only
negative result.

The useful retained information is narrower than an alpha claim: aggregate
SOMA securities-lending scarcity transitions are sufficiently frequent and
balanced, but their long quiet interval and RRP-adjacent timing fail the
precommitted deployment standard. A successor must use a different causal
object or genuinely independent timing mechanism under a new identity rather
than relaxing SLCS-72.
