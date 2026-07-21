# AMTR-48 source-support rejection — 2026-07-21

## Verdict

**Retire AMTR-48 before comparator novelty or economic outcomes.**

The source evaluator reproduced byte-identically and read only the promoted
Ethereum event panel. It opened no SQFD, SDDR, UCBR, supply-breadth, BTC market,
funding, future-return, PnL, absolute-return, CAGR, or strict-MDD row.

## Frozen primary support

| Statistic | Result | Gate | Pass |
|---|---:|---:|:---:|
| Globally non-overlapping pairs | **5** | at least 60 | no |
| LONG / SHORT | 1 / 4 | each at least 30% | no |
| 2021 pairs | **0** | at least 12 | no |
| 2022 pairs | **4** | at least 12 | no |
| 2023 pairs | **1** | at least 12 | no |
| Distinct minters | **1** | at least 5 | no |
| Maximum minter share | **100%** | at most 40% | no |
| Maximum minter share within each side | **100%** | at most 60% | no |
| Maximum mint-recipient share | **100%** | at most 50% | no |
| Maximum entry-month share | **80%** | at most 20% | no |

Four entries occurred in October 2022 and one in November 2023. The primary
clock is too sparse, one-sided, actor-concentrated, recipient-concentrated, and
calendar-concentrated for a standalone alpha test.

## Source-only controls

| Clock | Accepted events |
|---|---:|
| `cross_minter` | 459 |
| `no_amount_ratio` | 11 |
| `no_minimum_gap` | 5 |
| `stale_6h` | 5 |

The large cross-minter control count does not rescue AMTR-48. It demonstrates
that opposite large mint/burn activity is common across different authorized
roles while the frozen **same-role turnaround** mechanism is nearly absent.
Promoting that control would replace actor continuity with an aggregate
cross-actor flow sequence after seeing incidence, making it a new mechanism
and likely a disguised aggregate issuance/burn clock.

Removing the amount ratio only increases support from 5 to 11; removing the
minimum gap changes nothing. Neither is close to the frozen event, year,
actor, or concentration floors. Threshold, address identity, pair window,
direction, ratio, hold, and support gates remain unchanged.

## Decision boundary

Because the earliest conjunctive source-support stage failed, comparator
timestamps were not opened. No novelty statistic was calculated, and no
outcome evaluator is authorized. The correct next step is a separately frozen
observable or mechanism, not an AMTR control or threshold repair.

## Integrity anchors

- source promotion commit: `d367303`
- AMTR mechanism freeze commit: `fcdcdec`
- AMTR evaluator commit: `e09c721`
- source clock:
  `data/authorized_minter_turnaround_relay_clocks_2020_2023.csv.gz`
- clock SHA-256:
  `30875029daa4d6e2eff9a59f53d45eda57dbced05988df089c38a6c81abfa0f6`
- support report:
  `results/authorized_minter_turnaround_relay_source_support_2026-07-21.json`
- report SHA-256:
  `057dd07b7b5722c51305b35d5c9c77ee8bdbbe20df57f616821130d02de0d2e1`
- support manifest hash:
  `40642c24491d4b18928a6820510af1f3da6c84dc6f841dfc04fc40ec4de959a9`

