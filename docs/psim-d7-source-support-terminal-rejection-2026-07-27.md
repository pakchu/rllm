# PSIM-D7 terminal source-support rejection

Date: 2026-07-27 KST

## Decision

The sealed PSIM-D7 evaluator ran exactly once and terminally rejected at
Gate 5:

```text
split_annual_quarterly_unique_day_support
REJECT_PSIM_D7_UNCHANGED_BEFORE_MARKET_MODEL_OR_OUTCOMES
```

The canonical terminal artifact is:

```text
results/protocol_specification_intent_maturity_d7_source_rejection_2026-07-26.json
SHA-256   36702b4737f1bb37e901241a96e04f30e77132bb6a18ade1fab277a83f15557e
result    45846070617398860a03f5a401047c95a37c7ba3526c37fbcea5a11687e8658b
```

PSIM-D7 is terminal and must not be repaired, resumed, or rerun unchanged.
The retained `/tmp/psim-d7-source` root is forensic residue only.

## Official source statistics

| item | result |
|---|---:|
| source events | 5,356 |
| requested historical blobs | 11,280 |
| materialized Ethereum events | 4,985 |
| materialized Bitcoin events | 371 |
| semantic errors | 0 |
| daily cards completed | 0 |
| completed gates | 4 / 13 |
| first failed gate | 5 |

Bitcoin's exact source class roster was:

| class | blobs |
|---|---:|
| `D4_VALID` | 426 |
| `D7_BIP_LATER_HEADER` | 7 |
| `D7_BIP_PREFIXED_DEPENDENCY` | 1 |

The D7 Bitcoin grammar repair therefore worked as preregistered. Both
replicas produced identical histories, path incidence, event rosters,
semantic receipts, and transport receipts. Gate 4 completed with no
semantic error for either protocol.

## Why there is no return statistic

The run stopped while constructing the first daily-card schedule. It did not
open outcomes and did not load a model. Accordingly, this run has no
backtest, return, CAGR, strict MDD, trade count, hit rate, or economic alpha
claim. `profitability_result=false` means “not evaluated,” not negative
profitability.

The access ledger confirms zero:

- BTC market and funding rows;
- future-return and reward rows;
- model loads and model outputs;
- trades and PnL rows;
- CAGR and strict-MDD values.

## Gate-5 boundary

The official rejection intentionally published only `ValueError`, not the
exception text. A separate read-only post-terminal forensic census binds the
exact failure without invoking the official `run` command again. That census
is recorded independently so it cannot convert this rejection into a pass.

## Reproducibility authority

```text
implementation commit
0e8f22f2680a9edb2cf8497343444c16e4946df0

seal commit
3cb95185bad64b6e82fdd89f8e6f7f3eaa6fda72

seal hash
8088c0902479612bb7cc64f0c729c7375640fcb095bdd9c3d0fe62dcd35fa308

seal SHA-256
ea94ec6566b5925fb0be16bc30aae0e47f7215d42a202943e4d5213f144573d6

source authority hash
98ebc81f94bb14b8dd4f8ae8b10ee9e2a514683f2aa418830fa968cd0e1e8745
```
