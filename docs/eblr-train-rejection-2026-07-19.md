# EBLR-60/30 train rejection — 2026-07-19

## Decision

Reject the exact preregistered ETH→BTC liquidation relay rule. Train failed;
test and eval remain physically sealed and were not opened.

## Hardened train result

Calendar: `2023-06-25` inclusive → `2023-10-15` exclusive.  
Exposure: 1x. Cost: 6bp per side base, 10bp per side stress.  
Trades: 21 total, 8 long, 13 short.

| metric | result |
|---|---:|
| absolute return | -1.8146% |
| full-calendar CAGR | -5.7971% |
| strict MDD | 3.3476% |
| CAGR / strict MDD | -1.7317 |
| mean gross trade | +3.3441bp |
| mean net trade | -8.6513bp |
| 10bp/side stress absolute return | -3.4518% |
| stationary-bootstrap one-sided p | 0.874643 |

The preregistered target required positive base and stress returns,
CAGR/strict-MDD at least 3, MDD at most 15%, and bootstrap `p <= 0.10`.
Only the MDD and source-support/count checks passed.

## Mechanism diagnosis

- Following the ETH forced-flow sign had only **+3.34bp gross per trade**, far
  below the 12bp round-trip base cost.
- Flipping direction was also negative gross (`-3.34bp`), so the result does
  not support a simple exhaustion/fade repair.
- Removing the BTC quiet gate produced +1.5940% base return but failed 10bp
  stress (-2.4668%), ratio (0.5367), and significance (`p=0.3692`). The quiet
  gate did not create a robust relay edge.
- The deliberately noncausal future-ETH placebo returned +2.6492%, ratio 6.04,
  and stayed positive under stress, although it missed significance
  (`p=0.1427`). This is consistent with much of the move occurring **before**
  the archived ETH trigger becomes tradable, not with a causal post-trigger
  BTC relay.

## Integrity statement

- Primary and all six control clocks were frozen before train outcomes.
- The strict path includes global/pre-entry HWM, entry cost, every held 5m
  favorable then adverse OHLC extreme, exact funding settlement marks
  including dropped-credit boundary marks, virtual adverse exit cost, and
  actual exit cost.
- CAGR includes the full idle calendar.
- Result SHA-256:
  `1829a482bf0f99aea35814a6050ac3f6ad844bf8ab4aa7405332ddcbe29403fa`.
- Result hash:
  `7d42383aa2fd87bc8622870c5d7a88f01a1a972b67fb366ac613eb024d429f3d`.

No threshold or direction repair is allowed from this train outcome. The next
candidate must use a different source mechanism and should target a signal
available before the forced-flow price move, rather than another post-event
liquidation relay.
