# BAFR-24F train rejection — 2026-07-20

## Decision

**REJECT BAFR-24F before sealed selection.** The frozen 2020–2022 train stage
failed every economic gate. The 2023 OHLC and funding values remain unopened,
and no 2024+ value was available to this evaluator. The BAFR threshold, side,
holding period, controls, or accounting will not be repaired after observing
this result.

## Frozen train result

All figures use 0.5x leverage, next-five-minute-open entry/exit, exact realized
funding timestamps/rates with the frozen settlement-mark proxy, 6 bp of
notional per side, full-calendar CAGR including idle cash, and global
favorable-first/adverse-second strict MDD.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Mean gross move | Trades |
|---|---:|---:|---:|---:|---:|---:|
| 2020–2022 train | **-99.70%** | **-85.65%** | **99.72%** | **-0.86** | **-1.92 bp** | 8,220 |
| 2020 | -82.01% | -81.95% | 83.78% | -0.98 | -1.03 bp | 2,575 |
| 2021 | -88.08% | -88.09% | 89.27% | -0.99 | -3.16 bp | 2,758 |
| 2022 | -86.23% | -86.25% | 86.34% | -1.00 | -1.54 bp | 2,885 |
| 2020–2022, 10 bp/side stress | -99.99% | -95.21% | 99.99% | -0.95 | -1.92 bp | 8,220 |

Both directional sleeves failed independently:

| Sleeve | Absolute return | CAGR | Strict MDD | CAGR/MDD | Mean gross move | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Long only | -93.32% | -59.42% | 93.45% | -0.64 | -1.44 bp | 3,854 |
| Short only | -95.58% | -64.63% | 96.20% | -0.67 | -2.34 bp | 4,366 |

The weekly-cluster test, every calendar year, both side contributions, 10 bp
stress, 24 bp gross-edge floor, strict-MDD ceiling, CAGR/MDD floor, and control
superiority gate all failed.

## Why it failed

This is not a hidden long/short bias problem. The exact direction flip produced
only **+1.92 bp** mean gross underlying move and still lost **98.58%** absolute
with CAGR/MDD **-0.77** over 8,220 trades. Every simpler mechanism control also
lost approximately all capital after costs:

| Control | Absolute return | CAGR/MDD | Mean gross move | Trades |
|---|---:|---:|---:|---:|
| Direction flip | -98.58% | -0.77 | +1.92 bp | 8,220 |
| Aggressor flow only | -99.29% | -0.81 | +0.80 bp | 8,608 |
| Tick direction only | -99.78% | -0.87 | -1.92 bp | 8,687 |
| Strict nonzero-tick only | -99.78% | -0.87 | -2.64 bp | 8,234 |
| Carried zero-tick only | -99.28% | -0.81 | +0.34 bp | 8,328 |
| Completed-bar rejection | -99.64% | -0.85 | +0.13 bp | 9,224 |
| Stale 1h | -99.51% | -0.83 | -0.68 bp | 8,219 |
| Stale 24h | -99.67% | -0.85 | -1.62 bp | 8,211 |

At 0.5x leverage, the frozen 6 bp/side cost is about 6 bp of equity per round
trip. BAFR's already-small underlying movement has the wrong sign, while even
the best sign-flipped/null gross movement is far below the preregistered 24 bp
floor. The dense q90 clock therefore compounds weak microstructure noise and
transaction costs rather than a two-hour price edge.

## Leakage and seal evidence

- evaluator-freeze SHA256:
  `158f4ccf49874ce84d107d01760a8653cf21f8c389b627635b72a1dcd513908d`;
- train artifact SHA256:
  `1c6bd35b8b3f527b34394c9b1033351a2c69ca7505747151224da73512e564a9`;
- train canonical result hash:
  `c80e8e1b8f6eb775eef7cc100cb13660efd9230415fdae2ba384714c60af157f`;
- opened windows: only `train_2020_2022`;
- `selection_2023_opened`: `false`;
- decision: `reject_before_selection`; and
- selection artifact: absent.

## Research implication

The next candidate must not be another dense five-minute q90 transformation of
flow versus same-bar price response. Before opening outcomes it should define a
sparser event with an economic reason to persist beyond the signal bar and a
plausible gross move comfortably above round-trip costs. BAFR itself is retired;
its failed outcome cannot be reused to tune a threshold, hold, side, or gate.
