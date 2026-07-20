# UCBR-12 source-support rejection — 2026-07-20

## Verdict

**Reject UCBR-12 before opening outcomes.**

The frozen direct-stablecoin clock produced 31 balanced events but failed two
immutable support checks:

1. September 2023 produced 4 events, below the required 5;
2. 58.06% of UCBR entries were within plus/minus six hours of an SDDR primary
   event, above the 35% novelty limit.

No BTCUSDT perpetual OHLC, funding, future return, label, PnL, absolute return,
CAGR, or MDD was read. Threshold, breadth, side, and hold are not repaired.

## Primary support

| Statistic | Result | Gate | Pass |
|---|---:|---:|:---:|
| Events | 31 | at least 30 | yes |
| LONG / SHORT | 16 / 15 | each at least 30% | yes |
| LONG / SHORT share | 51.61% / 48.39% | each at least 30% | yes |
| September | **4** | at least 5 | **no** |
| October | 9 | at least 5 | yes |
| November | 7 | at least 5 | yes |
| December | 8 | at least 5 | yes |
| Largest month share | 29.03% | at most 45% | yes |
| First entry | 2023-08-29 22:05 UTC | — | — |
| Last exit | 2023-12-27 06:05 UTC | — | — |

The three August events occur after the strictly-prior 672-hour warmup and do
not substitute for the preregistered September minimum.

## Novelty result

| Comparator | Exact Jaccard | UCBR within ±6h | Comparator within ±6h | Maximum | Gate |
|---|---:|---:|---:|---:|:---:|
| SDDR `primary` | 5.83% | **58.06%** | 46.15% | **58.06%** | **fail** |
| SQFD `primary` | 1.18% | 16.13% | 9.09% | 16.13% | pass |
| SQFD `no_usdt_lag` | 0.69% | 25.81% | 7.89% | 25.81% | pass |
| SQFD `no_participation` | 0.84% | 19.35% | 6.74% | 19.35% | pass |

Direct stablecoin breadth is distinct from BTC flow, but it is not sufficiently
independent from SDDR's BTC cross-quote denominator clock. The high temporal
proximity supports the interpretation that both observe the same stablecoin
relative-value shock family. This conclusion uses timestamps only, not returns.

## Source-only controls

| Clock | Events | LONG / SHORT |
|---|---:|---:|
| `all_four` | 9 | 4 / 5 |
| `leave_out_usdc` | 20 | 11 / 9 |
| `leave_out_tusd` | 21 | 11 / 10 |
| `leave_out_usdp` | 9 | 4 / 5 |
| `leave_out_fdusd` | 11 | 4 / 7 |
| `median_only` | 42 | 22 / 20 |
| `stale_1h` | 31 | 16 / 15 |

These counts cannot be used to select a replacement control. UCBR-12 and its
same-source variants are retired from this research branch.

## Integrity anchors

- preregistration commit: `9f73a53`
- support-evaluator commit: `72d4b76`
- clock SHA-256:
  `20b3ee9f82696222a3adbde0045dfde53e0e240e85162e463166aa8fe90b1a8f`
- report SHA-256:
  `fad1959d09e7261d2d03fadc5abdae5f9ee0b3a78339763e3f5b6566bc42a8e8`
- report manifest hash:
  `e73f3522196e6a03f78b3404b2809eb9e3e09a18bfcfeae5a543fa7f4a87e8c4`

Two complete executions were byte-identical. The next candidate must use an
observable that is not another stablecoin relative-price or BTC stablecoin-book
transformation.
