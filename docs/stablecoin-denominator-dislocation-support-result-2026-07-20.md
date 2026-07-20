# SDDR-12 source-support result — 2026-07-20

## Verdict

**Reject SDDR-12 before opening outcomes.**

The frozen 2023 source clock had adequate incidence, direction balance, and
calendar dispersion, but failed the preregistered independence gate against
SQFD's broader `no_usdt_lag` flow clock. The SDDR primary clock had 44.87% of
its entries within plus/minus six hours of that comparator; the maximum allowed
bidirectional containment was 35%.

No BTCUSDT perpetual OHLC, funding cash flow, future return, label, PnL,
absolute return, CAGR, or MDD was opened. Threshold, sign, holding period, and
feature definitions remain unchanged. This candidate is not eligible for an
outcome evaluator or a renamed same-source repair.

## Frozen primary support

| Statistic | Result | Gate | Pass |
|---|---:|---:|:---:|
| Events | 78 | at least 30 | yes |
| LONG / SHORT | 40 / 38 | each at least 30% | yes |
| LONG / SHORT share | 51.28% / 48.72% | each at least 30% | yes |
| September events | 13 | at least 5 | yes |
| October events | 27 | at least 5 | yes |
| November events | 25 | at least 5 | yes |
| December events | 13 | at least 5 | yes |
| Largest month share | 34.62% | at most 45% | yes |
| First entry | 2023-09-02 18:05 UTC | — | — |
| Last exit | 2023-12-30 07:05 UTC | — | — |

Source-only control incidence was 74 events for `no_disagreement`, 341 for
`usdc_only`, 176 for `fdusd_only`, and 78 for `stale_1h`. These are diagnostic
counts, not alternative policies and not profitability results.

## Frozen novelty result

| SQFD comparator | Exact Jaccard | SDDR within ±6h | SQFD within ±6h | Max bidirectional | Gate |
|---|---:|---:|---:|---:|:---:|
| `primary` | 0.76% | 19.23% | 20.00% | 20.00% | pass |
| `no_usdt_lag` | 3.78% | 44.87% | 28.95% | **44.87%** | **fail** |
| `no_participation` | 0.60% | 21.79% | 15.73% | 21.79% | pass |

The low exact overlap but high six-hour proximity to `no_usdt_lag` indicates
that cross-quote price dislocations are often temporally adjacent to the same
alternative-stablecoin flow shocks. SDDR therefore adds less independent clock
information than its different algebra initially suggested. This is a source
interpretation only; no return was inspected to rationalize the rejection.

## Reproducibility and boundary

- preregistration commit: `0ea87da`
- support-evaluator commit: `f675ff3`
- event clock:
  `data/stablecoin_denominator_dislocation_clocks_2023.csv.gz`
  - SHA-256:
    `eaf2d6c187af9855e76474d2951fcdc12267174980a72649b73d068982ca8c69`
- support report:
  `results/stablecoin_denominator_dislocation_support_2026-07-20.json`
  - SHA-256:
    `1d7e8561963d903c5963bbd081c5cf0c9926dc9221f9a09d23fb565ab27f7bea`
  - manifest hash:
    `7292e1c1f8b1979ca95eabd2ada14cc2009f970a6e21d90a70b4c016d8de0302`

Two complete executions were byte-identical. The next alpha search must move
to an independent observable rather than optimize this failure. BitMEX
quote/trade/liquidation resiliency was considered but is not an eligible next
source: commit `75afc39` already rejected BitMEX market data for profit-seeking
production use because the repository has no written commercial-data consent.
The next candidate must first pass source entitlement, live parity, history,
and repository-novelty checks before any BTC outcome is opened.
