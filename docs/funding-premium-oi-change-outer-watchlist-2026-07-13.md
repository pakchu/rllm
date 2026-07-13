# Funding/premium squeeze × OI-change outer gate watchlist — 2026-07-13

## Decision

Keep manifest rank 7 as a **shadow/watchlist observation only**. Do not register
it in the alpha pool and do not treat it as live-grade. It is the strongest
additional low-correlation observation from the frozen external-state Top-10,
but it misses the predeclared automatic alpha-pool trade count by one trade in
2025 (19 versus 20).

## Fixed rule

First form the previously fixed long base:

1. `funding_available > 0.5`, `funding_rate <= -0.0000167`, and
   `trend_96 >= 0.007485218212390219`; or
2. `premium_available > 0.5`,
   `premium_index_change <= -0.00023471`, and
   `htf_1d_return_4 >= 0.0940403008961932`.

Then require a causally delayed OI-change outer state:

```text
oi_available > 0.5
and (
  oi_logchg288 <= -0.01575477998703718
  or oi_logchg288 >= 0.02278394401073455
)
```

`oi_logchg288` is the log change in Binance `sum_open_interest` over 288
five-minute bars (24 hours). The metrics stream is shifted by one complete
five-minute source bar before feature calculation. The rule therefore uses the
latest completed delayed OI value, never a same/future source row.

Execution is fixed: next 5-minute open, 576 bars (48 hours), stride 12, one
non-overlapping long, no TP/SL, 0.5x, and 6 bp per side.

## Selection and leakage controls

- External thresholds fitted on 2021-04-15 through 2022-12-31.
- Policies selected on 2023 plus separate H1/H2 stability.
- Market, funding, premium, Binance metrics, and DVOL source frames were all
  physically truncated before 2024 when the Top-10 manifest was written.
- Metrics use a 5-minute backward-asof tolerance and are then shifted one bar;
  the delayed OI source is therefore at least 5 and at most 10 minutes old.
- 30 external features were audited; 22 passed fit Spearman
  `max |rho| < 0.30` against BTC funding, premium change, 8-hour trend,
  completed daily momentum, and `lr_impact_72`.
- `oi_logchg288` has maximum fit `|rho| = 0.05455`; correlation with
  `lr_impact_72` is `0.00678`.
- 792 raw gate specs produced 789 unique masks and 24 eligible pre-2024
  variants. Only the frozen Top-10 were replayed on later years.
- Manifest rank: 7.
- Manifest internal SHA-256:
  `e3579c1a9b979f7dc55309f15a2a55298833bffc280f9d3716411bcf6197e74b`.
- Manifest file SHA-256:
  `cf25494be430e4907184c612bf7b6388d123b652ab6a3d8b8f63965460359c5c`.
- Replay reused the existing manifest without mutation and matched every
  selected row exactly.

## Strict results

All CAGR values count the entire named calendar window, including time with no
position. Strict MDD includes pre-entry equity high water, fees, and the
conservative favorable-then-adverse OHLC path over the complete holding period.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Approx. p |
|---|---:|---:|---:|---:|---:|---:|
| Fit 2021-04-15–2022 | +6.25% | 3.60% | 17.91% | 0.20 | 90 | 0.7144 |
| Select 2023 | +25.61% | 25.63% | 10.20% | 2.51 | 31 | 0.0528 |
| Test 2024 | +21.03% | 20.99% | 6.78% | 3.09 | 22 | 0.0180 |
| Eval 2025 | +17.20% | 17.21% | 6.56% | 2.62 | 19 | 0.0381 |
| 2026 to Jun02 | +11.87% | 30.94% | 6.16% | 5.03 | 20 | 0.0771 |
| 2024–2026 to Jun02 | **+58.70%** | **21.05%** | **6.78%** | **3.10** | **61** | **0.000258** |

The combined approximate p-value is about `0.00258` after a conservative
Bonferroni correction for the frozen Top-10. It would not survive correction
over all 792 exploratory specs, but those specs were reduced using pre-2024
data before the later windows were loaded.

Quarter stability is 8 positive, 1 flat/no-trade, and 1 negative quarter. The
only negative quarter is 2025Q4 at -2.33%; 2025Q3 has no completed
split-contained trade.

## Distinctness

The OOS executed-date Jaccard against the prior `lr_impact_72` candidate is
`0.1304` (15 intersections across 61 versus 69 trades). Annual Jaccard values
are 0.1163 in 2024, 0.2121 in 2025, and 0.0769 in 2026. This is materially
different timing despite sharing the same fixed funding/premium base.

The watchlist rule keeps 61 of the ungated base's 82 OOS executed opportunities;
executed-date Jaccard with the base is 0.3619 because non-overlap scheduling
changes subsequent eligible entries after a veto.

## Caveats and next action

- This is not a standalone OI alpha; OI is an external state gate on a fixed
  funding/premium setup.
- Automatic alpha-pool promotion remains false because 2025 has 19 trades,
  one short of the predeclared minimum. The threshold must not be relaxed after
  seeing OOS.
- The observation was identified among a frozen Top-10, and the broader program
  has repeatedly inspected 2024–2026. Fresh forward shadow evidence is still
  required.
- Before live use, verify that the DB's OI field is Binance
  `sum_open_interest` with the same timestamp, one-bar delay, and no stale
  carry. Until then the config remains research-only.

## Artifacts

- `training/search_funding_premium_external_state_gate_alpha.py`
- `tests/test_search_funding_premium_external_state_gate_alpha.py`
- `results/funding_premium_external_state_gate_top10_manifest_2026-07-13.json`
- `results/funding_premium_external_state_gate_alpha_scan_2026-07-13.json`
- `results/funding_premium_external_state_gate_replay_verification_2026-07-13.json`
- `configs/policies/funding_premium_oi_change_outer_shadow_watchlist.json`
