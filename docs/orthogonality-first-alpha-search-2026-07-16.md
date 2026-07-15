# Orthogonality-first alpha search protocol (2026-07-16)

## Decision

A candidate is independent only when its **executed trades** differ from the
existing portfolio. Raw feature correlation is diagnostic; it is not portfolio
orthogonality.

The required audit now measures:

1. exact executed-entry Jaccard;
2. the candidate-entry fraction within a preregistered time window of baseline
   entries;
3. occupied-position-bar Jaccard;
4. zero-filled daily marked-to-market PnL Pearson and Spearman correlations.

An undefined PnL correlation fails closed. At least 10 non-zero PnL days are
required. The default limits are exact-entry Jaccard `<=0.02`, near-6h entry
fraction `<=0.25`, position Jaccard `<=0.15`, and absolute daily-PnL Pearson
`<=0.30`; `<=0.10` near-6h overlap is the stronger target. Limits and policy
must be frozen before an untouched window is opened.

Orthogonality is only a diversification gate. A candidate still needs positive
absolute return, full-calendar CAGR, strict intratrade MDD, enough trades,
causal execution, and realistic costs/funding. A loss-making strategy is not an
alpha merely because it is uncorrelated.

## Current evidence versus frozen annual rank-7

| Candidate | Window | Exact entry Jaccard | Candidate entries near 6h | Position Jaccard | Daily PnL Pearson | Daily PnL Spearman | Economic result |
|---|---|---:|---:|---:|---:|---:|---|
| Fresh Kimchi/FX bidirectional | 2025-2026H1 | 0.0000 | 0.2222 | 0.1300 | 0.2109 | 0.1646 | +22.6205% absolute, 15.4963% CAGR, 5.5692% strict MDD, ratio 2.7825, 45 trades |
| Cross-collateral near-pressure sign | 2024-2026H1 | 0.0000 | 0.0431 | 0.0692 | -0.0406 | -0.0452 | -25.0844% absolute, -11.2604% CAGR, 32.9901% strict MDD, ratio -0.3413, 649 trades |

### Interpretation

- **Frozen annual rank-7 remains the live baseline.** Its 2024-2026H1 result is
  +45.3459% absolute, 16.7286% full-calendar CAGR, 4.9844% strict MDD,
  CAGR/MDD 3.3562, and 55 trades.
- **Fresh Kimchi/FX is the best current independent shadow candidate.** It has
  useful trade/PnL separation and positive results, but its combined ratio is
  below standalone live-grade 3.0 and 2025/2026 were previously viewed. Freeze
  it and seek forward evidence; do not retune it.
- **Cross-collateral near pressure is only a beta event clock.** Its actual
  timing is exceptionally independent, but its pressure sign loses money.
  Three alternative direction paths were rejected before additional future
  replay: generic direction search, fixed primary-weak ridge, and delayed price
  confirmation.

## Fixed-subaccount marginal-value audit

A single, non-optimized allocation was replayed with 75% frozen annual rank-7
and 25% Fresh Kimchi/FX. Both sleeves keep separate capital; their values are
combined on the same five-minute BTC OHLC clock before strict MDD is measured.

| Window | Absolute return | CAGR | Sync strict MDD | CAGR/MDD | Ratio delta vs rank-7 |
|---|---:|---:|---:|---:|---:|
| 2024 | 14.9511% | 14.9183% | 2.2856% | 6.5271 | +1.0623 |
| 2025 | 15.2518% | 15.2630% | 3.0216% | 5.0513 | +0.6649 |
| 2026H1 | 7.8748% | 19.9791% | 3.5167% | 5.6811 | +1.3803 |
| 2025-2026H1 | 24.3089% | 16.6176% | 3.5393% | 4.6951 | +0.7417 |
| 2024-2026H1 | 42.9189% | 15.9184% | 3.5683% | 4.4611 | +0.5686 |

The diversification effect is real in this replay: synchronized strict MDD and
CAGR/MDD improve in every window. Raw CAGR is slightly diluted versus rank-7
outside 2026H1, so this is a **risk-budget shadow candidate**, not a replacement
or pristine OOS promotion.

## Search rule going forward

1. Build and freeze a causal candidate without future-window ranking.
2. Pass the economic gate first on the declared validation window.
3. Audit actual entry, position, and marked-PnL independence against the frozen
   portfolio.
4. Estimate marginal value with synchronized mark-to-market subaccounts; never
   infer portfolio MDD from weighted summary statistics.
5. Promote only after a genuinely untouched window or forward shadow confirms
   both standalone robustness and portfolio marginal value.

This prevents two recurring errors: selecting a weak strategy merely because a
feature has low correlation, and calling an event clock an alpha when no
validated direction edge exists.

## Reproducible artifacts

- `results/fresh_kimchi_orthogonal_alpha_audit_2026-07-16.json`
- `docs/fresh-kimchi-orthogonal-alpha-audit-2026-07-16.md`
- `results/cross_collateral_near_pressure_oos_2026-07-16.json`
- `docs/cross-collateral-near-pressure-oos-2026-07-16.md`
- `results/ccnear_primaryweak_ridge_pre2024_2026-07-16.json`
- `results/ccnear_delayed_price_confirmation_pre2024_2026-07-16.json`
- `training/audit_rank7_fresh_kimchi_fixed_portfolio.py`
- `results/rank7_fresh_kimchi_fixed_portfolio_2026-07-16.json`
- `docs/rank7-fresh-kimchi-fixed-portfolio-2026-07-16.md`
