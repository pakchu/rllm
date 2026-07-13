# Liquidation-Scar First-Passage / One-Touch Alpha — Preflight

Date: 2026-07-13

## Structural retry

The repeated local-q80 scar policy overtraded and was rejected. This experiment did
not retune its threshold. Instead, each delayed-OI/taker event created an individual
signed price-level object with this causal lifecycle:

1. deposit after completed bar `t` has been queried;
2. require price to leave the level by at least 50 bp;
3. trigger only on the first later crossing back into a 10/20 bp zone;
4. consume the scar permanently after that one touch.

The opposite economic mappings were both tested: fade an exhausted liquidation
level, or continue through a cleared-inventory corridor.

## Protocol

- Physical rows strictly before `2024-01-01`; frozen OOS stayed unopened.
- Open interest delayed one complete 5-minute bar and standardized from prior bars.
- Query/consume prior scars before depositing the current completed event.
- 48 fixed policies: contraction tail `{q90,q95}`, expiry `{288,864,2016}` bars,
  touch zone `{10,20}` bp, mapping `{fade,permeability}`, hold `{24,72}` bars.
- Next-open execution, 0.5x, 6 bp/side, split-contained exits, conservative strict MDD.
- Admission required positive fit and 2023, ratio at least 3 in both, sufficient
  trades, and non-negative 2023 halves.

## Strongest adequately populated ranking

Parameters: q95 contraction, 288-bar expiry, 20 bp zone, permeability mapping,
72-bar hold. Median touch age was 49 bars.

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Fit (2020-06..2022) | -64.60% | -33.09% | 65.52% | -0.51 | 1,158 |
| Selection 2023 | +1.90% | +1.90% | 10.15% | +0.19 | 266 |
| 2023 H1 | +1.84% | +3.74% | 7.64% | +0.49 | 153 |
| 2023 H2 | +0.06% | +0.13% | 7.34% | +0.02 | 113 |

No one of 48 policies was positive in both fit and 2023. The only positive 2023
candidate above is a regime inversion, not stable evidence.

## Controls

| Variant | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| Exact direction flip | -37.71% / -0.37 | -29.65% / -0.95 |
| Monthly relocated scar price | -23.40% / -0.35, 63 trades | +5.91% / +1.61, 10 trades |
| Deposit at 12-bar-lagged price | -52.31% / -0.45 | -5.91% / -0.38 |
| Remove leave quarantine | -71.42% / -0.53 | -23.68% / -0.70 |

The relocated-price placebo has only ten 2023 trades and fails fit, so it is not
evidence for the real spatial mechanism.

## Cost decomposition of the top policy

| Cost per side | Fit return / ratio | 2023 return / ratio |
|---|---:|---:|
| 0 bp | -29.08% / -0.33 | +19.54% / +2.57 |
| 1 bp | -36.84% / -0.37 | +16.40% / +2.09 |
| 3 bp | -49.90% / -0.44 | +10.37% / +1.24 |
| 6 bp | -64.60% / -0.51 | +1.90% / +0.19 |

The 2023 direction specificity is real enough to retain the continuous spatial
state as a weak regime-context beta, but the full fit period is negative even at
zero cost. It cannot support a standalone alpha claim.

## Decision

Reject the exact first-passage/one-touch static mapping before OOS. With both the
repeated-local and one-touch usages now rejected, do not continue tuning price
zones, expiry, event tails, or holds. Any future use is limited to a continuous
regime token inside a genuinely different learner with fresh forward evidence.

Artifacts:

- `training/search_liquidation_scar_first_passage_alpha.py`
- `results/liquidation_scar_first_passage_alpha_scan_2026-07-13.json`
- `tests/test_search_liquidation_scar_first_passage_alpha.py`
