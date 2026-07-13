# Funding-settlement curl alpha search — 2026-07-13

## Hypothesis

A funding settlement is a known intervention on leveraged inventory.  The same
net premium and open-interest changes can imply different risk depending on
**their order** around settlement:

- before settlement, the funding-sign-aligned premium and OI may build together;
- after settlement, premium may normalize while OI remains trapped;
- the antisymmetric response
  `curl = z(B_pre) * z(OI_post) - z(OI_pre) * z(B_post)` distinguishes that
  ordering from an ordinary sum or correlation;
- `funding_sign * curl > 0` is interpreted as trapped inventory and traded
  contrarian to the funding sign.

This is a path-dependent mechanism, not raw funding, momentum or OI level.

## Causal protocol

- Every input was physically truncated before `2024-01-01`; OOS was not opened.
- Exact Binance `funding_time` was rounded **up** to the first 5-minute bar that
  cannot precede settlement.
- Premium uses completed hourly premium-index closes by backward-as-of join.
- Binance OI is delayed by one complete 5-minute source bar.
- The signal is formed only after a 1h or 2h post-settlement observation window
  and enters at the following 5-minute open.
- All event standardization uses `shift(1)` and the prior 180 settlement events.
- Thresholds use the fit window only (`2020-12-01` through 2022).
- Execution is 0.5x, 6bp/side, non-overlapping fixed holds and the canonical
  conservative favorable-first/adverse-second strict OHLC MDD.

Search: pre-window `{1h,2h}` × post-window `{1h,2h}` × absolute-funding fit
quantile `{70%,85%}` × curl fit quantile `{75%,85%,90%}` × mode
`{curl,trapped-structure}` × hold `{2h,4h,8h}` = 144 candidates.

## Best adequately populated candidate

Selected by predeclared segment stability, not by a single 2023 return:
1h pre / 1h post / funding q70 / curl q85 / 8h hold.

Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.

| Window | Result |
|---|---:|
| Fit through 2022 | `+26.44 / +11.92 / 10.53 / 1.13 / 192` |
| 2023 selection | `-2.55 / -2.55 / 7.82 / -0.33 / 65` |
| 2023 H1 | `+4.21 / +8.67 / 3.67 / 2.36 / 27` |
| 2023 H2 | `-6.48 / -12.46 / 7.74 / -1.61 / 38` |

Only five of six half-year robustness segments were positive.  No candidate
with enough fit and half-year trades cleared a positive minimum core ratio; the
best such minimum was negative.

## Controls

For the same selected timing and hold:

| Variant | Fit | 2023 |
|---|---:|---:|
| Direction flip | `-38.97 / -21.10 / 42.83 / -0.49 / 192` | `-5.37 / -5.37 / 10.60 / -0.51 / 65` |
| Symmetric dot, no curl | `+4.85 / +2.30 / 14.34 / 0.16 / 196` | `-1.71 / -1.71 / 8.18 / -0.21 / 57` |
| Funding-only contrarian | `-19.22 / -9.74 / 32.22 / -0.30 / 787` | `-20.73 / -20.75 / 21.70 / -0.96 / 268` |
| Fake settlement +4h | `-1.08 / -0.52 / 23.57 / -0.02 / 202` | `+5.59 / +5.59 / 8.06 / 0.69 / 73` |

Curl is not a complete no-op—the symmetric and funding-only controls are worse
in fit—but its edge changes sign in 2023 H2 and is weaker than a fake-time
control in 2023.  That invalidates the claimed settlement-specific mechanism.

## Decision

**Reject in preflight; do not open OOS and do not trade.**  Keep the exact static
mapping only as failure provenance.  A future retry must use genuinely new
information about settlement ownership or liquidation transfer, not another
threshold/hold sweep over these same four changes.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python -m training.search_funding_settlement_curl_alpha
PYTHONPATH=. .venv/bin/pytest -q tests/test_search_funding_settlement_curl_alpha.py
```
