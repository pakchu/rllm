# Hawkes liquidation-cascade alpha search — 2026-07-13

## Idea

This experiment treats large standardized BTC returns as a two-sided,
self-exciting point process rather than ordinary momentum.

1. Each completed 5-minute return is divided by volatility estimated through
   the preceding bar only.
2. Positive and negative `|z| >= 2` jumps excite separate exponentially
   decaying intensities (one-hour half-life).
3. Their normalized difference is a directional cascade pressure.
4. The pressure is followed only while one-bar-delayed Binance open interest
   is still higher than 24 hours earlier.  This distinguishes a still-loaded
   cascade from an OI-unwind/exhaustion state.
5. Decisions are sampled hourly and execute at the next 5-minute open for a
   fixed 24-hour hold.

This mechanism is distinct from raw momentum, rolling extrema and REX: it uses
the event-arrival memory of rare jumps plus derivatives inventory conservation.

## Frozen protocol

- Market and Binance metrics were physically truncated before `2024-01-01`
  for all feature fitting and policy selection.
- Binance metrics were delayed by one complete 5-minute source bar.
- Search: jump z `{2.0, 2.75}` × intensity half-life `{12, 48, 144}` bars ×
  absolute-imbalance fit quantile `{0.8, 0.9}` × OI mode
  `{unwind-follow, build-fade, build-follow, all-follow}` × hold
  `{72, 144, 288}` bars = 144 policies.
- Execution: next-bar open, 0.5x, 6bp/side, non-overlapping fixed holds,
  split-contained exits, conservative favorable-first/adverse-second strict
  OHLC MDD.
- The selected policy, threshold, market/feature prefix hashes and signal hash
  were written to the manifest before the full input was loaded.
- OOS replay did not change the policy, threshold, direction, hold or costs.

## Pre-2024 selection

Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.

Selected policy: jump z `2.0`, half-life `12`, fit q90 threshold
`0.44072451081084396`, OI-build follow, 288-bar hold.

| Window | Result |
|---|---:|
| Fit 2020-10-15 through 2022 | `+35.35 / +14.66 / 23.38 / 0.63 / 368` |
| 2023 selection | `+70.80 / +70.87 / 8.47 / 8.37 / 143` |
| 2023 H1 | `+43.73 / +107.93 / 8.47 / 12.75 / 78` |
| 2023 H2 | `+18.84 / +40.86 / 8.45 / 4.84 / 65` |

Six of seven pre-2024 robustness segments were positive.  The deliberate OOS
admission was exploratory, not a promotion: 2021 H1 was negative, 2021 H2 was
nearly flat, and the full fit ratio was only `0.63`.

## Frozen OOS result

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Test 2024 | +0.97% | +0.97% | 16.28% | 0.06 | 149 |
| Eval 2025 | +8.22% | +8.23% | 9.12% | 0.90 | 135 |
| 2026 through Jun 02 | -5.77% | -13.31% | 19.81% | -0.67 | 62 |
| Combined 2024–2026 | +0.65% | +0.27% | 19.81% | **0.01** | 347 |

The combined mean-trade approximation was not significant (`p=0.888`, effect
size `d=0.008`).  The apparent 2023 edge did not survive untouched replay.

## Negative controls and cost stress

| Variant, combined OOS | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Selected, 10bp/side | -12.40% | -5.33% | 24.74% | -0.22 | 347 |
| Direction flip, 6bp/side | -37.74% | -17.80% | 45.90% | -0.39 | 347 |
| Remove OI-build gate, 6bp/side | -27.88% | -12.64% | 48.38% | -0.26 | 554 |

The controls show that excitation direction and OI state are not no-ops, but
their incremental information is far too weak and unstable to pay costs.

## Decision

**Reject OOS; do not trade or retune this static mapping.**  Preserve continuous
up/down excitation intensity, normalized imbalance and OI-build state as weak
beta context for a materially different, predeclared learner.  Record the exact
q90/build-follow/24-hour mapping as gamma because its frozen OOS ratio is `0.01`.

## Reproduction

```bash
PYTHONPATH=. python -m training.search_hawkes_liquidation_cascade_alpha
PYTHONPATH=. .venv/bin/pytest -q tests/test_search_hawkes_liquidation_cascade_alpha.py
```

Artifacts:

- `training/search_hawkes_liquidation_cascade_alpha.py`
- `tests/test_search_hawkes_liquidation_cascade_alpha.py`
- `results/hawkes_liquidation_cascade_frozen_manifest_2026-07-13.json`
- `results/hawkes_liquidation_cascade_alpha_scan_2026-07-13.json`
- `results/hawkes_liquidation_cascade_replay_verification_2026-07-13.json`
