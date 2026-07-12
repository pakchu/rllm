# Jump + volume-clock bidirectional alpha (2026-07-12)

## Mechanism
Base: fixed 72-bar realized-jump continuation alpha. Gate: direction-confirming signed-flow speed measured over the amount of volume equal to 25% of the prior completed 24h quote volume.

The volume target is `rolling_24h_quote_volume.shift(1) * 0.25`; cumulative quote volume is searched backward only. No future bars are used.

Long gate: `vc_flow_speed_0p25 >= 0.0003536573`.
Short gate: `vc_flow_speed_0p25 <= -0.0004875875`.
Base TP1.5%, SL1%, cap96, stride6, 0.5x and 6bp/side remain fixed.

| split | return | CAGR | strict MDD | ratio | L/S | win L/S | Sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -11.85% | -3.10% | 16.09% | -0.19 | 152/158 | 42.8/38.0% | -1.28 |
| test2024 | 6.90% | 6.89% | 2.19% | **3.15** | 25/13 | 68.0/53.8% | 2.00 |
| eval2025 | 3.79% | 3.79% | 0.81% | **4.70** | 20/4 | 60.0/75.0% | 1.60 |
| ytd2026 | 0.24% | 0.58% | 2.98% | 0.19 | 16/13 | 43.8/61.5% | 0.10 |

## Integrity
- Train-fit volume thresholds and shifted trailing-volume target.
- Gate ranked on test2024 only; eval/2026 attached afterward.
- Exact second execution matched all metrics.
- Independent validator passed cost/split/trade/future-shift checks.

## Verdict
Mechanical live-grade test/eval candidate, but not operationally live-grade because train is negative and 2026 edge is negligible. Candidate alpha only.
