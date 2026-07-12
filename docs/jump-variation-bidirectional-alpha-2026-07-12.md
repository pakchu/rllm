# Jump-variation bidirectional alpha (2026-07-12)

## Mechanism
Past-only realized variance is decomposed with bipower variation. `jump_ratio=(RV-BV)+/RV`; signed cubic variation identifies jump direction. Taker-flow acceleration confirms continuation.

Long:
- 72-bar jump ratio >= 0.3173447786
- signed jump >= 0.1509897808
- taker-flow acceleration >= 0.0251275091

Short:
- same jump-ratio threshold
- signed jump <= -0.1495008468
- taker-flow acceleration <= -0.0252866649

Execution: TP1.5%, SL1%, cap96 bars, stride6, 0.5x, 6bp/side.

| split | return | CAGR | strict MDD | ratio | L/S | win L/S | Sharpe-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | -15.36% | -4.08% | 19.96% | -0.20 | 209/228 | 42.6/38.6% | -1.42 |
| test2024 | 7.05% | 7.03% | 2.77% | **2.54** | 34/18 | 61.8/50.0% | 1.75 |
| eval2025 | 3.88% | 3.89% | 1.26% | **3.09** | 28/8 | 53.6/62.5% | 1.29 |
| ytd2026 | 0.44% | 1.07% | 2.34% | 0.45 | 25/18 | 40.0/61.1% | 0.15 |

## Integrity evidence
- Thresholds fit on train only; rows ranked solely by test2024.
- Eval2025/YTD2026 attached only after top selection.
- No negative shift or centered rolling window.
- Full-window CAGR, strict intrabar MDD, next-open execution and split-contained exits.
- Exact second execution reproduced all conditions and metrics.
- Independent mechanical critic passed: `results/jump_variation_bidirectional_alpha_validator_2026-07-12.json`.

## Verdict
Qualifies for alpha_pool (test/eval ratio>=2.5, sufficient two-sided trades). It is not live-grade: train is negative and 2026 edge is weak. Candidate research alpha only.

## Artifacts
- `training/search_jump_variation_bidirectional_alpha.py`
- `results/jump_variation_bidirectional_alpha_scan_2026-07-12.json`
- `results/jump_variation_bidirectional_alpha_scan_repro_2026-07-12.json`
- `training/validate_alpha_research_candidate.py`
- `results/jump_variation_bidirectional_alpha_validator_2026-07-12.json`
