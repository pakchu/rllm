# Path-shape val-token-veto rolling check (2026-06-26)

## Purpose

The val-selected token veto overlay produced the first leakage-controlled positive OOS slice in this branch. This check asks whether that effect survives a shifted rolling split.

Dataset:

- Source rows: PA+micro augmented path-shape trader rows.
- Market alignment: `data/2023-01-01_2026-02-28_d2a88c0700504d6a5e15bc3839ad84b6.csv.gz`.
- Total rows: `3,457`, from `2023-01-01 02:55:00` to `2026-02-26 02:55:00`.

Protocol per split:

1. Fit token model on train only.
2. Mine bad tokens from val executed returns only.
3. Select veto/thresholds on val only.
4. Evaluate on untouched eval.

Config:

- `min_count=3`
- `top_k_tokens=24`
- `side_modes=normal`
- `min_token_trades=16`
- `max_veto_mean_ret_pct=-0.05`
- `veto_sizes=0,3,5,8,12,20`
- stop/take: `0.6% / 1.0%`

## Splits

| Split | Train | Val | Eval |
| --- | --- | --- | --- |
| r1 | 2023-01-01 to 2024-12-31 | 2025-01-01 to 2025-06-30 | 2025-07-01 to 2025-12-31 |
| r2 | 2023-01-01 to 2025-02-28 | 2025-03-01 to 2025-08-31 | 2025-09-01 to 2026-02-26 |

## Results

| Split | Selected veto | Selected p/m | Val CAGR | Val MDD | Val ratio | Val trades | Eval CAGR | Eval MDD | Eval ratio | Eval trades | Eval p |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | 20 | 0.65 / 0.00 | 45.59% | 3.61% | 12.64 | 69 | -17.10% | 9.71% | -1.76 | 49 | 0.060 |
| r2 | 20 | 0.34 / 0.30 | 31.50% | 3.88% | 8.13 | 72 | 19.50% | 7.82% | 2.49 | 79 | 0.184 |

Artifacts:

- `results/path_shape_val_token_veto_rolling_h144_t1p0_s0p6_pa_micro/r1/report.json`
- `results/path_shape_val_token_veto_rolling_h144_t1p0_s0p6_pa_micro/r2/report.json`

## Conclusion

The token-veto layer is a real improvement over raw token policy, but the edge is not yet stable:

- r2 remains positive and close to the CAGR/MDD target ratio.
- r1 overfits val badly and fails eval.
- Trade counts are still too small for high confidence.

Current status: **candidate direction, not deployable**.

Next work should not launch Gemma SFT yet. The correct next step is to make the abstention/veto layer more causal/stable:

1. Use rolling token statistics with decay instead of one val-mined veto set.
2. Restrict veto candidates to semantically stable families, avoiding highly specific recent-bar tokens.
3. Increase eval coverage beyond 6-month windows when more path-shape data is available.
