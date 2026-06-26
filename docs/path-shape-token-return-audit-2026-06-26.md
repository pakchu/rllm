# Path-shape token return audit (2026-06-26)

## Purpose

PA+micro token TTE improved damage but remained negative. This diagnostic joins the selected OOS executed trades back to prompt tokens to identify which token groups or individual tokens are conditionally associated with gains/losses.

Implementation:

- `training/path_shape_token_return_audit.py`
- `tests/test_path_shape_token_return_audit.py`

Artifact:

- `results/path_shape_token_policy_tte_h144_t1p0_s0p6_pa_micro_aug_mc3_tk24/token_return_audit.json`

Config audited:

- train: `economic_path_shape_trader_sft_h144_t1p0_s0p6_train_pa_micro_aug.jsonl`
- eval: `economic_path_shape_trader_sft_h144_t1p0_s0p6_oos_pa_micro_aug.jsonl`
- executed: `pa_micro_aug_mc3_tk24/selected_eval.bt.json`
- token model: `min_count=3`, `top_k_tokens=24`, normal side mode
- matched trades: `317`

## Diagnostic findings

Worst OOS tokens, diagnostic only:

| Token | Trades | Mean ret | Win rate |
| --- | ---: | ---: | ---: |
| `aug.pa.w576.return=>=5pct` | 15 | -0.380% | 20.0% |
| `recent=UP|NORMAL|SURGE|BALANCED` | 16 | -0.347% | 25.0% |
| `aug.micro.w12.return=-2..-0.75pct` | 22 | -0.345% | 22.7% |
| `augnum.micro.w72.return_pct=<=-3` | 18 | -0.344% | 22.2% |
| `recent=FLAT|TIGHT|ACTIVE|LOWER_REJECT` | 18 | -0.310% | 22.2% |

Best OOS tokens, diagnostic only:

| Token | Trades | Mean ret | Win rate |
| --- | ---: | ---: | ---: |
| `augnum.micro.w36.return_pct=1.5..3` | 14 | 0.388% | 71.4% |
| `augnum.pa.w36.return_pct=1.5..3` | 14 | 0.388% | 71.4% |
| `augnum.pa.w2016.return_pct=0.5..1.5` | 14 | 0.315% | 64.3% |
| `aug.pa.w576.to_high=NEAR` | 41 | 0.315% | 65.9% |
| `sym.Volume State=QUIET` | 35 | 0.259% | 60.0% |

## Interpretation

This is not deployment evidence because it looks at OOS realized returns. It is a diagnostic pointer:

- Strong recent down-move tokens and certain surge/rejection states are associated with losses.
- Moderate positive recent return and quiet-volume tokens are associated with gains.
- This supports a next train/val/eval test: select token veto/allow rules on val only, then apply unchanged to OOS.

Next unit:

- Build a val-selected bad-token veto overlay for the path-shape token policy.
- Selection must use val executed returns only.
- OOS must remain untouched until final evaluation.
