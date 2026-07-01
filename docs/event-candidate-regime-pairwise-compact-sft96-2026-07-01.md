# Compact Regime Pairwise Gemma SFT96 — 2026-07-01

## Setup

- Surface: compact regime-matched pairwise option rows
- Train: `data/event_candidate_regime_pairwise_option_compact_paext_rex_train_2022_2024_2026-06-30.jsonl`
- Eval: `data/event_candidate_regime_pairwise_option_compact_paext_rex_eval_2025_2026_2026-06-30.jsonl`
- Model: `google/gemma-4-E4B-it`
- Adapter: `checkpoints/event_candidate_regime_pairwise_compact_gemma4_sft_s96_2026-07-01`
- Samples: 4,096 balanced A/B
- LoRA: r=8, alpha=16, dropout=0.10
- LR: 5e-6
- Steps: 96
- Runtime: 1,665 sec (~27m45s)
- Train loss: 1.123

## Random1024 eval

Same eval sample as the base compact diagnostic: random seed 46.

| model | accuracy | pred A/B | target A acc | target B acc |
| --- | ---: | --- | ---: | ---: |
| base | 0.515625 | 598 / 426 | 0.5941 | 0.4274 |
| sft_s96 | 0.5078125 | 622 / 402 | 0.6089 | 0.3942 |

The adapter slightly increases A bias and reduces overall accuracy.

## Valid prediction backtest

Prediction rows preserve candidate metadata; sampled predictions are backtested directly.

| model | q | events | trades | CAGR | strict MDD | ratio | mean trade | p approx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.80 | 192 | 144 | 31.21 | 18.79 | 1.66 | 0.296 | 0.1569 |
| base | 0.85 | 149 | 117 | 11.58 | 18.64 | 0.62 | 0.157 | 0.4788 |
| base | 0.90 | 102 | 89 | 2.89 | 19.97 | 0.14 | 0.076 | 0.7824 |
| base | 0.95 | 52 | 48 | -0.72 | 17.56 | -0.04 | 0.002 | 0.9951 |
| sft_s96 | 0.80 | 197 | 151 | -24.74 | 46.24 | -0.54 | -0.226 | 0.2774 |
| sft_s96 | 0.85 | 151 | 119 | -3.09 | 32.78 | -0.09 | -0.002 | 0.9925 |
| sft_s96 | 0.90 | 101 | 84 | 5.09 | 18.85 | 0.27 | 0.112 | 0.6774 |
| sft_s96 | 0.95 | 52 | 48 | -10.34 | 26.27 | -0.39 | -0.275 | 0.4389 |

## Decision

Reject this SFT adapter. It learns the token task enough to reduce loss, but it worsens economic ranking. More steps on this SFT objective are unlikely to solve the core problem.

## Next direction

Use the base compact score as a weak signal and improve it with causal risk filters, score neutralization, or pairwise score diagnostics. Do not continue plain SFT on the same compact pairwise target without a new objective that optimizes economic ranking rather than label token likelihood.
