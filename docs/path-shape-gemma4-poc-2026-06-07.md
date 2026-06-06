# Gemma4 path-shape analyzer/trader POC (2026-06-07)

## Setup

- model alias: `gemma4-e4b` → `google/gemma-4-E4B-it`
- data: `economic_path_shape_*_h144_t1p0_s0p6`
- train sample: 512 balanced rows
- max steps: 16
- max seq length: 4096
- LoRA: r=16, alpha=32

## Dry-run

Both analyzer and trader SFT dry-runs succeeded on 256 balanced rows.

- analyzer prompt mean: ~2197 chars, target mean: ~626 chars
- trader prompt mean: ~2775 chars, target mean: ~92 chars

## Training results

| stage | checkpoint | runtime | train loss | notes |
| --- | --- | ---: | ---: | --- |
| analyzer | `checkpoints/path_shape_analyzer_gemma4_e4b_h144_t1p0_s0p6_step16` | 125.3s | 1.908 | learned weakly; generation collapsed |
| trader | `checkpoints/path_shape_trader_gemma4_e4b_h144_t1p0_s0p6_step16` | 247.3s | 0.6347 | learned action format/direction when oracle analyzer output is provided |

## Validation generation

Analyzer val32:

- direction_pressure accuracy: 40.6%
- exact pressure+grades: 3.1%
- collapse: predicted `NO_TRADE_FAVORED` and `NO_EDGE` almost everywhere

Trader val64, after repairing the common schema slip `gate=LONG/SHORT` → `gate=TRADE, side=LONG/SHORT`:

- gate accuracy: 70.3%
- side accuracy: 68.8%
- exact template accuracy: 46.9%
- failure mode: over-trading; NO_TRADE target was predicted correctly only 2/21 times

## Interpretation

The POC is useful but not profitable-ready:

1. Trader can use path-shape output if it is already supplied, so the trader stage is learnable.
2. Analyzer cannot yet generate the full long JSON path target from 16 steps; it collapses to safe/no-edge defaults.
3. The trader prompt schema was ambiguous enough that Gemma often wrote `gate: LONG/SHORT`. The SFT prompt and evaluator now repair/document this.

## Next move

Do not scale the full analyzer JSON target yet. First make a pressure-only analyzer target (`direction_pressure` only), verify it beats majority on val/OOS, then reintroduce grades/path numbers incrementally.
