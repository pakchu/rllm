# Path-shape analyzer/trader SFT data (2026-06-06)

## Outputs generated locally

Ignored data artifacts:

- `data/economic_path_shape_analyzer_sft_h144_t1p0_s0p6_{train,val,oos}.jsonl`
- `data/economic_path_shape_trader_sft_h144_t1p0_s0p6_{train,val,oos}.jsonl`

## Row counts

| split | analyzer rows | trader rows | period |
| --- | ---: | ---: | --- |
| train | 2370 | 2370 | 2023-01-01 to 2025-02-28 |
| val | 552 | 552 | 2025-03-01 to 2025-08-31 |
| OOS | 535 | 535 | 2025-09-01 to 2026-02-26 |

## Target distribution

| split | LONG_FAVORED | SHORT_FAVORED | NO_TRADE_FAVORED |
| --- | ---: | ---: | ---: |
| train | 749 | 749 | 872 |
| val | 162 | 163 | 227 |
| OOS | 151 | 176 | 208 |

## Prompt sizes

| split | analyzer prompt mean chars | trader prompt mean chars |
| --- | ---: | ---: |
| train | 2197 | 2775 |
| val | 2195 | 2773 |
| OOS | 2195 | 2773 |

## Training intent

- Analyzer SFT: past-only summary → path-shape JSON.
- Trader SFT: past-only summary + analyzer path-shape output → stop/target template action.

This is intentionally closer to LLM strengths: structured text classification/explanation of risk shape, not raw numeric action lottery selection.
