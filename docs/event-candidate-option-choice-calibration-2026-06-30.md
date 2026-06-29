# Option Choice Calibration Diagnostic — 2026-06-30

## Purpose

After fixing `training/eval_option_choice_logprob.py`, the corrected A/B/C predictions were tested with simple score offsets to see whether the current Gemma option-choice surface only needed bias calibration.

## Tool

`training/calibrate_option_choice_scores.py` scans relative offsets for B and C while fixing A at zero. It now supports `--validation-fraction` so offsets are selected on one split and evaluated on a held-out split. This is still diagnostic only; production model selection must use chronological train/test/eval splits, not random eval splits.

## Corrected eval256 holdout diagnostic

Input predictions:

- `base_eval256_corrected_predictions.jsonl`
- `sft_s64_eval256_corrected_predictions.jsonl`
- `sft_s256_eval256_corrected_predictions.jsonl`

Command shape:

```bash
.venv/bin/python -m training.calibrate_option_choice_scores \
  --predictions-jsonl <corrected_predictions.jsonl> \
  --output-json <calibration_holdout.json> \
  --offsets=-4,-3,-2,-1,0,1,2,3,4 \
  --objective balanced_accuracy \
  --validation-fraction 0.5 \
  --seed 7
```

## Result

| model | baseline validation acc | selected offsets | fit acc | validation acc | validation pred A/B/C |
| --- | ---: | --- | ---: | ---: | --- |
| base | 0.3750000 | B=-3, C=0 | 0.4296875 | 0.3671875 | 81 / 4 / 43 |
| sft_s64 | 0.3515625 | B=-4, C=0 | 0.4140625 | 0.3515625 | 76 / 0 / 52 |
| sft_s256 | 0.3359375 | B=-3, C=0 | 0.4296875 | 0.3671875 | 79 / 0 / 49 |

Reports:

- `results/event_candidate_option_choice_wavefull_ext_micro_c72_s2_2026-06-29/base_eval256_corrected_calibration_holdout.json`
- `results/event_candidate_option_choice_wavefull_ext_micro_c72_s2_2026-06-29/sft_s64_eval256_corrected_calibration_holdout.json`
- `results/event_candidate_option_choice_wavefull_ext_micro_c72_s2_2026-06-29/sft_s256_eval256_corrected_calibration_holdout.json`

## Interpretation

The offset search overfits the small calibration split and tends to suppress B heavily. This does not provide a reliable path to a tradable policy. The current supervised A/B/C target surface is not just miscalibrated; it is too noisy/ambiguous for the current SFT setup.

## Decision

Do not advance current s64/s256 A/B/C adapters to strict backtest. Move to a cleaner training objective/surface before spending more GPU time.
