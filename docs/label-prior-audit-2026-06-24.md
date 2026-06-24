# Label prior audit — 2026-06-24

## Purpose

Recent Gemma4 SFT experiments repeatedly failed because label/token priors dominated model skill:

- pairwise `A/B`: position/token prior;
- ordinal `AVOID/LOW/MID/HIGH`: semantic label prior, all predictions collapsed to `AVOID`.

`training/audit_label_priors.py` measures label logprob priors on a fixed prompt set before training or strategy selection.

## Audit setup

Prompt set:

- `data/event_action_ordinal_utility_eval2026_bal256_2026-06-24.jsonl`
- first 64 balanced ordinal rows
- base model: `google/gemma-4-E4B-it`
- no adapter
- score key: mean logprob
- batch size: 1

## Results

### Semantic ordinal labels

Labels: `AVOID,LOW,MID,HIGH`

| label | token count | mean score |
| --- | ---: | ---: |
| AVOID | 2 | -9.214 |
| LOW | 1 | -10.328 |
| MID | 1 | -10.789 |
| HIGH | 1 | -14.191 |

- dominant label: `AVOID`
- mean score spread: `4.978`
- prediction counts: AVOID 33 / LOW 16 / MID 12 / HIGH 3

This explains the ordinal SFT collapse: the base model already strongly prefers `AVOID` over `HIGH`.

### Neutral X labels

Labels: `XA,XB,XC,XD`

| label | token count | mean score |
| --- | ---: | ---: |
| XA | 1 | -20.046 |
| XB | 1 | -22.918 |
| XC | 1 | -19.930 |
| XD | 1 | -19.296 |

- dominant label: `XD`
- mean score spread: `3.622`
- prediction counts: XA 7 / XC 21 / XD 36

These are not neutral enough.

### Neutral Q labels

Labels: `Q1,Q2,Q3,Q4`

| label | token count | mean score |
| --- | ---: | ---: |
| Q1 | 2 | -14.921 |
| Q2 | 2 | -14.607 |
| Q3 | 2 | -14.706 |
| Q4 | 2 | -14.950 |

- dominant label: `Q2`
- mean score spread: `0.344`
- prediction counts: Q1 1 / Q2 57 / Q3 6 / Q4 0

`Q1..Q4` still has a dominant token prior in argmax counts, but mean spread is an order of magnitude smaller than semantic labels. It is the best tested candidate for a neutral-code label experiment.

## Conclusion

Do not use semantically loaded labels for logprob classification without prior correction. `AVOID/HIGH` is unusable as-is.

Next viable LLM-first experiment:

1. remap ordinal targets to neutral code labels, e.g. `Q1..Q4`;
2. train on code labels only;
3. calibrate label offsets on train/calibration scores;
4. evaluate on held-out eval scores;
5. only then connect code scores to action selection.

This still may fail, but it attacks the measured label-prior failure directly.
