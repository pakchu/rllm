# Frozen ExtraTrees top-10 OOS evaluation

Manifest: `c6e7d78a328118456eacf70bc42cb12a48f33e26d13edbe21f2edb3aedea4f8e`

- Future-pass candidates: `1/10`
- Full-window-pass candidates: `1/10`
- Frozen rank-1 future pass: `False`

| Frozen rank | 2025 abs/CAGR/MDD/ratio/trades | 2026H1 abs/CAGR/MDD/ratio/trades | Future ratio/trades | All abs/CAGR/MDD/ratio/trades | Future/full pass |
| ---: | --- | --- | --- | --- | --- |
| 1 | 10.00%/10.00%/5.22%/1.92/21 | 7.52%/19.03%/5.31%/3.58/16 | 2.09/37 | 55.56%/13.80%/6.01%/2.30/80 | False/False |
| 2 | 14.88%/14.90%/4.98%/2.99/19 | 5.48%/13.69%/5.31%/2.58/17 | 2.74/36 | 62.29%/15.23%/5.31%/2.87/78 | False/False |
| 3 | 14.88%/14.89%/4.98%/2.99/19 | 7.32%/18.51%/4.30%/4.30/14 | 3.20/33 | 63.37%/15.45%/4.98%/3.10/74 | False/False |
| 4 | 12.70%/12.71%/4.98%/2.55/18 | 7.32%/18.51%/4.30%/4.30/14 | 2.89/32 | 60.17%/14.78%/4.98%/2.97/72 | False/False |
| 5 | 14.15%/14.16%/4.98%/2.84/20 | 7.31%/18.48%/4.30%/4.30/12 | 3.09/32 | 61.79%/15.12%/4.98%/3.03/71 | False/False |
| 6 | 13.93%/13.94%/5.22%/2.67/22 | 5.48%/13.69%/5.31%/2.58/17 | 2.31/39 | 58.26%/14.38%/6.01%/2.39/81 | False/False |
| 7 | 16.36%/16.37%/4.98%/3.29/21 | 7.31%/18.48%/4.30%/4.30/12 | 3.41/33 | 64.04%/15.59%/4.98%/3.13/74 | True/True |
| 8 | 13.93%/13.94%/5.22%/2.67/22 | 7.32%/18.51%/4.30%/4.30/14 | 2.93/36 | 59.30%/14.60%/5.22%/2.80/77 | False/False |
| 9 | 11.77%/11.78%/5.22%/2.26/21 | 7.32%/18.51%/4.30%/4.30/14 | 2.63/35 | 56.18%/13.94%/5.22%/2.67/75 | False/False |
| 10 | 11.77%/11.78%/5.22%/2.26/21 | 7.32%/18.51%/4.30%/4.30/14 | 2.63/35 | 56.50%/14.01%/5.22%/2.68/76 | False/False |

## Integrity

- The pre-2025 feature, source-leg, activation, schedule, and metric prefixes reproduced exactly.
- Candidate order is the frozen 2023/2024 order; 2025/2026 did not rerank it.
- Annual expanding refits use only labels whose source-owned exits precede each cutoff.
- Prediction is deterministic (`n_jobs=1`); execution retains next-open, exact costs/funding, non-overlap, split containment, and strict MDD.

## Limitation

The evaluator is algorithmically isolated by a committed manifest, but this remains a retrospective reconstruction because earlier human research had viewed the future periods.
