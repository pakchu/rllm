# GOICR-12 train rejection

GOICR-12 passed source support (`44 / 36 / 70 / 47`) and every Gross9 novelty
limit. Its worst ±6-hour matched share was `0.125`; worst occupied-bar Jaccard
was `0.0630`; worst absolute signed-exposure correlation was `0.0453`.

The unchanged candidate then failed the first economic stage:

| metric | train result | gate |
|---|---:|---:|
| base absolute return | **-8.9585%** | > 0 |
| full-calendar CAGR / strict MDD | **-1.5143** | >= 3.0 |
| strict MDD | 11.2253% | <= 15% |
| mean gross underlying move | **-30.2139 bp** | >= 20 bp |
| weekly cluster sign-flip p | **0.99384** | <= 0.10 |
| 10bp stress return | **-10.5494%** | > 0 |
| 10bp stress CAGR / strict MDD | **-1.5775** | >= 2.5 |
| first / second half return | **-0.5896% / -8.4185%** | both > 0 |

Only the strict-MDD ceiling passed. Test, eval, final, and RV20 q90 outcomes
remain unopened. No OI window, churn/cancellation statistic, threshold, onset,
side, clock, hold, subset, comparator, or diagnostic control is repaired or
promoted.
