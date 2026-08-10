# HVCSER-12 train economic rejection

HVCSER-12 passed source support (`22 / 43 / 51 / 17`) with both sides present in every split, then passed every Gross9 novelty comparison. Its worst exact-entry Jaccard was `0`, worst one-to-one ±6-hour matched share was `0.292308`, worst occupied-bar Jaccard was `0.112255`, and worst absolute signed-exposure correlation was `0.133340`.

The unchanged candidate failed the first authorized economic stage:

| metric | 2023H2 train | gate |
|---|---:|---:|
| base absolute return | **-1.163968%** | > 0 |
| full-calendar CAGR / strict MDD | **-0.604435** | >= 3.0 |
| strict MDD | 3.800730% | <= 15% |
| mean gross underlying move | **1.352749 bp** | >= 20 bp |
| weekly cluster sign-flip p | **0.646934** | <= 0.10 |
| 10bp stress return | **-2.031071%** | > 0 |
| 10bp stress CAGR / strict MDD | **-0.926103** | >= 2.5 |
| first / second half return | +0.955300% / **-2.099214%** | both > 0 |

Only the strict-MDD ceiling passed. An immediate replay reproduced the result byte-for-byte (`a347853ab82cb9eb188f88f2cb59fa8fee71f0b40acf0cd5dd2c6ad92bc6635c`). Test, eval, final, and RV20 q90 outcomes remain unopened.

The Corwin-Schultz estimator, expansion and variation ranks, side, hold, 21:05 UTC clock, windows, subset, and diagnostic controls remain frozen. None may be repaired or promoted from this terminal candidate.
