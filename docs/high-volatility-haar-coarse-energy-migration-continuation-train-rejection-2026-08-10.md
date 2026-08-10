# HVHCEM-12 train economic rejection

HVHCEM-12 passed source support (`34 / 91 / 71 / 28`) with both sides in every split, then passed every Gross9 novelty comparison. Worst exact-entry Jaccard was `0`, worst one-to-one ±6-hour matched share `0.216216`, worst occupied-bar Jaccard `0.070891`, and worst absolute signed-exposure correlation `0.069229`.

The unchanged candidate failed the first authorized economic stage:

| metric | 2023H2 train | gate |
|---|---:|---:|
| base absolute return | **-6.253308%** | > 0 |
| full-calendar CAGR / strict MDD | **-1.266959** | >= 3.0 |
| strict MDD | 9.495745% | <= 15% |
| mean gross underlying move | **-25.450146 bp** | >= 20 bp |
| weekly cluster sign-flip p | **0.950900** | <= 0.10 |
| 10bp stress return | **-7.522351%** | > 0 |
| 10bp stress CAGR / strict MDD | **-1.399669** | >= 2.5 |
| first / second half return | **-5.198377% / -1.112777%** | both > 0 |

Only the strict-MDD ceiling passed. Immediate replay was byte-identical (`b4d03c93d08aadeda8b114a1855c2d93718ef2e6bac1cea942f8b33a1bec6370`). Test, eval, final, and RV20 q90 outcomes remain unopened.

Haar scales, coarse levels, migration and variation ranks, side, hold, clock, path, subset, and diagnostic controls remain frozen. No control may be promoted and no parameter repaired.
