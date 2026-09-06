# CATDCR-8 train economic rejection

CATDCR-8 passed source support (`37 / 119 / 143 / 67`) with both sides present
in every split, then passed every Gross9 novelty comparison. Its worst
one-to-one ±6-hour matched share was `0.235294`, worst occupied-bar Jaccard was
`0.075898`, and worst absolute signed-exposure correlation was `0.071632`.

The unchanged candidate failed the first authorized economic stage:

| metric | 2023H2 train | gate |
|---|---:|---:|
| base absolute return | **-0.633907%** | > 0 |
| full-calendar CAGR / strict MDD | **-0.163626** | >= 3.0 |
| strict MDD | 7.666313% | <= 15% |
| mean gross underlying move | **8.836462 bp** | >= 20 bp |
| weekly cluster sign-flip p | **0.565844** | <= 0.10 |
| 10bp stress return | **-2.093620%** | > 0 |
| 10bp stress CAGR / strict MDD | **-0.508230** | >= 2.5 |
| first / second half return | +1.757475% / **-2.350080%** | both > 0 |

Only the strict-MDD ceiling passed. An immediate replay reproduced the result
byte-for-byte (`20d1f4f97020c351e12392e819cefddee6f6b6dcba43cb72ae8c66acdd923e99`).
Test, eval, final, and RV20 q90 outcomes remain unopened.

The turnover normalization, entropy definition, rank, breadth, variation,
side, hold, clock, universe, subset, and diagnostic controls remain frozen.
None may be repaired or promoted from this terminal candidate.
