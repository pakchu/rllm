# MXRBR-12 terminal train rejection

MXRBR-12 passed source support and every frozen Gross9 novelty comparison, but
failed the first economic stage and is rejected unchanged. No 2024, 2025, or
2026 economic outcomes were opened.

## Passed gates

- Source events: 25/117/64/58 in train/test/eval/final.
- Every split passed direction balance and monthly concentration.
- Exact-entry Jaccard was zero against every Gross9 sleeve.
- Six-hour matched share ranged from 0.156 to 0.297, below 0.35.
- Occupied-bar Jaccard was at most 0.104 and absolute exposure correlation at
  most 0.066.

## Train economics

- 25 trades, 18 long and 7 short.
- Base return: **4.61%**; full-calendar CAGR: **9.37%**.
- Strict MDD: **5.50%**; CAGR/MDD: **1.70**, below the required 3.0.
- Mean gross underlying move: **49.48 bp**.
- Weekly sign-flip p-value: **0.153**, above the permitted 0.10.
- Stress return: **3.57%**, but stress CAGR/MDD was **1.31**, below 2.5.
- The first calendar half returned **-1.66%**; the second returned **6.39%**.

The candidate therefore failed base and stress risk-adjusted-return gates, the
cluster significance gate, and the both-halves-positive gate. Positive return,
gross edge, stress return, and low drawdown are insufficient to advance it. No
direction flip, threshold, hold, timing, volatility, or subset repair is
authorized; diagnostic controls remain non-promotable.
