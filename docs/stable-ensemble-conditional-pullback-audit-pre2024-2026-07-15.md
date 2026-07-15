# Stable ensemble conditional-pullback audit — 2026-07-15

**Pre-OOS audit passed; 2024+ is still sealed until a separate freeze commit.**

Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.

## Stability verdict

- 1,000-tree individual seeds passing: **4/5**.
- 2,000-tree individual seeds passing: **3/5**.
- Three- and five-model ensembles passing at both sizes: **True**.
- One-hour-delay individual seeds passing: **5/5**.
- One-hour-delay five-model ensemble passing: **True**.
- Decision: **promote_to_freeze**.

## Exact selected model

| Window | Result |
|---|---:|
| train | 104.16% / 33.01% / 8.15% / 4.05 / 128 |
| select_2023 | 11.09% / 11.10% / 3.12% / 3.56 / 19 |
| select_2023_h1 | 8.90% / 18.78% / 3.12% / 6.02 / 13 |
| select_2023_h2 | 2.01% / 4.03% / 2.30% / 1.75 / 6 |
| pre_2024 | 126.80% / 26.35% / 8.15% / 3.23 / 147 |

## One-hour information-delay control

All predictive inputs except current funding/premium event identity were shifted 12×5m rows before refitting. The delayed model still passed:

| Window | Result |
|---|---:|
| train | 103.73% / 32.89% / 8.02% / 4.10 / 127 |
| select_2023 | 11.99% / 12.00% / 3.12% / 3.85 / 26 |
| select_2023_h2 | 1.23% / 2.46% / 2.88% / 0.85 / 9 |
| pre_2024 | 128.16% / 26.56% / 8.02% / 3.31 / 153 |

## Interaction ablations

| Rule | Pass | train | 2023 | pre-2024 |
|---|---:|---:|---:|---:|
| conditional | True | 104.16% / 33.01% / 8.15% / 4.05 / 128 | 11.09% / 11.10% / 3.12% / 3.56 / 19 | 126.80% / 26.35% / 8.15% / 3.23 / 147 |
| source_only | False | 114.23% / 35.59% / 8.15% / 4.37 / 134 | 10.06% / 10.07% / 4.24% / 2.38 / 25 | 135.78% / 27.76% / 8.15% / 3.41 / 159 |
| unconditional_pullback | False | 61.05% / 20.98% / 9.44% / 2.22 / 103 | 10.59% / 10.60% / 3.12% / 3.40 / 18 | 78.11% / 17.92% / 9.44% / 1.90 / 121 |
| width_only | False | 91.06% / 29.53% / 8.15% / 3.62 / 122 | 10.37% / 10.38% / 3.12% / 3.33 / 15 | 110.86% / 23.74% / 8.15% / 2.91 / 137 |
| reversed_pullback | False | 101.11% / 32.21% / 9.19% / 3.50 / 131 | 8.91% / 8.92% / 4.50% / 1.98 / 23 | 119.02% / 25.09% / 9.19% / 2.73 / 154 |

Only the conditional combination passes. Score-only admits weak quiet-range entries; pullback-only destroys train/pre-2024 breadth; width-only loses 2023 H2 count; reversed pullback loses 2023 risk-adjusted return.

Audit hash: `366a72719a1bc6c86012597548327a4446e8d852af21d8701d79bc4063ef35db`
