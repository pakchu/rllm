# Year-walkforward weak responsibility search — 2026-07-15

**Decision: reject**. 2024+ remains sealed.

Metric: absolute return / CAGR / strict MDD / CAGR-to-strict-MDD / trades.

- Each scored year uses only fully purged earlier labels; no held-out year trains an earlier year.
- Ridge is deterministic and weighted by year/source. The model only executes or abstains.
- Funding events keep 48h/TP4/no-stop; premium events keep 12h/no-TP/SL3.
- A candidate needs the strict absolute gate and one adjacent passing ridge/margin cell.
- Market/funding/premium/OI/spot-premium sources are physically truncated before 2024.

Passing cells: `0/24`; adjacent-stable candidates: `0/24`.

| Rank | Spec | Train | 2023 | 2023 H2 | Pre-2024 | Pass | Adjacent |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `{"form": "linear", "hazard_hours": 168, "margin": 0.0, "ridge": 100.0}` | 13.32% / 5.12% / 12.50% / 0.41 / 47 | 4.75% / 4.75% / 3.08% / 1.54 / 8 | -0.89% / -1.76% / 2.64% / -0.67 / 2 | 18.71% / 5.02% / 12.50% / 0.40 / 55 | False | 0 |
| 2 | `{"form": "linear", "hazard_hours": 336, "margin": 0.0, "ridge": 100.0}` | 12.91% / 4.97% / 12.50% / 0.40 / 47 | 2.76% / 2.76% / 2.94% / 0.94 / 7 | -0.89% / -1.76% / 2.64% / -0.67 / 2 | 16.02% / 4.33% / 12.50% / 0.35 / 54 | False | 0 |
| 3 | `{"form": "tensor", "hazard_hours": 336, "margin": 0.001, "ridge": 10.0}` | 14.17% / 5.44% / 10.67% / 0.51 / 78 | 1.10% / 1.11% / 3.72% / 0.30 / 5 | -1.05% / -2.08% / 2.64% / -0.79 / 1 | 15.43% / 4.18% / 10.67% / 0.39 / 83 | False | 0 |
| 4 | `{"form": "tensor", "hazard_hours": 168, "margin": 0.001, "ridge": 10.0}` | 13.57% / 5.21% / 11.06% / 0.47 / 73 | 1.10% / 1.11% / 3.72% / 0.30 / 5 | -1.05% / -2.08% / 2.64% / -0.79 / 1 | 14.82% / 4.03% / 11.06% / 0.36 / 78 | False | 0 |
| 5 | `{"form": "tensor", "hazard_hours": 168, "margin": 0.0, "ridge": 10.0}` | 10.72% / 4.15% / 13.58% / 0.31 / 79 | 1.61% / 1.61% / 3.72% / 0.43 / 7 | -0.62% / -1.22% / 2.64% / -0.46 / 2 | 12.50% / 3.42% / 13.58% / 0.25 / 86 | False | 0 |
| 6 | `{"form": "tensor", "hazard_hours": 336, "margin": 0.0, "ridge": 10.0}` | 7.88% / 3.08% / 13.11% / 0.23 / 84 | 1.78% / 1.78% / 3.72% / 0.48 / 8 | -0.45% / -0.90% / 2.64% / -0.34 / 3 | 9.80% / 2.70% / 13.11% / 0.21 / 92 | False | 0 |
| 7 | `{"form": "tensor", "hazard_hours": 168, "margin": 0.0, "ridge": 100.0}` | 4.18% / 1.65% / 9.80% / 0.17 / 58 | 3.24% / 3.24% / 3.72% / 0.87 / 7 | -0.89% / -1.76% / 2.64% / -0.67 / 2 | 7.56% / 2.10% / 9.80% / 0.21 / 65 | False | 0 |
| 8 | `{"form": "tensor", "hazard_hours": 336, "margin": 0.0, "ridge": 100.0}` | 3.11% / 1.23% / 9.80% / 0.13 / 59 | 2.25% / 2.25% / 4.64% / 0.49 / 8 | -1.83% / -3.61% / 3.87% / -0.93 / 3 | 5.43% / 1.52% / 9.80% / 0.16 / 67 | False | 0 |
| 9 | `{"form": "tensor", "hazard_hours": 336, "margin": 0.002, "ridge": 10.0}` | 17.05% / 6.49% / 9.33% / 0.70 / 73 | 0.39% / 0.39% / 3.42% / 0.11 / 4 | -1.05% / -2.08% / 2.64% / -0.79 / 1 | 17.50% / 4.71% / 9.33% / 0.51 / 77 | False | 0 |
| 10 | `{"form": "tensor", "hazard_hours": 168, "margin": 0.002, "ridge": 10.0}` | 14.60% / 5.60% / 9.94% / 0.56 / 67 | 0.39% / 0.39% / 3.42% / 0.11 / 4 | -1.05% / -2.08% / 2.64% / -0.79 / 1 | 15.04% / 4.08% / 9.94% / 0.41 / 71 | False | 0 |

Implementation hash: `bd3104c71f1bd897ba00104c577ccef0f7a438ce5fbc926aae14549b61433af3`
