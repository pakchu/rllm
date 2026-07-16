# XFA-1 cross-sectional fragmentation absorption — pre-2026

## Decision

- Development status: **reject_before_2026**.
- 2026 post-entry outcomes were not read.
- Eight disclosed mechanism variants were evaluated on research-seen 2023-2025 only.
- XFA trades one idiosyncratic alt against ETH with causal factor-beta-neutral weights; it has no BTC leg.

## Ranked development policies

| Rank | Policy | Parameters | 2023 | 2024 | 2025 | 2024-25 combined | Pass |
|---:|---|---|---:|---:|---:|---:|:---:|
| 1 | XFA02 | flow>=2.0, resid<=0.5, size<=-0.5, hold=6h | -17.69/-17.70/20.99/-0.84/241 | -26.94/-26.89/29.16/-0.92/275 | -24.32/-24.34/27.58/-0.88/264 | -44.71/-25.63/45.87/-0.56/539 | FAIL |
| 2 | XFA04 | flow>=2.5, resid<=0.75, size<=-0.5, hold=6h | -13.60/-13.61/16.94/-0.80/135 | -12.76/-12.74/13.34/-0.96/148 | -7.95/-7.96/11.31/-0.70/118 | -19.70/-10.38/20.50/-0.51/266 | FAIL |
| 3 | XFA03 | flow>=2.5, resid<=0.75, size<=-0.5, hold=3h | -13.85/-13.86/16.48/-0.84/157 | -19.46/-19.43/20.01/-0.97/163 | -11.04/-11.05/12.20/-0.91/125 | -28.35/-15.35/28.70/-0.53/288 | FAIL |
| 4 | XFA08 | flow>=1.75, resid<=0.5, size<=-0.75, hold=6h | -28.27/-28.28/29.62/-0.95/319 | -19.60/-19.56/22.40/-0.87/337 | -35.98/-36.00/36.87/-0.98/288 | -48.52/-28.24/49.75/-0.57/625 | FAIL |
| 5 | XFA06 | flow>=2.0, resid<=0.35, size<=-0.25, hold=6h | -11.58/-11.59/14.61/-0.79/220 | -33.12/-33.06/34.55/-0.96/260 | -24.08/-24.09/24.63/-0.98/250 | -49.22/-28.72/50.29/-0.57/510 | FAIL |
| 6 | XFA01 | flow>=2.0, resid<=0.5, size<=-0.5, hold=3h | -27.70/-27.71/28.03/-0.99/291 | -28.85/-28.80/30.72/-0.94/330 | -32.52/-32.54/32.93/-0.99/306 | -51.99/-30.69/53.13/-0.58/636 | FAIL |
| 7 | XFA07 | flow>=1.75, resid<=0.5, size<=-0.75, hold=3h | -33.32/-33.34/33.99/-0.98/383 | -30.43/-30.38/32.09/-0.95/417 | -33.69/-33.71/34.04/-0.99/340 | -53.87/-32.06/54.73/-0.59/757 | FAIL |
| 8 | XFA05 | flow>=2.0, resid<=0.35, size<=-0.25, hold=3h | -24.75/-24.76/24.90/-0.99/266 | -33.22/-33.17/34.79/-0.95/307 | -33.91/-33.92/34.28/-0.99/300 | -55.86/-33.55/56.89/-0.59/607 | FAIL |

Metric cells are absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades.

## Causal and accounting contract

- Features use exactly twelve completed 5-minute bars per UTC hour; entry is the following +5-minute open.
- Flow, average-trade-size and beta standardizers use strictly prior rolling history.
- A signal requires extreme taker flow, muted factor-adjusted price response and unusually small average trade size.
- Direction is opposite the aggressive flow; ETH is the factor hedge, sized to zero estimated factor beta.
- Strict MDD includes global/pre-entry HWM, funding, favorable-before-adverse held OHLC and hypothetical liquidation cost.
- Controls include 10 bp/side, +5m entry/exit, exact direction flip and the same rule without fragmentation.

A development pass would only authorize a separately frozen 2026 one-shot replay. It would not authorize live trading.
