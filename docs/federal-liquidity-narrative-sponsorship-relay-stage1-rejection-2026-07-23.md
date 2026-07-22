# FLNSR-2016 Stage1 rejection — 2026-07-23

## Decision

**Terminal reject at 2020–2022 train.** The 2023 selection window remains
physically sealed. The preregistered no-repair rule forbids reversing the side,
retuning the liquidity tails, changing the narrative window, or promoting a
control after seeing this result.

## Audit identity

- Evaluator-freeze manifest: `09dade9c6e5198465a8480d8559c31f703d5517d9f2b0a58a1c6a87e8c427f50`
- Stage1 manifest: `ea26f41c20f21b633e088ae77f595bf974da9ffb60e2ee506195ec489f3702a6`
- Stage1 file SHA256: `efbc4eb91d5662d082d6f35a8cca14a366d96b8ba9f0f4994328024d36e4ef0d`
- Market prefix SHA256: `744ac1ad59e53c088e1b6697ecaa073b2cd12cec5823957ac6ffaf2feab896bd`
- Funding prefix SHA256: `9a211053a26eb6b3dd0f00a32cb43f2706cea2ca876ed42a936a669039ddff0b`
- Parsed 2023 market rows: **0**
- Parsed 2023 funding rows: **0**

## Primary result

| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | Mean gross |
|---|---:|---:|---:|---:|---:|---:|
| 2020–2022 | -14.10% | -4.94% | 51.54% | -0.10 | 67 | -2.56 bp |
| 2020 | 12.45% | 12.42% | 25.09% | 0.50 | 25 | 136.54 bp |
| 2021 | -11.70% | -11.71% | 39.44% | -0.30 | 23 | -52.68 bp |
| 2022 | 4.78% | 4.78% | 14.33% | 0.33 | 18 | 74.52 bp |

- Monthly-cluster sign-flip p-value: **0.5713** across 30 months.
- 10 bp/notional/side stress return: **-16.41%**.
- One-extra-bar-delay return: **-16.66%**.
- Every frozen economic gate failed.

## Mechanism diagnosis

The macro/narrative conjunction did not isolate a profitable subset:

| Clock | Absolute return | strict MDD | CAGR/MDD | Trades | Mean gross | Monthly p |
|---|---:|---:|---:|---:|---:|---:|
| primary agreement | -14.10% | 51.54% | -0.10 | 67 | -2.56 bp | 0.5713 |
| liquidity only | 48.40% | 43.76% | 0.32 | 119 | 107.00 bp | 0.1568 |
| narrative only | -56.75% | 69.32% | -0.35 | 144 | -79.18 bp | 0.8678 |
| disagreement | 37.08% | 23.26% | 0.48 | 54 | 155.69 bp | 0.1542 |
| exact side flip | -9.25% | 43.11% | -0.07 | 67 | 2.56 bp | 0.5178 |

Primary mean gross return was 109.57 bp below `liquidity_only` and 158.25 bp
below `disagreement`. The positive controls still had weak CAGR/MDD and failed
significance, so neither is promoted. The exact side flip also lost after costs
and funding, ruling out a defensible simple inversion.

The useful conclusion is negative but specific: weekly H.4.1 liquidity tails
contain some gross directional structure, while the frozen GDELT quality-sign
agreement removes rather than sponsors it. The next candidate must use a new
economic clock/geometry rather than another FLNSR threshold repair.
