# HVWMR-72 source-support pass

The preregistered and evaluator-frozen HVWMR-72 weekly clock passed every
minimum-count, side-balance, and month-concentration gate without opening
Gross9 rows, execution prices, funding, or post-entry outcomes.

| stage | events | long | short | minority share | max month share |
|---|---:|---:|---:|---:|---:|
| train | 8 | 6 | 2 | 0.2500 | 0.3750 |
| test | 27 | 16 | 11 | 0.4074 | 0.1481 |
| eval | 17 | 5 | 12 | 0.2941 | 0.2353 |
| final | 12 | 4 | 8 | 0.3333 | 0.3333 |

Two executions reproduced identical artifacts:

- source snapshot SHA-256: `4618b5357b1b24b9541b63b43ddea99d3b14578834f693ec8595953cd8c13b03`
- clock SHA-256: `c0c03b45c228edc8a7d8694ee9be862c84c08dc75f1ae9fe74b6a28af8570f13`
- result SHA-256: `c343727ff2e9f3d7e90a2c5c072f305a131f8ef14e67d0ca04ff6ad1a68e4cbf`

The unchanged singleton may advance only to the frozen Gross9 novelty gate.
