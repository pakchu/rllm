# HVCASFR-8 source-support pass — 2026-08-12

The preregistered cross-alt synchronized half-block sign-flip relay passed every
outcome-blind source gate without changing its alt set, clock, flip count,
variation threshold, side, or hold.

| Split | Events | Long / short | Minority share | Max month share |
| --- | ---: | ---: | ---: | ---: |
| train | 75 | 36 / 39 | 0.4800 | 0.2533 |
| test | 174 | 88 / 86 | 0.4943 | 0.1552 |
| eval | 164 | 71 / 93 | 0.4329 | 0.2134 |
| final | 71 | 32 / 39 | 0.4507 | 0.3944 |

Required minimum event counts were 8/12/12/8, minimum minority-side share was
0.20, and maximum month share was 0.45. The source builder opened only completed
pre-entry OHLC rows. It did not open execution prices, funding values, Gross9
rows, post-entry returns, or PnL.

Reproduction reran the frozen evaluator and compared all 14 generated JSON and
deterministic-gzip artifacts. Every SHA-256 digest was byte-identical. The
candidate is authorized to advance only to the frozen Gross9 novelty gate.
