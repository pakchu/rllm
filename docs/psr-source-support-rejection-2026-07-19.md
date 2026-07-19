# PSR-30/6 source-support rejection — 2026-07-19

## Verdict

**Reject before opening BTC outcomes.** No BTC execution price, return,
funding or PnL was read. The premium-only clock mechanism is distinct and has
ample aggregate support, but it missed one preregistered calendar-support gate.
That gate will not be relaxed after seeing the source-only counts.

## Source-only clocks

| split | total | long | short | max month share |
|---|---:|---:|---:|---:|
| train | 305 | 167 | 138 | 10.16% |
| test | 231 | 99 | 132 | 17.75% |
| eval | 611 | 330 | 281 | 7.36% |

Subperiod support:

- 2020 Feb-Dec: **15**, below the frozen minimum 20;
- 2021: 92, minimum 30;
- 2022: 198, minimum 30;
- 2023 H1/H2: 92 / 139, minimum 10 each;
- 2024 / 2025 / 2026 H1: 202 / 262 / 147, minimum 25 / 25 / 12.

All total-count, side-balance, month-concentration and later-subperiod gates
passed. The sole failure is 2020 partial-year breadth. The correct action under
the write-once contract is still rejection, not changing 20 to 15.

## Novelty

The source-only timing pattern passed every frozen novelty gate. The previously
selected PSI comparators were reconstructed under their original immediate
entry, 96-bar hold and global non-overlap contract.

| comparator | shared-span PSR clocks | exact Jaccard | PSR within 30m |
|---|---:|---:|---:|
| PSI-2016 | 1,147 | 0.00049 | 5.58% |
| PSI-8640 | 1,147 | 0.00054 | 5.32% |
| CLBR-24 | 327 | 0.00835 | 14.68% |
| ICLA-60 | 327 | 0.00524 | 9.79% |
| EBLR-60/30 | 327 | 0.00338 | 6.12% |

Frozen maximums were 0.10 exact Jaccard and 20% near-clock share.

## Frozen artifacts

- primary clocks: 1,147 rows,
  SHA-256 `cb209ed35f9baa08cc2fb3dd5bd60b8e747b1408c09507b774ca275e0b2b2db6`;
- support result SHA-256:
  `f33708368b089dd588051971b8d17b4174aaac304ead7a30b07ebb3ee3520b4f`;
- canonical manifest hash:
  `cd22c414d395dea4a45a63daf93888e4703560d0e5625e2d3ea64c172acc3fc8`;
- preregistration implementation SHA-256:
  `7f12921667e16711a3da188866367ca04ff7f588e82c95ad5b0068422855afd4`.

The PSI files above are novelty-only historical comparators. Their original
immediate entries precede the current source's `T+1s` availability and are
therefore explicitly forbidden from any BTC outcome/control evaluation.

Train, test and eval BTC outcomes remain unopened. A successor must introduce a
new pre-event mechanism; it may not repair PSR thresholds, direction, hold,
latency or the failed support gate.
