# EMFC-864 source-only support rejection — 2026-07-20

## Verdict

**REJECT.** EMFC-864 passed source integrity, event count, side balance,
calendar dispersion, density, feature-level distinctiveness, and every
existing-network-alpha exposure-correlation gate. It failed the frozen
mechanism-shadow gate because its scheduled exposure remained too similar to
both adjacent pseudo-maturity clocks:

- exact 100 versus pseudo 99 blocks: `0.8685818497`;
- exact 100 versus pseudo 101 blocks: `0.9297110748`; and
- frozen absolute maximum: `0.80`.

The exact 100-block boundary therefore does not create a sufficiently distinct
tradable clock after onset detection and 72-hour non-overlap. The candidate is
rejected without changing lag, thresholds, direction, onset, latency, or hold.

No BTC price, OHLC, funding, premium, OI, liquidation, future return, PnL,
equity, CAGR, or MDD value was loaded or calculated. There is deliberately no
performance table for this candidate.

## Frozen artifacts

- support result:
  `results/exact_maturity_fee_cadence_polarity_support_2026-07-20.json`;
- support-result file SHA-256:
  `1cfd359de4412972ed133e56523d3c713c372cb307304ce3bcd42c338b9e045d`;
- canonical result hash:
  `b32ec03347bc4ef3a80fc293ca1b122101286bbbe76e67d6c29801b1ac932880`;
- combined source clock:
  `results/exact_maturity_fee_cadence_polarity_clocks_2026-07-20.csv`;
- combined-clock SHA-256:
  `31af41f42ffe4dc73f0ff35ccf278e38c856d224184e802e46b370650d35951d`;
- source-support evaluator SHA-256:
  `c58327d32432ac07a8072cbca371a32fb849ca083afefba2f1cdd7b42fc1df3d`;
- preregistration SHA-256:
  `43f1505786ad5ddd8a076afebccc26bff65387d8ef9b7a443035136606157ff6`.

Two complete successful source-only runs reproduced both result and clock
files byte-for-byte. The successful run took 15.30 seconds wall time and
406,416 KiB maximum RSS. Two earlier implementation attempts stopped before
writing any artifact: one duplicate-column serialization error and one
infeasible greedy matched-null sampler. Both were regression-tested and
committed before the successful run; neither opened or printed market outcome.

## Source integrity

| Check | Result |
|---|---:|
| frozen source rows | 213,095 |
| exact candidate heights | 212,989 |
| valid candidate heights | 212,989 |
| positive elapsed-span ratio | 100.00% |
| maximum invalid elapsed run | 0 |
| six-successor containment | pass |
| non-negative fees | pass |
| pre-2024 containment | pass |
| outcome-boundary counters | all zero |

All ranks used an incremental prior-only ordered window. No full-series value
scan entered normalization. Header-time availability waited through `h+6`, a
two-hour embargo, ceiling to a five-minute boundary, and one complete latency
bar.

## Event support

The primary clock contained 175 events over 2021-2023:

| Window | Total | Long | Short |
|---|---:|---:|---:|
| 2021-2022 train | 114 | 72 | 42 |
| 2021 | 37 | 26 | 11 |
| 2022 | 77 | 46 | 31 |
| 2023 selection | 61 | 26 | 35 |
| 2023 H1 | 31 | 7 | 24 |
| 2023 H2 | 30 | 19 | 11 |

- train maximum month share: `7.8947%`;
- selection maximum month share: `13.1148%`;
- median entry gap: `105.75 hours`;
- exact 72-hour boundary-gap share: `0%`; and
- every frozen event-support gate: pass.

This is a statistically usable source clock. Its rejection is not caused by
sparsity, an always-on scheduler, one-sidedness, or calendar concentration.

## Novelty evidence

Feature-level source comparisons passed:

| Comparison | Spearman |
|---|---:|
| matured fee vs same-height fee | +0.6272 |
| exact state vs pseudo-99 state | +0.5000 |
| exact state vs pseudo-101 state | +0.4983 |

Existing network-alpha signed five-minute exposure correlations were all below
the frozen `0.35` ceiling:

| Comparator | Correlation |
|---|---:|
| BATE-288 | -0.0468 |
| BFC-3 | +0.0103 |
| MCR-7 | -0.0326 |
| NTB-7 | -0.0635 |
| UFCP-1 | -0.1671 |

Other internal shadows also passed the `0.80` ceiling: fee-only `0.2007`,
cadence-only `0.3618`, same-height fee `0.6600`, completed-day aggregate
`0.2343`, and seven-day stale `0.1189`.

The contradiction is specific: individual extreme-state classifications were
not identical, yet after onset and non-overlap the 99/100/101-block policies
occupied almost the same signed exposure intervals. The claimed alpha is a
smooth neighboring-lag family, not an exact consensus-maturity discontinuity.

## Stopping decision

The preregistration explicitly rejects a candidate whose internal shadow
exposure correlation exceeds 0.80. EMFC-864 is therefore stopped before any
strict market evaluator or 2021-2023 outcome source is created. Testing the
99- or 101-block variant, widening the shadow threshold, changing the hold, or
dropping the exact-boundary claim would be a post-incidence repair and is not
authorized under this singleton.

