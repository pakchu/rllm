# BFRT-288 support rejection — 2026-07-20

## Verdict

**Reject BFRT-288 permanently at the outcome-blind support gate.** Do not build
or run a market/funding evaluator for this singleton.

The preregistered rule required at least 20 accepted eval entries and at least
18 entries in 2026H1. The frozen clock produced 19 and 17 respectively. Every
other support check passed, but the stopping rule is conjunctive and permits no
near-miss waiver.

## Frozen identities

- preregistration manifest:
  `2d708231cbecaa5621c756f6cd9b7fbd259feb8baf32268464d68838487b9ebc`;
- policy hash:
  `06d2284866781a3c751857a6a049769cece2e64ea56e00d8e2180b5992825925`;
- support-builder commit: `75f41e6`;
- support-builder SHA-256:
  `de1ba0ac1424579b5869e8cd09986fca4383a95b3cfdffc0ef3694ecbd19ef1d`;
- support manifest hash:
  `20ef2c5e730f53aef48e79d3bbc640daa0e85036411a5563baa2df20f2aff7d5`;
- support manifest file SHA-256:
  `b980a0d76bd9a3084410d40e9fbf920acf1e2065bce174aa48823f41052f3bd8`;
- primary-clock SHA-256:
  `33428d29c2ace9b23672b2dc9dc3e9ba0e3020fa1a6e3845d55fa5d75230d64a`;
  and
- control-clock SHA-256:
  `d85197948f9418f4ab50e88825638fe33629f2db808fd1e572f8ed4d685c5a92`.

An independent replay recomputed the canonical support hash and all builder and
clock file hashes, reloaded both clock schemas, checked exact row counts and
split containment, and confirmed that the only failed checks were the two eval
support floors.

## Source-only incidence

The frozen source contained 2,191 rows. BFRT produced 1,384 valid source-feature
rows, 1,264 strict-prior rank-ready rows, and 224 accepted non-overlapping
primary entries.

| Window | Total | Long | Short | Maximum month share | Required total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 133 | 68 | 65 | 12.03% | 80 |
| Test | 72 | 32 | 40 | 12.50% | 35 |
| Eval | **19** | 8 | 11 | 21.05% | **20** |

Additional dispersion counts were:

- train 2023 Nov-Dec: 28; 2024H1: 55; 2024H2: 50;
- test 2025H1: 37; 2025H2: 35;
- test quarters: 16, 21, 25, and 10; and
- eval 2026H1: **17**, below the frozen minimum of **18**.

Train, test, both directions, all test quarters, all month-concentration limits,
and the zero-missing-bucket requirement passed. The failure is solely eval
sample support, not a performance result.

## Outcome boundary

The support run opened the frozen fee-rate source values and derived feature
and signal incidence, as permitted after preregistration. It loaded:

- zero BTC market rows or values;
- zero funding rows or values;
- zero premium or OI rows;
- zero return rows; and
- zero return, PnL, CAGR, or MDD fields.

No profitability statistic exists for BFRT-288.

## No repair

Although each failed floor missed by exactly one event, lowering either floor
after observing incidence would be post-registration selection. The frozen
stopping rule also forbids changing sign, magnitude/coherence/tail thresholds,
rank history, hold time, latency, non-overlap, or calendar boundaries.

BFRT-288 therefore ends here. Its result does not authorize a threshold-relaxed
BFRT-289, a side flip, or reuse of its sealed test/eval outcomes. A subsequent
candidate must use a genuinely distinct preregistered mechanism and begin again
at an outcome-blind source/support gate.
