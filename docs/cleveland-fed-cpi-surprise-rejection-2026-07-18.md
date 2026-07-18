# CFCS-1 rejection — 2026-07-18

## Decision

**REJECT_KEEP_2023_SEALED**

CFCS-1 produced a small positive immediate-release effect, but failed the
precommitted economic-efficiency, statistical-significance, and mechanism
specificity gates.  Its formula, direction, threshold, and clock will not be
repaired after observing these outcomes.

## Sealed Stage1 result

| Cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long / Short | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 bp/notional/side | 6.5035% | 2.1220% | 4.3509% | 0.4877 | 26 | 10 / 16 | 61.3281 | 0.3055 |
| 10 bp/notional/side | 5.4047% | 1.7696% | 4.4468% | 0.3980 | 26 | 10 / 16 | 61.3281 | 0.3904 |

Full-calendar CAGR counts the complete 2020–2022 interval, including idle
cash. Strict MDD includes pre-entry/global HWM and intratrade adverse 5-minute
OHLC under the frozen strict engine.

## Year containment

| Year | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp |
|---|---:|---:|---:|---:|---:|---:|
| 2020 | 0.7592% | 0.7577% | 2.2186% | 0.3415 | 9 | 29.2208 |
| 2021 | 3.7401% | 3.7428% | 2.5422% | 1.4722 | 8 | 103.0602 |
| 2022 | 1.8901% | 1.8914% | 4.3509% | 0.4347 | 9 | 56.3402 |

Every year was positive and all density/cost/MDD gates passed. The result was
nevertheless far below the required CAGR/strict-MDD ratio of 3 and failed the
weekly clustered sign-flip threshold of 0.10.

## Mechanism and placebo evidence

| Clock | Absolute return | CAGR/MDD | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|
| Primary concordant composite | 6.5035% | 0.4877 | 61.3281 | 0.3055 |
| Headline only | 5.6083% | 0.4218 | 51.9540 | 0.3988 |
| Core only | 8.6408% | 0.6436 | 68.3701 | 0.2174 |
| Composite without concordance | 6.2102% | 0.4661 | 57.4254 | 0.3279 |
| Direction flip | -9.3295% | -0.2923 | -61.3281 | 0.1275 |
| One-calendar-day delay | -3.2315% | -0.1844 | -11.9281 | 0.5837 |
| Seven-day placebo | 3.9161% | 0.0922 | 43.0545 | 0.3930 |

The direction flip and next-day loss are consistent with a short-lived
release response, but that is insufficient: core-only exceeded the registered
composite, so the claimed concordance mechanism did not clear its +0.25 ratio
margin. Controls are diagnostic only and cannot replace the failed singleton.

## Physical seal evidence

- market rows parsed: 315,648, ending `2022-12-31T23:55:00Z`;
- funding rows parsed: 3,288, ending `2022-12-31T16:00:00Z`;
- both parsers stopped before the 2023 boundary;
- Stage2 invocation fails before its loader with
  `CFCS-1 Stage1 failed; 2023 remains sealed`;
- no Stage2 artifact exists;
- evaluator SHA-256 remains
  `92aba5e648ee4a0ac7119d37a271edd86df99f4177b8a533d4338d0e88bb5ff2`;
- Stage1 result manifest:
  `bc47514ad06ad3e4a422d078a7436f13077c3634fe3234fc9b4ba04c416a08d6`.

Portfolio orthogonality was deliberately not inspected because standalone
performance did not pass.
