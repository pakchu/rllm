# RMSR-72 source-support rejection — 2026-07-23

## Decision

**Reject `RMSR-72-SOURCE-REUSE` before novelty and outcomes. Do not repair this
identity and do not open its BTC or funding returns.**

The committed source-only builder reproduced byte-identical artifacts:

- report: `results/ofr_repo_mix_shock_resolution_race_support_2026-07-23.json`;
- report SHA-256:
  `d42b97bb85f75eba4cb45ea3487af27a44e8bc659a1ee07d73656d3ec5f23cf9`;
- report manifest:
  `019ebd8ee55689b44c1e027f82ca57c62c2b05385657d5fa4dbd2fc099d016cd`;
- source/control clock:
  `results/ofr_repo_mix_shock_resolution_race_clocks_2026-07-23.csv.gz`;
- clock SHA-256:
  `bed2f522bd68fd389cb0af3d655980999324c0e6adb4f1dc027c6b6ea2cc5ae6`.

## Source integrity

The run read 77,369 normalized observations and 9,976 required collateral rows
over 1,249 source dates. It formed 1,245 complete exact-rational feature dates;
four dates were invalid for missing or null values, none failed materiality,
and 417 equal-availability rows were suppressed from decision use. Rank history
for every equal-availability batch was computed strictly from pre-batch rows.

The parser initially treated the normalized `disclosure_edit` boolean text
`"0"` as truthy. That implementation defect produced zero valid dates, was
diagnosed against the committed source schema, fixed without changing any
candidate rule, tested, and committed before the result below was regenerated.
Only flag `"1"` now invalidates a required row.

## Source-support result

The fixed race produced adequate total incidence, both sides, complete quarter
coverage, and acceptable venue concentration, but failed three frozen gates:

| Split | Events | Long | Short | Confirmation | Absorption | Max gap |
|---|---:|---:|---:|---:|---:|---:|
| train 2021–2022 | 37 | 15 | 22 | 6 (16.2%) | 31 (83.8%) | **129 days** |
| selection 2023 | 33 | 26 | 7 | 3 (9.1%) | 30 (90.9%) | 31 days |

Frozen failures:

- maximum accepted-entry gap: 129 days versus 90 allowed;
- train confirmation share: 16.2% versus 20% required;
- selection confirmation share: 9.1% versus 15% required.

The unscheduled state machine saw 132 collateral-mix extreme transitions. It
discarded 50 as already priced, armed 82 races, resolved 67 by quantity
absorption and 13 by price confirmation, cancelled one same-date ambiguity and
one continuity break, and had no timeout. This is decisive source evidence
that the frozen first-passage object is not a balanced two-terminal mechanism.

The venue diagnostic itself passed: among non-tie accepted events, the largest
collateral-rate-spread venue share was 83.3% in train and 59.3% in selection,
both below the frozen 85% maximum.

## Closed boundaries and retained lesson

Source failure short-circuited all comparator access. The run read zero
comparator rows, BTC bars, funding rows, and future returns and opened no PnL,
CAGR, or MDD. RMSR therefore ends as a clean pre-outcome negative result.

The retained information is not an alpha claim: cross-venue collateral-mix
extremes usually normalize before same-polarity collateral-rate repricing.
That asymmetry is stable enough to motivate a genuinely different future
causal object, but RMSR cannot be converted into an absorption-only trade,
given a longer gap, or rescued by lowering terminal-share floors. Any successor
must use a new ID, preregister a distinct mechanism, and pass source support
before outcome access.
