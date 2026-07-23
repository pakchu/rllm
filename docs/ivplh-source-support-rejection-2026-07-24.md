# IVPLH-72 source-support rejection

## Verdict

**Retire IVPLH-72 unchanged before comparator novelty or economic outcomes.**

The sealed candidate exactly reproduces its 66-row IVFHR `any_handoff`
lineage, including the frozen one-bar execution delay, but fails two
prospective source-support gates:

1. selection side support; and
2. maximum split month share.

No comparator row, post-entry price, funding row, future return, PnL, CAGR, or
strict-MDD value was decoded.

## Frozen execution

- preregistration artifact commit: `a6604c9`;
- source-support evaluator commit: `bb4eeee`;
- evaluator tests before the real run: 44 related tests passed;
- runtime: 2.33 seconds;
- peak RSS: 186,804 KiB.

Artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `data/intrinsic_volume_price_lag_handoff_clocks_2020_2023.csv.gz` | `2efca3b44b0512a9423da90171f43babcadec2316dc6148796f3e61f98138e80` |
| `results/intrinsic_volume_price_lag_handoff_support_2026-07-24.json` | `ef0b187c4de29c27583bfe7bef85c7a55db95eb193954fe587cf5cce23a17103` |

The report manifest hash is
`77ed74b30e7941eb5bfe47671f7d9e679ff56c70afea94c56abdec34bf4c8ba3`.
A clean rebuild reproduced both artifacts byte for byte.

## Identity and support

All predecessor checks passed:

- 66 rows exactly;
- `(source_day, side, decision_time)` exactly matches predecessor
  `(source_day, side, entry_time)`;
- candidate entry is predecessor entry plus five minutes; and
- candidate exit is predecessor exit plus five minutes.

| Split | Events | LONG | SHORT | LONG share | Max month | Max quarter | Max gap | Max side run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train 2021–2022 | 33 | 20 | 13 | 60.61% | 9.09% | 21.21% | 81.27 d | 6 |
| Selection 2023 | 18 | 3 | 15 | 16.67% | 22.22% | 33.33% | 58.06 d | 7 |

Selection required at least 20% per side. Its three LONG events satisfy the
absolute count floor but only represent `3/18 = 16.67%`. March and October
each contain four entries, so the maximum month share is
`4/18 = 22.22%`, above the frozen 20% ceiling.

Every remaining identity, timing, count, half-year, concentration, gap,
same-side-run, non-overlap, schema, and permutation-selectivity check passed.
Both SHA year-permutation controls stayed below the frozen Jaccard and
same-side-reproduction ceilings in train and selection.

## Outcome boundary

The terminal report records:

- comparator rows decoded: `0`;
- post-entry price rows decoded: `0`;
- funding rows decoded: `0`;
- future-return rows decoded: `0`;
- return/PnL fields decoded: `0`;
- PnL/CAGR/MDD values decoded: `0`; and
- network calls: `0`.

Therefore IVPLH-72 has no profitability result. Relaxing the side or month
gate, changing the anchor, or adding an outcome-aware filter under the same
identity is prohibited.

## Process disclosure

During independent code review, a reviewer executed a source-only dry run with
the commit guard bypassed before the final evaluator commit, despite being
assigned review-only work. That run opened no comparator or economic data and
reported the same two source failures. The only subsequent change corrected
clock text serialization (`source_day` as `YYYY-MM-DD`) and could not affect
candidate incidence or gate statistics. The committed evaluator then
reproduced the source verdict exactly.

This incident prevents presenting the support run as a perfectly ordered
clean-room execution. It does not authorize repair: because the candidate is
source-seen and terminally fails the frozen gates, IVPLH-72 is retired rather
than tuned. Any continuation must use a new mechanism identity and a newly
sealed preregistration.
