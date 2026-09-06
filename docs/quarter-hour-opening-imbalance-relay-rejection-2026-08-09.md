# QHOIR-8 terminal novelty rejection

QHOIR-8 passed its frozen source-support gate but failed its frozen Gross9
structural-novelty gate. It is rejected unchanged and cannot advance to economic
evaluation.

## Frozen evidence sequence

- Preregistration was committed before candidate incidence was opened.
- Source support passed with 234/455/434/201 train/test/eval/final events.
- The source-support artifact kept post-entry returns, execution prices, funding,
  and Gross9 rows sealed.
- The Gross9 novelty evaluator was committed and pushed before comparator clocks
  were opened.

## Failure

All five Gross9 sleeves passed exact-entry Jaccard, occupied-5m-bar Jaccard, and
absolute signed-exposure correlation. All five failed only the preregistered
one-to-one six-hour matched-share ceiling of 0.35:

| Gross9 sleeve | Candidate matched within 6h |
| --- | ---: |
| `cand_rex_veto_7` | 0.6200 |
| `fresh_kimchi_fx` | 0.7838 |
| `frozen_annual_rank7` | 0.6897 |
| `markov_transition_long` | 0.7647 |
| `rex_taker_low_range_position` | 0.5833 |

Exact-entry Jaccard was zero for every sleeve, but that does not override the
failed near-time gate. The candidate's frequent reserved eight-hour schedule is
therefore not sufficiently orthogonal to the existing Gross9 event clocks under
the frozen definition.

## Sealed boundary and decision

No BTC execution rows, post-entry prices or returns, funding rows, economic
outcomes, portfolio returns, or PnL metrics were opened or computed. No phase,
threshold, direction, hold, volatility, or subset repair is authorized, and none
of the diagnostic controls may be promoted. QHOIR-8 is terminally rejected at
novelty.
