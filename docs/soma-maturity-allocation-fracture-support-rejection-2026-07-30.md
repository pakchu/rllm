# SMAF-72 source-support rejection — 2026-07-30

## Verdict

**Retire SMAF-72 unchanged before external novelty or BTC outcomes.**

The frozen maturity parser failed the third preregistered source gate,
`parser_coverage_and_complete_operations`. The exact first failing check was
`security_description_parser`. The preregistration makes any newly observed
parser failure terminal, so the grammar, maturity statistic, polarity, rank
window, tail, onset, latency, hold, support floor, comparator, cost, and
threshold remain unchanged.

No external comparator row, BTC market row, funding row, forward return, PnL,
CAGR, or MDD was opened. Profitability statistics are therefore `N/A`, not
zero.

## Frozen source result

| Split | Description rows | Parsed rows | Parser coverage | Operations | Complete operations | Complete share | Availability batches | Singleton batches | Singleton share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 182,616 | 154,552 | 0.8463223375826872 | 1,259 | 180 | 0.14297061159650518 | 1,249 | 178 | 0.14251401120896717 |
| Warmup | 26,880 | 26,129 | 0.9720610119047619 | 251 | 180 | 0.7171314741035857 | 249 | 178 | 0.714859437751004 |
| Train | 113,667 | 95,288 | 0.8383083920575013 | 757 | 0 | 0.0 | 751 | 0 | 0.0 |
| Selection | 42,069 | 33,135 | 0.7876346002995079 | 251 | 0 | 0.0 | 249 | 0 | 0.0 |

The frozen parser rejected 28,064 joined detail rows. Under the fail-closed
contract, one rejected detail invalidates its whole operation; 1,079
operations were invalid. The exact required parser coverage, complete-operation
share, and singleton-batch share were all `1.0` in every split.

The frozen identity/header and the preceding schema, join, uniqueness, and
exact-reconciliation checks passed. The evaluator stopped the decision at the
first failing ordered gate. It emitted no source clock events:

```text
primary rows: 0
all control rows: 0
total clock rows: 0
```

The incidence cannot be repaired by extending the now-seen parser grammar,
dropping unmatched rows, retaining partial operations, changing the maturity
centroid, weakening coverage floors, or changing any downstream gate.

## Sealed boundaries

The terminal report records:

```text
comparator_rows_decoded: 0
novelty_authorized: false
novelty_passed: false
btc_market_rows_loaded: 0
funding_rows_loaded: 0
forward_return_rows_loaded: 0
pnl_cagr_mdd_opened: false
outcomes_opened: false
economic_evaluator_authorized: false
model_or_gpu_calls: 0
network_calls: 0
```

SCAF and SLCS comparator identities remained hash-bound, but zero comparator
data rows were decoded because every source gate had to pass before external
novelty. The sealed economic evaluator was not authorized.

## Integrity evidence

- preregistration commit:
  `01bdc8b923f1ddd4e218df28239e2b814fc47f62`
- source-support evaluator commits:
  `67320969cd75a55a002cbcd85d580501ad5288ce` and
  `74fbf91c1c338d2f087299f68db67c7d3e2701a7`
- evaluator protocol:
  `soma_maturity_allocation_fracture_support_v1`
- terminal decision:
  `retire_SMAF_72_unchanged_before_outcomes`
- evaluator implementation SHA-256:
  `08f7278bd7e285e2f8dac2cf06f71d77de90aba697eee43d8051537ef6941264`
- evaluator test SHA-256:
  `c222ced3b9718ac2f2a91d898b3bb6d501629cab555edc7f98a96a50884d0829`
- support clock:
  `data/soma_maturity_allocation_fracture_clocks_2020_2023.csv.gz`
- support clock rows: `0`
- support clock SHA-256:
  `8c76f8207cd9e9fa609e123e290b0f07e7d30defcaa6fa9ee6cbc57a654321b3`
- support report:
  `results/soma_maturity_allocation_fracture_support_2026-07-30.json`
- support report SHA-256:
  `afacd9e048e4846322ff5fef41fc8947123565fa75ca2589d8d0531a6f408f9d`
- support report manifest hash:
  `8b33d545ba70901fdd0b1e15f779fc44f953541d064688f00a96cf107ea38e31`

The first successful direct evaluator invocation created both write-once
artifacts. A second invocation validated the existing artifacts byte for byte,
reported `verified_existing` for each, and left both SHA-256 digests unchanged.
The report also passes the evaluator's exact terminal-report schema and
cross-field validation.

## Research implication

Do not create `SMAF-72B` by repairing this parser or relaxing source support.
A future maturity candidate must use a genuinely different, newly frozen
information geometry, disclose that SMAF source incidence has been seen, and
receive a new identity and pre-outcome preregistration.
