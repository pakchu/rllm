# CEFS-D2 source-support retirement

Date: 2026-07-25

## Decision

**RETIRE CEFS-D2 unchanged before market-outcome access, reward
construction, model training, or economic evaluation.**

CEFS-D2 was the clean-room successor allowed after CEFS-D1 failed before
source decoding. It preserved CEFS-D1's scientific mechanism and changed
only the candidate identity and execution authority hardening.

Machine-readable terminal evidence:

```text
results/cboe_edge_flip_sequence_policy_d2_source_rejection_2026-07-25.json
SHA256 7c4ee86ad540ad1eefb92d35859948818a0637dc1be7ddbb2d527cfb6f2924bb
result_hash ade0d792d7231693482ab713c9f848fcce0aa1f3abafb0d0c14aed959358ea2b
source_row_hash 4c0f7c4bf398a5f0fa7266a025d72e7bc6fae65a8d3bb6bf0df67c678c4b3c89
```

## Exact result

The first three frozen gates passed and Gate 4 failed:

```text
1  runtime_authority_forbidden_access  pass
2  schema_chronology                    pass
3  schedule_support                     pass
4  primitive_edge_support               fail
```

Only these two preregistered checks failed:

```text
eval_term_back_level_two_levels  false
eval_term_back_level_max_share   false
```

The decoded `TERM_BACK_LEVEL` support was:

| split | rows | HIGHER | LOWER | dominant share |
|---|---:|---:|---:|---:|
| TRAIN | 498 | 55 | 443 | 88.96% |
| TEST | 251 | 15 | 236 | 94.02% |
| EVAL | 250 | 0 | 250 | 100.00% |

The 2023 EVAL source language therefore collapsed to one state. The frozen
primitive could not express a level flip in that split and failed both the
minimum-level and maximum-dominance requirements.

## Evidence boundary

This is a genuine source-support rejection, not an alpha or profitability
result:

- the official source rows were decoded only through the primitive-support
  gate;
- no control rows were built;
- no BTC market, funding, or future-return row was opened;
- no reward, model, selected action, or trade row was built;
- no PnL, absolute return, CAGR, or strict MDD was computed; and
- no source/control pass artifact was published.

All forbidden counters are zero. The non-null source-row hash is
stage-consistent because the frozen schedules were built before Gate 4.
The control-row hash is null because the runner never reached Gate 6.

## No repair or CEFS-D3

Do not relax the level-count or dominance thresholds, remove
`TERM_BACK_LEVEL`, change its definition, alter split dates, or rerun the
same identity. Those changes would use observed EVAL source incidence to
repair the candidate.

CEFS-D2 is permanently retired. The next research unit must use a genuinely
independent mechanism and source family rather than a CEFS-D3 incidence
repair. Economic/RLLM evaluator freezing remains unauthorized.
