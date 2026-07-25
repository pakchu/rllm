# CLOR-D1 source-support rejection

Date: 2026-07-25

## Terminal decision

CLOR-D1 is retired unchanged before outcomes.

The sealed source-support evaluator was invoked exactly once from clean commit
`6ebdeb1`. It stopped at the first frozen gate:

```text
gate 1
source_schema_chronology_reconciliation
decision
reject
terminal action
retire_clor_d1_unchanged_before_outcomes
```

The strict exact-decimal parser encountered a signed-zero representation and
raised `CLOR-D1 decimal uses signed zero`. The parser grammar, runner bytes,
tests, runtime, and gate order had already been committed and execution-sealed
before any source value row was decoded. Normalizing or relaxing that grammar
after observing the failure would be a post-incidence repair, so this candidate
must not be rerun or amended.

## Source-only evidence

The single run decoded the four frozen allowlist projections:

| Projection | Rows |
|---|---:|
| Treasury | 445 |
| SOMA operations | 1,259 |
| SOMA details | 182,616 |
| OFR | 77,369 |
| predecessor value rows | 0 |

No joint source schedule, control schedule, model row, action, market row,
funding row, future return, reward, trade, PnL, CAGR, or MDD was built.
Every frozen forbidden-access counter remained zero.

## Terminal artifact

```text
path
  results/collateral_liquidity_ordering_relation_source_rejection_2026-07-25.json
SHA256
  2fd8bf4546ddac9a566bb6a8a7ca34c077d4335b2920a6b4e91b7490e41dc58f
result_hash
  eb158cd4634be0bd4452dd9e0fd4f8cffeba714bebbf35a123fec36b6cb8e8fc
```

The pass source CSV, pass controls CSV, and pass report are absent. This result
contains no alpha or profitability evidence. Any subsequent alpha candidate
requires a new identifier, boundary, preregistration, implementation contract,
and one-shot source evaluation rather than a CLOR-D1 repair.
