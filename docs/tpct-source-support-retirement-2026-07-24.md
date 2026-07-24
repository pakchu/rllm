# TPCT-120 source-support retirement

## Decision

`TPCT-120` is retired unchanged before comparator, market, funding, return,
label, model-training, or PnL access.

```text
decision =
  retire_TPCT_120_unchanged_before_comparators_or_outcomes
```

The committed protocol was executed once from clean commit:

```text
7d49facfabdfb1450235563b01e617efd962bb0d
```

## Frozen artifacts

```text
results/tri_party_composition_topology_source_support_2026-07-24.json
SHA256 3ab639e33b62aed13aa6471e99dfb38a7ba10a19a79270f043e5071b6f8263a1
manifest e888172f073d202304adef4e5bff23a780aaa3f6c72846efcd8be71b83ad49b8

results/tri_party_composition_topology_source_clock_2026-07-24.csv.gz
SHA256 77a1ec73182d4a6889ab5778ed504a66119d222e740544c3b6abf35a02bba3ed
```

The clock contains only its header because no complete vector existed.

## Source-support result

```text
physical source rows read                 77,369
eligible selected rows seen              11,844
eligible selected values converted       11,844
eligible TPCT source dates                   987
complete 16-row vectors                        0
invalid TPCT source dates                    987
rank-complete decisions                        0
token states                                   0
train opportunities                            0
2022 selection opportunities                   0
sealed values converted                        0
sealed candidate statistics                    0
```

Every eligible source date contained exactly twelve of the sixteen frozen
mnemonics. These four required series had zero eligible rows:

```text
REPO-TRIV1_AR_B27-P
REPO-TRIV1_TV_B27-P
REPO-TRIV1_AR_B830-P
REPO-TRIV1_TV_B830-P
```

All other twelve required series had 987 eligible rows. The missing four were
metadata definitions without observations in the frozen window. The aggregate
source audit had disclosed that some definitions were metadata-only, but it did
not identify these candidate-specific mnemonics before the preregistered source
support run.

## Outcome boundary

```text
comparator rows read       0
market rows read           0
funding rows read          0
return/PnL rows read       0
model labels created       0
model training runs        0
network calls              0
sealed values converted    0
```

No performance statistic exists for TPCT-120.

## Consequence

Removing `B27`/`B830`, replacing them with `LE30`/`TOT`, changing the primitive
simplex, or relaxing vector completeness would repair a failed identity and is
forbidden. Any later OFR candidate must use a new candidate ID, a newly frozen
mechanism, and a new source-support contract. TPCT-120 cannot proceed to
comparator novelty, cheap economic evaluation, Gemma training, or 2023
evaluation.
