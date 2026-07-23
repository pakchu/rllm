# OPRR-288 preregistration — frozen before incidence

## Decision

The OPRR-288 source, ordinal-state, transition, session-calendar, control,
support, composition, comparator-novelty, execution, and RLLM contracts are
write-once frozen in:

```text
results/cboe_option_pressure_rank_rotation_preregistration_2026-07-24.json
```

Artifact SHA-256:

```text
76db2b61fe35599acaa9eb52d3406eac891bad3f0c95c17e6ccd212aea719d99
```

Canonical manifest hash:

```text
a8f45ab7339eb773650830ed73f541820082de6c9f86a7dfa40a69b430d2fb99
```

## Evidence boundary

The preregistration process validated only:

- frozen document, source-file, source-manifest, and prior-clock byte hashes;
- the first CSV header line for each source and comparator;
- exact source allowlists and comparator headers; and
- internal consistency of the canonical JSON contract.

It decoded no source data row and no comparator data row. It computed no OPRR
pressure, ordinal position, transition, timestamp, side, count, overlap,
return, funding, PnL, CAGR, or strict MDD. The artifact records all four flags
as `false`:

```text
source_rows_decoded
source_incidence_opened
comparator_rows_decoded
outcomes_opened
```

## Frozen sequence

1. commit this preregistration, generator, artifact, and tests;
2. implement and commit an outcome-blind source-support evaluator;
3. open source incidence only with that committed evaluator;
4. stop unchanged at the first source, support, composition, or novelty failure;
5. open no BTC or funding row unless every earlier gate passes;
6. freeze a separate economic/RLLM evaluator before any market join.

The source-support evaluator may not change the policy, calendar, formulas,
controls, thresholds, comparator set, hashes, or failure action in the frozen
artifact.
