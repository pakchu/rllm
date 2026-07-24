# GitHub Advisory first-add source rejection — 2026-07-24

## Decision

Permanently retire:

```text
GHAD-GRFA-D1 — GitHub-reviewed Advisory first-add daily ledger
```

at the frozen first-parent history source gate.

The sole authorized production source audit ended in:

```text
decision: TERMINAL_REJECT
stage: history
exception_class: HistoryError
```

No retry, resume, repair, alternate history parser, shallower interval,
current-`main` substitution, API fallback, archive fallback, mirror,
semantic mechanism selection, or outcome test is authorized.

## Immutable evidence

The source boundary was committed as:

- commit: `1ca886c4a08fb6f6bda19bee81c78e1f2915dc80`;
- document:
  `docs/github-advisory-first-add-source-axis-decision-2026-07-24.md`;
- SHA-256:
  `b167da46a43308a5ce6be70563c455b1c4209499ae5a0423efbdad15080bb25f`.

The independently reviewed verifier was committed before source access as:

- commit: `9bf903ae0345520149824cb9ac813e45b1a1d3ec`;
- runner Git blob: `353936bc4fa512f9bd854e41809b24f0c5c0cf9d`;
- script SHA-256:
  `f5acea9b86f4f0633dc61e8d65584e384096e84012b20ecc2f5fc62054b15808`;
- test SHA-256:
  `c9bfc216c3fde197de13e463acefed94848474be0bd2c79895162bba687eefd0`;
- protocol: `GHAD-GRFA-D1-source-audit-2026-07-24`; and
- fresh synthetic verification: `29 passed`.

The authoritative generic rejection report is:

- commit: `6bc3296db1e26419255deeddbe203be09f4b9763`;
- path:
  `results/github_advisory_first_add_source_2026-07-24.json`;
- SHA-256:
  `8a2694952157ff442a49c697d1081a5fc679d992fb3e985f7569dde94eae348e`.

The report confirms:

- `execution_authority = production_one_shot`;
- `source_audit_authoritative = true`;
- `mechanism_preregistration_authorized = false`;
- `retry_or_resume_authorized = false`; and
- every no-outcome exposure flag remained `false`.

## Exposure boundary

The production attempt consumed the immutable sentinel and stopped during
first-parent history validation, before candidate-blob materialization.
The ignored local object database contains source transport evidence only.
Neither it nor the private retrieval manifest was opened during rejection
inspection.

Only the committed aggregate report was inspected. No advisory body, package
name, purl, advisory identity, vulnerability meaning, semantic label, model,
tokenizer, prompt, adapter, market data, candidate incidence, return, PnL,
CAGR, MDD, reward, checkpoint, or portfolio result was inspected or
published.

## Consequence

GHAD-GRFA-D1 contributes no alpha, feature, gate, model input, portfolio
weight, or live dependency.

The alpha search must move to a different preregistered source axis. The
GitHub Advisory first-add axis may not be reopened under another label.
