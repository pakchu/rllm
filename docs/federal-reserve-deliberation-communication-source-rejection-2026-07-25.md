# FRDCL-D1 Source Rejection

## Verdict

**TERMINAL_REJECT — no repair, resume, or retry is authorized.**

The authoritative one-shot source audit was executed from clean commit
`01aaebae240bb98bc1a9685b96d6715684aa76ec` after the source boundary,
identity ledger, verifier, and synthetic tests were committed.

The committed aggregate report is:

```text
results/federal_reserve_deliberation_communication_source_2026-07-25.json
```

Its file SHA-256 is:

```text
7d9ee7a007dc1a066dc60ca27090c4f1f9fb68511a4439150d3e9d28c2a73801
```

Its self-bound report hash is:

```text
d36f44b01da6357edf980f77e873276ec4d7229415667175bd28280887506b8d
```

## Failure boundary

The frozen aggregate failure is:

```text
stage: historical_indexes
exception_class: IndexError
```

The source axis failed before document-corpus support, model access, or market
access. Under the preregistered stop condition, an index-parser or index-shape
failure after sentinel creation permanently retires FRDCL-D1. The implementation
may not be loosened and the source attempt may not be rerun.

No ignored raw HTML or manifest content was opened during failure review. Only
the committed aggregate report and filesystem-level hashes/counts were used.

## Confirmed closed boundaries

The aggregate report confirms:

- article text emitted: false;
- database opened: false;
- market, price, return, or funding opened: false;
- model, tokenizer, adapter, prompt, or checkpoint opened: false;
- portfolio, reward, or performance opened: false;
- post-2020 document body opened: false; and
- semantic label, embedding, or inference called: false.

Therefore this rejection contains no alpha outcome, no direction choice, and
no information that can tune a later trading mechanism.

## Consequence

FRDCL-D1 is retired. A future alpha search must select a different source axis
under a new boundary and one-shot identity. It may reuse generic source-audit
engineering, but it may not repair, resume, or rerun this FOMC source attempt.
