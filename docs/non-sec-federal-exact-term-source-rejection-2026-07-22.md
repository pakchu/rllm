# NFET positive-member source rejection — 2026-07-22

## Decision

**`REJECT_NO_REPAIR`.** The exact frozen Non-SEC Federal Exact-Term (`NFET`)
source policy is retired before source-support, novelty, market, or outcome
evaluation.

The official-membership build started from an empty output/archive state. It
completed 355 of the 486 frozen candidates and stopped at candidate 356,
Federal Register document `2023-07229`.

## Exact positive-member failure

The frozen GovInfo HTML parser found one exact supporting term:

```text
pattern_id = blockchain
substring = blockchain
span = [524717, 524727)
```

This makes the document a positive member. Its official GovInfo MODS, HTML,
PDF, and Federal Register detail responses all returned HTTP 200. GovInfo MODS
produced the canonical agency set:

```text
DEPARTMENT OF HEALTH AND HUMAN SERVICES
OFFICE OF THE SECRETARY
```

The official detail response is:

<https://www.federalregister.gov/api/v1/documents/2023-07229.json>

Its second agency object was exactly:

```json
{"raw_name":"Office of the Secretary"}
```

It had no `agencies[1].slug`, so the frozen positive-member reconciliation
stopped with:

```text
ValueError: NFET detail JSON has a malformed agency slug
```

## Sealed evidence

The machine-readable rejection is
`results/non_sec_federal_exact_term_source_rejection_2026-07-22.json`, with
result hash
`65b388afd59a2fd42ff1dcbad3a29badcd46d147591e8f01e4f3c5b20e21aa75`.

Only the failing positive member's four source objects remain in the repository:

- GovInfo MODS raw SHA-256:
  `38f122b41075e281a0b5b54c67f3c07b8ddbc1c4d3c92e7908b7d2b363f4c305`;
- GovInfo HTML raw SHA-256:
  `368799e0372d84b9bf0b41e50745d38b5968b9e1c0f6d616f8bfec039647f1c3`;
- Federal Register detail raw SHA-256:
  `fd8690100b07b16d4f89f937e2f50b368887e534ad09d63ff6eabcbdbae2ce57`;
- GovInfo PDF raw SHA-256:
  `4945eff81df97460d9dec57bf2a5260ea6c3e05716e981f08299bc2ae2bc5dd9`;
  and
- canonical visible-text SHA-256:
  `9bb0edb36627744d4f529d096318f83fed33afaf0b7dceb892f21992d36d5441`.

The regression test decompresses every object, verifies its raw and gzip hash,
replays the exact HTML match, parses MODS, verifies PDF magic, and reproduces
the agency-slug failure.

## Why this cannot be repaired

The source protocol froze these requirements before official membership was
opened:

- every positive member must reconcile across GovInfo MODS, exact HTML, PDF,
  and Federal Register detail JSON;
- detail `agencies[].slug` is the mandatory independent agency cross-check;
- GovInfo SEC routing and the detail slug set must agree;
- no alias, parent roll-up, quarantine, or imputation is permitted; and
- one positive parse or reconciliation failure retires the whole source.

Synthesizing the missing slug, deleting the child agency, routing from GovInfo
alone, or skipping this positive would alter the policy after seeing its source
data.

## Opened and closed boundaries

The build opened a partial membership prefix and retained 1,267 network
responses before stopping: 356 MODS, 356 HTML, 356 detail JSON, and 199 positive
PDF responses. The final exact-member incidence was not completed, and the
partial prefix cannot be used as a signal.

No market clock, price, funding row, future return, PnL, CAGR, strict MDD,
novelty score, semantic model, selected event source, source manifest, or
support result was opened or produced. The resume state and every unrelated
partial object were deleted.

Any successor must use a genuinely different observable and a new
outcome-blind preregistration rather than a repaired NFET policy.
