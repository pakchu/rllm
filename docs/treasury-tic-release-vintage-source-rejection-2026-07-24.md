# Treasury TIC release-vintage source rejection — 2026-07-24

## Decision

Permanently retire:

```text
TICRV-v1 — Treasury International Capital Release-Vintage Ledger
```

at the frozen ZIP-structure source gate.

The sole authorized production source audit ended in:

```text
decision: TERMINAL_REJECT
stage: zip
exception_class: ZipContractError
```

No retry, resume, repair, narrower archive range, alternate parser, mirror,
cache, current-vintage substitution, mechanism selection, or outcome test is
authorized.

## Immutable evidence

The source boundary was committed as:

- commit: `a4b06f9ffad983d197cbb91e965e1a46ba1a15ad`;
- document:
  `docs/treasury-tic-release-vintage-source-axis-decision-2026-07-24.md`;
- SHA-256:
  `ed7008daff527d0bdfe13dd03686ca7113e39dfff8ab4a7056100cbd36799419`.

The independently reviewed verifier was committed before source access as:

- commit: `15c745f8839371f2473deff56d8613817a45f608`;
- runner Git blob: `b686e13f7eae3fb225f6eb6a5eb9810b8d65ecfb`;
- protocol: `TICRV-v1-source-audit-2026-07-24`.

The authoritative generic rejection report is:

- commit: `68fdae14752ce81c56bd836a1d3c109ef15a3d9d`;
- path:
  `results/treasury_tic_release_vintage_source_2026-07-24.json`;
- SHA-256:
  `b5104531e20dfbab233f72ec4d1eabc59cc2e509de0c2e03700f14455e0654e1`.

The report confirms:

- `execution_authority = production_one_shot`;
- `source_audit_authoritative = true`;
- `mechanism_preregistration_authorized = false`; and
- `retry_or_resume_authorized = false`.

## Exposure boundary

The production attempt consumed the immutable sentinel and performed only the
frozen source audit. Raw index, redirect, and archive bytes remain ignored
local artifacts.

The committed report intentionally does not identify the failing archive or
member. No TIC table, header, row, field, country, sector, instrument, flow,
candidate incidence, BTC market value, return, funding, PnL, CAGR, MDD, model,
or portfolio result was inspected or published.

## Consequence

TICRV-v1 contributes no alpha, feature, gate, model input, portfolio weight,
or live dependency.

The alpha search must move to a different preregistered source axis. The
Treasury TIC release-vintage axis may not be reopened under another label.
