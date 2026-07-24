# DOL Weekly Claims Release-Vintage Source Rejection

## Decision

`DOL-WCRV-D1` is permanently retired as an alpha-research source axis.

The single authorized production audit returned:

```text
decision: TERMINAL_REJECT
stage: newsroom
exception_class: NewsroomError
retry_or_resume_authorized: false
mechanism_preregistration_authorized: false
```

No retry, parser repair, alternate endpoint, mutable-series fallback, source
subset, or market-assisted rescue is authorized.

## Authoritative evidence

- Boundary:
  `docs/dol-weekly-claims-release-vintage-source-axis-decision-2026-07-24.md`
- Boundary SHA-256:
  `7119044d3278a4fc152131e90756cacd1cd28c5dc4b1614a98ff17b6cc8289fd`
- Aggregate report:
  `results/dol_weekly_claims_release_vintage_source_2026-07-24.json`
- Aggregate report SHA-256:
  `84582f517abb1e2e795ec6254c2649bc7d674c4b409d2a9cfbc5ade3b8eacb31`
- Aggregate report self-hash:
  `410ae23b445f86d64efec61f0b69df7490d21176cb002b16ba5672398a3f6cfd`
- Verifier commit:
  `01cf8cc5d50e620b25517bfa117b2184ec51b839`
- Manifest SHA-256:
  `678aef729a82186323f81b7ab2bba099564ea2c668b7e707a9c9a4c76f2ced22`
- Sentinel SHA-256:
  `8e44792d42b2dec5cd0d2596db10baee80474816993413e247108be8fd5a62be`

The aggregate report self-hash and the committed boundary, verifier, and test
hash bindings were independently recomputed after execution.

## No-leak outcome

The authoritative report records all outcome-boundary flags as false:

- no market or price data opened;
- no model, tokenizer, adapter, or prompt opened;
- no portfolio, reward, checkpoint, or performance data opened;
- no mutable current claims series opened;
- no post-window teaser body opened; and
- no private forecast or consensus opened.

Therefore the rejection does not create a hidden model-selection trial. It
ends this source axis before any DOL mechanism or alpha evaluation.

## Next action

Return to source selection. A new source must receive its own immutable causal
availability boundary and source-support audit before any market, model, or
portfolio outcome is opened.
