# Frozen PPOSM → Gross9 same-gross marginal preregistration

- preregistration SHA-256: `0fbb2680383899b867f6ff3d6c381be7796f3075bee4f2a9a2573da550ce1553`
- candidate freeze: `1360cf620b8afcead476e2aa0c1394e1419b70c42cac7e03dcc91912ab60c81f`
- Status: frozen before any PPOSM/Gross9 merged portfolio statistic is computed.
- Caveat: standalone 2024–2026 outcomes were already exposed; this is a practical portfolio-interaction audit, not pristine discovery OOS.

## Fixed question

Does the exact frozen 0.5x PPOSM path improve authoritative Gross9 versus pro-rata Gross9 at the same configured gross?

## Selection

- Candidate weights: `0.25, 0.50, 0.75, 1.00`; total configured gross `9+c` (maximum 10).
- Rank on full train calendar and 2024 only; freeze one top1.
- Require positive/stress-positive standalone behavior, return retention, same-gross ratio improvement, no worse strict MDD, and exact-entry Jaccard ≤ 0.25.

## Future veto

- Open only the frozen top1 on 2025 and 2026H1; no rerank, rank2, threshold repair, or weight repair.
- Each future window must remain positive and improve same-gross risk efficiency without worsening strict MDD.
- Combined 2025–2026H1 additionally requires ≥20 candidate trades, ≥20 active weeks, sign-flip p≤0.20, and positive 90% bootstrap lower mean paired effect.

## Stop rule

- No pre-2025 passer: reject without future portfolio opening.
- Frozen top1 future failure: terminally reject this exact candidate and move to a materially different alpha family.
