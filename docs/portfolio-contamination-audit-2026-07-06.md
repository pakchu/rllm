# Portfolio Contamination Audit — 2026-07-06

## Scope
Recent portfolio candidates combine discovered sleeves and portfolio weights across:
- train: `<2024`
- test: `2024`
- eval: `2025`
- diagnostic/live-adjacent: `2026YTD`

## Critical contamination warning
The latest portfolio weights are **research/discovery candidates**, not clean OOS validation results.

Reasons:
1. Portfolio-level weights were selected after observing 2024, 2025, and 2026YTD performance.
2. Constraint changes were iterated interactively using test/eval/2026 metrics.
3. 2026YTD was used as a selection constraint in later scans, so it cannot be interpreted as untouched forward validation.
4. Some sleeves had selectors/filters derived from prior searches; each sleeve must be audited separately for feature/label split purity.

## Safe interpretation
- Use current candidates as **alpha/portfolio hypotheses**.
- Do not report 2024/2025/2026 metrics as clean validation of the portfolio selection procedure.
- Treat train metrics as diagnostic only, and not as acceptance evidence.
- Treat 2026YTD metrics as robustness diagnostics only if the exact candidate was frozen before looking at later 2026 data; otherwise it is also contaminated.

## Required clean protocol before live confidence
1. Freeze candidate definition:
   - sleeve definitions
   - selector rules
   - portfolio weights
   - cost/slippage assumptions
   - evaluation code hash or script path
2. Run a walk-forward selection protocol:
   - Fit/discover sleeves and weights using data up to cutoff T.
   - Validate only on T+1 untouched period.
   - Roll forward multiple times.
3. Add a final paper/live shadow period with no further selection changes.
4. If using an LLM selector, freeze prompt/schema/rules and train cutoff; log every blocked/allowed decision.

## Current candidate status
- `portfolio_gross6_mdd20_ratio5_return_best_candidate`: discovery candidate only.
- `portfolio_gross6_oos_mdd20_return_best_candidate`: discovery candidate only.
- `portfolio_gross3_*`: discovery/paper candidates only unless separately frozen before forward data.

## Reporting language
Use:
> “Selected on historical 2024/2025/2026YTD constraints; metrics are in-sample-to-selection / research diagnostics.”

Do not use:
> “Clean OOS proven” or “validated on 2026.”
