# Gross9 invariant-ensemble Top10 marginal battery preregistration

`G9-IEM-1` freezes the portfolio test before any 2024 marginal result is computed.

## Frozen universe

- Candidates are exactly the ten invariant V-REx/GroupDRO uncertainty policies ranked in 2023 before later metrics.
- Each policy is long-only, enters at the next 5-minute open, holds 576 bars, and is evaluated at weights 0.25, 0.50, 0.75, and 1.00.
- No feature, threshold, rank, side, hold, stride, or exit may be changed.

## Portfolio acceptance

- Test `[2020-09-01, 2024-01-01)` and `[2024-01-01, 2025-01-01)` only. Candidate returns are explicitly flat before its support begins on 2023-01-01 while the full Gross9 calendar remains active.
- Compare every addition against pro-rata Gross9 at identical gross, not against unlevered Gross9 alone.
- Require positive standalone returns and at least 30 trades in both windows, >=0.05 CAGR/strict-MDD improvement in both windows, >=97% Gross9 return retention, MDD reduction in at least one window, <=0.50 entry Jaccard, and positive 10bp/side stress returns.
- Freeze one stable top-1 only. 2025 and 2026 may veto that exact winner but cannot rerank, repair, or substitute another rank.

Jaccard acceptance is the maximum exact-entry overlap across every baseline
sleeve and each selection split. Standalone statistics always use a fixed
1.00 sleeve multiplier (0.50x effective leverage), independent of the
portfolio weight cell. MDD reduction and return retention are measured
against unscaled Gross9; ratio improvement is measured against the same-gross
pro-rata comparator.

## Future support

- A later committed runner at `training/build_invariant_ensemble_frozen_top1_future_support.py` may reconstruct only the frozen winner for `[2025-01-01, 2026-01-01)` and `[2026-01-01, 2026-06-03)`.
- Its support schema, split-reset non-overlap semantics, source/result/freeze bindings, deterministic hashes, and forbidden columns are fixed in the machine contract.
- July 2026 is outside this battery entirely: it cannot rank, veto, or provide report-only evidence.

## Claim boundary

All candidates and later periods are already research-exposed. A survivor is eligible only for prospective forward shadowing; it is not pristine OOS evidence or automatic live-capital authorization.

The exact machine contract, candidate specs, hashes, source paths, thresholds, diagnostics, and future-veto rules are in `results/gross9_invariant_ensemble_top10_marginal_preregistration_2026-07-28.json`.
