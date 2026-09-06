# Additional-alpha portfolio recheck — 2026-09-06

## Scope and decision

Three retained additions: macro-flow/regime switch, OI divergence pullback, and the weak regional-demand trend diversifier. Exhaustive nonnegative 5% simplex: **231 weights**, sum 100%, net exposure capped at 1x at execution events. Overlap is allowed and opposite positions offset before fees/funding. No fee-ratio or frequency exclusion.

**No allocation passes the subsequent-period return check. Do not promote the optimizer output.**

Legacy Gross9 integration is not complete: its historical execution clocks and the new sleeves require a common causal replay adapter. These results do not establish the optimal portfolio across every historical alpha. Basis/calendar/direct-funding failures are not added to the retained universe.

## Protocol

- June 1–July 1: rank by 10bp/side CAGR/MDD.
- July 1–August 3 21:45 UTC: fixed-weight report, no replacement.
- June 1–August 3: retrospective best reported separately.
- Every period was previously exposed; the chronological split is a diagnostic, **not pristine OOS**.
- OI source ends August 3; 8 OI trades across the common period.
- Funding uses realized mark-price transfers. MDD uses conservative five-minute high-before-low ordering.
- Sleeves update their own quantities only. Unlike the prior v3 blend audit, an hourly macro update does not implicitly retarget the OI sleeve. Net-risk cap intervention is a separate event-driven exception.
- Each window starts in cash and initializes the known sleeve state; this may initialize a carried-in signal. All final holdings are liquidated and charged.

## Results at 6bp per side

| Allocation (macro / OI / regional) | Full-window return | Full MDD | Full CAGR | July–Aug return |
|---|---:|---:|---:|---:|
| June-selected 80 / 20 / 0 | +2.08% | 1.72% | 12.50% | -1.41% |
| Retrospective stress-Calmar best 60 / 40 / 0 | +3.30% | 1.90% | 20.38% | -1.32% |
| Fixed 50 / 50 / 0 | +3.91% | 2.37% | see report | -1.28% |

June-selected 80/20 at 10bp: full +1.61%, subsequent period **-1.65%**.
Retrospective 60/40 at 10bp: full +2.77%, subsequent period **-1.52%**.
All **231/231 allocations lose** in the subsequent period at 10bp. Regional trend receives zero in both selected and retrospective winners.

80/20 has 23 net entry episodes and 437 order changes including terminal liquidation across the common window; fees are 0.716% of initial equity. Entry episodes are not independent sleeve trades. Short-window CAGR is descriptive and highly unstable.

## Artifacts

- `training/optimize_added_alpha_portfolio.py`
- `research/added_alpha_portfolio_optimization_v2/design.json`
- `research/added_alpha_portfolio_optimization_v2/selection_freeze.json`
- `research/added_alpha_portfolio_optimization_v2/report.json`
- `research/added_alpha_portfolio_optimization_v2/shadow_config.json` (disabled diagnostic, not a promoted candidate)

The first run was superseded because zero-weight sleeve events could trigger a net-cap resize. Its hash and reason are preserved in `research/added_alpha_portfolio_optimization/failure.json`; the corrected implementation has a regression test for that case.
