# DCRM-1 preregistration — 2026-07-17

## Mechanism

DCRM-1 is a weekly, market-neutral cross-alt pair. At Monday 00:00 UTC it
uses only completed bars through Sunday 23:55, ranks six alts by beta-residual
30-day momentum, then buys the maximum and shorts the minimum at the Monday
00:05 open. The pair exits seven days later.

A strictly-prior 26-week dispersion q80 controls risk, not direction: gross is
1.0 below or at q80 and 0.25 above it. The current week is excluded from the
reference distribution. The weights are positive and beta-neutral before that
gross scale is applied.

## Why this is structurally different

The candidate has no BTC leg and uses no REX, OI, funding signal, premium,
Kimchi, FX, DXY, tree model, Markov state, or LLM. Its weekly cross-sectional
ranking and seven-day holding period also differ from the active event-driven
sleeves. Correlation is nevertheless an outcome and will be opened only after
the standalone gates pass.

## Evidence boundary

Only the causal weekly feature/support clock was inspected. Two outcome-blind
risk treatments were compared: abstention above q80 left 59 events, while
quarter gross retained 92. No post-entry price, trade return, or equity curve
was calculated. The latter is frozen as DCRM-1.

## Qualification

Support must first pass at least 85 events, 35 per year, 10 per half, 12
ordered pairs, <=20% pair concentration, <=15% month concentration, and all
six symbols on both sides. Then 2023 is opened exactly once. It must be
positive, have CAGR/strict-MDD >=2, strict MDD <=15%, at least 35 trades,
positive halves and all controls. Only a complete pass opens 2024, which must
reach CAGR/strict-MDD >=3; 2025 and 2026 remain sequentially sealed.

Strict MDD uses the global/pre-entry HWM, held two-leg OHLC with
favorable-before-adverse ordering, funding, and hypothetical liquidation
costs. CAGR always spans the full declared calendar.

## Research context

- 30-day cross-sectional continuation with a seven-day horizon:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4322637
- dispersion-conditioned momentum (recent 2026 working paper):
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6648082
- adverse realism check on implementable crypto momentum (2026 working paper):
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4675565
- horizon dependence and reversal warning:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3913263

These sources motivate the test; they do not validate this candidate.

Protocol hash: `e41f3acdb7297c6704db2f225eea0764d2e8252285713f282d07bdc8a6ffb4eb`
