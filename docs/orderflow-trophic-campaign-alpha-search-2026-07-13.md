# Order-Flow Trophic Campaign Alpha — Preflight

Date: 2026-07-13

## Structural follow-up

The parent one-shot trophic-succession policy contained a weak gross edge but lost
money after 6 bp/side costs in fit. This follow-up freezes the parent q95
continuation roles and changes the object being traded: a single succession event
is treated as noise, while repeated same-direction sponsor-to-crowd successions are
treated as one latent institutional **campaign**.

At each current completed event, the algorithm counts only current and trailing
events. A campaign is confirmed after two or three same-direction events in a
6-hour or 12-hour trailing window, permits at most one opposite event, then consumes
all further confirmations during one full lookback cooldown. This is an event-memory
representation, not another role-score-tail retune.

## Protocol

- Physical source rows strictly before `2024-01-01`; 2024+ OOS stayed unopened.
- Parent role definitions, q95 thresholds and continuation mapping are frozen.
- 48 predeclared policies: six parent phase profiles, lookback `{72,144}` bars,
  minimum same-direction events `{2,3}`, and hold `{144,288}` bars.
- Current event and trailing counts only; next-open entry, 0.5x, 6 bp/side,
  split-contained exits and conservative strict MDD.
- Admission required fit and 2023 CAGR/MDD at least 3, positive returns, adequate
  two-sided trade support, and non-negative 2023 halves.

## Most temporally robust adequately populated policy

Profile `(12,24,6)`, 144-bar lookback, two-event confirmation, 144-bar hold:

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Fit (2020-06..2022) | +24.31% | +8.78% | 11.97% | +0.73 | 100 |
| Selection 2023 | +14.18% | +14.19% | 5.82% | +2.44 | 40 |
| 2023 H1 | +14.11% | +30.51% | 4.85% | +6.28 | 24 |
| 2023 H2 | +0.06% | +0.13% | 5.82% | +0.02 | 16 |

All five fit half-years and both 2023 halves were positive. However, 2023 H2 is
economically flat and full-fit risk efficiency remains far below the target.

The strongest minimum full-window ratio used profile `(12,12,6)`, a 72-bar
lookback, three-event confirmation and 288-bar hold. It produced fit `+37.12%`
return / `+12.99%` CAGR / `9.23%` MDD / `1.41` ratio with 41 trades, and 2023
`+15.70%` / `+15.71%` CAGR / `6.71%` MDD / `2.34` ratio with 23 trades. Its lower
support and 2023 H2 ratio `0.28` also prevent admission.

## Cost decomposition of the robust policy

| Cost per side | Fit return / CAGR / MDD / ratio | 2023 return / CAGR / MDD / ratio |
|---|---:|---:|
| 0 bp | +32.00% / +11.34% / 11.23% / +1.01 | +16.95% / +16.96% / 5.73% / +2.96 |
| 1 bp | +30.68% / +10.91% / 11.35% / +0.96 | +16.48% / +16.50% / 5.73% / +2.88 |
| 3 bp | +28.09% / +10.05% / 11.60% / +0.87 | +15.56% / +15.57% / 5.73% / +2.72 |
| 6 bp | +24.31% / +8.78% / 11.97% / +0.73 | +14.18% / +14.19% / 5.82% / +2.44 |

Unlike the parent one-shot policy, campaign aggregation survives standard costs.
Its weakness is path/regime efficiency rather than turnover alone.

## Controls at 6 bp/side

| Variant | Fit return / ratio / trades | 2023 return / ratio / trades |
|---|---:|---:|
| Exact direction flip | -30.06% / -0.42 / 100 | -16.98% / -0.83 / 40 |
| One-event cooldown baseline | +3.04% / +0.06 / 235 | +13.42% / +1.98 / 91 |
| Sponsor/crowd phase-order swap | +23.71% / +0.55 / 107 | +7.97% / +1.11 / 68 |
| Campaign delayed by 42 bars | +28.14% / +0.79 / 100 | +4.15% / +0.80 / 40 |

Repeated confirmation materially improves the parent and one-event baseline, and
the exact direction is strongly specific. The phase-order and delay controls remain
partly positive, so campaign persistence is broader than the exact trophic story.

## Decision

Reject the exact 48 count/lookback/fixed-hold policies as a standalone alpha because
none reaches CAGR/MDD 3 in both pre-2024 windows. Preserve campaign density as weak
beta evidence: it turns a cost-fragile event into positive fit and 2023 returns, but
must not be promoted or replay-selected on 2024+. A future retry must change the
economic state transition or use an independent data mechanism, not sweep nearby
counts, lookbacks or holds.

Artifacts:

- `training/search_orderflow_trophic_campaign_alpha.py`
- `results/orderflow_trophic_campaign_alpha_scan_2026-07-13.json`
- `tests/test_search_orderflow_trophic_campaign_alpha.py`
