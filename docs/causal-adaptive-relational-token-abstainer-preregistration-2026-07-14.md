# CARTA v1 preregistration — 2026-07-14

## Status and boundary

**CARTA-specific returns have not been opened.** This stage computes only
causal features, symbolic tokens, timestamps for split/support accounting, and
candidate counts.

- name: **CARTA — Causal Adaptive Relational Token Abstainer**
- support artifact: `results/causal_adaptive_relational_tokens_support_2026-07-14.json`
- support artifact SHA256: `77dfd1d0b0ad444744157972aa437f805901bc56428a4e5d76029bf64100d339`
- sealed policy windows: 2024, 2025, and 2026 YTD

Pre-2024 prices have been used by many earlier experiments, and NETF-specific
2020–2023 returns are already known. CARTA therefore does not claim a market
history clean room. It claims a newly frozen event clock, state abstraction,
and future evaluation protocol while keeping 2024+ unopened.

## Why this is not a NETF repair

NETF v1 is rejected. Its fixed capital-revelation rule reversed after 2021,
including within the dominant `010` notional-concentration structure. CARTA
does not invert NETF, change its `0.875` threshold, select one structure mark,
or reuse its 30/60-minute confirmations and 4/8-hour holds.

CARTA instead freezes:

- a broader topology-fracture event without a required price-revelation
  direction;
- a new 45-minute observation window;
- a new 6-hour consequence horizon;
- a policy action set of `ABSTAIN`, `FOLLOW`, and `FADE`;
- relational transition tokens rather than one threshold conjunction.

The novel inference is that the economic meaning of a topology fracture is
carried by **how capital, event breadth, price, concentration, arrival shape,
and market context change together**, not by the level of one setup statistic.
This is a hypothesis, not established profitability.

## Verified source and causality contract

CARTA reuses the fail-closed official Binance data foundation.

- aggregate-trade feature SHA256:
  `c2bb0e6742f8cdc4e13315e7f0a13d6ab9cd536fb40d9cb4484b7a6ba30131cf`
- official 5-minute kline SHA256:
  `e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d`
- available range: 2020-01-01 through 2023-12-31, UTC

The complete source-gap day, each missing aggregate-trade slot, and the next
24 bars remain quarantined. The 24-hour context, setup, 45-minute transition,
signal, next-open entry, full hold, and scheduled-open exit must be clean. A
candidate whose setup origin, entry, or exit crosses its evaluated split is
dropped, never force-closed.

Rolling percentiles use only clean observations, shift by one completed bar,
cover at most the preceding 8,640 bars (30 days), and require 2,016 valid
observations (7 days). Future changes must leave every earlier token unchanged.

## Event clock

At completed bar `t0`, define:

- capital direction: sign of signed aggressive quote notional;
- crowd direction: sign of signed aggressive-event-count imbalance;
- topology tension:
  `sqrt(flow_coherence × abs(event_imbalance)) × abs(size_log_ratio)`.

A setup requires:

1. a clean source and at least 64 aggregate-trade events;
2. nonzero, opposite capital and crowd directions;
3. topology tension at or above its strictly lagged 97.5th percentile;
4. at least one feature at or above its lagged 80th percentile: arrival
   burstiness, aggregate-event notional HHI, or underlying trade-ID span per
   aggregate event.

Unlike NETF, immediate price direction is not a gate. After exactly nine more
completed bars, the candidate state is emitted if the full causal context is
clean and all token histories are available. The reference direction is the
setup's capital direction.

The candidate clock is fixed before any model action. It reserves a 72-bar
hold interval even when the eventual action is `ABSTAIN`; abstention therefore
cannot release later candidates and create an action-dependent opportunity
set. Entry, when selected, is the next 5-minute open and exit is the open 72
bars later.

## Symbolic state

The model receives no raw timestamp, row identifier, price, future reward, or
raw feature magnitude. Its 36 fixed fields are short categorical relations:

### Direction and transition relations

- reference side (`LONG` or `SHORT`);
- setup price with/against/flat to the reference;
- 45-minute capital, crowd, and price relation to the reference;
- setup-direction relation to the prior 24-hour trend;
- whether observed transition excursion was reference- or opposite-dominant;
- count of capital/crowd/price relations aligned with the reference;
- current position, frozen as `FLAT` for this one-step policy.

### Structure and market context

- origin and signal three-bit structure marks in fixed order: arrival burst,
  notional concentration, trade-ID span;
- trailing 24-hour range location: lower, middle, or upper third.

### Lagged ranks and changes

Each feature has a rank `0..4` from lagged 20/40/60/80th percentiles and an
origin-to-signal transition `FALL`, `STABLE`, or `RISE`:

1. topology tension;
2. arrival burstiness;
3. notional HHI;
4. underlying trades per aggregate event;
5. flow coherence;
6. normalized effective event count;
7. sign-flip rate;
8. absolute event imbalance;
9. absolute buy/sell size asymmetry;
10. absolute signed price response;
11. trailing 24-hour realized volatility;
12. drawdown from the trailing 24-hour high.

These are relation tokens, not prose summaries from a separate analyzer. CARTA
must remain one compact policy model.

## Support-only calibration

Only the setup tension percentile was varied. The grid and stopping rule were
committed in code: choose the highest tested percentile passing every frozen
support floor (500 total, 50 per year, 80 per 2023 half, and at least 25% per
reference side).

| percentile | total | 2020 | 2021 | 2022 | 2023 | H1 | H2 | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.950 | 858 | 320 | 116 | 102 | 320 | 128 | 191 | yes |
| 0.960 | 765 | 284 | 98 | 88 | 295 | 119 | 175 | yes |
| **0.975** | **559** | **205** | **60** | **58** | **236** | **91** | **144** | **yes** |
| 0.980 | 490 | 176 | 50 | 48 | 216 | 85 | 131 | no |
| 0.990 | 330 | 117 | 30 | 24 | 159 | 61 | 98 | no |

At 0.975, the reference direction is 44.54% long and 55.46% short. No emitted
model token is unavailable.

## Memorization boundary

The split-contained candidate clock yields 323 training candidates in
2020–2022 and 236 selection candidates in 2023. All 323 training token
signatures are unique; all 236 2023 signatures are also unique, and none is an
exact training signature.

This high-dimensional uniqueness is not evidence of alpha. It prevents exact
prompt-memory lookup from being accepted as the policy. CARTA must learn
compositional relations and beat causal linear, Naive Bayes, and memory
baselines. Model evaluation must also report per-token support and action
collapse.

## Frozen next-stage protocol

Before opening any CARTA return, a separate commit must freeze and hash the
reward builder, linear contextual-bandit baseline, Gemma data builder, model
configuration, and evaluator.

### Actions and delayed reward

- `ABSTAIN`: no position, utility zero;
- `FOLLOW`: trade in the setup capital/reference direction;
- `FADE`: trade in the opposite direction.

For both trade actions, the exact account multiplier must use `0.5x`, 5 bp fee
plus 1 bp slippage per side, next-open entry, and the scheduled 72-bar exit.
The training utility will be frozen as exact log account multiplier minus one
third of the trade-local favorable-first held-path drawdown. Labels become
available only after scheduled exit.

### Causal training/update split

1. initial train: labels with exits complete by 2022-01-01;
2. fixed quarterly updates during 2022, each using only labels whose exit is
   complete before that quarter boundary;
3. freeze the final policy/checkpoint before 2023;
4. selection: full 2023, with H1 and H2 reported;
5. 2024+ remains sealed unless the complete 2023 gate passes.

No year/month/timestamp token enters the model. Quarterly updates adapt weights
from delayed outcomes; they do not tell the model which historical era it is
in.

### Required baselines

- always abstain, always follow, and always fade;
- shuffled-label control;
- exact-signature memory (expected to abstain on unseen signatures);
- categorical linear contextual-bandit value model;
- token Naive Bayes;
- base Gemma zero-shot action-token score.

The cheap linear baseline is evaluated before GPU training. Gemma training is
not justified unless causal tokens show 2023 economic learnability beyond the
constant and shuffled controls.

### Execution and final qualification

Every performance table must include absolute return, full-clock CAGR, held-path
strict MDD, CAGR/strict-MDD, and trades. The final single-Gemma policy advances
only if full 2023:

- has positive absolute return;
- has CAGR/strict-MDD at least 3.0;
- has strict MDD at most 15%;
- has at least 60 trades and at least 20 in each half;
- has positive H1 and H2 absolute return;
- has weekly-cluster one-sided `p < 0.10`;
- beats the strongest causal cheap baseline without action collapse.

If it passes, 2024 and 2025 must independently meet the same return, ratio, MDD,
minimum-40-trade, and `p < 0.10` gates; combined 2024–2025 requires `p < 0.05`.
2026 is forward reporting only and cannot alter model or selection.

## Rejection rules

CARTA v1 is rejected without repair if:

- a 2024+ outcome affects tokens, reward, thresholds, model choice, or training;
- 2023 labels enter model weights before 2023 evaluation;
- target echo or oracle actions are presented as policy performance;
- source quarantine or split containment is bypassed;
- a timestamp/raw-price identifier reaches the model;
- performance depends on one exact-signature memory cell;
- the model collapses to one action or cannot beat a cheap causal baseline;
- any full-2023 or half-year economic gate fails.

Changing the 0.975 setup quantile, 9-bar observation, 72-bar hold, token schema,
or action semantics after return opening requires a newly named experiment.
