# Price-action event feature scan (2026-06-23)

## Purpose

The previous numeric ridge bundle failed rolling validation. This scan changes the representation toward LLM-friendly symbolic price-action state:

- breakout above / below prior range;
- failed breakout / failed breakdown;
- liquidity sweep and reclaim;
- mid-range reclaim / rejection;
- volume-confirmed variants.

The goal is not to deploy a single event rule. The goal is to find event tokens worth feeding into RLLM context.

## Causal feature construction

For each lookback window `w`, prior range levels are computed as:

- `prior_high = high.shift(1).rolling(w).max()`
- `prior_low = low.shift(1).rolling(w).min()`

Therefore a bar at time `t` can break or sweep the prior range without contaminating the range definition with its own high/low.

Windows scanned: `36, 72, 144, 288, 576, 2016` 5m bars.
Horizons scanned: `36, 72, 144, 288` bars.

## Protocol

- Train side fit: `2020-01-01` through `2023-12-31`.
- Test diagnostic: `2024-01-01` through `2025-12-31`.
- Untouched eval: `2026-01-01` through `2026-06-01`.
- Side mapping: train-only conditional mean; positive => LONG, negative => SHORT.
- Backtest: event-triggered strict bar-by-bar MDD, 0.5x leverage, 4bp fee, 1bp slippage, next-bar entry.
- Output: `results/price_action_event_scan_2026-06-23.json`.

Scanned: 90 sparse event tokens × 4 horizons = 360 rules.

## Result summary

Strict filters:

| Filter | Passing rules |
| --- | ---: |
| test ratio > 1 and eval ratio > 1 | 0 |
| test ratio > 1 and eval CAGR > 0 | 0 |
| test/eval CAGR > 0 and both strict MDD <= 15 | 4 |
| train p < 0.05 and test CAGR > 0 | 67 |

Top test-ranked event:

| Event | Horizon | Side | Train n | Train mean | Train p | Test CAGR | Test MDD | Test ratio | Test trades | Test p | Eval CAGR | Eval MDD | Eval ratio | Eval trades | Eval p |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pae_w576_failed_breakdown_long` | 144 | LONG | 777 | 0.199% | 0.159 | 13.49% | 9.26% | 1.46 | 218 | 0.098 | -18.35% | 9.03% | -2.03 | 47 | 0.273 |

Best same-sign but weak cases:

| Event | Horizon | Side | Train p | Test CAGR | Test ratio | Eval CAGR | Eval ratio | Comment |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `pae_w2016_break_below` | 72 | LONG | 0.006 | 2.93% | 0.25 | 6.42% | 0.98 | positive but too weak |
| `pae_w2016_break_below_with_volume` | 72 | LONG | 0.006 | 2.93% | 0.25 | 6.42% | 0.98 | duplicate due volume always true in this subset |
| `pae_w2016_break_below` | 144 | LONG | 0.916 | 6.57% | 0.67 | 1.26% | 0.15 | train not significant |
| `pae_w2016_break_below_with_volume` | 144 | LONG | 0.916 | 6.57% | 0.67 | 1.26% | 0.15 | train not significant |

## Interpretation

Event tokens are a better representation for RLLM than raw numeric ridge outputs, but single event rules are still not deployable.

The important signal is structural:

- `break_below`, `failed_breakdown_long`, and `low_sweep_reclaim` tend to behave as long mean-reversion events in 2020-2025.
- The same family often fails in 2026, so the missing component is regime qualification, not another static event rule.
- This is exactly where an LLM/RLLM context policy can be useful: it should reason over event state plus regime state, not memorize numeric action labels.

## Decision

Promote these as **context tokens**, not direct labels:

- `break_below` / `failed_breakdown_long` / `low_sweep_reclaim`
- long-window variants, especially `w576` and `w2016`
- paired regime fields: trend, volatility, funding/premium, DXY/Kimchi when available

Next step: build a compact RLLM context dataset where the prompt describes recent event state and asks for wait/long/short only after regime qualification. The target should be event-conditioned utility, not raw forward return.
