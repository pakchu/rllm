# ORFR-1 Stage1 rejection postmortem — 2026-07-17

## Verdict

ORFR-1 is rejected at the frozen 2021–2022 Stage1 gate. It produced a real
direction-specific gross effect, but not a stable risk-adjusted standalone
alpha. The 2023 outcome remains physically unopened.

| Window / cost | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Mean gross bp | p(two-sided) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2021–2022, 6 bp/side | +57.8571% | +25.6609% | 17.9620% | 1.4286 | 111 | +102.67 | 0.0175 |
| 2021–2022, 10 bp/side | +51.0072% | +22.9023% | 18.9961% | 1.2056 | 111 | +102.67 | 0.0308 |
| 2021 | +72.6922% | +72.7569% | 15.1162% | 4.8132 | 58 | +215.71 | 0.0011 |
| 2022 | -8.9438% | -8.9497% | 15.2350% | -0.5874 | 52 | -22.98 | 0.2645 |

## Why it failed

- full-window CAGR/strict-MDD was 1.43, below 3;
- strict MDD was 17.96%, above 15%;
- the sign was not temporally stable: strong in 2021, negative in 2022;
- the one-day-delta mechanism control was better risk-adjusted at 1.88, so the
  five-operation residual did not add the preregistered mechanism value.

The one-day-delta control itself is not promotable from this result. It earned
`+49.7141%` absolute return, `22.3746%` CAGR, `11.8964%` strict MDD, ratio
`1.8808`, and 98 trades, but lost `-2.9846%` in 2022. It is falsification and
development evidence, not a replacement selected after viewing outcomes.

## Useful negative evidence

| Control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Direction flip | -47.6787% | -27.6826% | 53.1723% | -0.5206 | 111 |
| One-release delay | +4.8109% | +2.3789% | 20.1330% | 0.1182 | 111 |
| Hash-random side | +28.5708% | +13.3988% | 19.3277% | 0.6932 | 111 |

The exact direction flip loses heavily and a one-operation delay removes most
of the effect. This supports a same-release directional relation rather than a
generic weekday or long-beta artifact. However, 2021 concentration and 2022
failure show that the effect depends on the facility's macro regime and cannot
be used unconditionally.

## Isolation and no-repair boundary

- physically parsed market window: `[2021-01-01, 2023-01-01)`;
- market rows parsed: 210,240;
- funding rows parsed: 2,190;
- parser stopped before the first 2023 row;
- Stage2 rejects before its execution loader with
  `ORFR-1 Stage1 failed; 2023 remains sealed`;
- thresholds, direction, baseline, hold clock, and size remain unchanged;
- 2023 and all later outcomes remain sealed for ORFR-1.

## Integrity

- evaluator SHA-256:
  `8bd60256d065da1750c9852b7c7b47375ad1cd65842b4b8e256cc43b470a8567`;
- Stage1 JSON SHA-256:
  `57dcfc8d5cf945250f8e1ee18e95dc341d81c5dad372ead166c64ebc38e4d63d`;
- Stage1 manifest:
  `db7e3333913a0f2d1eb2c38fdca7144121b957ad980c25479c7267b8d3fce939`.

The next candidate must use a different mechanism. Any later ORFR-derived
research must explicitly treat 2021–2022 as development and use a new
sequential split; it may not relabel this failed Stage1 as OOS.
