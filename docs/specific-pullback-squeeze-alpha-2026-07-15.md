# Specific pullback-squeeze weak-signal alpha — 2026-07-15

## Verdict

Promote the rule below to the **research/shadow alpha-candidate pool**. It is a
specific interaction of weak signals, not a linear blend. It clears
`CAGR / strict MDD >= 3` in train, 2023 selection, 2024 test, and the combined
2025-2026 evaluation. It also survives 10 bp per side and a two/three-bar entry
delay.

It is **not live-grade proof**. The complete 2020H2-2026 diagnostic ratio is
`2.65`, the programme has already inspected the later years, and this rule came
from a 29,133-member pre-2024 interaction search. Fresh shadow/forward evidence
is still mandatory.

## Rule

All thresholds below were fitted only on hourly decision points from
2020-07-01 through 2022-12-31.

### Opportunity legs

Funding/trend leg:

```text
funding_available
and funding_rate <= q10 = -0.00002222
and trend_96 >= q70 = 0.009017208457522975
```

Premium/momentum leg:

```text
premium_available
and premium_index_change <= q20 = -0.00026817
and completed htf_1d_return_4 >= q90 = 0.09605902316678483
```

### Context interaction

```text
common pullback:
  rex_576_range_pos <= q60 = 0.4633067898243715

funding leg overheat veto:
  completed htf_1d_return_1 <= q70 = 0.028038610394397256

premium leg overheat veto:
  completed htf_3d_return_1 <= q70 = 0.045815363295544476

entry = common pullback
        and (
          funding/trend leg and funding overheat veto
          or
          premium/momentum leg and premium overheat veto
        )
```

Interpretation: depressed funding or premium plus strong medium-horizon trend
creates the squeeze opportunity. The rule waits until price is back in the
lower 60% of its trailing 48-hour range and rejects a funding event after an
overextended completed daily move or a premium event after an overextended
completed three-day move. The edge is therefore **momentum/squeeze conditioned
on a non-overheated pullback**, not unconditional momentum chasing.

## Frozen execution contract

- signal: completed 5-minute bar, evaluated hourly;
- entry: next 5-minute open;
- exit: fixed 576 bars / 48 hours;
- one non-overlapping long position;
- leverage: 0.5x;
- cost: 5 bp fee + 1 bp slippage per side;
- CAGR: full named wall-clock period, including idle time;
- strict MDD: pre-entry equity high water plus favorable-then-adverse intrabar
  marking over every held bar;
- any trade whose exit crosses a split boundary is purged;
- funding and premium availability are mandatory, so stale/missing auxiliary
  values cannot create an entry.

## Strict results

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | Approx. mean-return p |
|---|---:|---:|---:|---:|---:|---:|
| Train 2020H2-2022 | +105.72% | 33.41% | 9.65% | **3.46** | 83 | 0.00164 |
| Select 2023 | +17.90% | 17.92% | 3.15% | **5.68** | 18 | 0.0921 |
| Test 2024 | +21.99% | 21.94% | 5.24% | **4.19** | 16 | 0.0000104 |
| Eval 2025-2026-05-31 | +30.13% | 20.51% | 6.21% | **3.30** | 35 | 0.00620 |
| 2024-2026-05-31 combined | +58.74% | 21.10% | 6.21% | **3.40** | 51 | 0.0000155 |
| Full 2020H2-2026-05-31 diagnostic | +285.04% | 25.60% | 9.65% | **2.65** | 152 | 0.000000630 |

Annual strict ratios are positive and above 3 for each evaluated year:

- 2020H2: `7.94`, 13 trades;
- 2021: `3.71`, 26 trades;
- 2022: `3.06`, 43 trades;
- 2024: `4.19`, 16 trades;
- 2025: `4.27`, 13 trades;
- 2026 through May 31: `4.40`, 22 trades.

2023H2 is the weak subwindow: +2.52% absolute return, ratio `1.85`, 10 trades.
The full 2023 selection year remains positive and passes the target.

## Stress and robustness

### 10 bp per side

| Window | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades |
|---|---:|---:|---:|---:|---:|
| Train | +99.00% | 31.65% | 10.01% | **3.16** | 83 |
| Select 2023 | +17.06% | 17.07% | 3.27% | **5.22** | 18 |
| Test 2024 | +21.21% | 21.16% | 5.28% | **4.01** | 16 |
| Eval 2025-2026 | +28.32% | 19.32% | 6.40% | **3.02** | 35 |
| OOS 2024-2026 | +55.54% | 20.08% | 6.40% | **3.14** | 51 |

Entry delayed by two and three bars also leaves train, selection, test, and
evaluation ratios above 3. This argues against a single-open fill artifact.

The 48-hour hold is structural rather than arbitrary robustness: 36 hours
reduces eval ratio to `1.18`; 60 hours reduces it to `2.63`. Do not retune the
hold from later periods.

Nearby threshold recipes are mixed but not needle-like. The fixed q60/q70 rule
passes every primary split. q60/q65 still gives train `3.12`, 2024 `3.84`, and
eval `3.05`; q55/q70 gives train `3.47`, 2024 `4.59`, and eval `2.90`.

Four-trade moving-block bootstrap mean-return intervals remain above zero:

- train 95% interval: `+0.410%` to `+1.477%` per trade;
- 2024 test: `+0.767%` to `+1.724%`;
- 2025-2026 eval: `+0.311%` to `+1.199%`;
- combined 2024-2026: `+0.578%` to `+1.338%`.

These intervals are descriptive. They do not correct the full research
multiplicity.

## Why the combination matters

Neither source leg is sufficient alone under identical context:

| Ablation | Train ratio / trades | 2024 ratio / trades | 2025-26 ratio / trades | OOS ratio / trades |
|---|---:|---:|---:|---:|
| Funding only | 1.83 / 55 | 1.24 / 7 | 2.24 / 31 | 1.66 / 38 |
| Premium only | 1.68 / 29 | 4.00 / 9 | 1.05 / 4 | 2.18 / 13 |
| Conditional union | **3.46 / 83** | **4.19 / 16** | **3.30 / 35** | **3.40 / 51** |

The two branches cover different regimes, and non-overlap scheduling changes
which subsequent opportunities remain executable. The result supports the
hypothesis that multiple weak, mechanism-compatible signals can form an edge
even when each branch is inadequate alone.

## Leakage and live-gap audit

- `rex_576_range_pos` is calculated from trailing/current bars only.
- `htf_1d_return_*` and `htf_3d_return_*` use completed, shifted higher-timeframe
  candles; the in-progress higher-timeframe candle is excluded.
- Funding and premium are backward-as-of joined with explicit availability and
  tolerance limits. Premium uses its kline close timestamp.
- Threshold fitting ends before 2023; 2024+ rows cannot change the rule.
- The simulator enforces t+1 entry, complete split-contained exits, costs,
  non-overlap, and strict bar-level MDD.

Remaining gaps:

1. Actual funding cash flows during the 48-hour position are not included.
2. The broader programme has already viewed 2024-2026, so those periods are not
   pristine global OOS despite this search freezing a Top-20 family first.
3. The full six-year diagnostic ratio is `2.65`, below the global target of 3.
4. Selection 2023 contains only 18 trades. Aggregate evidence is meaningful,
   but per-year samples remain small.

## Artifacts

- `training/search_specific_pullback_squeeze_alpha.py`
- `tests/test_search_specific_pullback_squeeze_alpha.py`
- `results/specific_pullback_squeeze_alpha_audit_2026-07-15.json`

Reproduce:

```bash
.venv/bin/python training/search_specific_pullback_squeeze_alpha.py
.venv/bin/python -m pytest -q tests/test_search_specific_pullback_squeeze_alpha.py
```
