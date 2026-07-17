# Cross-collateral positioning recoil (CCPR-1) preregistration — 2026-07-17

## Outcome boundary

This work unit freezes one new alpha family before opening any executable BTC
return. The source-only density scan joined no OHLC, funding, future label,
portfolio return, CAGR, or drawdown. Therefore performance metrics are **N/A**.

The physical source is the checksum-audited Binance USD-M `BTCUSDT` and COIN-M
`BTCUSD_PERP` five-minute metrics panel documented in
`docs/binance-cross-collateral-positioning-metrics-source-audit-2026-07-17.md`.
The experiment uses no price, funding, premium, REX, Kimchi/FX, Upbit, order
book, options, CFTC, or on-chain signal input.

## Mechanism

USD-margined and coin-margined perpetuals expose the same BTC direction through
different collateral constituencies. CCPR-1 treats an unusually large six-hour
USD-M-versus-COIN-M OI rotation that agrees with an unusually large one-hour
taker-ratio gap as a collateral-specific leveraged crowding burst. The frozen
action fades that burst rather than following it.

This is structurally different from the current portfolio:

- `oi_upbit_ratio288_low` combines one futures OI stream with Korean spot flow;
- `new_long_minimal_funding_premium` uses carry state;
- REX sleeves use price action and taker/range state;
- Fresh Kimchi uses KRW premium and FX;
- Markov and Rank7 sleeves use price/derived state.

Economic orthogonality is not presumed. It is tested only if CCPR-1 first
passes its standalone sealed evaluation.

## Frozen feature and clock

At each UTC hourly `:55` source row `t`:

1. `R[t]` is the six-hour log OI growth of USD-M notional minus the six-hour
   log OI growth of COIN-M contract count.
2. `T[t]` is the median over the final twelve complete five-minute rows of
   `log(USD-M taker ratio) - log(COIN-M taker ratio)`.
3. `A[t]` and `G[t]` are strict-prior empirical mid-ranks of `abs(R[t])` and
   `abs(T[t])` against exactly 168 immediately prior hourly anchors.
4. Setup: `A[t] >= Q`, `G[t] >= 0.60`, and `sign(R[t]) = sign(T[t]) != 0`.
5. Only a false-to-true transition creates an episode.
6. Side: `-sign(T[t])`.
7. The metrics timestamp waits one complete availability bucket; entry is the
   next five-minute open at `t+10m`.

No gap is filled. A gap invalidates the current 73-row OI path and all 168
strict-prior rank anchors, so signals remain quarantined until the full causal
history is rebuilt.

## Frozen support and outcome sequence

- Support-only grid: `Q in {0.80, 0.85, 0.90}`.
- Select the highest Q passing all frozen density, temporal coverage,
  side-balance, concentration, and component-overlap floors.
- Source-only preflight episode counts:

| Q | 2021 partial | 2022 | 2023 support-seen/outcome-sealed |
|---:|---:|---:|---:|
| 0.80 | 47 | 100 | 66 |
| 0.85 | 35 | 78 | 49 |
| 0.90 | 24 | 51 | 34 |

At `Q=0.90`, 2023H2 has only nine episodes and fails the frozen half-year
support floor of ten. This is a source-density decision, not a return decision.

After support selection, only two candidates exist:

- `CCPR-H4`: 48 five-minute bars;
- `CCPR-H8`: 96 five-minute bars.

Stage1 is physically restricted to `[2021-07-08, 2023-01-01)`. The 2023
execution window stays sealed unless a Stage1 candidate passes every gate.

## Mandatory gates

Stage1 requires positive absolute return, full-calendar CAGR/strict-MDD at
least 3.0, strict MDD at most 15%, at least 80 trades, two-sided weekly-cluster
sign-flip `p <= 0.025`, positive return in 2021 partial, 2022H1, and 2022H2,
and positive 10bp-stress return with stress ratio at least 2.5. The primary
ratio must beat each OI-only, taker-only, USD-M-only, and COIN-M-only mechanism
control by at least 0.25.

Every control receives the complete profitability, risk, significance,
trade-count, stress, and subperiod battery. Direction flip, deterministic
random side, and a one-hour entry shift are also frozen.

## Orthogonality gate

Only a standalone passer is compared with the hash-frozen deduplicated alpha
universe and current selected portfolios. Required limits include exact-entry
Jaccard at most 0.02, six-hour-near-entry fraction at most 0.20, position-time
Jaccard at most 0.15, and absolute daily-PnL correlation at most 0.30, followed
by a positive marginal portfolio contribution test.

## Artifacts

- Preregistration code:
  `training/preregister_cross_collateral_positioning_recoil.py`
- Frozen JSON:
  `results/cross_collateral_positioning_recoil_preregistration_2026-07-17.json`
- Tests:
  `tests/test_preregister_cross_collateral_positioning_recoil.py`

If support fails, no outcome is opened. If Stage1 fails, 2023 remains sealed.
Threshold, direction, horizon, or extra-gate repair after outcomes is forbidden.
