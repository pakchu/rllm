# EM-FX coherent pressure reversal — frozen OOS result

The pre-2024 policy and manifest
`3adb3d041f2096d51effc899319ee5653b28f4a49e003ba7544e8ba2a8b60910`
were committed before this family opened 2024+ BTC outcomes. The pre-2024
market, funding, EM-FX and feature prefixes, schedules, execution economics and
statistics reproduced exactly. No threshold, direction or holding-period
parameter was changed.

## Frozen policy

- signal: the upper/lower 20% tails of prior-only standardized one-session
  AUD, CNY, INR and MXN common pressure;
- direction: fade the already-completed BTC 24-hour move;
- hold: seven days without overlap;
- execution: complete UTC FX day plus five minutes, one completed 5-minute
  signal bar, next-open fill, 0.5x, 6 bp per side, realized funding,
  full-calendar CAGR and intratrade strict MDD.

## Frozen result

| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| test 2024 | -34.2143% | -34.1578% | 39.8386% | -0.8574 | 27 | 14/13 |
| eval 2025 (partial source) | 2.2657% | 2.2673% | 7.7560% | 0.2923 | 4 | 2/2 |
| holdout 2026H1 (not evaluable) | 0.0000% | 0.0000% | 0.0000% | 0.0000 | 0 | 0/0 |
| OOS 2024–2026H1 (partial source) | -32.7238% | -15.1219% | 40.5497% | -0.3729 | 31 | 16/15 |
| all 2021–2026H1 | 111.6722% | 14.8513% | 39.2363% | 0.3785 | 130 | 68/62 |

At doubled transaction cost, combined OOS absolute return is `-33.9641%`.
The 31 OOS trades have approximate t statistic `-1.8026` and p value
`0.0714`; these remain descriptive post-selection diagnostics.

## Source-coverage failure

The fixed quote-count contract yields `207` complete sessions in 2024, only
`16` in 2025 and `0` in 2026. Complete fixed-panel sessions end on
`2025-02-10` because the local `USDCNY` feed collapses from hundreds of
observations per session to roughly one or two. Raw per-symbol date coverage is
therefore not sufficient; the OOS runner now fails the coverage gate unless a
complete session reaches at least `2026-05-29`.

The database snapshot is also a historical backfill rather than a point-in-time
capture. Backfill use is opt-in, frozen in the manifest and explicitly blocks
promotion until a live point-in-time forward window exists.

## Verdict

**Rejected.** The fully covered 2024 test already loses `34.21%` with
`39.84%` strict MDD. The incomplete later source cannot rescue or validate the
family. Do not invert, retune or rerank this EM-FX family on the consumed
2024–2026 windows. Trade/PnL orthogonality was intentionally not tested because
the frozen performance and coverage gates failed first.
