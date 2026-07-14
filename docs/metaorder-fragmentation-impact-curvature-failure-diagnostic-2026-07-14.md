# MFIC v1 failure diagnostic — 2026-07-14

## Scope

This is post-hoc decomposition of the already opened 2020–2023 MFIC v1 outcomes. It does not repair, promote, invert, or retune MFIC. The 2024, 2025, and 2026 windows remain unopened.

- frozen rejection: `e3f2fb8`
- diagnostic: `results/metaorder_fragmentation_impact_curvature_failure_diagnostic_2026-07-14.json`
- diagnostic SHA256: `33a68398edf1ceab51a7df3b5b3c6b0487c8c55f63895a9718001a1843832727`
- flat round-trip account cost at `0.5x`: `5.9991 bp`
- equivalent underlying break-even move before variance/compounding: approximately `12 bp`

## Gross edge versus cost

| candidate | split | trades | mean underlying raw | mean account gross | mean account net | post-hoc inverted net |
|---|---|---:|---:|---:|---:|---:|
| fast | train 2020–2022 | 1,348 | -1.81 bp | -0.90 bp | -6.90 bp | -5.10 bp |
| fast | select 2023 | 218 | -1.30 bp | -0.65 bp | -6.65 bp | -5.35 bp |
| slow | train 2020–2022 | 1,392 | +1.13 bp | +0.57 bp | -5.43 bp | -6.56 bp |
| slow | select 2023 | 243 | +0.91 bp | +0.45 bp | -5.55 bp | -6.45 bp |

The fast signal has the wrong gross direction on average. The slow signal has a small positive gross edge in both train and 2023, but it captures less than one tenth of the required round-trip account cost. Direction inversion does not solve the problem: every inverted net mean also remains negative.

## Best stable subgroup is still untradeable

The most consistent subgroup is `mfic_slow / fade / long`:

| split | trades | account gross | account net | gross win rate |
|---|---:|---:|---:|---:|
| train 2020–2022 | 309 | +2.57 bp | -3.43 bp | 58.25% |
| select 2023 | 52 | +2.63 bp | -3.37 bp | 61.54% |

This is useful mechanistic evidence—some absorption/exhaustion behavior exists—but it is not an alpha after executable costs. Branch-pruning it would be an impermissible post-hoc MFIC repair, and even the selected subgroup is approximately `3.4 bp` short per trade at account level.

Other seemingly favorable cells are weaker or unstable:

- fast fade short: `+1.17 bp` gross in train and `+1.78 bp` in 2023, both net negative;
- slow continuation short: `-1.68 bp` gross in train but `+1.58 bp` in 2023;
- slow fade short: `+0.47 bp` gross in train but `-2.38 bp` in 2023.

## Root cause

MFIC v1 detects a contemporaneous microstructure state, but its 15–60 minute scheduled horizons do not convert that state into enough subsequent displacement. The signal fires roughly daily, while the mean underlying move is only about `1 bp` for the better slow candidate. Transaction cost, not insufficient leverage, dominates:

- leverage scales both expected PnL and notional-linked fees, so increasing leverage does not improve gross-edge-to-cost ratio;
- the large sample makes “not enough trades” implausible as the explanation;
- both 2023 halves lose, so one isolated regime is not responsible;
- exact inversion also loses after costs, so the signal is not merely sign-flipped alpha.

## Successor constraints

A successor must be a new preregistered mechanism, not MFIC v1 tuning. It should:

1. target rare state transitions whose expected underlying displacement can plausibly exceed `12 bp`;
2. use multi-hour consequence horizons only when a causal completion/transition event occurs, rather than blindly lengthening MFIC holds;
3. reduce turnover by requiring an event sequence, not a static threshold conjunction;
4. exploit currently unused aggregate-trade dimensions—arrival burstiness, notional tail concentration, underlying-trades-per-aggregate-event, signed event imbalance, and run-length structure;
5. preserve next-open execution, full split clocks, source-gap quarantine, strict MDD, train/selection rejection rules, and sealed OOS handling.

The leading independent direction is a **liquidity-vacuum/replenishment transition**: detect an event-time collapse in trade-arrival entropy and notional concentration, then wait for either failed replenishment (continuation) or confirmed refill plus flow reversal (reversion). That sequential completion condition is materially different from MFIC's same-window impact curvature.
