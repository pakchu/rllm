# BFMWD-144 novelty comparator freeze — 2026-07-20

This amendment closes the comparator-registry field left abstract in the
BFMWD-144 preregistration. It was frozen before any Bitfinex source value,
candidate clock, BTC outcome, return, or PnL was parsed.

Only the `primary` clock from each frozen artifact is admissible:

| family | path | SHA-256 | common window |
|---|---|---|---|
| AMTR-48 | `data/authorized_minter_turnaround_relay_clocks_2020_2023.csv.gz` | `30875029daa4d6e2eff9a59f53d45eda57dbced05988df089c38a6c81abfa0f6` | 2021–2023 |
| CMSR-36 | `data/coinm_next_maturity_shock_relay_clocks_2020_2023.csv.gz` | `e81450d4e76ffd0ce2ae96edf97106f2f4c473da233be0db18dc2530c8da8e87` | 2021–2023 |
| CCIPA-48 | `data/cross_collateral_inventory_pressure_absorption_clock_2021_2023.csv.gz` | `a96d06ecda35fd7f0f75a8015ab907e280c4d4b8c06620a9da3d874adb6523f9` | 2021–2023 |
| SQFD-6 | `data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz` | `a81e144eea1e80ae5439fc66db1fad5bbd00cd9ac177e25142b5cfb5a07bcc5b` | 2023 source prefix only |
| DLPD-12 | `data/btcdom_leverage_polarity_decomposition_clocks_2022_2023.csv.gz` | `b33990f1629465caa837aa1f6f74430054b7185b68ece47b8c7540f9c11bf0fb` | 2022–2023 |
| CPR | `results/cross_collateral_positioning_recoil_clocks_2026-07-17.csv` | `2a864ec2b616a3118bf9ffa44f99f96fbe19e79d82870f21a0d7d9010d5c993a` | 2021–2023 |

The frozen entry field is `entry_time`, exact duplicates are removed after the
`control == primary` filter, and timestamps outside
`[2021-01-01, 2024-01-01)` are discarded. No alternative control may be
substituted after BFMWD clocks are observed.

For every BFMWD candidate/comparator pair with at least ten candidate and five
comparator entries in the common interval, both conditions must hold:

- exact-entry Jaccard `<= 0.10`; and
- both candidate-within-comparator and comparator-within-candidate containment
  over a symmetric ±6-hour window `<= 0.35`.

If a comparator has too few common entries, it is reported as insufficient and
does not count as evidence of novelty. At least four of the six comparators
must have sufficient common support, including CPR or CCIPA and at least one of
AMTR/SQFD. Failure retires BFMWD before market evaluation.
