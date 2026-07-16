# BTC difficulty/hashprice alpha preflight — 2026-07-16

## Hypothesis

Bitcoin difficulty-adjustment blocks create an event clock that is independent
of exchange OI, funding, order flow and price-bar thresholds. After six blocks
of confirmation, difficulty, prior-epoch duration, block subsidy and the BTC
price response over the completed epoch are immutable and causally available.

The exploratory features were difficulty change, prior-epoch BTC return,
subsidy-adjusted hashprice change and miner squeeze, plus prior-only rolling
standardizations. Both feature-tail direction mappings and event-gated BTC
momentum/reversal were tested with fixed 3/7/14-day holds.

## Protocol

- Blockstream Esplora adjustment-height blocks and height `+6` confirmation
  blocks; events before 2024 only.
- First completed 5-minute signal bar after the sixth confirmation; next-open
  fill.
- 0.5x, 6 bp per side, realized funding, full-calendar CAGR and intratrade
  strict MDD.
- 720 exploratory cells: eight features × three tails × ten mappings × three
  holds.
- Fit blocks: 2020H2, 2021, 2022; selection blocks: 2023H1 and 2023H2.
- 2024+ block events and performance were not opened.

## Best adequately populated candidate

Extreme subsidy-adjusted hashprice-change events, completed BTC 24-hour
momentum direction, 14-day hold:

| Window | Absolute return | CAGR/MDD | Trades |
|---|---:|---:|---:|
| 2020H2 | 8.1% | 0.82 | 4 |
| 2021 | 43.8% | 2.10 | 10 |
| 2022 | 31.8% | 2.62 | 10 |
| 2023H1 | 11.6% | 1.89 | 4 |
| 2023H2 | 5.0% | 0.78 | 5 |
| 2023 | 19.5% | 1.13 | 9 |

The exact ±1/3/7-day clock shifts did not maintain positive results across all
blocks, but the selected rule also failed the project's robustness target: its
minimum block CAGR/MDD was only `0.78`, and its feature sign was unused by the
price-momentum mapping. Economically it was close to a sparse calendar-gated
momentum rule rather than a miner-state alpha.

## Verdict

**Reject before OOS.** Do not spend the globally research-seen 2024–2026 window
on this 720-cell family. Keep difficulty-adjustment timing as a potentially
orthogonal future feature, but any retry needs a direction-specific miner
mechanism rather than another price-momentum gate.

Source API: <https://github.com/Blockstream/esplora/blob/master/API.md>
