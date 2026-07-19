# IFAR-288 support rejection — 2026-07-20

Decision: **rejected before any post-decision execution bar, return, funding
mark, PnL, CAGR, or strict MDD was loaded**.

The exact insurance-loss threshold, completed-price threshold, one-day source
embargo, reversal side, 24-hour hold, calendar requirements, and source hashes
were frozen in commits `02944eb` and `9559a2a` before incidence was calculated.

## Outcome-blind support result

| Window | Events | Frozen minimum | Result |
|---|---:|---:|---|
| 2020H2–2022 total | 25 | 50 | fail |
| 2020H2–2021 train | 17 | 30 | fail |
| 2020H2 | 0 | 8 | fail |
| 2021 | 17 | 20 | fail |
| 2022 test | 8 | 20 | fail |
| 2022 H1 | 2 | 8 | fail |
| 2022 H2 | 6 | 8 | fail |

All, train, and test side-balance gates passed:

- all: 48.00% long / 52.00% short;
- train: 41.18% long / 58.82% short;
- test: 62.50% long / 37.50% short.

The candidate was nevertheless too sparse and uneven. There were no eligible
events in either 2020 quarter, only one in each of `2022Q1` and `2022Q2`, and
the largest quarter represented 28% of all events versus the frozen 20% cap.
Only the three side-balance checks passed; total, train/test incidence, both
test halves, per-quarter support, and quarter concentration failed.

No event clock was written.

## Physical source audit

- BitMEX insurance rows parsed: 1,826;
- completed Binance daily snapshot closes parsed: 1,096;
- eligible entry days: 914;
- eligible days with both strictly prior thresholds ready: 805;
- candidate days: 25;
- Binance non-date fields parsed outside the 11:55 daily source row: 0;
- funding rows loaded: 0;
- rows used to construct a post-decision execution or outcome: 0;
- rows at or after `2023-01-01` loaded: 0.

The threshold-ready count shows that this is not merely a warm-up artifact.
Daily net default-fund losses that also coincide with a material directional
BTC move are intrinsically too sparse for the requested standalone validation.

## Integrity anchors

- support result hash:
  `085fffd6addbd0e450bc70605e5c04e844d7b05b4c44cb916bff5b5a21959630`;
- support JSON SHA-256:
  `5b32c92bf79abcf031698e87f155c19f8b0398fb4d85df591943a1f156143a02`;
- protocol hash:
  `a68ade9d78a86f57d9873eb5050d75c046340c9752e7419d21fa8720cec919e2`;
- event-clock hash (not written):
  `d826f11d71e39cee8a22359e059a286d670bc2c4220663eb4655c18de21c1003`;
- source manifest hash:
  `4c751b96a4d877bc558bf37e693396fc529326feb050862ea5ddb9100cde8612`;
- private source SHA-256:
  `523d179d4a4ac51e3ebf5ce24f188f23cda02f31d8f879e0d256361af333c6dc`.

## Decision boundary

Lowering the insurance-loss threshold, weakening the price confirmation,
removing the one-day publication embargo, admitting 2023+, or relaxing the
calendar gate after seeing these counts would be a post-hoc repair. The frozen
contract prohibits all of them.

IFAR-288 is not eligible for a return evaluator, 2023+ source download, alpha
registration, or portfolio promotion. The insurance series may be retained as
a future risk-regime or sizing input only after an independently specified
portfolio-level study; this rejected singleton may not be relabeled as alpha.
