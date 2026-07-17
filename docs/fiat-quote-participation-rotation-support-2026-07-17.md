# FQPR-3 support result (2026-07-17)

## Decision

**PASS the source-only support gate and advance only to Stage-1 train
outcomes.** No execution OHLC, funding, return, win rate, CAGR, or drawdown was
read in this work unit.

- frozen policy: `FQPR-3`
- preregistration commit:
  `b7f17846497fc9cd7e367a4c84d3ff3296996fb9`
- selected participation quantile: `Q = 0.65`
- selection source: 2021-2022 support only
- 2023 action: pass/reject the already selected `Q`; no fallback
- 2024-2026: sealed

The preregistration discloses that 2023 source-support density was seen during
pre-freeze design. It remains outcome-sealed. This replay therefore proves
deterministic clock construction and gate compliance, not a clean support
holdout and not profitability.

## Source isolation

The support builder reads only:

1. the committed machine-readable preregistration;
2. the checksum-frozen fiat-quote source panel;
3. that panel's source manifest.

It calls `validate_manifest(..., verify_sources=False)` and separately hashes
only the files above. A test instruments every support-side SHA read and fails
if a market-kline or funding path is touched.

Opened outcome sources: `[]`.

## Train-only Q selection

The grid is evaluated in descending `Q`. The first train-support passer is
selected; lower values cannot compete after that point.

| Q | Train entries | 2021 after warm-up | 2022 | Train support |
|---:|---:|---:|---:|:---:|
| 0.70 | 36 | 18 | 17 | FAIL |
| **0.65** | **44** | **23** | **20** | **PASS / selected** |
| 0.60 | 55 | 28 | 26 | PASS, ignored |
| 0.55 | 58 | 29 | 28 | PASS, ignored |
| 0.50 | 60 | 30 | 29 | PASS, ignored |

`Q=0.70` fails the frozen train and subperiod count floors. `Q=0.65` is the
highest train passer. No 2023 statistic participates in that choice.

### Selected train support

| Check | Result | Gate |
|---|---:|---:|
| Non-overlapping entries | 44 | >= 40 |
| 2021 after warm-up | 23 | >= 20 |
| 2022 | 20 | >= 18 |
| Maximum one-month share | 13.64% | <= 25% |
| EUR involvement | 40.91% | >= 30% |
| TRY involvement | 88.64% | >= 30% |
| BRL involvement | 75.00% | >= 30% |
| Largest participating-book set | 59.09% | <= 80% |

The split counts do not have to sum to the parent count: a trade crossing the
2021/2022 boundary remains in the parent train clock but is excluded from both
contained calendar subperiods.

All frozen train control-Jaccard limits pass. The largest are no-taker
`0.4333`, no-ticket `0.3913`, and single-BRL `0.3134`; BTCUSDT-only is `0.0361`
and the one-day delayed clock is `0.0000`.

## Unchanged 2023 support check

Only the already selected `Q=0.65` is evaluated.

| Check | Result | Gate |
|---|---:|---:|
| Non-overlapping entries | 28 | >= 20 |
| 2023 H1 | 17 | >= 8 |
| 2023 H2 | 11 | >= 8 |
| Maximum one-month share | 17.86% | <= 25% |
| EUR involvement | 82.14% | >= 30% |
| TRY involvement | 71.43% | >= 30% |
| BRL involvement | 96.43% | >= 30% |
| Largest participating-book set | 50.00% | <= 80% |

All frozen 2023 Jaccard limits pass. The largest are no-taker `0.6552`,
no-ticket `0.4524`, and single-EUR `0.3659`. BTCUSDT-only and the one-day delay
are both `0.0000`.

## Causal clock

For source day `d`:

1. ranks compare `d` only with exact complete days `d-180..d-1`;
2. the daily source becomes available after `d 23:59:59.999 UTC`;
3. the strategy waits through the complete next `00:00-00:05` bucket;
4. entry is `d+1 00:05 UTC`;
5. scheduled exit is 864 five-minute bars later;
6. false-to-true episodes and positions are globally non-overlapping;
7. split reports retain only signal, entry, and exit contained in that split.

The selected primary clock has 72 globally non-overlapping pre-2024 entries.
The combined source-only primary/control clock file has 1,127 split-contained
rows. No entry or exit crosses the exclusive 2024 boundary.

## Artifacts and replay

- builder:
  `training/build_fiat_quote_participation_rotation_support.py`
- tests:
  `tests/test_build_fiat_quote_participation_rotation_support.py`
- support result:
  `results/fiat_quote_participation_rotation_support_2026-07-17.json`
- clocks:
  `results/fiat_quote_participation_rotation_clocks_2026-07-17.csv`
- support JSON SHA-256:
  `635bfa69b0d10d6766bf65f8673e573e0b9614c8585723f90b32193fd7343d4b`
- clock CSV SHA-256:
  `54a70cce565d4f1727d095707471235f01345b94179a6c37df9f4c37d1a458a2`

Two consecutive full support builds were byte-identical.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m training.build_fiat_quote_participation_rotation_support

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m pytest -q -p no:cacheprovider \
  tests/test_build_fiat_quote_participation_rotation_support.py
```

## Next gate

Stage 1 may now parse execution OHLC and realized funding only through
2022-12-31. It must evaluate the unchanged `Q=0.65` primary and all frozen
controls under strict MDD. **Absolute return, CAGR, strict MDD, CAGR/MDD, and
trade count remain N/A here**, not zero.
