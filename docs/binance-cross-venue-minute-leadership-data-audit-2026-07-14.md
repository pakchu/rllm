# Binance Spot↔USD-M minute-order data audit — 2026-07-14

## Verdict

**PASS for outcome-blind alpha preregistration.** The official 2020–2023 source
was built on a complete five-minute clock, all accepted rows satisfy the frozen
availability and denominator contract, and no 2024+ observation or outcome was
opened.

This verdict approves the data source only. It is not evidence that any
directional rule is profitable.

## Frozen artifacts

- Combined source:
  `data/binance_cross_venue_minute_leadership_btc_2020_2023/BTCUSDT_cross_venue_minute_leadership_5m_2020-01_2023-12.csv.gz`
- Combined SHA-256:
  `00ab6a55fc7bfeb3012584db5bc97a7d7b98dd995491acfd3f865c6bd41f92cc`
- Audit result:
  `results/binance_cross_venue_minute_leadership_audit_2026-07-14.json`
- Audit-result SHA-256:
  `ffe0124ac9c5c0c3f1d1c284b672618cf910dc16cae36e65c1efe79710f039af`
- Official archive/checksum specification:
  <https://github.com/binance/binance-public-data>

All 48 Spot and 48 USD-M monthly payloads were verified against their adjacent
official `.CHECKSUM` files. No raw ZIP remains on disk.

## Coverage

| Year | Expected 5m rows | Source-complete | Feature-valid |
|---:|---:|---:|---:|
| 2020 | 105,408 | 105,146 | 105,144 |
| 2021 | 105,120 | 104,889 | 104,889 |
| 2022 | 105,120 | 105,107 | 105,093 |
| 2023 | 105,120 | 105,063 | 104,729 |
| **Total** | **420,768** | **420,205** | **419,855** |

- Source-complete fraction: **99.8662%**
- Feature-valid fraction: **99.7830%**
- Quarantined rows: **913**
- Interval: `2020-01-01 00:00:00` through `2023-12-31 23:55:00`

The complete output grid has no missing five-minute timestamp. Source defects
are represented as quarantined rows rather than silently deleting clock time:
2,325 missing Spot minutes, zero missing USD-M minutes, 216 invalid Spot
minutes, and 243 invalid USD-M minutes.

## Structural checks

All audit checks passed:

- combined and monthly artifact hashes match the manifest;
- exact UTC five-minute grid, unique and monotonic;
- `feature_available_time_utc = trade_earliest_time_utc = date + 5m`;
- only within-bar lag edges `0→1`, `1→2`, `2→3`, `3→4` are accepted;
- reverse-time `later→earlier` descriptors exist only as placebo controls and
  also require four complete within-bar pairs;
- accepted rows have exactly four lagged pairs and finite feature values;
- quarantined rows contain no usable feature value and record explicit reasons;
- antisymmetric leadership and sign-agreement fields remain within `[-1, 1]`;
- all per-venue timing centroids remain within `[0, 1]`;
- response-difference, timing-difference, and basis-change identities reconcile;
- no label, target, reward, action, PnL, forward, or future column exists;
- manifest records `outcomes_opened=false` and the hard seal at `2024-01-01`.

The most common quarantine was a documented source-incomplete bar (549 rows).
The remaining denominator failures are primarily flat or zero-response paths;
they are intentionally failed closed rather than imputed.

## Resource footprint

- Retained source directory: approximately **247 MiB**
- Full rebuild: **1m 35s** wall time on this workstation
- Peak resident memory: approximately **1.0 GiB**
- WSL filesystem after build: approximately **295 GiB used**, below the user's
  300 GiB ceiling but close enough that redundant smoke/checkpoint artifacts
  should be removed.

## Next boundary

The next commit may preregister one support/novelty-selected alpha using the
frozen SHA above. It must not read return outcomes during support selection and
must include venue-swap, reversed-minute-order, simultaneous-only,
aggregate-only, stale, exact-flip, CSPR-overlap, and RIFT-overlap controls.
2024+ remains sealed until a frozen pre-2024 evaluator passes.
