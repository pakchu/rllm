# Continual DVOL + routed REX complement alpha (2026-07-13)

## Verdict

The chronological single-position combination improved 2024 and 2026, but did
not repair 2025.  No alpha or live candidate was promoted.

## Protocol

- Components: frozen positioning + DVOL Top-10 and frozen regime-routed REX
  Top-10 manifests.
- Combination selection: 2023 full/H1/H2 only; 200 combinations.
- Execution: one position at a time, with either ordinary union scheduling or
  causal early exit of a DVOL long when a REX short signal arrives.
- Combined pre-future manifest hash:
  `4355a9a4a880bc364bbb10a877c90e574dce4049ae1c3900ff3cb5c46723c7fe`
- Full market and future candidate files are opened after the combined manifest.
- Costs: 0.5x and 6 bp per side.
- CAGR: full calendar window including idle time.
- strict MDD: favorable-to-adverse intraposition high-water path.

## Best selected combination

- DVOL policy index 4: 48-hour continual long critic.
- REX routed pair index 0.
- Mode: REX-short preempts an open DVOL long.

| Period | Absolute return | CAGR | strict MDD | CAGR/MDD | Trades | Sources | Preemptions |
|---|---:|---:|---:|---:|---:|---|---:|
| 2023 Select | +34.80% | 34.82% | 6.19% | 5.63 | 69 | REX 37 / DVOL 32 | 2 |
| 2024 Test | +36.42% | 36.33% | 8.34% | 4.36 | 67 | REX 31 / DVOL 36 | 0 |
| 2025 Eval | +3.18% | 3.18% | 12.50% | 0.25 | 58 | REX 16 / DVOL 42 | 2 |
| 2026 YTD | +11.59% | 30.15% | 6.39% | 4.72 | 31 | REX 11 / DVOL 20 | 3 |

## Interpretation

The components are complementary outside 2025, but the sparse routed REX
critic does not fire often enough during the DVOL policy's 2025 drawdown.  Only
two preemptions occurred, so the loss path remained nearly unchanged.  More
priority tuning on this event set would not create missing short-side edge.

The next justified audit is the independently discovered REX8640-width / USDKRW
gate that was selected through 2024 and originally evaluated on later data.  It
must be replayed with the corrected intraposition high-water MDD before it can
be considered as a denser bearish sleeve.

## Artifacts

- Search: `training/search_dvol_rex_complement_alpha.py`
- Tests: `tests/test_search_dvol_rex_complement_alpha.py`
- Manifest: `results/dvol_rex_complement_top10_manifest_2026-07-13.json`
- Result: `results/dvol_rex_complement_alpha_scan_2026-07-13.json`
