# LORE v1 outcome-blind support freeze — 2026-07-17

## Decision

**PASS support only.** All four preregistered policies have sufficient causal
event incidence, year/half coverage, symbol breadth, pair diversity, and source
quality to advance to a separately frozen 2023–2024 selector. No post-entry
return, PnL, CAGR, MDD, 2025 row, or 2026 row was calculated.

- preregistration protocol hash:
  `18480ed99902cecc126fcd4e5d9f5df40c98e65878bfecfb547e2941084be840`
- source manifest hash:
  `1c54fddc45fcc516d8ce42741904e018e8d00e646eff40be514273cf10eee7ed`
- support manifest hash:
  `1dc91c0775825a6bcbc76ba8956639e020bdcf5a59d6188fd3d06235f8ce177e`
- frozen clock SHA-256:
  `76c0d78c7c703dc16145a5ff86a32700afe77c8ecce46b0d5042afc3ead5135c`
- frozen support-feature SHA-256:
  `9dba4f34a53ed3efc17dc4e25c9011a490ece6fdbf495fedd366f7cba14c978f`

## Support statistics

| Policy | Residual/hold | Events | 2023 | 2024 | Half-years | Ordered pairs | Max pair share | Long/short symbols | Pass |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| L01 | 6h/12h | 284 | 127 | 157 | 41/86/58/99 | 30 | 5.28% | 6/6 | PASS |
| L02 | 6h/24h | 214 | 95 | 119 | 34/61/46/73 | 30 | 5.61% | 6/6 | PASS |
| L03 | 12h/12h | 224 | 107 | 117 | 37/70/47/70 | 30 | 6.25% | 6/6 | PASS |
| L04 | 12h/24h | 163 | 77 | 86 | 27/50/35/51 | 30 | 5.52% | 6/6 | PASS |

Half-year counts are ordered `2023H1 / 2023H2 / 2024H1 / 2024H2`.

## Source quality

- exact 2023–2024 hourly opportunities: `17,543`;
- all-six-symbol clean hours: `17,539`;
- quarantined hours: `4`;
- global quarantine: `0.0228%`;
- maximum monthly quarantine: `0.4032%`, below the frozen `1%` ceiling;
- 2025+ rows in clocks: `0`.

The 19 zero-volume five-minute bars per symbol were preserved as valid source
facts but made their four affected joint hours ineligible. They were not filled,
interpolated, or used to release a later event during an active reservation.

## Boundary

The next step is to commit a selector that replays these exact clocks with
per-symbol OHLC, exact funding, two-leg costs, full-calendar CAGR, and the
preregistered strict multi-leg MDD. If no policy passes, LORE v1 ends before
2025 is opened.
