# DCRM-1 2023 execution-source freeze — 2026-07-17

- Outcome return/PnL calculated: **no**
- Market rows written per symbol: `105120`
- Funding rows written per symbol: `1095`
- Maximum timestamp exclusive: `2024-01-01 00:00:00`
- 2024 rows parsed or written: **0**
- Symbols: `['ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT']`
- Manifest hash: `95d875bc1a4c56027fb75b3cff1abe3f496a08680c472844277a2df23f3b2d15`

The exporter reads fixed row counts from the already-frozen 2023–2024 source
and writes physically separate 2023-only files. It does not hash or parse the
combined source beyond those prefixes and calculates no return, PnL, label,
equity, or drawdown. The 2023 evaluator is required to read only these files.
