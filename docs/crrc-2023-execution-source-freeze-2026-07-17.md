# CRRC-72 2023 execution-source freeze — 2026-07-17

- Outcome return/PnL calculated: **no**
- Market rows: `105120`
- Funding rows: `1095`
- Maximum timestamp exclusive: `2024-01-01 00:00:00`
- 2024 rows parsed or written: **0**
- Exact funding timestamps preserved: **yes**
- Manifest hash: `8ac689f0ca7024b3b3a748981647a14947fa2b4bab75768b88f5e1b1c730b3bc`

The exporter reads only the physically pre-2024 source tails and writes a
dedicated calendar-2023 BTCUSDT OHLC file plus exact millisecond funding
settlement timestamps/rates. It constructs no signal, return, label, equity,
CAGR, MDD, or PnL. The CRRC evaluator must load only these two outputs.
