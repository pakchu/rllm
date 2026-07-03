# REX+LLM Binance live bridge (bear-regime pilot)

## Intent

The current REX+LLM structure appears useful mainly when the long timeframe regime is bearish.  The live bridge therefore treats the long-regime call as a **manual operator input** and only lets model decisions reach Binance execution when the manual regime gate permits it.

## Boundary

- RLLM owns: REX candidate generation, LLM/gate decision (`TRADE` -> candidate side, `ABSTAIN` -> `HOLD`), manual regime gate, and risk policy.
- `../workspace/wave_trading` owns: Binance USDT-M futures client, post-only maker order mechanics, leverage/margin initialization, and position handling.
- This repo imports only the execution classes from `wave_trading`; it does not reuse that repo's research signal generator.

## Current safety defaults

Config: `configs/live/rex_llm_binance_testnet_bear_pilot.json`

- `dry_run: true`: no order call is made.
- `allow_live_orders: false`: even if `--live` is passed, execution is blocked until explicitly changed.
- `testnet: true`: testnet endpoint is selected by default.
- `manual_regime: UNKNOWN` + `require_bear_regime: true`: trades are blocked until the operator sets `BEAR`.
- `allowed_signals: ["SHORT"]`: initial bear pilot is short-only.  This can be widened to `LONG,SHORT` after testnet replay confirms long entries are intentional.
- `position_size_pct: 0.02`, `leverage: 1`: small pilot sizing.

## CLI dry-run smoke tests

Blocked by manual regime:

```bash
python -m execution.wave_execution \
  --config configs/live/rex_llm_binance_testnet_bear_pilot.json \
  --signal SHORT --probability 0.8 --current-close 100000 --current-atr 500 \
  --signal-id dryrun-blocked --dry-run
```

Approved dry-run after manually editing/copying config with `manual_regime: "BEAR"`:

```bash
python -m execution.wave_execution \
  --config /path/to/bear-config.json \
  --signal SHORT --probability 0.8 --current-close 100000 --current-atr 500 \
  --signal-id dryrun-approved --dry-run
```

Testnet live orders are still blocked unless **both** are true:

1. config/env has `dry_run=false`
2. config/env has `allow_live_orders=true`

Do not enable those until the dry-run event stream has been checked against expected REX+LLM decisions.

## Credential handling

The bridge can load `BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_API_SECRET` from `../workspace/wave_trading/.env`, but it never prints secret values.  Live mainnet requires `BINANCE_LIVE_API_KEY` / `BINANCE_LIVE_API_SECRET` and should remain disabled until separate testnet verification is complete.

## Next live-integration steps

1. Wire the frozen REX+LLM scorer to produce `ExecutionDecision` objects from the latest closed 5m candle.
2. Run `--dry-run` continuously for at least several days in manually confirmed BEAR regime.
3. Compare dry-run signals with the offline backtest event definitions: no current candle leakage, no duplicate entries, no stale decisions.
4. Only then enable testnet orders with `allow_live_orders=true` and keep `allowed_signals=["SHORT"]` for the first pilot.

## Pre-scored REX+LLM record mapping

The bridge now accepts a pre-scored policy JSON object.  This is the intended boundary for the next scorer step:

```json
{
  "prediction": "TRADE",
  "action": {"side": "SHORT"},
  "current_close": 100000,
  "current_atr": 500,
  "signal_id": "latest-closed-candle",
  "probability": 0.8
}
```

Mapping contract:

- `prediction/decision/target = TRADE` + candidate `side=SHORT` -> execution signal `SHORT`.
- `prediction/decision/target = TRADE` + candidate `side=LONG` -> execution signal `LONG`, then still subject to `allowed_signals`.
- `ABSTAIN`, `NO_TRADE`, `BLOCK`, `HOLD`, or unknown label -> execution signal `HOLD`.
- The LLM does **not** choose long/short; it only allows or blocks the REX candidate side.

Smoke command:

```bash
python -m execution.wave_execution \
  --config /path/to/bear-config.json \
  --policy-record-json '{"prediction":"TRADE","action":{"side":"SHORT"},"current_close":100000,"current_atr":500,"signal_id":"record-smoke","probability":0.8}' \
  --dry-run
```
