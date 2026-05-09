# Wave Trading Binance Futures Execution Bridge

This project keeps LLM inference/regime/risk decisions in `rllm` and delegates
Binance USDT-M Futures order execution to the sibling repository at
`/home/pakchu/workspace/wave_trading`.

## Boundary

- `rllm` owns: analyzer/trader inference, regime filter, risk overlay, final `LONG`/`SHORT`/`HOLD` decision.
- `wave_trading` owns: Binance REST client, leverage/margin setup, post-only maker orders, fill polling, position open/close, trade CSV logging.

The bridge is implemented in `execution/wave_execution.py`.

## Safety default

The bridge is dry-run by default. It will not place orders unless either:

```bash
RLLM_EXECUTION_DRY_RUN=false
```

or the CLI is called with `--live`.

Live mode requires Binance credentials:

```bash
# testnet by default
export BINANCE_TESTNET=true
export BINANCE_TESTNET_API_KEY=...
export BINANCE_TESTNET_API_SECRET=...

# live account
export BINANCE_TESTNET=false
export BINANCE_LIVE_API_KEY=...
export BINANCE_LIVE_API_SECRET=...
```

## One-shot smoke command

Dry-run:

```bash
PYTHONPATH=. uv run python -m execution.wave_execution \
  --dry-run \
  --signal LONG \
  --probability 0.7 \
  --current-close 100000 \
  --current-atr 500 \
  --signal-id rllm-smoke-001
```

Live/testnet order path:

```bash
RLLM_EXECUTION_DRY_RUN=false \
PYTHONPATH=. uv run python -m execution.wave_execution \
  --live \
  --signal LONG \
  --probability 0.7 \
  --current-close 100000 \
  --current-atr 500 \
  --signal-id rllm-live-001
```

## Runtime knobs

- `RLLM_WAVE_TRADING_PATH` default `/home/pakchu/workspace/wave_trading`
- `TRADING_SYMBOL` default `BTCUSDT`
- `TRADING_LEVERAGE` default `1`
- `POSITION_SIZE_PCT` default `0.10`
- `INTERVAL_MINUTES` default `5`
- `RLLM_EXECUTION_MAX_HOLDING_BARS` default `288`

## Implementation note

`wave_trading.trading.executor` imports its research-heavy signal generator at
module import time. The bridge injects a minimal signal-generator stub so this
repo can depend on the executor/client layer without pulling in wavelet research
dependencies. RLLM provides the final signal directly to
`TradingExecutor.handle_signal(...)`.
