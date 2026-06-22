# Wave Trading Binance Futures Execution Bridge

This project keeps LLM inference/regime/risk decisions in `rllm` and delegates
Binance USDT-M Futures order execution to the sibling repository at
`/home/pakchu/workspace/wave_trading`.

## Boundary

- `rllm` owns: single-policy inference, regime/risk state interpretation, deterministic conversion to final `LONG`/`SHORT`/`HOLD` decision.
- `wave_trading` owns: Binance REST client, leverage/margin setup, post-only maker orders, fill polling, position open/close, trade CSV logging.

The bridge is implemented in `execution/wave_execution.py`.

## Safety default

The bridge is dry-run by default. It will not place orders unless either:

```bash
RLLM_EXECUTION_DRY_RUN=false
```

or the CLI is called with `--live`.

Live mode requires Binance credentials. The bridge first loads the sibling
`wave_trading/.env` file, then respects any environment variables already set in
the current process. Do not print or commit credential values.

```bash
# testnet by default; values may come from /home/pakchu/workspace/wave_trading/.env
export BINANCE_TESTNET=true

# live account; requires live keys in environment or wave_trading/.env
export BINANCE_TESTNET=false
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
`TradingExecutor.handle_signal(...)`. The deprecated analyzer/trader cascade must not be
revived in the live execution path; any LLM output should first be reduced to the
single compact policy contract and then converted deterministically to an executor signal.
