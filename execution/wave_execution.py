"""Bridge RLLM policy signals to the wave_trading Binance futures executor.

This module intentionally keeps the boundary narrow: RLLM owns model inference,
regime filtering, and risk-overlay decisions; wave_trading owns Binance REST
client details and maker-order execution.  Live order placement is opt-in.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
import types
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal, Optional

TradeSignal = Literal["LONG", "SHORT", "HOLD"]

DEFAULT_WAVE_TRADING_PATH = "/home/pakchu/workspace/wave_trading"


def load_env_file(path: str | Path, *, override: bool = False) -> dict[str, str]:
    """Load simple KEY=VALUE lines from a dotenv file without printing secrets.

    This avoids adding python-dotenv as a dependency while allowing the bridge
    to reuse the sibling wave_trading repo's existing credential file.
    Values already present in the process environment win unless override=True.
    """

    env_path = Path(path).expanduser()
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


@dataclass(frozen=True)
class WaveExecutionConfig:
    """Configuration for the wave_trading execution bridge."""

    wave_trading_path: str = DEFAULT_WAVE_TRADING_PATH
    symbol: str = "BTCUSDT"
    testnet: bool = True
    leverage: int = 1
    position_size_pct: float = 0.10
    maker_offset_pct: float = 0.0001
    max_retries: int = 3
    order_timeout_sec: int = 60
    dry_run: bool = True
    interval_minutes: int = 5
    # Values consumed by wave_trading.TradingExecutor for position tracking if
    # caller chooses to use its trailing/max-holding helpers.
    atr_period: int = 15
    pt_mult: float = 3.75
    max_holding_bars: int = 288

    @classmethod
    def from_env(cls) -> "WaveExecutionConfig":
        def env_bool(name: str, default: bool) -> bool:
            return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "y", "on"}

        return cls(
            wave_trading_path=os.environ.get("RLLM_WAVE_TRADING_PATH", DEFAULT_WAVE_TRADING_PATH),
            symbol=os.environ.get("TRADING_SYMBOL", "BTCUSDT"),
            testnet=env_bool("BINANCE_TESTNET", True),
            leverage=int(os.environ.get("TRADING_LEVERAGE", "1")),
            position_size_pct=float(os.environ.get("POSITION_SIZE_PCT", "0.10")),
            maker_offset_pct=float(os.environ.get("MAKER_OFFSET_PCT", "0.0001")),
            max_retries=int(os.environ.get("MAX_RETRIES", "3")),
            order_timeout_sec=int(os.environ.get("ORDER_TIMEOUT_SEC", "60")),
            dry_run=env_bool("RLLM_EXECUTION_DRY_RUN", True),
            interval_minutes=int(os.environ.get("INTERVAL_MINUTES", "5")),
            atr_period=int(os.environ.get("RLLM_EXECUTION_ATR_PERIOD", "15")),
            pt_mult=float(os.environ.get("RLLM_EXECUTION_PT_MULT", "3.75")),
            max_holding_bars=int(os.environ.get("RLLM_EXECUTION_MAX_HOLDING_BARS", "288")),
        )


@dataclass(frozen=True)
class ExecutionDecision:
    """A model/risk-layer decision ready for execution."""

    signal: TradeSignal
    probability: float
    current_close: float
    current_atr: float
    signal_id: str
    reason: str = ""


class _StaticSignalGenerator:
    """Minimal object satisfying wave_trading.TradingExecutor expectations."""

    def __init__(self, *, atr_period: int, pt_mult: float, max_holding_bars: int) -> None:
        self._best_params = {
            "atr_period": int(atr_period),
            "pt_mult": float(pt_mult),
            "holding_period": int(max_holding_bars),
        }

    @property
    def BEST_PARAMS(self) -> dict[str, Any]:
        return self._best_params

    def generate_signal(self, *args: Any, **kwargs: Any) -> tuple[TradeSignal, float, None]:
        # RLLM policy inference happens outside wave_trading.  This is present
        # only to satisfy the executor constructor; bridge calls handle_signal
        # directly with an already-produced decision.
        return "HOLD", 0.5, None


def _install_signal_generator_stub() -> None:
    """Avoid importing wave_trading's research-heavy SignalGenerator.

    wave_trading.trading.executor imports `.signal_generator` at module import
    time.  The real module pulls in polars/PyWavelets/sklearn/research code,
    which is unnecessary for execution when RLLM supplies the signal.  A stub
    keeps the dependency on executor/client code without adding research deps to
    this project.
    """

    module_name = "trading.signal_generator"
    if module_name in sys.modules:
        return
    stub = types.ModuleType(module_name)
    stub.SignalGenerator = _StaticSignalGenerator
    sys.modules[module_name] = stub


def load_wave_execution_classes(wave_trading_path: str = DEFAULT_WAVE_TRADING_PATH) -> tuple[type, type]:
    """Load wave_trading BinanceFuturesClient and TradingExecutor classes."""

    root = Path(wave_trading_path).expanduser().resolve()
    if not (root / "trading" / "executor.py").exists():
        raise FileNotFoundError(f"wave_trading executor not found under {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _install_signal_generator_stub()
    client_cls = importlib.import_module("trading.binance_client").BinanceFuturesClient
    executor_cls = importlib.import_module("trading.executor").TradingExecutor
    return client_cls, executor_cls


class WaveExecutionBridge:
    """Execute RLLM decisions through wave_trading's Binance futures executor."""

    def __init__(self, *, config: WaveExecutionConfig, client: Any, executor: Any) -> None:
        self.config = config
        self.client = client
        self.executor = executor
        self._initialized = False

    @classmethod
    def from_env(cls, *, config: WaveExecutionConfig | None = None) -> "WaveExecutionBridge":
        cfg = config or WaveExecutionConfig.from_env()
        api_key, api_secret = _load_api_credentials(cfg.testnet, cfg.wave_trading_path)
        client_cls, executor_cls = load_wave_execution_classes(cfg.wave_trading_path)
        client = client_cls(api_key=api_key, api_secret=api_secret, testnet=cfg.testnet)
        signal_generator = _StaticSignalGenerator(
            atr_period=cfg.atr_period,
            pt_mult=cfg.pt_mult,
            max_holding_bars=cfg.max_holding_bars,
        )
        executor = executor_cls(
            client=client,
            signal_generator=signal_generator,
            symbol=cfg.symbol,
            leverage=cfg.leverage,
            position_size_pct=cfg.position_size_pct,
            maker_offset_pct=cfg.maker_offset_pct,
            max_retries=cfg.max_retries,
            order_timeout_sec=cfg.order_timeout_sec,
        )
        return cls(config=cfg, client=client, executor=executor)

    async def initialize(self) -> None:
        if self.config.dry_run:
            self._initialized = True
            return
        if not self._initialized:
            await self.client.sync_time()
            await self.executor.initialize()
            self._initialized = True

    async def execute_decision(self, decision: ExecutionDecision, *, ws_client: Any = None) -> dict[str, Any]:
        """Execute or dry-run one already-approved RLLM decision."""

        _validate_decision(decision)
        if self.config.dry_run:
            return {
                "dry_run": True,
                "action": "NOOP" if decision.signal == "HOLD" else "EXECUTE_SIGNAL",
                "decision": asdict(decision),
                "config": asdict(self.config),
            }

        await self.initialize()
        await self.executor.handle_signal(
            decision.signal,
            decision.probability,
            decision.current_close,
            decision.current_atr,
            decision.signal_id,
            self.config.interval_minutes,
            ws_client=ws_client,
        )
        return {"dry_run": False, "action": "HANDLED", "decision": asdict(decision)}

    async def close_position(self, *, reason: str = "MANUAL_EXIT", ws_client: Any = None) -> dict[str, Any]:
        if self.config.dry_run:
            return {"dry_run": True, "action": "CLOSE_POSITION", "reason": reason, "config": asdict(self.config)}
        await self.initialize()
        order = await self.executor.close_position(ws_client=ws_client, signal_id=reason, order_type="EXIT")
        return {"dry_run": False, "action": "CLOSE_POSITION", "order": order, "reason": reason}

    async def aclose(self) -> None:
        close = getattr(self.client, "aclose", None)
        if close is not None:
            await close()


def _load_api_credentials(testnet: bool, wave_trading_path: str = DEFAULT_WAVE_TRADING_PATH) -> tuple[str, str]:
    load_env_file(Path(wave_trading_path) / ".env", override=False)
    if testnet:
        key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
        secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
    else:
        key = os.environ.get("BINANCE_LIVE_API_KEY", "")
        secret = os.environ.get("BINANCE_LIVE_API_SECRET", "")
    # Dry-run can be constructed without credentials, but live mode cannot.
    if os.environ.get("RLLM_EXECUTION_DRY_RUN", "true").strip().lower() not in {"1", "true", "yes", "y", "on"}:
        if not key or not secret:
            venue = "testnet" if testnet else "live"
            raise ValueError(f"Missing Binance {venue} API credentials")
    return key, secret


def _validate_decision(decision: ExecutionDecision) -> None:
    if decision.signal not in {"LONG", "SHORT", "HOLD"}:
        raise ValueError(f"Unsupported signal: {decision.signal}")
    if not (0.0 <= float(decision.probability) <= 1.0):
        raise ValueError("probability must be in [0, 1]")
    if float(decision.current_close) <= 0.0:
        raise ValueError("current_close must be positive")
    if float(decision.current_atr) < 0.0:
        raise ValueError("current_atr must be non-negative")
    if not decision.signal_id:
        raise ValueError("signal_id is required for execution traceability")


async def _amain(args: argparse.Namespace) -> None:
    cfg = WaveExecutionConfig.from_env()
    if args.live:
        cfg = WaveExecutionConfig(**{**asdict(cfg), "dry_run": False})
    elif args.dry_run:
        cfg = WaveExecutionConfig(**{**asdict(cfg), "dry_run": True})
    bridge = WaveExecutionBridge.from_env(config=cfg)
    try:
        decision = ExecutionDecision(
            signal=args.signal,
            probability=args.probability,
            current_close=args.current_close,
            current_atr=args.current_atr,
            signal_id=args.signal_id,
            reason=args.reason,
        )
        result = await bridge.execute_decision(decision)
        print(result)
    finally:
        await bridge.aclose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one RLLM signal via wave_trading Binance futures executor")
    parser.add_argument("--signal", choices=["LONG", "SHORT", "HOLD"], required=True)
    parser.add_argument("--probability", type=float, default=0.5)
    parser.add_argument("--current-close", type=float, required=True)
    parser.add_argument("--current-atr", type=float, default=0.0)
    parser.add_argument("--signal-id", required=True)
    parser.add_argument("--reason", default="manual-cli")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=False)
    mode.add_argument("--live", action="store_true", default=False, help="Actually place/cancel orders; requires credentials")
    return parser.parse_args()


def main() -> None:
    asyncio.run(_amain(parse_args()))


if __name__ == "__main__":
    main()
