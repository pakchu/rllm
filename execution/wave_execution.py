"""Bridge RLLM policy signals to the wave_trading Binance futures executor.

This module intentionally keeps the boundary narrow: RLLM owns model inference,
manual long-regime filtering, and risk-overlay decisions; wave_trading owns
Binance REST client details and maker-order execution.  Live order placement is
opt-in and blocked unless multiple explicit safety gates are enabled.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import types
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Literal, Optional

TradeSignal = Literal["LONG", "SHORT", "HOLD"]
ManualRegime = Literal["UNKNOWN", "BEAR", "BULL", "SIDEWAYS"]

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


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_allowed_signals(value: str) -> tuple[str, ...]:
    out = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    return out or ("LONG", "SHORT")


@dataclass(frozen=True)
class WaveExecutionConfig:
    """Configuration for the wave_trading execution bridge.

    The default is intentionally safe: dry-run, testnet, manual regime UNKNOWN,
    and live order placement disabled.  To place even testnet orders, caller must
    set dry_run=false, allow_live_orders=true, and satisfy the regime/signal gates.
    """

    wave_trading_path: str = DEFAULT_WAVE_TRADING_PATH
    symbol: str = "BTCUSDT"
    testnet: bool = True
    leverage: int = 1
    position_size_pct: float = 0.02
    maker_offset_pct: float = 0.0001
    max_retries: int = 3
    order_timeout_sec: int = 60
    dry_run: bool = True
    allow_live_orders: bool = False
    interval_minutes: int = 5
    manual_regime: ManualRegime = "UNKNOWN"
    require_bear_regime: bool = True
    allowed_signals: tuple[str, ...] = ("LONG", "SHORT")
    max_probability_age_sec: int = 600
    require_flat_position: bool = True
    require_no_open_orders: bool = True
    # Values consumed by wave_trading.TradingExecutor for position tracking if
    # caller chooses to use its trailing/max-holding helpers.
    atr_period: int = 15
    pt_mult: float = 3.75
    max_holding_bars: int = 144

    @classmethod
    def from_env(cls) -> "WaveExecutionConfig":
        return cls(
            wave_trading_path=os.environ.get("RLLM_WAVE_TRADING_PATH", DEFAULT_WAVE_TRADING_PATH),
            symbol=os.environ.get("TRADING_SYMBOL", "BTCUSDT"),
            testnet=_env_bool("BINANCE_TESTNET", True),
            leverage=int(os.environ.get("TRADING_LEVERAGE", "1")),
            position_size_pct=float(os.environ.get("POSITION_SIZE_PCT", "0.02")),
            maker_offset_pct=float(os.environ.get("MAKER_OFFSET_PCT", "0.0001")),
            max_retries=int(os.environ.get("MAX_RETRIES", "3")),
            order_timeout_sec=int(os.environ.get("ORDER_TIMEOUT_SEC", "60")),
            dry_run=_env_bool("RLLM_EXECUTION_DRY_RUN", True),
            allow_live_orders=_env_bool("RLLM_ALLOW_LIVE_ORDERS", False),
            interval_minutes=int(os.environ.get("INTERVAL_MINUTES", "5")),
            manual_regime=os.environ.get("RLLM_MANUAL_REGIME", "UNKNOWN").strip().upper(),  # type: ignore[arg-type]
            require_bear_regime=_env_bool("RLLM_REQUIRE_BEAR_REGIME", True),
            allowed_signals=_split_allowed_signals(os.environ.get("RLLM_ALLOWED_SIGNALS", "LONG,SHORT")),
            max_probability_age_sec=int(os.environ.get("RLLM_MAX_PROBABILITY_AGE_SEC", "600")),
            require_flat_position=_env_bool("RLLM_REQUIRE_FLAT_POSITION", True),
            require_no_open_orders=_env_bool("RLLM_REQUIRE_NO_OPEN_ORDERS", True),
            atr_period=int(os.environ.get("RLLM_EXECUTION_ATR_PERIOD", "15")),
            pt_mult=float(os.environ.get("RLLM_EXECUTION_PT_MULT", "3.75")),
            max_holding_bars=int(os.environ.get("RLLM_EXECUTION_MAX_HOLDING_BARS", "144")),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "WaveExecutionConfig":
        raw = json.loads(Path(path).expanduser().read_text())
        if not isinstance(raw, dict):
            raise ValueError("live config JSON must be an object")
        field_names = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - field_names)
        if unknown:
            raise ValueError(f"Unknown live config keys: {unknown}")
        if "allowed_signals" in raw:
            raw["allowed_signals"] = tuple(str(x).upper() for x in raw["allowed_signals"])
        return cls(**raw)


@dataclass(frozen=True)
class ExecutionDecision:
    """A model/risk-layer decision ready for execution."""

    signal: TradeSignal
    probability: float
    current_close: float
    current_atr: float
    signal_id: str
    reason: str = ""
    age_sec: float = 0.0


@dataclass(frozen=True)
class ExecutionGateResult:
    allowed: bool
    reason: str
    action: str


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


class _DryRunClient:
    async def aclose(self) -> None:
        return None


class _DryRunExecutor:
    async def handle_signal(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("dry-run executor must not be called")

    async def close_position(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("dry-run executor must not be called")


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


def _nested_get(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _normalize_policy_label(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("decision", "gate", "label", "prediction", "action"):
            normalized = _normalize_policy_label(value.get(key))
            if normalized != "UNKNOWN":
                return normalized
        return "UNKNOWN"
    text = str(value or "").strip().upper()
    if text in {"TRADE", "ALLOW", "TAKE", "EXECUTE", "LONG", "SHORT"}:
        return "TRADE"
    if text in {"ABSTAIN", "NO_TRADE", "NOTRADE", "BLOCK", "HOLD", "SKIP", "NONE"}:
        return "ABSTAIN"
    return "UNKNOWN"


def _extract_policy_label(record: dict[str, Any]) -> str:
    candidates = (
        ("decision",),
        ("prediction",),
        ("label",),
        ("target",),
        ("output",),
        ("metadata", "target"),
        ("metadata", "prediction"),
    )
    for path in candidates:
        label = _normalize_policy_label(_nested_get(record, path))
        if label != "UNKNOWN":
            return label
    return "UNKNOWN"


def _extract_candidate_side(record: dict[str, Any]) -> str:
    candidates = (
        ("candidate_side",),
        ("side",),
        ("action", "side"),
        ("metadata", "action", "side"),
        ("metadata", "target", "action_side"),
    )
    for path in candidates:
        value = _nested_get(record, path)
        side = str(value or "").strip().upper()
        if side in {"LONG", "SHORT"}:
            return side
    return "NONE"


def decision_from_policy_record(record: dict[str, Any], *, default_close: float = 1.0, default_atr: float = 0.0) -> ExecutionDecision:
    """Convert a pre-scored REX+LLM policy record into an execution decision.

    The LLM gate remains binary: TRADE maps to the candidate side, while
    ABSTAIN/NO_TRADE/BLOCK maps to HOLD.  This preserves the current architecture
    where REX chooses the side and the LLM decides whether that candidate is
    allowed in the current regime.
    """

    label = _extract_policy_label(record)
    side = _extract_candidate_side(record)
    signal: TradeSignal = side if label == "TRADE" and side in {"LONG", "SHORT"} else "HOLD"  # type: ignore[assignment]
    snap = record.get("feature_snapshot", {}) if isinstance(record.get("feature_snapshot"), dict) else {}
    close = record.get("current_close", record.get("close", snap.get("close", default_close)))
    atr = record.get("current_atr", record.get("atr", snap.get("atr", default_atr)))
    signal_id = str(record.get("signal_id") or record.get("id") or f"{record.get('date', 'unknown')}:{record.get('signal_pos', 'na')}")
    probability = float(record.get("probability", record.get("score", 0.5)))
    age_sec = float(record.get("age_sec", 0.0))
    return ExecutionDecision(
        signal=signal,
        probability=max(0.0, min(1.0, probability)),
        current_close=float(close),
        current_atr=float(atr),
        signal_id=signal_id,
        reason=f"policy_label={label};candidate_side={side}",
        age_sec=age_sec,
    )


def evaluate_execution_gate(config: WaveExecutionConfig, decision: ExecutionDecision) -> ExecutionGateResult:
    """Decide whether a model decision may reach the Binance executor."""

    _validate_config(config)
    _validate_decision(decision)
    if decision.signal == "HOLD":
        return ExecutionGateResult(True, "hold/noop", "NOOP")
    if decision.signal not in set(config.allowed_signals):
        return ExecutionGateResult(False, f"signal {decision.signal} not in allowed_signals", "BLOCKED")
    if config.require_bear_regime and config.manual_regime != "BEAR":
        return ExecutionGateResult(False, f"manual_regime={config.manual_regime}; BEAR required", "BLOCKED")
    if decision.age_sec > config.max_probability_age_sec:
        return ExecutionGateResult(False, f"decision too stale: {decision.age_sec:.1f}s", "BLOCKED")
    if not config.dry_run and not config.allow_live_orders:
        return ExecutionGateResult(False, "live orders require allow_live_orders=true", "BLOCKED")
    return ExecutionGateResult(True, "approved", "EXECUTE_SIGNAL")


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
        _validate_config(cfg)
        if cfg.dry_run:
            return cls(config=cfg, client=_DryRunClient(), executor=_DryRunExecutor())
        api_key, api_secret = _load_api_credentials(cfg.testnet, cfg.wave_trading_path, dry_run=cfg.dry_run)
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

        gate = evaluate_execution_gate(self.config, decision)
        if not gate.allowed:
            return {
                "dry_run": self.config.dry_run,
                "action": gate.action,
                "gate_reason": gate.reason,
                "decision": asdict(decision),
                "config": _safe_config_dict(self.config),
            }
        if gate.action == "NOOP":
            return {
                "dry_run": self.config.dry_run,
                "action": gate.action,
                "gate_reason": gate.reason,
                "decision": asdict(decision),
                "config": _safe_config_dict(self.config),
            }
        if self.config.dry_run:
            return {
                "dry_run": True,
                "action": gate.action,
                "gate_reason": gate.reason,
                "decision": asdict(decision),
                "config": _safe_config_dict(self.config),
            }

        sync_time = getattr(self.client, "sync_time", None)
        if self.config.require_flat_position or self.config.require_no_open_orders:
            if sync_time is not None:
                await sync_time()
        if self.config.require_flat_position:
            get_position = getattr(self.client, "get_position", None)
            if get_position is None:
                return {
                    "dry_run": False,
                    "action": "BLOCKED",
                    "gate_reason": "require_flat_position=true but client has no get_position",
                    "decision": asdict(decision),
                    "config": _safe_config_dict(self.config),
                }
            position = await get_position(self.config.symbol)
            if str(position.get("side", "NONE")).upper() != "NONE" or float(position.get("quantity", 0.0)) != 0.0:
                return {
                    "dry_run": False,
                    "action": "BLOCKED",
                    "gate_reason": "existing position present; require_flat_position=true",
                    "position": position,
                    "decision": asdict(decision),
                    "config": _safe_config_dict(self.config),
                }
        if self.config.require_no_open_orders:
            get_open_orders = getattr(self.client, "get_open_orders", None)
            if get_open_orders is None:
                return {
                    "dry_run": False,
                    "action": "BLOCKED",
                    "gate_reason": "require_no_open_orders=true but client has no get_open_orders",
                    "decision": asdict(decision),
                    "config": _safe_config_dict(self.config),
                }
            open_orders = await get_open_orders(self.config.symbol)
            if open_orders:
                return {
                    "dry_run": False,
                    "action": "BLOCKED",
                    "gate_reason": "open orders present; require_no_open_orders=true",
                    "open_orders_count": len(open_orders),
                    "decision": asdict(decision),
                    "config": _safe_config_dict(self.config),
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
        return {"dry_run": False, "action": "HANDLED", "gate_reason": gate.reason, "decision": asdict(decision)}

    async def close_position(self, *, reason: str = "MANUAL_EXIT", ws_client: Any = None) -> dict[str, Any]:
        if self.config.dry_run:
            return {"dry_run": True, "action": "CLOSE_POSITION", "reason": reason, "config": _safe_config_dict(self.config)}
        if not self.config.allow_live_orders:
            return {"dry_run": False, "action": "BLOCKED", "gate_reason": "close requires allow_live_orders=true"}
        await self.initialize()
        order = await self.executor.close_position(ws_client=ws_client, signal_id=reason, order_type="EXIT")
        return {"dry_run": False, "action": "CLOSE_POSITION", "order": order, "reason": reason}

    async def aclose(self) -> None:
        close = getattr(self.client, "aclose", None)
        if close is not None:
            await close()


def _load_api_credentials(
    testnet: bool,
    wave_trading_path: str = DEFAULT_WAVE_TRADING_PATH,
    *,
    dry_run: Optional[bool] = None,
) -> tuple[str, str]:
    load_env_file(Path(wave_trading_path) / ".env", override=False)
    if testnet:
        key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
        secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
    else:
        key = os.environ.get("BINANCE_LIVE_API_KEY", "")
        secret = os.environ.get("BINANCE_LIVE_API_SECRET", "")
    # Dry-run can be constructed without credentials, but live mode cannot.
    dry = _env_bool("RLLM_EXECUTION_DRY_RUN", True) if dry_run is None else dry_run
    if not dry:
        if not key or not secret:
            venue = "testnet" if testnet else "live"
            raise ValueError(f"Missing Binance {venue} API credentials")
    return key, secret


def _validate_config(config: WaveExecutionConfig) -> None:
    if config.manual_regime not in {"UNKNOWN", "BEAR", "BULL", "SIDEWAYS"}:
        raise ValueError(f"Unsupported manual_regime: {config.manual_regime}")
    bad = set(config.allowed_signals) - {"LONG", "SHORT", "HOLD"}
    if bad:
        raise ValueError(f"Unsupported allowed_signals: {sorted(bad)}")
    if config.leverage < 1 or config.leverage > 125:
        raise ValueError("leverage must be in [1, 125]")
    if not (0.0 < config.position_size_pct <= 1.0):
        raise ValueError("position_size_pct must be in (0, 1]")
    if config.interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    if config.max_probability_age_sec <= 0:
        raise ValueError("max_probability_age_sec must be positive")


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
    if float(decision.age_sec) < 0.0:
        raise ValueError("age_sec must be non-negative")


def _safe_config_dict(config: WaveExecutionConfig) -> dict[str, Any]:
    data = asdict(config)
    # No secrets are stored here; keep helper explicit for future fields.
    return data


async def _amain(args: argparse.Namespace) -> None:
    cfg = WaveExecutionConfig.from_json(args.config) if args.config else WaveExecutionConfig.from_env()
    if args.live:
        cfg = WaveExecutionConfig(**{**asdict(cfg), "dry_run": False})
    elif args.dry_run:
        cfg = WaveExecutionConfig(**{**asdict(cfg), "dry_run": True})
    bridge = WaveExecutionBridge.from_env(config=cfg)
    try:
        if args.policy_record_json:
            decision = decision_from_policy_record(json.loads(args.policy_record_json))
        elif args.policy_record_file:
            decision = decision_from_policy_record(json.loads(Path(args.policy_record_file).read_text()))
        else:
            decision = ExecutionDecision(
                signal=args.signal,
                probability=args.probability,
                current_close=args.current_close,
                current_atr=args.current_atr,
                signal_id=args.signal_id,
                reason=args.reason,
                age_sec=args.age_sec,
            )
        result = await bridge.execute_decision(decision)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        await bridge.aclose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one RLLM signal via wave_trading Binance futures executor")
    parser.add_argument("--config", help="JSON config path; defaults to environment variables")
    parser.add_argument("--policy-record-json", help="Pre-scored REX+LLM JSON object; TRADE executes candidate side")
    parser.add_argument("--policy-record-file", help="Path to a pre-scored REX+LLM JSON object")
    parser.add_argument("--signal", choices=["LONG", "SHORT", "HOLD"], required=False)
    parser.add_argument("--probability", type=float, default=0.5)
    parser.add_argument("--current-close", type=float, default=1.0)
    parser.add_argument("--current-atr", type=float, default=0.0)
    parser.add_argument("--signal-id", default="manual-cli")
    parser.add_argument("--reason", default="manual-cli")
    parser.add_argument("--age-sec", type=float, default=0.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=False)
    mode.add_argument("--live", action="store_true", default=False, help="Actually place/cancel orders; requires credentials + allow_live_orders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.policy_record_json and not args.policy_record_file and not args.signal:
        raise SystemExit("--signal is required unless --policy-record-json/--policy-record-file is provided")
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
