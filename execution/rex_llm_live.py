"""Live REX+RLLM policy runner backed by the live market database.

This module connects the already-validated research pipeline to the safe
``wave_trading`` execution bridge:

1. read BTC/FX/kimchi/premium/funding frames from PostgreSQL,
2. build the same leak-safe feature frame used by research,
3. select the current REX candidate side from past-only strength quantiles,
4. apply the frozen RLLM gate thesis (range-vol + kimchi-flow), and
5. pass a dry-run/live decision into ``WaveExecutionBridge``.

The defaults are intentionally execution-safe.  The bundled live config remains
``dry_run=true`` and ``manual_regime=UNKNOWN``, so even a TRADE candidate is
blocked until the operator deliberately sets a bearish manual regime and enables
live orders.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from execution.wave_execution import WaveExecutionBridge, WaveExecutionConfig, decision_from_policy_record
from preprocessing.live_db_features import (
    LiveDbFeatureConfig,
    build_live_feature_frame_from_frames,
    latest_live_feature_snapshot,
    query_live_source_frames,
    sqlalchemy_engine_from_env,
)
from training.event_candidate_pool_probe import _feature_candidates

Side = Literal["LONG", "SHORT", "NONE"]


@dataclass(frozen=True)
class FrozenGate:
    """Simple numeric gate learned/frozen outside the live loop."""

    feature: str
    op: Literal[">=", "<=", ">", "<"]
    value: float

    def evaluate(self, snapshot: dict[str, float]) -> bool:
        actual = float(snapshot.get(self.feature, 0.0))
        if self.op == ">=":
            return actual >= self.value
        if self.op == "<=":
            return actual <= self.value
        if self.op == ">":
            return actual > self.value
        if self.op == "<":
            return actual < self.value
        raise ValueError(f"unsupported gate op: {self.op}")


@dataclass(frozen=True)
class RexLivePolicyConfig:
    """Frozen live policy parameters for the current REX+RLLM pilot."""

    family: str = "rex_htf_pullback_reclaim"
    strength_quantile: float = 0.75
    min_positive_strengths: int = 50
    hold_bars: int = 144
    entry_delay_bars: int = 1
    # Frozen RLLM thesis from the current best 2025+2026H1 OOS gate.
    gates: tuple[FrozenGate, ...] = (
        FrozenGate("range_vol", ">=", 0.023959233645008706),
        FrozenGate("kimchi_premium_change", "<=", 0.0),
    )
    require_core_external: bool = True
    allow_missing_core_external_on_weekend: bool = True
    require_binance_aux: bool = False


@dataclass(frozen=True)
class RexLiveResult:
    policy_record: dict[str, Any]
    execution_result: dict[str, Any]


def _current_atr(enriched: pd.DataFrame, period: int = 15) -> float:
    if enriched.empty:
        return 0.0
    high = pd.to_numeric(enriched.get("high"), errors="coerce")
    low = pd.to_numeric(enriched.get("low"), errors="coerce")
    close = pd.to_numeric(enriched.get("close"), errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(max(1, int(period)), min_periods=1).mean().iloc[-1]
    if not np.isfinite(float(atr)):
        return 0.0
    return float(atr)


def _side_from_direction(direction: float) -> Side:
    if direction > 0.0:
        return "LONG"
    if direction < 0.0:
        return "SHORT"
    return "NONE"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _is_weekend_or_fx_closed(date_value: Any) -> bool:
    """Return True when historical train/eval would normally have stale/missing FX.

    FX closes around Friday 22:00 UTC and reopens around Sunday 22:00 UTC.
    Historical datasets used a short as-of tolerance and represented these rows
    as availability=0 with neutral numeric external features, not as hard blocks.
    """

    ts = pd.Timestamp(date_value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    dow = int(ts.dayofweek)
    hour = int(ts.hour)
    return dow == 5 or (dow == 6 and hour < 22) or (dow == 4 and hour >= 22)


def _quality_ok(data_quality: dict[str, Any], cfg: RexLivePolicyConfig, *, date_value: Any) -> tuple[bool, list[str]]:
    missing: list[str] = []
    weekend_fx_closed = _is_weekend_or_fx_closed(date_value)
    if cfg.require_core_external and not (cfg.allow_missing_core_external_on_weekend and weekend_fx_closed):
        for key in ("dxy_available", "kimchi_available", "usdkrw_available"):
            if _safe_float(data_quality.get(key)) < 1.0:
                missing.append(key)
    if cfg.require_binance_aux:
        for key in ("premium_available", "funding_available"):
            if _safe_float(data_quality.get(key)) < 1.0:
                missing.append(key)
    return not missing, missing


def build_rex_live_policy_record(
    enriched: pd.DataFrame,
    features: pd.DataFrame,
    *,
    policy_cfg: RexLivePolicyConfig = RexLivePolicyConfig(),
    execution_cfg: WaveExecutionConfig | None = None,
) -> dict[str, Any]:
    """Build one leak-safe live policy record from completed feature rows."""

    if enriched.empty or features.empty:
        raise ValueError("cannot score empty live frame")
    if len(enriched) != len(features):
        raise ValueError(f"enriched/features row mismatch: {len(enriched)} != {len(features)}")
    if not (0.0 < float(policy_cfg.strength_quantile) < 1.0):
        raise ValueError("strength_quantile must be in (0, 1)")

    families = _feature_candidates(features)
    if policy_cfg.family not in families:
        raise ValueError(f"unknown REX family: {policy_cfg.family}")
    strength, direction = families[policy_cfg.family]
    pos = len(features) - 1
    historical = np.asarray(strength[: pos + 1], dtype=float)
    positive = historical[np.isfinite(historical) & (historical > 0.0)]
    enough_history = len(positive) >= int(policy_cfg.min_positive_strengths)
    threshold = float(np.quantile(positive, float(policy_cfg.strength_quantile))) if enough_history else float("inf")
    current_strength = _safe_float(strength[pos], 0.0)
    current_direction = _safe_float(direction[pos], 0.0)
    candidate_active = bool(enough_history and np.isfinite(threshold) and current_strength > threshold and current_direction != 0.0)
    side = _side_from_direction(current_direction) if candidate_active else "NONE"

    snapshot = latest_live_feature_snapshot(enriched, features)
    feature_snapshot = dict(snapshot["feature_snapshot"])
    feature_snapshot.update(
        {
            "rex_candidate_strength": current_strength,
            "rex_candidate_threshold": threshold if np.isfinite(threshold) else 0.0,
            "rex_candidate_margin": current_strength - threshold if np.isfinite(threshold) else 0.0,
        }
    )
    data_quality = dict(snapshot["data_quality"])
    quality_ok, missing_quality = _quality_ok(data_quality, policy_cfg, date_value=snapshot["date"])
    gate_results = [
        {
            "feature": gate.feature,
            "op": gate.op,
            "value": float(gate.value),
            "actual": _safe_float(feature_snapshot.get(gate.feature)),
            "passed": bool(gate.evaluate(feature_snapshot)),
        }
        for gate in policy_cfg.gates
    ]
    gates_pass = all(item["passed"] for item in gate_results)
    prediction = "TRADE" if candidate_active and gates_pass and quality_ok else "ABSTAIN"
    close = _safe_float(enriched.iloc[-1].get("close"), 1.0)
    atr_period = execution_cfg.atr_period if execution_cfg is not None else 15
    current_atr = _current_atr(enriched, atr_period)
    margin_prob = 0.5
    if candidate_active and np.isfinite(threshold) and threshold > 0.0:
        margin_prob = float(np.clip(0.5 + 0.25 * ((current_strength / threshold) - 1.0), 0.5, 0.95))

    reasons: list[str] = []
    if not enough_history:
        reasons.append(f"insufficient_positive_strengths={len(positive)}<{policy_cfg.min_positive_strengths}")
    if not candidate_active and enough_history:
        reasons.append("rex_candidate_inactive")
    if not gates_pass:
        reasons.append("frozen_gate_block")
    if not quality_ok:
        reasons.append("missing_quality=" + ",".join(missing_quality))
    if prediction == "TRADE":
        reasons.append("rex_candidate_and_frozen_gate_pass")

    date = str(snapshot["date"])
    return {
        "date": date,
        "prediction": prediction,
        "candidate_side": side,
        "action": {
            "family": policy_cfg.family,
            "side": side,
            "strength": current_strength,
            "threshold": threshold if np.isfinite(threshold) else None,
            "strength_quantile": float(policy_cfg.strength_quantile),
            "hold_bars": int(policy_cfg.hold_bars),
            "entry_delay_bars": int(policy_cfg.entry_delay_bars),
        },
        "feature_snapshot": feature_snapshot,
        "data_quality": data_quality,
        "gate_results": gate_results,
        "current_close": close,
        "current_atr": current_atr,
        "probability": margin_prob,
        "signal_id": f"{policy_cfg.family}:{date}",
        "reason": ";".join(reasons),
        "policy_config": {
            **asdict(policy_cfg),
            "gates": [asdict(gate) for gate in policy_cfg.gates],
        },
    }


async def run_rex_live_once(
    *,
    execution_cfg: WaveExecutionConfig,
    policy_cfg: RexLivePolicyConfig = RexLivePolicyConfig(),
    live_db_cfg: LiveDbFeatureConfig = LiveDbFeatureConfig(),
    asof: str | pd.Timestamp | None = None,
    env_path: str | Path = ".env",
) -> RexLiveResult:
    """Query DB, score the latest completed bar, and execute/dry-run once."""

    asof_ts = pd.Timestamp.utcnow() if asof is None else pd.Timestamp(asof)
    if asof_ts.tzinfo is None:
        asof_ts = asof_ts.tz_localize("UTC")
    else:
        asof_ts = asof_ts.tz_convert("UTC")
    engine = sqlalchemy_engine_from_env(env_path)
    frames = query_live_source_frames(engine, asof=asof_ts, cfg=live_db_cfg)
    enriched, features = build_live_feature_frame_from_frames(cfg=live_db_cfg, **frames)
    record = build_rex_live_policy_record(enriched, features, policy_cfg=policy_cfg, execution_cfg=execution_cfg)
    decision = decision_from_policy_record(record)
    bridge = WaveExecutionBridge.from_env(config=execution_cfg)
    try:
        execution_result = await bridge.execute_decision(decision)
    finally:
        await bridge.aclose()
    return RexLiveResult(policy_record=record, execution_result=execution_result)


def _parse_gate(raw: str) -> FrozenGate:
    for op in (">=", "<=", ">", "<"):
        if op in raw:
            left, right = raw.split(op, 1)
            return FrozenGate(left.strip(), op, float(right.strip()))  # type: ignore[arg-type]
    raise argparse.ArgumentTypeError(f"gate must look like feature>=value: {raw}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DB-backed REX+RLLM live decision through the wave_trading bridge")
    parser.add_argument("--config", default="configs/live/rex_llm_binance_testnet_bear_pilot.json", help="WaveExecutionConfig JSON")
    parser.add_argument("--env", default=".env", help="dotenv containing PG_* keys")
    parser.add_argument("--asof", default=None, help="UTC timestamp; defaults to now")
    parser.add_argument("--family", default=RexLivePolicyConfig.family)
    parser.add_argument("--strength-quantile", type=float, default=RexLivePolicyConfig.strength_quantile)
    parser.add_argument("--min-positive-strengths", type=int, default=RexLivePolicyConfig.min_positive_strengths)
    parser.add_argument("--lookback-minutes", type=int, default=LiveDbFeatureConfig.lookback_minutes)
    parser.add_argument("--require-binance-aux", action="store_true", default=False)
    parser.add_argument("--strict-core-external", action="store_true", default=False, help="Block missing DXY/kimchi/USDKRW even during FX weekend closures")
    parser.add_argument("--allow-missing-core-external", action="store_true", default=False, help="Always allow missing DXY/kimchi/USDKRW like historical neutral-fill evaluation")
    parser.add_argument("--gate", action="append", type=_parse_gate, default=None, help="Override frozen gate, e.g. range_vol>=0.02")
    parser.add_argument("--manual-regime", choices=["UNKNOWN", "BEAR", "BULL", "SIDEWAYS"], help="Override config manual regime")
    parser.add_argument("--allow-live-orders", action="store_true", default=False, help="Set allow_live_orders=true; still needs --live")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=False)
    mode.add_argument("--live", action="store_true", default=False, help="Set dry_run=false; requires explicit allow_live_orders")
    return parser.parse_args()


async def _amain(args: argparse.Namespace) -> None:
    exec_cfg = WaveExecutionConfig.from_json(args.config)
    overrides = asdict(exec_cfg)
    if args.manual_regime:
        overrides["manual_regime"] = args.manual_regime
    if args.allow_live_orders:
        overrides["allow_live_orders"] = True
    if args.live:
        overrides["dry_run"] = False
    elif args.dry_run:
        overrides["dry_run"] = True
    exec_cfg = WaveExecutionConfig(**overrides)
    policy_cfg = RexLivePolicyConfig(
        family=args.family,
        strength_quantile=args.strength_quantile,
        min_positive_strengths=args.min_positive_strengths,
        gates=tuple(args.gate) if args.gate else RexLivePolicyConfig.gates,
        require_core_external=not bool(args.allow_missing_core_external),
        allow_missing_core_external_on_weekend=not bool(args.strict_core_external),
        require_binance_aux=bool(args.require_binance_aux),
    )
    live_db_cfg = LiveDbFeatureConfig(lookback_minutes=int(args.lookback_minutes))
    result = await run_rex_live_once(
        execution_cfg=exec_cfg,
        policy_cfg=policy_cfg,
        live_db_cfg=live_db_cfg,
        asof=args.asof,
        env_path=args.env,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True, default=str))


def main() -> None:
    asyncio.run(_amain(parse_args()))


if __name__ == "__main__":
    main()
