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
from functools import lru_cache
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
class RexLlmSelectorConfig:
    """Bounded adapter selector for REX candidates.

    The selector can only convert an already-active REX+frozen-gate candidate
    from TRADE to ABSTAIN.  It cannot create a candidate, change side, change
    hold bars, or alter sizing.  Fail-closed keeps live execution safe if the
    adapter/model stack is unavailable at decision time.
    """

    enabled: bool = False
    adapter_dir: str = "checkpoints/rex_regime_thesis_range_kimchi_label_gemma4_s32_2026-07-03"
    model_name: str = "gemma4-e4b-it"
    score_normalization: str = "sum"
    fail_closed: bool = True
    require_cuda: bool = True

@dataclass(frozen=True)
class RexLiveResult:
    policy_record: dict[str, Any]
    execution_result: dict[str, Any]


@dataclass(frozen=True)
class RexLiveLoopConfig:
    """Runtime controls for repeated live scoring/execution."""

    state_file: Path = Path(".omx/state/rex_llm_live_loop_state.json")
    close_delay_sec: float = 2.0
    run_immediately: bool = True
    max_iterations: int | None = None
    json_log: bool = False


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



def _selector_enabled(cfg: RexLlmSelectorConfig | None) -> bool:
    return bool(cfg and cfg.enabled and str(cfg.adapter_dir).strip())


@lru_cache(maxsize=2)
def _load_rex_selector_model(model_name: str, adapter_dir: str):
    from training.eval_text_label import _load_text_model

    return _load_text_model(model_name, adapter_dir)


def _chat_prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if getattr(tokenizer, "chat_template", None)
        else f"<|user|>\n{prompt}\n<|assistant|>\n"
    )


def _selector_prompt_from_record(record: dict[str, Any], gates: tuple[FrozenGate, ...]) -> str:
    from training.build_rex_regime_thesis_sft import Gate, _prompt

    row = {
        "date": record.get("date"),
        "action": record.get("action", {}),
        "feature_snapshot": record.get("feature_snapshot", {}),
    }
    gate_rows = tuple(Gate(feature=g.feature, op=g.op, threshold=float(g.value)) for g in gates if g.op in {">=", "<="})
    return _prompt(row, gate_rows)


def _score_rex_llm_selector(record: dict[str, Any], *, gates: tuple[FrozenGate, ...], cfg: RexLlmSelectorConfig) -> dict[str, Any]:
    """Return bounded TRADE/ABSTAIN selector decision with label logprobs."""

    import torch

    if cfg.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("REX selector requires CUDA, but torch.cuda.is_available() is false")
    if cfg.score_normalization not in {"sum", "mean", "first_token"}:
        raise ValueError("score_normalization must be one of {'sum','mean','first_token'}")
    tokenizer, model = _load_rex_selector_model(str(cfg.model_name), str(cfg.adapter_dir))
    prompt = _selector_prompt_from_record(record, gates)
    prompt_ids = tokenizer(_chat_prompt_text(tokenizer, prompt), add_special_tokens=False)["input_ids"]
    labels = ["ABSTAIN", "TRADE"]
    sequences: list[list[int]] = []
    spans: list[tuple[int, int]] = []
    for label in labels:
        label_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            label_ids = label_ids + [int(tokenizer.eos_token_id)]
        start = len(prompt_ids)
        end = start + len(label_ids)
        sequences.append(prompt_ids + label_ids)
        spans.append((start, end))
    encoded = tokenizer.pad({"input_ids": sequences}, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    scores: dict[str, float] = {}
    for i, (start, end) in enumerate(spans):
        positions = torch.arange(start - 1, end - 1, device=log_probs.device)
        label_tensor = input_ids[i, start:end]
        token_scores = log_probs[i, positions, label_tensor]
        if cfg.score_normalization == "first_token":
            score = token_scores[0]
        elif cfg.score_normalization == "mean":
            score = token_scores.mean()
        else:
            score = token_scores.sum()
        scores[labels[i]] = float(score.detach().cpu())
    decision = max(labels, key=lambda lab: scores[lab])
    return {
        "enabled": True,
        "backend": "adapter_candidate_logprob",
        "adapter_dir": str(cfg.adapter_dir),
        "model_name": str(cfg.model_name),
        "score_normalization": str(cfg.score_normalization),
        "decision": decision,
        "scores": scores,
        "bounded_contract": {
            "can_create_signal": False,
            "can_change_side": False,
            "can_change_hold": False,
            "can_change_size": False,
            "allowed_outputs": ["TRADE", "ABSTAIN"],
        },
    }

def build_rex_live_policy_record(
    enriched: pd.DataFrame,
    features: pd.DataFrame,
    *,
    policy_cfg: RexLivePolicyConfig = RexLivePolicyConfig(),
    execution_cfg: WaveExecutionConfig | None = None,
    scorer_asof: str | pd.Timestamp | None = None,
    selector_cfg: RexLlmSelectorConfig | None = None,
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
    selector_record: dict[str, Any] = {"enabled": False}
    if prediction == "TRADE" and _selector_enabled(selector_cfg):
        assert selector_cfg is not None
        preview_record = {
            "date": str(snapshot["date"]),
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
        }
        try:
            selector_record = _score_rex_llm_selector(preview_record, gates=policy_cfg.gates, cfg=selector_cfg)
            if selector_record.get("decision") != "TRADE":
                prediction = "ABSTAIN"
                reasons.append("llm_selector_block")
            else:
                reasons.append("llm_selector_allow")
        except Exception as exc:
            selector_record = {
                "enabled": True,
                "backend": "adapter_candidate_logprob",
                "adapter_dir": str(selector_cfg.adapter_dir),
                "model_name": str(selector_cfg.model_name),
                "decision": "ABSTAIN" if selector_cfg.fail_closed else "ERROR_ALLOW_FALLBACK",
                "error": str(exc),
                "fail_closed": bool(selector_cfg.fail_closed),
                "require_cuda": bool(selector_cfg.require_cuda),
            }
            if selector_cfg.fail_closed:
                prediction = "ABSTAIN"
                reasons.append("llm_selector_error_fail_closed")
            else:
                reasons.append("llm_selector_error_allow_fallback")
    elif prediction == "TRADE":
        reasons.append("rex_candidate_and_frozen_gate_pass")

    date = str(snapshot["date"])
    if scorer_asof is None:
        age_sec = 0.0
    else:
        snap_ts = pd.Timestamp(snapshot["date"])
        asof_ts = pd.Timestamp(scorer_asof)
        if snap_ts.tzinfo is None:
            snap_ts = snap_ts.tz_localize("UTC")
        else:
            snap_ts = snap_ts.tz_convert("UTC")
        if asof_ts.tzinfo is None:
            asof_ts = asof_ts.tz_localize("UTC")
        else:
            asof_ts = asof_ts.tz_convert("UTC")
        age_sec = max(0.0, float((asof_ts - snap_ts).total_seconds()))
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
        "llm_selector": selector_record,
        "current_close": close,
        "current_atr": current_atr,
        "probability": margin_prob,
        "age_sec": age_sec,
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
    selector_cfg: RexLlmSelectorConfig | None = None,
) -> RexLiveResult:
    """Query DB, score the latest completed bar, and execute/dry-run once."""

    record = await score_rex_live_once(
        execution_cfg=execution_cfg,
        policy_cfg=policy_cfg,
        live_db_cfg=live_db_cfg,
        asof=asof,
        env_path=env_path,
        selector_cfg=selector_cfg,
    )
    decision = decision_from_policy_record(record)
    bridge = WaveExecutionBridge.from_env(config=execution_cfg)
    try:
        execution_result = await bridge.execute_decision(decision)
    finally:
        await bridge.aclose()
    return RexLiveResult(policy_record=record, execution_result=execution_result)


async def score_rex_live_once(
    *,
    execution_cfg: WaveExecutionConfig,
    policy_cfg: RexLivePolicyConfig = RexLivePolicyConfig(),
    live_db_cfg: LiveDbFeatureConfig = LiveDbFeatureConfig(),
    asof: str | pd.Timestamp | None = None,
    env_path: str | Path = ".env",
    selector_cfg: RexLlmSelectorConfig | None = None,
) -> dict[str, Any]:
    """Query DB and score the latest completed bar without placing orders."""

    asof_ts = pd.Timestamp.utcnow() if asof is None else pd.Timestamp(asof)
    if asof_ts.tzinfo is None:
        asof_ts = asof_ts.tz_localize("UTC")
    else:
        asof_ts = asof_ts.tz_convert("UTC")
    engine = sqlalchemy_engine_from_env(env_path)
    frames = query_live_source_frames(engine, asof=asof_ts, cfg=live_db_cfg)
    enriched, features = build_live_feature_frame_from_frames(cfg=live_db_cfg, **frames)
    return build_rex_live_policy_record(
        enriched,
        features,
        policy_cfg=policy_cfg,
        execution_cfg=execution_cfg,
        scorer_asof=asof_ts,
        selector_cfg=selector_cfg,
    )


def _load_loop_state(path: str | Path) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    try:
        loaded = json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_loop_state(path: str | Path, state: dict[str, Any]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def _seconds_until_next_interval(
    now: pd.Timestamp,
    *,
    interval_minutes: int,
    close_delay_sec: float,
) -> float:
    """Seconds until the next completed-candle decision time.

    For a 5-minute interval and 15s close delay, decision times are
    00:00:15, 00:05:15, 00:10:15, ... UTC.  If called exactly on or after one
    decision time, this returns the following interval rather than 0 so a loop
    never spins.
    """

    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    if close_delay_sec < 0:
        raise ValueError("close_delay_sec must be non-negative")
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    interval_sec = int(interval_minutes) * 60
    current_sec = ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1_000_000
    shifted = current_sec - float(close_delay_sec)
    intervals_elapsed = int(np.floor(shifted / interval_sec)) + 1
    next_sec = intervals_elapsed * interval_sec + float(close_delay_sec)
    wait = next_sec - current_sec
    if wait <= 0:
        wait += interval_sec
    return float(wait)


def _emit_loop_status(message: str, *, json_log: bool = False, payload: dict[str, Any] | None = None) -> None:
    """Emit loop progress as either append-only JSON or a single updating line."""

    if json_log:
        print(json.dumps(payload or {"event": message}, ensure_ascii=False, sort_keys=True, default=str), flush=True)
        return
    # Clear the previous line after carriage return.  Keep it short enough for
    # tmux/terminal status monitoring while avoiding append-only log spam.
    print("\r" + message[:240].ljust(240), end="", flush=True)


def _result_status(result: RexLiveResult) -> str:
    record = result.policy_record
    exec_result = result.execution_result
    signal_id = str(record.get("signal_id", ""))
    return (
        f"[rex-live] {pd.Timestamp.utcnow().isoformat()} "
        f"date={record.get('date')} pred={record.get('prediction')} side={record.get('candidate_side')} "
        f"exec={exec_result.get('action')} gate={exec_result.get('gate_reason', exec_result.get('reason', ''))} "
        f"signal={signal_id}"
    )


async def run_rex_live_loop(
    *,
    execution_cfg: WaveExecutionConfig,
    policy_cfg: RexLivePolicyConfig = RexLivePolicyConfig(),
    live_db_cfg: LiveDbFeatureConfig = LiveDbFeatureConfig(),
    env_path: str | Path = ".env",
    loop_cfg: RexLiveLoopConfig = RexLiveLoopConfig(),
    selector_cfg: RexLlmSelectorConfig | None = None,
) -> None:
    """Continuously execute one REX live decision per completed candle.

    The loop stores the latest processed ``signal_id`` in ``loop_cfg.state_file``
    and skips repeats.  This protects restarts near a candle boundary from
    re-submitting the same completed-bar decision.
    """

    iterations = 0
    first = True
    while True:
        if first and loop_cfg.run_immediately:
            first = False
        else:
            first = False
            wait_sec = _seconds_until_next_interval(
                pd.Timestamp.utcnow(),
                interval_minutes=execution_cfg.interval_minutes,
                close_delay_sec=loop_cfg.close_delay_sec,
            )
            _emit_loop_status(
                f"[rex-live] waiting {wait_sec:.1f}s for next {execution_cfg.interval_minutes}m close",
                json_log=loop_cfg.json_log,
                payload={"event": "sleep", "seconds": wait_sec},
            )
            await asyncio.sleep(wait_sec)

        asof_ts = pd.Timestamp.utcnow()
        record = await score_rex_live_once(
            execution_cfg=execution_cfg,
            policy_cfg=policy_cfg,
            live_db_cfg=live_db_cfg,
            asof=asof_ts,
            env_path=env_path,
            selector_cfg=selector_cfg,
        )
        signal_id = str(record.get("signal_id", ""))
        state = _load_loop_state(loop_cfg.state_file)
        if signal_id and state.get("last_signal_id") == signal_id:
            _emit_loop_status(
                f"[rex-live] duplicate skipped asof={asof_ts} signal={signal_id}",
                json_log=loop_cfg.json_log,
                payload={
                    "event": "duplicate_signal_skipped",
                    "signal_id": signal_id,
                    "asof": str(asof_ts),
                },
            )
        else:
            decision = decision_from_policy_record(record)
            bridge = WaveExecutionBridge.from_env(config=execution_cfg)
            try:
                execution_result = await bridge.execute_decision(decision)
            finally:
                await bridge.aclose()
            result = RexLiveResult(policy_record=record, execution_result=execution_result)
            _write_loop_state(
                loop_cfg.state_file,
                {
                    "last_signal_id": signal_id,
                    "last_policy_date": record.get("date"),
                    "last_prediction": record.get("prediction"),
                    "last_execution_action": result.execution_result.get("action"),
                    "updated_at": str(pd.Timestamp.utcnow()),
                },
            )
            _emit_loop_status(
                _result_status(result),
                json_log=loop_cfg.json_log,
                payload=asdict(result),
            )

        iterations += 1
        if loop_cfg.max_iterations is not None and iterations >= loop_cfg.max_iterations:
            if not loop_cfg.json_log:
                print()
            return


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
    parser.add_argument("--rex-selector-adapter-dir", default="", help="Optional bounded REX TRADE/ABSTAIN LoRA adapter directory")
    parser.add_argument("--rex-selector-model-name", default="gemma4-e4b-it")
    parser.add_argument("--rex-selector-score-normalization", choices=["sum", "mean", "first_token"], default="sum")
    parser.add_argument("--rex-selector-fail-open", action="store_true", default=False, help="If set, adapter errors do not block an otherwise valid candidate")
    parser.add_argument("--rex-selector-allow-cpu", action="store_true", default=False, help="Allow selector inference without CUDA; default is fail-closed when CUDA is unavailable")
    parser.add_argument("--manual-regime", choices=["UNKNOWN", "BEAR", "BULL", "SIDEWAYS"], help="Override config manual regime")
    parser.add_argument("--allow-live-orders", action="store_true", default=False, help="Set allow_live_orders=true; still needs --live")
    parser.add_argument("--loop", action="store_true", default=False, help="Keep running and evaluate once per completed interval")
    parser.add_argument("--loop-state-file", default=".omx/state/rex_llm_live_loop_state.json", help="State file used to skip already processed signal_id values")
    parser.add_argument("--close-delay-sec", type=float, default=2.0, help="Seconds after each interval boundary before evaluating the closed candle")
    parser.add_argument("--no-run-immediately", action="store_true", default=False, help="In --loop mode, wait for the next scheduled boundary before first evaluation")
    parser.add_argument("--json-log", action="store_true", default=False, help="In --loop mode, print append-only JSON logs instead of updating one terminal line")
    parser.add_argument("--max-iterations", type=int, default=None, help=argparse.SUPPRESS)
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
    selector_cfg = RexLlmSelectorConfig(
        enabled=bool(args.rex_selector_adapter_dir),
        adapter_dir=str(args.rex_selector_adapter_dir or RexLlmSelectorConfig.adapter_dir),
        model_name=str(args.rex_selector_model_name),
        score_normalization=str(args.rex_selector_score_normalization),
        fail_closed=not bool(args.rex_selector_fail_open),
        require_cuda=not bool(args.rex_selector_allow_cpu),
    )
    if args.loop:
        if args.asof:
            raise SystemExit("--asof cannot be used with --loop because it would keep targeting the same timestamp")
        loop_cfg = RexLiveLoopConfig(
            state_file=Path(args.loop_state_file),
            close_delay_sec=float(args.close_delay_sec),
            run_immediately=not bool(args.no_run_immediately),
            max_iterations=args.max_iterations,
            json_log=bool(args.json_log),
        )
        await run_rex_live_loop(
            execution_cfg=exec_cfg,
            policy_cfg=policy_cfg,
            live_db_cfg=live_db_cfg,
            env_path=args.env,
            loop_cfg=loop_cfg,
            selector_cfg=selector_cfg,
        )
        return
    result = await run_rex_live_once(
        execution_cfg=exec_cfg,
        policy_cfg=policy_cfg,
        live_db_cfg=live_db_cfg,
        asof=args.asof,
        env_path=args.env,
        selector_cfg=selector_cfg,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True, default=str))


def main() -> None:
    asyncio.run(_amain(parse_args()))


if __name__ == "__main__":
    main()
