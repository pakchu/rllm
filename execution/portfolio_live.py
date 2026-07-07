"""Live executor for fixed-weight portfolio sleeve candidates.

This runner is intentionally narrower than a general portfolio engine.  It
supports the current gross<=6 BTCUSDT candidate family by:

* building one latest 5m feature row from the live DB, including open interest,
* evaluating fixed sleeve gates from ``configs/live/portfolio_gross6_*.json``,
* allocating margin by fixed sleeve weights against the research leverage
  budget (weight/leverage by default),
* placing hedge-mode LONG/SHORT maker orders through ``wave_trading``, and
* tracking per-sleeve exit timestamps in a local ledger.

It does not select new weights, change sleeve rules, or net active sleeves.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import select
import time
from dataclasses import dataclass, asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from execution.rex_llm_live import RexLivePolicyConfig, RexLlmSelectorConfig, build_rex_live_policy_record
from execution.wave_execution import (
    WaveExecutionConfig,
    _StaticSignalGenerator,
    _load_api_credentials,
    load_wave_execution_classes,
)
from preprocessing.binance_aux_features import attach_binance_um_aux_frames
from preprocessing.external_features import attach_external_features, build_external_feature_frame, DXY_WEIGHTS
from preprocessing.live_db_features import (
    LiveDbFeatureConfig,
    _normalise_bar_frame,
    query_live_source_frames,
    resample_market_bars,
    sqlalchemy_engine_from_env,
)
from preprocessing.market_features import build_market_feature_frame
from training.evaluate_oi_llm_selector import _context_id, _tokens
from training.evaluate_portfolio_llm_selector import _base_context_tokens
from training.long_regime_interest_gate_validation import build_interest_features
from training.long_regime_score_gate_validation import _build_score_frame, _score_variant


Side = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class PortfolioLiveConfig:
    portfolio_config: Path
    execution_config: Path
    env_path: Path = Path(".env")
    state_file: Path = Path(".omx/state/portfolio_live_state.json")
    lookback_minutes: int = 45_000
    close_delay_sec: float = 0.25
    max_freshness_wait_sec: float = 8.0
    freshness_poll_sec: float = 0.5
    freshness_notify_channel: str = "market_data_bar"
    run_immediately: bool = False
    live: bool = False
    allow_live_orders: bool = False
    leverage: int = 6
    allocation_mode: Literal["research_gross", "normalize_weights"] = "research_gross"
    max_iterations: int | None = None
    entry_timeout_fraction: float = 0.25
    max_entry_wait_sec: int = 300
    cancel_stale_open_orders: bool = True
    portfolio_selector_overlay: Path | None = None
    rex_selector_adapter_dir: Path | None = None
    rex_selector_model_name: str = "gemma4-e4b-it"
    rex_selector_score_normalization: str = "sum"
    rex_selector_fail_closed: bool = True
    rex_selector_require_cuda: bool = True


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text())


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def _load_state(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"open_sleeves": {}, "processed_signals": {}}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"open_sleeves": {}, "processed_signals": {}}
    if not isinstance(data, dict):
        return {"open_sleeves": {}, "processed_signals": {}}
    data.setdefault("open_sleeves", {})
    data.setdefault("processed_signals", {})
    return data


def _gate_pass(row: pd.Series, gates: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok = True
    for gate in gates:
        feature = str(gate["feature"])
        op = str(gate["op"])
        threshold = float(gate["threshold"])
        value = float(row.get(feature, np.nan))
        passed = np.isfinite(value) and ((value >= threshold) if op in {">=", "ge"} else (value <= threshold))
        reasons.append(f"{feature}={value:.6g}{op}{threshold:.6g}:{'pass' if passed else 'fail'}")
        ok &= bool(passed)
    return ok, reasons


def _interval_slot(ts: pd.Timestamp, stride_bars: int, interval_minutes: int = 5) -> bool:
    """Return True when a timestamp is on the sleeve's stride grid."""

    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    minutes = int(t.timestamp() // 60)
    return (minutes // int(interval_minutes)) % int(stride_bars) == 0


async def _query_oi(engine: Any, *, asof: pd.Timestamp, start: pd.Timestamp, symbol: str) -> pd.DataFrame:
    from sqlalchemy import text

    sql = text(
        """
        SELECT ts AS date, sum_open_interest AS open_interest
        FROM open_interest_binance
        WHERE symbol = :symbol AND period = '5m' AND ts >= :start AND ts <= :asof
        ORDER BY ts
        """
    )
    with engine.connect() as conn:
        return pd.read_sql_query(sql, conn, params={"symbol": symbol, "start": start.to_pydatetime(), "asof": asof.to_pydatetime()})


async def build_live_portfolio_frames(
    *,
    engine: Any,
    asof: pd.Timestamp,
    cfg: LiveDbFeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build enriched market and feature frame with live OI included."""

    start = asof - pd.Timedelta(minutes=int(cfg.lookback_minutes))
    frames = query_live_source_frames(engine, asof=asof, cfg=cfg)
    oi = await _query_oi(engine, asof=asof, start=start, symbol=cfg.symbol)

    market = resample_market_bars(frames["btcusdt_1m"], cfg.decision_interval)
    if oi.empty:
        raise RuntimeError("open_interest_binance returned no rows; portfolio OI sleeves cannot run")
    oi = oi.copy()
    oi["date"] = pd.to_datetime(oi["date"], utc=True).dt.tz_convert(None)
    oi["open_interest"] = pd.to_numeric(oi["open_interest"], errors="coerce")
    market = pd.merge_asof(
        market.sort_values("date"),
        oi[["date", "open_interest"]].sort_values("date"),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("10min"),
    )

    btckrw = _normalise_bar_frame(frames["btckrw_1m"], tic="KRW-BTC")
    usdkrw = _normalise_bar_frame(frames["usdkrw_1m"], tic="USDKRW")
    forex = _normalise_bar_frame(frames["forex_1m"])
    external = build_external_feature_frame(
        market,
        forex_bars=forex,
        btckrw_bars=btckrw,
        usdkrw_bars=usdkrw,
        interval=cfg.decision_interval,
        include_forex_components=cfg.include_forex_components,
    )
    enriched = attach_external_features(
        market,
        external,
        tolerance=cfg.external_tolerance,
        zscore_window=cfg.zscore_window,
        momentum_period=cfg.zscore_window,
    )
    enriched = attach_binance_um_aux_frames(
        enriched,
        funding_frame=frames["funding"],
        premium_frame=frames["premium_1m"],
        funding_tolerance=cfg.funding_tolerance,
        premium_tolerance=cfg.premium_tolerance,
        zscore_window=cfg.zscore_window,
    )
    features = build_market_feature_frame(
        enriched,
        window_size=cfg.feature_window_size,
        zscore_window=cfg.zscore_window,
        volume_window=cfg.volume_window,
    ).copy()

    oi_s = pd.Series(enriched["open_interest"].astype(float).replace(0, np.nan).ffill(), index=enriched.index)
    px = pd.Series(enriched["close"].astype(float), index=enriched.index)
    for w, name in [(6, "30m"), (12, "1h"), (24, "2h"), (48, "4h"), (96, "8h")]:
        oi_ret = np.log(oi_s / oi_s.shift(w)).replace([np.inf, -np.inf], np.nan)
        px_ret = np.log(px / px.shift(w)).replace([np.inf, -np.inf], np.nan)
        div = oi_ret - px_ret
        for nm, series in [
            (f"oi_ret_{name}", oi_ret),
            (f"px_ret_{name}", px_ret),
            (f"oi_minus_px_{name}", div),
            (f"px_minus_oi_{name}", px_ret - oi_ret),
        ]:
            mu = series.rolling(288, min_periods=50).mean()
            sd = series.rolling(288, min_periods=50).std(ddof=0)
            features[nm] = series
            features[nm + "_z"] = ((series - mu) / sd.replace(0, np.nan)).clip(-5, 5)

    # Restore activity_flow_htf used by some portfolio selector experiments.
    try:
        interest = build_interest_features(enriched, features)
        raw = _build_score_frame(enriched, features, interest)
        train_mask = np.ones(len(enriched), dtype=bool)
        score, _ = _score_variant(raw, train_mask, "activity_flow_htf")
        features["activity_flow_htf"] = score
    except Exception:
        pass

    return enriched, features.replace([np.inf, -np.inf], np.nan)


def _current_atr(enriched: pd.DataFrame, period: int = 15) -> float:
    high = pd.to_numeric(enriched["high"], errors="coerce")
    low = pd.to_numeric(enriched["low"], errors="coerce")
    close = pd.to_numeric(enriched["close"], errors="coerce")
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out = float(tr.rolling(max(1, int(period)), min_periods=1).mean().iloc[-1])
    return out if np.isfinite(out) else 0.0



def _load_portfolio_selector_overlay(portfolio: dict[str, Any], explicit_path: Path | None) -> dict[str, Any] | None:
    """Load a bounded portfolio-level selector overlay if configured.

    The selector is intentionally constrained to ALLOW/BLOCK_RISK for already
    triggered sleeves.  It cannot create signals, alter exits, resize sleeves,
    or change leverage.
    """

    raw_path = explicit_path or portfolio.get("portfolio_selector_overlay") or portfolio.get("llm_selector_overlay")
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"portfolio selector overlay not found: {path}")
    overlay = _load_json(path)
    sel = overlay.get("symbolic_proxy")
    if not isinstance(sel, dict):
        raise ValueError(f"portfolio selector overlay missing symbolic_proxy: {path}")
    keys = sel.get("context_keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError(f"portfolio selector overlay missing context_keys: {path}")
    return {**overlay, "_path": str(path)}


def _apply_portfolio_selector_overlay(
    sleeve_scores: list[dict[str, Any]],
    *,
    overlay: dict[str, Any] | None,
    enriched: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, Any] | None:
    """Apply the bounded LLM-selector proxy to pending live entries.

    This mirrors ``training.evaluate_portfolio_llm_selector``: build the same
    compact state tokens from signal-time features, compute a context id, and
    block only if that train-only bad context is in the frozen overlay. Existing
    open sleeves are not force-closed; only new entries are suppressed.
    """

    active = [s for s in sleeve_scores if s.get("active")]
    if not overlay or not active:
        return None
    sel = overlay.get("symbolic_proxy", {})
    keys = tuple(str(k) for k in sel.get("context_keys", []))
    blocked = {str(x.get("context_id")) for x in sel.get("blocked_contexts", []) if isinstance(x, dict)}
    pos = len(features) - 1
    tokens = _base_context_tokens(pos, market=enriched, feat=features)
    cid = _context_id(tokens, keys) if keys else ""
    allowed = cid not in blocked
    action = "ALLOW" if allowed else "BLOCK_RISK"
    pending = [str(s.get("name")) for s in active]
    record = {
        "selector": str(overlay.get("name") or overlay.get("_path") or "portfolio_selector_overlay"),
        "overlay_path": overlay.get("_path"),
        "output_space": overlay.get("output_space", ["ALLOW", "BLOCK_RISK"]),
        "action": action,
        "allowed": allowed,
        "context_id": cid,
        "context_keys": list(keys),
        "pending_sleeves": pending,
        "state_tokens": tokens,
        "bounded_contract": {
            "can_create_signals": False,
            "can_change_side": False,
            "can_change_size": False,
            "can_change_exit": False,
            "applies_to": "new_entries_only",
        },
    }
    reason = f"portfolio_selector_context={cid}:{action}"
    for sleeve in active:
        sleeve.setdefault("reasons", []).append(reason)
        sleeve["portfolio_selector"] = {k: v for k, v in record.items() if k != "state_tokens"}
        if not allowed:
            sleeve["active"] = False
    return record

def _score_sleeves(
    *,
    portfolio: dict[str, Any],
    enriched: pd.DataFrame,
    features: pd.DataFrame,
    exec_cfg: WaveExecutionConfig,
    asof: pd.Timestamp,
    rex_selector_cfg: RexLlmSelectorConfig | None = None,
) -> list[dict[str, Any]]:
    row = features.iloc[-1]
    ts = pd.Timestamp(enriched.iloc[-1]["date"])
    close = float(enriched.iloc[-1]["close"])
    atr = _current_atr(enriched, exec_cfg.atr_period)
    out: list[dict[str, Any]] = []
    for sleeve in portfolio["base_sleeves"]:
        name = str(sleeve["name"])
        source = str(sleeve["source"])
        weight = float(sleeve["weight"])
        side = str(sleeve["side"]).upper()
        active = False
        reasons: list[str] = []
        hold = 0
        stride = 1

        if source.endswith(".json") and Path(source).exists():
            cfg = _load_json(source)
            overlay_cfg = cfg if "selector_overlay" in source else None
            if "base_candidate" in cfg and "gates" not in cfg and "signal" not in cfg:
                cfg = _load_json(str(cfg["base_candidate"]))
            if "signal" in cfg:
                gates = cfg["signal"]["gates"]
                hold = int(cfg["signal"].get("hold_bars_5m", cfg["signal"].get("hold_bars", 0)))
                stride = int(cfg["signal"].get("stride_bars_5m", cfg["signal"].get("stride_bars", 1)))
            else:
                gates = cfg["gates"]
                hold = int(cfg.get("hold_bars", cfg.get("hold_bars_5m", 0)))
                stride = int(cfg.get("stride_bars", cfg.get("stride_bars_5m", 1)))
            gate_ok, reasons = _gate_pass(row, gates)
            stride_ok = _interval_slot(ts, stride, exec_cfg.interval_minutes)
            active = bool(gate_ok and stride_ok)
            reasons.append(f"stride={stride}:{'pass' if stride_ok else 'fail'}")

            if overlay_cfg is not None:
                sel = overlay_cfg.get("symbolic_proxy", {})
                blocked = {x["context_id"] for x in sel.get("blocked_contexts", [])}
                keys = tuple(sel.get("context_keys", []))
                cid = _context_id(_tokens(len(features) - 1, market=enriched, feat=features), keys) if keys else ""
                selector_ok = cid not in blocked
                active &= selector_ok
                reasons.append(f"selector_context={cid}:{'ALLOW' if selector_ok else 'BLOCK'}")
        else:
            # Bear REX short sleeve uses the frozen REX live policy path.
            record = build_rex_live_policy_record(
                enriched,
                features,
                policy_cfg=RexLivePolicyConfig(),
                execution_cfg=exec_cfg,
                scorer_asof=asof,
                selector_cfg=rex_selector_cfg,
            )
            active = record.get("prediction") == "TRADE" and record.get("candidate_side") == "SHORT"
            hold = int(record.get("action", {}).get("hold_bars", 144))
            stride = 1
            reasons = [str(record.get("reason", ""))]

        out.append(
            {
                "name": name,
                "source": source,
                "side": side,
                "weight": weight,
                "active": active,
                "hold_bars": hold,
                "stride_bars": stride,
                "date": str(ts),
                "current_close": close,
                "current_atr": atr,
                "signal_id": f"{name}:{ts.isoformat()}",
                "reasons": reasons,
            }
        )
    return out


def _seconds_until_next_interval(now: pd.Timestamp, *, interval_minutes: int, close_delay_sec: float) -> float:
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    interval_sec = int(interval_minutes) * 60
    current_sec = ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1_000_000
    shifted = current_sec - float(close_delay_sec)
    next_sec = (int(np.floor(shifted / interval_sec)) + 1) * interval_sec + float(close_delay_sec)
    wait = next_sec - current_sec
    return float(wait if wait > 0 else wait + interval_sec)


def _status(msg: str) -> None:
    print("\r" + msg[:240].ljust(240), end="", flush=True)



PORTFOLIO_ORDER_PREFIX = "rpf"


def _entry_ttl_seconds(
    sleeve: dict[str, Any],
    *,
    interval_minutes: int,
    timeout_fraction: float,
    max_entry_wait_sec: int,
) -> int:
    """Bound post-only entry waiting by the sleeve's signal cycle.

    Entry freshness is tied to the sleeve stride because a missed maker entry
    should not keep chasing after the next decision opportunity is near.  Hold
    bars are only used as a fallback for sleeves without a positive stride.
    """

    stride = int(sleeve.get("stride_bars") or 0)
    hold = int(sleeve.get("hold_bars") or 0)
    cycle_bars = stride if stride > 0 else max(1, hold)
    cycle_sec = max(1, cycle_bars) * int(interval_minutes) * 60
    ttl = int(cycle_sec * max(0.0, float(timeout_fraction)))
    ttl = max(30, ttl)
    if max_entry_wait_sec > 0:
        ttl = min(ttl, int(max_entry_wait_sec))
    return ttl


def _portfolio_sleeve_key(sleeve_name: str) -> str:
    return hashlib.sha1(str(sleeve_name).encode("utf-8")).hexdigest()[:6]


def _portfolio_client_order_id(signal_id: str, *, sleeve_name: str, now_sec: int | None = None) -> str:
    """Create a Binance-safe, bot-owned client order id.

    Binance's limit is short; this prefix lets stale-order cleanup avoid
    touching unrelated bots that may share the same symbol/account.
    """

    epoch = int(now_sec if now_sec is not None else time.time())
    sleeve_key = _portfolio_sleeve_key(sleeve_name)
    digest = hashlib.sha1(signal_id.encode("utf-8")).hexdigest()[:8]
    return f"{PORTFOLIO_ORDER_PREFIX}_{epoch}_{sleeve_key}_{digest}"


def _order_timestamp_ms(order: dict[str, Any]) -> int | None:
    for key in ("time", "updateTime", "workingTime"):
        value = order.get(key)
        if value in (None, ""):
            continue
        try:
            value_i = int(value)
        except (TypeError, ValueError):
            continue
        if value_i > 0:
            return value_i
    cid = str(order.get("clientOrderId") or order.get("origClientOrderId") or "")
    parts = cid.split("_")
    if len(parts) >= 2 and parts[0] == PORTFOLIO_ORDER_PREFIX:
        try:
            return int(parts[1]) * 1000
        except ValueError:
            return None
    return None


async def _cancel_stale_portfolio_orders(
    *,
    client: Any,
    symbol: str,
    now: pd.Timestamp,
    max_age_sec: int,
) -> list[dict[str, Any]]:
    """Cancel only this runner's stale post-only orders.

    Other live bots can share BTCUSDT, so cleanup is restricted to the
    client-order-id prefix generated by this module.
    """

    if max_age_sec <= 0:
        return []
    now_ms = int(pd.Timestamp(now).timestamp() * 1000)
    cancelled: list[dict[str, Any]] = []
    try:
        open_orders = await client.get_open_orders(symbol)
    except Exception as exc:
        return [{"status": "scan_failed", "error": str(exc)}]
    for order in open_orders:
        cid = str(order.get("clientOrderId") or "")
        if not cid.startswith(PORTFOLIO_ORDER_PREFIX + "_"):
            continue
        ts_ms = _order_timestamp_ms(order)
        if ts_ms is None or (now_ms - ts_ms) < int(max_age_sec) * 1000:
            continue
        try:
            result = await client.cancel_order(symbol, client_order_id=cid)
            cancelled.append(
                {
                    "status": "cancelled",
                    "order_id": order.get("orderId"),
                    "client_order_id": cid,
                    "age_sec": round((now_ms - ts_ms) / 1000, 3),
                    "result": result,
                }
            )
        except Exception as exc:
            if "-2011" not in str(exc):
                cancelled.append(
                    {
                        "status": "cancel_failed",
                        "order_id": order.get("orderId"),
                        "client_order_id": cid,
                        "age_sec": round((now_ms - ts_ms) / 1000, 3),
                        "error": str(exc),
                    }
                )
    return cancelled


async def _cancel_portfolio_orders_for_sleeve(
    *,
    client: Any,
    symbol: str,
    sleeve_name: str,
    reason: str,
) -> list[dict[str, Any]]:
    """Cancel this runner's open entry orders for one sleeve before replacing them."""

    sleeve_key = _portfolio_sleeve_key(sleeve_name)
    target_prefix = f"{PORTFOLIO_ORDER_PREFIX}_"
    cancelled: list[dict[str, Any]] = []
    try:
        open_orders = await client.get_open_orders(symbol)
    except Exception as exc:
        return [{"status": "scan_failed", "sleeve": sleeve_name, "reason": reason, "error": str(exc)}]
    for order in open_orders:
        cid = str(order.get("clientOrderId") or "")
        parts = cid.split("_")
        if not (cid.startswith(target_prefix) and len(parts) >= 4 and parts[2] == sleeve_key):
            continue
        try:
            result = await client.cancel_order(symbol, client_order_id=cid)
            cancelled.append(
                {
                    "status": "cancelled",
                    "reason": reason,
                    "sleeve": sleeve_name,
                    "order_id": order.get("orderId"),
                    "client_order_id": cid,
                    "result": result,
                }
            )
        except Exception as exc:
            if "-2011" not in str(exc):
                cancelled.append(
                    {
                        "status": "cancel_failed",
                        "reason": reason,
                        "sleeve": sleeve_name,
                        "order_id": order.get("orderId"),
                        "client_order_id": cid,
                        "error": str(exc),
                    }
                )
    return cancelled


async def _place_portfolio_maker_order_with_deadline(
    *,
    client: Any,
    executor: Any,
    exec_cfg: WaveExecutionConfig,
    order_side: Literal["BUY", "SELL"],
    quantity: Decimal,
    position_side: Side,
    signal_id: str,
    sleeve_name: str,
    ttl_sec: int,
    poll_interval_sec: float = 1.0,
) -> dict[str, Any]:
    """Place one post-only entry and cancel unfilled remainder by deadline."""

    maker_price = await executor.get_maker_price(order_side, None)
    client_order_id = _portfolio_client_order_id(signal_id, sleeve_name=sleeve_name)
    started = pd.Timestamp.utcnow()
    try:
        order = await client.place_order(
            symbol=exec_cfg.symbol,
            side=order_side,
            order_type="LIMIT",
            quantity=float(quantity),
            price=float(maker_price),
            time_in_force="GTX",
            reduce_only=False,
            client_order_id=client_order_id,
            position_side=position_side,
        )
    except Exception as exc:
        return {
            "status": "REJECTED",
            "client_order_id": client_order_id,
            "requested_quantity": str(quantity),
            "filled_quantity": "0",
            "avg_price": str(maker_price),
            "ttl_sec": int(ttl_sec),
            "error": str(exc),
        }

    order_id = order.get("orderId")
    deadline = asyncio.get_event_loop().time() + max(0, int(ttl_sec))
    last_status = dict(order)
    filled_qty = Decimal("0")
    avg_price = Decimal(str(maker_price))
    terminal_statuses = {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}

    while True:
        try:
            last_status = await client.get_order(exec_cfg.symbol, order_id=order_id)
        except Exception as exc:
            if "-2013" in str(exc):
                last_status = {**last_status, "status": "UNKNOWN", "query_error": str(exc)}
                break
            last_status = {**last_status, "query_error": str(exc)}
        status = str(last_status.get("status") or "UNKNOWN")
        try:
            filled_qty = Decimal(str(last_status.get("executedQty", "0") or "0"))
        except Exception:
            filled_qty = Decimal("0")
        try:
            reported_avg = Decimal(str(last_status.get("avgPrice", "0") or "0"))
            if reported_avg > 0:
                avg_price = reported_avg
        except Exception:
            pass
        if status == "FILLED" or status in terminal_statuses or asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(max(0.05, float(poll_interval_sec)))

    final_status = str(last_status.get("status") or "UNKNOWN")
    cancel_result: dict[str, Any] | None = None
    if final_status not in terminal_statuses or final_status == "PARTIALLY_FILLED":
        try:
            cancel_result = await client.cancel_order(exec_cfg.symbol, client_order_id=client_order_id)
            if final_status not in {"FILLED", "PARTIALLY_FILLED"}:
                final_status = "TIMEOUT_CANCELLED"
            elif final_status == "PARTIALLY_FILLED":
                final_status = "PARTIAL_CANCELLED"
        except Exception as exc:
            if "-2011" not in str(exc):
                cancel_result = {"status": "cancel_failed", "error": str(exc)}

    return {
        "status": final_status,
        "order_id": order_id,
        "client_order_id": client_order_id,
        "requested_quantity": str(quantity),
        "filled_quantity": str(filled_qty),
        "avg_price": str(avg_price),
        "price": str(maker_price),
        "ttl_sec": int(ttl_sec),
        "started_at": str(started),
        "deadline_at": str(started + pd.Timedelta(seconds=int(ttl_sec))),
        "cancel_result": cancel_result,
        "raw_order": order,
        "last_status": last_status,
    }


def _margin_fraction_for_weight(
    *,
    weight: float,
    total_weight: float,
    leverage_budget: float,
    allocation_mode: str,
) -> float:
    """Return account-equity margin fraction for a sleeve weight.

    Research portfolio metrics apply sleeve weights directly as notional
    exposure per 1.0 account equity.  On an exchange account with leverage L,
    matching that assumption requires margin_fraction = weight / L.

    ``normalize_weights`` intentionally scales all weights to consume the full
    leverage budget even when gross_weight < leverage_budget; this is more
    aggressive than the saved research candidate.
    """

    if weight < 0:
        raise ValueError("sleeve weight must be non-negative")
    if leverage_budget <= 0:
        raise ValueError("leverage_budget must be positive")
    if allocation_mode == "research_gross":
        return float(weight) / float(leverage_budget)
    if allocation_mode == "normalize_weights":
        if total_weight <= 0:
            raise ValueError("total_weight must be positive for normalize_weights")
        return float(weight) / float(total_weight)
    raise ValueError(f"unsupported allocation_mode: {allocation_mode}")


def _allocation_audit(portfolio: dict[str, Any], *, leverage_budget: float, allocation_mode: str) -> dict[str, Any]:
    sleeves = portfolio.get("base_sleeves", [])
    total_weight = float(sum(float(s["weight"]) for s in sleeves))
    rows = []
    for sleeve in sleeves:
        w = float(sleeve["weight"])
        mf = _margin_fraction_for_weight(
            weight=w,
            total_weight=total_weight,
            leverage_budget=leverage_budget,
            allocation_mode=allocation_mode,
        )
        rows.append(
            {
                "name": sleeve["name"],
                "side": sleeve["side"],
                "research_weight_notional_per_equity": w,
                "margin_fraction": mf,
                "live_notional_per_equity_at_budget": mf * float(leverage_budget),
                "research_match": abs((mf * float(leverage_budget)) - w) < 1e-9,
            }
        )
    return {
        "allocation_mode": allocation_mode,
        "leverage_budget": float(leverage_budget),
        "research_gross_weight": total_weight,
        "margin_fraction_sum_if_all_active": float(sum(r["margin_fraction"] for r in rows)),
        "live_gross_if_all_active": float(sum(r["live_notional_per_equity_at_budget"] for r in rows)),
        "unused_margin_fraction_vs_full_budget": max(0.0, 1.0 - float(sum(r["margin_fraction"] for r in rows))),
        "sleeves": rows,
        "notes": [
            "research_gross mode matches saved research weights exactly and leaves unused leverage capacity when gross_weight < leverage_budget",
            "normalize_weights consumes 100% margin but scales risk/return versus research when gross_weight != leverage_budget",
        ],
    }


async def _make_executor(exec_cfg: WaveExecutionConfig):
    key, secret = _load_api_credentials(exec_cfg.testnet, exec_cfg.wave_trading_path, dry_run=exec_cfg.dry_run)
    client_cls, executor_cls = load_wave_execution_classes(exec_cfg.wave_trading_path)
    client = client_cls(api_key=key, api_secret=secret, testnet=exec_cfg.testnet)
    signal_generator = _StaticSignalGenerator(
        atr_period=exec_cfg.atr_period,
        pt_mult=exec_cfg.pt_mult,
        max_holding_bars=exec_cfg.max_holding_bars,
    )
    executor = executor_cls(
        client=client,
        signal_generator=signal_generator,
        symbol=exec_cfg.symbol,
        leverage=exec_cfg.leverage,
        position_size_pct=1.0,
        maker_offset_pct=exec_cfg.maker_offset_pct,
        max_retries=exec_cfg.max_retries,
        order_timeout_sec=exec_cfg.order_timeout_sec,
    )
    await client.sync_time()
    if not await client.is_hedge_mode(force_refresh=True):
        raise RuntimeError("Binance account must be in hedge mode for distributed LONG/SHORT portfolio execution")
    await client.set_leverage(exec_cfg.symbol, exec_cfg.leverage)
    return client, executor


async def _open_sleeve(
    *,
    client: Any,
    executor: Any,
    exec_cfg: WaveExecutionConfig,
    sleeve: dict[str, Any],
    margin_fraction: float,
    entry_ttl_sec: int,
) -> dict[str, Any]:
    balance = await client.get_usdt_balance()
    total_equity = float(balance["total"])
    price = await client.get_ticker_price(exec_cfg.symbol)
    notional = total_equity * float(margin_fraction) * float(exec_cfg.leverage)
    quantity = Decimal(str(notional / float(price)))
    side: Side = sleeve["side"]
    order_side = "BUY" if side == "LONG" else "SELL"
    order = await _place_portfolio_maker_order_with_deadline(
        client=client,
        executor=executor,
        exec_cfg=exec_cfg,
        order_side=order_side,
        quantity=quantity,
        position_side=side,
        signal_id=sleeve["signal_id"],
        sleeve_name=str(sleeve["name"]),
        ttl_sec=entry_ttl_sec,
    )
    return {
        "order": order,
        "requested_quantity": str(quantity),
        "filled_quantity": order.get("filled_quantity", "0"),
        "entry_ttl_sec": int(entry_ttl_sec),
        "notional": notional,
        "margin_fraction": margin_fraction,
        "equity_basis": total_equity,
    }

async def _close_sleeve(*, executor: Any, sleeve_state: dict[str, Any], exec_cfg: WaveExecutionConfig) -> dict[str, Any]:
    side: Side = sleeve_state["side"]
    close_side = "SELL" if side == "LONG" else "BUY"
    quantity = Decimal(str(sleeve_state["quantity"]))
    order = await executor.place_maker_order_with_retry(
        close_side,
        quantity,
        reduce_only=True,
        max_retries=None,
        signal_id=str(sleeve_state.get("signal_id", "portfolio-exit")),
        position_side=side,
        order_type="EXIT",
    )
    return {"order": order, "closed_quantity": str(quantity)}



def _expected_decision_bar(now: pd.Timestamp, *, interval_minutes: int) -> pd.Timestamp:
    ts = pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    interval = f"{int(interval_minutes)}min"
    return ts.floor(interval) - pd.Timedelta(minutes=int(interval_minutes))


def _validate_pg_notify_channel(channel: str) -> str:
    name = str(channel or "").strip()
    if not name or not name.replace("_", "a").isalnum() or not name[0].isalpha():
        raise ValueError(f"invalid Postgres notify channel: {channel!r}")
    return name


def _latest_binance_1m_ts(engine: Any, *, symbol: str) -> pd.Timestamp | None:
    from sqlalchemy import text

    sql = text(
        """
        SELECT MAX(ts) AS max_ts
        FROM bars_binance
        WHERE symbol = :symbol AND interval = '1m'
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"symbol": symbol}).mappings().one()
    value = row.get("max_ts")
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _wait_for_bar_update_notify(
    engine: Any,
    *,
    symbol: str,
    required_1m: pd.Timestamp,
    max_wait_sec: float,
    channel: str,
) -> tuple[float, pd.Timestamp | None, bool]:
    """Block on Postgres LISTEN/NOTIFY until the required 1m bar is committed."""

    deadline = time.monotonic() + max(0.0, float(max_wait_sec))
    latest: pd.Timestamp | None = None
    raw = engine.raw_connection()
    try:
        dbapi_conn = getattr(raw, "driver_connection", None) or getattr(raw, "connection", raw)
        if hasattr(dbapi_conn, "autocommit"):
            dbapi_conn.autocommit = True
        cur = raw.cursor()
        channel = _validate_pg_notify_channel(channel)
        cur.execute(f"LISTEN {channel}")

        # Check after LISTEN to avoid missing a notification between the initial
        # schedule wake-up and listener registration.
        latest = _latest_binance_1m_ts(engine, symbol=symbol)
        if latest is not None and latest >= required_1m:
            return 0.0, latest, True

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            readable, _, _ = select.select([dbapi_conn], [], [], remaining)
            if not readable:
                break
            dbapi_conn.poll()
            while getattr(dbapi_conn, "notifies", []):
                notify = dbapi_conn.notifies.pop(0)
                try:
                    payload = json.loads(notify.payload)
                except Exception:
                    continue
                if (
                    payload.get("table") != "bars_binance"
                    or str(payload.get("symbol", "")).upper() != symbol.upper()
                    or payload.get("interval") != "1m"
                ):
                    continue
                ts = pd.Timestamp(payload.get("ts"))
                ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                latest = ts if latest is None or ts > latest else latest
                if latest >= required_1m:
                    return max(0.0, time.monotonic() - (deadline - max(0.0, float(max_wait_sec)))), latest, True

        latest = _latest_binance_1m_ts(engine, symbol=symbol)
        return max(0.0, time.monotonic() - (deadline - max(0.0, float(max_wait_sec)))), latest, bool(latest is not None and latest >= required_1m)
    finally:
        try:
            raw.close()
        except Exception:
            pass


async def _wait_for_expected_1m_tail(
    *,
    engine: Any,
    symbol: str,
    asof: pd.Timestamp,
    interval_minutes: int,
    max_wait_sec: float,
    notify_channel: str,
) -> tuple[float, pd.Timestamp | None, pd.Timestamp, str]:
    """Wait for ingest's Postgres notification for the expected decision bar."""

    expected_bar = _expected_decision_bar(asof, interval_minutes=interval_minutes)
    required_1m = expected_bar + pd.Timedelta(minutes=max(0, int(interval_minutes) - 1))
    waited, latest, fresh = await asyncio.to_thread(
        _wait_for_bar_update_notify,
        engine,
        symbol=symbol,
        required_1m=required_1m,
        max_wait_sec=max_wait_sec,
        channel=notify_channel,
    )
    return waited, latest, expected_bar, "notify" if fresh else "notify_timeout"


async def run_portfolio_loop(cfg: PortfolioLiveConfig) -> None:
    portfolio = _load_json(cfg.portfolio_config)
    selector_overlay = _load_portfolio_selector_overlay(portfolio, cfg.portfolio_selector_overlay)
    rex_selector_cfg = RexLlmSelectorConfig(
        enabled=bool(cfg.rex_selector_adapter_dir),
        adapter_dir=str(cfg.rex_selector_adapter_dir or RexLlmSelectorConfig.adapter_dir),
        model_name=str(cfg.rex_selector_model_name),
        score_normalization=str(cfg.rex_selector_score_normalization),
        fail_closed=bool(cfg.rex_selector_fail_closed),
        require_cuda=bool(cfg.rex_selector_require_cuda),
    )
    exec_cfg_raw = WaveExecutionConfig.from_json(cfg.execution_config)
    exec_cfg = WaveExecutionConfig(
        **{
            **asdict(exec_cfg_raw),
            "dry_run": not cfg.live,
            "allow_live_orders": bool(cfg.allow_live_orders),
            "leverage": int(cfg.leverage),
            "position_size_pct": 1.0,
            "allowed_signals": ("LONG", "SHORT"),
            "require_flat_position": False,
            "require_no_open_orders": True,
        }
    )
    if cfg.live and not cfg.allow_live_orders:
        raise SystemExit("--live requires --allow-live-orders")
    weights = {str(s["name"]): float(s["weight"]) for s in portfolio["base_sleeves"]}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise RuntimeError("portfolio weights sum to zero")
    audit = _allocation_audit(portfolio, leverage_budget=float(cfg.leverage), allocation_mode=cfg.allocation_mode)

    engine = sqlalchemy_engine_from_env(cfg.env_path)
    client = executor = None
    if not exec_cfg.dry_run:
        client, executor = await _make_executor(exec_cfg)
    first = True
    try:
        iterations = 0
        while True:
            if first and cfg.run_immediately:
                first = False
            else:
                first = False
                wait = _seconds_until_next_interval(pd.Timestamp.utcnow(), interval_minutes=exec_cfg.interval_minutes, close_delay_sec=cfg.close_delay_sec)
                _status(f"[portfolio-live] waiting {wait:.1f}s for next {exec_cfg.interval_minutes}m close")
                await asyncio.sleep(wait)

            asof = pd.Timestamp.utcnow()
            if asof.tzinfo is None:
                asof = asof.tz_localize("UTC")
            live_cfg = LiveDbFeatureConfig(lookback_minutes=int(cfg.lookback_minutes))
            freshness_waited, latest_1m_ts, expected_bar, freshness_mode = await _wait_for_expected_1m_tail(
                engine=engine,
                symbol=live_cfg.symbol,
                asof=asof,
                interval_minutes=exec_cfg.interval_minutes,
                max_wait_sec=cfg.max_freshness_wait_sec,
                notify_channel=cfg.freshness_notify_channel,
            )
            frame_t0 = time.perf_counter()
            enriched, features = await build_live_portfolio_frames(engine=engine, asof=asof, cfg=live_cfg)
            frame_build_sec = time.perf_counter() - frame_t0
            sleeve_scores = _score_sleeves(
                portfolio=portfolio,
                enriched=enriched,
                features=features,
                exec_cfg=exec_cfg,
                asof=asof,
                rex_selector_cfg=rex_selector_cfg,
            )
            portfolio_selector_record = _apply_portfolio_selector_overlay(
                sleeve_scores,
                overlay=selector_overlay,
                enriched=enriched,
                features=features,
            )
            state = _load_state(cfg.state_file)
            if portfolio_selector_record is not None:
                state["last_portfolio_selector"] = portfolio_selector_record
            now = pd.Timestamp.utcnow()
            stale_cancelled: list[dict[str, Any]] = []
            if cfg.cancel_stale_open_orders and not exec_cfg.dry_run:
                assert client is not None
                stale_cancelled = await _cancel_stale_portfolio_orders(
                    client=client,
                    symbol=exec_cfg.symbol,
                    now=now,
                    max_age_sec=int(cfg.max_entry_wait_sec),
                )
                if stale_cancelled:
                    state["last_stale_order_cancels"] = stale_cancelled
                    history = list(state.get("stale_order_cancel_history", []))
                    history.extend(stale_cancelled)
                    state["stale_order_cancel_history"] = history[-200:]

            closed: list[str] = []
            for key, open_state in list(state["open_sleeves"].items()):
                if pd.Timestamp(open_state["exit_at"]) <= now:
                    if not exec_cfg.dry_run:
                        assert executor is not None
                        await _close_sleeve(executor=executor, sleeve_state=open_state, exec_cfg=exec_cfg)
                    closed.append(key)
                    state["open_sleeves"].pop(key, None)

            opened: list[str] = []
            replaced_entry_orders = 0
            for sleeve in sleeve_scores:
                name = sleeve["name"]
                signal_id = sleeve["signal_id"]
                if not sleeve["active"]:
                    continue
                if name in state["open_sleeves"]:
                    continue
                if state["processed_signals"].get(name) == signal_id:
                    continue
                margin_fraction = _margin_fraction_for_weight(
                    weight=float(sleeve["weight"]),
                    total_weight=total_weight,
                    leverage_budget=float(cfg.leverage),
                    allocation_mode=cfg.allocation_mode,
                )
                quantity = "0"
                order_info: dict[str, Any] = {}
                entry_ttl_sec = _entry_ttl_seconds(
                    sleeve,
                    interval_minutes=exec_cfg.interval_minutes,
                    timeout_fraction=cfg.entry_timeout_fraction,
                    max_entry_wait_sec=cfg.max_entry_wait_sec,
                )
                if not exec_cfg.dry_run:
                    assert client is not None and executor is not None
                    replaced = await _cancel_portfolio_orders_for_sleeve(
                        client=client,
                        symbol=exec_cfg.symbol,
                        sleeve_name=name,
                        reason="new_signal_replaces_stale_entry",
                    )
                    if replaced:
                        replaced_entry_orders += len(replaced)
                        state["last_replaced_entry_orders"] = replaced
                        history = list(state.get("replaced_entry_order_history", []))
                        history.extend(replaced)
                        state["replaced_entry_order_history"] = history[-200:]
                    order_info = await _open_sleeve(
                        client=client,
                        executor=executor,
                        exec_cfg=exec_cfg,
                        sleeve=sleeve,
                        margin_fraction=margin_fraction,
                        entry_ttl_sec=entry_ttl_sec,
                    )
                    quantity = str(order_info.get("filled_quantity") or "0")
                    if Decimal(str(quantity)) <= Decimal("0"):
                        state["processed_signals"][name] = signal_id
                        missed = list(state.get("missed_entries", []))
                        missed.append(
                            {
                                "name": name,
                                "signal_id": signal_id,
                                "reason": "post_only_entry_not_filled",
                                "entry_ttl_sec": entry_ttl_sec,
                                "order_info": order_info,
                                "recorded_at": str(pd.Timestamp.utcnow()),
                            }
                        )
                        state["missed_entries"] = missed[-200:]
                        continue
                else:
                    quantity = str((100.0 * margin_fraction * exec_cfg.leverage) / float(sleeve["current_close"]))
                    order_info = {"status": "DRY_RUN", "entry_ttl_sec": entry_ttl_sec, "filled_quantity": quantity}
                signal_ts = pd.Timestamp(sleeve["date"])
                if signal_ts.tzinfo is None:
                    signal_ts = signal_ts.tz_localize("UTC")
                exit_at = signal_ts + pd.Timedelta(minutes=exec_cfg.interval_minutes * (1 + int(sleeve["hold_bars"])))
                state["open_sleeves"][name] = {
                    "name": name,
                    "side": sleeve["side"],
                    "signal_id": signal_id,
                    "signal_date": sleeve["date"],
                    "exit_at": str(exit_at),
                    "weight": sleeve["weight"],
                    "margin_fraction": margin_fraction,
                    "allocation_mode": cfg.allocation_mode,
                    "quantity": quantity,
                    "order_info": order_info,
                }
                state["processed_signals"][name] = signal_id
                opened.append(name)

            state["updated_at"] = str(pd.Timestamp.utcnow())
            state["last_scores"] = sleeve_scores
            state["allocation_audit"] = audit
            state["last_timing"] = {
                "frame_build_sec": round(frame_build_sec, 3),
                "freshness_waited_sec": round(freshness_waited, 3),
                "latest_bar": str(enriched.iloc[-1]["date"]),
                "latest_1m_ts": str(latest_1m_ts),
                "expected_bar": str(expected_bar),
                "asof": str(asof),
                "freshness_mode": freshness_mode,
            }
            _write_json(cfg.state_file, state)
            active = [s["name"] for s in sleeve_scores if s["active"]]
            selector_status = "none"
            if portfolio_selector_record is not None:
                selector_status = f"{portfolio_selector_record.get('action')}:{portfolio_selector_record.get('context_id')}"
            _status(
                f"[portfolio-live] {pd.Timestamp.utcnow().isoformat()} active={active} opened={opened} closed={closed} "
                f"open={list(state['open_sleeves'])} stale_cancel={len(stale_cancelled)} repl={replaced_entry_orders} gross={total_weight:.3f} lev={exec_cfg.leverage} "
                f"alloc={cfg.allocation_mode} live_gross={audit['live_gross_if_all_active']:.3f} selector={selector_status} "
                f"fb={frame_build_sec:.2f}s fw={freshness_waited:.1f}s fm={freshness_mode} dry_run={exec_cfg.dry_run}"
            )
            iterations += 1
            if cfg.max_iterations is not None and iterations >= cfg.max_iterations:
                print()
                return
    finally:
        if client is not None:
            await client.aclose()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fixed-weight gross portfolio live executor")
    p.add_argument("--portfolio-config", default="configs/live/portfolio_gross6_mdd20_ratio5_return_best_candidate.json")
    p.add_argument("--execution-config", default="configs/live/rex_llm_binance_mainnet_bear_pilot_lev6.local.json")
    p.add_argument("--env", default=".env")
    p.add_argument("--state-file", default=".omx/state/portfolio_live_state.json")
    p.add_argument("--lookback-minutes", type=int, default=45_000)
    p.add_argument("--close-delay-sec", type=float, default=0.25)
    p.add_argument("--max-freshness-wait-sec", type=float, default=8.0)
    p.add_argument("--freshness-poll-sec", type=float, default=0.5, help="Deprecated fallback knob; live freshness uses Postgres NOTIFY")
    p.add_argument("--freshness-notify-channel", default="market_data_bar")
    p.add_argument("--run-immediately", action="store_true", default=False)
    p.add_argument("--live", action="store_true", default=False)
    p.add_argument("--allow-live-orders", action="store_true", default=False)
    p.add_argument("--leverage", type=int, default=6)
    p.add_argument(
        "--allocation-mode",
        choices=["research_gross", "normalize_weights"],
        default="research_gross",
        help="research_gross matches saved weights as notional exposure; normalize_weights uses 100% margin budget",
    )
    p.add_argument("--entry-timeout-fraction", type=float, default=0.25, help="Fraction of a sleeve stride cycle to wait for a post-only entry fill")
    p.add_argument("--max-entry-wait-sec", type=int, default=300, help="Hard cap for post-only entry wait/stale cancel age")
    p.add_argument(
        "--portfolio-selector-overlay",
        default="",
        help="Optional bounded portfolio LLM selector overlay; ALLOW/BLOCK_RISK only for new entries",
    )
    p.add_argument("--rex-selector-adapter-dir", default="", help="Optional bounded REX TRADE/ABSTAIN LoRA adapter directory")
    p.add_argument("--rex-selector-model-name", default="gemma4-e4b-it")
    p.add_argument("--rex-selector-score-normalization", choices=["sum", "mean", "first_token"], default="sum")
    p.add_argument("--rex-selector-fail-open", action="store_true", default=False, help="If set, adapter errors do not block an otherwise valid REX candidate")
    p.add_argument("--rex-selector-allow-cpu", action="store_true", default=False, help="Allow selector inference without CUDA; default is fail-closed when CUDA is unavailable")
    stale = p.add_mutually_exclusive_group()
    stale.add_argument("--cancel-stale-open-orders", dest="cancel_stale_open_orders", action="store_true", default=True)
    stale.add_argument("--no-cancel-stale-open-orders", dest="cancel_stale_open_orders", action="store_false")
    p.add_argument("--max-iterations", type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    asyncio.run(
        run_portfolio_loop(
            PortfolioLiveConfig(
                portfolio_config=Path(a.portfolio_config),
                execution_config=Path(a.execution_config),
                env_path=Path(a.env),
                state_file=Path(a.state_file),
                lookback_minutes=int(a.lookback_minutes),
                close_delay_sec=float(a.close_delay_sec),
                max_freshness_wait_sec=float(a.max_freshness_wait_sec),
                freshness_poll_sec=float(a.freshness_poll_sec),
                freshness_notify_channel=str(a.freshness_notify_channel),
                run_immediately=bool(a.run_immediately),
                live=bool(a.live),
                allow_live_orders=bool(a.allow_live_orders),
                leverage=int(a.leverage),
                allocation_mode=a.allocation_mode,
                max_iterations=a.max_iterations,
                entry_timeout_fraction=float(a.entry_timeout_fraction),
                max_entry_wait_sec=int(a.max_entry_wait_sec),
                cancel_stale_open_orders=bool(a.cancel_stale_open_orders),
                portfolio_selector_overlay=Path(a.portfolio_selector_overlay) if a.portfolio_selector_overlay else None,
                rex_selector_adapter_dir=Path(a.rex_selector_adapter_dir) if a.rex_selector_adapter_dir else None,
                rex_selector_model_name=str(a.rex_selector_model_name),
                rex_selector_score_normalization=str(a.rex_selector_score_normalization),
                rex_selector_fail_closed=not bool(a.rex_selector_fail_open),
                rex_selector_require_cuda=not bool(a.rex_selector_allow_cpu),
            )
        )
    )


if __name__ == "__main__":
    main()
