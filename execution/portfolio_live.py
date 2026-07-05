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
import json
from dataclasses import dataclass, asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from execution.rex_llm_live import RexLivePolicyConfig, build_rex_live_policy_record
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
    close_delay_sec: float = 15.0
    run_immediately: bool = False
    live: bool = False
    allow_live_orders: bool = False
    leverage: int = 6
    allocation_mode: Literal["research_gross", "normalize_weights"] = "research_gross"
    max_iterations: int | None = None


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


def _score_sleeves(
    *,
    portfolio: dict[str, Any],
    enriched: pd.DataFrame,
    features: pd.DataFrame,
    exec_cfg: WaveExecutionConfig,
    asof: pd.Timestamp,
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
) -> dict[str, Any]:
    balance = await client.get_usdt_balance()
    total_equity = float(balance["total"])
    price = await client.get_ticker_price(exec_cfg.symbol)
    notional = total_equity * float(margin_fraction) * float(exec_cfg.leverage)
    quantity = Decimal(str(notional / float(price)))
    side: Side = sleeve["side"]
    order_side = "BUY" if side == "LONG" else "SELL"
    order = await executor.place_maker_order_with_retry(
        order_side,
        quantity,
        reduce_only=False,
        max_retries=exec_cfg.max_retries,
        signal_id=sleeve["signal_id"],
        position_side=side,
        order_type="ENTRY",
    )
    return {
        "order": order,
        "requested_quantity": str(quantity),
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


async def run_portfolio_loop(cfg: PortfolioLiveConfig) -> None:
    portfolio = _load_json(cfg.portfolio_config)
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
            enriched, features = await build_live_portfolio_frames(engine=engine, asof=asof, cfg=live_cfg)
            sleeve_scores = _score_sleeves(portfolio=portfolio, enriched=enriched, features=features, exec_cfg=exec_cfg, asof=asof)
            state = _load_state(cfg.state_file)
            now = pd.Timestamp.utcnow()

            closed: list[str] = []
            for key, open_state in list(state["open_sleeves"].items()):
                if pd.Timestamp(open_state["exit_at"]) <= now:
                    if not exec_cfg.dry_run:
                        assert executor is not None
                        await _close_sleeve(executor=executor, sleeve_state=open_state, exec_cfg=exec_cfg)
                    closed.append(key)
                    state["open_sleeves"].pop(key, None)

            opened: list[str] = []
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
                if not exec_cfg.dry_run:
                    assert client is not None and executor is not None
                    order_info = await _open_sleeve(client=client, executor=executor, exec_cfg=exec_cfg, sleeve=sleeve, margin_fraction=margin_fraction)
                    quantity = order_info["requested_quantity"]
                else:
                    quantity = str((100.0 * margin_fraction * exec_cfg.leverage) / float(sleeve["current_close"]))
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
            _write_json(cfg.state_file, state)
            active = [s["name"] for s in sleeve_scores if s["active"]]
            _status(
                f"[portfolio-live] {pd.Timestamp.utcnow().isoformat()} active={active} opened={opened} closed={closed} "
                f"open={list(state['open_sleeves'])} gross={total_weight:.3f} lev={exec_cfg.leverage} "
                f"alloc={cfg.allocation_mode} live_gross={audit['live_gross_if_all_active']:.3f} dry_run={exec_cfg.dry_run}"
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
    p.add_argument("--close-delay-sec", type=float, default=15.0)
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
                run_immediately=bool(a.run_immediately),
                live=bool(a.live),
                allow_live_orders=bool(a.allow_live_orders),
                leverage=int(a.leverage),
                allocation_mode=a.allocation_mode,
                max_iterations=a.max_iterations,
            )
        )
    )


if __name__ == "__main__":
    main()
