#!/usr/bin/env python3
"""Replay the promoted added-alpha portfolio on a completed calendar window.

The report uses the exact live signal adapters and the frozen research
execution/accounting contracts:

* completed 5-minute bars and next-bar-open entries;
* absolute timestamp stride grids;
* stale external inputs fail closed;
* 6 bp/notional/side and 0.5x sleeve leverage;
* source-specific Rank7/Fresh barriers and realized funding;
* fixed-hold REX/Markov paths matching their frozen research accounting; and
* same-BTC low/high, upper-before-lower strict MDD.

This is a retrospective replay, not pristine OOS evidence.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.portfolio_live import (
    _rank7_effective_lookback_minutes,
    _required_availability_flags,
    build_live_portfolio_frames,
)
from execution.portfolio_shadow_policies import (
    build_fresh_kimchi_feature_frame,
    build_markov_feature_frame,
    observable_markov_transition_keys,
)
from execution.rank7_runtime import (
    NO_BARRIER_BPS,
    Rank7Bundle,
    build_rank7_feature_context,
    score_rank7_row,
)
from execution.rex_llm_live import _rex_policy_features
from preprocessing.binance_aux_features import normalise_funding_history_frame
from preprocessing.live_db_features import (
    LiveDbFeatureConfig,
    sqlalchemy_engine_from_env,
)
from training.audit_rank7_fresh_kimchi_fixed_portfolio import subaccount_bar_path
from training.build_rex_event_reasoning_policy_data import (
    _rex_pullback_reclaim_arrays,
)
from training.event_candidate_pool_probe import _feature_candidates
from training.portfolio_opt_added_alpha_update import favorable_path, path_event
from training.portfolio_opt_new_alpha_pool import _event_path
from training.search_inventory_purge_reclaim_alpha import (
    Config as ResearchExecutionConfig,
)
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine

DEFAULT_PORTFOLIO = Path(
    "configs/live/portfolio_added_alpha_mainnet_live_2026-07-18.json"
)
DEFAULT_OUTPUT = Path(
    "results/portfolio_added_alpha_july_2026_performance_2026-07-27.json"
)
DEFAULT_DOCS = Path(
    "docs/portfolio-added-alpha-july-2026-performance-2026-07-27.md"
)
BASE_LEVERAGE = 0.5
COST_RATE = 0.0006
INTERVAL_MINUTES = 5
RANK7_RECENT_SOURCE_ROWS = 3_000
CURRENT_SLEEVES = (
    "fresh_kimchi_fx",
    "frozen_annual_rank7",
    "rex_taker_low_range_position",
    "cand_rex_veto_7",
    "markov_transition_long",
)


@dataclass(frozen=True)
class Config:
    portfolio_config: Path = DEFAULT_PORTFOLIO
    env_path: Path = Path("/home/pakchu/rllm/.env")
    output: Path = DEFAULT_OUTPUT
    docs_output: Path = DEFAULT_DOCS
    start: str = "2026-07-01T00:00:00Z"
    end: str = "2026-08-01T00:00:00Z"
    asof: str = "2026-07-27T15:03:00Z"
    lookback_minutes: int = 150_000
    enriched_cache: Path | None = None
    features_cache: Path | None = None
    funding_cache: Path | None = None


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _naive(value: Any) -> pd.Timestamp:
    return _utc(value).tz_localize(None)


def _interval_slots(
    dates: pd.Series | pd.Index,
    stride_bars: int,
    stride_offset_bars: int,
) -> np.ndarray:
    timestamps = pd.to_datetime(dates, utc=True).astype("int64").to_numpy()
    minutes = timestamps // 60_000_000_000
    return ((minutes // INTERVAL_MINUTES) % int(stride_bars)) == (
        int(stride_offset_bars) % int(stride_bars)
    )


def _vector_gate_pass(
    frame: pd.DataFrame,
    gates: list[dict[str, Any]],
) -> np.ndarray:
    active = np.ones(len(frame), dtype=bool)
    for gate in gates:
        feature = str(gate["feature"])
        for flag in _required_availability_flags(feature):
            values = pd.to_numeric(frame.get(flag, np.nan), errors="coerce")
            if not isinstance(values, pd.Series):
                values = pd.Series(values, index=frame.index)
            array = values.to_numpy(dtype=float)
            active &= np.isfinite(array) & (array > 0.5)
        if feature not in frame:
            active &= False
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
        threshold = float(gate["threshold"])
        op = str(gate["op"])
        if op in {">=", "ge"}:
            passed = values >= threshold
        elif op in {"<=", "le"}:
            passed = values <= threshold
        else:
            raise ValueError(f"unsupported gate op: {op}")
        active &= np.isfinite(values) & passed
    return active


def _vector_gate_clauses(
    frame: pd.DataFrame,
    clauses: list[list[dict[str, Any]]],
) -> np.ndarray:
    if not clauses:
        return np.zeros(len(frame), dtype=bool)
    return np.logical_or.reduce(
        [_vector_gate_pass(frame, clause) for clause in clauses]
    )


def _empty_arrays(length: int) -> dict[str, Any]:
    return {
        "R": np.zeros(length, dtype=np.float64),
        "L": np.zeros(length, dtype=np.float64),
        "H": np.zeros(length, dtype=np.float64),
        "trades": [],
        "skipped_overlap": 0,
        "skipped_boundary": 0,
    }


def _fixed_hold_arrays(
    market: pd.DataFrame,
    signal: np.ndarray,
    *,
    name: str,
    hold_bars: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build the exact legacy fixed-hold REX/Markov event path."""

    dates = pd.to_datetime(market["date"])
    output = _empty_arrays(len(market))
    next_allowed = 0
    for raw_position in np.flatnonzero(signal):
        position = int(raw_position)
        if not (start <= dates.iloc[position] < end):
            continue
        if position < next_allowed:
            output["skipped_overlap"] += 1
            continue
        side = "long" if int(signal[position]) > 0 else "short"
        path = _event_path(
            market,
            position,
            side=side,
            hold=int(hold_bars),
            cost_rate=COST_RATE,
            entry_delay=1,
            leverage=BASE_LEVERAGE,
        )
        if path is None:
            output["skipped_boundary"] += 1
            continue
        event_return, event_adverse, realized = path
        nonzero = np.flatnonzero(np.abs(event_return) > 1e-15)
        if not len(nonzero):
            output["skipped_boundary"] += 1
            continue
        exit_position = int(nonzero[-1])
        if not (dates.iloc[exit_position] < end):
            output["skipped_boundary"] += 1
            continue
        event_favorable = favorable_path(
            market,
            signal_position=position,
            exit_position=exit_position,
            side=side,
            leverage=BASE_LEVERAGE,
        )
        output["R"] += event_return
        if side == "long":
            output["L"] += event_adverse
            output["H"] += event_favorable
        else:
            output["L"] += event_favorable
            output["H"] += event_adverse
        output["trades"].append(
            {
                "sleeve": name,
                "signal_date": str(dates.iloc[position]),
                "entry_date": str(dates.iloc[position + 1]),
                "exit_date": str(dates.iloc[exit_position]),
                "side": side.upper(),
                "net_return": float(realized),
                "source": None,
            }
        )
        next_allowed = exit_position + 1
    return output


def _research_execution_config(end: pd.Timestamp) -> ResearchExecutionConfig:
    return ResearchExecutionConfig(
        input_csv="",
        metrics_csv="",
        funding_csv="",
        output="",
        manifest_output="",
        exclude_from=str(end),
        leverage=BASE_LEVERAGE,
        fee_rate=0.0005,
        slippage_rate=0.0001,
    )


def _barrier_arrays(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    signal: np.ndarray,
    *,
    name: str,
    lifecycle: Callable[[int], dict[str, Any]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    """Build exact funded barrier paths for Fresh Kimchi and Rank7."""

    dates = pd.to_datetime(market["date"])
    output = _empty_arrays(len(market))
    execution_cfg = _research_execution_config(end)
    engine = ExecutionEngine(market, funding, execution_cfg)
    trades = []
    hold_by_signal: dict[int, int] = {}
    metadata_by_signal: dict[int, dict[str, Any]] = {}
    next_allowed = 0
    for raw_position in np.flatnonzero(signal):
        position = int(raw_position)
        if not (start <= dates.iloc[position] < end):
            continue
        if position < next_allowed:
            output["skipped_overlap"] += 1
            continue
        spec = lifecycle(position)
        hold = int(spec["hold_bars"])
        take_bps = (
            NO_BARRIER_BPS if spec.get("take_bps") is None else float(spec["take_bps"])
        )
        stop_bps = (
            NO_BARRIER_BPS if spec.get("stop_bps") is None else float(spec["stop_bps"])
        )
        trade = engine.trade_at(
            position,
            int(signal[position]),
            hold,
            round(take_bps),
            round(stop_bps),
        )
        if trade is None or not (dates.iloc[trade.exit_position] < end):
            output["skipped_boundary"] += 1
            continue
        trades.append(trade)
        hold_by_signal[position] = hold
        metadata_by_signal[position] = spec
        next_allowed = int(trade.exit_position) + 1

    if not trades:
        return output
    path = subaccount_bar_path(
        market,
        funding,
        trades,
        execution_cfg,
        start=str(start),
        end=str(end),
        hold_bars=lambda trade: hold_by_signal[int(trade.signal_position)],
    )
    event = path_event(
        market,
        path,
        split="window",
        sleeve=name,
        trades=trades,
    )
    output["R"] = event["ret"]
    output["L"] = event["low"]
    output["H"] = event["high"]
    cost_factor = 1.0 - BASE_LEVERAGE * COST_RATE
    for trade in trades:
        spec = metadata_by_signal[int(trade.signal_position)]
        output["trades"].append(
            {
                "sleeve": name,
                "signal_date": str(dates.iloc[trade.signal_position]),
                "entry_date": str(dates.iloc[trade.entry_position]),
                "exit_date": str(dates.iloc[trade.exit_position]),
                "side": "LONG" if int(trade.side) > 0 else "SHORT",
                "net_return": float(
                    cost_factor
                    * float(trade.price_factor)
                    * float(trade.funding_factor)
                    * cost_factor
                    - 1.0
                ),
                "source": spec.get("source"),
            }
        )
    return output


def _strict_metric(
    arrays: dict[str, dict[str, Any]],
    weights: dict[str, float],
    *,
    dates: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    mask = ((dates >= start) & (dates < end)).to_numpy(dtype=bool)
    positions = np.flatnonzero(mask)
    if not len(positions):
        raise ValueError("empty metric window")
    returns = np.zeros(len(positions), dtype=float)
    market_low = np.zeros(len(positions), dtype=float)
    market_high = np.zeros(len(positions), dtype=float)
    for name, weight in weights.items():
        returns += float(weight) * arrays[name]["R"][positions]
        market_low += float(weight) * arrays[name]["L"][positions]
        market_high += float(weight) * arrays[name]["H"][positions]
    equity_after = np.cumprod(np.maximum(0.0, 1.0 + returns))
    equity_before = np.r_[1.0, equity_after[:-1]]
    low_value = equity_before * np.maximum(0.0, 1.0 + market_low)
    high_value = equity_before * np.maximum(0.0, 1.0 + market_high)
    upper = np.maximum.reduce(
        [equity_before, equity_after, low_value, high_value]
    )
    lower = np.minimum.reduce(
        [equity_before, equity_after, low_value, high_value]
    )
    peak = np.maximum.accumulate(upper)
    mdd = float(np.max(1.0 - lower / np.maximum(peak, 1e-12))) * 100.0
    final = float(equity_after[-1])
    absolute_return = (final - 1.0) * 100.0
    years = (end - start).total_seconds() / (365.25 * 86_400.0)
    cagr = (final ** (1.0 / years) - 1.0) * 100.0 if final > 0.0 else -100.0
    trades = [
        trade
        for name in weights
        for trade in arrays[name]["trades"]
        if start <= pd.Timestamp(trade["entry_date"]) < end
    ]
    wins = sum(float(trade["net_return"]) > 0.0 for trade in trades)
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(mdd),
        "cagr_to_strict_mdd": float(cagr / mdd) if mdd > 1e-12 else 0.0,
        "return_to_strict_mdd": (
            float(absolute_return / mdd) if mdd > 1e-12 else 0.0
        ),
        "trades": len(trades),
        "longs": sum(trade["side"] == "LONG" for trade in trades),
        "shorts": sum(trade["side"] == "SHORT" for trade in trades),
        "win_rate": float(wins / len(trades)) if trades else 0.0,
        "active_bars": int(
            np.count_nonzero(
                (np.abs(returns) > 1e-15)
                | (np.abs(market_low) > 1e-15)
                | (np.abs(market_high) > 1e-15)
            )
        ),
        "final_equity": final,
        "calendar_days": (end - start).total_seconds() / 86_400.0,
        "trades_by_sleeve": {
            name: sum(
                start <= pd.Timestamp(trade["entry_date"]) < end
                for trade in arrays[name]["trades"]
            )
            for name in weights
        },
    }


def _fresh_signal(
    market: pd.DataFrame,
    features: pd.DataFrame,
    cfg: dict[str, Any],
) -> np.ndarray:
    policy_features = build_fresh_kimchi_feature_frame(market, features)
    long_active = _vector_gate_pass(policy_features, cfg["long_gates"])
    short_active = _vector_gate_pass(policy_features, cfg["short_gates"])
    slots = _interval_slots(
        market["date"],
        int(cfg["stride_bars"]),
        int(cfg.get("stride_offset_bars", 0)),
    )
    return np.where(
        slots & np.logical_xor(long_active, short_active),
        np.where(long_active, 1, -1),
        0,
    ).astype(np.int8)


def _markov_signal(
    market: pd.DataFrame,
    features: pd.DataFrame,
    cfg: dict[str, Any],
) -> np.ndarray:
    contract = cfg["feature_contract"]
    policy_features = build_markov_feature_frame(
        market,
        features,
        window_size=int(contract["window_size"]),
        zscore_window=int(contract["zscore_window"]),
        volume_window=int(contract["volume_window"]),
    )
    base = _vector_gate_clauses(policy_features, cfg["gate_clauses"])
    transitions = observable_markov_transition_keys(market, cfg["state_model"])
    slots = _interval_slots(
        market["date"],
        int(cfg["stride_bars"]),
        int(cfg.get("stride_offset_bars", 0)),
    )
    return (
        slots
        & base
        & np.isin(
            transitions,
            np.asarray(cfg["state_model"]["allowed_transition_keys"], dtype=int),
        )
    ).astype(np.int8)


def _rex_signal(
    market: pd.DataFrame,
    features: pd.DataFrame,
    cfg: dict[str, Any],
) -> np.ndarray:
    policy = cfg["rex_policy"]
    if any(
        (
            policy.get("rule_gates"),
            policy.get("alternate_rule_gate_sets"),
            policy.get("require_core_external"),
            policy.get("require_binance_aux"),
        )
    ):
        raise ValueError("monthly vector replay supports only the frozen current REX contract")
    policy_features = _rex_policy_features(
        market,
        features,
        str(policy["feature_contract"]),
    )
    if str(policy["feature_contract"]) == "rex_event_reasoning_20260712":
        strength, direction = _rex_pullback_reclaim_arrays(policy_features)
    else:
        strength, direction = _feature_candidates(policy_features)[
            str(policy["family"])
        ]
    active = (
        (strength > float(policy["strength_threshold"]))
        & (direction != 0)
        & _vector_gate_pass(policy_features, cfg["gates"])
        & _interval_slots(
            market["date"],
            int(cfg["stride_bars"]),
            int(cfg.get("stride_offset_bars", 0)),
        )
    )
    return np.where(active, np.sign(direction), 0).astype(np.int8)


def _rank7_signal(
    market: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[int, dict[str, Any]], dict[str, Any]]:
    bundle = Rank7Bundle.load(str(cfg["bundle_path"]))
    if str(cfg.get("bundle_manifest_hash")) != str(
        bundle.manifest["bundle_manifest_hash"]
    ):
        raise RuntimeError("Rank7 config/bundle manifest mismatch")
    context = build_rank7_feature_context(market, bundle)
    dates = pd.to_datetime(context["dates"])
    source_ready = (
        pd.to_numeric(market["spot_rows"], errors="coerce")
        .eq(5)
        .rolling(RANK7_RECENT_SOURCE_ROWS, min_periods=RANK7_RECENT_SOURCE_ROWS)
        .sum()
        .eq(RANK7_RECENT_SOURCE_ROWS)
        & pd.to_numeric(market["premium_rows"], errors="coerce")
        .eq(5)
        .rolling(RANK7_RECENT_SOURCE_ROWS, min_periods=RANK7_RECENT_SOURCE_ROWS)
        .sum()
        .eq(RANK7_RECENT_SOURCE_ROWS)
        & (pd.to_numeric(market["open_interest_available"], errors="coerce") > 0.5)
        & (pd.to_numeric(market["funding_available"], errors="coerce") > 0.5)
        & (pd.to_numeric(market["premium_available"], errors="coerce") > 0.5)
        & (pd.to_numeric(market["open_interest"], errors="coerce") > 0.0)
    ).to_numpy(dtype=bool)
    candidates = np.flatnonzero(
        np.asarray(context["anchors"], dtype=bool)
        & (dates.dt.minute.to_numpy() == 0)
        & (dates.dt.second.to_numpy() == 0)
    )
    signal = np.zeros(len(market), dtype=np.int8)
    lifecycles: dict[int, dict[str, Any]] = {}
    raw_active = blocked_source = 0
    for position in candidates:
        decision = score_rank7_row(
            bundle,
            context["matrix"][position],
            decision_ts=dates.iloc[position],
            is_anchor=True,
        )
        if not decision.active:
            continue
        raw_active += 1
        if not source_ready[position]:
            blocked_source += 1
            continue
        signal[position] = 1
        barrier = decision.barrier_exit or {}
        lifecycles[int(position)] = {
            "hold_bars": int(decision.hold_bars),
            "take_bps": barrier.get("take_bps"),
            "stop_bps": barrier.get("stop_bps"),
            "source": decision.source,
        }
    return signal, lifecycles, {
        "candidate_anchors": len(candidates),
        "raw_active": int(raw_active),
        "blocked_runtime_source": int(blocked_source),
        "bundle_manifest_hash": str(bundle.manifest["bundle_manifest_hash"]),
        "model_version": bundle.model_version,
    }


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path, compression="infer")


async def _query_frames(
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    engine = sqlalchemy_engine_from_env(cfg.env_path)
    asof = _utc(cfg.asof)
    effective_lookback_minutes = _rank7_effective_lookback_minutes(
        _load_json(cfg.portfolio_config),
        asof=asof,
        configured_minutes=int(cfg.lookback_minutes),
        interval_minutes=INTERVAL_MINUTES,
    )
    live_cfg = LiveDbFeatureConfig(
        lookback_minutes=effective_lookback_minutes,
        include_spot_source=True,
    )
    enriched, features = await build_live_portfolio_frames(
        engine=engine,
        asof=asof,
        cfg=live_cfg,
        live_oi_snapshot_cutoff=asof + pd.Timedelta(minutes=2),
        include_activity_flow=False,
        include_alt_pool=False,
    )
    from sqlalchemy import text

    with engine.connect() as conn:
        funding = pd.read_sql_query(
            text(
                """
                SELECT funding_time AS date, funding_rate
                FROM funding_rates_binance
                WHERE symbol = 'BTCUSDT'
                  AND funding_time <= :asof
                ORDER BY funding_time
                """
            ),
            conn,
            params={"asof": asof.to_pydatetime()},
        )
    return enriched, features, funding, engine


def _load_frames(
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any | None]:
    cache_paths = (cfg.enriched_cache, cfg.features_cache, cfg.funding_cache)
    if any(cache_paths):
        if not all(cache_paths):
            raise ValueError("all three frame caches must be supplied together")
        assert cfg.enriched_cache is not None
        assert cfg.features_cache is not None
        assert cfg.funding_cache is not None
        return (
            _read_frame(cfg.enriched_cache),
            _read_frame(cfg.features_cache),
            _read_frame(cfg.funding_cache),
            None,
        )
    return asyncio.run(_query_frames(cfg))


def _frame_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column in selected:
        if "date" in column:
            selected[column] = pd.to_datetime(selected[column], utc=True).astype("int64")
    payload = pd.util.hash_pandas_object(selected, index=False).to_numpy(
        dtype="<u8", copy=False
    )
    digest = hashlib.sha256()
    digest.update("\x1f".join(columns).encode())
    digest.update(payload.tobytes())
    return digest.hexdigest()


def _ledger_audit(
    engine: Any | None,
    *,
    promotion: pd.Timestamp,
) -> dict[str, Any]:
    if engine is None:
        return {"queried": False, "reason": "cache_mode"}
    from sqlalchemy import bindparam, text

    names = tuple(CURRENT_SLEEVES)
    with engine.connect() as conn:
        current = conn.execute(
            text(
                """
                SELECT sub_strategy_name, action, status, COUNT(*) AS rows,
                       COALESCE(SUM(net_realized_pnl), 0) AS net_realized_pnl
                FROM trade_executions
                WHERE created_at >= :promotion
                  AND sub_strategy_name IN :names
                GROUP BY sub_strategy_name, action, status
                ORDER BY sub_strategy_name, action, status
                """
            ).bindparams(bindparam("names", expanding=True)),
            {"promotion": promotion.to_pydatetime(), "names": names},
        ).mappings().all()
        unsupported = conn.execute(
            text(
                """
                SELECT action, status, COUNT(*) AS rows,
                       COALESCE(SUM(net_realized_pnl), 0) AS net_realized_pnl
                FROM trade_executions
                WHERE sub_strategy_name = 'cand_rex_veto_7'
                  AND created_at >= TIMESTAMPTZ '2026-07-14 00:00:00+00'
                  AND created_at < TIMESTAMPTZ '2026-07-15 01:00:00+00'
                GROUP BY action, status
                ORDER BY action, status
                """
            )
        ).mappings().all()
    return {
        "queried": True,
        "post_promotion_current_sleeves": [
            {key: str(value) if key == "net_realized_pnl" else value for key, value in row.items()}
            for row in current
        ],
        "post_promotion_execution_rows": int(sum(int(row["rows"]) for row in current)),
        "july14_cand_rex_ledger_rows": [
            {key: str(value) if key == "net_realized_pnl" else value for key, value in row.items()}
            for row in unsupported
        ],
        "july14_classification": (
            "unsupported_live_entries_excluded_from_valid_alpha_stats; "
            "commit 55f1ad452 froze the research feature/threshold/stride contract"
        ),
    }


def _availability_summary(
    market: pd.DataFrame,
    mask: np.ndarray,
) -> dict[str, Any]:
    columns = (
        "open_interest_available",
        "funding_available",
        "premium_available",
        "usdkrw_available",
        "dxy_available",
        "kimchi_available",
    )
    return {
        column: {
            "available_bars": int(
                np.count_nonzero(
                    pd.to_numeric(market[column], errors="coerce").to_numpy(float)[
                        mask
                    ]
                    > 0.5
                )
            ),
            "coverage_pct": float(
                np.mean(
                    pd.to_numeric(market[column], errors="coerce").to_numpy(float)[
                        mask
                    ]
                    > 0.5
                )
                * 100.0
            ),
        }
        for column in columns
    }


def _render_docs(report: dict[str, Any]) -> str:
    def metric(row: dict[str, Any]) -> str:
        return (
            f"{row['absolute_return_pct']:.4f}% / {row['cagr_pct']:.2f}% / "
            f"{row['strict_mdd_pct']:.4f}% / {row['cagr_to_strict_mdd']:.2f} / "
            f"{row['trades']}"
        )

    lines = [
        "# Added-alpha portfolio — 2026년 7월 성과 감사",
        "",
        f"- 데이터: `{report['window']['start']}` ~ `{report['window']['end_exclusive']}`",
        f"- 마지막 완결 5분봉: `{report['window']['last_completed_bar']}`",
        f"- 포트폴리오: `{report['portfolio']['name']}`; gross `{report['portfolio']['gross_weight']:.2f}`",
        "- 표 셀: `절대수익 / 관측기간 연율화 CAGR / strict MDD / CAGR-MDD / 거래수`.",
        "- Standalone은 슬리브 weight 1.0, weighted contribution은 현재 포트폴리오 비중을 적용한다.",
        "- 한 달 미만 CAGR와 CAGR/MDD는 매우 불안정하므로 절대수익과 strict MDD를 우선 해석한다.",
        "",
        "## 7월 전체 retrospective replay",
        "",
        "| 알파 | 비중 | Standalone | Weighted contribution | Long/Short | 승률 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in CURRENT_SLEEVES:
        row = report["full_window"]["sleeves"][name]
        weighted = report["full_window"]["weighted_sleeves"][name]
        lines.append(
            f"| `{name}` | {report['portfolio']['weights'][name]:.2f} | "
            f"{metric(row)} | {metric(weighted)} | {row['longs']}/{row['shorts']} | "
            f"{row['win_rate'] * 100:.1f}% |"
        )
    portfolio = report["full_window"]["portfolio"]
    lines.extend(
        [
            (
                f"| **gross 8 portfolio** | **{report['portfolio']['gross_weight']:.2f}** | "
                f"— | **{metric(portfolio)}** | **{portfolio['longs']}/{portfolio['shorts']}** | "
                f"**{portfolio['win_rate'] * 100:.1f}%** |"
            ),
            "",
            "## 실제 승격 이후 forward 구간",
            "",
            f"- 승격 시각: `{report['portfolio']['promotion_start']}`",
            f"- 성과: **{metric(report['post_promotion']['portfolio'])}**",
            f"- DB 주문/체결 장부 행: **{report['live_ledger'].get('post_promotion_execution_rows', '미조회')}**",
            "",
            "## 체결된 retrospective 거래",
            "",
            "| 알파 | 신호 | 진입 | 청산 | 방향 | 순수익률 |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for trade in report["trades"]:
        lines.append(
            f"| `{trade['sleeve']}` | {trade['signal_date']} | {trade['entry_date']} | "
            f"{trade['exit_date']} | {trade['side']} | {trade['net_return'] * 100:.4f}% |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 현재 동결 계약으로는 7월에 Fresh Kimchi, Rank7, Markov 신호가 없었다.",
            "- 유효 신호는 REX 두 계열의 숏뿐이었고, 포트폴리오 전체는 손실이었다.",
            (
                "- 현재 gross 8 포트폴리오는 7월 18일 승격되었으므로 7월 1일부터의 수치는 "
                "현재 계약을 과거에 대입한 retrospective 진단이지 forward OOS가 아니다."
            ),
            (
                "- 7월 14일 `cand_rex_veto_7` 실거래 2건은 이후 커밋 `55f1ad452`에서 "
                "재계산 threshold/feature/stride 오류로 판정된 unsupported entry다. 해당 실현손익은 "
                "유효 알파 통계에서 제외했다."
            ),
            "- 모든 외부 데이터는 backward-as-of로만 결합하며, FX 주말·stale 구간은 fail-closed다.",
            (
                "- 비용은 6 bp/notional/side, 기본 슬리브 레버리지는 0.5x다. strict MDD는 "
                "동일 BTC 저가/고가를 공유하고 bar 내 상단을 먼저 본 뒤 하단을 보는 보수적 경로다."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run(cfg: Config) -> dict[str, Any]:
    portfolio = _load_json(cfg.portfolio_config)
    weights = {str(key): float(value) for key, value in portfolio["weights"].items()}
    if tuple(weights) != CURRENT_SLEEVES:
        raise RuntimeError(
            f"unexpected promoted sleeve order: {tuple(weights)} != {CURRENT_SLEEVES}"
        )
    cache_paths = (cfg.enriched_cache, cfg.features_cache, cfg.funding_cache)
    effective_lookback_minutes = (
        None
        if any(cache_paths)
        else _rank7_effective_lookback_minutes(
            portfolio,
            asof=_utc(cfg.asof),
            configured_minutes=int(cfg.lookback_minutes),
            interval_minutes=INTERVAL_MINUTES,
        )
    )
    enriched, features, raw_funding, engine = _load_frames(cfg)
    enriched = enriched.copy()
    enriched["date"] = pd.to_datetime(enriched["date"], utc=True).dt.tz_convert(None)
    features = features.reset_index(drop=True)
    if len(enriched) != len(features):
        raise RuntimeError("enriched/features length mismatch")
    dates = pd.to_datetime(enriched["date"])
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta(minutes=INTERVAL_MINUTES)).all():
        raise RuntimeError("market frame is not a complete 5-minute grid")
    funding = normalise_funding_history_frame(raw_funding)
    requested_start = _naive(cfg.start)
    requested_end = _naive(cfg.end)
    data_end = pd.Timestamp(dates.iloc[-1]) + pd.Timedelta(minutes=INTERVAL_MINUTES)
    end = min(requested_end, data_end)
    start = requested_start
    if start >= end:
        raise RuntimeError("requested report window has no completed data")
    window_mask = ((dates >= start) & (dates < end)).to_numpy(dtype=bool)
    expected_bars = int((end - start) / pd.Timedelta(minutes=INTERVAL_MINUTES))
    if int(window_mask.sum()) != expected_bars:
        raise RuntimeError(
            f"window completeness mismatch: {int(window_mask.sum())} != {expected_bars}"
        )

    source_cfg = {
        row["name"]: _load_json(row["source"])
        for row in portfolio["base_sleeves"]
    }
    fresh_signal = _fresh_signal(
        enriched, features, source_cfg["fresh_kimchi_fx"]
    )
    rank7_signal, rank7_lifecycles, rank7_diagnostics = _rank7_signal(
        enriched, source_cfg["frozen_annual_rank7"]
    )
    rex_taker_signal = _rex_signal(
        enriched, features, source_cfg["rex_taker_low_range_position"]
    )
    rex_veto_signal = _rex_signal(
        enriched, features, source_cfg["cand_rex_veto_7"]
    )
    markov_signal = _markov_signal(
        enriched, features, source_cfg["markov_transition_long"]
    )

    fresh_cfg = source_cfg["fresh_kimchi_fx"]
    arrays = {
        "fresh_kimchi_fx": _barrier_arrays(
            enriched,
            funding,
            fresh_signal,
            name="fresh_kimchi_fx",
            lifecycle=lambda _position: {
                "hold_bars": int(fresh_cfg["hold_bars"]),
                "take_bps": float(fresh_cfg["take_bps"]),
                "stop_bps": float(fresh_cfg["stop_bps"]),
                "source": None,
            },
            start=start,
            end=end,
        ),
        "frozen_annual_rank7": _barrier_arrays(
            enriched,
            funding,
            rank7_signal,
            name="frozen_annual_rank7",
            lifecycle=lambda position: rank7_lifecycles[int(position)],
            start=start,
            end=end,
        ),
        "rex_taker_low_range_position": _fixed_hold_arrays(
            enriched,
            rex_taker_signal,
            name="rex_taker_low_range_position",
            hold_bars=int(
                source_cfg["rex_taker_low_range_position"]["hold_bars"]
            ),
            start=start,
            end=end,
        ),
        "cand_rex_veto_7": _fixed_hold_arrays(
            enriched,
            rex_veto_signal,
            name="cand_rex_veto_7",
            hold_bars=int(source_cfg["cand_rex_veto_7"]["hold_bars"]),
            start=start,
            end=end,
        ),
        "markov_transition_long": _fixed_hold_arrays(
            enriched,
            markov_signal,
            name="markov_transition_long",
            hold_bars=int(source_cfg["markov_transition_long"]["hold_bars"]),
            start=start,
            end=end,
        ),
    }
    signals = {
        "fresh_kimchi_fx": fresh_signal,
        "frozen_annual_rank7": rank7_signal,
        "rex_taker_low_range_position": rex_taker_signal,
        "cand_rex_veto_7": rex_veto_signal,
        "markov_transition_long": markov_signal,
    }
    sleeve_metrics = {
        name: _strict_metric(
            arrays,
            {name: 1.0},
            dates=dates,
            start=start,
            end=end,
        )
        for name in CURRENT_SLEEVES
    }
    weighted_sleeve_metrics = {
        name: _strict_metric(
            arrays,
            {name: weights[name]},
            dates=dates,
            start=start,
            end=end,
        )
        for name in CURRENT_SLEEVES
    }
    portfolio_metric = _strict_metric(
        arrays,
        weights,
        dates=dates,
        start=start,
        end=end,
    )
    promotion = _naive(portfolio["as_of"])
    post_promotion = {
        "start": str(promotion),
        "end_exclusive": str(end),
        "portfolio": _strict_metric(
            arrays,
            weights,
            dates=dates,
            start=promotion,
            end=end,
        ),
        "sleeves": {
            name: _strict_metric(
                arrays,
                {name: 1.0},
                dates=dates,
                start=promotion,
                end=end,
            )
            for name in CURRENT_SLEEVES
        },
    }
    trades = sorted(
        [
            trade
            for name in CURRENT_SLEEVES
            for trade in arrays[name]["trades"]
        ],
        key=lambda trade: trade["entry_date"],
    )
    source_files = {
        str(cfg.portfolio_config): _sha256(cfg.portfolio_config),
        **{
            str(row["source"]): _sha256(row["source"])
            for row in portfolio["base_sleeves"]
        },
        str(
            Path(source_cfg["frozen_annual_rank7"]["bundle_path"]) / "manifest.json"
        ): _sha256(
            Path(source_cfg["frozen_annual_rank7"]["bundle_path"]) / "manifest.json"
        ),
    }
    report = {
        "schema_version": 1,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "mode": "promoted_added_alpha_completed_bar_monthly_replay",
        "accounting_version": "same_btc_low_high_v1",
        "retrospective_not_pristine_oos": True,
        "config": {
            **asdict(cfg),
            "portfolio_config": str(cfg.portfolio_config),
            "env_path": "<redacted>",
            "output": str(cfg.output),
            "docs_output": str(cfg.docs_output),
            "enriched_cache": (
                None if cfg.enriched_cache is None else str(cfg.enriched_cache)
            ),
            "features_cache": (
                None if cfg.features_cache is None else str(cfg.features_cache)
            ),
            "funding_cache": (
                None if cfg.funding_cache is None else str(cfg.funding_cache)
            ),
            "effective_lookback_minutes": effective_lookback_minutes,
        },
        "window": {
            "requested_start": str(requested_start),
            "requested_end_exclusive": str(requested_end),
            "start": str(start),
            "end_exclusive": str(end),
            "last_completed_bar": str(dates.iloc[-1]),
            "bars": int(window_mask.sum()),
            "calendar_days": (end - start).total_seconds() / 86_400.0,
        },
        "portfolio": {
            "name": portfolio["name"],
            "status": portfolio["status"],
            "weights": weights,
            "gross_weight": float(sum(weights.values())),
            "promotion_start": str(promotion),
        },
        "execution_contract": {
            "bar_interval_minutes": INTERVAL_MINUTES,
            "entry": "next_completed_5m_bar_open",
            "cost_rate_each_side": COST_RATE,
            "base_sleeve_leverage": BASE_LEVERAGE,
            "fixed_hold_funding": "excluded to preserve frozen REX/Markov research accounting",
            "rank7_fresh_funding": "realized funding included",
            "strict_mdd": (
                "same BTC low/high across sleeves; upper envelope before lower "
                "envelope on each bar"
            ),
            "split_boundary": "flat start; only split-contained exits admitted",
        },
        "data_quality": {
            "market_rows_with_warmup": len(enriched),
            "market_start": str(dates.iloc[0]),
            "market_end": str(dates.iloc[-1]),
            "spot_incomplete_5m_bars_full_warmup": int(
                np.count_nonzero(
                    pd.to_numeric(enriched["spot_rows"], errors="coerce").to_numpy()
                    != 5
                )
            ),
            "premium_incomplete_5m_bars_full_warmup": int(
                np.count_nonzero(
                    pd.to_numeric(
                        enriched["premium_rows"], errors="coerce"
                    ).to_numpy()
                    != 5
                )
            ),
            "availability": _availability_summary(enriched, window_mask),
            "window_market_hash": _frame_hash(
                enriched.loc[window_mask].reset_index(drop=True),
                [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "open_interest",
                    "funding_rate",
                    "premium_index",
                    "usdkrw",
                    "kimchi_premium",
                ],
            ),
        },
        "signal_diagnostics": {
            name: {
                "scheduled_raw": int(
                    np.count_nonzero(signal[window_mask])
                ),
                "raw_longs": int(np.count_nonzero(signal[window_mask] > 0)),
                "raw_shorts": int(np.count_nonzero(signal[window_mask] < 0)),
                "accepted_trades": len(arrays[name]["trades"]),
                "skipped_overlap": int(arrays[name]["skipped_overlap"]),
                "skipped_boundary": int(arrays[name]["skipped_boundary"]),
            }
            for name, signal in signals.items()
        },
        "rank7_diagnostics": rank7_diagnostics,
        "full_window": {
            "portfolio": portfolio_metric,
            "sleeves": sleeve_metrics,
            "weighted_sleeves": weighted_sleeve_metrics,
        },
        "post_promotion": post_promotion,
        "trades": trades,
        "live_ledger": _ledger_audit(
            engine,
            promotion=_utc(portfolio["as_of"]),
        ),
        "source_sha256": source_files,
        "interpretation": {
            "primary_metric": "absolute_return_pct and strict_mdd_pct",
            "partial_period_cagr_warning": (
                "CAGR and CAGR/MDD annualize only the observed partial month and "
                "are not statistically stable"
            ),
            "july14_live_exception": (
                "Two cand_rex_veto_7 entries preceded the frozen-contract fix "
                "55f1ad452 and are excluded from valid replay statistics"
            ),
            "post_promotion_forward_evidence": (
                "Only the interval on/after 2026-07-18 is post-promotion; the full "
                "July replay is retrospective"
            ),
        },
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    cfg.docs_output.parent.mkdir(parents=True, exist_ok=True)
    cfg.docs_output.write_text(_render_docs(report) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-config", default=str(DEFAULT_PORTFOLIO))
    parser.add_argument("--env", default="/home/pakchu/rllm/.env")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--docs-output", default=str(DEFAULT_DOCS))
    parser.add_argument("--start", default=Config.start)
    parser.add_argument("--end", default=Config.end)
    parser.add_argument("--asof", default=Config.asof)
    parser.add_argument("--lookback-minutes", type=int, default=Config.lookback_minutes)
    parser.add_argument("--enriched-cache", default="")
    parser.add_argument("--features-cache", default="")
    parser.add_argument("--funding-cache", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(
        Config(
            portfolio_config=Path(args.portfolio_config),
            env_path=Path(args.env),
            output=Path(args.output),
            docs_output=Path(args.docs_output),
            start=str(args.start),
            end=str(args.end),
            asof=str(args.asof),
            lookback_minutes=int(args.lookback_minutes),
            enriched_cache=(
                Path(args.enriched_cache) if str(args.enriched_cache) else None
            ),
            features_cache=(
                Path(args.features_cache) if str(args.features_cache) else None
            ),
            funding_cache=(
                Path(args.funding_cache) if str(args.funding_cache) else None
            ),
        )
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "docs": str(args.docs_output),
                "window": report["window"],
                "portfolio": report["full_window"]["portfolio"],
                "post_promotion": report["post_promotion"]["portfolio"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
