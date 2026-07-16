"""One-shot, no-order scorer for forward-shadow portfolio candidates."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from execution.portfolio_live import (
    _completed_decision_data_asof,
    _expected_decision_bar,
    _load_json,
    _score_sleeves,
    _validate_portfolio_mode,
    build_live_portfolio_frames,
)
from execution.wave_execution import WaveExecutionConfig
from preprocessing.live_db_features import LiveDbFeatureConfig, sqlalchemy_engine_from_env


@dataclass(frozen=True)
class PortfolioShadowConfig:
    portfolio_config: Path = Path(
        "configs/live/portfolio_added_alpha_shadow_candidate_2026-07-16.json"
    )
    execution_config: Path = Path("configs/live/rex_llm_binance_testnet_bear_pilot.json")
    env_path: Path = Path(".env")
    output: Path | None = Path(".omx/state/portfolio_added_alpha_shadow_score.json")
    lookback_minutes: int = 90_000
    asof: str | None = None


def build_shadow_report(
    *,
    portfolio: dict[str, Any],
    enriched: pd.DataFrame,
    features: pd.DataFrame,
    execution_cfg: WaveExecutionConfig,
    decision_asof: pd.Timestamp,
) -> dict[str, Any]:
    """Score configured sleeves without importing or invoking any order path."""

    _validate_portfolio_mode(portfolio, live=False)
    if not bool(portfolio.get("shadow_only")):
        raise RuntimeError("portfolio_shadow requires shadow_only=true")
    scores = _score_sleeves(
        portfolio=portfolio,
        enriched=enriched,
        features=features,
        exec_cfg=execution_cfg,
        asof=decision_asof,
    )
    blocked = [
        score["name"]
        for score in scores
        if any(str(reason).startswith("runtime_bridge=missing") for reason in score["reasons"])
    ]
    scoreable = [score["name"] for score in scores if score["name"] not in blocked]
    return {
        "mode": "forward_shadow_score_only",
        "orders_enabled": False,
        "portfolio": str(portfolio.get("name", "")),
        "portfolio_status": str(portfolio.get("status", "")),
        "decision_asof": str(decision_asof),
        "latest_completed_bar": str(enriched.iloc[-1]["date"]),
        "gross_weight": float(sum(float(row["weight"]) for row in portfolio["base_sleeves"])),
        "runtime_blocked_sleeves": blocked,
        "signal_scoring_ready_sleeves": scoreable,
        "signal_scoring_ready_count": len(scoreable),
        "complete_portfolio_runtime_ready": not blocked,
        "live_promotion_ready": False,
        "scores": scores,
    }


async def score_shadow_once(cfg: PortfolioShadowConfig) -> dict[str, Any]:
    portfolio = _load_json(cfg.portfolio_config)
    _validate_portfolio_mode(portfolio, live=False)
    if not bool(portfolio.get("shadow_only")):
        raise RuntimeError("portfolio_shadow requires shadow_only=true")
    minimum_history = int(portfolio.get("minimum_feature_history_minutes", 0))
    if int(cfg.lookback_minutes) < minimum_history:
        raise RuntimeError(
            "shadow lookback is shorter than the portfolio feature-history contract: "
            f"{cfg.lookback_minutes} < {minimum_history} minutes"
        )
    execution_cfg = WaveExecutionConfig.from_json(cfg.execution_config)
    wall_clock = pd.Timestamp.utcnow() if cfg.asof is None else pd.Timestamp(cfg.asof)
    if wall_clock.tzinfo is None:
        wall_clock = wall_clock.tz_localize("UTC")
    else:
        wall_clock = wall_clock.tz_convert("UTC")
    expected_bar = _expected_decision_bar(
        wall_clock,
        interval_minutes=int(execution_cfg.interval_minutes),
    )
    decision_asof = _completed_decision_data_asof(
        expected_bar,
        interval_minutes=int(execution_cfg.interval_minutes),
    )
    engine = sqlalchemy_engine_from_env(cfg.env_path)
    live_cfg = LiveDbFeatureConfig(lookback_minutes=int(cfg.lookback_minutes))
    enriched, features = await build_live_portfolio_frames(
        engine=engine,
        asof=decision_asof,
        cfg=live_cfg,
        live_oi_snapshot_cutoff=expected_bar + pd.Timedelta(minutes=execution_cfg.interval_minutes),
        include_activity_flow=False,
    )
    required_rows = int(
        math.ceil(minimum_history / max(1, int(execution_cfg.interval_minutes)))
    )
    if len(enriched) < required_rows:
        raise RuntimeError(
            "shadow market history is shorter than the portfolio feature-history contract: "
            f"{len(enriched)} < {required_rows} completed bars"
        )
    report = build_shadow_report(
        portfolio=portfolio,
        enriched=enriched,
        features=features,
        execution_cfg=execution_cfg,
        decision_asof=decision_asof,
    )
    latest = pd.Timestamp(enriched.iloc[-1]["date"])
    if latest.tzinfo is None:
        latest = latest.tz_localize("UTC")
    else:
        latest = latest.tz_convert("UTC")
    report["expected_completed_bar"] = str(expected_bar)
    report["completed_bar_fresh"] = bool(latest == expected_bar)
    report["feature_history_rows"] = len(enriched)
    report["required_feature_history_rows"] = required_rows
    report["feature_history_ready"] = True
    if latest != expected_bar:
        for score in report["scores"]:
            score["active"] = False
            score.setdefault("reasons", []).append(
                f"completed_decision_bar=fail:expected={expected_bar.isoformat()},actual={latest.isoformat()}"
            )
    if cfg.output is not None:
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        cfg.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a forward-shadow portfolio without orders")
    parser.add_argument(
        "--portfolio-config",
        default=str(PortfolioShadowConfig.portfolio_config),
    )
    parser.add_argument(
        "--execution-config",
        default=str(PortfolioShadowConfig.execution_config),
    )
    parser.add_argument("--env", default=str(PortfolioShadowConfig.env_path))
    parser.add_argument("--output", default=str(PortfolioShadowConfig.output))
    parser.add_argument("--lookback-minutes", type=int, default=PortfolioShadowConfig.lookback_minutes)
    parser.add_argument("--asof", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PortfolioShadowConfig(
        portfolio_config=Path(args.portfolio_config),
        execution_config=Path(args.execution_config),
        env_path=Path(args.env),
        output=Path(args.output) if args.output else None,
        lookback_minutes=int(args.lookback_minutes),
        asof=str(args.asof) or None,
    )
    report = asyncio.run(score_shadow_once(cfg))
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "orders_enabled": report["orders_enabled"],
                "completed_bar_fresh": report["completed_bar_fresh"],
                "runtime_blocked_sleeves": report["runtime_blocked_sleeves"],
                "active": [row["name"] for row in report["scores"] if row["active"]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
