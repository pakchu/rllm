"""Strict backtest for a predeclared episode-template portfolio.

This is the follow-up to the feature stability audit.  It does not search or
rank templates; callers must pass a fixed list of event+horizon specs.  The
script exists to separate structural diagnosis from deployable validation and
avoid accidentally tuning on eval.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.alpha_linear_combo_scan import _load_market, _parse_list
from training.price_action_episode_policy import (
    EPISODE_SIDES,
    EpisodePolicyCfg,
    add_sequence_context_features,
    build_episode_event_features,
    simulate_triggers,
    template_triggers,
)


@dataclass(frozen=True)
class FixedEpisodeBacktestCfg:
    input_csv: str
    output: str
    specs: str
    train_start: str = "2024-01-01"
    train_end: str = "2024-12-31 23:59:59"
    test_start: str = "2025-01-01"
    test_end: str = "2025-12-31 23:59:59"
    eval_start: str = "2026-01-01"
    eval_end: str = "2026-06-01"
    windows: str = "36,72,144,288,576,2016,4032,8640"
    include_sequence_context: bool = True
    entry_delay_bars: int = 1
    leverage: float = 0.5
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    max_trigger_overlap: float = 0.80


def _policy_cfg(cfg: FixedEpisodeBacktestCfg) -> EpisodePolicyCfg:
    return EpisodePolicyCfg(
        input_csv=cfg.input_csv,
        output=cfg.output,
        train_start=cfg.train_start,
        train_end=cfg.train_end,
        test_start=cfg.test_start,
        test_end=cfg.test_end,
        eval_start=cfg.eval_start,
        eval_end=cfg.eval_end,
        windows=cfg.windows,
        horizons="1",
        entry_delay_bars=cfg.entry_delay_bars,
        leverage=cfg.leverage,
        fee_rate=cfg.fee_rate,
        slippage_rate=cfg.slippage_rate,
        max_trigger_overlap=cfg.max_trigger_overlap,
        include_sequence_context=cfg.include_sequence_context,
    )


def _event_type(event: str) -> str:
    for suffix in sorted(EPISODE_SIDES, key=len, reverse=True):
        if event.endswith("_" + suffix):
            return suffix
    raise ValueError(f"cannot infer episode event type from {event!r}")


def _parse_specs(raw: str) -> list[dict[str, Any]]:
    """Parse comma-separated ``event@horizon[:score]`` specs."""
    specs = []
    for chunk in [x.strip() for x in raw.split(",") if x.strip()]:
        event_part, sep, rest = chunk.partition("@")
        if not sep:
            raise ValueError(f"spec lacks @horizon: {chunk!r}")
        horizon_part, _, score_part = rest.partition(":")
        event_type = _event_type(event_part)
        side, episode = EPISODE_SIDES[event_type]
        specs.append(
            {
                "event": event_part,
                "event_type": event_type,
                "episode": episode,
                "side": side,
                "window": int(event_part.split("_w", 1)[1].split("_", 1)[0]),
                "horizon": int(horizon_part),
                "score": float(score_part) if score_part else 0.0,
            }
        )
    if not specs:
        raise ValueError("at least one fixed template spec is required")
    return specs


def _overlap(a: set[int], b: set[int]) -> float:
    return len(a & b) / max(1, len(a | b))


def run(cfg: FixedEpisodeBacktestCfg) -> dict[str, Any]:
    market = _load_market(cfg.input_csv)
    dates = pd.to_datetime(market["date"])
    windows = _parse_list(cfg.windows, int)
    features = build_episode_event_features(market, windows)
    if cfg.include_sequence_context:
        features = add_sequence_context_features(market, features, windows)

    specs = _parse_specs(cfg.specs)
    selected = []
    triggers: list[dict[str, Any]] = []
    position_sets: list[set[int]] = []
    rejected = []
    for spec in specs:
        event = str(spec["event"])
        if event not in features.columns:
            rejected.append({"spec": spec, "reason": "missing_event_column"})
            continue
        events = features[event].to_numpy(dtype=float)
        positions = set(int(x) for x in np.flatnonzero(events > 0.5))
        overlap_hits = [
            {"selected_event": selected[i]["event"], "jaccard": _overlap(positions, prev)}
            for i, prev in enumerate(position_sets)
            if _overlap(positions, prev) > float(cfg.max_trigger_overlap)
        ]
        if overlap_hits:
            rejected.append({"spec": spec, "reason": "trigger_overlap_above_max", "overlaps": overlap_hits})
            continue
        selected.append(spec)
        position_sets.append(positions)
        triggers.extend(template_triggers(spec | {"events": events}, score=float(spec.get("score", 0.0))))

    policy_cfg = _policy_cfg(cfg)
    portfolio = {
        "train": simulate_triggers(market, dates, triggers, start=cfg.train_start, end=cfg.train_end, cfg=policy_cfg),
        "test": simulate_triggers(market, dates, triggers, start=cfg.test_start, end=cfg.test_end, cfg=policy_cfg),
        "eval": simulate_triggers(market, dates, triggers, start=cfg.eval_start, end=cfg.eval_end, cfg=policy_cfg),
    }
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "selected_templates": selected,
        "rejected_templates": rejected,
        "portfolio": {k: {"period": v["period"], "sim": v["sim"], "trade_stats": v["trade_stats"], "executed_sample": v["executed"][:20]} for k, v in portfolio.items()},
        "protocol": "fixed predeclared templates only; no search/ranking; eval is reported after fixed specs are instantiated",
        "leakage_guard": {
            "template_selection_uses_eval": False,
            "features_use_rows_at_or_before_t": True,
            "entry_uses_next_open": int(cfg.entry_delay_bars) >= 1,
            "strict_mdd_includes_intrabar_adverse_excursion": True,
        },
    }
    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--specs", required=True, help="comma-separated event@horizon[:score] specs")
    for name in ("train-start", "train-end", "test-start", "test-end", "eval-start", "eval-end"):
        p.add_argument(f"--{name}", default=getattr(FixedEpisodeBacktestCfg, name.replace("-", "_")))
    p.add_argument("--windows", default=FixedEpisodeBacktestCfg.windows)
    p.add_argument("--no-sequence-context", dest="include_sequence_context", action="store_false")
    p.set_defaults(include_sequence_context=FixedEpisodeBacktestCfg.include_sequence_context)
    p.add_argument("--entry-delay-bars", type=int, default=FixedEpisodeBacktestCfg.entry_delay_bars)
    p.add_argument("--leverage", type=float, default=FixedEpisodeBacktestCfg.leverage)
    p.add_argument("--fee-rate", type=float, default=FixedEpisodeBacktestCfg.fee_rate)
    p.add_argument("--slippage-rate", type=float, default=FixedEpisodeBacktestCfg.slippage_rate)
    p.add_argument("--max-trigger-overlap", type=float, default=FixedEpisodeBacktestCfg.max_trigger_overlap)
    return p.parse_args()


def main() -> None:
    report = run(FixedEpisodeBacktestCfg(**vars(parse_args())))
    print(json.dumps({
        "output": report["config"]["output"],
        "selected_templates": report["selected_templates"],
        "rejected_templates": report["rejected_templates"],
        "portfolio": {k: v["sim"] | {"p": v["trade_stats"].get("p_value_mean_ret_approx")} for k, v in report["portfolio"].items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
