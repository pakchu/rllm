"""Update the live portfolio anchor with executable post-anchor alpha sleeves.

The allocation protocol is deliberately frozen before reading the future
diagnostics produced by this run:

* candidate definitions are fixed by committed artifacts;
* allocation ranking uses only ``train`` and ``test2024``;
* ``eval2025`` and ``ytd2026`` can veto the already-ranked top row, but they
  cannot rerank it or select a lower row;
* non-zero weights are at least 0.25 and lie on a 0.05 grid;
* gross exposure is capped at 10 and correlated families are capped at 2.

The research history has already inspected every reported window.  Therefore
the output is a forward-shadow candidate, never an automatic live promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import training.portfolio_opt_all_discovered_alpha_gross10 as legacy_all
import training.portfolio_opt_combined_rex_new_alpha as legacy_base
import training.portfolio_opt_new_alpha_pool as new_alpha
from preprocessing.market_features import build_market_feature_frame
from training.audit_fresh_kimchi_orthogonal_alpha import (
    CANDIDATE_SPEC,
    DEFAULT_FUNDING,
    DEFAULT_INPUT,
    DEFAULT_PREMIUM,
    Config as FreshAuditConfig,
    build_candidate_context,
    build_rank7_context,
    candidate_schedule,
    rank7_schedule,
)
from training.audit_rank7_fresh_kimchi_fixed_portfolio import subaccount_bar_path
from training.audit_rex8640_usdkrw_gate import gate_match as rex_gate_match
from training.audit_weak_feature_responsibility_stability import (
    _action_spec as rank7_action_spec,
)
from training.evaluate_expanding_extratrees_top10_oos import FULL_CUTOFF
from training.long_regime_interest_gate_validation import build_interest_features
from training.search_funding_premium_independent_gate_alpha import (
    _apply_gate as apply_independent_gate,
    _build_base_components as build_independent_base,
    _build_gate_features as build_independent_features,
    _gate_mask as independent_gate_mask,
)
from training.search_gaussian_hmm_regime_alpha import hourly_features


OUTPUT = "results/portfolio_added_alpha_update_2026-07-16.json"
DOCS_OUTPUT = "docs/portfolio-added-alpha-update-2026-07-16.md"
CANDIDATE_CONFIG = "configs/live/portfolio_added_alpha_shadow_candidate_2026-07-16.json"

LIVE_WEIGHTS = {
    "oi_upbit_ratio288_low": 0.65,
    "new_long_minimal_funding_premium": 1.75,
    "cand_rex_veto_7": 1.45,
}
NEW_SLEEVES = (
    "frozen_annual_rank7",
    "fresh_kimchi_fx",
    "markov_transition_long",
    "funding_premium_lr_impact_central",
    "rex_taker_low_range_position",
)
SLEEVES = tuple(LIVE_WEIGHTS) + NEW_SLEEVES
FAMILIES = {
    "oi_upbit_ratio288_low": "oi",
    "new_long_minimal_funding_premium": "funding_premium",
    "cand_rex_veto_7": "rex",
    "frozen_annual_rank7": "rank7",
    "fresh_kimchi_fx": "kimchi_fx",
    "markov_transition_long": "funding_premium",
    "funding_premium_lr_impact_central": "funding_premium",
    "rex_taker_low_range_position": "rex",
}
SPLIT_BOUNDS = {
    "train": ("2020-09-01", "2024-01-01"),
    "test2024": ("2024-01-01", "2025-01-01"),
    "eval2025": ("2025-01-01", "2026-01-01"),
    # Match the authoritative live portfolio clock.  The end is exclusive and
    # intentionally counts the full no-trade calendar through June 2.
    "ytd2026": ("2026-01-01", "2026-06-03"),
}
REX_GATES = (
    {"feature": "taker_imbalance", "op": "<=", "threshold": -0.07073595391836504},
    {"feature": "rex_2016_range_pos", "op": "<=", "threshold": 0.6865011402825759},
)
EXPECTED_LIVE_LEGACY = {
    "train": (523.6041861154301, 31.897074004308923, 818),
    "test2024": (66.9367849572657, 13.878107437867515, 172),
    "eval2025": (61.2009149651342, 10.005706358844435, 109),
    "ytd2026": (24.89053385433926, 7.271147051069471, 65),
}


@dataclass(frozen=True)
class Config:
    input_csv: str = DEFAULT_INPUT
    funding_csv: str = DEFAULT_FUNDING
    premium_csv: str = DEFAULT_PREMIUM
    output: str = OUTPUT
    docs_output: str = DOCS_OUTPUT
    candidate_config: str = CANDIDATE_CONFIG
    gross_cap: float = 10.0
    family_gross_cap: float = 2.0
    min_nonzero_weight: float = 0.25
    weight_step: float = 0.05
    train_mdd_cap: float = 40.0
    test_mdd_cap: float = 20.0
    future_mdd_cap: float = 20.0
    min_test_trades: int = 80
    min_test_ratio: float = 3.0
    min_future_ratio: float = 3.0
    random_samples: int = 60_000
    exact_batch_size: int = 16
    seed: int = 71616
    seed_count: int = 2
    refinement_rounds: int = 20
    refinement_top_n: int = 20
    refinement_patience: int = 3
    cost_rate: float = 0.0006


def resolve_existing(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    fallback = Path("/home/pakchu/rllm") / candidate
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(path)


def ensure_runtime_inputs() -> None:
    """Restore ignored data links/temp OI input without copying large files."""
    aux = Path("data/binance_um_aux_btc_2020_2026")
    fallback_aux = Path("/home/pakchu/rllm/data/binance_um_aux_btc_2020_2026")
    if not aux.exists() and fallback_aux.exists():
        aux.symlink_to(fallback_aux, target_is_directory=True)
    oi_target = Path("/tmp/btcusdt_open_interest_5m_2020_2026.csv")
    if not oi_target.exists():
        source = resolve_existing(
            "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
        )
        pd.read_csv(source, usecols=["date", "open_interest"]).to_csv(oi_target, index=False)


def json_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    resolved = resolve_existing(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def favorable_path(
    market: pd.DataFrame,
    *,
    signal_position: int,
    exit_position: int,
    side: str,
    leverage: float,
) -> np.ndarray:
    """Return the favorable OHLC envelope relative to each bar open."""
    favorable = np.zeros(len(market), dtype=np.float64)
    entry = int(signal_position) + 1
    end = min(int(exit_position), len(market) - 1)
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    for position in range(entry, end):
        open_price = float(opens[position])
        if open_price <= 0.0:
            continue
        move = (
            (float(highs[position]) - open_price) / open_price
            if side == "long"
            else (open_price - float(lows[position])) / open_price
        )
        favorable[position] = max(0.0, float(leverage) * move)
    return favorable


def append_mask_policy(
    events: list[dict[str, Any]],
    market: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    name: str,
    long_active: np.ndarray,
    short_active: np.ndarray,
    hold: int,
    stride: int,
    cost_rate: float,
    take: float | None = None,
    stop: float | None = None,
) -> dict[str, int]:
    dates = pd.to_datetime(market["date"])
    positions = np.arange(143, max(0, len(market) - hold - 2), stride, dtype=np.int64)
    counts: dict[str, int] = {}
    for split, split_mask in masks.items():
        returns = np.zeros(len(market), dtype=np.float64)
        adverse = np.zeros(len(market), dtype=np.float64)
        favorable = np.zeros(len(market), dtype=np.float64)
        next_allowed = 0
        trades = wins = 0
        first_signal: int | None = None
        active = split_mask[positions] & (long_active[positions] | short_active[positions])
        for raw_position in positions[active]:
            position = int(raw_position)
            if position < next_allowed:
                continue
            side = (
                "long"
                if bool(long_active[position]) and not bool(short_active[position])
                else "short"
                if bool(short_active[position]) and not bool(long_active[position])
                else ""
            )
            if not side:
                continue
            path = new_alpha._event_path(
                market,
                position,
                side=side,
                hold=hold,
                cost_rate=cost_rate,
                tp=take,
                sl=stop,
                entry_delay=1,
                leverage=0.5,
            )
            if path is None:
                continue
            event_return, event_adverse, realized = path
            nonzero = np.flatnonzero(np.abs(event_return) > 1e-15)
            exit_position = int(nonzero[-1]) if len(nonzero) else position + hold + 1
            if exit_position >= len(split_mask) or not split_mask[exit_position]:
                continue
            returns += event_return
            adverse += event_adverse
            favorable += favorable_path(
                market,
                signal_position=position,
                exit_position=exit_position,
                side=side,
                leverage=0.5,
            )
            trades += 1
            wins += int(float(realized) > 0.0)
            first_signal = position if first_signal is None else first_signal
            next_allowed = exit_position + 1
        counts[split] = trades
        if trades:
            events.append(
                {
                    "split": split,
                    "sleeve": name,
                    "side": "mixed",
                    "signal_pos": int(first_signal or 0),
                    "date": str(dates.iloc[int(first_signal or 0)]),
                    "ret": returns,
                    "adv": adverse,
                    "fav": favorable,
                    "trade_count": trades,
                    "win_count": wins,
                }
            )
    return counts


def feature_frame(market: pd.DataFrame) -> pd.DataFrame:
    base = build_market_feature_frame(market, window_size=144)
    return pd.concat([base, build_interest_features(market, base)], axis=1).loc[
        :, lambda frame: ~frame.columns.duplicated(keep="last")
    ]


def markov_active(market: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
    record = json.loads(
        Path("research/pools/alphas/markov_persistent_funding_premium_long_20260712.json").read_text()
    )
    spec = record["state_model"]
    setup = new_alpha._alpha_active(features, "long_minimal_funding_premium")
    _, hourly = hourly_features(market)
    trend = np.where(
        hourly["trend24"] <= float(spec["trend_low"]),
        0,
        np.where(hourly["trend24"] >= float(spec["trend_high"]), 2, 1),
    )
    volatility = (hourly["vol24"] >= float(spec["vol_median"])).astype(int)
    flow = (hourly["flow24"] >= float(spec["flow_median"])).astype(int)
    state = trend * 4 + volatility * 2 + flow
    previous = pd.Series(state, index=hourly.index).shift(1).fillna(-1).astype(int)
    transitions = previous * 12 + state
    mapped = pd.merge_asof(
        pd.DataFrame({"date": pd.to_datetime(market["date"]), "position": np.arange(len(market))}),
        pd.DataFrame({"date": hourly.index.to_numpy(), "transition": transitions.to_numpy()}),
        on="date",
        direction="backward",
        tolerance=pd.Timedelta("2h"),
    ).sort_values("position")
    transition = mapped["transition"].fillna(-1).to_numpy(int)
    return setup & np.isin(transition, np.asarray(spec["allowed_transition_keys"], dtype=int))


def funding_lr_active(market: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    candidate = json.loads(
        Path("configs/live/funding_premium_lr_impact_central_research_candidate.json").read_text()
    )
    manifest = json.loads(
        Path("results/funding_premium_independent_gate_top10_manifest_2026-07-13.json").read_text()
    )
    expected_hash = str(candidate["selection"]["manifest_sha256"])
    if str(manifest.get("sha256")) != expected_hash:
        raise RuntimeError("funding-premium manifest hash drifted from the frozen candidate")
    rank = int(candidate["selection"]["manifest_rank"])
    frozen = manifest["selected"][rank - 1]
    spec = {
        key: frozen[key]
        for key in ("feature", "tail", "lower", "upper", "gate_mode", "target_component")
    }
    funding_component, premium_component = build_independent_base(market)
    features = build_independent_features(market)
    gate = independent_gate_mask(features[spec["feature"]].to_numpy(float), spec)
    active = apply_independent_gate(
        funding_component, premium_component, gate, spec["target_component"]
    )
    return active, {"manifest_rank": rank, "spec": spec, "manifest_sha256": manifest["sha256"]}


def append_rex_taker_policy(
    events: list[dict[str, Any]],
    market: pd.DataFrame,
    masks: dict[str, np.ndarray],
    *,
    cost_rate: float,
) -> dict[str, int]:
    paths = (
        "data/rex_pullback_reclaim_q075_h144_ranker_train_2021_2023.jsonl",
        "data/rex_pullback_reclaim_q075_h144_ranker_test_2024.jsonl",
        "data/rex_pullback_reclaim_q075_h144_ranker_eval_2025_2026h1.jsonl",
    )
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for path in paths:
        for line in resolve_existing(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(int(row["signal_pos"]), str(row["date"]))] = row
    ordered = sorted(rows.values(), key=lambda row: int(row["signal_pos"]))
    dates = pd.to_datetime(market["date"])
    counts: dict[str, int] = {}
    for split, split_mask in masks.items():
        returns = np.zeros(len(market), dtype=np.float64)
        adverse = np.zeros(len(market), dtype=np.float64)
        favorable = np.zeros(len(market), dtype=np.float64)
        next_allowed = 0
        trades = wins = 0
        first_signal: int | None = None
        for row in ordered:
            position = int(row["signal_pos"])
            if position < next_allowed or position >= len(split_mask) or not split_mask[position]:
                continue
            if pd.Timestamp(row["date"]) != pd.Timestamp(dates.iloc[position]):
                raise RuntimeError("REX source row is not aligned to the shared market grid")
            if not rex_gate_match(row, list(REX_GATES)):
                continue
            side = str((row.get("action") or {}).get("side", "")).lower()
            if side not in {"long", "short"}:
                continue
            path = new_alpha._event_path(
                market,
                position,
                side=side,
                hold=144,
                cost_rate=cost_rate,
                entry_delay=1,
                leverage=0.5,
            )
            if path is None:
                continue
            event_return, event_adverse, realized = path
            exit_position = position + 145
            if exit_position >= len(split_mask) or not split_mask[exit_position]:
                continue
            returns += event_return
            adverse += event_adverse
            favorable += favorable_path(
                market,
                signal_position=position,
                exit_position=exit_position,
                side=side,
                leverage=0.5,
            )
            trades += 1
            wins += int(float(realized) > 0.0)
            first_signal = position if first_signal is None else first_signal
            next_allowed = exit_position + 1
        counts[split] = trades
        if trades:
            events.append(
                {
                    "split": split,
                    "sleeve": "rex_taker_low_range_position",
                    "side": "mixed",
                    "signal_pos": int(first_signal or 0),
                    "date": str(dates.iloc[int(first_signal or 0)]),
                    "ret": returns,
                    "adv": adverse,
                    "fav": favorable,
                    "trade_count": trades,
                    "win_count": wins,
                }
            )
    return counts


def attach_live_rex_favorable(
    events: list[dict[str, Any]],
    market: pd.DataFrame,
    masks: dict[str, np.ndarray],
    cfg: Config,
) -> None:
    """Rebuild the live aggregated REX sleeve's favorable OHLC envelope."""
    report = legacy_all.load_json(legacy_all.SCAN_FILES["rex_veto"])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bucket in ("top", "tte_top"):
        for row in report.get(bucket, [])[:50]:
            key = json.dumps(row.get("gates", []), sort_keys=True)
            if key not in seen:
                seen.add(key)
                rows.append(row)
    row_index = 7
    if row_index >= len(rows):
        raise RuntimeError("frozen live REX-veto row 7 is missing")
    gate_row = rows[row_index]
    source = [
        json.loads(line)
        for line in Path("data/rex_event_reasoning_policy_sft_20260712.jsonl").read_text().splitlines()
        if line.strip()
    ]
    features = legacy_all._build_light_rex_features(market)
    dates = pd.to_datetime(market["date"])
    for split, split_mask in masks.items():
        favorable = np.zeros(len(market), dtype=np.float64)
        next_allowed = 0
        trades = 0
        for source_row in source:
            position = int(source_row.get("signal_pos", -1))
            if (
                position < 0
                or position >= len(market)
                or not split_mask[position]
                or position < next_allowed
            ):
                continue
            if pd.Timestamp(source_row["date"]) != pd.Timestamp(dates.iloc[position]):
                raise RuntimeError("live REX source row is not aligned to the shared market grid")
            side = str((source_row.get("base_event") or {}).get("base_side", "")).lower()
            if side not in {"long", "short"} or not legacy_all._rex_row_matches(
                gate_row.get("gates", []), features, source_row
            ):
                continue
            exit_position = position + 145
            if exit_position >= len(market) or not split_mask[exit_position]:
                continue
            favorable += favorable_path(
                market,
                signal_position=position,
                exit_position=exit_position,
                side=side,
                leverage=0.5,
            )
            trades += 1
            next_allowed = exit_position + 1
        matches = [
            event
            for event in events
            if event["split"] == split and event["sleeve"] == "cand_rex_veto_7"
        ]
        if len(matches) != 1 or int(matches[0].get("trade_count", 0)) != trades:
            raise RuntimeError(f"live REX favorable replay drifted in {split}")
        matches[0]["fav"] = favorable


def attach_default_favorable(events: list[dict[str, Any]], market: pd.DataFrame) -> None:
    """Attach exact OHLC upper envelopes to non-aggregated legacy events."""
    leverage_by_sleeve = {
        "oi_upbit_ratio288_low": 1.0,
        "new_long_minimal_funding_premium": 0.5,
    }
    for event in events:
        if "fav" in event:
            continue
        side = str(event.get("side", "")).lower()
        if side not in {"long", "short"}:
            raise RuntimeError(f"missing favorable replay for aggregated sleeve {event['sleeve']}")
        nonzero = np.flatnonzero(np.abs(event["ret"]) > 1e-15)
        if not len(nonzero):
            event["fav"] = np.zeros(len(market), dtype=np.float64)
            continue
        event["fav"] = favorable_path(
            market,
            signal_position=int(event["signal_pos"]),
            exit_position=int(nonzero[-1]),
            side=side,
            leverage=leverage_by_sleeve[event["sleeve"]],
        )


def path_event(
    market: pd.DataFrame,
    path: Any,
    *,
    split: str,
    sleeve: str,
    trades: Iterable[Any],
) -> dict[str, Any]:
    dates = pd.DatetimeIndex(pd.to_datetime(market["date"]))
    positions = dates.get_indexer(path.dates)
    if (positions < 0).any():
        raise RuntimeError(f"{sleeve} path is not aligned to the shared market grid")
    previous_close = np.r_[1.0, path.close_value[:-1]]
    local_return = path.close_value / previous_close - 1.0
    local_adverse = (
        np.minimum.reduce(
            [path.open_value, path.market_low_value, path.market_high_value, path.close_value]
        )
        / previous_close
        - 1.0
    )
    local_favorable = (
        np.maximum.reduce(
            [path.open_value, path.market_low_value, path.market_high_value, path.close_value]
        )
        / previous_close
        - 1.0
    )
    local_adverse = np.minimum(0.0, local_adverse)
    local_favorable = np.maximum(0.0, local_favorable)
    returns = np.zeros(len(market), dtype=np.float64)
    adverse = np.zeros(len(market), dtype=np.float64)
    favorable = np.zeros(len(market), dtype=np.float64)
    returns[positions] = local_return
    adverse[positions] = local_adverse
    favorable[positions] = local_favorable
    trades = list(trades)
    return {
        "split": split,
        "sleeve": sleeve,
        "side": "mixed",
        "signal_pos": int(trades[0].signal_position) if trades else int(positions[0]),
        "date": str(path.dates[0]),
        "ret": returns,
        "adv": adverse,
        "fav": favorable,
        "trade_count": len(trades),
        "win_count": sum(
            float(trade.price_factor) * float(trade.funding_factor) > 1.0 for trade in trades
        ),
    }


def append_rank7_and_fresh(
    events: list[dict[str, Any]], cfg: Config
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    audit_cfg = FreshAuditConfig(
        input_csv=str(resolve_existing(cfg.input_csv)),
        funding_csv=str(resolve_existing(cfg.funding_csv)),
        premium_csv=str(resolve_existing(cfg.premium_csv)),
        output="/tmp/no_write_portfolio_added_alpha.json",
        docs_output="",
        exclude_from=FULL_CUTOFF,
    )
    fresh = build_candidate_context(audit_cfg)
    rank7 = build_rank7_context(audit_cfg)
    market = fresh["market"]
    rank7_market = rank7["base"]["context"]["market"]
    if not np.array_equal(pd.to_datetime(market["date"]), pd.to_datetime(rank7_market["date"])):
        raise RuntimeError("rank7/fresh market grids differ")
    funding_leg = np.asarray(rank7["base"]["context"]["funding_leg"], dtype=bool)
    counts = {"frozen_annual_rank7": {}, "fresh_kimchi_fx": {}}
    hashes: dict[str, Any] = {}
    for split, (start, end) in SPLIT_BOUNDS.items():
        fresh_trades = candidate_schedule(fresh, start=start, end=end)
        rank7_trades = rank7_schedule(rank7, start=start, end=end)
        fresh_path = subaccount_bar_path(
            market,
            fresh["funding"],
            fresh_trades,
            fresh["execution_cfg"],
            start=start,
            end=end,
            hold_bars=lambda _trade: int(CANDIDATE_SPEC["hold_bars"]),
        )
        rank7_path = subaccount_bar_path(
            rank7_market,
            rank7["base"]["context"]["funding"],
            rank7_trades,
            rank7["base"]["execution_cfg"],
            start=start,
            end=end,
            hold_bars=lambda trade: int(
                rank7_action_spec(bool(funding_leg[trade.signal_position]))[0]
            ),
        )
        events.append(
            path_event(
                market,
                rank7_path,
                split=split,
                sleeve="frozen_annual_rank7",
                trades=rank7_trades,
            )
        )
        events.append(
            path_event(
                market,
                fresh_path,
                split=split,
                sleeve="fresh_kimchi_fx",
                trades=fresh_trades,
            )
        )
        counts["frozen_annual_rank7"][split] = len(rank7_trades)
        counts["fresh_kimchi_fx"][split] = len(fresh_trades)
        hashes[split] = {
            "rank7_final_equity": rank7_path.final_equity,
            "fresh_final_equity": fresh_path.final_equity,
        }
    return counts, hashes


def split_arrays(
    events: list[dict[str, Any]],
    market: pd.DataFrame,
    masks: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    dates = pd.to_datetime(market["date"])
    output: dict[str, dict[str, Any]] = {}
    for split, split_mask in masks.items():
        positions = np.flatnonzero(split_mask)
        start, end = int(positions[0]), int(positions[-1]) + 1
        returns = np.zeros((len(SLEEVES), end - start), dtype=np.float64)
        adverse = np.zeros_like(returns)
        favorable = np.zeros_like(returns)
        counts = np.zeros(len(SLEEVES), dtype=np.int64)
        wins = np.zeros(len(SLEEVES), dtype=np.int64)
        for event in events:
            if event["split"] != split or event["sleeve"] not in SLEEVES:
                continue
            index = SLEEVES.index(event["sleeve"])
            returns[index] += event["ret"][start:end]
            adverse[index] += event["adv"][start:end]
            favorable[index] += event["fav"][start:end]
            trade_count = int(event.get("trade_count", 1))
            counts[index] += trade_count
            if "win_count" in event:
                wins[index] += int(event["win_count"])
            elif trade_count == 1:
                wins[index] += int(float(event.get("ret_bps", 0.0)) > 0.0)
        output[split] = {
            "R": returns,
            "A": adverse,
            "U": favorable,
            "counts": counts,
            "wins": wins,
            "dates": pd.DatetimeIndex(dates.iloc[start:end]),
        }
    return output


def legacy_metric(data: dict[str, Any], years: float, weights: dict[str, float]) -> dict[str, Any]:
    vector = np.asarray([weights.get(name, 0.0) for name in SLEEVES], dtype=float)
    returns = vector @ data["R"]
    adverse = vector @ data["A"]
    factor = np.maximum(0.0, 1.0 + returns)
    equity_after = np.cumprod(factor)
    equity_before = np.r_[1.0, equity_after[:-1]]
    peak_after = np.maximum.accumulate(equity_after)
    peak_before = np.maximum.accumulate(equity_before)
    drawdown_after = np.max(1.0 - equity_after / np.maximum(peak_after, 1e-12))
    drawdown_adverse = np.max(
        1.0
        - equity_before * np.maximum(0.0, 1.0 + adverse) / np.maximum(peak_before, 1e-12)
    )
    final = float(equity_after[-1])
    total_return = (final - 1.0) * 100.0
    cagr = (final ** (1.0 / years) - 1.0) * 100.0 if final > 0 else -100.0
    mdd = max(float(drawdown_after), float(drawdown_adverse)) * 100.0
    selected = vector > 0.0
    return {
        "absolute_return_pct": total_return,
        "cagr_pct": cagr,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": cagr / mdd if mdd > 1e-12 else 0.0,
        "trades": int(data["counts"][selected].sum()),
    }


def strict_metric(data: dict[str, Any], years: float, weights: dict[str, float]) -> dict[str, Any]:
    """Conservative OHLC upper-before-lower MDD on the shared portfolio clock."""
    vector = np.asarray([weights.get(name, 0.0) for name in SLEEVES], dtype=float)
    returns = vector @ data["R"]
    adverse = vector @ data["A"]
    favorable = vector @ data["U"]
    equity_after = np.cumprod(np.maximum(0.0, 1.0 + returns))
    equity_before = np.r_[1.0, equity_after[:-1]]
    upper = np.maximum.reduce(
        [equity_before, equity_after, equity_before * np.maximum(0.0, 1.0 + favorable)]
    )
    peak = np.maximum.accumulate(upper)
    lower = np.minimum(
        equity_after,
        equity_before * np.maximum(0.0, 1.0 + adverse),
    )
    mdd = float(np.max(1.0 - lower / np.maximum(peak, 1e-12))) * 100.0
    final = float(equity_after[-1])
    total_return = (final - 1.0) * 100.0
    cagr = (final ** (1.0 / years) - 1.0) * 100.0 if final > 0 else -100.0
    selected = vector > 0.0
    trades = int(data["counts"][selected].sum())
    wins = int(data["wins"][selected].sum())
    return {
        "absolute_return_pct": total_return,
        "cagr_pct": cagr,
        "strict_mdd_pct": mdd,
        "cagr_to_strict_mdd": cagr / mdd if mdd > 1e-12 else 0.0,
        "trades": trades,
        "win_rate": wins / trades if trades else 0.0,
        "trades_by_sleeve": {
            name: int(data["counts"][index])
            for index, name in enumerate(SLEEVES)
            if vector[index] > 0.0
        },
    }


def years_for(split: str) -> float:
    start, end = SPLIT_BOUNDS[split]
    return (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / (365.25 * 86_400.0)


def quantize_weights(weights: dict[str, float], cfg: Config) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, raw in weights.items():
        if name not in SLEEVES or float(raw) <= 0.0:
            continue
        value = round(float(raw) / cfg.weight_step) * cfg.weight_step
        if value + 1e-12 >= cfg.min_nonzero_weight:
            output[name] = round(value, 10)
    return output


def valid_weights(weights: dict[str, float], cfg: Config) -> bool:
    if not weights or any(name not in SLEEVES for name in weights):
        return False
    if sum(weights.values()) > cfg.gross_cap + 1e-9:
        return False
    family_gross: dict[str, float] = {}
    for name, weight in weights.items():
        if weight + 1e-12 < cfg.min_nonzero_weight:
            return False
        units = weight / cfg.weight_step
        if not np.isclose(units, round(units), atol=1e-9):
            return False
        family = FAMILIES[name]
        family_gross[family] = family_gross.get(family, 0.0) + float(weight)
    return all(value <= cfg.family_gross_cap + 1e-9 for value in family_gross.values())


def weight_candidates(cfg: Config) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()

    def add(raw: dict[str, float]) -> None:
        weights = quantize_weights(raw, cfg)
        if not valid_weights(weights, cfg):
            return
        key = tuple(round(weights.get(name, 0.0), 4) for name in SLEEVES)
        if key not in seen:
            seen.add(key)
            candidates.append(weights)

    add(LIVE_WEIGHTS)
    for scale in (0.5, 0.75, 1.0):
        anchor = {name: weight * scale for name, weight in LIVE_WEIGHTS.items()}
        add(anchor)
        for sleeve in NEW_SLEEVES:
            for weight in np.arange(0.25, 2.001, 0.25):
                add({**anchor, sleeve: float(weight)})
    for left_index, left in enumerate(NEW_SLEEVES):
        for right in NEW_SLEEVES[left_index + 1 :]:
            for left_weight in (0.25, 0.5, 1.0, 1.5, 2.0):
                for right_weight in (0.25, 0.5, 1.0, 1.5, 2.0):
                    add({**LIVE_WEIGHTS, left: left_weight, right: right_weight})
                    add({left: left_weight, right: right_weight})

    for seed in range(int(cfg.seed), int(cfg.seed) + int(cfg.seed_count)):
        rng = random.Random(seed)
        for _ in range(cfg.random_samples):
            chosen = rng.sample(list(SLEEVES), rng.randint(2, min(7, len(SLEEVES))))
            gross = rng.choice(np.arange(2.0, cfg.gross_cap + 0.001, 0.5).tolist())
            raw = np.asarray([rng.random() ** 1.35 for _ in chosen], dtype=float)
            raw *= float(gross) / raw.sum()
            add({name: float(value) for name, value in zip(chosen, raw, strict=True)})
    return candidates


def weight_neighbors(weights: dict[str, float], cfg: Config) -> list[dict[str, float]]:
    """Generate deterministic one-step and capital-transfer grid neighbors."""
    output: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()

    def add(raw: dict[str, float]) -> None:
        candidate = quantize_weights(raw, cfg)
        if not valid_weights(candidate, cfg):
            return
        key = tuple(candidate.get(name, 0.0) for name in SLEEVES)
        if key not in seen:
            seen.add(key)
            output.append(candidate)

    for name in SLEEVES:
        current = float(weights.get(name, 0.0))
        if current <= 0.0:
            for value in (cfg.min_nonzero_weight, 0.5):
                add({**weights, name: value})
            continue
        removed = dict(weights)
        removed.pop(name, None)
        add(removed)
        for delta in (-0.25, -cfg.weight_step, cfg.weight_step, 0.25):
            add({**weights, name: current + delta})

    for left in SLEEVES:
        left_weight = float(weights.get(left, 0.0))
        if left_weight <= 0.0:
            continue
        for right in SLEEVES:
            if right == left:
                continue
            right_weight = float(weights.get(right, 0.0))
            for delta in (cfg.weight_step, 0.25):
                moved = dict(weights)
                moved[left] = left_weight - delta
                moved[right] = right_weight + delta
                add(moved)
    return output


def batch_metrics(data: dict[str, Any], years: float, weight_matrix: np.ndarray) -> dict[str, np.ndarray]:
    returns = weight_matrix @ data["R"]
    adverse = weight_matrix @ data["A"]
    favorable = weight_matrix @ data["U"]
    equity_after = np.cumprod(np.maximum(0.0, 1.0 + returns), axis=1)
    equity_before = np.concatenate([np.ones((len(weight_matrix), 1)), equity_after[:, :-1]], axis=1)
    upper = np.maximum.reduce(
        [equity_before, equity_after, equity_before * np.maximum(0.0, 1.0 + favorable)]
    )
    peak = np.maximum.accumulate(upper, axis=1)
    lower = np.minimum(
        equity_after,
        equity_before * np.maximum(0.0, 1.0 + adverse),
    )
    mdd = np.max(1.0 - lower / np.maximum(peak, 1e-12), axis=1) * 100.0
    final = equity_after[:, -1]
    total_return = (final - 1.0) * 100.0
    cagr = np.where(final > 0.0, (np.power(final, 1.0 / years) - 1.0) * 100.0, -100.0)
    ratio = np.divide(cagr, mdd, out=np.zeros_like(cagr), where=mdd > 1e-12)
    trades = (weight_matrix > 0.0).astype(np.int64) @ data["counts"]
    return {"return": total_return, "cagr": cagr, "mdd": mdd, "ratio": ratio, "trades": trades}


def pre2025_passes(stats: dict[str, Any], cfg: Config) -> bool:
    train = stats["train"]
    test = stats["test2024"]
    return bool(
        train["absolute_return_pct"] > 0.0
        and test["absolute_return_pct"] > 0.0
        and train["strict_mdd_pct"] <= cfg.train_mdd_cap
        and test["strict_mdd_pct"] <= cfg.test_mdd_cap
        and test["cagr_to_strict_mdd"] >= cfg.min_test_ratio
        and test["trades"] >= cfg.min_test_trades
    )


def pre2025_selection_key(stats: dict[str, Any], cfg: Config) -> tuple[Any, ...]:
    train = stats["train"]
    test = stats["test2024"]
    train_ratio = float(train["cagr_to_strict_mdd"])
    test_ratio = float(test["cagr_to_strict_mdd"])
    return (
        pre2025_passes(stats, cfg),
        min(train_ratio, test_ratio),
        float(np.sqrt(max(0.0, train_ratio) * max(0.0, test_ratio))),
        test_ratio,
        train["absolute_return_pct"] + test["absolute_return_pct"],
        -max(train["strict_mdd_pct"], test["strict_mdd_pct"]),
    )


def forward_veto(stats: dict[str, Any], cfg: Config) -> bool:
    return all(
        stats[split]["absolute_return_pct"] > 0.0
        and stats[split]["strict_mdd_pct"] <= cfg.future_mdd_cap
        and stats[split]["cagr_to_strict_mdd"] >= cfg.min_future_ratio
        for split in ("eval2025", "ytd2026")
    )


def _batch_metric_at(metrics: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    return {
        "absolute_return_pct": float(metrics["return"][index]),
        "cagr_pct": float(metrics["cagr"][index]),
        "strict_mdd_pct": float(metrics["mdd"][index]),
        "cagr_to_strict_mdd": float(metrics["ratio"][index]),
        "trades": int(metrics["trades"][index]),
    }


def exact_pre2025_rows(
    arrays: dict[str, dict[str, Any]],
    candidates: list[dict[str, float]],
    cfg: Config,
) -> list[dict[str, Any]]:
    """Rank every generated candidate on exact 5-minute train/2024 paths."""
    matrix = np.asarray(
        [[weights.get(name, 0.0) for name in SLEEVES] for weights in candidates], dtype=float
    )
    rows: list[dict[str, Any]] = []
    chunk_size = max(1, int(cfg.exact_batch_size))
    for start in range(0, len(matrix), chunk_size):
        end = min(len(matrix), start + chunk_size)
        weights = matrix[start:end]
        train = batch_metrics(arrays["train"], years_for("train"), weights)
        test = batch_metrics(arrays["test2024"], years_for("test2024"), weights)
        passes = (
            (train["return"] > 0.0)
            & (test["return"] > 0.0)
            & (train["mdd"] <= cfg.train_mdd_cap)
            & (test["mdd"] <= cfg.test_mdd_cap)
            & (test["ratio"] >= cfg.min_test_ratio)
            & (test["trades"] >= cfg.min_test_trades)
        )
        for offset in np.flatnonzero(passes):
            candidate_index = start + int(offset)
            stats = {
                "train": _batch_metric_at(train, int(offset)),
                "test2024": _batch_metric_at(test, int(offset)),
            }
            rows.append(
                {
                    "weights": candidates[candidate_index],
                    "gross": round(sum(candidates[candidate_index].values()), 6),
                    "stats": stats,
                    "selection_key": pre2025_selection_key(stats, cfg),
                }
            )
    rows.sort(key=pre2025_row_sort_key, reverse=True)
    return rows


def _weight_key(weights: dict[str, float]) -> tuple[float, ...]:
    return tuple(round(float(weights.get(name, 0.0)), 10) for name in SLEEVES)


def pre2025_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Prefer lower gross/lexicographic weights when performance is exactly tied."""
    weights = _weight_key(row["weights"])
    return (row["selection_key"], -float(row["gross"]), tuple(-value for value in weights))


def refine_pre2025_rows(
    arrays: dict[str, dict[str, Any]],
    initial_candidates: list[dict[str, float]],
    initial_rows: list[dict[str, Any]],
    cfg: Config,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Climb the exact 0.05 grid without ever reading future windows."""
    ranked = list(initial_rows)
    seen = {_weight_key(weights) for weights in initial_candidates}
    rounds: list[dict[str, Any]] = []
    total_evaluated = 0
    frontier = ranked[: int(cfg.refinement_top_n)]
    stalled_rounds = 0
    for round_index in range(int(cfg.refinement_rounds)):
        neighbors: list[dict[str, float]] = []
        for row in frontier:
            for candidate in weight_neighbors(row["weights"], cfg):
                key = _weight_key(candidate)
                if key not in seen:
                    seen.add(key)
                    neighbors.append(candidate)
        if not neighbors:
            break
        previous_key = ranked[0]["selection_key"]
        new_rows = exact_pre2025_rows(arrays, neighbors, cfg)
        total_evaluated += len(neighbors)
        by_weight = {_weight_key(row["weights"]): row for row in ranked}
        by_weight.update({_weight_key(row["weights"]): row for row in new_rows})
        ranked = sorted(
            by_weight.values(), key=pre2025_row_sort_key, reverse=True
        )
        frontier = sorted(new_rows, key=pre2025_row_sort_key, reverse=True)[
            : int(cfg.refinement_top_n)
        ]
        improved = ranked[0]["selection_key"] > previous_key
        stalled_rounds = 0 if improved else stalled_rounds + 1
        rounds.append(
            {
                "round": round_index + 1,
                "evaluated": len(neighbors),
                "passed": len(new_rows),
                "top_improved": improved,
                "top_weights": ranked[0]["weights"],
                "stalled_rounds": stalled_rounds,
            }
        )
        if stalled_rounds >= int(cfg.refinement_patience):
            break
    return ranked, {
        "evaluated": total_evaluated,
        "rounds": rounds,
        "stopped_after_stalled_rounds": stalled_rounds,
        "patience": int(cfg.refinement_patience),
    }


def _format_metric(metric: dict[str, Any]) -> str:
    return (
        f"{metric['absolute_return_pct']:.2f}/{metric['cagr_pct']:.2f}/"
        f"{metric['strict_mdd_pct']:.2f}/{metric['cagr_to_strict_mdd']:.2f}/"
        f"{metric['trades']}"
    )


def render_docs(report: dict[str, Any]) -> str:
    selected = report["frozen_pre2025_top1"]
    baseline = report["baseline_live_replay"]["corrected_strict"]
    lines = [
        "# Added-alpha portfolio allocation update (2026-07-16)",
        "",
        "Metric cells: `absolute return / full-calendar CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        "## Frozen protocol",
        "",
        f"- Gross <= {report['config']['gross_cap']}; family gross <= {report['config']['family_gross_cap']}.",
        f"- Non-zero weight >= {report['config']['min_nonzero_weight']}; step = {report['config']['weight_step']}.",
        "- Allocation ranking uses train and 2024 only.",
        "- Two deterministic seed pools plus exact 0.05-grid beam refinement (3 stalled rounds patience) are ranked on the shared 5-minute clock; there is no daily shortlist.",
        "- Exact score ties prefer lower gross, then lexicographically lower sleeve weights.",
        "- 2025 and 2026 may veto frozen rank 1, but never rerank or select rank 2+.",
        "- All future windows have prior research exposure; result is shadow-only.",
        "",
        "## Decision",
        "",
        f"- Frozen rank-1 weights: `{selected['weights']}` (gross {selected['gross']:.2f}).",
        f"- Frozen rank-1 future veto: **{'PASS' if selected['future_veto_passed'] else 'FAIL'}**.",
        f"- Deployment disposition: **{report['deployment_disposition']}**.",
        "",
        "| Portfolio | Train | 2024 selection | 2025 report | 2026H1 report |",
        "|---|---:|---:|---:|---:|",
        f"| Previous live | {_format_metric(baseline['train'])} | {_format_metric(baseline['test2024'])} | {_format_metric(baseline['eval2025'])} | {_format_metric(baseline['ytd2026'])} |",
        f"| Frozen rank 1 | {_format_metric(selected['stats']['train'])} | {_format_metric(selected['stats']['test2024'])} | {_format_metric(selected['stats']['eval2025'])} | {_format_metric(selected['stats']['ytd2026'])} |",
        "",
        "## Top pre-2025 allocation ranks",
        "",
        "| # | Gross | Weights | Train | 2024 | 2025 report | 2026H1 report | Future veto |",
        "|---:|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for index, row in enumerate(report["top_pre2025"][:20], start=1):
        stats = row["stats"]
        lines.append(
            f"| {index} | {row['gross']:.2f} | `{row['weights']}` | "
            f"{_format_metric(stats['train'])} | {_format_metric(stats['test2024'])} | "
            f"{_format_metric(stats['eval2025'])} | {_format_metric(stats['ytd2026'])} | "
            f"{'PASS' if row['future_veto_passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Candidate and accounting notes",
        "",
        "- The old live row is reproduced exactly under its legacy MDD engine before comparison.",
        "- Selection uses the corrected same-bar upper-before-lower strict MDD clock.",
        "- OHLC favorable and adverse envelopes are both retained; upper is applied before lower on each bar.",
        "- The reported row is the best found in a deterministic seeded candidate search, not a proof of the global discrete-grid optimum.",
        "- Rank7 and Fresh Kimchi retain their canonical execution/funding schedules.",
        "- Advanced-state representatives selected by inspecting future passers were excluded.",
        "- This experiment does not overwrite the current live config.",
    ]
    return "\n".join(lines) + "\n"


def run(cfg: Config) -> dict[str, Any]:
    ensure_runtime_inputs()
    input_provenance = {
        "market": file_record(cfg.input_csv),
        "funding": file_record(cfg.funding_csv),
        "premium": file_record(cfg.premium_csv),
        "open_interest_cache": file_record("/tmp/btcusdt_open_interest_5m_2020_2026.csv"),
        "live_anchor": file_record(
            "configs/live/portfolio_gross385_trainmdd40_2026-07-12.json"
        ),
        "markov_candidate": file_record(
            "research/pools/alphas/markov_persistent_funding_premium_long_20260712.json"
        ),
        "funding_lr_candidate": file_record(
            "configs/live/funding_premium_lr_impact_central_research_candidate.json"
        ),
        "funding_lr_manifest": file_record(
            "results/funding_premium_independent_gate_top10_manifest_2026-07-13.json"
        ),
        "rex_reasoning_source": file_record("data/rex_event_reasoning_policy_sft_20260712.jsonl"),
    }
    legacy_cfg = legacy_all.Config(
        random_samples=0,
        candidate_rex_top_n=50,
        train_mdd_cap=cfg.train_mdd_cap,
        oos_mdd_cap=cfg.future_mdd_cap,
        gross_cap=cfg.gross_cap,
        min_nonzero_weight=cfg.min_nonzero_weight,
        weight_step=cfg.weight_step,
        cost_rate=cfg.cost_rate,
    )
    market, _, masks, _, events, _ = legacy_base.build_combined_events(legacy_cfg)
    legacy_all.add_rex_veto_candidates(events, market, masks, legacy_cfg)
    events = [event for event in events if event["sleeve"] in LIVE_WEIGHTS]
    attach_live_rex_favorable(events, market, masks, cfg)
    attach_default_favorable(events, market)
    features = feature_frame(market)
    markov = markov_active(market, features)
    markov_counts = append_mask_policy(
        events,
        market,
        masks,
        name="markov_transition_long",
        long_active=markov,
        short_active=np.zeros(len(market), dtype=bool),
        hold=576,
        stride=12,
        cost_rate=cfg.cost_rate,
    )
    funding_active, funding_meta = funding_lr_active(market)
    funding_counts = append_mask_policy(
        events,
        market,
        masks,
        name="funding_premium_lr_impact_central",
        long_active=funding_active,
        short_active=np.zeros(len(market), dtype=bool),
        hold=576,
        stride=12,
        cost_rate=cfg.cost_rate,
    )
    rex_counts = append_rex_taker_policy(events, market, masks, cost_rate=cfg.cost_rate)
    path_counts, path_meta = append_rank7_and_fresh(events, cfg)
    arrays = split_arrays(events, market, masks)
    del events

    legacy_replay = {
        split: legacy_metric(arrays[split], years_for(split), LIVE_WEIGHTS) for split in SPLIT_BOUNDS
    }
    for split, (expected_return, expected_mdd, expected_trades) in EXPECTED_LIVE_LEGACY.items():
        actual = legacy_replay[split]
        if not np.isclose(actual["absolute_return_pct"], expected_return, rtol=0.0, atol=1e-9):
            raise RuntimeError(f"live baseline return drifted in {split}")
        if not np.isclose(actual["strict_mdd_pct"], expected_mdd, rtol=0.0, atol=1e-9):
            raise RuntimeError(f"live baseline MDD drifted in {split}")
        if actual["trades"] != expected_trades:
            raise RuntimeError(f"live baseline trade count drifted in {split}")
    strict_baseline = {
        split: strict_metric(arrays[split], years_for(split), LIVE_WEIGHTS) for split in SPLIT_BOUNDS
    }

    candidates = weight_candidates(cfg)
    ranked_pre2025 = exact_pre2025_rows(arrays, candidates, cfg)
    if not ranked_pre2025:
        raise RuntimeError("no exact allocation passed the frozen pre-2025 constraints")
    ranked_pre2025, refinement_meta = refine_pre2025_rows(
        arrays, candidates, ranked_pre2025, cfg
    )
    rows = ranked_pre2025[:100]
    # Freeze ordering first. Future windows are read only after this list is fixed.
    frozen_weight_order = [tuple(row["weights"].get(name, 0.0) for name in SLEEVES) for row in rows]
    for row in rows:
        weights = row["weights"]
        row["stats"] = {
            split: strict_metric(arrays[split], years_for(split), weights)
            for split in SPLIT_BOUNDS
        }
        row["future_veto_passed"] = forward_veto(row["stats"], cfg)
        row.pop("selection_key", None)
    if frozen_weight_order != [
        tuple(row["weights"].get(name, 0.0) for name in SLEEVES) for row in rows
    ]:
        raise RuntimeError("future diagnostics changed the frozen pre-2025 ordering")
    selected = rows[0]
    uses_added_alpha = any(name in selected["weights"] for name in NEW_SLEEVES)
    disposition = (
        "forward_shadow_candidate_not_live"
        if uses_added_alpha and selected["future_veto_passed"]
        else "retain_existing_live_portfolio"
    )
    source_validation = {
        "markov_counts": markov_counts,
        "funding_lr_counts": funding_counts,
        "rex_taker_counts": rex_counts,
        "path_counts": path_counts,
        "path_final_equities": path_meta,
        "funding_lr_manifest": funding_meta,
    }
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "mode": "frozen_pre2025_allocation_rank_future_veto_only",
        "config": asdict(cfg),
        "protocol_hash": json_hash(
            {
                "sleeves": SLEEVES,
                "families": FAMILIES,
                "splits": SPLIT_BOUNDS,
                "constraints": {
                    key: value
                    for key, value in asdict(cfg).items()
                    if key
                    in {
                        "gross_cap",
                        "family_gross_cap",
                        "min_nonzero_weight",
                        "weight_step",
                        "train_mdd_cap",
                        "test_mdd_cap",
                        "future_mdd_cap",
                        "min_test_trades",
                        "min_test_ratio",
                        "min_future_ratio",
                        "random_samples",
                        "seed",
                        "seed_count",
                        "refinement_rounds",
                        "refinement_top_n",
                        "refinement_patience",
                        "cost_rate",
                    }
                },
                "input_sha256": {
                    name: record["sha256"] for name, record in input_provenance.items()
                },
                "selection": (
                    "train+test2024 only; exact multi-seed beam refinement; "
                    "tie=lower gross then lexicographic weights; future veto cannot rerank"
                ),
            }
        ),
        "future_used_for_allocation_ranking": False,
        "future_can_only_veto_frozen_rank1": True,
        "future_is_pristine_discovery_oos": False,
        "contamination_caveat": (
            "Every reported future window has prior research exposure. The allocation ranking adds no "
            "new future reranking, but the alpha universe itself remains research-contaminated."
        ),
        "candidate_universe": {
            "live_anchor": list(LIVE_WEIGHTS),
            "added": list(NEW_SLEEVES),
            "excluded": {
                "kalman_bocpd_semimarkov_representatives": (
                    "exact representative chosen from pre-evaluation Top-10 after observing future passers"
                ),
                "funding_premium_alt_outer": "failed its own 2026 and combined live-grade ratios",
                "cross_collateral_pressure": "rejected on frozen OOS",
            },
        },
        "input_provenance": input_provenance,
        "baseline_live_weights": LIVE_WEIGHTS,
        "baseline_live_replay": {
            "legacy_exact_reproduction": legacy_replay,
            "corrected_strict": strict_baseline,
        },
        "candidate_generation": {
            "generated_initial": len(candidates),
            "seed_range": [cfg.seed, cfg.seed + cfg.seed_count - 1],
            "exact_pre2025_evaluated": len(candidates) + refinement_meta["evaluated"],
            "exact_pre2025_passed": len(ranked_pre2025),
            "refinement": refinement_meta,
            "search_scope": "deterministic seeded/random candidate set; not a proof of global grid optimum",
            "selection_clock": "shared 5-minute OHLC upper-before-lower strict MDD",
        },
        "source_validation": source_validation,
        "frozen_pre2025_top1": selected,
        "top_pre2025": rows[:100],
        "deployment_disposition": disposition,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    docs = Path(cfg.docs_output)
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(render_docs(report))
    candidate_config = {
        "name": "portfolio_added_alpha_shadow_candidate_2026_07_16",
        "status": disposition,
        "as_of": "2026-07-16",
        "weights": selected["weights"],
        "gross_weight": selected["gross"],
        "selection": "frozen train+2024 rank 1; 2025/2026 veto only; no reranking",
        "future_veto_passed": selected["future_veto_passed"],
        "research_contaminated": True,
        "source_result": cfg.output,
        "protocol_hash": report["protocol_hash"],
    }
    candidate_path = Path(cfg.candidate_config)
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps(candidate_config, indent=2, ensure_ascii=False) + "\n")
    return report


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in Config.__dataclass_fields__.values():
        name = "--" + field.name.replace("_", "-")
        if field.type in (int, "int"):
            parser.add_argument(name, type=int, default=field.default)
        elif field.type in (float, "float"):
            parser.add_argument(name, type=float, default=field.default)
        else:
            parser.add_argument(name, default=field.default)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "output": report["config"]["output"],
                "deployment_disposition": report["deployment_disposition"],
                "frozen_pre2025_top1": report["frozen_pre2025_top1"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
