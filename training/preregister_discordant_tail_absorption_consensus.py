"""Outcome-blind preregistration for DTAC-8 discordant-tail consensus clocks.

DTAC-8 asks whether sign-conditional tail disagreement between aggressive
six-alt taker flow and same-hour USD-M premium-index impulse identifies passive
inventory absorption that subsequently relays into BTC. This module reads no
BTC execution price, BTC funding, post-entry return, PnL, or equity.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.build_six_alt_price_free_flow_panel import (  # noqa: E402
    OUTPUT_COLUMNS as FLOW_SCHEMA,
    SYMBOLS,
    deterministic_gzip_csv,
    sha256_file,
)


POLICY_ID = "DTAC-8"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_discordant_tail_absorption_consensus.py"
)
FLOW_PANEL = Path(
    "data/binance_six_alt_price_free_flow_2023_2026/"
    "six_alt_price_free_flow_1h_2023-01-01_2026-06-01.csv.gz"
)
FLOW_MANIFEST = Path(
    "data/binance_six_alt_price_free_flow_2023_2026/build_manifest.json"
)
FLOW_PANEL_SHA256 = "bf4d67ee02948444712a6ff7862a0d4f4ae4ae2a704c9d0586538043c169f6b9"
FLOW_MANIFEST_SHA256 = (
    "eab61cbc7f5fc51e78f574e8bef163b3a3b91bd027136cae8efd7aaf26edc0f1"
)
PREMIUM_DIR = Path("data/binance_um_aux_2023_2026")
PREMIUM_SUMMARY = PREMIUM_DIR / "download_summary_aux_2023-01-01_2026-06-01.json"
PREMIUM_SUMMARY_SHA256 = (
    "83026318c3b3caeea08a1528224d819c08cf4a76d8e3b58d18f308f92d5db0b4"
)
PREMIUM_SCHEMA = ("date", "symbol", "open", "high", "low", "close", "close_time")
PREMIUM_SHA256 = {
    "ADAUSDT": "fcdb8d4a8733e677740e2996aaffe665f129601c0228fef873794646bb994a28",
    "BNBUSDT": "0d86c250f39bf38379622787e46ec06a23e350216221ecc71339c30c1c52a17d",
    "DOGEUSDT": "9aceab02b1a3d25ba69acee09eba3e473ddd769b7795cc3a399928890de2b5d2",
    "ETHUSDT": "88eb7d2e7413bf7066371c0f7ed4f8b549b1089b8339a29fdae3344bcfd2b8df",
    "SOLUSDT": "72c2a0bb8d2db31ece6f41d636d262d52949711ee921a37c3f270b05153aaf54",
    "XRPUSDT": "1c62ffcbdf34f2922eaf54c8e955b2513361f9080be4dc9ae94e02304ab48bae",
}

SELECTION_END = cast(pd.Timestamp, pd.Timestamp("2024-01-01"))
SOURCE_END = cast(pd.Timestamp, pd.Timestamp("2026-06-01"))
SPLITS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (cast(pd.Timestamp, pd.Timestamp("2023-01-01")), SELECTION_END),
    "test": (
        cast(pd.Timestamp, pd.Timestamp("2024-01-01")),
        cast(pd.Timestamp, pd.Timestamp("2025-01-01")),
    ),
    "eval": (
        cast(pd.Timestamp, pd.Timestamp("2025-01-01")),
        cast(pd.Timestamp, pd.Timestamp("2026-01-01")),
    ),
    "final": (cast(pd.Timestamp, pd.Timestamp("2026-01-01")), SOURCE_END),
}

COMPARATORS = {
    "FCIR-12": {
        "path": "data/flow_centrality_incubation_relay_clocks_2023_2026.csv.gz",
        "sha256": "d4bb6245f0bac34885e780e35ff1edb9b5cf2114dc3c13088ec19613ad8056ea",
        "kind": "plain_csv",
    },
    "SQFD-6": {
        "path": "data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz",
        "sha256": "a81e144eea1e80ae5439fc66db1fad5bbd00cd9ac177e25142b5cfb5a07bcc5b",
        "kind": "primary_csv",
    },
    "OPDR-24": {
        "path": "data/options_perpetual_demand_relay_clocks_2023_2026.csv.gz",
        "sha256": "ceb79b206c3e1f6bf78b02cd2ace9a94f875ce930a704cc6e7a5a8b255021b99",
        "kind": "plain_csv",
    },
    "PCBR-12": {
        "path": "data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz",
        "sha256": "659fc1b6b6e3a20e60031ed1d50f51c8c7d2836956f911f62ad13e4152740cda",
        "kind": "plain_csv",
    },
    "PSR-30/6": {
        "path": "data/premium_snapback_recenter_clocks_2020_2026.csv.gz",
        "sha256": "cb209ed35f9baa08cc2fb3dd5bd60b8e747b1408c09507b774ca275e0b2b2db6",
        "kind": "plain_csv",
    },
    "TGR-12": {
        "path": "data/ticket_gap_release_clocks_2023_2026.csv.gz",
        "sha256": "166c6e214f43d19eb0c33adbc7deed5fd81eeee222de6a737daf061fd2c3ffc2",
        "kind": "plain_csv",
    },
}

FLOW_COLUMNS = (
    "feature_available_time_utc",
    "symbol",
    "taker_flow_fraction",
    "feature_valid",
)
EVENT_COLUMNS = (
    "candidate",
    "split",
    "source_hour_open_utc",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "long_votes",
    "short_votes",
    "long_vote_symbols",
    "short_vote_symbols",
    "consensus_count",
    "flow_tail_quantile",
    "premium_tail_quantile",
    "mean_vote_flow",
    "mean_vote_premium_impulse",
)
FORBIDDEN_SELECTION_TOKENS = (
    "return",
    "pnl",
    "profit",
    "loss",
    "cagr",
    "mdd",
    "drawdown",
    "equity",
    "sharpe",
    "hit_rate",
    "excursion",
    "future_price",
)


@dataclass(frozen=True)
class Config:
    result_output: str = (
        "results/discordant_tail_absorption_consensus_support_2026-07-19.json"
    )
    clock_output: str = (
        "data/discordant_tail_absorption_consensus_clocks_2023_2026.csv.gz"
    )
    docs_output: str = (
        "docs/discordant-tail-absorption-consensus-preregistration-2026-07-19.md"
    )
    threshold_window_hours: int = 2160
    threshold_minimum_sign_observations: int = 360
    flow_tail_quantiles: tuple[float, ...] = (0.65, 0.70, 0.75, 0.80, 0.85)
    premium_tail_quantiles: tuple[float, ...] = (0.50, 0.60, 0.70, 0.75, 0.80)
    consensus_counts: tuple[int, ...] = (2, 3, 4)
    maximum_opposite_votes: int = 1
    entry_delay_minutes: int = 5
    hold_hours: int = 8
    minimum_train_events: int = 100
    minimum_train_half_events: int = 45
    minimum_active_quarter_events: int = 30
    minimum_train_side_share: float = 0.40
    minimum_stage_side_share: float = 0.35
    maximum_train_month_share: float = 0.20
    minimum_stage_events: tuple[tuple[str, int], ...] = (
        ("train", 100),
        ("test", 100),
        ("eval", 100),
        ("final", 50),
    )
    novelty_tolerance_hours: int = 2
    maximum_exact_jaccard: float = 0.05
    maximum_near_share: float = 0.35


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def premium_path(symbol: str) -> Path:
    return PREMIUM_DIR / f"{symbol}_premium_1h_2023-01-01_2026-06-01.csv.gz"


def protocol() -> dict[str, Any]:
    cfg = Config()
    return {
        "policy_id": POLICY_ID,
        "hypothesis": (
            "same-hour sign-conditional tail disagreement between aggressive alt "
            "taker flow and USD-M premium impulse marks passive absorption; a "
            "cross-sectional consensus relays into BTC in the premium direction"
        ),
        "evidence_boundary": {
            "allowed": [
                "frozen completed-hour six-alt normalized taker flow",
                "frozen completed-hour six-alt premium-index open and close",
                "strictly-prior sign-conditional rolling quantiles",
                "source-only event incidence, sides, concentration, and overlap",
            ],
            "forbidden": [
                "BTC execution OHLC or funding",
                "entry or later prices",
                "post-entry return or excursion",
                "PnL, equity, CAGR, MDD, hit rate, or payoff",
            ],
            "post_entry_outcomes_opened": False,
        },
        "tail_calibration": {
            "history": (
                f"prior {cfg.threshold_window_hours} completed hours with at least "
                f"{cfg.threshold_minimum_sign_observations} observations per sign"
            ),
            "current_hour_excluded": True,
            "positive_and_negative_tails_calibrated_separately": True,
            "reason": (
                "prevent raw distribution skew from manufacturing a directional bias"
            ),
        },
        "signal": {
            "long_vote": (
                "same symbol has negative-flow magnitude in its prior negative tail "
                "and positive premium impulse in its prior positive tail"
            ),
            "short_vote": (
                "same symbol has positive flow in its prior positive tail and "
                "negative-premium magnitude in its prior negative tail"
            ),
            "long_side": (
                "at least selected consensus long votes and no more than one short vote"
            ),
            "short_side": (
                "at least selected consensus short votes and no more than one long vote"
            ),
            "side": "premium impulse direction, opposite the absorbed taker-flow tail",
            "trigger": "zero-to-directional state onset or direct polarity change",
        },
        "selection": {
            "source_window": "2023 only",
            "outcomes_used": False,
            "grid": {
                "flow_tail_quantiles": list(cfg.flow_tail_quantiles),
                "premium_tail_quantiles": list(cfg.premium_tail_quantiles),
                "consensus_counts": list(cfg.consensus_counts),
            },
            "rule": (
                "among 2023 source-support-passing cells maximize consensus count, "
                "then flow-tail quantile, then premium-tail quantile"
            ),
            "future_source_incidence": (
                "opened only after selection and used solely as a feasibility veto"
            ),
            "honest_design_history": (
                "a shared unsigned threshold prototype was source-only rejected for "
                "direction bias before this sign-conditional family was frozen"
            ),
        },
        "clock": {
            "decision": "right edge of the completed UTC source hour",
            "entry": "decision + 5 minutes",
            "exit": "entry + 8 hours",
            "position_state": "one BTC position maximum per split",
        },
        "eventual_execution": {
            "instrument": "Binance BTCUSDT USD-M perpetual",
            "leverage": 0.5,
            "cost_bp_per_notional_side": 6.0,
            "funding": (
                "interior exact-time events symmetric; exact entry/exit credits "
                "dropped and debits retained; every settlement mark visited"
            ),
            "strict_mdd": (
                "global pre-entry HWM, costs, funding marks, every held five-minute "
                "favorable-then-adverse path, and virtual adverse-mark exit cost"
            ),
            "full_calendar_cagr": True,
            "controls": [
                "direction flip on identical clocks",
                "all-six premium-impulse side on identical clocks",
                "all-six flow-fade side on identical clocks",
                "deterministic symbol-permuted premium pairing",
                "24-hour stale premium pairing",
                "deterministic random side on identical clocks",
                "one-hour additional latency",
                "10 bp per-notional-side cost stress",
            ],
        },
        "outcome_gate": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "minimum_trades": dict(cfg.minimum_stage_events),
            "weekly_cluster_signflip_p_max": 0.10,
            "mean_gross_underlying_move_bp_min": 20.0,
            "each_contained_half_absolute_return_positive": True,
            "stress_cost_notional_per_side": 0.001,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "mechanism_control_margin_min": 0.25,
            "sequential_opening": (
                "train then test then eval then final; stop on first failed gate"
            ),
            "ratio_definition": (
                "CAGR_pct / strict_MDD_pct; exact zero-MDD uses signed infinity or zero"
            ),
            "statistical_test": {
                "cluster_key": "UTC entry timestamp ISO year/week",
                "cluster_value": (
                    "sum of net account trade returns after costs and funding"
                ),
                "exact_cluster_max": 20,
                "monte_carlo_draws": 20_000,
                "seed": 20_260_719,
            },
        },
    }


def _validate_config(cfg: Config) -> None:
    if cfg != Config(
        result_output=cfg.result_output,
        clock_output=cfg.clock_output,
        docs_output=cfg.docs_output,
    ):
        raise ValueError("DTAC-8 source signal and support configuration is frozen")


def _validate_source_artifacts() -> dict[str, Any]:
    if sha256_file(FLOW_PANEL) != FLOW_PANEL_SHA256:
        raise RuntimeError("DTAC-8 flow panel hash changed")
    if sha256_file(FLOW_MANIFEST) != FLOW_MANIFEST_SHA256:
        raise RuntimeError("DTAC-8 flow manifest hash changed")
    flow_manifest = json.loads(FLOW_MANIFEST.read_text())
    source_protocol = flow_manifest["protocol"]
    if source_protocol["post_entry_outcomes_opened"] is not False:
        raise RuntimeError("DTAC-8 flow source opened outcomes")
    if source_protocol["price_values_read"] is not False:
        raise RuntimeError("DTAC-8 flow source read price values")
    if flow_manifest["combined_sha256"] != FLOW_PANEL_SHA256:
        raise RuntimeError("DTAC-8 flow panel and manifest disagree")
    if sha256_file(PREMIUM_SUMMARY) != PREMIUM_SUMMARY_SHA256:
        raise RuntimeError("DTAC-8 premium summary hash changed")
    summary = json.loads(PREMIUM_SUMMARY.read_text())
    if set(summary["symbols"]) != set(SYMBOLS) or "premium" not in summary["kinds"]:
        raise RuntimeError("DTAC-8 premium source universe changed")
    for symbol, expected in PREMIUM_SHA256.items():
        if sha256_file(premium_path(symbol)) != expected:
            raise RuntimeError(f"DTAC-8 premium source hash changed: {symbol}")
    return {"flow": flow_manifest, "premium": summary}


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid DTAC-8 source boolean: {value}")
    return normalized == "true"


def load_flow_prefix(*, end_exclusive: pd.Timestamp | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with gzip.open(FLOW_PANEL, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FLOW_SCHEMA:
            raise RuntimeError("DTAC-8 flow source schema changed")
        for raw in reader:
            timestamp = cast(
                pd.Timestamp, pd.Timestamp(raw["feature_available_time_utc"])
            )
            if end_exclusive is not None and timestamp >= end_exclusive:
                break
            rows.append(
                {
                    "feature_available_time_utc": timestamp,
                    "symbol": raw["symbol"],
                    "taker_flow_fraction": (
                        float(raw["taker_flow_fraction"])
                        if raw["taker_flow_fraction"]
                        else np.nan
                    ),
                    "feature_valid": _parse_bool(raw["feature_valid"]),
                }
            )
    frame = pd.DataFrame(rows, columns=pd.Index(FLOW_COLUMNS))
    if frame.empty or set(frame["symbol"]) != set(SYMBOLS):
        raise RuntimeError("DTAC-8 flow prefix is empty or incomplete")
    return frame


def load_premium_prefix(*, end_exclusive: pd.Timestamp | None) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for symbol in sorted(SYMBOLS):
        rows: list[tuple[pd.Timestamp, float]] = []
        with gzip.open(
            premium_path(symbol), "rt", encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PREMIUM_SCHEMA:
                raise RuntimeError(f"DTAC-8 premium schema changed: {symbol}")
            for raw in reader:
                available = cast(
                    pd.Timestamp,
                    pd.to_datetime(int(raw["close_time"]) + 1, unit="ms"),
                )
                if end_exclusive is not None and available >= end_exclusive:
                    break
                if raw["symbol"] != symbol:
                    raise RuntimeError(f"DTAC-8 premium symbol changed: {symbol}")
                rows.append((available, float(raw["close"]) - float(raw["open"])))
        if not rows:
            raise RuntimeError(f"DTAC-8 premium prefix is empty: {symbol}")
        times = pd.DatetimeIndex([timestamp for timestamp, _ in rows])
        values = np.asarray([value for _, value in rows], dtype=float)
        if not times.is_monotonic_increasing or not times.is_unique:
            raise RuntimeError(f"DTAC-8 premium time grid changed: {symbol}")
        on_hour = (times.astype("int64").to_numpy() % pd.Timedelta(hours=1).value) == 0
        if not bool(on_hour.all()) or not np.isfinite(values).all():
            raise RuntimeError(f"DTAC-8 premium values are invalid: {symbol}")
        series[symbol] = pd.Series(
            values,
            index=times,
            name=symbol,
        )
    return pd.DataFrame(series).sort_index().reindex(columns=pd.Index(sorted(SYMBOLS)))


def flow_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    valid = cast(pd.DataFrame, frame.loc[frame["feature_valid"].eq(True)].copy())
    matrix = valid.pivot(
        index="feature_available_time_utc",
        columns="symbol",
        values="taker_flow_fraction",
    ).reindex(columns=pd.Index(sorted(SYMBOLS)))
    expected = pd.date_range(
        cast(pd.Timestamp, frame["feature_available_time_utc"].min()),
        cast(pd.Timestamp, frame["feature_available_time_utc"].max()),
        freq="1h",
    )
    return matrix.reindex(expected)


def _sign_tail_thresholds(
    frame: pd.DataFrame,
    quantile: float,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive = frame.where(frame.gt(0.0))
    negative_magnitude = (-frame).where(frame.lt(0.0))
    positive_threshold = (
        positive.rolling(
            cfg.threshold_window_hours,
            min_periods=cfg.threshold_minimum_sign_observations,
        )
        .quantile(quantile)
        .shift(1)
    )
    negative_threshold = (
        negative_magnitude.rolling(
            cfg.threshold_window_hours,
            min_periods=cfg.threshold_minimum_sign_observations,
        )
        .quantile(quantile)
        .shift(1)
    )
    return cast(pd.DataFrame, positive_threshold), cast(
        pd.DataFrame, negative_threshold
    )


def feature_panel(
    flow: pd.DataFrame,
    premium: pd.DataFrame,
    *,
    flow_tail_quantile: float,
    premium_tail_quantile: float,
    consensus_count: int,
    cfg: Config,
) -> pd.DataFrame:
    index = flow.index.intersection(premium.index)
    aligned_flow = flow.reindex(index)
    aligned_premium = premium.reindex(index)
    flow_positive, flow_negative = _sign_tail_thresholds(
        aligned_flow, flow_tail_quantile, cfg
    )
    premium_positive, premium_negative = _sign_tail_thresholds(
        aligned_premium, premium_tail_quantile, cfg
    )
    long_vote = (-aligned_flow).ge(flow_negative) & aligned_premium.ge(premium_positive)
    short_vote = aligned_flow.ge(flow_positive) & (-aligned_premium).ge(
        premium_negative
    )
    long_votes = long_vote.sum(axis=1)
    short_votes = short_vote.sum(axis=1)
    long_state = long_votes.ge(consensus_count) & short_votes.le(
        cfg.maximum_opposite_votes
    )
    short_state = short_votes.ge(consensus_count) & long_votes.le(
        cfg.maximum_opposite_votes
    )
    side = pd.Series(
        np.where(
            long_state & ~short_state,
            1,
            np.where(short_state & ~long_state, -1, 0),
        ),
        index=index,
    )
    selected_vote = long_vote.where(side.eq(1), short_vote.where(side.eq(-1), False))
    panel = pd.DataFrame(index=index)
    panel["side"] = side
    panel["long_votes"] = long_votes
    panel["short_votes"] = short_votes
    panel["long_vote_symbols"] = long_vote.apply(
        lambda row: ";".join(str(symbol) for symbol in row.index[row]), axis=1
    )
    panel["short_vote_symbols"] = short_vote.apply(
        lambda row: ";".join(str(symbol) for symbol in row.index[row]), axis=1
    )
    panel["mean_vote_flow"] = aligned_flow.where(selected_vote).mean(axis=1)
    panel["mean_vote_premium_impulse"] = aligned_premium.where(selected_vote).mean(
        axis=1
    )
    panel["consensus_count"] = consensus_count
    panel["flow_tail_quantile"] = flow_tail_quantile
    panel["premium_tail_quantile"] = premium_tail_quantile
    return panel


def signal_onset(features: pd.DataFrame) -> pd.Series:
    side = cast(pd.Series, features["side"])
    return side.ne(0) & side.ne(side.shift(1, fill_value=0))


def schedule_events(features: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    onset = signal_onset(features)
    records: list[dict[str, Any]] = []
    for split, (start, end) in SPLITS.items():
        reserved_until: pd.Timestamp | None = None
        times = cast(pd.DatetimeIndex, features.index[onset]).sort_values()
        for decision in times[(times >= start) & (times < end)]:
            entry = cast(
                pd.Timestamp,
                decision + pd.Timedelta(minutes=cfg.entry_delay_minutes),
            )
            exit_time = cast(pd.Timestamp, entry + pd.Timedelta(hours=cfg.hold_hours))
            if exit_time > end:
                continue
            if reserved_until is not None and entry < reserved_until:
                continue
            row = features.loc[decision]
            records.append(
                {
                    "candidate": POLICY_ID,
                    "split": split,
                    "source_hour_open_utc": decision - pd.Timedelta(hours=1),
                    "decision_time": decision,
                    "feature_available_time": decision,
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "side": int(row["side"]),
                    "long_votes": int(row["long_votes"]),
                    "short_votes": int(row["short_votes"]),
                    "long_vote_symbols": str(row["long_vote_symbols"]),
                    "short_vote_symbols": str(row["short_vote_symbols"]),
                    "consensus_count": int(row["consensus_count"]),
                    "flow_tail_quantile": float(row["flow_tail_quantile"]),
                    "premium_tail_quantile": float(row["premium_tail_quantile"]),
                    "mean_vote_flow": float(row["mean_vote_flow"]),
                    "mean_vote_premium_impulse": float(
                        row["mean_vote_premium_impulse"]
                    ),
                }
            )
            reserved_until = exit_time
    return pd.DataFrame(records, columns=pd.Index(EVENT_COLUMNS))


def support_summary(events: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = cast(pd.DataFrame, events.loc[events["split"].eq(split)].copy())
    selected["month"] = selected["entry_time"].dt.to_period("M").astype(str)
    selected["quarter"] = selected["entry_time"].dt.to_period("Q").astype(str)
    month_counts = selected["month"].value_counts().sort_index()
    quarter_counts = selected["quarter"].value_counts().sort_index()
    start, end = SPLITS[split]
    midpoint = start + (end - start) / 2
    left = int(selected["entry_time"].lt(midpoint).sum())
    right = int(selected["entry_time"].ge(midpoint).sum())
    events_count = len(selected)
    long_count = int(selected["side"].eq(1).sum())
    short_count = int(selected["side"].eq(-1).sum())
    return {
        "events": events_count,
        "long": long_count,
        "short": short_count,
        "side_share_min": (
            min(long_count, short_count) / events_count if events_count else 0.0
        ),
        "maximum_month_share": (
            float(month_counts.max() / events_count) if events_count else 1.0
        ),
        "month_counts": {str(key): int(value) for key, value in month_counts.items()},
        "quarter_counts": {
            str(key): int(value) for key, value in quarter_counts.items()
        },
        "subwindows": {f"{split}_h1": left, f"{split}_h2": right},
    }


def train_support_checks(summary: dict[str, Any], cfg: Config) -> dict[str, bool]:
    quarters = summary["quarter_counts"]
    return {
        "train_events": summary["events"] >= cfg.minimum_train_events,
        "train_side_balance": summary["side_share_min"] >= cfg.minimum_train_side_share,
        "train_half_coverage": min(summary["subwindows"].values())
        >= cfg.minimum_train_half_events,
        "train_active_quarter_coverage": all(
            int(quarters.get(quarter, 0)) >= cfg.minimum_active_quarter_events
            for quarter in ("2023Q2", "2023Q3", "2023Q4")
        ),
        "train_month_concentration": summary["maximum_month_share"]
        <= cfg.maximum_train_month_share,
    }


def _reject_outcome_fields(payload: Any, path: str = "root") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_SELECTION_TOKENS):
                raise ValueError(
                    f"forbidden outcome field in DTAC support selection: {path}.{key}"
                )
            _reject_outcome_fields(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _reject_outcome_fields(value, f"{path}[{index}]")


def select_support_cell(cells: list[dict[str, Any]]) -> dict[str, Any]:
    _reject_outcome_fields(cells)
    passing = [cell for cell in cells if cell["passes"]]
    if not passing:
        raise RuntimeError("no DTAC-8 source-support cell passed")
    return max(
        passing,
        key=lambda cell: (
            int(cell["consensus_count"]),
            float(cell["flow_tail_quantile"]),
            float(cell["premium_tail_quantile"]),
        ),
    )


def _load_comparator_entries() -> dict[str, pd.DatetimeIndex]:
    entries: dict[str, pd.DatetimeIndex] = {}
    for name, spec in COMPARATORS.items():
        path = Path(spec["path"])
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"DTAC-8 comparator hash changed: {name}")
        if spec["kind"] == "primary_csv":
            frame = cast(
                pd.DataFrame,
                pd.read_csv(path, usecols=cast(Any, ["control", "entry_time"])),
            )
            raw = frame.loc[frame["control"].eq("primary"), "entry_time"]
        elif spec["kind"] == "plain_csv":
            frame = cast(
                pd.DataFrame,
                pd.read_csv(path, usecols=cast(Any, ["entry_time"])),
            )
            raw = frame["entry_time"]
        else:
            raise RuntimeError(f"unknown DTAC-8 comparator kind: {spec['kind']}")
        values = pd.to_datetime(raw, utc=True).dt.tz_localize(None)
        entries[name] = pd.DatetimeIndex(values).sort_values()
    return entries


def _near_share(
    source: pd.DatetimeIndex,
    target: pd.DatetimeIndex,
    tolerance: pd.Timedelta,
) -> float:
    if len(source) == 0 or len(target) == 0:
        return 0.0
    source_ns = source.astype("int64").to_numpy()
    target_ns = target.astype("int64").to_numpy()
    positions = np.searchsorted(target_ns, source_ns)
    matched = np.zeros(len(source_ns), dtype=bool)
    for offset in (0, -1):
        candidate = positions + offset
        valid = (candidate >= 0) & (candidate < len(target_ns))
        matched[valid] |= (
            np.abs(target_ns[candidate[valid]] - source_ns[valid]) <= tolerance.value
        )
    return float(matched.mean())


def novelty_metrics(
    new_entries: pd.DatetimeIndex,
    prior_entries: pd.DatetimeIndex,
    *,
    tolerance: pd.Timedelta,
) -> dict[str, Any]:
    coverage_start = max(
        cast(pd.Timestamp, new_entries.min()),
        cast(pd.Timestamp, prior_entries.min()),
    )
    coverage_end = min(
        cast(pd.Timestamp, new_entries.max()),
        cast(pd.Timestamp, prior_entries.max()),
    )
    if coverage_start > coverage_end:
        return {
            "shared_coverage_start": None,
            "shared_coverage_end": None,
            "new_entries": 0,
            "prior_entries": 0,
            "exact_jaccard": 0.0,
            "new_near_prior_share": 0.0,
            "prior_near_new_share": 0.0,
            "max_bidirectional_near_share": 0.0,
        }
    new = new_entries[(new_entries >= coverage_start) & (new_entries <= coverage_end)]
    prior = prior_entries[
        (prior_entries >= coverage_start) & (prior_entries <= coverage_end)
    ]
    new_set = set(new.astype("int64"))
    prior_set = set(prior.astype("int64"))
    union = new_set | prior_set
    new_near = _near_share(new, prior, tolerance)
    prior_near = _near_share(prior, new, tolerance)
    return {
        "shared_coverage_start": coverage_start.isoformat(),
        "shared_coverage_end": coverage_end.isoformat(),
        "new_entries": len(new),
        "prior_entries": len(prior),
        "exact_jaccard": len(new_set & prior_set) / len(union) if union else 0.0,
        "new_near_prior_share": new_near,
        "prior_near_new_share": prior_near,
        "max_bidirectional_near_share": max(new_near, prior_near),
    }


def _write_once(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite immutable DTAC artifact: {path}")
        return
    destination.write_bytes(payload)


def _docs(report: dict[str, Any]) -> str:
    selected = report["selected"]
    support = report["support"]
    novelty_lines = "\n".join(
        f"- {name}: exact Jaccard {values['exact_jaccard']:.4f}, "
        f"±2h max near-share {values['max_bidirectional_near_share']:.4f}"
        for name, values in report["novelty"].items()
    )
    return f"""# DTAC-8 source-only preregistration — 2026-07-19

## Status

DTAC-8 passed source-only support, direction-balance, future-incidence
feasibility, and novelty gates. **No BTC execution price, funding, return,
excursion, PnL, equity, CAGR, or MDD was opened.** This permits an evaluator
freeze; it is not profitability evidence.

## Frozen mechanism

- Sources: completed-hour normalized taker flow and premium-index open/close for
  ADA, BNB, DOGE, ETH, SOL, and XRP USD-M perpetuals.
- Tail calibration: positive and negative magnitudes use separate strictly-prior
  2160-hour rolling quantiles, each requiring 360 same-sign observations.
- Long vote: negative-flow tail plus positive premium-impulse tail on the same
  symbol. Short vote is the exact sign mirror.
- Consensus: at least {selected["consensus_count"]} matching votes and at most
  one opposite vote.
- Selected tails: flow q{selected["flow_tail_quantile"] * 100:g}, premium
  q{selected["premium_tail_quantile"] * 100:g}.
- Clock: directional-state onset/polarity change, entry +5m, fixed 8h hold,
  one position.

The selected cell maximized consensus count, then flow-tail strength, then
premium-tail strength among cells passing 2023 source-incidence gates. A naïve
shared unsigned threshold prototype was rejected source-only for direction bias;
it opened no BTC outcome and is not an alternate policy.

## Source-only incidence

| Stage | Events | Long | Short | Max month share |
|---|---:|---:|---:|---:|
| train 2023 | {support["train"]["events"]} | {support["train"]["long"]} | {support["train"]["short"]} | {support["train"]["maximum_month_share"]:.3f} |
| test 2024 | {support["test"]["events"]} | {support["test"]["long"]} | {support["test"]["short"]} | {support["test"]["maximum_month_share"]:.3f} |
| eval 2025 | {support["eval"]["events"]} | {support["eval"]["long"]} | {support["eval"]["short"]} | {support["eval"]["maximum_month_share"]:.3f} |
| final 2026H1 | {support["final"]["events"]} | {support["final"]["long"]} | {support["final"]["short"]} | {support["final"]["maximum_month_share"]:.3f} |

## Clock novelty

{novelty_lines}

## Sequential outcome rule

The strict evaluator and every source-only control must be committed before
2023 BTC execution outcomes are opened. Train must pass all frozen economic,
significance, half-stability, stress, and mechanism-margin gates. Failure
retires the exact policy and keeps 2024+ sealed; no threshold, side, hold, or
control repair is allowed.
"""


def _cell(
    flow: pd.DataFrame,
    premium: pd.DataFrame,
    *,
    flow_tail_quantile: float,
    premium_tail_quantile: float,
    consensus_count: int,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = feature_panel(
        flow,
        premium,
        flow_tail_quantile=flow_tail_quantile,
        premium_tail_quantile=premium_tail_quantile,
        consensus_count=consensus_count,
        cfg=cfg,
    )
    events = schedule_events(features, cfg)
    return events, support_summary(events, "train")


def run(cfg: Config = Config()) -> dict[str, Any]:
    _validate_config(cfg)
    source_manifests = _validate_source_artifacts()

    train_flow = flow_matrix(load_flow_prefix(end_exclusive=SELECTION_END))
    train_premium = load_premium_prefix(end_exclusive=SELECTION_END)
    cells: list[dict[str, Any]] = []
    for flow_tail_quantile in cfg.flow_tail_quantiles:
        for premium_tail_quantile in cfg.premium_tail_quantiles:
            for consensus_count in cfg.consensus_counts:
                _, summary = _cell(
                    train_flow,
                    train_premium,
                    flow_tail_quantile=flow_tail_quantile,
                    premium_tail_quantile=premium_tail_quantile,
                    consensus_count=consensus_count,
                    cfg=cfg,
                )
                checks = train_support_checks(summary, cfg)
                cells.append(
                    {
                        "flow_tail_quantile": flow_tail_quantile,
                        "premium_tail_quantile": premium_tail_quantile,
                        "consensus_count": consensus_count,
                        "train_support": summary,
                        "checks": checks,
                        "passes": all(checks.values()),
                    }
                )
    selected = select_support_cell(cells)

    selected_train_events, _ = _cell(
        train_flow,
        train_premium,
        flow_tail_quantile=float(selected["flow_tail_quantile"]),
        premium_tail_quantile=float(selected["premium_tail_quantile"]),
        consensus_count=int(selected["consensus_count"]),
        cfg=cfg,
    )
    full_flow = flow_matrix(load_flow_prefix(end_exclusive=None))
    full_premium = load_premium_prefix(end_exclusive=None)
    events, _ = _cell(
        full_flow,
        full_premium,
        flow_tail_quantile=float(selected["flow_tail_quantile"]),
        premium_tail_quantile=float(selected["premium_tail_quantile"]),
        consensus_count=int(selected["consensus_count"]),
        cfg=cfg,
    )
    full_train = cast(pd.DataFrame, events.loc[events["split"].eq("train")])
    if not full_train.reset_index(drop=True).equals(
        selected_train_events.reset_index(drop=True)
    ):
        raise RuntimeError("future DTAC source changed train clock semantics")

    support = {split: support_summary(events, split) for split in SPLITS}
    stage_minimums = dict(cfg.minimum_stage_events)
    stage_feasibility_checks = {
        f"{split}_minimum_trade_incidence": support[split]["events"]
        >= int(stage_minimums[split])
        for split in SPLITS
    }
    stage_feasibility_checks.update(
        {
            f"{split}_minimum_side_share": support[split]["side_share_min"]
            >= cfg.minimum_stage_side_share
            for split in SPLITS
        }
    )
    new_entries = pd.DatetimeIndex(events["entry_time"]).sort_values()
    comparator_entries = _load_comparator_entries()
    tolerance = cast(pd.Timedelta, pd.Timedelta(hours=cfg.novelty_tolerance_hours))
    novelty = {
        name: novelty_metrics(new_entries, prior, tolerance=tolerance)
        for name, prior in comparator_entries.items()
    }
    novelty_checks = {
        f"{name}_exact_jaccard": values["exact_jaccard"] <= cfg.maximum_exact_jaccard
        for name, values in novelty.items()
    }
    novelty_checks.update(
        {
            f"{name}_near_share": values["max_bidirectional_near_share"]
            <= cfg.maximum_near_share
            for name, values in novelty.items()
        }
    )
    source_checks = {
        **cast(dict[str, bool], selected["checks"]),
        **novelty_checks,
        **stage_feasibility_checks,
    }

    clock_path = Path(cfg.clock_output)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    if clock_path.exists():
        temporary = clock_path.with_name(clock_path.name + ".rebuild")
        deterministic_gzip_csv(events, temporary)
        rebuilt = temporary.read_bytes()
        temporary.unlink()
        _write_once(clock_path, rebuilt)
    else:
        deterministic_gzip_csv(events, clock_path)

    report_core = {
        "candidate": POLICY_ID,
        "protocol": protocol(),
        "config": asdict(cfg),
        "preregistration_source": str(PREREGISTRATION_SOURCE),
        "preregistration_source_sha256": sha256_file(PREREGISTRATION_SOURCE),
        "flow_panel": str(FLOW_PANEL),
        "flow_panel_sha256": FLOW_PANEL_SHA256,
        "flow_manifest": str(FLOW_MANIFEST),
        "flow_manifest_sha256": FLOW_MANIFEST_SHA256,
        "premium_summary": str(PREMIUM_SUMMARY),
        "premium_summary_sha256": PREMIUM_SUMMARY_SHA256,
        "premium_sources": {
            symbol: {"path": str(premium_path(symbol)), "sha256": digest}
            for symbol, digest in PREMIUM_SHA256.items()
        },
        "source_manifest_outcomes_opened": source_manifests["flow"]["protocol"][
            "post_entry_outcomes_opened"
        ],
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "btc_execution_rows_loaded": 0,
        "btc_funding_rows_loaded": 0,
        "future_source_values_opened_before_selection": False,
        "selection_source_end_exclusive": SELECTION_END.isoformat(),
        "tested_cells": cells,
        "selected": {
            "flow_tail_quantile": selected["flow_tail_quantile"],
            "premium_tail_quantile": selected["premium_tail_quantile"],
            "consensus_count": selected["consensus_count"],
            "selection_rule_used_future_source_metrics": False,
            "selection_rule_used_outcomes": False,
        },
        "support": support,
        "clock_output": str(clock_path),
        "clock_sha256": sha256_file(clock_path),
        "clock_rows": len(events),
        "comparators": COMPARATORS,
        "novelty": novelty,
        "checks": source_checks,
        "support_passed": all(source_checks.values()),
        "advance_to_evaluator_freeze": all(source_checks.values()),
        "disposition": (
            "ADVANCE_TO_EVALUATOR_FREEZE"
            if all(source_checks.values())
            else "REJECT_SOURCE_SUPPORT_NO_OUTCOME_OPEN"
        ),
    }
    report = {**report_core, "manifest_hash": canonical_hash(report_core)}
    result_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    docs_bytes = _docs(report).encode()
    _write_once(cfg.result_output, result_bytes)
    _write_once(cfg.docs_output, docs_bytes)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-output", default=Config.result_output)
    parser.add_argument("--clock-output", default=Config.clock_output)
    parser.add_argument("--docs-output", default=Config.docs_output)
    args = parser.parse_args()
    report = run(
        Config(
            result_output=args.result_output,
            clock_output=args.clock_output,
            docs_output=args.docs_output,
        )
    )
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "selected": report["selected"],
                "support": report["support"],
                "clock_sha256": report["clock_sha256"],
                "manifest_hash": report["manifest_hash"],
                "support_passed": report["support_passed"],
                "outcomes_opened": report["outcomes_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
