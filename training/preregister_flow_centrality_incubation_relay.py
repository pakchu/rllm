"""Outcome-blind support selection for FCIR-12.

FCIR-12 asks whether dynamically influential alt-perpetual taker flow leads BTC
while the equal-weight alt crowd is still quiet.  Parameter selection uses only
2023 source incidence.  BTC OHLC, funding, returns, excursions, PnL, and equity
are forbidden in this module.
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

from training.build_six_alt_price_free_flow_panel import (
    OUTPUT_COLUMNS as SOURCE_COLUMNS,
)
from training.build_six_alt_price_free_flow_panel import SYMBOLS
from training.build_six_alt_price_free_flow_panel import (
    deterministic_gzip_csv,
    sha256_file,
)


POLICY_ID = "FCIR-12"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_flow_centrality_incubation_relay.py"
)
SOURCE_PANEL = Path(
    "data/binance_six_alt_price_free_flow_2023_2026/"
    "six_alt_price_free_flow_1h_2023-01-01_2026-06-01.csv.gz"
)
SOURCE_MANIFEST = Path(
    "data/binance_six_alt_price_free_flow_2023_2026/build_manifest.json"
)
SOURCE_PANEL_SHA256 = (
    "bf4d67ee02948444712a6ff7862a0d4f4ae4ae2a704c9d0586538043c169f6b9"
)
SOURCE_MANIFEST_SHA256 = (
    "eab61cbc7f5fc51e78f574e8bef163b3a3b91bd027136cae8efd7aaf26edc0f1"
)
SOURCE_START = cast(pd.Timestamp, pd.Timestamp("2023-01-01 00:00:00"))
SOURCE_END = cast(pd.Timestamp, pd.Timestamp("2026-06-01 00:00:00"))
SELECTION_END = cast(pd.Timestamp, pd.Timestamp("2024-01-01 00:00:00"))

SPLITS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "train": (
        cast(pd.Timestamp, pd.Timestamp("2023-01-01")),
        cast(pd.Timestamp, pd.Timestamp("2024-01-01")),
    ),
    "test": (
        cast(pd.Timestamp, pd.Timestamp("2024-01-01")),
        cast(pd.Timestamp, pd.Timestamp("2025-01-01")),
    ),
    "eval": (
        cast(pd.Timestamp, pd.Timestamp("2025-01-01")),
        cast(pd.Timestamp, pd.Timestamp("2026-01-01")),
    ),
    "final": (
        cast(pd.Timestamp, pd.Timestamp("2026-01-01")),
        SOURCE_END,
    ),
}

COMPARATORS = {
    "CLD-72": {
        "path": "results/cross_sectional_leadership_diffusion_event_clock_2026-07-18.json",
        "sha256": "089ae3f854459a76bade4e3fd6682d1b1a9a6d600dc990a367840c179c0e623d",
        "kind": "cld_json",
    },
    "SQFD-6": {
        "path": "data/stablecoin_quote_flow_diffusion_clocks_2023_2026.csv.gz",
        "sha256": "a81e144eea1e80ae5439fc66db1fad5bbd00cd9ac177e25142b5cfb5a07bcc5b",
        "kind": "primary_csv",
    },
    "OPDR-24": {
        "path": "data/options_perpetual_demand_relay_clocks_2023_2026.csv.gz",
        "sha256": "ceb79b206c3e1f6bf78b02cd2ace9a94f875ce930a704cc6e7a5a8b255021b99",
        "kind": "primary_csv",
    },
    "PCBR-12": {
        "path": "data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz",
        "sha256": "659fc1b6b6e3a20e60031ed1d50f51c8c7d2836956f911f62ad13e4152740cda",
        "kind": "primary_csv",
    },
    "PSR-30/6": {
        "path": "data/premium_snapback_recenter_clocks_2020_2026.csv.gz",
        "sha256": "cb209ed35f9baa08cc2fb3dd5bd60b8e747b1408c09507b774ca275e0b2b2db6",
        "kind": "psr_csv",
    },
}

EVENT_COLUMNS = (
    "candidate",
    "split",
    "source_hour_open_utc",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "central_flow",
    "equal_weight_flow",
    "central_abs_threshold",
    "crowd_quiet_threshold",
    "effective_names",
    *(f"weight_{symbol.lower()}" for symbol in SYMBOLS),
)
FORBIDDEN_OUTCOME_TOKENS = (
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
        "results/flow_centrality_incubation_relay_support_2026-07-19.json"
    )
    clock_output: str = (
        "data/flow_centrality_incubation_relay_clocks_2023_2026.csv.gz"
    )
    docs_output: str = (
        "docs/flow-centrality-incubation-relay-preregistration-2026-07-19.md"
    )
    centrality_window_hours: int = 720
    centrality_minimum_hours: int = 672
    threshold_window_hours: int = 2160
    threshold_minimum_hours: int = 720
    central_flow_quantiles: tuple[float, ...] = (0.65, 0.70, 0.75, 0.80)
    effective_name_minima: tuple[float, ...] = (2.2, 2.6, 3.0)
    crowd_quiet_quantile: float = 0.50
    entry_delay_minutes: int = 5
    hold_hours: int = 12
    minimum_train_events: int = 60
    minimum_train_half_events: int = 20
    minimum_train_quarter_events: int = 8
    minimum_train_side_share: float = 0.35
    maximum_train_month_share: float = 0.25
    novelty_tolerance_hours: int = 6
    maximum_exact_jaccard: float = 0.05
    maximum_near_share: float = 0.35


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def protocol() -> dict[str, Any]:
    cfg = Config()
    return {
        "policy_id": POLICY_ID,
        "hypothesis": (
            "BTC follows the direction of strong completed-hour taker imbalance "
            "in alt symbols that have strictly-prior directed flow influence while "
            "equal-weight alt flow remains historically quiet"
        ),
        "evidence_boundary": {
            "allowed": [
                "frozen six-alt completed-hour normalized taker flow",
                "strictly-prior rolling directed correlations and thresholds",
                "source-only event timestamps, sides, concentration, and clock overlap",
            ],
            "forbidden": [
                "BTC OHLC or funding",
                "entry or later prices",
                "post-entry return or excursion",
                "PnL, equity, CAGR, MDD, hit rate, or payoff",
            ],
            "post_entry_outcomes_opened": False,
        },
        "directed_network": {
            "flow": "(2 * taker_buy_quote - quote_volume) / quote_volume",
            "edge": (
                "rho(i[u-1], j[u]) - rho(j[u-1], i[u]), clipped below at zero"
            ),
            "edge_window": (
                f"prior {cfg.centrality_window_hours} target hours ending at t-1; "
                f"minimum {cfg.centrality_minimum_hours}; current t excluded"
            ),
            "weight": "positive outgoing net-lead edge sum normalized across symbols",
            "effective_names": "1 / sum(weight_i ** 2)",
        },
        "signal": {
            "central_flow": "sum(prior_network_weight_i * current_flow_i)",
            "equal_weight_flow": "mean(current_flow_i)",
            "strong_influential_flow": (
                "abs(central_flow) >= its strictly-prior selected quantile"
            ),
            "quiet_crowd": (
                "abs(equal_weight_flow) <= its strictly-prior q50"
            ),
            "threshold_history": (
                f"rolling {cfg.threshold_window_hours}-hour window, activates after "
                f"{cfg.threshold_minimum_hours} prior valid observations, current t excluded"
            ),
            "side": "sign(central_flow); direction flip is a frozen control",
            "trigger": "false-to-true onset only",
        },
        "clock": {
            "decision": "right edge of a completed UTC hour",
            "entry": "decision + 5 minutes",
            "exit": "entry + 12 hours",
            "position_state": "one BTC position maximum per split; skip overlapping onsets",
        },
        "selection": {
            "source_window": "2023 only",
            "outcomes_used": False,
            "rule": (
                "among train-support-passing cells maximize central-flow quantile, "
                "then minimum effective names; never use BTC or future-source metrics"
            ),
            "future_source_incidence": (
                "opened only after selection and reported as non-selecting diagnostics"
            ),
        },
        "eventual_execution": {
            "instrument": "Binance BTCUSDT USD-M perpetual",
            "leverage": 0.5,
            "cost_bp_per_notional_side": 6.0,
            "funding": "exact realized funding ownership on [entry, exit)",
            "strict_mdd": "global pre-entry HWM plus every held five-minute path",
            "full_calendar_cagr": True,
            "controls": [
                "direction flip on identical clocks",
                "equal-weight-flow side on identical clocks",
                "24-hour stale network",
                "deterministic symbol-permuted network",
                "deterministic random side on identical clocks",
                "one-hour additional latency",
                "10 bp per-notional-side cost stress",
            ],
        },
        "outcome_gate": {
            "absolute_return_positive": True,
            "cagr_to_strict_mdd_min": 3.0,
            "strict_mdd_max_pct": 15.0,
            "minimum_trades": {
                "train": 60,
                "test": 80,
                "eval": 55,
                "final": 30,
            },
            "weekly_cluster_signflip_p_max": 0.10,
            "mean_gross_underlying_move_bp_min": 20.0,
            "each_contained_half_absolute_return_positive": True,
            "stress_cost_notional_per_side": 0.0010,
            "stress_absolute_return_positive": True,
            "stress_cagr_to_strict_mdd_min": 2.5,
            "mechanism_control_margin_min": 0.25,
            "ratio_definition": (
                "CAGR_pct / strict_MDD_pct; exact zero-MDD uses +inf, 0, or "
                "-inf according to CAGR sign; +inf controls fail margin"
            ),
            "statistical_test": {
                "cluster_key": "UTC entry timestamp ISO year/week",
                "cluster_value": (
                    "sum of net account trade returns after base costs and "
                    "realized funding"
                ),
                "observed": "absolute mean net trade return",
                "null": (
                    "independent Rademacher sign per weekly sum; signed total "
                    "divided by trade count"
                ),
                "exact_cluster_max": 20,
                "monte_carlo_draws": 20_000,
                "seed": 20_260_719,
                "p_value_rule": (
                    "exact exceed/2^K when K<=20; otherwise "
                    "p=(1+exceed)/(20000+1)"
                ),
            },
            "sequential_opening": (
                "train then test then eval then final; stop on first failed gate"
            ),
        },
        "sequential_oos": {
            "stages": ["2023 train", "2024 test", "2025 eval", "2026H1 final"],
            "failed_stage_action": "retire exact frozen policy without repair",
        },
    }


def _validate_config(cfg: Config) -> None:
    if cfg != Config(
        result_output=cfg.result_output,
        clock_output=cfg.clock_output,
        docs_output=cfg.docs_output,
    ):
        raise ValueError("FCIR signal and support configuration is frozen")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid source boolean: {value}")
    return normalized == "true"


def _validate_source_artifacts() -> dict[str, Any]:
    if sha256_file(SOURCE_PANEL) != SOURCE_PANEL_SHA256:
        raise RuntimeError("FCIR source panel hash mismatch")
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("FCIR source manifest hash mismatch")
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    source_protocol = manifest["protocol"]
    if source_protocol["post_entry_outcomes_opened"] is not False:
        raise RuntimeError("FCIR source manifest opened post-entry outcomes")
    if source_protocol["price_values_read"] is not False:
        raise RuntimeError("FCIR source manifest read price values")
    if manifest["combined_sha256"] != SOURCE_PANEL_SHA256:
        raise RuntimeError("FCIR panel and source manifest disagree")
    return manifest


def load_source_prefix(
    path: str | Path = SOURCE_PANEL,
    *,
    end_exclusive: pd.Timestamp | None,
) -> pd.DataFrame:
    """Read source rows sequentially and stop before a sealed boundary."""
    selected = {
        "feature_available_time_utc",
        "symbol",
        "taker_flow_fraction",
        "feature_valid",
    }
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise RuntimeError("FCIR source panel schema changed")
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
    frame = pd.DataFrame(rows, columns=cast(Any, sorted(selected)))
    if frame.empty:
        raise RuntimeError("FCIR source prefix is empty")
    return frame


def flow_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    symbols = tuple(cast(pd.Series, frame["symbol"]).drop_duplicates())
    if set(symbols) != set(SYMBOLS):
        raise RuntimeError("FCIR source prefix does not contain the frozen universe")
    if frame[["feature_available_time_utc", "symbol"]].duplicated().any():
        raise RuntimeError("FCIR source prefix contains duplicate hour-symbol rows")
    counts = frame.groupby("feature_available_time_utc")["symbol"].nunique()
    if not bool(counts.eq(len(SYMBOLS)).all()):
        raise RuntimeError("FCIR source prefix has an incomplete hour-symbol grid")
    values = cast(
        pd.DataFrame,
        frame.pivot(
            index="feature_available_time_utc",
            columns="symbol",
            values="taker_flow_fraction",
        ).reindex(columns=SYMBOLS),
    )
    validity = cast(
        pd.DataFrame,
        frame.pivot(
            index="feature_available_time_utc",
            columns="symbol",
            values="feature_valid",
        ).reindex(columns=SYMBOLS),
    )
    expected = pd.date_range(values.index.min(), values.index.max(), freq="1h")
    if not values.index.equals(expected):
        raise RuntimeError("FCIR source prefix is not an exact hourly boundary grid")
    all_valid = validity.astype(bool).all(axis=1)
    return values.where(all_valid)


def prior_directed_weights(
    flow: pd.DataFrame,
    *,
    window: int,
    minimum: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute positive net lead weights whose newest target is t-1."""
    if minimum > window or minimum < 2:
        raise ValueError("invalid directed-correlation window")
    correlations: dict[tuple[str, str], pd.Series] = {}
    for leader in SYMBOLS:
        for follower in SYMBOLS:
            if leader == follower:
                continue
            correlations[leader, follower] = cast(
                pd.Series,
                flow[leader]
                .shift(1)
                .rolling(window, min_periods=minimum)
                .corr(flow[follower])
                .shift(1),
            )
    outgoing = pd.DataFrame(
        0.0, index=flow.index, columns=cast(Any, list(SYMBOLS))
    )
    for leader in SYMBOLS:
        for follower in SYMBOLS:
            if leader == follower:
                continue
            advantage = cast(
                pd.Series,
                correlations[leader, follower] - correlations[follower, leader]
            ).clip(lower=0.0)
            outgoing[leader] = outgoing[leader] + advantage
    total = cast(pd.Series, outgoing.sum(axis=1))
    weights = outgoing.div(total.replace(0.0, np.nan), axis=0)
    weight_square_sum = cast(pd.Series, weights.pow(2).sum(axis=1))
    effective_names = (1.0 / weight_square_sum).where(total.gt(0.0))
    return weights, effective_names


def prior_quantile(
    values: pd.Series,
    *,
    quantile: float,
    window: int,
    minimum: int,
) -> pd.Series:
    return cast(
        pd.Series,
        values.rolling(window, min_periods=minimum).quantile(quantile).shift(1),
    )


def feature_panel(flow: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    weights, effective = prior_directed_weights(
        flow,
        window=cfg.centrality_window_hours,
        minimum=cfg.centrality_minimum_hours,
    )
    valid = flow.notna().all(axis=1) & weights.notna().all(axis=1)
    central = cast(
        pd.Series, weights.mul(flow).sum(axis=1, min_count=len(SYMBOLS))
    ).where(valid)
    equal = cast(pd.Series, flow.mean(axis=1)).where(valid)
    features = pd.DataFrame(
        {
            "central_flow": central,
            "equal_weight_flow": equal,
            "effective_names": effective.where(valid),
            "crowd_quiet_threshold": prior_quantile(
                equal.abs(),
                quantile=cfg.crowd_quiet_quantile,
                window=cfg.threshold_window_hours,
                minimum=cfg.threshold_minimum_hours,
            ),
            "source_valid": valid,
        },
        index=flow.index,
    )
    for quantile in cfg.central_flow_quantiles:
        features[f"central_abs_q{int(round(quantile * 100)):02d}"] = prior_quantile(
            central.abs(),
            quantile=quantile,
            window=cfg.threshold_window_hours,
            minimum=cfg.threshold_minimum_hours,
        )
    for symbol in SYMBOLS:
        features[f"weight_{symbol.lower()}"] = weights[symbol]
    return features


def signal_state(
    features: pd.DataFrame,
    *,
    central_quantile: float,
    minimum_effective_names: float,
) -> pd.Series:
    threshold = features[f"central_abs_q{int(round(central_quantile * 100)):02d}"]
    return cast(
        pd.Series,
        (
            features["source_valid"]
            & features["central_flow"].notna()
            & features["central_flow"].ne(0.0)
            & features["central_flow"].abs().ge(threshold)
            & features["equal_weight_flow"].abs().le(
                features["crowd_quiet_threshold"]
            )
            & features["effective_names"].ge(minimum_effective_names)
        ).fillna(False),
    )


def schedule_events(
    features: pd.DataFrame,
    state: pd.Series,
    *,
    central_quantile: float,
    cfg: Config,
) -> pd.DataFrame:
    onset = state & ~state.shift(1, fill_value=False)
    threshold_column = f"central_abs_q{int(round(central_quantile * 100)):02d}"
    rows: list[dict[str, Any]] = []
    for split, (start, end) in SPLITS.items():
        last_exit = start
        eligible = features.index[
            onset
            & features.index.to_series().ge(start)
            & features.index.to_series().lt(end)
        ]
        for decision in eligible:
            decision_time = cast(pd.Timestamp, decision)
            entry = cast(
                pd.Timestamp,
                decision_time + pd.Timedelta(minutes=cfg.entry_delay_minutes),
            )
            exit_time = cast(
                pd.Timestamp, entry + pd.Timedelta(hours=cfg.hold_hours)
            )
            if entry < last_exit or exit_time > end:
                continue
            row = features.loc[decision_time]
            event = {
                "candidate": POLICY_ID,
                "split": split,
                "source_hour_open_utc": decision_time - pd.Timedelta(hours=1),
                "decision_time": decision_time,
                "feature_available_time": decision_time,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": 1 if float(row["central_flow"]) > 0.0 else -1,
                "central_flow": float(row["central_flow"]),
                "equal_weight_flow": float(row["equal_weight_flow"]),
                "central_abs_threshold": float(row[threshold_column]),
                "crowd_quiet_threshold": float(row["crowd_quiet_threshold"]),
                "effective_names": float(row["effective_names"]),
            }
            for symbol in SYMBOLS:
                event[f"weight_{symbol.lower()}"] = float(
                    row[f"weight_{symbol.lower()}"]
                )
            rows.append(event)
            last_exit = exit_time
    return pd.DataFrame(rows, columns=cast(Any, list(EVENT_COLUMNS)))


def _window_counts(events: pd.DataFrame, split: str) -> dict[str, int]:
    start, end = SPLITS[split]
    window = cast(pd.DataFrame, events.loc[events["split"].eq(split)])
    if split == "final":
        boundaries = {
            "2026_q1": (start, cast(pd.Timestamp, pd.Timestamp("2026-04-01"))),
            "2026_q2": (cast(pd.Timestamp, pd.Timestamp("2026-04-01")), end),
        }
    else:
        mid = cast(pd.Timestamp, start + (end - start) / 2)
        boundaries = {f"{split}_h1": (start, mid), f"{split}_h2": (mid, end)}
    return {
        name: int(
            (
                cast(pd.Series, window["entry_time"]).ge(left)
                & cast(pd.Series, window["exit_time"]).le(right)
            ).sum()
        )
        for name, (left, right) in boundaries.items()
    }


def support_summary(events: pd.DataFrame, split: str) -> dict[str, Any]:
    window = cast(pd.DataFrame, events.loc[events["split"].eq(split)]).copy()
    total = len(window)
    month_counts = (
        cast(pd.Series, window["entry_time"]).dt.to_period("M").value_counts()
        if total
        else pd.Series(dtype="int64")
    )
    quarter_counts = (
        cast(pd.Series, window["entry_time"]).dt.to_period("Q").value_counts()
        if total
        else pd.Series(dtype="int64")
    )
    return {
        "events": total,
        "long": int(cast(pd.Series, window["side"]).eq(1).sum()),
        "short": int(cast(pd.Series, window["side"]).eq(-1).sum()),
        "side_share_min": (
            float(
                min(
                    cast(pd.Series, window["side"]).eq(1).sum(),
                    cast(pd.Series, window["side"]).eq(-1).sum(),
                )
                / total
            )
            if total
            else 0.0
        ),
        "maximum_month_share": float(month_counts.max() / total) if total else 0.0,
        "month_counts": {str(key): int(value) for key, value in month_counts.items()},
        "quarter_counts": {
            str(key): int(value) for key, value in quarter_counts.sort_index().items()
        },
        "subwindows": _window_counts(events, split),
    }


def train_support_checks(summary: dict[str, Any], cfg: Config) -> dict[str, bool]:
    return {
        "train_events": summary["events"] >= cfg.minimum_train_events,
        "train_side_balance": summary["side_share_min"]
        >= cfg.minimum_train_side_share,
        "train_half_counts": min(summary["subwindows"].values())
        >= cfg.minimum_train_half_events,
        "train_quarter_counts": len(summary["quarter_counts"]) == 4
        and min(summary["quarter_counts"].values())
        >= cfg.minimum_train_quarter_events,
        "train_month_concentration": summary["maximum_month_share"]
        <= cfg.maximum_train_month_share,
    }


def _reject_outcome_fields(payload: Any, path: str = "root") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_OUTCOME_TOKENS):
                raise ValueError(f"forbidden outcome field in support selection: {path}.{key}")
            _reject_outcome_fields(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _reject_outcome_fields(value, f"{path}[{index}]")


def select_support_cell(cells: list[dict[str, Any]]) -> dict[str, Any]:
    _reject_outcome_fields(cells)
    passing = [cell for cell in cells if cell["passes"]]
    if not passing:
        raise RuntimeError("no FCIR source-support cell passed")
    return max(
        passing,
        key=lambda cell: (
            float(cell["central_flow_quantile"]),
            float(cell["minimum_effective_names"]),
        ),
    )


def _load_comparator_entries() -> dict[str, pd.DatetimeIndex]:
    entries: dict[str, pd.DatetimeIndex] = {}
    for name, spec in COMPARATORS.items():
        path = Path(spec["path"])
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"FCIR comparator clock hash mismatch: {name}")
        if spec["kind"] == "cld_json":
            payload = json.loads(path.read_text())
            values = pd.to_datetime([row["entry_date"] for row in payload["events"]])
        elif spec["kind"] == "primary_csv":
            frame = cast(
                pd.DataFrame,
                pd.read_csv(
                    path, usecols=cast(Any, ["control", "entry_time"])
                ),
            )
            values = pd.to_datetime(
                frame.loc[frame["control"].eq("primary"), "entry_time"], utc=True
            ).dt.tz_localize(None)
        elif spec["kind"] == "psr_csv":
            frame = cast(
                pd.DataFrame,
                pd.read_csv(path, usecols=cast(Any, ["entry_time"])),
            )
            values = pd.to_datetime(frame["entry_time"], utc=True).dt.tz_localize(None)
        else:
            raise RuntimeError(f"unknown FCIR comparator kind: {spec['kind']}")
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
    new_start = cast(pd.Timestamp, new_entries.min())
    prior_start = cast(pd.Timestamp, prior_entries.min())
    new_end = cast(pd.Timestamp, new_entries.max())
    prior_end = cast(pd.Timestamp, prior_entries.max())
    coverage_start = max(new_start, prior_start)
    coverage_end = min(new_end, prior_end)
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
    new = new_entries[
        (new_entries >= coverage_start) & (new_entries <= coverage_end)
    ]
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
            raise RuntimeError(f"refusing to overwrite immutable FCIR artifact: {path}")
        return
    destination.write_bytes(payload)


def _docs(report: dict[str, Any]) -> str:
    selected = report["selected"]
    support = report["support"]
    novelty = report["novelty"]
    novelty_lines = "\n".join(
        f"- {name}: exact Jaccard `{values['exact_jaccard']:.4f}`, "
        f"±6h max near-share `{values['max_bidirectional_near_share']:.4f}`"
        for name, values in novelty.items()
    )
    return f"""# FCIR-12 source-only preregistration — 2026-07-19

## Status

`FCIR-12` passed source-only support and novelty gates. **No BTC price,
funding, return, excursion, PnL, equity, CAGR, or MDD was opened.** This is
permission to freeze an evaluator, not evidence of profitability.

## Frozen mechanism

- Source: normalized completed-hour taker flow for six USD-M alts.
- Directed edge: lag-one correlation advantage, estimated from the prior
  720 target hours with at least 672 observations; newest target is `t-1`.
- Central flow: current flow weighted by the strictly-prior outgoing net-lead
  network.
- Crowd gate: absolute equal-weight current flow is at or below its strictly
  prior median from a rolling 90-day window, activated after at least 720
  prior valid observations.
- Strength gate: absolute central flow is at or above its strictly-prior
  q{int(round(selected['central_flow_quantile'] * 100))} from the same rolling
  90-day/minimum-720-prior-observation contract.
- Network breadth: effective names at least
  `{selected['minimum_effective_names']:.1f}`.
- Side: sign of central flow.
- Clock: false-to-true onset, entry `+5m`, fixed `12h` hold, one position.

The selected cell was chosen only from 2023 source incidence by maximizing
mechanism strength among support-passing cells. Later source incidence was
opened only after selection and did not alter the cell.

## Source-only incidence

| Stage | Events | Long | Short | Max month share |
|---|---:|---:|---:|---:|
| train 2023 | {support['train']['events']} | {support['train']['long']} | {support['train']['short']} | {support['train']['maximum_month_share']:.3f} |
| test 2024 | {support['test']['events']} | {support['test']['long']} | {support['test']['short']} | {support['test']['maximum_month_share']:.3f} |
| eval 2025 | {support['eval']['events']} | {support['eval']['long']} | {support['eval']['short']} | {support['eval']['maximum_month_share']:.3f} |
| final 2026H1 | {support['final']['events']} | {support['final']['long']} | {support['final']['short']} | {support['final']['maximum_month_share']:.3f} |

## Clock novelty

{novelty_lines}

## Why the earlier strict-disagreement variant was dropped

Requiring at least four of six raw flow signs to oppose central flow produced
only 14 non-overlapping 2023 events. It was retired before outcomes. FCIR-12
tests a different and more coherent incubation claim: influential flow is
strong while aggregate crowd flow is still quiet, not necessarily opposite.

## Sequential outcome rule

The evaluator and all controls must be committed before 2023 BTC outcomes are
opened. Train must pass every frozen economic, significance, stability,
stress, and mechanism-margin gate. Failure retires the exact policy without
opening 2024 or repairing from controls.
"""


def run(cfg: Config = Config()) -> dict[str, Any]:
    _validate_config(cfg)
    source_manifest = _validate_source_artifacts()

    # Physical prefix read: future source feature values remain unopened until
    # the 2023-only support selector has returned one immutable cell.
    train_source = load_source_prefix(end_exclusive=SELECTION_END)
    train_flow = flow_matrix(train_source)
    train_features = feature_panel(train_flow, cfg)
    cells: list[dict[str, Any]] = []
    for central_quantile in cfg.central_flow_quantiles:
        for minimum_effective_names in cfg.effective_name_minima:
            state = signal_state(
                train_features,
                central_quantile=central_quantile,
                minimum_effective_names=minimum_effective_names,
            )
            events = schedule_events(
                train_features,
                state,
                central_quantile=central_quantile,
                cfg=cfg,
            )
            summary = support_summary(events, "train")
            checks = train_support_checks(summary, cfg)
            cells.append(
                {
                    "central_flow_quantile": central_quantile,
                    "minimum_effective_names": minimum_effective_names,
                    "train_support": summary,
                    "checks": checks,
                    "passes": all(checks.values()),
                }
            )
    selected = select_support_cell(cells)

    full_source = load_source_prefix(end_exclusive=None)
    full_flow = flow_matrix(full_source)
    full_features = feature_panel(full_flow, cfg)
    full_state = signal_state(
        full_features,
        central_quantile=float(selected["central_flow_quantile"]),
        minimum_effective_names=float(selected["minimum_effective_names"]),
    )
    events = schedule_events(
        full_features,
        full_state,
        central_quantile=float(selected["central_flow_quantile"]),
        cfg=cfg,
    )
    prefix_events = cast(pd.DataFrame, events.loc[events["split"].eq("train")])
    selected_train_state = signal_state(
        train_features,
        central_quantile=float(selected["central_flow_quantile"]),
        minimum_effective_names=float(selected["minimum_effective_names"]),
    )
    selected_train_events = schedule_events(
        train_features,
        selected_train_state,
        central_quantile=float(selected["central_flow_quantile"]),
        cfg=cfg,
    )
    if not prefix_events["entry_time"].reset_index(drop=True).equals(
        selected_train_events["entry_time"].reset_index(drop=True)
    ):
        raise RuntimeError("future FCIR source incidence changed the selected train clock")

    support = {split: support_summary(events, split) for split in SPLITS}
    new_entries = pd.DatetimeIndex(events["entry_time"]).sort_values()
    comparator_entries = _load_comparator_entries()
    tolerance = cast(
        pd.Timedelta, pd.Timedelta(hours=cfg.novelty_tolerance_hours)
    )
    novelty = {
        name: novelty_metrics(new_entries, prior, tolerance=tolerance)
        for name, prior in comparator_entries.items()
    }
    novelty_checks = {
        f"{name}_exact_jaccard": values["exact_jaccard"]
        <= cfg.maximum_exact_jaccard
        for name, values in novelty.items()
    }
    novelty_checks.update(
        {
            f"{name}_near_share": values["max_bidirectional_near_share"]
            <= cfg.maximum_near_share
            for name, values in novelty.items()
        }
    )
    selected_checks = cast(dict[str, bool], selected["checks"])
    all_checks = {**selected_checks, **novelty_checks}

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
        "source_panel": str(SOURCE_PANEL),
        "source_panel_sha256": SOURCE_PANEL_SHA256,
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_manifest_outcomes_opened": source_manifest["protocol"][
            "post_entry_outcomes_opened"
        ],
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "btc_execution_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "future_source_values_opened_before_selection": False,
        "selection_source_end_exclusive": SELECTION_END.isoformat(),
        "tested_cells": cells,
        "selected": {
            "central_flow_quantile": selected["central_flow_quantile"],
            "minimum_effective_names": selected["minimum_effective_names"],
            "selection_rule_used_future_source_metrics": False,
            "selection_rule_used_outcomes": False,
        },
        "support": support,
        "clock_output": str(clock_path),
        "clock_sha256": sha256_file(clock_path),
        "clock_rows": len(events),
        "comparators": COMPARATORS,
        "novelty": novelty,
        "checks": all_checks,
        "support_passed": all(all_checks.values()),
        "advance_to_evaluator_freeze": all(all_checks.values()),
    }
    report = {**report_core, "manifest_hash": canonical_hash(report_core)}
    if not report["support_passed"]:
        raise RuntimeError("FCIR source-only support or novelty gate failed")
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
                "support": {
                    split: {
                        key: report["support"][split][key]
                        for key in ("events", "long", "short", "maximum_month_share")
                    }
                    for split in SPLITS
                },
                "support_passed": report["support_passed"],
                "outcomes_opened": report["outcomes_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
