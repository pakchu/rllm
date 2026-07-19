"""Outcome-blind support selection for TGR-12 ticket-gap release clocks.

TGR-12 asks whether flow from the two alt perpetuals with the largest causal
mean-ticket surprise leads BTC when the remaining four-symbol crowd is quiet.
This module reads no BTC execution price, funding, return, PnL, or equity.
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
    OUTPUT_COLUMNS as SOURCE_SCHEMA,
    SYMBOLS,
    deterministic_gzip_csv,
    sha256_file,
)


POLICY_ID = "TGR-12"
PREREGISTRATION_SOURCE = Path("training/preregister_ticket_gap_release.py")
SOURCE_PANEL = Path(
    "data/binance_six_alt_price_free_flow_2023_2026/"
    "six_alt_price_free_flow_1h_2023-01-01_2026-06-01.csv.gz"
)
SOURCE_MANIFEST = Path(
    "data/binance_six_alt_price_free_flow_2023_2026/build_manifest.json"
)
SOURCE_PANEL_SHA256 = "bf4d67ee02948444712a6ff7862a0d4f4ae4ae2a704c9d0586538043c169f6b9"
SOURCE_MANIFEST_SHA256 = (
    "eab61cbc7f5fc51e78f574e8bef163b3a3b91bd027136cae8efd7aaf26edc0f1"
)
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
}

SOURCE_COLUMNS = (
    "feature_available_time_utc",
    "symbol",
    "taker_flow_fraction",
    "mean_ticket_usdt",
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
    "top_symbol_1",
    "top_symbol_2",
    "top_ticket_flow",
    "bottom_crowd_flow",
    "ticket_gap",
    "top_flow_abs_threshold",
    "bottom_quiet_threshold",
    "ticket_gap_threshold",
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
    result_output: str = "results/ticket_gap_release_support_2026-07-19.json"
    clock_output: str = "data/ticket_gap_release_clocks_2023_2026.csv.gz"
    docs_output: str = "docs/ticket-gap-release-preregistration-2026-07-19.md"
    ticket_window_hours: int = 720
    ticket_minimum_hours: int = 672
    threshold_window_hours: int = 2160
    threshold_minimum_hours: int = 720
    top_flow_quantiles: tuple[float, ...] = (0.75, 0.80, 0.85, 0.875, 0.90)
    ticket_gap_quantiles: tuple[float, ...] = (0.70, 0.75, 0.80, 0.825, 0.85)
    bottom_quiet_quantile: float = 0.50
    top_symbols: int = 2
    entry_delay_minutes: int = 5
    hold_hours: int = 12
    minimum_train_events: int = 55
    minimum_train_half_events: int = 20
    minimum_active_quarter_events: int = 15
    minimum_train_side_share: float = 0.40
    maximum_train_month_share: float = 0.20
    novelty_tolerance_hours: int = 6
    maximum_exact_jaccard: float = 0.05
    maximum_near_share: float = 0.35


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def protocol() -> dict[str, Any]:
    cfg = Config()
    return {
        "policy_id": POLICY_ID,
        "hypothesis": (
            "after unusually large mean-ticket separation across six alt "
            "perpetuals, aligned flow from the two high-surprise symbols leads "
            "BTC while the remaining four-symbol crowd is quiet"
        ),
        "evidence_boundary": {
            "allowed": [
                "frozen completed-hour six-alt mean ticket and normalized taker flow",
                "strictly-prior rolling robust ticket baselines and thresholds",
                "source-only event incidence, sides, concentration, and overlap",
            ],
            "forbidden": [
                "BTC OHLC or funding",
                "entry or later prices",
                "post-entry return or excursion",
                "PnL, equity, CAGR, MDD, hit rate, or payoff",
            ],
            "post_entry_outcomes_opened": False,
        },
        "ticket_rank": {
            "ticket_measure": "log(completed-hour quote volume / trade count)",
            "robust_baseline": (
                f"prior {cfg.ticket_window_hours} hours, minimum "
                f"{cfg.ticket_minimum_hours}; current hour excluded"
            ),
            "z_score": (
                "(log_ticket - prior rolling median) / "
                "(1.4826 * prior rolling median absolute deviation)"
            ),
            "tie_break": "fixed alphabetical symbol order",
            "leaders": "two largest current robust ticket surprises",
        },
        "signal": {
            "top_ticket_flow": "equal mean normalized taker flow of the two leaders",
            "bottom_crowd_flow": "equal mean normalized taker flow of the other four",
            "ticket_gap": "mean top-two ticket z minus mean bottom-four ticket z",
            "requirements": [
                "both top-two flows have the same nonzero sign as top_ticket_flow",
                "abs(top_ticket_flow) reaches its strictly-prior selected quantile",
                "ticket_gap reaches its strictly-prior selected quantile",
                "abs(bottom_crowd_flow) is at or below its strictly-prior median",
            ],
            "threshold_history": (
                f"rolling {cfg.threshold_window_hours} hours with minimum "
                f"{cfg.threshold_minimum_hours}, shifted one hour"
            ),
            "side": "sign(top_ticket_flow)",
            "trigger": "false-to-true state onset only",
        },
        "selection": {
            "source_window": "2023 only",
            "outcomes_used": False,
            "rule": (
                "among source-support-passing cells maximize top-flow quantile, "
                "then ticket-gap quantile; never use BTC or future-source metrics"
            ),
            "future_source_incidence": (
                "opened only after selection and reported as non-selecting diagnostics"
            ),
        },
        "clock": {
            "decision": "right edge of the completed UTC source hour",
            "entry": "decision + 5 minutes",
            "exit": "entry + 12 hours",
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
                "equal-weight all-six-flow side on identical clocks",
                "24-hour stale ticket ranks",
                "deterministic symbol-permuted ticket ranks",
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
                "train": 55,
                "test": 75,
                "eval": 75,
                "final": 40,
            },
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
                "CAGR_pct / strict_MDD_pct; exact zero-MDD uses +inf, 0, or "
                "-inf according to CAGR sign; +inf controls fail margin"
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
        raise ValueError("TGR-12 source signal and support configuration is frozen")


def _validate_source_artifacts() -> dict[str, Any]:
    if sha256_file(SOURCE_PANEL) != SOURCE_PANEL_SHA256:
        raise RuntimeError("TGR-12 source panel hash changed")
    if sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("TGR-12 source manifest hash changed")
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    source_protocol = manifest["protocol"]
    if source_protocol["post_entry_outcomes_opened"] is not False:
        raise RuntimeError("TGR-12 source manifest opened outcomes")
    if source_protocol["price_values_read"] is not False:
        raise RuntimeError("TGR-12 source manifest read prices")
    if manifest["combined_sha256"] != SOURCE_PANEL_SHA256:
        raise RuntimeError("TGR-12 panel and source manifest disagree")
    return manifest


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"invalid TGR-12 source boolean: {value}")
    return normalized == "true"


def load_source_prefix(*, end_exclusive: pd.Timestamp | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with gzip.open(SOURCE_PANEL, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SOURCE_SCHEMA:
            raise RuntimeError("TGR-12 source panel schema changed")
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
                    "mean_ticket_usdt": (
                        float(raw["mean_ticket_usdt"])
                        if raw["mean_ticket_usdt"]
                        else np.nan
                    ),
                    "feature_valid": _parse_bool(raw["feature_valid"]),
                }
            )
    frame = pd.DataFrame(rows, columns=pd.Index(SOURCE_COLUMNS))
    if frame.empty or set(frame["symbol"]) != set(SYMBOLS):
        raise RuntimeError("TGR-12 source prefix is empty or incomplete")
    return frame


def source_matrices(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = cast(pd.DataFrame, frame.loc[frame["feature_valid"].eq(True)].copy())
    flow = valid.pivot(
        index="feature_available_time_utc",
        columns="symbol",
        values="taker_flow_fraction",
    ).reindex(columns=pd.Index(sorted(SYMBOLS)))
    ticket = valid.pivot(
        index="feature_available_time_utc",
        columns="symbol",
        values="mean_ticket_usdt",
    ).reindex(columns=pd.Index(sorted(SYMBOLS)))
    expected = pd.date_range(
        cast(pd.Timestamp, frame["feature_available_time_utc"].min()),
        cast(pd.Timestamp, frame["feature_available_time_utc"].max()),
        freq="1h",
    )
    flow = flow.reindex(expected)
    ticket = ticket.reindex(expected)
    return flow, ticket


def _median_absolute_deviation(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Return the NaN-tolerant median deviation around this window's median."""
    center = float(np.nanmedian(values))
    return float(np.nanmedian(np.abs(values - center)))


def robust_ticket_z(ticket: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute ticket surprise from one strictly-prior robust rolling window."""
    log_ticket = cast(pd.DataFrame, np.log(ticket.where(ticket.gt(0.0))))
    median = (
        log_ticket.rolling(
            cfg.ticket_window_hours, min_periods=cfg.ticket_minimum_hours
        )
        .median()
        .shift(1)
    )
    mad = (
        log_ticket.rolling(
            cfg.ticket_window_hours, min_periods=cfg.ticket_minimum_hours
        )
        .apply(_median_absolute_deviation, raw=True)
        .shift(1)
    )
    return cast(pd.DataFrame, (log_ticket - median) / (1.4826 * mad.clip(lower=1e-6)))


def base_feature_panel(
    flow: pd.DataFrame,
    ticket: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    ticket_z = robust_ticket_z(ticket, cfg)
    ranks = ticket_z.rank(axis=1, method="first", ascending=False)
    top = ranks.le(cfg.top_symbols)
    bottom = ranks.gt(cfg.top_symbols)
    top_flow = cast(pd.Series, flow.where(top).mean(axis=1))
    bottom_flow = cast(pd.Series, flow.where(bottom).mean(axis=1))
    ticket_gap = cast(
        pd.Series,
        ticket_z.where(top).mean(axis=1) - ticket_z.where(bottom).mean(axis=1),
    )
    side = cast(pd.Series, np.sign(top_flow))
    top_agreement = flow.where(top).mul(side, axis=0).gt(0.0).sum(axis=1)
    bottom_quiet_threshold = (
        bottom_flow.abs()
        .rolling(cfg.threshold_window_hours, min_periods=cfg.threshold_minimum_hours)
        .quantile(cfg.bottom_quiet_quantile)
        .shift(1)
    )
    top_names = ranks.apply(
        lambda row: tuple(row.nsmallest(cfg.top_symbols).index), axis=1
    )
    panel = pd.DataFrame(index=flow.index)
    panel["top_symbol_1"] = top_names.map(
        lambda names: names[0] if len(names) == cfg.top_symbols else ""
    )
    panel["top_symbol_2"] = top_names.map(
        lambda names: names[1] if len(names) == cfg.top_symbols else ""
    )
    panel["top_ticket_flow"] = top_flow
    panel["bottom_crowd_flow"] = bottom_flow
    panel["ticket_gap"] = ticket_gap
    panel["bottom_quiet_threshold"] = bottom_quiet_threshold
    panel["top_agreement"] = top_agreement
    panel["side"] = side
    return panel


def feature_panel(
    base: pd.DataFrame,
    *,
    top_flow_quantile: float,
    ticket_gap_quantile: float,
    cfg: Config,
) -> pd.DataFrame:
    panel = base.copy()
    panel["top_flow_abs_threshold"] = (
        cast(pd.Series, panel["top_ticket_flow"])
        .abs()
        .rolling(cfg.threshold_window_hours, min_periods=cfg.threshold_minimum_hours)
        .quantile(top_flow_quantile)
        .shift(1)
    )
    panel["ticket_gap_threshold"] = (
        cast(pd.Series, panel["ticket_gap"])
        .rolling(cfg.threshold_window_hours, min_periods=cfg.threshold_minimum_hours)
        .quantile(ticket_gap_quantile)
        .shift(1)
    )
    return panel


def signal_state(features: pd.DataFrame, cfg: Config) -> pd.Series:
    return cast(
        pd.Series,
        features["top_ticket_flow"].abs().ge(features["top_flow_abs_threshold"])
        & features["ticket_gap"].ge(features["ticket_gap_threshold"])
        & features["bottom_crowd_flow"].abs().le(features["bottom_quiet_threshold"])
        & features["top_agreement"].eq(cfg.top_symbols)
        & features["side"].isin((-1.0, 1.0))
        & features["top_symbol_1"].ne("")
        & features["top_symbol_2"].ne(""),
    ).fillna(False)


def schedule_events(
    features: pd.DataFrame,
    state: pd.Series,
    cfg: Config,
) -> pd.DataFrame:
    onset = state & ~state.shift(1, fill_value=False)
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
                    "top_symbol_1": str(row["top_symbol_1"]),
                    "top_symbol_2": str(row["top_symbol_2"]),
                    "top_ticket_flow": float(row["top_ticket_flow"]),
                    "bottom_crowd_flow": float(row["bottom_crowd_flow"]),
                    "ticket_gap": float(row["ticket_gap"]),
                    "top_flow_abs_threshold": float(row["top_flow_abs_threshold"]),
                    "bottom_quiet_threshold": float(row["bottom_quiet_threshold"]),
                    "ticket_gap_threshold": float(row["ticket_gap_threshold"]),
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
                    f"forbidden outcome field in TGR support selection: {path}.{key}"
                )
            _reject_outcome_fields(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _reject_outcome_fields(value, f"{path}[{index}]")


def select_support_cell(cells: list[dict[str, Any]]) -> dict[str, Any]:
    _reject_outcome_fields(cells)
    passing = [cell for cell in cells if cell["passes"]]
    if not passing:
        raise RuntimeError("no TGR-12 source-support cell passed")
    return max(
        passing,
        key=lambda cell: (
            float(cell["top_flow_quantile"]),
            float(cell["ticket_gap_quantile"]),
        ),
    )


def _load_comparator_entries() -> dict[str, pd.DatetimeIndex]:
    entries: dict[str, pd.DatetimeIndex] = {}
    for name, spec in COMPARATORS.items():
        path = Path(spec["path"])
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"TGR-12 comparator hash changed: {name}")
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
            raise RuntimeError(f"unknown TGR-12 comparator kind: {spec['kind']}")
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
            raise RuntimeError(f"refusing to overwrite immutable TGR artifact: {path}")
        return
    destination.write_bytes(payload)


def _docs(report: dict[str, Any]) -> str:
    selected = report["selected"]
    support = report["support"]
    if report["advance_to_evaluator_freeze"]:
        status = "`TGR-12` passed source-only support, feasibility, and novelty gates."
        disposition = (
            "This permits an evaluator freeze; it is not profitability evidence."
        )
    else:
        status = "`TGR-12` was rejected before opening any BTC outcome."
        disposition = (
            "The corrected causal MAD produced only "
            f"{support['test']['events']} test-year events versus the frozen "
            f"minimum of {report['protocol']['outcome_gate']['minimum_trades']['test']}; "
            "the exact policy therefore cannot pass its sequential gate."
        )
    novelty_lines = "\n".join(
        f"- {name}: exact Jaccard `{values['exact_jaccard']:.4f}`, "
        f"±6h max near-share `{values['max_bidirectional_near_share']:.4f}`"
        for name, values in report["novelty"].items()
    )
    return f"""# TGR-12 source-only preregistration — 2026-07-19

## Status

{status} **No BTC price, funding, return, excursion, PnL, equity, CAGR, or MDD
was opened.** {disposition}

## Frozen mechanism

- Source: completed-hour mean ticket and normalized taker flow for six USD-M
  alt perpetuals; no OHLC is present in this source artifact.
- Ticket surprise: robust per-symbol z-score against the strictly prior 720
  hours with at least 672 observations.
- Leaders: the two largest current ticket surprises, ties broken by frozen
  alphabetical symbol order.
- Release: both leader flows agree, their mean absolute flow reaches strictly
  prior q{int(round(selected["top_flow_quantile"] * 1000)) / 10:g}, and the
  leader-versus-crowd ticket gap reaches strictly prior
  q{int(round(selected["ticket_gap_quantile"] * 1000)) / 10:g}, while the
  bottom-four crowd flow remains below its prior median.
- Side: sign of the top-two mean flow.
- Clock: false-to-true onset, entry `+5m`, fixed `12h` hold, one position.

The selected cell maximized mechanism strength among cells passing 2023
source-incidence gates. Future source incidence was opened only after selection
and could not alter the cell.

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

The strict evaluator and every control would have to be committed before 2023
BTC execution outcomes could be opened. Because the frozen annual test trade
minimum is already impossible from source incidence, no evaluator is frozen and
all BTC outcomes remain sealed. No threshold, side, hold, or control repair is
allowed for this exact policy.
"""


def _cell(
    base: pd.DataFrame,
    *,
    top_flow_quantile: float,
    ticket_gap_quantile: float,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = feature_panel(
        base,
        top_flow_quantile=top_flow_quantile,
        ticket_gap_quantile=ticket_gap_quantile,
        cfg=cfg,
    )
    events = schedule_events(features, signal_state(features, cfg), cfg)
    return events, support_summary(events, "train")


def run(cfg: Config = Config()) -> dict[str, Any]:
    _validate_config(cfg)
    source_manifest = _validate_source_artifacts()

    train_source = load_source_prefix(end_exclusive=SELECTION_END)
    train_flow, train_ticket = source_matrices(train_source)
    train_base = base_feature_panel(train_flow, train_ticket, cfg)
    cells: list[dict[str, Any]] = []
    for top_flow_quantile in cfg.top_flow_quantiles:
        for ticket_gap_quantile in cfg.ticket_gap_quantiles:
            _, summary = _cell(
                train_base,
                top_flow_quantile=top_flow_quantile,
                ticket_gap_quantile=ticket_gap_quantile,
                cfg=cfg,
            )
            checks = train_support_checks(summary, cfg)
            cells.append(
                {
                    "top_flow_quantile": top_flow_quantile,
                    "ticket_gap_quantile": ticket_gap_quantile,
                    "train_support": summary,
                    "checks": checks,
                    "passes": all(checks.values()),
                }
            )
    selected = select_support_cell(cells)

    selected_train_events, _ = _cell(
        train_base,
        top_flow_quantile=float(selected["top_flow_quantile"]),
        ticket_gap_quantile=float(selected["ticket_gap_quantile"]),
        cfg=cfg,
    )
    full_source = load_source_prefix(end_exclusive=None)
    full_flow, full_ticket = source_matrices(full_source)
    full_base = base_feature_panel(full_flow, full_ticket, cfg)
    events, _ = _cell(
        full_base,
        top_flow_quantile=float(selected["top_flow_quantile"]),
        ticket_gap_quantile=float(selected["ticket_gap_quantile"]),
        cfg=cfg,
    )
    full_train = cast(pd.DataFrame, events.loc[events["split"].eq("train")])
    if (
        not full_train["entry_time"]
        .reset_index(drop=True)
        .equals(selected_train_events["entry_time"].reset_index(drop=True))
    ):
        raise RuntimeError("future TGR source incidence changed the train clock")

    support = {split: support_summary(events, split) for split in SPLITS}
    minimum_trades = protocol()["outcome_gate"]["minimum_trades"]
    stage_feasibility_checks = {
        f"{split}_minimum_trade_incidence": support[split]["events"]
        >= int(minimum_trades[split])
        for split in SPLITS
    }
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
    source_checks = {**cast(dict[str, bool], selected["checks"]), **novelty_checks}
    all_checks = {**source_checks, **stage_feasibility_checks}

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
            "top_flow_quantile": selected["top_flow_quantile"],
            "ticket_gap_quantile": selected["ticket_gap_quantile"],
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
        "source_support_passed": all(source_checks.values()),
        "support_passed": all(all_checks.values()),
        "advance_to_evaluator_freeze": all(all_checks.values()),
        "disposition": (
            "ADVANCE_TO_EVALUATOR_FREEZE"
            if all(all_checks.values())
            else "REJECT_SOURCE_INCIDENCE_NO_OUTCOME_OPEN"
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
                "outcomes_opened": report["outcomes_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
