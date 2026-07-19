"""Build outcome-blind PCBR-12 clocks and source-support evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.build_binance_aggtrade_microstructure import _write_gzip_csv  # noqa: E402
from training import preregister_premium_compression_breakout_relay as prereg  # noqa: E402


PREREG_COMMIT = "bc819d9db29a99bff3eb5b563665f25815a69936"
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
EXPECTED_PREREGISTRATION_SHA256 = (
    "df3a45cfaf36503793159cb969524803519ef95fd4421d19bfb58d266405bcce"
)
EXPECTED_PREREGISTRATION_MANIFEST_HASH = (
    "26cc00df2baa24a9df32c37e904155104c52928da89eda2dd9975a252c964dd2"
)
DEFAULT_CLOCK = Path("data/premium_compression_breakout_relay_clocks_2020_2026.csv.gz")
DEFAULT_CONTROLS_DIR = Path("data/premium_compression_breakout_relay_controls_2020_2026")
DEFAULT_RESULT = Path(
    "results/premium_compression_breakout_relay_support_2026-07-19.json"
)
COMPARATORS = {
    "PSR-30/6": {
        "path": Path("data/premium_snapback_recenter_clocks_2020_2026.csv.gz"),
        "sha256": "cb209ed35f9baa08cc2fb3dd5bd60b8e747b1408c09507b774ca275e0b2b2db6",
        "time_column": "entry_time",
        "near_minutes": 60,
        "near_limit": 0.25,
        "coverage_start": "2020-03-01T00:00:00Z",
        "coverage_end": "2026-07-01T00:00:00Z",
    },
    "PSI-2016": {
        "path": Path("data/premium_snapback_recenter_comparators_2020_2026/psi_2016.csv.gz"),
        "sha256": "4e413c1eb6d656f541734ba17b2a010aceb50b508005acadd8e2cb8bbbb7e03a",
        "time_column": "entry_time",
        "near_minutes": 60,
        "near_limit": 0.25,
        "coverage_start": "2020-03-01T00:00:00Z",
        "coverage_end": "2026-07-01T00:00:00Z",
    },
    "PSI-8640": {
        "path": Path("data/premium_snapback_recenter_comparators_2020_2026/psi_8640.csv.gz"),
        "sha256": "58fde45f300949b5c55e6e3025be6a9a4fe95451d3476e8ab0c03a83e3d81410",
        "time_column": "entry_time",
        "near_minutes": 60,
        "near_limit": 0.25,
        "coverage_start": "2020-03-01T00:00:00Z",
        "coverage_end": "2026-07-01T00:00:00Z",
    },
    "CMSR-36": {
        "path": Path("data/coinm_next_maturity_shock_relay_clocks_2020_2023.csv.gz"),
        "sha256": "e81450d4e76ffd0ce2ae96edf97106f2f4c473da233be0db18dc2530c8da8e87",
        "time_column": "entry_time",
        "near_minutes": 360,
        "near_limit": 0.25,
        "coverage_start": "2020-08-01T00:00:00Z",
        "coverage_end": "2024-01-01T00:00:00Z",
    },
}
SPLITS = {
    "train": (pd.Timestamp("2020-03-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")),
    "test": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")),
    "eval": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")),
}
SUBPERIODS = {
    "2020_mar_dec": (pd.Timestamp("2020-03-01", tz="UTC"), pd.Timestamp("2021-01-01", tz="UTC"), 30),
    "2021": (pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"), 50),
    "2022": (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC"), 50),
    "2023_h1": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2023-07-01", tz="UTC"), 20),
    "2023_h2": (pd.Timestamp("2023-07-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"), 20),
    "2024": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"), 40),
    "2025": (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC"), 40),
    "2026_h1": (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC"), 20),
}
SPLIT_SUBPERIODS = {
    "train": ("2020_mar_dec", "2021", "2022"),
    "test": ("2023_h1", "2023_h2"),
    "eval": ("2024", "2025", "2026_h1"),
}
CONTROL_NAMES = (
    "no_compression",
    "no_terminal_pin",
    "no_outside_cage",
    "direction_flip",
    "extra_latency_1h",
    "deterministic_random_side",
)
SOURCE_COLUMNS = (
    "date",
    "source_close_time",
    "feature_available_time",
    "source_valid",
    "premium_open",
    "premium_high",
    "premium_low",
    "premium_close",
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "context_start_time",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "context_range",
    "trigger_move",
    "trigger_efficiency",
    "terminal_location",
    "outside_distance",
)


@dataclass(frozen=True)
class Config:
    preregistration: str = str(PREREGISTRATION)
    source: str = prereg.SOURCE_PATH
    source_manifest: str = prereg.SOURCE_MANIFEST_PATH
    output_clock: str = str(DEFAULT_CLOCK)
    output_controls_dir: str = str(DEFAULT_CONTROLS_DIR)
    output_result: str = str(DEFAULT_RESULT)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_preregistration(cfg: Config) -> dict[str, Any]:
    if _sha256(cfg.preregistration) != EXPECTED_PREREGISTRATION_SHA256:
        raise ValueError("PCBR-12 preregistration bytes changed")
    report = json.loads(Path(cfg.preregistration).read_text(encoding="utf-8"))
    prereg.validate_manifest(report, verify_feature_source=False)
    if report.get("manifest_hash") != EXPECTED_PREREGISTRATION_MANIFEST_HASH:
        raise ValueError("PCBR-12 preregistration manifest changed")
    if report.get("outcomes_opened") is not False:
        raise ValueError("PCBR-12 preregistration opened outcomes")
    return report


def load_source(cfg: Config) -> pd.DataFrame:
    """Open premium-only source fields and no BTC execution source."""

    registration = _verify_preregistration(cfg)
    source_contract = registration["source_contract"]
    if _sha256(cfg.source_manifest) != source_contract["premium_manifest_sha256"]:
        raise ValueError("PCBR-12 premium manifest bytes changed")
    manifest = json.loads(Path(cfg.source_manifest).read_text(encoding="utf-8"))
    if manifest.get("protocol", {}).get("source_only") is not True:
        raise ValueError("PCBR-12 premium source is not source-only")
    if manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("PCBR-12 premium source manifest opened outcomes")
    if manifest.get("protocol", {}).get("btc_execution_prices_retained") is not False:
        raise ValueError("PCBR-12 premium source retained BTC prices")
    if _sha256(cfg.source) != source_contract["premium_sha256"]:
        raise ValueError("PCBR-12 premium source bytes changed")
    frame = pd.read_csv(
        cfg.source,
        compression="gzip",
        usecols=cast(Any, list(SOURCE_COLUMNS)),
    ).loc[:, list(SOURCE_COLUMNS)]
    for column in ("date", "source_close_time", "feature_available_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    expected = pd.date_range(
        "2020-01-01",
        "2026-07-01",
        freq="1min",
        inclusive="left",
        tz="UTC",
    )
    if not pd.DatetimeIndex(frame["date"]).equals(expected):
        raise ValueError("PCBR-12 premium source grid changed")
    valid = cast(pd.Series, frame["source_valid"]).astype(bool)
    premium = list(SOURCE_COLUMNS[4:])
    if not bool(frame.loc[valid, premium].notna().all().all()):
        raise ValueError("PCBR-12 valid source row has missing premium")
    if not bool(frame.loc[~valid, premium].isna().all().all()):
        raise ValueError("PCBR-12 invalid source row retains premium")
    return frame


def aggregate_5m(source: pd.DataFrame) -> pd.DataFrame:
    if len(source) % 5:
        raise ValueError("PCBR-12 minute source is not divisible into five-minute bars")
    minute_times = pd.DatetimeIndex(source["date"])
    if minute_times.empty:
        raise ValueError("PCBR-12 minute source is empty")
    first_minute = cast(pd.Timestamp, minute_times[0])
    if first_minute.minute % 5:
        raise ValueError("PCBR-12 minute source is not aligned to a five-minute boundary")
    expected_minutes = pd.date_range(
        first_minute, periods=len(source), freq="1min"
    )
    if not minute_times.equals(expected_minutes):
        raise ValueError("PCBR-12 minute source grid is not contiguous")
    groups = np.arange(len(source), dtype=np.int64) // 5
    valid = cast(pd.Series, source["source_valid"]).astype(bool)
    numeric = {
        column: cast(pd.Series, pd.to_numeric(source[column], errors="coerce")) * 10_000.0
        for column in SOURCE_COLUMNS[4:]
    }
    frame = pd.DataFrame(
        {
            "bar_open_time": source["date"].groupby(groups).first(),
            "feature_available_time": source["feature_available_time"].groupby(groups).last(),
            "valid": valid.astype(int).groupby(groups).sum().eq(5),
            "open": numeric["premium_open"].groupby(groups).first(),
            "high": numeric["premium_high"].groupby(groups).max(),
            "low": numeric["premium_low"].groupby(groups).min(),
            "close": numeric["premium_close"].groupby(groups).last(),
        }
    ).reset_index(drop=True)
    frame["decision_time"] = frame["bar_open_time"] + pd.Timedelta(minutes=5)
    invalid = ~cast(pd.Series, frame["valid"]).astype(bool)
    frame.loc[invalid, ["open", "high", "low", "close"]] = np.nan
    expected = pd.date_range(
        first_minute, periods=len(source) // 5, freq="5min"
    )
    if not pd.DatetimeIndex(frame["bar_open_time"]).equals(expected):
        raise ValueError("PCBR-12 five-minute grid changed")
    return frame


def derive_state(
    bars: pd.DataFrame, policy: prereg.Policy = prereg.Policy()
) -> pd.DataFrame:
    valid = cast(pd.Series, bars["valid"]).astype(bool)
    context = policy.context_bars_5m
    trigger = policy.trigger_bars_5m
    context_count = cast(
        pd.Series,
        valid.astype(int)
        .shift(trigger)
        .rolling(context, min_periods=context)
        .sum(),
    )
    trigger_count = cast(
        pd.Series,
        valid.astype(int).rolling(trigger, min_periods=trigger).sum(),
    )
    context_valid = context_count.eq(context)
    trigger_valid = trigger_count.eq(trigger)
    context_high = cast(pd.Series, bars["high"]).shift(trigger).rolling(
        context, min_periods=context
    ).max()
    context_low = cast(pd.Series, bars["low"]).shift(trigger).rolling(
        context, min_periods=context
    ).min()
    context_range = context_high - context_low
    trigger_open = cast(pd.Series, bars["open"]).shift(trigger - 1)
    trigger_close = cast(pd.Series, bars["close"])
    trigger_high = cast(pd.Series, bars["high"]).rolling(
        trigger, min_periods=trigger
    ).max()
    trigger_low = cast(pd.Series, bars["low"]).rolling(
        trigger, min_periods=trigger
    ).min()
    trigger_move = trigger_close - trigger_open
    trigger_path_range = (cast(pd.Series, bars["high"]) - bars["low"]).rolling(
        trigger, min_periods=trigger
    ).sum()
    trigger_efficiency = trigger_move.abs().div(
        trigger_path_range.where(trigger_path_range.gt(0.0))
    )
    trigger_span = (trigger_high - trigger_low).replace(0.0, np.nan)
    terminal_location = 2.0 * (trigger_close - trigger_low) / trigger_span - 1.0
    positive = trigger_move.gt(0.0)
    negative = trigger_move.lt(0.0)
    outside_distance = np.select(
        [positive, negative],
        [trigger_close - context_high, context_low - trigger_close],
        default=np.nan,
    )
    current_valid = context_valid & trigger_valid
    features = pd.DataFrame(
        {
            "context_range": context_range.where(current_valid),
            "trigger_move_abs": trigger_move.abs().where(current_valid),
            "trigger_efficiency": trigger_efficiency.where(current_valid),
        }
    )
    shift = policy.prior_nonoverlap_shift_bars_5m
    thresholds = pd.DataFrame(index=bars.index)
    for feature, quantile, name in (
        ("context_range", policy.compression_range_quantile, "prior_q25_context_range"),
        ("trigger_move_abs", policy.trigger_move_abs_quantile, "prior_q90_trigger_move_abs"),
        ("trigger_efficiency", policy.trigger_efficiency_quantile, "prior_q70_trigger_efficiency"),
    ):
        thresholds[name] = features[feature].shift(shift).rolling(
            policy.prior_window_bars_5m,
            min_periods=policy.prior_min_periods_5m,
        ).quantile(quantile)
    base = current_valid & np.isfinite(thresholds.to_numpy(float)).all(axis=1)
    compressed = context_range.le(thresholds["prior_q25_context_range"])
    displaced = trigger_move.abs().ge(thresholds["prior_q90_trigger_move_abs"])
    efficient = trigger_efficiency.ge(thresholds["prior_q70_trigger_efficiency"])
    outside = cast(pd.Series, pd.Series(outside_distance, index=bars.index)).gt(0.0)
    aligned_terminal = (
        np.sign(trigger_move) * terminal_location
    ).ge(policy.terminal_location_abs_min)
    families = {
        "primary_active": base & compressed & displaced & efficient & outside & aligned_terminal,
        "no_compression_active": base & displaced & efficient & outside & aligned_terminal,
        "no_terminal_pin_active": base & compressed & displaced & efficient & outside,
        "no_outside_cage_active": base & compressed & displaced & efficient & aligned_terminal,
    }
    output = pd.DataFrame(
        {
            "context_start_time": bars["decision_time"]
            - pd.Timedelta(minutes=5 * (context + trigger)),
            "decision_time": bars["decision_time"],
            "feature_available_time": bars["feature_available_time"],
            "side": np.sign(trigger_move).fillna(0.0).astype(int),
            "context_range": context_range,
            "trigger_move": trigger_move,
            "trigger_efficiency": trigger_efficiency,
            "terminal_location": terminal_location,
            "outside_distance": outside_distance,
            **{name: active & ~active.shift(1, fill_value=False) for name, active in families.items()},
        }
    )
    return output


def _deterministic_side(decision_time: pd.Timestamp) -> int:
    digest = hashlib.sha256(
        f"{prereg.CANDIDATE}|{decision_time.isoformat()}".encode("utf-8")
    ).digest()
    return 1 if digest[0] & 1 else -1


def build_clocks(
    state: pd.DataFrame,
    *,
    control: str = "primary",
    policy: prereg.Policy = prereg.Policy(),
) -> pd.DataFrame:
    active_column = {
        "primary": "primary_active",
        "no_compression": "no_compression_active",
        "no_terminal_pin": "no_terminal_pin_active",
        "no_outside_cage": "no_outside_cage_active",
        "direction_flip": "primary_active",
        "extra_latency_1h": "primary_active",
        "deterministic_random_side": "primary_active",
    }[control]
    rows: list[dict[str, Any]] = []
    for split, (start, end) in SPLITS.items():
        split_start = cast(pd.Timestamp, start)
        split_end = cast(pd.Timestamp, end)
        active = cast(pd.Series, state[active_column]).astype(bool)
        decision = cast(pd.Series, state["decision_time"])
        selected = state.loc[
            active & decision.ge(split_start) & decision.lt(split_end)
        ]
        previous_exit = split_start
        for row in selected.to_dict(orient="records"):
            decision_time = cast(pd.Timestamp, pd.Timestamp(row["decision_time"]))
            entry = cast(
                pd.Timestamp,
                decision_time
                + pd.Timedelta(minutes=5 * policy.entry_delay_bars_5m),
            )
            if control == "extra_latency_1h":
                entry = cast(pd.Timestamp, entry + pd.Timedelta(hours=1))
            exit_time = cast(
                pd.Timestamp,
                entry + pd.Timedelta(minutes=5 * policy.hold_bars_5m),
            )
            if (
                entry < split_start
                or exit_time >= split_end
                or entry < previous_exit
            ):
                continue
            feature_available = cast(
                pd.Timestamp, pd.Timestamp(row["feature_available_time"])
            )
            if feature_available >= entry:
                raise ValueError("PCBR-12 feature is unavailable at entry")
            side = int(row["side"])
            if control == "direction_flip":
                side *= -1
            elif control == "deterministic_random_side":
                side = _deterministic_side(decision_time)
            if side not in {-1, 1}:
                raise ValueError("PCBR-12 active clock has invalid side")
            rows.append(
                {
                    "candidate": prereg.CANDIDATE,
                    "control": control,
                    "split": split,
                    "context_start_time": row["context_start_time"],
                    "decision_time": decision_time,
                    "feature_available_time": feature_available,
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "side": side,
                    "context_range": float(row["context_range"]),
                    "trigger_move": float(row["trigger_move"]),
                    "trigger_efficiency": float(row["trigger_efficiency"]),
                    "terminal_location": float(row["terminal_location"]),
                    "outside_distance": float(row["outside_distance"]),
                }
            )
            previous_exit = exit_time
    return pd.DataFrame(rows, columns=cast(Any, list(CLOCK_COLUMNS)))


def _support_stats(clocks: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = clocks.loc[clocks["split"].eq(split)].copy()
    entry = pd.to_datetime(selected["entry_time"], utc=True)
    total = int(len(selected))
    months = entry.dt.strftime("%Y-%m").value_counts().sort_index()
    subperiods = {}
    for name in SPLIT_SUBPERIODS[split]:
        start, end, minimum = SUBPERIODS[name]
        count = int(entry.ge(start).mul(entry.lt(end)).sum())
        subperiods[name] = {"events": count, "minimum": minimum, "passed": count >= minimum}
    return {
        "events": total,
        "long": int(selected["side"].eq(1).sum()),
        "short": int(selected["side"].eq(-1).sum()),
        "long_share": float(selected["side"].eq(1).mean()) if total else 0.0,
        "short_share": float(selected["side"].eq(-1).mean()) if total else 0.0,
        "max_month_share": float(months.max() / total) if total else 1.0,
        "month_counts": {str(key): int(value) for key, value in months.items()},
        "subperiods": subperiods,
    }


def _support_checks(stats: dict[str, dict[str, Any]], registration: dict[str, Any]) -> dict[str, bool]:
    contract = registration["support_gate"]
    checks: dict[str, bool] = {}
    for split, row in stats.items():
        checks[f"{split}_events"] = row["events"] >= contract["minimum_events"][split]
        checks[f"{split}_side_balance"] = min(row["long_share"], row["short_share"]) >= contract["minimum_each_side_share"]
        checks[f"{split}_month_concentration"] = row["max_month_share"] <= contract["maximum_month_share"][split]
        checks[f"{split}_subperiods"] = all(item["passed"] for item in row["subperiods"].values())
    return checks


def _clock_times(path: Path, expected_hash: str, column: str) -> pd.DatetimeIndex:
    if _sha256(path) != expected_hash:
        raise ValueError(f"PCBR-12 comparator changed: {path}")
    frame = pd.read_csv(path, compression="gzip", usecols=cast(Any, [column]))
    return pd.DatetimeIndex(
        pd.to_datetime(frame[column], utc=True, errors="raise")
        .drop_duplicates()
        .sort_values()
    )


def _novelty(
    primary: pd.DatetimeIndex,
    other: pd.DatetimeIndex,
    near_minutes: int,
    *,
    coverage_start: str | None = None,
    coverage_end: str | None = None,
) -> dict[str, Any]:
    if (coverage_start is None) != (coverage_end is None):
        raise ValueError("PCBR-12 novelty coverage requires both boundaries")
    left_full = primary.dropna().drop_duplicates().sort_values()
    right_full = other.dropna().drop_duplicates().sort_values()
    left = left_full
    right = right_full
    if coverage_start is not None and coverage_end is not None:
        start = cast(pd.Timestamp, pd.Timestamp(coverage_start))
        end = cast(pd.Timestamp, pd.Timestamp(coverage_end))
        left = left[(left >= start) & (left < end)]
        right = right[(right >= start) & (right < end)]
    elif len(left) and len(right):
        start = min(cast(pd.Timestamp, left.min()), cast(pd.Timestamp, right.min()))
        end = max(cast(pd.Timestamp, left.max()), cast(pd.Timestamp, right.max()))
    else:
        start = None
        end = None
    if not len(left) or not len(right):
        return {
            "primary_full": int(len(left_full)),
            "other_full": int(len(right_full)),
            "shared_start": start.isoformat() if start is not None else None,
            "shared_end": end.isoformat() if end is not None else None,
            "primary_events": int(len(left)),
            "other_events": int(len(right)),
            "exact_intersection": 0,
            "exact_jaccard": 0.0,
            "near_minutes": near_minutes,
            "near_primary_events": 0,
            "near_primary_share": 0.0,
        }
    left_ns = set(left.view("int64").tolist())
    right_ns = set(right.view("int64").tolist())
    intersection = len(left_ns & right_ns)
    union = len(left_ns | right_ns)
    ordered = np.sort(np.fromiter(right_ns, dtype=np.int64))
    tolerance = int(pd.Timedelta(minutes=near_minutes).value)
    near = 0
    for value in left_ns:
        position = int(np.searchsorted(ordered, value))
        candidates = ordered[max(0, position - 1) : min(len(ordered), position + 1)]
        if len(candidates) and int(np.min(np.abs(candidates - value))) <= tolerance:
            near += 1
    shared_start = cast(pd.Timestamp, start)
    shared_end = cast(pd.Timestamp, end)
    return {
        "primary_full": int(len(left_full)),
        "other_full": int(len(right_full)),
        "shared_start": shared_start.isoformat(),
        "shared_end": shared_end.isoformat(),
        "primary_events": len(left_ns),
        "other_events": len(right_ns),
        "exact_intersection": intersection,
        "exact_jaccard": float(intersection / union) if union else 0.0,
        "near_minutes": near_minutes,
        "near_primary_events": near,
        "near_primary_share": float(near / len(left_ns)) if left_ns else 0.0,
    }


def build_result(
    cfg: Config,
    registration: dict[str, Any],
    source: pd.DataFrame,
    bars: pd.DataFrame,
    primary: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    support = {split: _support_stats(primary, split) for split in SPLITS}
    support_checks = _support_checks(support, registration)
    primary_times = pd.DatetimeIndex(pd.to_datetime(primary["entry_time"], utc=True))
    novelty = {}
    novelty_checks = {}
    for name, spec in COMPARATORS.items():
        row = _novelty(
            primary_times,
            _clock_times(spec["path"], spec["sha256"], spec["time_column"]),
            int(spec["near_minutes"]),
            coverage_start=str(spec["coverage_start"]),
            coverage_end=str(spec["coverage_end"]),
        )
        novelty[name] = row
        novelty_checks[f"{name}_exact"] = row["exact_jaccard"] <= 0.10
        novelty_checks[f"{name}_near"] = row["near_primary_share"] <= float(spec["near_limit"])
    all_checks = {**support_checks, **novelty_checks}
    core: dict[str, Any] = {
        "protocol_version": "premium_compression_breakout_relay_support_v1",
        "policy_id": prereg.CANDIDATE,
        "as_of_date": "2026-07-19",
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "preregistration": {
            "path": cfg.preregistration,
            "sha256": EXPECTED_PREREGISTRATION_SHA256,
            "commit": PREREG_COMMIT,
            "manifest_hash": registration["manifest_hash"],
        },
        "source": {
            "path": cfg.source,
            "sha256": registration["source_contract"]["premium_sha256"],
            "minute_rows": int(len(source)),
            "five_minute_rows": int(len(bars)),
            "btc_execution_rows_loaded": 0,
            "funding_rows_loaded": 0,
        },
        "clock": {
            "path": cfg.output_clock,
            "sha256": _sha256(cfg.output_clock),
            "rows": int(len(primary)),
        },
        "controls": {
            name: {
                "path": str(Path(cfg.output_controls_dir) / f"{name}.csv.gz"),
                "sha256": _sha256(Path(cfg.output_controls_dir) / f"{name}.csv.gz"),
                "rows": int(len(frame)),
            }
            for name, frame in controls.items()
        },
        "support": support,
        "support_checks": support_checks,
        "novelty": novelty,
        "novelty_checks": novelty_checks,
        "all_checks": all_checks,
        "support_passed": bool(all(all_checks.values())),
        "failed_checks": [name for name, passed in all_checks.items() if not passed],
        "advance_to_train_outcomes": bool(all(all_checks.values())),
        "sealed_outcome_windows": ["train_2020_2022", "test_2023", "eval_2024_2026"],
        "implementation_sha256": _sha256(__file__),
    }
    return {**core, "manifest_hash": _canonical_hash(core)}


def run(cfg: Config = Config()) -> dict[str, Any]:
    registration = _verify_preregistration(cfg)
    source = load_source(cfg)
    bars = aggregate_5m(source)
    state = derive_state(bars)
    primary = build_clocks(state)
    controls = {name: build_clocks(state, control=name) for name in CONTROL_NAMES}
    clock_path = Path(cfg.output_clock)
    clock_path.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(primary, clock_path)
    controls_dir = Path(cfg.output_controls_dir)
    controls_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in controls.items():
        _write_gzip_csv(frame, controls_dir / f"{name}.csv.gz")
    result = build_result(cfg, registration, source, bars, primary, controls)
    output = Path(cfg.output_result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default=Config.preregistration)
    parser.add_argument("--source", default=Config.source)
    parser.add_argument("--source-manifest", default=Config.source_manifest)
    parser.add_argument("--output-clock", default=Config.output_clock)
    parser.add_argument("--output-controls-dir", default=Config.output_controls_dir)
    parser.add_argument("--output-result", default=Config.output_result)
    result = run(Config(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                "support": result["support"],
                "novelty": result["novelty"],
                "support_passed": result["support_passed"],
                "failed_checks": result["failed_checks"],
                "manifest_hash": result["manifest_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
