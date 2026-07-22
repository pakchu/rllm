"""Build IVLIR-72 source-only clocks and support diagnostics without outcomes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import preregister_intrinsic_volume_latent_impact_relay as prereg


PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "6b94d4fe58d16d7cb62ee7b3888ca37710c8aadd75ce5ebe008ee230f7221be7"
)
DEFAULT_OUTPUT = Path(
    "results/intrinsic_volume_latent_impact_relay_support_2026-07-23.json"
)
DEFAULT_CLOCK = Path("data/intrinsic_volume_latent_impact_relay_clocks_2020_2023.csv.gz")
SOURCE_START = pd.Timestamp("2020-01-01T00:00:00Z")
SOURCE_END = pd.Timestamp("2024-01-01T00:00:00Z")
TRAIN_START = pd.Timestamp("2020-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2023-01-01T00:00:00Z")
SELECTION_START = TRAIN_END
SELECTION_END = SOURCE_END
BAR = pd.Timedelta(minutes=5)
CLOCK_COLUMNS = [
    "clock_name",
    "source_day",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="raise")
    if parsed.isna().any():
        raise RuntimeError("IVLIR source timestamp is NaT")
    return parsed


def validate_frozen_inputs() -> dict[str, Any]:
    prereg_hash = sha256_file(PREREGISTRATION)
    if prereg_hash != PREREGISTRATION_SHA256:
        raise RuntimeError("IVLIR preregistration file changed")
    frozen = json.loads(PREREGISTRATION.read_text())
    prereg.validate_manifest(frozen)
    if frozen["policy"] != asdict(prereg.Policy()):
        raise RuntimeError("IVLIR policy changed after preregistration")
    for path, expected in (
        (prereg.MARKET_MANIFEST, frozen["source_contract"]["market_manifest_sha256"]),
        (prereg.MARKET_SOURCE, frozen["source_contract"]["market_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"IVLIR frozen source changed: {path}")
    return frozen


def load_source(path: str | Path = prereg.MARKET_SOURCE) -> pd.DataFrame:
    required = prereg._core_manifest()["source_contract"]["required_columns"]
    frame = pd.read_csv(path, usecols=required)
    if list(frame.columns) != required:
        frame = frame.loc[:, required]
    frame["date"] = _utc(frame["date"])
    if len(frame) != 420_768:
        raise RuntimeError(f"IVLIR source row count changed: {len(frame)}")
    expected = pd.date_range(SOURCE_START, SOURCE_END, freq=BAR, inclusive="left")
    dates = pd.DatetimeIndex(frame["date"])
    if not dates.equals(expected):
        raise RuntimeError("IVLIR source is not the exact five-minute grid")
    numeric_columns = [column for column in required if column != "date"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise RuntimeError(f"IVLIR non-finite source column: {column}")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError("IVLIR source contains non-positive price")
    if not (
        (frame["high"] >= frame[["open", "close", "low"]].max(axis=1)).all()
        and (frame["low"] <= frame[["open", "close", "high"]].min(axis=1)).all()
    ):
        raise RuntimeError("IVLIR OHLC invariant failed")
    quote = frame["quote_asset_volume"].to_numpy(float)
    taker = frame["taker_buy_quote"].to_numpy(float)
    tolerance = np.maximum(1e-8, np.abs(quote) * 1e-10)
    if (quote < 0).any() or (taker < -tolerance).any() or (taker - quote > tolerance).any():
        raise RuntimeError("IVLIR quote/taker accounting invariant failed")
    return frame


def _linear_quantile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan")
    return float(np.quantile(array, quantile, method="linear"))


def _daily_completeness(frame: pd.DataFrame) -> pd.DataFrame:
    days = frame["date"].dt.floor("D")
    counts = frame.groupby(days, sort=True).size()
    if not len(counts) or not counts.eq(288).all():
        raise RuntimeError("IVLIR source does not contain complete UTC days")
    total = frame.groupby(days, sort=True)["quote_asset_volume"].sum().astype(float)
    if (total <= 0).any() or not np.isfinite(total.to_numpy()).all():
        raise RuntimeError("IVLIR daily quote volume is invalid")
    return pd.DataFrame({"day": total.index, "daily_quote_volume": total.to_numpy()})


def build_anchor_features(
    frame: pd.DataFrame,
    policy: prereg.Policy = prereg.Policy(),
    *,
    fixed_noon: bool = False,
) -> pd.DataFrame:
    """Build causal daily anchor features; no post-anchor value is computed."""
    daily = _daily_completeness(frame)
    daily["expected_quote_volume"] = (
        daily["daily_quote_volume"]
        .shift(1)
        .rolling(
            policy.utc_day_volume_lookback_days,
            min_periods=policy.utc_day_volume_min_days,
        )
        .median()
    )
    high_roll = frame["high"].rolling(
        policy.rolling_extrema_bars, min_periods=policy.rolling_extrema_bars
    ).max()
    low_roll = frame["low"].rolling(
        policy.rolling_extrema_bars, min_periods=policy.rolling_extrema_bars
    ).min()
    signed_quote = 2.0 * frame["taker_buy_quote"] - frame["quote_asset_volume"]
    rows: list[dict[str, Any]] = []
    for daily_row in daily.itertuples(index=False):
        expected_volume = float(daily_row.expected_quote_volume)
        if not np.isfinite(expected_volume) or expected_volume <= 0:
            continue
        day = pd.Timestamp(daily_row.day)
        start = int((day - SOURCE_START) / BAR)
        end = start + 288
        if start < 0 or end > len(frame):
            raise RuntimeError("IVLIR UTC-day slice escaped the source")
        day_quote = frame["quote_asset_volume"].iloc[start:end].to_numpy(float)
        cumulative_quote = np.cumsum(day_quote)
        target = policy.intrinsic_volume_fraction * expected_volume
        if fixed_noon:
            local_index = 11 * 12 + 11  # completed 11:55 UTC bar
            if cumulative_quote[local_index] < target:
                continue
        else:
            local_index = int(np.searchsorted(cumulative_quote, target, side="left"))
            if local_index >= len(cumulative_quote):
                continue
        anchor_index = start + local_index
        anchor_time = pd.Timestamp(frame.at[anchor_index, "date"])
        minute_of_day = anchor_time.hour * 60 + anchor_time.minute
        if minute_of_day > policy.latest_anchor_minute_utc:
            continue
        cumulative_volume = float(cumulative_quote[local_index])
        if cumulative_volume <= 0:
            continue
        cumulative_signed = float(signed_quote.iloc[start : anchor_index + 1].sum())
        cumulative_flow = cumulative_signed / cumulative_volume
        if not np.isfinite(cumulative_flow) or cumulative_flow == 0.0:
            continue
        day_open = float(frame.at[start, "open"])
        anchor_close = float(frame.at[anchor_index, "close"])
        anchor_return = float(np.log(anchor_close / day_open))
        side_sign = 1 if cumulative_flow > 0 else -1
        directional_return = float(side_sign * anchor_return)
        impact_ratio = directional_return / max(abs(cumulative_flow), 1e-12)
        rolling_high = float(high_roll.iloc[anchor_index])
        rolling_low = float(low_roll.iloc[anchor_index])
        width = rolling_high - rolling_low
        if not np.isfinite(width) or width <= 0:
            continue
        range_position = (anchor_close - rolling_low) / width
        if not np.isfinite(range_position):
            continue
        rows.append(
            {
                "source_day": day,
                "anchor_index": anchor_index,
                "anchor_time": anchor_time,
                "entry_time": anchor_time + policy.entry_delay_bars * BAR,
                "exit_time": anchor_time
                + (policy.entry_delay_bars + policy.hold_bars) * BAR,
                "side_sign": side_sign,
                "side": "LONG" if side_sign > 0 else "SHORT",
                "cumulative_flow": cumulative_flow,
                "anchor_return": anchor_return,
                "directional_return": directional_return,
                "impact_ratio": impact_ratio,
                "range_position": range_position,
                "anchor_minute_utc": minute_of_day,
                "target_quote_volume": target,
                "cumulative_quote_volume": cumulative_volume,
            }
        )
    return pd.DataFrame(rows)


def apply_causal_event_references(
    anchors: pd.DataFrame, policy: prereg.Policy = prereg.Policy()
) -> pd.DataFrame:
    if anchors.empty:
        return anchors.assign(
            reference_ready=pd.Series(dtype=bool),
            flow_threshold=pd.Series(dtype=float),
            impact_threshold=pd.Series(dtype=float),
            flow_pass=pd.Series(dtype=bool),
            alignment_pass=pd.Series(dtype=bool),
            impact_pass=pd.Series(dtype=bool),
            headroom_pass=pd.Series(dtype=bool),
            primary=pd.Series(dtype=bool),
            previous_anchor_side=pd.Series(dtype=str),
        )
    ordered = anchors.sort_values("anchor_time", kind="mergesort").reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for index, row in ordered.iterrows():
        start = max(0, index - policy.event_reference_days)
        prior = ordered.iloc[start:index]
        reference_ready = len(prior) >= policy.event_reference_min_days
        flow_threshold = float("nan")
        impact_threshold = float("nan")
        aligned_reference_count = 0
        if reference_ready:
            flow_threshold = _linear_quantile(
                prior["cumulative_flow"].abs(), policy.absolute_flow_quantile
            )
            aligned = prior.loc[
                prior["directional_return"].ge(0)
                & np.isfinite(prior["impact_ratio"]),
                "impact_ratio",
            ]
            aligned_reference_count = len(aligned)
            impact_threshold = _linear_quantile(
                aligned, policy.maximum_impact_quantile
            )
        flow_pass = bool(
            reference_ready
            and np.isfinite(flow_threshold)
            and abs(float(row["cumulative_flow"])) >= flow_threshold
        )
        alignment_pass = bool(float(row["directional_return"]) >= 0.0)
        impact_pass = bool(
            reference_ready
            and np.isfinite(impact_threshold)
            and float(row["impact_ratio"]) <= impact_threshold
        )
        if int(row["side_sign"]) > 0:
            headroom_pass = bool(
                float(row["range_position"]) <= policy.long_max_range_position
            )
        else:
            headroom_pass = bool(
                float(row["range_position"]) >= policy.short_min_range_position
            )
        records.append(
            {
                "reference_ready": reference_ready,
                "reference_count": len(prior),
                "aligned_reference_count": aligned_reference_count,
                "flow_threshold": flow_threshold,
                "impact_threshold": impact_threshold,
                "flow_pass": flow_pass,
                "alignment_pass": alignment_pass,
                "impact_pass": impact_pass,
                "headroom_pass": headroom_pass,
                "primary": bool(
                    flow_pass and alignment_pass and impact_pass and headroom_pass
                ),
                "previous_anchor_side": (
                    str(ordered.iloc[index - 1]["side"]) if index else ""
                ),
            }
        )
    return pd.concat([ordered, pd.DataFrame(records)], axis=1)


def _contained(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[
        frame["anchor_time"].ge(start)
        & frame["entry_time"].ge(start)
        & frame["exit_time"].le(end)
    ].copy()


def _greedy_schedule(frame: pd.DataFrame, clock_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=CLOCK_COLUMNS)
    selected: list[dict[str, Any]] = []
    previous_exit: pd.Timestamp | None = None
    for row in frame.sort_values("entry_time", kind="mergesort").itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        exit_time = pd.Timestamp(row.exit_time)
        if previous_exit is not None and entry < previous_exit:
            continue
        selected.append(
            {
                "clock_name": clock_name,
                "source_day": pd.Timestamp(row.source_day),
                "decision_time": pd.Timestamp(row.anchor_time),
                "entry_time": entry,
                "exit_time": exit_time,
                "side": str(row.side),
            }
        )
        previous_exit = exit_time
    return pd.DataFrame(selected, columns=CLOCK_COLUMNS)


def schedule_across_splits(frame: pd.DataFrame, clock_name: str) -> pd.DataFrame:
    parts = [
        _greedy_schedule(_contained(frame, TRAIN_START, TRAIN_END), clock_name),
        _greedy_schedule(
            _contained(frame, SELECTION_START, SELECTION_END), clock_name
        ),
    ]
    return pd.concat(parts, ignore_index=True).sort_values(
        ["entry_time", "clock_name"], kind="mergesort"
    ).reset_index(drop=True)


def _exact_clock_variant(
    primary: pd.DataFrame, clock_name: str, sides: Iterable[str]
) -> pd.DataFrame:
    variant = primary.copy()
    variant["clock_name"] = clock_name
    variant["side"] = list(sides)
    return variant.loc[:, CLOCK_COLUMNS]


def build_clocks(
    features: pd.DataFrame,
    fixed_noon_features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    primary_rows = features.loc[features["primary"]].copy()
    primary = schedule_across_splits(primary_rows, "primary")
    flow_only_rows = features.loc[features["reference_ready"] & features["flow_pass"]]
    no_under_rows = features.loc[
        features["flow_pass"]
        & features["alignment_pass"]
        & features["headroom_pass"]
    ]
    no_headroom_rows = features.loc[
        features["flow_pass"]
        & features["alignment_pass"]
        & features["impact_pass"]
    ]
    fixed_noon_rows = fixed_noon_features.loc[fixed_noon_features["primary"]]
    clocks = {
        "primary": primary,
        "flow_only": schedule_across_splits(flow_only_rows, "flow_only"),
        "no_under_response": schedule_across_splits(
            no_under_rows, "no_under_response"
        ),
        "no_headroom": schedule_across_splits(no_headroom_rows, "no_headroom"),
        "fixed_noon": schedule_across_splits(fixed_noon_rows, "fixed_noon"),
    }
    clocks["exact_side_flip"] = _exact_clock_variant(
        primary,
        "exact_side_flip",
        ["SHORT" if side == "LONG" else "LONG" for side in primary["side"]],
    )
    previous_by_entry = primary[["entry_time"]].merge(
        primary_rows[["entry_time", "previous_anchor_side"]],
        on="entry_time",
        how="left",
        validate="one_to_one",
    )
    stale_side = previous_by_entry["previous_anchor_side"].where(
        previous_by_entry["previous_anchor_side"].isin(["LONG", "SHORT"]),
        primary["side"].reset_index(drop=True),
    )
    clocks["stale_previous_anchor_side"] = _exact_clock_variant(
        primary, "stale_previous_anchor_side", stale_side
    )
    seed = int.from_bytes(
        hashlib.sha256(b"IVLIR-72|side").digest()[:8], "big", signed=False
    )
    rng = np.random.default_rng(seed)
    random_side = np.where(rng.integers(0, 2, len(primary)) == 0, "SHORT", "LONG")
    clocks["deterministic_random_side"] = _exact_clock_variant(
        primary, "deterministic_random_side", random_side
    )
    return clocks


def _window(clock: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return clock.loc[clock["entry_time"].ge(start) & clock["exit_time"].le(end)]


def _longest_run(sides: Iterable[str]) -> int:
    best = 0
    current = 0
    previous = None
    for side in sides:
        if side == previous:
            current += 1
        else:
            current = 1
            previous = side
        best = max(best, current)
    return best


def clock_stats(clock: pd.DataFrame) -> dict[str, Any]:
    ordered = clock.sort_values("entry_time", kind="mergesort")
    total = len(ordered)
    side_counts = ordered["side"].value_counts().to_dict()
    calendar_time = ordered["entry_time"].dt.tz_convert(None)
    month_counts = calendar_time.dt.to_period("M").astype(str).value_counts()
    quarter_counts = calendar_time.dt.to_period("Q").astype(str).value_counts()
    return {
        "events": total,
        "long": int(side_counts.get("LONG", 0)),
        "short": int(side_counts.get("SHORT", 0)),
        "long_share": float(side_counts.get("LONG", 0) / total) if total else None,
        "short_share": float(side_counts.get("SHORT", 0) / total) if total else None,
        "active_months": int(len(month_counts)),
        "maximum_month_share": float(month_counts.max() / total) if total else None,
        "maximum_quarter_share": float(quarter_counts.max() / total) if total else None,
        "maximum_same_side_run": _longest_run(ordered["side"]),
    }


def _support_report(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    fixed_noon_features: pd.DataFrame,
    clocks: dict[str, pd.DataFrame],
    clock_path: Path,
) -> dict[str, Any]:
    policy = prereg.Policy()
    primary = clocks["primary"]
    train = _window(primary, TRAIN_START, TRAIN_END)
    selection = _window(primary, SELECTION_START, SELECTION_END)
    windows = {
        "all": clock_stats(primary),
        "train_2020_2022": clock_stats(train),
        "2020": clock_stats(_window(primary, TRAIN_START, pd.Timestamp("2021-01-01T00:00:00Z"))),
        "2021": clock_stats(_window(primary, pd.Timestamp("2021-01-01T00:00:00Z"), pd.Timestamp("2022-01-01T00:00:00Z"))),
        "2022": clock_stats(_window(primary, pd.Timestamp("2022-01-01T00:00:00Z"), TRAIN_END)),
        "selection_2023": clock_stats(selection),
        "selection_2023_h1": clock_stats(_window(primary, SELECTION_START, pd.Timestamp("2023-07-01T00:00:00Z"))),
        "selection_2023_h2": clock_stats(_window(primary, pd.Timestamp("2023-07-01T00:00:00Z"), SELECTION_END)),
    }
    low_share, high_share = prereg._core_manifest()["source_support_gate"][
        "each_side_share_range_all_train_selection"
    ]

    def side_ok(stats: dict[str, Any]) -> bool:
        return bool(
            stats["events"]
            and stats["long_share"] is not None
            and low_share <= stats["long_share"] <= high_share
            and low_share <= stats["short_share"] <= high_share
        )

    all_stats = windows["all"]
    checks = {
        "source_exact_grid": len(frame) == 420_768,
        "train_events_min": windows["train_2020_2022"]["events"]
        >= prereg._core_manifest()["source_support_gate"]["train_events_min"],
        "each_train_year_events_min": all(
            windows[str(year)]["events"]
            >= prereg._core_manifest()["source_support_gate"][
                "each_train_year_events_min"
            ]
            for year in (2020, 2021, 2022)
        ),
        "selection_events_min": windows["selection_2023"]["events"]
        >= prereg._core_manifest()["source_support_gate"]["selection_events_min"],
        "each_selection_half_events_min": all(
            windows[name]["events"]
            >= prereg._core_manifest()["source_support_gate"][
                "each_selection_half_events_min"
            ]
            for name in ("selection_2023_h1", "selection_2023_h2")
        ),
        "side_share_all_train_selection": all(
            side_ok(windows[name])
            for name in ("all", "train_2020_2022", "selection_2023")
        ),
        "active_months_min": all_stats["active_months"]
        >= prereg._core_manifest()["source_support_gate"]["active_months_min"],
        "maximum_single_month_share": bool(
            all_stats["maximum_month_share"] is not None
            and all_stats["maximum_month_share"]
            <= prereg._core_manifest()["source_support_gate"][
                "maximum_single_month_share"
            ]
        ),
        "maximum_single_quarter_share": bool(
            all_stats["maximum_quarter_share"] is not None
            and all_stats["maximum_quarter_share"]
            <= prereg._core_manifest()["source_support_gate"][
                "maximum_single_quarter_share"
            ]
        ),
        "maximum_same_side_run": all_stats["maximum_same_side_run"]
        <= prereg._core_manifest()["source_support_gate"]["maximum_same_side_run"],
        "clock_has_no_market_value_or_outcome_columns": list(
            pd.concat(clocks.values(), ignore_index=True).columns
        )
        == CLOCK_COLUMNS,
    }
    failed = [name for name, passed in checks.items() if not passed]
    all_clocks = pd.concat(clocks.values(), ignore_index=True).sort_values(
        ["entry_time", "clock_name"], kind="mergesort"
    )
    return {
        "protocol_version": "intrinsic_volume_latent_impact_relay_support_v1",
        "policy_id": policy.policy_id,
        "outcomes_opened": False,
        "post_entry_return_computed": False,
        "funding_loaded": False,
        "source_incidence_opened": True,
        "support_passed": not failed,
        "authorized_next_stage": "freeze_strict_evaluator" if not failed else None,
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": sha256_file(PREREGISTRATION),
            "manifest_hash": json.loads(PREREGISTRATION.read_text())["manifest_hash"],
        },
        "source": {
            "path": prereg.MARKET_SOURCE,
            "sha256": sha256_file(prereg.MARKET_SOURCE),
            "rows": len(frame),
            "first_timestamp": frame["date"].iloc[0].isoformat(),
            "last_timestamp": frame["date"].iloc[-1].isoformat(),
            "complete_utc_days": int(frame["date"].dt.floor("D").nunique()),
        },
        "feature_funnel": {
            "first_passage_anchors": len(features),
            "reference_ready": int(features["reference_ready"].sum()),
            "flow_pass": int(features["flow_pass"].sum()),
            "alignment_pass": int(features["alignment_pass"].sum()),
            "impact_pass": int(features["impact_pass"].sum()),
            "headroom_pass": int(features["headroom_pass"].sum()),
            "raw_primary": int(features["primary"].sum()),
            "fixed_noon_anchors": len(fixed_noon_features),
            "fixed_noon_raw_primary": int(fixed_noon_features["primary"].sum()),
        },
        "windows": windows,
        "controls": {
            name: clock_stats(clock) for name, clock in clocks.items()
        },
        "support_checks": checks,
        "failed_checks": failed,
        "clock": {
            "path": str(clock_path),
            "rows": len(all_clocks),
            "columns": CLOCK_COLUMNS,
            "sha256": sha256_file(clock_path),
        },
        "leakage_guard": {
            "daily_volume_reference_shifted_one_complete_day": True,
            "event_reference_strictly_prior": True,
            "current_anchor_excluded_from_quantiles": True,
            "signals_use_only_day_start_through_completed_anchor": True,
            "no_post_entry_price_access": True,
            "no_return_or_pnl_field_written": True,
            "llm_or_model_trained": False,
        },
    }


def write_clock(path: str | Path, clocks: dict[str, pd.DataFrame]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(clocks.values(), ignore_index=True).sort_values(
        ["entry_time", "clock_name"], kind="mergesort"
    )
    if list(combined.columns) != CLOCK_COLUMNS:
        raise RuntimeError("IVLIR clock schema changed")
    csv_bytes = combined.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%SZ").encode()
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
            zipped.write(csv_bytes)


def build_support(
    *,
    source: str | Path = prereg.MARKET_SOURCE,
    output: str | Path = DEFAULT_OUTPUT,
    clock_output: str | Path = DEFAULT_CLOCK,
) -> dict[str, Any]:
    validate_frozen_inputs()
    if str(source) != prereg.MARKET_SOURCE:
        raise RuntimeError("IVLIR support must use the frozen source path")
    frame = load_source(source)
    features = apply_causal_event_references(build_anchor_features(frame))
    fixed_noon_features = apply_causal_event_references(
        build_anchor_features(frame, fixed_noon=True)
    )
    clocks = build_clocks(features, fixed_noon_features)
    clock_path = Path(clock_output)
    write_clock(clock_path, clocks)
    report = _support_report(frame, features, fixed_noon_features, clocks, clock_path)
    core = {key: value for key, value in report.items() if key != "created_at"}
    report["report_manifest_hash"] = canonical_hash(core)
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=prereg.MARKET_SOURCE)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--clock-output", default=str(DEFAULT_CLOCK))
    args = parser.parse_args()
    report = build_support(
        source=args.source, output=args.output, clock_output=args.clock_output
    )
    print(
        json.dumps(
            {
                "policy_id": report["policy_id"],
                "support_passed": report["support_passed"],
                "failed_checks": report["failed_checks"],
                "windows": report["windows"],
                "feature_funnel": report["feature_funnel"],
                "clock": report["clock"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
