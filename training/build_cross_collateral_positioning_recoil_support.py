"""Build source-only CCPR-1 features, controls, and frozen event clocks.

This module may parse only the preregistration and the audited cross-collateral
positioning panel.  It must not open executable OHLC, funding, future returns,
labels, portfolio PnL, CAGR, or drawdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training import preregister_cross_collateral_positioning_recoil as prereg


PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "a2ae7b4ee86cd98c409966d4c35227c62989046758d15620eec2ad8cb05dc7fd"
)
PREREGISTRATION_COMMIT = "d546b84"
DEFAULT_OUTPUT = Path(
    "results/cross_collateral_positioning_recoil_support_2026-07-17.json"
)
DEFAULT_CLOCKS = Path(
    "results/cross_collateral_positioning_recoil_clocks_2026-07-17.csv"
)

REQUIRED_COLUMNS = (
    "date",
    "um_symbol",
    "um_sum_open_interest_value",
    "um_sum_taker_long_short_vol_ratio",
    "cm_symbol",
    "cm_sum_open_interest",
    "cm_sum_taker_long_short_vol_ratio",
    "source_complete",
)
CONTROL_ORDER = ("primary", "oi_only", "taker_only", "um_only", "cm_only")
TimeBounds = tuple[Any, Any]
TRAIN: TimeBounds = (
    pd.Timestamp("2021-07-08", tz="UTC"),
    pd.Timestamp("2023-01-01", tz="UTC"),
)
YEAR_2021: TimeBounds = (
    pd.Timestamp("2021-07-08", tz="UTC"),
    pd.Timestamp("2022-01-01", tz="UTC"),
)
YEAR_2022: TimeBounds = (
    pd.Timestamp("2022-01-01", tz="UTC"),
    pd.Timestamp("2023-01-01", tz="UTC"),
)
YEAR_2023: TimeBounds = (
    pd.Timestamp("2023-01-01", tz="UTC"),
    pd.Timestamp("2024-01-01", tz="UTC"),
)
H1_2023: TimeBounds = (
    pd.Timestamp("2023-01-01", tz="UTC"),
    pd.Timestamp("2023-07-01", tz="UTC"),
)
H2_2023: TimeBounds = (
    pd.Timestamp("2023-07-01", tz="UTC"),
    pd.Timestamp("2024-01-01", tz="UTC"),
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _series(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, frame[name])


def _isoformat(value: Any) -> str:
    timestamp = cast(pd.Timestamp, pd.Timestamp(value))
    return timestamp.isoformat()


def load_preregistration(
    path: str | Path = PREREGISTRATION,
) -> dict[str, Any]:
    if _sha256(path) != PREREGISTRATION_SHA256:
        raise ValueError("CCPR-1 preregistration artifact hash mismatch")
    manifest = _load_json(path)
    prereg.validate_manifest(manifest)
    if manifest["outcomes_opened"] is not False:
        raise ValueError("CCPR-1 support cannot open outcomes")
    return manifest


def load_source(manifest: dict[str, Any]) -> pd.DataFrame:
    source = manifest["source_contract"]
    path = Path(source["positioning"])
    if _sha256(path) != source["positioning_sha256"]:
        raise ValueError("CCPR-1 positioning source hash mismatch")
    frame = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("CCPR-1 positioning schema mismatch")
    frame = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(_series(frame, "date"), utc=True, errors="raise")
    if bool(_series(frame, "date").duplicated().any()):
        raise ValueError("CCPR-1 duplicate positioning timestamp")
    frame = frame.sort_values("date").set_index("date")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("CCPR-1 positioning clock is not monotonic")
    if (
        len(frame) < 2
        or not frame.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all()
    ):
        raise ValueError("CCPR-1 positioning grid is not exact 5m")
    observed_um = _series(frame, "um_symbol").dropna()
    observed_cm = _series(frame, "cm_symbol").dropna()
    if not bool(observed_um.eq(manifest["policy"]["usd_m_symbol"]).all()):
        raise ValueError("CCPR-1 unexpected USD-M symbol")
    if not bool(observed_cm.eq(manifest["policy"]["coin_m_symbol"]).all()):
        raise ValueError("CCPR-1 unexpected COIN-M symbol")
    numeric = frame[
        [
            "um_sum_open_interest_value",
            "um_sum_taker_long_short_vol_ratio",
            "cm_sum_open_interest",
            "cm_sum_taker_long_short_vol_ratio",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    finite_positive = pd.Series(
        np.isfinite(numeric.to_numpy()).all(axis=1)
        & numeric.gt(0.0).all(axis=1).to_numpy(),
        index=frame.index,
    )
    symbols_complete = _series(frame, "um_symbol").eq(
        manifest["policy"]["usd_m_symbol"]
    ) & _series(frame, "cm_symbol").eq(manifest["policy"]["coin_m_symbol"])
    frame["source_complete"] = (
        _series(frame, "source_complete").astype(bool)
        & finite_positive
        & symbols_complete
    )
    frame[numeric.columns] = numeric
    return frame


def prior_midrank(values: pd.Series, valid: pd.Series, window: int) -> pd.Series:
    """Rank current magnitude against exact prior anchors, excluding current."""
    array = values.to_numpy(dtype=float)
    validity = valid.to_numpy(dtype=bool)
    output = np.full(len(array), np.nan, dtype=float)
    for position in range(window, len(array)):
        prior = array[position - window : position]
        if (
            not validity[position]
            or not validity[position - window : position].all()
            or not np.isfinite(array[position])
            or not np.isfinite(prior).all()
        ):
            continue
        current = array[position]
        output[position] = (
            float(np.count_nonzero(prior < current))
            + 0.5 * float(np.count_nonzero(prior == current))
        ) / float(window)
    return pd.Series(output, index=values.index, dtype=float)


def build_features(source: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    policy = manifest["policy"]
    complete = _series(source, "source_complete").astype(bool)
    oi_bars = int(policy["oi_change_bars"])
    taker_bars = int(policy["taker_median_bars"])
    rank_anchors = int(policy["prior_rank_hourly_anchors"])

    um_oi = cast(
        pd.Series,
        np.log(_series(source, "um_sum_open_interest_value").where(complete)),
    )
    cm_oi = cast(
        pd.Series, np.log(_series(source, "cm_sum_open_interest").where(complete))
    )
    um_taker = cast(
        pd.Series,
        np.log(_series(source, "um_sum_taker_long_short_vol_ratio").where(complete)),
    )
    cm_taker = cast(
        pd.Series,
        np.log(_series(source, "cm_sum_taker_long_short_vol_ratio").where(complete)),
    )

    um_oi_change = um_oi.diff(oi_bars)
    cm_oi_change = cm_oi.diff(oi_bars)
    oi_rotation = um_oi_change - cm_oi_change
    um_taker_median = um_taker.rolling(taker_bars, min_periods=taker_bars).median()
    cm_taker_median = cm_taker.rolling(taker_bars, min_periods=taker_bars).median()
    taker_gap = (
        (um_taker - cm_taker).rolling(taker_bars, min_periods=taker_bars).median()
    )

    source_index = pd.DatetimeIndex(source.index)
    anchors = np.fromiter(
        (
            pd.Timestamp(value).minute == int(policy["anchor_minute"])
            for value in source_index
        ),
        dtype=bool,
        count=len(source_index),
    )
    feature = pd.DataFrame(index=source_index[anchors])
    valid_history = cast(
        pd.Series,
        complete.rolling(oi_bars + 1, min_periods=oi_bars + 1).sum(),
    )
    feature["anchor_valid"] = valid_history.eq(oi_bars + 1).loc[feature.index]
    feature["oi_rotation"] = oi_rotation.loc[feature.index]
    feature["taker_gap"] = taker_gap.loc[feature.index]
    feature["um_oi_change"] = um_oi_change.loc[feature.index]
    feature["cm_oi_change"] = cm_oi_change.loc[feature.index]
    feature["um_taker"] = um_taker_median.loc[feature.index]
    feature["cm_taker"] = cm_taker_median.loc[feature.index]

    for name in (
        "oi_rotation",
        "taker_gap",
        "um_oi_change",
        "cm_oi_change",
        "um_taker",
        "cm_taker",
    ):
        feature[f"{name}_rank"] = prior_midrank(
            _series(feature, name).abs(),
            _series(feature, "anchor_valid"),
            rank_anchors,
        )
    feature["feature_complete"] = (
        feature[
            [
                "oi_rotation_rank",
                "taker_gap_rank",
                "um_oi_change_rank",
                "cm_oi_change_rank",
                "um_taker_rank",
                "cm_taker_rank",
            ]
        ]
        .notna()
        .all(axis=1)
    )
    return feature


def _nonzero_sign(series: pd.Series) -> pd.Series:
    return pd.Series(np.sign(series.to_numpy(dtype=float)), index=series.index)


def build_flags(
    feature: pd.DataFrame, q: float, taker_floor: float
) -> dict[str, pd.Series]:
    rotation_sign = _nonzero_sign(_series(feature, "oi_rotation"))
    taker_sign = _nonzero_sign(_series(feature, "taker_gap"))
    um_oi_sign = _nonzero_sign(_series(feature, "um_oi_change"))
    cm_oi_sign = _nonzero_sign(_series(feature, "cm_oi_change"))
    um_taker_sign = _nonzero_sign(_series(feature, "um_taker"))
    cm_taker_sign = _nonzero_sign(_series(feature, "cm_taker"))
    valid = _series(feature, "feature_complete")
    return {
        "primary": (
            valid
            & _series(feature, "oi_rotation_rank").ge(q)
            & _series(feature, "taker_gap_rank").ge(taker_floor)
            & rotation_sign.eq(taker_sign)
            & rotation_sign.ne(0.0)
        ),
        "oi_only": valid
        & _series(feature, "oi_rotation_rank").ge(q)
        & rotation_sign.ne(0.0),
        "taker_only": valid
        & _series(feature, "taker_gap_rank").ge(taker_floor)
        & taker_sign.ne(0.0),
        "um_only": (
            valid
            & _series(feature, "um_oi_change_rank").ge(q)
            & _series(feature, "um_taker_rank").ge(taker_floor)
            & um_oi_sign.eq(um_taker_sign)
            & um_oi_sign.ne(0.0)
        ),
        "cm_only": (
            valid
            & _series(feature, "cm_oi_change_rank").ge(q)
            & _series(feature, "cm_taker_rank").ge(taker_floor)
            & cm_oi_sign.eq(cm_taker_sign)
            & cm_oi_sign.ne(0.0)
        ),
    }


def false_to_true(flag: pd.Series) -> pd.Series:
    current = flag.fillna(False).astype(bool)
    return current & ~current.shift(1, fill_value=False)


def _sides(feature: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "primary": -_nonzero_sign(_series(feature, "taker_gap")),
        "oi_only": -_nonzero_sign(_series(feature, "oi_rotation")),
        "taker_only": -_nonzero_sign(_series(feature, "taker_gap")),
        "um_only": -_nonzero_sign(_series(feature, "um_taker")),
        "cm_only": -_nonzero_sign(_series(feature, "cm_taker")),
    }


def build_clocks(feature: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    policy = manifest["policy"]
    rows: list[pd.DataFrame] = []
    side_series = _sides(feature)
    for q in policy["rotation_quantiles"]:
        flags = build_flags(feature, float(q), float(policy["taker_rank_floor"]))
        for control in CONTROL_ORDER:
            episodes = false_to_true(flags[control])
            selected = feature.loc[episodes].copy()
            selected.insert(0, "signal_time", selected.index)
            selected.insert(
                1,
                "entry_time",
                selected.index
                + pd.Timedelta(minutes=5 * int(policy["execution_delay_bars"])),
            )
            selected.insert(2, "q", float(q))
            selected.insert(3, "control", control)
            selected.insert(
                4,
                "side",
                side_series[control].loc[selected.index].astype(int).to_numpy(),
            )
            rows.append(selected.reset_index(drop=True))
    clocks = pd.concat(rows, ignore_index=True)
    clocks["signal_time"] = pd.to_datetime(clocks["signal_time"], utc=True)
    clocks["entry_time"] = pd.to_datetime(clocks["entry_time"], utc=True)
    clocks = clocks.sort_values(
        ["q", "control", "signal_time"], kind="stable"
    ).reset_index(drop=True)
    if not bool(_series(clocks, "side").isin((-1, 1)).all()):
        raise ValueError("CCPR-1 emitted a zero or invalid side")
    return clocks


def _mask(clocks: pd.DataFrame, bounds: TimeBounds) -> pd.Series:
    start, end = bounds
    return _series(clocks, "signal_time").ge(start) & _series(clocks, "signal_time").lt(
        end
    )


def _jaccard(left: pd.Series, right: pd.Series) -> float:
    left_set = set(pd.to_datetime(left, utc=True))
    right_set = set(pd.to_datetime(right, utc=True))
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 0.0


def _primary_summary(clocks: pd.DataFrame) -> dict[str, Any]:
    counts = {
        "train": int(_mask(clocks, TRAIN).sum()),
        "2021_partial": int(_mask(clocks, YEAR_2021).sum()),
        "2022": int(_mask(clocks, YEAR_2022).sum()),
        "2023": int(_mask(clocks, YEAR_2023).sum()),
        "2023_H1": int(_mask(clocks, H1_2023).sum()),
        "2023_H2": int(_mask(clocks, H2_2023).sum()),
    }
    split_details: dict[str, Any] = {}
    for name, bounds in (("train", TRAIN), ("2023", YEAR_2023)):
        subset = clocks.loc[_mask(clocks, bounds)]
        side_counts = subset["side"].value_counts().to_dict()
        total = len(subset)
        monthly = subset["signal_time"].dt.strftime("%Y-%m").value_counts()
        split_details[name] = {
            "side_counts": {
                "long": int(side_counts.get(1, 0)),
                "short": int(side_counts.get(-1, 0)),
            },
            "minimum_side_share": (
                float(min(side_counts.get(1, 0), side_counts.get(-1, 0)) / total)
                if total
                else 0.0
            ),
            "maximum_single_month_share": (
                float(monthly.max() / total) if total else 0.0
            ),
        }
    return {"counts": counts, "split_details": split_details}


def _control_jaccards(
    all_q: pd.DataFrame, primary: pd.DataFrame
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for control in CONTROL_ORDER[1:]:
        component = all_q.loc[all_q["control"].eq(control)]
        output[control] = {
            "train": _jaccard(
                primary.loc[_mask(primary, TRAIN), "signal_time"],
                component.loc[_mask(component, TRAIN), "signal_time"],
            ),
            "2023": _jaccard(
                primary.loc[_mask(primary, YEAR_2023), "signal_time"],
                component.loc[_mask(component, YEAR_2023), "signal_time"],
            ),
        }
    return output


def evaluate_q(
    clocks: pd.DataFrame, q: float, manifest: dict[str, Any]
) -> dict[str, Any]:
    all_q = clocks.loc[np.isclose(clocks["q"], q)].copy()
    primary = all_q.loc[all_q["control"].eq("primary")].copy()
    summary = _primary_summary(primary)
    jaccards = _control_jaccards(all_q, primary)
    support = manifest["support_calibration"]
    counts = summary["counts"]
    details = summary["split_details"]
    floors = {
        "train_episodes": counts["train"] >= support["minimum_train_episodes"],
        "2021_partial_episodes": counts["2021_partial"]
        >= support["minimum_2021_partial_episodes"],
        "2022_episodes": counts["2022"] >= support["minimum_2022_episodes"],
        "2023_episodes": counts["2023"] >= support["minimum_2023_episodes"],
        "2023_halves": min(counts["2023_H1"], counts["2023_H2"])
        >= support["minimum_each_2023_half"],
        "train_side_balance": details["train"]["minimum_side_share"]
        >= support["minimum_each_side_share"],
        "2023_side_balance": details["2023"]["minimum_side_share"]
        >= support["minimum_each_side_share"],
        "train_month_concentration": details["train"]["maximum_single_month_share"]
        <= support["maximum_single_month_share"],
        "2023_month_concentration": details["2023"]["maximum_single_month_share"]
        <= support["maximum_single_month_share"],
    }
    for control, maximum in support["maximum_signal_jaccard"].items():
        floors[f"{control}_jaccard_train"] = jaccards[control]["train"] <= maximum
        floors[f"{control}_jaccard_2023"] = jaccards[control]["2023"] <= maximum
    return {
        "q": q,
        **summary,
        "control_signal_jaccard": jaccards,
        "floors": floors,
        "passed": bool(all(floors.values())),
    }


def _write_clocks(clocks: pd.DataFrame, path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serial = clocks.copy()
    for column in ("signal_time", "entry_time"):
        serial[column] = serial[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    serial.to_csv(output, index=False, lineterminator="\n", float_format="%.12g")
    return _sha256(output)


def build(
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
    clocks_path: str | Path = DEFAULT_CLOCKS,
) -> dict[str, Any]:
    manifest = load_preregistration()
    source = load_source(manifest)
    feature = build_features(source, manifest)
    clocks = build_clocks(feature, manifest)
    evaluations = [
        evaluate_q(clocks, float(q), manifest)
        for q in manifest["policy"]["rotation_quantiles"]
    ]
    passing = [item for item in evaluations if item["passed"]]
    selected_q = max((item["q"] for item in passing), default=None)
    clocks_hash = _write_clocks(clocks, clocks_path)
    feature_complete = _series(feature, "feature_complete").astype(bool)
    complete_feature_index = pd.DatetimeIndex(feature.index[feature_complete])
    source_index = pd.DatetimeIndex(source.index)
    core: dict[str, Any] = {
        "protocol_version": "cross_collateral_positioning_recoil_support_v1",
        "policy_id": manifest["policy"]["policy_id"],
        "as_of_date": manifest["as_of_date"],
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "preregistration": {
            "path": str(PREREGISTRATION),
            "sha256": PREREGISTRATION_SHA256,
            "commit": PREREGISTRATION_COMMIT,
            "manifest_hash": manifest["manifest_hash"],
        },
        "source": {
            "path": manifest["source_contract"]["positioning"],
            "sha256": manifest["source_contract"]["positioning_sha256"],
            "rows": len(source),
            "source_complete_rows": int(_series(source, "source_complete").sum()),
            "first_timestamp": _isoformat(source_index[0]),
            "last_timestamp": _isoformat(source_index[-1]),
        },
        "feature_support": {
            "hourly_anchors": len(feature),
            "feature_complete_anchors": int(feature_complete.sum()),
            "first_feature_complete": (
                _isoformat(complete_feature_index[0])
                if len(complete_feature_index)
                else None
            ),
            "last_feature_complete": (
                _isoformat(complete_feature_index[-1])
                if len(complete_feature_index)
                else None
            ),
        },
        "q_evaluations": evaluations,
        "selection_rule": manifest["support_calibration"]["selection_rule"],
        "selected_q": selected_q,
        "support_passed": selected_q is not None,
        "advance_to_stage1_outcomes": selected_q is not None,
        "clocks": {
            "path": str(clocks_path),
            "sha256": clocks_hash,
            "rows": len(clocks),
            "controls": list(CONTROL_ORDER),
        },
        "sealed": {
            "stage1_execution_window": ["2021-07-08", "2023-01-01"],
            "stage2_execution_window": ["2023-01-01", "2024-01-01"],
            "stage1_outcomes_opened": False,
            "stage2_outcomes_opened": False,
            "market_rows_parsed": 0,
            "funding_rows_parsed": 0,
            "simulations_run": 0,
        },
    }
    report = {**core, "manifest_hash": _canonical_hash(core)}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--clocks", default=str(DEFAULT_CLOCKS))
    args = parser.parse_args()
    report = build(output_path=args.output, clocks_path=args.clocks)
    print(json.dumps(report, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
