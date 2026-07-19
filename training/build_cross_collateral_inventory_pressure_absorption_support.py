"""Build the outcome-blind CIPA-48 clock and support decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from training import build_cross_collateral_positioning_recoil_support as source_tools
from training import preregister_cross_collateral_inventory_pressure_absorption as p


PREREGISTRATION = Path(p.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = (
    "ecb28bb25f842d383d368baaea56dcbcd85e062c07275f2f338db123e7c91a31"
)
PREREGISTRATION_COMMIT = "4e41c33"
DEFAULT_OUTPUT = Path(
    "results/cross_collateral_inventory_pressure_absorption_support_2026-07-19.json"
)
DEFAULT_CLOCK = Path(
    "data/cross_collateral_inventory_pressure_absorption_clock_2021_2023.csv.gz"
)
CCPR_CLOCK = Path("results/cross_collateral_positioning_recoil_clocks_2026-07-17.csv")
CCPR_CLOCK_SHA256 = "2a864ec2b616a3118bf9ffa44f99f96fbe19e79d82870f21a0d7d9010d5c993a"
CONTROL_ORDER = ("primary", "oi_only", "taker_only")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sign(series: pd.Series) -> pd.Series:
    return pd.Series(np.sign(series.to_numpy(dtype=float)), index=series.index)


def _series(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, frame[name])


def _timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result is pd.NaT:
        raise ValueError("CIPA-48 timestamp is NaT")
    return cast(pd.Timestamp, result)


def load_preregistration() -> dict[str, Any]:
    if _sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("CIPA-48 preregistration hash mismatch")
    manifest = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    p.validate_manifest(manifest, verify_sources=False)
    if (
        _sha256(manifest["source_contract"]["positioning"])
        != manifest["source_contract"]["positioning_sha256"]
    ):
        raise ValueError("CIPA-48 positioning source hash mismatch")
    return manifest


def build_features(manifest: dict[str, Any]) -> pd.DataFrame:
    source = source_tools.load_source(manifest)
    return source_tools.build_features(source, manifest)


def build_flags(
    feature: pd.DataFrame, manifest: dict[str, Any]
) -> dict[str, pd.Series]:
    policy = manifest["policy"]
    rotation = _sign(_series(feature, "oi_rotation"))
    taker = _sign(_series(feature, "taker_gap"))
    valid = _series(feature, "feature_complete").astype(bool)
    oi_extreme = _series(feature, "oi_rotation_rank").ge(policy["oi_rotation_rank_min"])
    taker_extreme = _series(feature, "taker_gap_rank").ge(policy["taker_gap_rank_min"])
    return {
        "primary": valid
        & oi_extreme
        & taker_extreme
        & rotation.eq(-taker)
        & rotation.ne(0),
        "oi_only": valid & oi_extreme & rotation.ne(0),
        "taker_only": valid & taker_extreme & taker.ne(0),
    }


def build_clock(feature: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    flags = build_flags(feature, manifest)
    rotation = _sign(_series(feature, "oi_rotation"))
    taker = _sign(_series(feature, "taker_gap"))
    side = {
        "primary": rotation,
        "oi_only": rotation,
        "taker_only": -taker,
    }
    rows: list[dict[str, Any]] = []
    delay = pd.Timedelta(minutes=5 * manifest["policy"]["execution_delay_bars"])
    hold = pd.Timedelta(minutes=5 * manifest["policy"]["hold_bars"])
    for control in CONTROL_ORDER:
        onset = source_tools.false_to_true(flags[control])
        previous_exit: pd.Timestamp | None = None
        for signal_time in feature.index[onset]:
            signal = _timestamp(signal_time)
            entry = _timestamp(signal + delay)
            exit_time = _timestamp(entry + hold)
            if previous_exit is not None and entry < previous_exit:
                continue
            previous_exit = exit_time
            rows.append(
                {
                    "control": control,
                    "signal_time": signal,
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "side": int(side[control].loc[signal]),
                    "oi_rotation": float(feature.loc[signal, "oi_rotation"]),
                    "taker_gap": float(feature.loc[signal, "taker_gap"]),
                    "oi_rotation_rank": float(feature.loc[signal, "oi_rotation_rank"]),
                    "taker_gap_rank": float(feature.loc[signal, "taker_gap_rank"]),
                }
            )
    clock = (
        pd.DataFrame(rows).sort_values(["control", "entry_time"]).reset_index(drop=True)
    )
    if not bool(_series(clock, "side").isin((-1, 1)).all()):
        raise ValueError("CIPA-48 emitted an invalid side")
    return clock


def _window(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    lower = pd.Timestamp(start, tz="UTC")
    upper = pd.Timestamp(end, tz="UTC")
    return frame.loc[
        _series(frame, "signal_time").ge(lower)
        & _series(frame, "entry_time").ge(lower)
        & _series(frame, "exit_time").lt(upper)
    ].copy()


def _summary(frame: pd.DataFrame) -> dict[str, Any]:
    side = _series(frame, "side")
    longs = int(side.eq(1).sum())
    shorts = int(side.eq(-1).sum())
    months = cast(
        pd.Series, _series(frame, "entry_time").dt.strftime("%Y-%m").value_counts()
    )
    return {
        "events": int(len(frame)),
        "longs": longs,
        "shorts": shorts,
        "minimum_side_share": (
            float(min(longs, shorts) / len(frame)) if len(frame) else 0.0
        ),
        "maximum_month_share": float(months.max() / len(frame)) if len(frame) else 0.0,
        "month_counts": {
            str(key): int(value) for key, value in months.sort_index().items()
        },
    }


def _ccpr_overlap(primary: pd.DataFrame) -> dict[str, Any]:
    if _sha256(CCPR_CLOCK) != CCPR_CLOCK_SHA256:
        raise ValueError("CIPA-48 CCPR comparator hash mismatch")
    ccpr = pd.read_csv(CCPR_CLOCK)
    ccpr["entry_time"] = pd.to_datetime(
        _series(ccpr, "entry_time"), utc=True, errors="raise"
    )
    ccpr = ccpr.loc[
        _series(ccpr, "control").eq("primary") & np.isclose(_series(ccpr, "q"), 0.85)
    ]
    output: dict[str, Any] = {}
    for name, start, end in (
        ("train", "2021-07-08", "2023-01-01"),
        ("test_support", "2023-01-01", "2024-01-01"),
    ):
        left = _series(_window(primary, start, end), "entry_time")
        lower, upper = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        right = _series(
            ccpr.loc[
                _series(ccpr, "entry_time").ge(lower)
                & _series(ccpr, "entry_time").lt(upper)
            ],
            "entry_time",
        )
        left_set, right_set = set(left), set(right)
        union = left_set | right_set
        near = (
            sum(
                ((right - value).abs() <= pd.Timedelta(hours=1)).any() for value in left
            )
            / len(left)
            if len(left)
            else 0.0
        )
        output[name] = {
            "cipa_events": int(len(left)),
            "ccpr_events": int(len(right)),
            "exact_entry_jaccard": float(len(left_set & right_set) / len(union))
            if union
            else 0.0,
            "cipa_near_one_hour_fraction": float(near),
        }
    return output


def _write_clock(frame: pd.DataFrame, path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serial = frame.copy()
    for column in ("signal_time", "entry_time", "exit_time"):
        serial[column] = serial[column].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    serial.to_csv(
        output,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    return _sha256(output)


def build(
    output_path: str | Path = DEFAULT_OUTPUT,
    clock_path: str | Path = DEFAULT_CLOCK,
) -> dict[str, Any]:
    manifest = load_preregistration()
    feature = build_features(manifest)
    clock = build_clock(feature, manifest)
    primary = clock.loc[_series(clock, "control").eq("primary")].copy()
    partitions = {
        "train": _summary(_window(primary, "2021-07-08", "2023-01-01")),
        "2021_partial": _summary(_window(primary, "2021-07-08", "2022-01-01")),
        "2022": _summary(_window(primary, "2022-01-01", "2023-01-01")),
        "test_support": _summary(_window(primary, "2023-01-01", "2024-01-01")),
        "2023_H1": _summary(_window(primary, "2023-01-01", "2023-07-01")),
        "2023_H2": _summary(_window(primary, "2023-07-01", "2024-01-01")),
    }
    overlap = _ccpr_overlap(primary)
    gate = manifest["support_gate"]
    checks = {
        "train_events": partitions["train"]["events"]
        >= gate["minimum_nonoverlap_train"],
        "2021_partial_events": partitions["2021_partial"]["events"]
        >= gate["minimum_2021_partial"],
        "2022_events": partitions["2022"]["events"] >= gate["minimum_2022"],
        "2023_events": partitions["test_support"]["events"] >= gate["minimum_2023"],
        "2023_halves": min(
            partitions["2023_H1"]["events"], partitions["2023_H2"]["events"]
        )
        >= gate["minimum_each_2023_half"],
        "train_side_balance": partitions["train"]["minimum_side_share"]
        >= gate["minimum_each_side_share"],
        "2023_side_balance": partitions["test_support"]["minimum_side_share"]
        >= gate["minimum_each_side_share"],
        "train_month_concentration": partitions["train"]["maximum_month_share"]
        <= gate["maximum_single_month_share"],
        "2023_month_concentration": partitions["test_support"]["maximum_month_share"]
        <= gate["maximum_single_month_share"],
        "ccpr_exact_train": overlap["train"]["exact_entry_jaccard"]
        <= gate["ccpr_exact_entry_jaccard_max"],
        "ccpr_exact_2023": overlap["test_support"]["exact_entry_jaccard"]
        <= gate["ccpr_exact_entry_jaccard_max"],
        "ccpr_near_train": overlap["train"]["cipa_near_one_hour_fraction"]
        <= gate["ccpr_near_one_hour_fraction_max"],
        "ccpr_near_2023": overlap["test_support"]["cipa_near_one_hour_fraction"]
        <= gate["ccpr_near_one_hour_fraction_max"],
    }
    clock_sha = _write_clock(clock, clock_path)
    core = {
        "protocol_version": "cross_collateral_inventory_pressure_absorption_support_v1",
        "policy_id": "CIPA-48",
        "as_of_date": "2026-07-19",
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
            "feature_complete_hourly_anchors": int(
                _series(feature, "feature_complete").sum()
            ),
            "execution_market_rows_loaded": 0,
            "funding_rows_loaded": 0,
        },
        "clock": {
            "path": str(clock_path),
            "sha256": clock_sha,
            "rows": int(len(clock)),
        },
        "partitions": partitions,
        "ccpr_novelty": overlap,
        "support_checks": checks,
        "support_passed": bool(all(checks.values())),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "advance_to_train_outcomes": bool(all(checks.values())),
        "sealed_outcome_windows": ["train_2021_2022", "test_2023", "2024_plus"],
    }
    report = {**core, "manifest_hash": _canonical_hash(core)}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clock", default=DEFAULT_CLOCK)
    args = parser.parse_args()
    report = build(args.output, args.clock)
    print(
        json.dumps(
            {
                "support_passed": report["support_passed"],
                "failed": report["failed_checks"],
            }
        )
    )


if __name__ == "__main__":
    main()
