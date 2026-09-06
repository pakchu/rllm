"""Materialize outcome-blind source support for frozen HVMDPAC-6 pairs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from training import preregister_high_volatility_multi_domain_pairwise_concordance as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv


PREREG_SHA = "ba21fd9cb7d89de2391497d7db8642d15eafe0a5973606be9c4528774938957b"
CLOCK_DIR = Path("data/high_volatility_multi_domain_pairwise_concordance_clocks_2023_2026")
RESULT = Path(
    "results/high_volatility_multi_domain_pairwise_concordance_support_2026-08-14.json"
)
HOLD = pd.Timedelta("6h")
INPUT_COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
)
OUTPUT_COLUMNS = (
    "candidate",
    "control",
    "split",
    "left_component_id",
    "right_component_id",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"HVMDPAC-6 expected JSON object: {path}")
    return value


def verify_frozen_inputs() -> dict[str, Any]:
    """Verify the preregistration and every bound component artifact."""
    if sha256_file(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVMDPAC-6 preregistration artifact drift")
    registration = _read_json(prereg.DEFAULT_OUTPUT)
    prereg.validate(registration)

    verified: dict[str, Any] = {}
    for component in prereg.COMPONENT_ORDER:
        artifacts = prereg.COMPONENT_ARTIFACTS[component]
        component_result: dict[str, Any] = {}
        for artifact_type in ("preregistration", "support", "gross9", "clock"):
            artifact = artifacts[artifact_type]
            actual = sha256_file(artifact["path"])
            if actual != artifact["sha256"]:
                raise RuntimeError(
                    f"HVMDPAC-6 {component} {artifact_type} artifact drift"
                )
            component_result[artifact_type] = {
                "path": artifact["path"],
                "sha256": actual,
                "verified": True,
            }

        # Consult frozen result artifacts only for their preregistered pass states.
        source_support = _read_json(Path(artifacts["support"]["path"]))
        source_support_passed = source_support.get("support_passed") is True
        if not source_support_passed:
            raise RuntimeError(
                f"HVMDPAC-6 {component} frozen source-support artifact did not pass"
            )

        gross9 = _read_json(Path(artifacts["gross9"]["path"]))
        gross9_passed = (
            gross9.get("source_support_passed") is True
            and gross9.get("every_gross9_sleeve_passed") is True
            and gross9.get("gross9_novelty_status") == "passed"
        )
        if not gross9_passed:
            raise RuntimeError(f"HVMDPAC-6 {component} frozen Gross9 artifact did not pass")
        component_result["source_support_passed"] = True
        component_result["gross9_passed"] = True
        verified[component] = component_result
    return verified


def _stage_for(entry: pd.Timestamp, exit_time: pd.Timestamp) -> str | None:
    for name, bounds in prereg.build()["stages"].items():
        start, end = (pd.Timestamp(value) for value in bounds)
        if entry >= start and exit_time <= end:
            return name
    return None


def load_component_clock(component: str) -> pd.DataFrame:
    """Read only the frozen clock fields needed for source-support materialization."""
    if component not in prereg.COMPONENT_ARTIFACTS:
        raise ValueError(f"unknown HVMDPAC-6 component: {component}")
    path = prereg.COMPONENT_ARTIFACTS[component]["clock"]["path"]
    frame = pd.read_csv(path, usecols=list(INPUT_COLUMNS))
    if frame.columns.tolist() != list(INPUT_COLUMNS):
        raise RuntimeError(f"HVMDPAC-6 {component} clock schema drift")
    for column in ("decision_time", "feature_available_time", "entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["side"] = pd.to_numeric(frame["side"], errors="raise")

    if frame["entry_time"].duplicated().any():
        raise RuntimeError(f"HVMDPAC-6 {component} duplicate entry_time")
    if not frame["candidate"].eq(component).all() or not frame["control"].eq("primary").all():
        raise RuntimeError(f"HVMDPAC-6 {component} clock identity drift")
    if not frame["side"].isin((-1, 1)).all():
        raise RuntimeError(f"HVMDPAC-6 {component} clock has non-strict side")
    if not frame["exit_time"].eq(frame["entry_time"] + HOLD).all():
        raise RuntimeError(f"HVMDPAC-6 {component} clock hold drift")
    if not frame["decision_time"].le(frame["entry_time"]).all():
        raise RuntimeError(f"HVMDPAC-6 {component} decision after entry")
    if not frame["feature_available_time"].le(frame["entry_time"]).all():
        raise RuntimeError(f"HVMDPAC-6 {component} feature availability after entry")
    expected_splits = [
        _stage_for(entry, exit_time)
        for entry, exit_time in zip(frame["entry_time"], frame["exit_time"])
    ]
    if any(split is None for split in expected_splits) or frame["split"].tolist() != expected_splits:
        raise RuntimeError(f"HVMDPAC-6 {component} split drift")
    return frame.sort_values("entry_time", kind="stable").reset_index(drop=True)


def reserve_half_open(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply chronological first-eligible reservation with [entry, exit) intervals."""
    ordered = frame.sort_values("entry_time", kind="stable")
    keep: list[int] = []
    next_available: pd.Timestamp | None = None
    for index, row in ordered.iterrows():
        if next_available is not None and row["entry_time"] < next_available:
            continue
        keep.append(index)
        next_available = row["exit_time"]
    return ordered.loc[keep].reset_index(drop=True)


def intersect_pair(
    left: str, right: str, clocks: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    if f"{left}__AND__{right}" not in prereg.CANDIDATE_FAMILY:
        raise ValueError("HVMDPAC-6 pair must use exact frozen unordered order")
    joined = clocks[left].merge(
        clocks[right], on=["entry_time", "side"], how="inner", suffixes=("_left", "_right"),
        validate="one_to_one", sort=True,
    )
    if joined.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not joined["side"].isin((-1, 1)).all():
        raise RuntimeError("HVMDPAC-6 intersection has non-strict side")
    if not joined["exit_time_left"].eq(joined["entry_time"] + HOLD).all() or not joined[
        "exit_time_right"
    ].eq(joined["entry_time"] + HOLD).all():
        raise RuntimeError("HVMDPAC-6 component exit mismatch")
    if not joined["split_left"].eq(joined["split_right"]).all():
        raise RuntimeError("HVMDPAC-6 component split mismatch")

    output = pd.DataFrame(
        {
            "candidate": f"{left}__AND__{right}",
            "control": "primary",
            "split": joined["split_left"],
            "left_component_id": left,
            "right_component_id": right,
            "decision_time": joined[["decision_time_left", "decision_time_right"]].max(axis=1),
            "feature_available_time": joined[
                ["feature_available_time_left", "feature_available_time_right"]
            ].max(axis=1),
            "entry_time": joined["entry_time"],
            "exit_time": joined["entry_time"] + HOLD,
            "side": joined["side"].astype(int),
        },
        columns=OUTPUT_COLUMNS,
    )
    if not output["decision_time"].le(output["entry_time"]).all() or not output[
        "feature_available_time"
    ].le(output["entry_time"]).all():
        raise RuntimeError("HVMDPAC-6 combined availability after entry")
    return reserve_half_open(output)


def support_stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock["split"].eq(split)]
    if selected.empty:
        return {
            "events": 0, "longs": 0, "shorts": 0,
            "minority_side_share": 0.0, "max_month_share": 0.0,
        }
    longs = int(selected["side"].eq(1).sum())
    shorts = int(selected["side"].eq(-1).sum())
    count = len(selected)
    return {
        "events": count,
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / count,
        "max_month_share": int(
            selected["entry_time"].dt.strftime("%Y-%m").value_counts().max()
        ) / count,
    }


def _support_checks(stats: Mapping[str, Mapping[str, float | int]]) -> dict[str, bool]:
    registration = prereg.build()
    stages = tuple(registration["stages"])
    if set(stats) != set(stages):
        raise RuntimeError("HVMDPAC-6 support checks require every frozen stage exactly once")
    gates = registration["source_support_gates"]
    checks: dict[str, bool] = {}
    for split in stages:
        values = stats[split]
        checks[f"{split}_minimum_events"] = (
            values["events"] >= gates["minimum_events"][split]
        )
        checks[f"{split}_side_balance"] = (
            values["minority_side_share"] >= gates["minority_side_share_min"]
        )
        checks[f"{split}_month_concentration"] = (
            values["max_month_share"] <= gates["max_month_share"]
        )
    return checks


def run() -> dict[str, Any]:
    verified = verify_frozen_inputs()
    clocks = {component: load_component_clock(component) for component in prereg.COMPONENT_ORDER}
    CLOCK_DIR.mkdir(parents=True, exist_ok=True)

    pairs: dict[str, Any] = {}
    eligible_pairs: list[str] = []
    for index, left in enumerate(prereg.COMPONENT_ORDER):
        for right in prereg.COMPONENT_ORDER[index + 1 :]:
            candidate = f"{left}__AND__{right}"
            combined = intersect_pair(left, right, clocks)
            path = CLOCK_DIR / f"{candidate}.csv.gz"
            _write_gzip_csv(combined, path)
            stats = {split: support_stats(combined, split) for split in prereg.build()["stages"]}
            checks = _support_checks(stats)
            passed = all(checks.values())
            if passed:
                eligible_pairs.append(candidate)
            pairs[candidate] = {
                "components": [left, right],
                "clock": {"path": str(path), "sha256": sha256_file(path), "rows": len(combined)},
                "support": stats,
                "support_checks": checks,
                "support_passed": passed,
                "advance_to_combination_gross9": passed,
                "advance_to_combination_economic_outcomes": False,
                "decision": "pass_to_combination_gross9" if passed else "terminal_source_support_reject",
            }

    registration = _read_json(prereg.DEFAULT_OUTPUT)
    core = {
        "protocol_version": "hvmdpac_6_source_support_v1",
        "policy_id": prereg.POLICY_ID,
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
            "manifest_hash": registration["manifest_hash"],
        },
        "verified_component_artifacts": verified,
        "frozen_component_clocks_opened": True,
        "combination_incidence_opened": True,
        "combination_outcomes_opened": False,
        "combination_postentry_returns_or_pnl_opened": False,
        "entry_exit_prices_opened": False,
        "funding_opened": False,
        "gross9_rows_opened": False,
        "pairs": pairs,
        "eligible_pairs_for_combination_gross9": eligible_pairs,
        "advance_to_combination_gross9": bool(eligible_pairs),
        "advance_to_combination_economic_outcomes": False,
        "decision": "eligible_pairs_to_combination_gross9" if eligible_pairs else "terminal_no_source_supported_pairs",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({"eligible_pairs": report["eligible_pairs_for_combination_gross9"]}, indent=2))
