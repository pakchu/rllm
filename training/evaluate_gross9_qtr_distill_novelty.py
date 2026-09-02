"""Train-only Gross9 structural novelty gate for G9QTR-DISTILL-8.

This evaluator opens only the committed G9QTR-DISTILL-8 preregistration,
materialized clock-source package, portfolio schedules, and frozen Gross9
structural clocks. It does not open market prices, funding, returns, PnL, or any
out-of-sample economic outcome.

Candidate event semantics are portfolio-level: exact-entry and near-6h matching
use signed portfolio episode starts, represented as ``(start_time, side)`` from
the materialized train ``signed_episodes`` schedule. Occupied-bar Jaccard and
signed-exposure Pearson use the train 5m weighted net exposure array
reconstructed from ``segments.target_exposure``. Each Gross9 comparator is
reconstructed as a weighted signed exposure array using that sleeve's frozen
Gross9 weight.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import build_gross9_qtr_distill_clocks as source_support
from training import evaluate_gross9_async_pair_train_novelty as pair_novelty
from training import export_gross9_structural_clocks as gross9
from training import preregister_gross9_qtr_distill as prereg

POLICY_ID = prereg.POLICY_ID
PROTOCOL_VERSION = "gross9_qtr_distill_train_gross9_novelty_v1"
PREREGISTRATION = prereg.DEFAULT_OUTPUT
SOURCE_PACKAGE = source_support.DEFAULT_RESULT
GROSS9_MANIFEST = gross9.DEFAULT_MANIFEST
OUTPUT = Path("results/gross9_qtr_distill_train_gross9_novelty_2026-09-02.json")
TRAIN_START = pair_novelty.TRAIN_START
TRAIN_END = pair_novelty.TRAIN_END
BAR = pair_novelty.BAR
NEAR_LIMIT = pd.Timedelta(hours=6)
LIMITS = dict(pair_novelty.LIMITS)
SCHEDULE_NAMES = ("transitions", "segments", "signed_episodes")


def sha256_file(path: str | Path) -> str:
    return pair_novelty.sha256_file(path)


def canonical_hash(value: Any) -> str:
    return gross9.canonical_hash(value)


def _load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} JSON artifact is not an object: {path}")
    return value


def _verify_manifest_hash(value: Mapping[str, Any], label: str) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} {label} manifest drift")


def _verify_bound_file(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path")))
    if not path.is_file():
        raise RuntimeError(f"{POLICY_ID} missing {label}: {path}")
    if sha256_file(path) != record.get("sha256"):
        raise RuntimeError(f"{POLICY_ID} {label} hash drift")
    return path


def _iso_z(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _check_no_outcome_boundary(boundary: Mapping[str, Any], *, label: str) -> None:
    false_flags = (
        "market_rows_opened",
        "entry_exit_prices_opened",
        "funding_opened",
        "returns_or_pnl_opened",
        "economic_outcomes_opened",
        "oos_outcomes_opened",
        "oos_outcomes_opened_by_this_preregistration",
    )
    for flag in false_flags:
        if flag in boundary and boundary.get(flag) is not False:
            raise RuntimeError(f"{POLICY_ID} {label} boundary opened outcomes: {flag}")


def _expected_gross9_gate_payload() -> dict[str, Any]:
    return {
        "exact_entry_jaccard_max": 0.1,
        "candidate_near_6h_share_max": 0.35,
        "occupied_5m_jaccard_max": 0.25,
        "absolute_signed_exposure_pearson_max": 0.35,
        "must_pass_before_economics": True,
    }


def load_validated_controls() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    registration = _load_json_object(PREREGISTRATION)
    _verify_manifest_hash(registration, "preregistration")
    prereg.validate(registration)
    if registration != prereg.build():
        raise RuntimeError(f"{POLICY_ID} preregistration artifact differs from code")
    prereg_sha = sha256_file(PREREGISTRATION)
    if registration.get("implementation", {}).get("preregister", {}).get("sha256") != sha256_file(prereg.__file__):
        raise RuntimeError(f"{POLICY_ID} preregistration implementation hash drift")
    gates = dict(registration.get("oos_gate_rule", {}).get("gross9_novelty_gates", {}))
    comparator = gates.pop("comparator", {})
    if gates != _expected_gross9_gate_payload():
        raise RuntimeError(f"{POLICY_ID} preregistered Gross9 novelty gate drift")
    if (
        comparator.get("path") != GROSS9_MANIFEST.as_posix()
        or comparator.get("sha256") != pair_novelty.GROSS9_MANIFEST_SHA256
        or comparator.get("manifest_hash") != pair_novelty.GROSS9_MANIFEST_HASH
        or comparator.get("authority_sha256") != gross9.ANCHOR_SHA256
    ):
        raise RuntimeError(f"{POLICY_ID} preregistered Gross9 comparator drift")
    _check_no_outcome_boundary(registration.get("evidence_boundary", {}), label="preregistration")

    source = _load_json_object(SOURCE_PACKAGE)
    _verify_manifest_hash(source, "source package")
    if (
        source.get("policy_id") != POLICY_ID
        or source.get("protocol_version") != source_support.PROTOCOL_VERSION
        or source.get("decision") != "materialized_shadow_distilled_clock_package"
        or source.get("preregistration", {}).get("path") != str(PREREGISTRATION)
        or source.get("preregistration", {}).get("sha256") != prereg_sha
        or source.get("preregistration", {}).get("manifest_hash") != registration.get("manifest_hash")
        or source.get("preregistration", {}).get("status") != "validated_against_committed_preregistration"
    ):
        raise RuntimeError(f"{POLICY_ID} source-package preregistration binding drift")
    builder = source.get("implementation", {}).get("builder", {})
    if builder.get("path") != "training/build_gross9_qtr_distill_clocks.py" or builder.get("sha256") != sha256_file(source_support.__file__):
        raise RuntimeError(f"{POLICY_ID} source-package builder binding drift")
    components = source.get("components", {})
    if (
        tuple(components.get("base_order", ())) != source_support.BASE_ORDER
        or components.get("veto_component") != source_support.VETO_COMPONENT
        or components.get("base_weights") != source_support.BASE_WEIGHTS
        or not math.isclose(float(components.get("max_gross", -1.0)), source_support.MAX_GROSS)
    ):
        raise RuntimeError(f"{POLICY_ID} source-package component/weight drift")
    for name in SCHEDULE_NAMES:
        rec = source.get("portfolio_schedules", {}).get(name)
        if not isinstance(rec, Mapping):
            raise RuntimeError(f"{POLICY_ID} source-package missing schedule binding: {name}")
        _verify_bound_file(rec, label=f"{name} schedule")
        if int(rec.get("rows", -1)) < 0:
            raise RuntimeError(f"{POLICY_ID} source-package invalid schedule row binding: {name}")
    _check_no_outcome_boundary(source.get("evidence_boundary", {}), label="source-package")

    _verify_bound_file({"path": GROSS9_MANIFEST, "sha256": pair_novelty.GROSS9_MANIFEST_SHA256}, label="Gross9 structural manifest")
    manifest = _load_json_object(GROSS9_MANIFEST)
    _verify_manifest_hash(manifest, "Gross9 structural clocks")
    if (
        manifest.get("manifest_hash") != pair_novelty.GROSS9_MANIFEST_HASH
        or manifest.get("protocol_version") != gross9.PROTOCOL_VERSION
        or manifest.get("authority", {}).get("sha256") != gross9.ANCHOR_SHA256
        or manifest.get("authority", {}).get("weights") != gross9.EXPECTED_WEIGHTS
        or manifest.get("all_authoritative_counts_verified") is not True
        or set(manifest.get("clocks", {})) != set(gross9.EXPECTED_WEIGHTS)
    ):
        raise RuntimeError(f"{POLICY_ID} Gross9 structural clock authority drift")
    return registration, source, manifest


def _read_bound_schedule(source: Mapping[str, Any], name: str, required: set[str]) -> pd.DataFrame:
    rec = source["portfolio_schedules"][name]
    path = _verify_bound_file(rec, label=f"{name} schedule")
    frame = pd.read_csv(path, compression="gzip")
    if len(frame) != int(rec.get("rows", -1)):
        raise RuntimeError(f"{POLICY_ID} {name} schedule row-count drift")
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{POLICY_ID} {name} schedule schema drift")
    return frame


def load_train_episodes(source: Mapping[str, Any]) -> pd.DataFrame:
    frame = _read_bound_schedule(source, "signed_episodes", {"candidate", "split", "start_time", "end_time", "side"})
    out = frame[["candidate", "split", "start_time", "end_time", "side"]].copy()
    out["split"] = out["split"].astype(str)
    out["start_time"] = pd.to_datetime(out["start_time"], utc=True, errors="raise")
    out["end_time"] = pd.to_datetime(out["end_time"], utc=True, errors="raise")
    out["side"] = pd.to_numeric(out["side"], errors="raise").astype(int)
    if not out["candidate"].eq(POLICY_ID).all() or set(out["side"]) - {-1, 1}:
        raise RuntimeError(f"{POLICY_ID} signed episode identity/side drift")
    train = out.loc[out["split"].eq("train")].copy()
    if train.empty:
        raise RuntimeError(f"{POLICY_ID} train signed episodes are empty")
    if not (train["start_time"].ge(TRAIN_START) & train["start_time"].lt(TRAIN_END) & train["end_time"].le(TRAIN_END) & train["start_time"].lt(train["end_time"])).all():
        raise RuntimeError(f"{POLICY_ID} signed episodes leaked outside train window")
    return train.sort_values(["start_time", "end_time", "side"], kind="mergesort").reset_index(drop=True)


def load_train_segments(source: Mapping[str, Any]) -> pd.DataFrame:
    frame = _read_bound_schedule(source, "segments", {"candidate", "split", "start_time", "end_time", "target_exposure"})
    out = frame[["candidate", "split", "start_time", "end_time", "target_exposure"]].copy()
    out["split"] = out["split"].astype(str)
    out["start_time"] = pd.to_datetime(out["start_time"], utc=True, errors="raise")
    out["end_time"] = pd.to_datetime(out["end_time"], utc=True, errors="raise")
    out["target_exposure"] = pd.to_numeric(out["target_exposure"], errors="raise").astype(float)
    if not out["candidate"].eq(POLICY_ID).all() or not np.isfinite(out["target_exposure"].to_numpy()).all():
        raise RuntimeError(f"{POLICY_ID} segment identity/exposure drift")
    train = out.loc[out["split"].eq("train")].copy()
    if train.empty:
        raise RuntimeError(f"{POLICY_ID} train segments are empty")
    if not (train["start_time"].ge(TRAIN_START) & train["start_time"].lt(TRAIN_END) & train["end_time"].le(TRAIN_END) & train["start_time"].lt(train["end_time"])).all():
        raise RuntimeError(f"{POLICY_ID} segments leaked outside train window")
    return train.sort_values(["start_time", "end_time"], kind="mergesort").reset_index(drop=True)


def validate_train_transitions(source: Mapping[str, Any]) -> int:
    frame = _read_bound_schedule(source, "transitions", {"candidate", "split", "timestamp", "target_exposure"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    train = frame.loc[frame["split"].astype(str).eq("train")]
    if train.empty:
        raise RuntimeError(f"{POLICY_ID} train transitions are empty")
    if not (train["timestamp"].ge(TRAIN_START) & train["timestamp"].le(TRAIN_END)).all():
        raise RuntimeError(f"{POLICY_ID} transitions leaked outside train window")
    return int(len(train))


def signed_episode_starts(episodes: pd.DataFrame) -> tuple[tuple[pd.Timestamp, int], ...]:
    return tuple((pd.Timestamp(row.start_time), int(row.side)) for row in episodes.itertuples(index=False))


def exact_signed_entry_jaccard(candidate: Sequence[tuple[pd.Timestamp, int]], comparator: Sequence[tuple[pd.Timestamp, int]]) -> float:
    left, right = set(candidate), set(comparator)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def optimal_signed_near_matches(candidate: Sequence[tuple[pd.Timestamp, int]], comparator: Sequence[tuple[pd.Timestamp, int]]) -> tuple[tuple[tuple[tuple[pd.Timestamp, int], tuple[pd.Timestamp, int]], ...], int]:
    left = tuple(sorted(set(candidate), key=lambda item: (item[0], item[1])))
    right = tuple(sorted(set(comparator), key=lambda item: (item[0], item[1])))
    rows, columns = len(left), len(right)
    cardinality = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    lag = np.zeros((rows + 1, columns + 1), dtype=np.int64)
    choice = np.full((rows, columns), 1, dtype=np.int8)
    limit_seconds = int(NEAR_LIMIT.total_seconds())

    def rank(item: tuple[int, int, int]) -> tuple[int, int, int]:
        count, total_lag, priority = item
        return (-count, total_lag, priority)

    for i in range(rows - 1, -1, -1):
        for j in range(columns - 1, -1, -1):
            options = [(int(cardinality[i + 1, j]), int(lag[i + 1, j]), 1), (int(cardinality[i, j + 1]), int(lag[i, j + 1]), 2)]
            seconds = int(abs((left[i][0] - right[j][0]).total_seconds()))
            if left[i][1] == right[j][1] and seconds <= limit_seconds:
                options.append((1 + int(cardinality[i + 1, j + 1]), seconds + int(lag[i + 1, j + 1]), 0))
            best = min(options, key=rank)
            cardinality[i, j], lag[i, j], choice[i, j] = best

    matches: list[tuple[tuple[pd.Timestamp, int], tuple[pd.Timestamp, int]]] = []
    i = j = 0
    while i < rows and j < columns:
        selected = int(choice[i, j])
        if selected == 0:
            matches.append((left[i], right[j]))
            i += 1
            j += 1
        elif selected == 1:
            i += 1
        else:
            j += 1
    return tuple(matches), int(lag[0, 0])


def _bar_count() -> int:
    return int((TRAIN_END - TRAIN_START) / BAR)


def weighted_segment_exposure_train(segments: pd.DataFrame) -> np.ndarray:
    exposure = np.zeros(_bar_count(), dtype=float)
    for row in segments.itertuples(index=False):
        start = int((pd.Timestamp(row.start_time) - TRAIN_START) / BAR)
        end = int((pd.Timestamp(row.end_time) - TRAIN_START) / BAR)
        if start < 0 or end > len(exposure) or end <= start:
            raise RuntimeError(f"{POLICY_ID} train exposure received invalid segment")
        exposure[start:end] = float(row.target_exposure)
    return exposure


def load_gross9_train_clock(sleeve: str, record: Mapping[str, Any]) -> tuple[pd.DataFrame, int, int]:
    return pair_novelty.load_gross9_train_clock(sleeve, record)


def weighted_gross9_exposure_train(clock: pd.DataFrame, sleeve: str) -> np.ndarray:
    bar_count = _bar_count()
    changes = np.zeros(bar_count + 1, dtype=float)
    weight = float(gross9.EXPECTED_WEIGHTS[sleeve])
    train = pair_novelty.train_contained(clock)
    if train.empty:
        raise RuntimeError(f"{POLICY_ID} Gross9 train comparator is empty: {sleeve}")
    for row in train.itertuples(index=False):
        start = int((pd.Timestamp(row.entry_time) - TRAIN_START) / BAR)
        end = int((pd.Timestamp(row.exit_time) - TRAIN_START) / BAR)
        if start < 0 or end > bar_count or end <= start:
            raise RuntimeError(f"{POLICY_ID} Gross9 exposure received invalid interval")
        signed = int(row.side) * weight
        changes[start] += signed
        changes[end] -= signed
    return np.cumsum(changes[:-1])


def occupied_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_occ = np.abs(left) > 1e-12
    right_occ = np.abs(right) > 1e-12
    union = left_occ | right_occ
    return float((left_occ & right_occ).sum() / union.sum()) if bool(union.any()) else 0.0


def pearson_or_nan(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise RuntimeError(f"{POLICY_ID} exposure vectors have different shapes")
    if left.size == 0 or np.nanstd(left) == 0.0 or np.nanstd(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _corr_check(value: float) -> bool:
    return True if math.isnan(value) else abs(value) <= LIMITS["absolute_signed_exposure_pearson"]


def evaluate_against_gross9(candidate_events: Sequence[tuple[pd.Timestamp, int]], candidate_exposure: np.ndarray, gross9_clock: pd.DataFrame, sleeve: str) -> dict[str, Any]:
    comp_train = pair_novelty.train_contained(gross9_clock)
    comparator_events = tuple((pd.Timestamp(row.entry_time), int(row.side)) for row in comp_train.itertuples(index=False))
    matches, total_lag_seconds = optimal_signed_near_matches(candidate_events, comparator_events)
    denominator = min(len(set(candidate_events)), len(set(comparator_events)))
    matched_share = len(matches) / denominator if denominator else 0.0
    comparator_exposure = weighted_gross9_exposure_train(comp_train, sleeve)
    correlation = pearson_or_nan(candidate_exposure, comparator_exposure)
    metrics = {
        "exact_entry_jaccard": exact_signed_entry_jaccard(candidate_events, comparator_events),
        "one_to_one_6h_max_matched_share": matched_share,
        "occupied_5m_bar_jaccard": occupied_jaccard(candidate_exposure, comparator_exposure),
        "signed_exposure_pearson": correlation,
        "absolute_signed_exposure_pearson": None if math.isnan(correlation) else abs(correlation),
    }
    checks = {
        "exact_entry_jaccard": metrics["exact_entry_jaccard"] <= LIMITS["exact_entry_jaccard"],
        "one_to_one_6h_max_matched_share": metrics["one_to_one_6h_max_matched_share"] <= LIMITS["one_to_one_6h_max_matched_share"],
        "occupied_5m_bar_jaccard": metrics["occupied_5m_bar_jaccard"] <= LIMITS["occupied_5m_bar_jaccard"],
        "absolute_signed_exposure_pearson": _corr_check(correlation),
    }
    match_rows = [
        {"candidate_entry": _iso_z(left[0]), "candidate_side": int(left[1]), "comparator_entry": _iso_z(right[0]), "comparator_side": int(right[1]), "absolute_lag_seconds": int(abs((left[0] - right[0]).total_seconds()))}
        for left, right in matches
    ]
    return {
        "common_window": [_iso_z(TRAIN_START), _iso_z(TRAIN_END)],
        "entry_event_semantics": "signed portfolio episode starts: (signed_episodes.start_time, signed_episodes.side)",
        "exposure_semantics": "5m weighted net exposure from candidate segments.target_exposure; Gross9 sleeve side * frozen sleeve weight",
        "candidate_signed_episode_starts": int(len(candidate_events)),
        "comparator_signed_entry_starts": int(len(comparator_events)),
        "matching": {"objective": "same-side maximum cardinality, then minimum total absolute lag, deterministic non-crossing UTC order", "maximum_cardinality": int(len(matches)), "minimum_total_absolute_lag_seconds": int(total_lag_seconds), "matched_pairs_sha256": canonical_hash(match_rows)},
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run(output: str | Path = OUTPUT) -> dict[str, Any]:
    registration, source, manifest = load_validated_controls()
    train_transition_rows = validate_train_transitions(source)
    train_episodes = load_train_episodes(source)
    train_segments = load_train_segments(source)
    candidate_events = signed_episode_starts(train_episodes)
    candidate_exposure = weighted_segment_exposure_train(train_segments)

    gross9_clocks: dict[str, pd.DataFrame] = {}
    gross9_counts: dict[str, Any] = {}
    for sleeve in gross9.EXPECTED_WEIGHTS:
        record = manifest["clocks"][sleeve]
        clock, total_rows, common_rows = load_gross9_train_clock(sleeve, record)
        gross9_clocks[sleeve] = clock
        gross9_counts[sleeve] = {"path": record["path"], "sha256": record["sha256"], "manifest_total_rows": int(record["rows"]), "full_file_rows_verified": int(total_rows), "split_counts_verified": {key: int(value) for key, value in record["counts"].items()}, "train_common_window_rows_evaluated": int(common_rows), "frozen_weight": float(gross9.EXPECTED_WEIGHTS[sleeve])}

    sleeve_results = {sleeve: evaluate_against_gross9(candidate_events, candidate_exposure, gross9_clocks[sleeve], sleeve) for sleeve in gross9.EXPECTED_WEIGHTS}
    all_passed = all(item["passed"] for item in sleeve_results.values())
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": prereg.AS_OF_DATE,
        "implementation": {"evaluator": {"path": Path(__file__).as_posix(), "sha256": sha256_file(Path(__file__))}},
        "preregistration": {"path": str(PREREGISTRATION), "sha256": sha256_file(PREREGISTRATION), "manifest_hash": registration["manifest_hash"]},
        "source_package": {"path": str(SOURCE_PACKAGE), "sha256": sha256_file(SOURCE_PACKAGE), "manifest_hash": source["manifest_hash"], "predecessor_mutated": False},
        "gross9_structural_clocks": {"path": str(GROSS9_MANIFEST), "sha256": pair_novelty.GROSS9_MANIFEST_SHA256, "manifest_hash": manifest["manifest_hash"], "authority_sha256": gross9.ANCHOR_SHA256, "complete_roster": list(gross9.EXPECTED_WEIGHTS), "counts": gross9_counts},
        "candidate": POLICY_ID,
        "candidate_source": {"portfolio_schedules": source["portfolio_schedules"], "train_transition_rows_opened_for_leakage_check": train_transition_rows, "train_signed_episode_rows_evaluated": int(len(train_episodes)), "train_segment_rows_evaluated": int(len(train_segments)), "weighted_net_exposure_nonzero_5m_bars": int((np.abs(candidate_exposure) > 1e-12).sum()), "max_abs_weighted_net_exposure": float(np.max(np.abs(candidate_exposure))) if candidate_exposure.size else 0.0},
        "window": {"start_inclusive": _iso_z(TRAIN_START), "end_exclusive": _iso_z(TRAIN_END), "containment_policy": "split == train AND timestamp/segment/episode fully contained; no clip, shift, or split"},
        "limits": LIMITS,
        "gross9_sleeves": sleeve_results,
        "gross9_pass": all_passed,
        "gross9_novelty_status": "passed" if all_passed else "failed",
        "advance_to_economic_outcomes": all_passed,
        "decision": "pass_g9qtr_distill_to_economic_outcomes" if all_passed else "terminal_gross9_novelty_reject",
        "evidence_boundary": {"portfolio_transition_rows_opened_train_only_for_leakage_check": train_transition_rows, "portfolio_signed_episode_rows_opened_train_only": int(len(train_episodes)), "portfolio_segment_rows_opened_train_only": int(len(train_segments)), "entry_event_semantics": "signed portfolio episode starts", "weighted_net_exposure_source": "segments.target_exposure", "gross9_structural_clock_rows_opened_full_files_for_count_hash_split_verification": sum(item["full_file_rows_verified"] for item in gross9_counts.values()), "gross9_structural_clock_rows_evaluated_after_train_filter": sum(item["train_common_window_rows_evaluated"] for item in gross9_counts.values()), "market_rows_opened": 0, "entry_exit_prices_opened": 0, "funding_rows_opened": 0, "returns_or_pnl_rows_opened": 0, "economic_outcome_rows_opened": 0, "oos_schedule_rows_opened": 0, "oos_outcomes_opened": False, "portfolio_return_or_pnl_metrics_computed": False},
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({"output": str(args.output), "gross9_pass": result["gross9_pass"], "decision": result["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
