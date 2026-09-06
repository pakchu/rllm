"""Train-only economics for Gross9 async active opposite-veto candidates.

Only source-supported, exact-duplicate-clean, and Gross9-novel candidates from
frozen G9ASYNCACTIVEVETO-8 are opened for train economic outcomes.  The raw
rank-one candidate is selected by the preregistered train CAGR/strict-MDD
ranking; if that rank-one candidate fails any train economic gate, the family
terminates without substitution.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from training import evaluate_gross9_async_active_veto_train_novelty as novelty
from training import evaluate_options_led_volatility_expansion_premium_relay_economics_v5 as econ
from training import preregister_gross9_async_active_veto_search as prereg


POLICY_ID = prereg.POLICY_ID
PROTOCOL_VERSION = "gross9_async_active_veto_train_economics_v1"
NOVELTY = novelty.OUTPUT
NOVELTY_SHA256 = "64a8cbd12edc04ebb02d30649687c3319a1d06558005ddd8c73ab22f81d884cf"
NOVELTY_MANIFEST_HASH = "27cd576b8175556ff23e0ee87ea1e99092dcaa41cdb8b78fcf6cb202e845c40f"
OUTPUT = Path("results/gross9_async_active_veto_train_economics_2026-09-02.json")
TRAIN_START = pd.Timestamp("2023-07-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2024-01-01T00:00:00Z")
EXPECTED_NOVEL_CANDIDATES = 14


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return prereg.canonical_hash(value)


def _load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} JSON artifact is not an object: {path}")
    return value


def _verify_manifest_hash(value: Mapping[str, Any], label: str) -> None:
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} {label} manifest hash drift")


def _iso_z(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat().replace("+00:00", "Z")


def _resolve_hash_bound(relative: str | Path, expected_sha256: str) -> Path:
    path = Path(relative)
    candidates = [path, Path("/home/pakchu/rllm") / path]
    for candidate in candidates:
        if candidate.is_file():
            if sha256_file(candidate) != expected_sha256:
                raise RuntimeError(f"{POLICY_ID} hash drift: {candidate}")
            return candidate
    raise FileNotFoundError(path)


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _stream_gzip(path: Path, start: pd.Timestamp, end: pd.Timestamp, columns: tuple[str, ...], date_col: str, include_end: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(columns).issubset(reader.fieldnames or []):
            raise RuntimeError(f"{POLICY_ID} schema drift: {path}")
        for row in reader:
            timestamp = _utc(row[date_col])
            if timestamp > end or (timestamp == end and not include_end):
                break
            if timestamp >= start:
                rows.append({column: row[column] for column in columns})
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        raise RuntimeError(f"{POLICY_ID} empty physical prefix: {path}")
    frame[date_col] = pd.to_datetime(frame[date_col], utc=True, errors="raise")
    return frame


def load_market_hash_bound(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = _resolve_hash_bound(econ.v1.MARKET, econ.v1.MARKET_SHA)
    market = _stream_gzip(path, start, end, ("date", "open", "high", "low", "close"), "date", True)
    for column in ("open", "high", "low", "close"):
        market[column] = pd.to_numeric(market[column], errors="raise")
    return market


def load_train_funding_hash_bound(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = _resolve_hash_bound(econ.TRAIN_FUNDING, econ.TRAIN_FUNDING_SHA)
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"funding_time_utc", "funding_rate", "settlement_mark_price"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"{POLICY_ID} train funding mark schema drift")
        for row in reader:
            timestamp = _utc(row["funding_time_utc"])
            if timestamp >= end:
                break
            if timestamp >= start:
                rows.append({"date": timestamp, "funding_rate": row["funding_rate"], "mark_price": row["settlement_mark_price"]})
    funding = pd.DataFrame(rows, columns=["date", "funding_rate", "mark_price"])
    if funding.empty:
        raise RuntimeError(f"{POLICY_ID} empty train funding mark prefix")
    for column in ("funding_rate", "mark_price"):
        funding[column] = pd.to_numeric(funding[column], errors="raise")
    return funding


def load_novelty_authorization() -> dict[str, Any]:
    if not NOVELTY.is_file():
        raise RuntimeError(f"{POLICY_ID} missing train novelty artifact: {NOVELTY}")
    if sha256_file(NOVELTY) != NOVELTY_SHA256:
        raise RuntimeError(f"{POLICY_ID} train novelty artifact hash drift")
    report = _load_json_object(NOVELTY)
    _verify_manifest_hash(report, "train Gross9 novelty")
    if report.get("manifest_hash") != NOVELTY_MANIFEST_HASH:
        raise RuntimeError(f"{POLICY_ID} train novelty manifest binding drift")
    if (
        report.get("protocol_version") != novelty.PROTOCOL_VERSION
        or report.get("policy_id") != POLICY_ID
        or report.get("candidate_family") != list(prereg.CANDIDATE_FAMILY)
        or report.get("candidate_family_size") != prereg.FAMILY_SIZE
        or report.get("gross9_passed_any_candidate") is not True
        or report.get("advance_to_economic_outcomes") is not True
        or report.get("gross9_novelty_passed_candidate_count") != EXPECTED_NOVEL_CANDIDATES
        or report.get("decision") != "pass_gross9_novel_candidates_to_train_economics"
    ):
        raise RuntimeError(f"{POLICY_ID} train novelty did not authorize economics")
    boundary = report.get("evidence_boundary", {})
    if (
        boundary.get("candidate_family_rows_counted") != prereg.FAMILY_SIZE
        or boundary.get("source_and_exact_duplicate_supported_candidates_expected") != EXPECTED_NOVEL_CANDIDATES
        or boundary.get("exact_duplicate_gate_projected_for_all_72") is not True
        or boundary.get("btc_price_or_return_rows_opened") != 0
        or boundary.get("entry_exit_prices_opened") != 0
        or boundary.get("funding_rows_opened") != 0
        or boundary.get("economic_outcome_rows_opened") != 0
        or boundary.get("portfolio_return_or_pnl_metrics_computed") is not False
        or boundary.get("outcomes_opened") is not False
    ):
        raise RuntimeError(f"{POLICY_ID} train novelty boundary already opened economics or lost 72/14 authorization")
    candidates = report.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != set(prereg.CANDIDATE_FAMILY) or len(candidates) != prereg.FAMILY_SIZE:
        raise RuntimeError(f"{POLICY_ID} train novelty candidate-family drift")
    eligible = [
        candidate
        for candidate in prereg.CANDIDATE_FAMILY
        if candidates[candidate].get("source_pass") is True
        and candidates[candidate].get("exact_duplicate_pass") is True
        and candidates[candidate].get("gross9_pass") is True
    ]
    if eligible != report.get("gross9_novelty_passed_candidates") or len(eligible) != EXPECTED_NOVEL_CANDIDATES:
        raise RuntimeError(f"{POLICY_ID} train novelty eligible roster drift")
    return report


def load_candidate_clock(clock_record: Mapping[str, Any], candidate: str) -> pd.DataFrame:
    path = Path(str(clock_record["path"]))
    if sha256_file(path) != clock_record.get("sha256"):
        raise RuntimeError(f"{POLICY_ID} candidate clock hash drift: {candidate}")
    clock = econ.load_clock(path, "train", TRAIN_START, TRAIN_END)
    if len(clock) != int(clock_record.get("rows", -1)):
        raise RuntimeError(f"{POLICY_ID} candidate clock row-count drift: {candidate}")
    return clock


def public_metric(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "trade_rows"}


def evaluate_candidate(candidate: str, clock_record: Mapping[str, Any], market: pd.DataFrame, funding: pd.DataFrame) -> dict[str, Any]:
    clock = load_candidate_clock(clock_record, candidate)
    base = econ.simulate(clock, market, funding, TRAIN_START, TRAIN_END, econ.BASE_COST)
    stress = econ.simulate(clock, market, funding, TRAIN_START, TRAIN_END, econ.STRESS_COST)
    midpoint = TRAIN_START + (TRAIN_END - TRAIN_START) / 2
    halves = {
        name: public_metric(econ.simulate(clock[(clock.entry_time >= start) & (clock.exit_time <= end)], market, funding, start, end, econ.BASE_COST))
        for name, start, end in (
            ("first", TRAIN_START, midpoint),
            ("second", midpoint, TRAIN_END),
        )
    }
    cluster = econ.cluster_p(base["trade_rows"])
    checks = {
        "absolute_return_positive": base["absolute_return_pct"] > 0,
        "cagr_to_strict_mdd_min_3": base["cagr_to_strict_mdd"] >= 3.0,
        "strict_mdd_max_15": base["strict_mdd_pct"] <= 15.0,
        "mean_gross_move_min_20bp": base["mean_gross_underlying_bp"] >= 20.0,
        "cluster_signflip_p_max_bonferroni_0_1_over_72": cluster["pvalue"] <= prereg.BONFERRONI_RAW_P_MAX,
        "stress_absolute_return_positive": stress["absolute_return_pct"] > 0,
        "stress_cagr_to_strict_mdd_min_2_5": stress["cagr_to_strict_mdd"] >= 2.5,
        "each_calendar_half_positive": all(item["absolute_return_pct"] > 0 for item in halves.values()),
    }
    passed = all(checks.values())
    return {
        "candidate_clock_rows_opened": len(clock),
        "primary": {
            "base": public_metric(base),
            "stress": public_metric(stress),
            "cluster_signflip": cluster,
            "calendar_halves": halves,
        },
        "checks": checks,
        "source_pass": True,
        "exact_duplicate_pass": True,
        "gross9_pass": True,
        "train_economic_pass": passed,
        "train_cagr_to_strict_mdd": base["cagr_to_strict_mdd"],
        "train_absolute_return": base["absolute_return_pct"],
        "decision": "train_pass" if passed else "train_economic_reject",
    }


def _fallback_raw_rank_one(rows_for_selection: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows_for_selection if row["source_pass"] and row["exact_duplicate_pass"] and row["gross9_pass"]]
    eligible.sort(
        key=lambda row: (
            -float(row["train_cagr_to_strict_mdd"]),
            -float(row["train_absolute_return"]),
            list(prereg.CANDIDATE_FAMILY).index(str(row["candidate"])),
        )
    )
    if not eligible:
        return None
    top = eligible[0]
    return {
        "candidate": top["candidate"],
        "train_cagr_to_strict_mdd": top["train_cagr_to_strict_mdd"],
        "train_absolute_return": top["train_absolute_return"],
        "frozen_before_test": True,
        "substitution_authorized": False,
        "rerank_authorized": False,
        "train_economic_pass": False,
    }


def run(output: str | Path = OUTPUT) -> dict[str, Any]:
    report = load_novelty_authorization()
    novelty_sha = sha256_file(NOVELTY)
    market = load_market_hash_bound(TRAIN_START, TRAIN_END)
    funding = load_train_funding_hash_bound(TRAIN_START, TRAIN_END)
    econ.validate_market(market, TRAIN_START, TRAIN_END)
    econ.validate_funding(funding, TRAIN_START, TRAIN_END)

    rows_for_selection: list[dict[str, Any]] = []
    candidates: dict[str, Any] = {}
    economics_evaluated_candidates: list[str] = []
    for candidate in prereg.CANDIDATE_FAMILY:
        novelty_row = report["candidates"][candidate]
        source_pass = novelty_row.get("source_pass") is True
        exact_duplicate_pass = novelty_row.get("exact_duplicate_pass") is True
        gross9_pass = novelty_row.get("gross9_pass") is True
        row: dict[str, Any] = {
            "candidate": candidate,
            "source_pass": source_pass,
            "exact_duplicate_pass": exact_duplicate_pass,
            "gross9_pass": gross9_pass,
            "train_economic_pass": False,
            "train_cagr_to_strict_mdd": 0.0,
            "train_absolute_return": 0.0,
            "decision": "not_evaluated_prereq_failed",
        }
        if source_pass and exact_duplicate_pass and gross9_pass:
            economics_evaluated_candidates.append(candidate)
            row.update(evaluate_candidate(candidate, novelty_row["clock"], market, funding))
        candidates[candidate] = row
        rows_for_selection.append(
            {
                "candidate": candidate,
                "source_pass": row["source_pass"],
                "exact_duplicate_pass": row["exact_duplicate_pass"],
                "gross9_pass": row["gross9_pass"],
                "train_economic_pass": row["train_economic_pass"],
                "train_cagr_to_strict_mdd": row["train_cagr_to_strict_mdd"],
                "train_absolute_return": row["train_absolute_return"],
            }
        )

    raw_rank_one: dict[str, Any] | None = None
    selection_error: str | None = None
    try:
        raw_rank_one = prereg.select_train_winner(rows_for_selection)
    except RuntimeError as exc:
        selection_error = str(exc)
        raw_rank_one = _fallback_raw_rank_one(rows_for_selection)

    train_passed = raw_rank_one is not None and selection_error is None
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": "2026-09-02",
        "stage": "train",
        "window": [_iso_z(TRAIN_START), _iso_z(TRAIN_END)],
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": novelty.PREREGISTRATION_SHA256,
            "manifest_hash": report["preregistration"]["manifest_hash"],
        },
        "source_support": report["source_support"],
        "novelty_authorization": {
            "path": str(NOVELTY),
            "sha256": novelty_sha,
            "manifest_hash": report["manifest_hash"],
        },
        "source": {
            "mode": "hash_bound_gzip_physical_prefix",
            "market_sha256": econ.v1.MARKET_SHA,
            "funding_marks_sha256": econ.TRAIN_FUNDING_SHA,
            "costs": {"base_each_notional_side_bp": 6, "stress_each_notional_side_bp": 10},
            "accounting": "fixed quantity, exact offset funding marks, held 5m favorable/adverse strict MDD, global HWM, full-calendar CAGR",
        },
        "candidate_family": list(prereg.CANDIDATE_FAMILY),
        "candidate_family_size": prereg.FAMILY_SIZE,
        "bonferroni_raw_p_max": prereg.BONFERRONI_RAW_P_MAX,
        "economics_evaluated_candidates": economics_evaluated_candidates,
        "candidates": candidates,
        "selection": {
            "raw_rank_one": raw_rank_one,
            "selection_error": selection_error,
            "train_passed": train_passed,
            "substitution_authorized": False,
            "rerank_authorized": False,
        },
        "physical_rows_opened": {
            "market": len(market),
            "funding": len(funding),
            "candidate_clock": sum(candidates[c].get("candidate_clock_rows_opened", 0) for c in candidates),
        },
        "later_stage_outcomes_opened": False,
        "decision": "freeze_train_winner_before_test" if train_passed else "terminal_train_reject_no_substitution",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "economics_evaluated_candidates": result["economics_evaluated_candidates"],
                "raw_rank_one": result["selection"]["raw_rank_one"],
                "selection_error": result["selection"]["selection_error"],
                "decision": result["decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
