"""Evaluate strict pre-2024 premium-path support for both PPOSM actions."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing.live_db_features import postgres_url_from_env
from training import build_pposm_sft_rlvr_data as frozen
from training import preregister_pposm_premium1m_path_two_action as prereg

ACTIONS = ("SKIP", "TP12")


@dataclass(frozen=True)
class Config:
    preregistration: Path = prereg.DEFAULT_OUTPUT
    output: Path = prereg.DEFAULT_RESULT
    labels: Path = prereg.LABELS
    manifest: Path = prereg.PPOSM_MANIFEST
    env_file: Path = Path("/home/pakchu/rllm/.env")


def _naive_utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_convert("UTC").tz_localize(None) if timestamp.tzinfo is not None else timestamp


def _iso_z(value: Any) -> str:
    return _naive_utc(value).isoformat() + "Z"


def _decimal_text(value: Any) -> str:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"invalid premium close: {value!r}") from exc
    if not number.is_finite():
        raise ValueError(f"non-finite premium close: {value!r}")
    return format(number, "f")


def validate_preregistration(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = prereg.build_preregistration()
    strip = lambda item: {k: v for k, v in item.items() if k not in {"created_at", "preregistration_hash"}}
    if observed.get("preregistration_hash") != expected.get("preregistration_hash") or strip(observed) != strip(expected):
        raise RuntimeError("premium1m path preregistration drift")
    return observed


def query_premium(env_file: Path) -> pd.DataFrame:
    from sqlalchemy import create_engine, text

    engine = create_engine(postgres_url_from_env(env_file), connect_args={"connect_timeout": 10})
    try:
        with engine.connect() as connection:
            frame = pd.read_sql_query(
                text(prereg.PREMIUM_QUERY),
                connection,
                params={"symbol": "BTCUSDT", "start": pd.Timestamp(prereg.QUERY_START), "end": pd.Timestamp(prereg.QUERY_END_EXCLUSIVE)},
                coerce_float=False,
            )
    finally:
        engine.dispose()
    if frame.columns.tolist() != ["ts", "close_time", "close"]:
        raise RuntimeError(f"premium query schema drift: {frame.columns.tolist()}")
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="raise").dt.tz_convert(None)
    frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True, errors="raise").dt.tz_convert(None)
    frame = frame.sort_values("ts", kind="mergesort").reset_index(drop=True)
    if frame["ts"].duplicated().any() or (frame["ts"] < _naive_utc(prereg.QUERY_START)).any() or (frame["ts"] >= _naive_utc(prereg.QUERY_END_EXCLUSIVE)).any():
        raise RuntimeError("premium source timestamp contract failed")
    frame["close_exact"] = frame["close"].map(_decimal_text)
    frame["close"] = pd.to_numeric(frame["close_exact"], errors="raise")
    return frame


def _load_market_for_maturity(manifest_path: Path) -> pd.DataFrame:
    manifest, strategy_cfg = frozen.load_frozen_manifest(manifest_path)
    market, raw_features, _, source_hashes = frozen._load_bundle(
        strategy_cfg,
        cutoff="2024-01-01",
        premium_tolerance=strategy_cfg.live_premium_tolerance,
    )
    if source_hashes != manifest.get("source_prefix_hashes"):
        raise RuntimeError("frozen pre-2024 source hashes changed")
    _, state, _, base_thresholds = frozen._state_inputs(market, raw_features, strategy_cfg)
    if base_thresholds != manifest.get("base_thresholds") or frozen.pposm.feature_hash(state) != manifest.get("feature_prefix_hash"):
        raise RuntimeError("frozen pre-2024 feature identity changed")
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"])
    return market


def load_pair_labels(path: Path, manifest_path: Path) -> pd.DataFrame:
    if prereg.sha256_file(path) != prereg.LABELS_SHA256:
        raise RuntimeError("pairwise train label bytes changed")
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw) != 924:
        raise RuntimeError(f"expected 924 pair rows, got {len(raw)}")
    market = _load_market_for_maturity(manifest_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in raw:
        if row.get("task") != "pposm_residual_action" or row.get("split") != "train":
            raise RuntimeError("pairwise label row task/split changed")
        if row.get("target") not in {"KEEP", "SWITCH"}:
            raise RuntimeError("pairwise label target changed")
        metadata = row.get("metadata", {})
        grouped.setdefault(str(metadata.get("base_identity")), []).append(row)
    records: list[dict[str, Any]] = []
    for base_identity, rows in sorted(grouped.items()):
        if len(rows) != 2:
            raise RuntimeError(f"base anchor lacks two action rows: {base_identity}")
        candidates = {str(row["metadata"].get("candidate_action")): row for row in rows}
        if set(candidates) != set(ACTIONS):
            raise RuntimeError(f"base anchor action set changed: {base_identity}")
        first = rows[0]["metadata"]
        if any(row["metadata"].get("signal_position") != first.get("signal_position") or row["metadata"].get("signal_time") != first.get("signal_time") for row in rows):
            raise RuntimeError(f"paired anchor identity mismatch: {base_identity}")
        exits = first.get("executable_positions", {})
        if any(row["metadata"].get("executable_positions") != exits for row in rows):
            raise RuntimeError(f"paired executable positions mismatch: {base_identity}")
        exit_position = max(int(exits[action]["exit_position"]) for action in ("TP4", "TP12"))
        signal_position = int(first["signal_position"])
        if signal_position < 0 or signal_position >= len(market) or exit_position >= len(market):
            raise RuntimeError("label maturity position exceeds frozen market")
        signal_time = _naive_utc(first["signal_time"])
        if _naive_utc(market.iloc[signal_position]["date"]) != signal_time:
            raise RuntimeError("pairwise signal position/time changed")
        # Exit-bar high/low/close becomes known only at completion, not at its
        # open timestamp.
        maturity_time = _naive_utc(market.iloc[exit_position]["date"]) + pd.Timedelta(minutes=5)
        if maturity_time >= _naive_utc(prereg.QUERY_END_EXCLUSIVE):
            raise RuntimeError("pre-2024 label does not mature before source boundary")
        records.append(
            {
                "base_identity": base_identity,
                "signal_position": signal_position,
                "decision_time": signal_time,
                "maturity_time": maturity_time,
                "SKIP": int(candidates["SKIP"]["target"] == "SWITCH"),
                "TP12": int(candidates["TP12"]["target"] == "SWITCH"),
            }
        )
    frame = pd.DataFrame(records).sort_values("decision_time", kind="mergesort").reset_index(drop=True)
    if len(frame) != 462 or frame["base_identity"].duplicated().any():
        raise RuntimeError("pairwise base-anchor population changed")
    return frame


def path_features(close: np.ndarray) -> dict[str, float]:
    close = np.asarray(close, dtype=float)
    if len(close) != 60 or not np.isfinite(close).all():
        raise ValueError("premium path must contain exactly 60 finite closes")
    increments = np.diff(close)
    total_variation = float(np.abs(increments).sum())
    quadratic_variation = float(np.sqrt(np.square(increments).sum()))
    nonzero_signs = np.sign(increments[increments != 0.0])
    reversals = int((nonzero_signs[1:] != nonzero_signs[:-1]).sum()) if len(nonzero_signs) > 1 else 0
    level_signs = np.sign(close)
    return {
        "path_total_variation": total_variation,
        "path_quadratic_variation": quadratic_variation,
        "signed_path_efficiency": 0.0 if total_variation == 0.0 else float((close[-1] - close[0]) / total_variation),
        "first_half_variation_share": 0.0 if total_variation == 0.0 else float(np.abs(increments[:29]).sum() / total_variation),
        "last_quarter_variation_share": 0.0 if total_variation == 0.0 else float(np.abs(increments[44:]).sum() / total_variation),
        "positive_occupation_centered": float(np.mean(close > 0.0) - 0.5),
        "level_zero_cross_rate": float((level_signs[1:] != level_signs[:-1]).sum() / 59.0),
        "increment_reversal_rate": 0.0 if len(nonzero_signs) <= 1 else float(reversals / (len(nonzero_signs) - 1)),
    }


def attach_features(labels: pd.DataFrame, premium: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = premium.set_index("ts", drop=False)
    records: list[dict[str, Any]] = []
    for anchor in labels.itertuples(index=False):
        decision = _naive_utc(anchor.decision_time)
        expected = pd.date_range(decision - pd.Timedelta(minutes=60), decision - pd.Timedelta(minutes=1), freq="1min")
        block = source.reindex(expected)
        present = block["ts"].notna()
        valid = bool(present.all())
        reason = "pass" if valid else "missing_minute"
        if valid:
            causal = (block["close_time"] >= block["ts"]) & (block["close_time"] < block["ts"] + pd.Timedelta(minutes=1)) & (block["close_time"] <= decision)
            valid = bool(causal.all())
            if not valid:
                reason = "invalid_close_time"
        features = {name: np.nan for name in prereg.FEATURE_COLUMNS}
        if valid:
            features = path_features(block["close"].to_numpy(float))
        records.append({"base_identity": anchor.base_identity, "source_valid": valid, "source_reason": reason, **features})
    joined = labels.merge(pd.DataFrame(records), on="base_identity", validate="one_to_one")
    year = joined["decision_time"].dt.year.astype(str)
    valid = joined["source_valid"].astype(bool)
    by_year = {key: {"valid": int(valid[year == key].sum()), "total": int((year == key).sum()), "coverage": float(valid[year == key].mean())} for key in sorted(year.unique())}
    overall = float(valid.mean())
    passed = overall >= prereg.COVERAGE_GATES["overall_min"] and all(item["coverage"] >= prereg.COVERAGE_GATES["per_year_min"] for item in by_year.values())
    return joined, {"valid": int(valid.sum()), "total": len(joined), "overall": overall, "by_year": by_year, "gates": prereg.COVERAGE_GATES, "passed": passed}


def fold_masks(frame: pd.DataFrame, test_year: int) -> tuple[np.ndarray, np.ndarray]:
    start = pd.Timestamp(f"{test_year}-01-01")
    end = pd.Timestamp(f"{test_year + 1}-01-01")
    valid = frame["source_valid"].to_numpy(bool)
    train = valid & (frame["decision_time"] < start).to_numpy(bool) & (frame["maturity_time"] < start).to_numpy(bool)
    test = valid & (frame["decision_time"] >= start).to_numpy(bool) & (frame["decision_time"] < end).to_numpy(bool) & (frame["maturity_time"] < end).to_numpy(bool)
    return train, test


def _model() -> Any:
    return make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", C=0.25, solver="liblinear", class_weight="balanced", max_iter=1000, random_state=20260906))


def expanding_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    scored = frame.copy()
    reports: dict[str, list[dict[str, Any]]] = {action: [] for action in ACTIONS}
    for action in ACTIONS:
        scored[f"score_{action}"] = np.nan
    for fold in prereg.FOLDS:
        year = int(fold["test_year"])
        train, test = fold_masks(scored, year)
        for action in ACTIONS:
            y_train = scored.loc[train, action].to_numpy(int)
            y_test = scored.loc[test, action].to_numpy(int)
            report: dict[str, Any] = {"fold": fold["name"], "test_year": year, "train": int(train.sum()), "test": int(test.sum()), "purged_train": int(((scored["decision_time"] < pd.Timestamp(f"{year}-01-01")) & ~(scored["maturity_time"] < pd.Timestamp(f"{year}-01-01"))).sum()), "evaluated": False}
            if int(test.sum()) < prereg.COVERAGE_GATES["test_fold_min"] or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                report["reason"] = "fold_coverage_or_class_failure"
                reports[action].append(report)
                continue
            model = _model()
            model.fit(scored.loc[train, prereg.FEATURE_COLUMNS].to_numpy(float), y_train)
            score = model.predict_proba(scored.loc[test, prereg.FEATURE_COLUMNS].to_numpy(float))[:, 1]
            scored.loc[test, f"score_{action}"] = score
            report.update({"evaluated": True, "auc": float(roc_auc_score(y_test, score)), "balanced_accuracy": float(balanced_accuracy_score(y_test, score >= 0.5)), "positives": int(y_test.sum()), "negatives": int(len(y_test) - y_test.sum())})
            reports[action].append(report)
    eligible = scored["source_valid"] & scored["decision_time"].dt.year.isin([2021, 2022, 2023])
    for year in (2021, 2022, 2023):
        eligible &= ~((scored["decision_time"].dt.year == year) & ~(scored["maturity_time"] < pd.Timestamp(f"{year + 1}-01-01")))
    if any(scored.loc[eligible, f"score_{action}"].isna().any() for action in ACTIONS):
        raise RuntimeError("OOF scores do not cover every valid mature validation anchor")
    return scored.loc[eligible].reset_index(drop=True), reports


def stationary_indices(years: np.ndarray, *, rng: np.random.Generator, mean_block: int = 8) -> np.ndarray:
    output: list[int] = []
    for year in sorted(np.unique(years)):
        positions = np.flatnonzero(years == year)
        current = int(rng.integers(len(positions)))
        for step in range(len(positions)):
            if step and rng.random() < 1.0 / mean_block:
                current = int(rng.integers(len(positions)))
            elif step:
                current = (current + 1) % len(positions)
            output.append(int(positions[current]))
    return np.asarray(output, dtype=int)


def paired_simultaneous_bounds(scored: pd.DataFrame) -> dict[str, Any]:
    observed = {action: float(roc_auc_score(scored[action], scored[f"score_{action}"])) for action in ACTIONS}
    rng = np.random.default_rng(prereg.BOOTSTRAP["random_seed"])
    years = scored["decision_time"].dt.year.to_numpy(int)
    draws: dict[str, list[float]] = {action: [] for action in ACTIONS}
    losses: list[float] = []
    for _ in range(prereg.BOOTSTRAP["iterations"]):
        take = stationary_indices(years, rng=rng, mean_block=prereg.BOOTSTRAP["expected_block_length_anchors"])
        values: dict[str, float] = {}
        for action in ACTIONS:
            y = scored[action].to_numpy(int)[take]
            if len(np.unique(y)) < 2:
                break
            values[action] = float(roc_auc_score(y, scored[f"score_{action}"].to_numpy(float)[take]))
        if len(values) != len(ACTIONS):
            continue
        for action in ACTIONS:
            draws[action].append(values[action])
        losses.append(max(observed[action] - values[action] for action in ACTIONS))
    if len(losses) < prereg.BOOTSTRAP["iterations"] * 0.95:
        raise RuntimeError("too many degenerate paired bootstrap draws")
    penalty = float(np.quantile(losses, 0.95))
    return {"observed_auc": observed, "valid_iterations": len(losses), "simultaneous_loss_q95": penalty, "simultaneous_lower95": {action: observed[action] - penalty for action in ACTIONS}, "individual_lower95": {action: float(np.quantile(draws[action], 0.05)) for action in ACTIONS}}


class _HashSink:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()
    def write(self, value: str) -> None:
        self.digest.update(value.encode())


def frame_csv_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    sink = _HashSink()
    frame.loc[:, columns].to_csv(sink, index=False, lineterminator="\n")
    return sink.digest.hexdigest()


def evaluate(cfg: Config = Config()) -> dict[str, Any]:
    registration = validate_preregistration(cfg.preregistration)
    labels = load_pair_labels(cfg.labels, cfg.manifest)
    premium = query_premium(cfg.env_file)
    frame, coverage = attach_features(labels, premium)
    if not coverage["passed"]:
        core = {"protocol": "pposm_premium1m_path_two_action_source_support_v1", "preregistration_hash": registration["preregistration_hash"], "terminal": "source_coverage_fail", "coverage": coverage, "metrics_computed": False, "oos_opened": False, "sft_or_rlvr_started": False}
    else:
        scored, folds = expanding_scores(frame)
        bounds = paired_simultaneous_bounds(scored)
        actions: dict[str, Any] = {}
        for action in ACTIONS:
            score = scored[f"score_{action}"].to_numpy(float)
            target = scored[action].to_numpy(int)
            auc = float(roc_auc_score(target, score))
            bacc = float(balanced_accuracy_score(target, score >= 0.5))
            year_auc = {str(year): float(roc_auc_score(scored.loc[scored["decision_time"].dt.year == year, action], scored.loc[scored["decision_time"].dt.year == year, f"score_{action}"])) for year in (2021, 2022, 2023)}
            checks = {"pooled_auc": auc >= prereg.SUPPORT_GATES["pooled_auc_min"], "simultaneous_lower95": bounds["simultaneous_lower95"][action] > prereg.SUPPORT_GATES["simultaneous_auc_lower95_strict_min"], "balanced_accuracy": bacc >= prereg.SUPPORT_GATES["balanced_accuracy_min"], "consistent_year_direction": all(value > prereg.SUPPORT_GATES["each_year_auc_strict_min"] for value in year_auc.values()), "all_folds_evaluated": all(item["evaluated"] for item in folds[action])}
            actions[action] = {"auc": auc, "balanced_accuracy": bacc, "simultaneous_lower95_auc": bounds["simultaneous_lower95"][action], "individual_lower95_auc": bounds["individual_lower95"][action], "year_auc": year_auc, "folds": folds[action], "checks": checks, "passed": all(checks.values())}
        passed = all(item["passed"] for item in actions.values())
        score_rows = scored[["base_identity", "decision_time", "SKIP", "TP12", "score_SKIP", "score_TP12"]].copy()
        core = {"protocol": "pposm_premium1m_path_two_action_source_support_v1", "preregistration_hash": registration["preregistration_hash"], "terminal": "pass_to_separate_sft_rlvr_preregistration" if passed else "pre2024_source_support_reject", "coverage": coverage, "source": {"rows": len(premium), "first": _iso_z(premium["ts"].min()), "last": _iso_z(premium["ts"].max()), "sha256": frame_csv_hash(premium, ["ts", "close_time", "close_exact"])}, "labels": {"rows": 924, "anchors": len(labels), "sha256": prereg.LABELS_SHA256, "canonical_maturity_sha256": frame_csv_hash(labels, ["base_identity", "signal_position", "decision_time", "maturity_time", "SKIP", "TP12"])}, "features": {"columns": list(prereg.FEATURE_COLUMNS), "sha256": frame_csv_hash(frame, ["base_identity", "source_valid", *prereg.FEATURE_COLUMNS])}, "scores": {"rows": len(scored), "sha256": frame_csv_hash(score_rows, list(score_rows.columns))}, "bootstrap": {**prereg.BOOTSTRAP, **bounds}, "actions": actions, "support_passed": passed, "metrics_computed": True, "oos_opened": False, "fresh_outcomes_opened": False, "sft_or_rlvr_started": False}
    core["result_hash"] = prereg.sha256_bytes(prereg.canonical_json(core).encode())
    payload = json.dumps(core, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n"
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    if cfg.output.exists() and json.loads(cfg.output.read_text()).get("result_hash") != core["result_hash"]:
        raise RuntimeError("refusing to overwrite different premium1m path result")
    cfg.output.write_text(payload, encoding="utf-8")
    return core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=prereg.DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=prereg.DEFAULT_RESULT)
    parser.add_argument("--labels", type=Path, default=prereg.LABELS)
    parser.add_argument("--manifest", type=Path, default=prereg.PPOSM_MANIFEST)
    parser.add_argument("--env-file", type=Path, default=Path("/home/pakchu/rllm/.env"))
    return parser.parse_args()


if __name__ == "__main__":
    result = evaluate(Config(**vars(parse_args())))
    print(json.dumps({"terminal": result["terminal"], "support_passed": result.get("support_passed", False), "actions": result.get("actions", {})}, indent=2, default=str))
