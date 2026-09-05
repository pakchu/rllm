"""Evaluate pre-2024 premium-versus-price path support for PPOSM actions."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from preprocessing.live_db_features import postgres_url_from_env
from training import evaluate_pposm_premium1m_path_two_action as base
from training import preregister_pposm_premium_price_divergence_two_action as prereg


@dataclass(frozen=True)
class Config:
    preregistration: Path = prereg.DEFAULT_OUTPUT
    output: Path = prereg.DEFAULT_RESULT
    labels: Path = prereg.premium.LABELS
    manifest: Path = prereg.premium.PPOSM_MANIFEST
    env_file: Path = Path("/home/pakchu/rllm/.env")


def validate_preregistration(path: Path) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = prereg.build_preregistration()
    strip = lambda item: {key: value for key, value in item.items() if key not in {"created_at", "preregistration_hash"}}
    if observed.get("preregistration_hash") != expected.get("preregistration_hash") or strip(observed) != strip(expected):
        raise RuntimeError("premium-price divergence preregistration drift")
    return observed


def query_sources(env_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sqlalchemy import create_engine, text

    engine = create_engine(postgres_url_from_env(env_file), connect_args={"connect_timeout": 10})
    params = {"symbol": "BTCUSDT", "start": pd.Timestamp(prereg.premium.QUERY_START), "end": pd.Timestamp(prereg.premium.QUERY_END_EXCLUSIVE)}
    try:
        with engine.connect() as connection:
            premium = pd.read_sql_query(text(prereg.premium.PREMIUM_QUERY), connection, params=params, coerce_float=False)
            market = pd.read_sql_query(text(prereg.MARKET_QUERY), connection, params=params, coerce_float=False)
    finally:
        engine.dispose()
    if premium.columns.tolist() != ["ts", "close_time", "close"] or market.columns.tolist() != ["ts", "close"]:
        raise RuntimeError("premium/market query schema drift")
    for frame in (premium, market):
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="raise").dt.tz_convert(None)
        frame.sort_values("ts", kind="mergesort", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if frame["ts"].duplicated().any() or (frame["ts"] < base._naive_utc(prereg.premium.QUERY_START)).any() or (frame["ts"] >= base._naive_utc(prereg.premium.QUERY_END_EXCLUSIVE)).any():
            raise RuntimeError("source timestamp contract failed")
        frame["close_exact"] = frame["close"].map(base._decimal_text)
        frame["close"] = pd.to_numeric(frame["close_exact"], errors="raise")
    premium["close_time"] = pd.to_datetime(premium["close_time"], utc=True, errors="raise").dt.tz_convert(None)
    if (market["close"] <= 0).any():
        raise RuntimeError("market close must be positive")
    return premium, market


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or float(np.std(left)) <= prereg.NUMERIC_EPSILON or float(np.std(right)) <= prereg.NUMERIC_EPSILON:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def divergence_features(premium_close: np.ndarray, market_close: np.ndarray) -> dict[str, float]:
    premium_close = np.asarray(premium_close, dtype=float)
    market_close = np.asarray(market_close, dtype=float)
    if len(premium_close) != 60 or len(market_close) != 60 or not np.isfinite(premium_close).all() or not np.isfinite(market_close).all() or np.any(market_close <= 0):
        raise ValueError("divergence paths must be exact finite 60-row paths")
    premium_increment = np.diff(premium_close)
    price_level = np.log(market_close)
    price_increment = np.diff(price_level)
    price_variance = float(np.var(price_increment))
    premium_total = float(np.abs(premium_increment).sum())
    price_total = float(np.abs(price_increment).sum())
    premium_efficiency = 0.0 if premium_total <= prereg.NUMERIC_EPSILON else float((premium_close[-1] - premium_close[0]) / premium_total)
    price_efficiency = 0.0 if price_total <= prereg.NUMERIC_EPSILON else float((price_level[-1] - price_level[0]) / price_total)
    premium_std = float(np.std(premium_close))
    price_std = float(np.std(price_level))
    premium_terminal_z = 0.0 if premium_std <= prereg.NUMERIC_EPSILON else float((premium_close[-1] - premium_close.mean()) / premium_std)
    price_terminal_z = 0.0 if price_std <= prereg.NUMERIC_EPSILON else float((price_level[-1] - price_level.mean()) / price_std)
    premium_half = 0.0 if premium_std <= prereg.NUMERIC_EPSILON else float((premium_close[30:].mean() - premium_close[:30].mean()) / premium_std)
    price_half = 0.0 if price_std <= prereg.NUMERIC_EPSILON else float((price_level[30:].mean() - price_level[:30].mean()) / price_std)
    values = {
        "increment_correlation": _safe_corr(premium_increment, price_increment),
        "premium_leads_price_correlation": _safe_corr(premium_increment[:-1], price_increment[1:]),
        "price_leads_premium_correlation": _safe_corr(price_increment[:-1], premium_increment[1:]),
        "signed_comovement_mean": float(np.mean(np.sign(premium_increment) * np.sign(price_increment))),
        "premium_beta_on_price": 0.0 if price_variance <= prereg.NUMERIC_EPSILON else float(np.cov(premium_increment, price_increment, ddof=0)[0, 1] / price_variance),
        "path_efficiency_difference": premium_efficiency - price_efficiency,
        "terminal_zscore_difference": premium_terminal_z - price_terminal_z,
        "half_drift_zscore_difference": premium_half - price_half,
    }
    if tuple(values) != prereg.FEATURE_COLUMNS or not np.isfinite(list(values.values())).all():
        raise RuntimeError("divergence feature contract failed")
    return values


def attach_features(labels: pd.DataFrame, premium: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    premium_index = premium.set_index("ts", drop=False)
    market_index = market.set_index("ts", drop=False)
    records: list[dict[str, Any]] = []
    for anchor in labels.itertuples(index=False):
        decision = base._naive_utc(anchor.decision_time)
        expected = pd.date_range(decision - pd.Timedelta(minutes=60), decision - pd.Timedelta(minutes=1), freq="1min")
        premium_block = premium_index.reindex(expected)
        market_block = market_index.reindex(expected)
        valid = bool(premium_block["ts"].notna().all() and market_block["ts"].notna().all())
        reason = "pass" if valid else "missing_minute"
        if valid:
            causal = (premium_block["close_time"] >= premium_block["ts"]) & (premium_block["close_time"] < premium_block["ts"] + pd.Timedelta(minutes=1)) & (premium_block["close_time"] <= decision)
            valid = bool(causal.all())
            if not valid:
                reason = "invalid_close_time"
        features = {name: np.nan for name in prereg.FEATURE_COLUMNS}
        if valid:
            features = divergence_features(premium_block["close"].to_numpy(float), market_block["close"].to_numpy(float))
        records.append({"base_identity": anchor.base_identity, "source_valid": valid, "source_reason": reason, **features})
    frame = labels.merge(pd.DataFrame(records), on="base_identity", validate="one_to_one")
    years = frame["decision_time"].dt.year.astype(str)
    valid = frame["source_valid"].to_numpy(bool)
    by_year = {year: {"valid": int(valid[years == year].sum()), "total": int((years == year).sum()), "coverage": float(valid[years == year].mean())} for year in sorted(years.unique())}
    overall = float(valid.mean())
    gates = prereg.premium.COVERAGE_GATES
    passed = overall >= gates["overall_min"] and all(item["coverage"] >= gates["per_year_min"] for item in by_year.values())
    return frame, {"valid": int(valid.sum()), "total": len(frame), "overall": overall, "by_year": by_year, "gates": gates, "passed": passed}


def expanding_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    scored = frame.copy()
    reports: dict[str, list[dict[str, Any]]] = {action: [] for action in base.ACTIONS}
    for action in base.ACTIONS:
        scored[f"score_{action}"] = np.nan
    for fold in prereg.premium.FOLDS:
        year = int(fold["test_year"])
        train, test = base.fold_masks(scored, year)
        for action in base.ACTIONS:
            y_train = scored.loc[train, action].to_numpy(int)
            y_test = scored.loc[test, action].to_numpy(int)
            report: dict[str, Any] = {"fold": fold["name"], "test_year": year, "train": int(train.sum()), "test": int(test.sum()), "evaluated": False}
            if int(test.sum()) < prereg.premium.COVERAGE_GATES["test_fold_min"] or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                report["reason"] = "fold_coverage_or_class_failure"
                reports[action].append(report)
                continue
            model = base._model()
            model.fit(scored.loc[train, prereg.FEATURE_COLUMNS].to_numpy(float), y_train)
            score = model.predict_proba(scored.loc[test, prereg.FEATURE_COLUMNS].to_numpy(float))[:, 1]
            scored.loc[test, f"score_{action}"] = score
            report.update({"evaluated": True, "auc": float(roc_auc_score(y_test, score)), "balanced_accuracy": float(balanced_accuracy_score(y_test, score >= 0.5))})
            reports[action].append(report)
    eligible = scored["source_valid"] & scored["decision_time"].dt.year.isin([2021, 2022, 2023])
    for year in (2021, 2022, 2023):
        eligible &= ~((scored["decision_time"].dt.year == year) & ~(scored["maturity_time"] < pd.Timestamp(f"{year + 1}-01-01")))
    if any(scored.loc[eligible, f"score_{action}"].isna().any() for action in base.ACTIONS):
        raise RuntimeError("divergence OOF scores are incomplete")
    return scored.loc[eligible].reset_index(drop=True), reports


def evaluate(cfg: Config = Config()) -> dict[str, Any]:
    registration = validate_preregistration(cfg.preregistration)
    labels = base.load_pair_labels(cfg.labels, cfg.manifest)
    label_receipt = {
        "rows": 924,
        "anchors": len(labels),
        "sha256": prereg.premium.LABELS_SHA256,
        "canonical_maturity_sha256": base.frame_csv_hash(
            labels,
            ["base_identity", "signal_position", "decision_time", "maturity_time", "SKIP", "TP12"],
        ),
    }
    premium, market = query_sources(cfg.env_file)
    frame, coverage = attach_features(labels, premium, market)
    if not coverage["passed"]:
        core: dict[str, Any] = {"protocol": "pposm_premium_price_divergence_two_action_source_support_v1", "terminal": "source_coverage_fail", "preregistration_hash": registration["preregistration_hash"], "coverage": coverage, "labels": label_receipt, "support_passed": False, "metrics_computed": False, "oos_opened": False, "fresh_outcomes_opened": False, "sft_or_rlvr_started": False}
    else:
        scored, folds = expanding_scores(frame)
        bounds = base.paired_simultaneous_bounds(scored)
        actions: dict[str, Any] = {}
        for action in base.ACTIONS:
            target = scored[action].to_numpy(int); score = scored[f"score_{action}"].to_numpy(float)
            auc = float(roc_auc_score(target, score)); bacc = float(balanced_accuracy_score(target, score >= 0.5))
            year_auc = {str(year): float(roc_auc_score(scored.loc[scored["decision_time"].dt.year == year, action], scored.loc[scored["decision_time"].dt.year == year, f"score_{action}"])) for year in (2021, 2022, 2023)}
            gates = prereg.premium.SUPPORT_GATES
            checks = {"pooled_auc": auc >= gates["pooled_auc_min"], "simultaneous_lower95": bounds["simultaneous_lower95"][action] > gates["simultaneous_auc_lower95_strict_min"], "balanced_accuracy": bacc >= gates["balanced_accuracy_min"], "consistent_year_direction": all(value > gates["each_year_auc_strict_min"] for value in year_auc.values()), "all_folds_evaluated": all(item["evaluated"] for item in folds[action])}
            actions[action] = {"auc": auc, "balanced_accuracy": bacc, "simultaneous_lower95_auc": bounds["simultaneous_lower95"][action], "year_auc": year_auc, "folds": folds[action], "checks": checks, "passed": all(checks.values())}
        passed = all(item["passed"] for item in actions.values())
        score_rows = scored[["base_identity", "decision_time", "SKIP", "TP12", "score_SKIP", "score_TP12"]]
        core = {"protocol": "pposm_premium_price_divergence_two_action_source_support_v1", "terminal": "pass_to_separate_sft_rlvr_preregistration" if passed else "pre2024_source_support_reject", "preregistration_hash": registration["preregistration_hash"], "coverage": coverage, "sources": {"premium_rows": len(premium), "market_rows": len(market), "premium_sha256": base.frame_csv_hash(premium, ["ts", "close_time", "close_exact"]), "market_sha256": base.frame_csv_hash(market, ["ts", "close_exact"])}, "labels": label_receipt, "features": {"columns": list(prereg.FEATURE_COLUMNS), "sha256": base.frame_csv_hash(frame, ["base_identity", "source_valid", *prereg.FEATURE_COLUMNS])}, "scores": {"rows": len(scored), "sha256": base.frame_csv_hash(score_rows, list(score_rows.columns))}, "bootstrap": {**prereg.premium.BOOTSTRAP, **bounds}, "actions": actions, "support_passed": passed, "metrics_computed": True, "oos_opened": False, "fresh_outcomes_opened": False, "sft_or_rlvr_started": False}
    core["result_hash"] = prereg.premium.sha256_bytes(prereg.premium.canonical_json(core).encode())
    payload = json.dumps(core, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n"
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    if cfg.output.exists() and json.loads(cfg.output.read_text()).get("result_hash") != core["result_hash"]:
        raise RuntimeError("refusing to overwrite different divergence result")
    cfg.output.write_text(payload, encoding="utf-8")
    return core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=prereg.DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=prereg.DEFAULT_RESULT)
    parser.add_argument("--labels", type=Path, default=prereg.premium.LABELS)
    parser.add_argument("--manifest", type=Path, default=prereg.premium.PPOSM_MANIFEST)
    parser.add_argument("--env-file", type=Path, default=Path("/home/pakchu/rllm/.env"))
    return parser.parse_args()


if __name__ == "__main__":
    result = evaluate(Config(**vars(parse_args())))
    print(json.dumps({"terminal": result["terminal"], "support_passed": result.get("support_passed", False), "actions": result.get("actions", {})}, indent=2, default=str))
