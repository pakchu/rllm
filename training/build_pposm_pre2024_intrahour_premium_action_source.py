"""Evaluate pre-2024 intrahour-premium support for PPOSM residual actions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import build_pposm_counterfactual_action_data as counterfactual
from training import build_pposm_sft_rlvr_data as frozen
from training import build_pposm_state_router_data as numeric
from training.audit_confirmed_pullback_squeeze_live_parity import _execution_config
from training.search_inventory_purge_reclaim_alpha import ExecutionEngine
from training import preregister_pposm_pre2024_intrahour_premium_action_source as prereg

DEFAULT_OUTPUT = Path("results/pposm_pre2024_intrahour_premium_action_source_2026-09-05.json")
DEFAULT_ENV_FILE = Path("/home/pakchu/rllm/.env")
CANDIDATES = ("SKIP", "TP12")
FEATURE_WINDOWS = tuple(prereg.FEATURE_WINDOWS_MINUTES)
EPS = 1e-12


@dataclass(frozen=True)
class Config:
    preregistration: Path = prereg.DEFAULT_OUTPUT
    output: Path = DEFAULT_OUTPUT
    env_file: Path = DEFAULT_ENV_FILE
    manifest: Path = numeric.DEFAULT_MANIFEST


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def immutable_write_json(path: Path, obj: dict[str, Any]) -> None:
    payload = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("result_hash") != obj.get("result_hash"):
            raise RuntimeError(f"refusing to overwrite different result: {path}")
    path.write_text(payload, encoding="utf-8")


def load_preregistration(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("policy_id") != prereg.POLICY_ID:
        raise RuntimeError("wrong preregistration policy_id")
    expected = data.get("preregistration_hash")
    observed = sha256_bytes(canonical_json({k: v for k, v in data.items() if k != "preregistration_hash"}).encode("utf-8"))
    if expected != observed:
        raise RuntimeError("preregistration hash mismatch")
    return data


def db_engine(env_file: Path):
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(env_file)
    return create_engine(postgres_url_from_env(env_file), connect_args={"connect_timeout": 10})


def query_premium(env_file: Path) -> pd.DataFrame:
    from sqlalchemy import text

    engine = db_engine(env_file)
    try:
        with engine.connect() as conn:
            frame = pd.read_sql_query(
                text(prereg.PREMIUM_QUERY),
                conn,
                params={"symbol": "BTCUSDT", "start": pd.Timestamp(prereg.QUERY_START), "end": pd.Timestamp(prereg.QUERY_END_EXCLUSIVE)},
            )
    finally:
        engine.dispose()
    expected_cols = ["date", "open", "high", "low", "close"]
    if frame.columns.tolist() != expected_cols:
        raise RuntimeError(f"premium query schema drift: {frame.columns.tolist()}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame


def pre2024_label_frame(manifest_path: Path) -> pd.DataFrame:
    manifest, strategy_cfg = frozen.load_frozen_manifest(manifest_path)
    market, funding, state, active = numeric.replay_frozen_decisions(manifest, strategy_cfg)
    positions = numeric.decision_positions(market, active).get("pre_2024", [])
    engine = ExecutionEngine(market, funding, _execution_config(strategy_cfg, strategy_cfg.leverage))
    rows: list[dict[str, Any]] = []
    for pos in positions:
        trades = counterfactual.counterfactual_trades(engine, int(pos), manifest)
        utilities = counterfactual.action_utilities(trades, strategy_cfg)
        signal_time = pd.Timestamp(market.iloc[int(pos)]["date"], tz="UTC") if pd.Timestamp(market.iloc[int(pos)]["date"]).tz is None else pd.Timestamp(market.iloc[int(pos)]["date"]).tz_convert("UTC")
        if signal_time >= pd.Timestamp("2024-01-01T00:00:00Z"):
            raise RuntimeError("non-pre2024 label attempted")
        rows.append(
            {
                "signal_position": int(pos),
                "decision_time": signal_time,
                "SKIP": int(float(utilities["SKIP"]) > float(utilities["TP4"])),
                "TP12": int(float(utilities["TP12"]) > float(utilities["TP4"])),
                "tp4_utility": float(utilities["TP4"]),
                "skip_advantage": float(utilities["SKIP"]) - float(utilities["TP4"]),
                "tp12_advantage": float(utilities["TP12"]) - float(utilities["TP4"]),
            }
        )
    out = pd.DataFrame(rows).sort_values("decision_time").reset_index(drop=True)
    if out.empty or out["decision_time"].max() >= pd.Timestamp("2024-01-01T00:00:00Z"):
        raise RuntimeError("pre-2024 labels unavailable or contaminated")
    return out


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def feature_row(decision_time: pd.Timestamp, premium: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {"decision_time": decision_time, "source_valid": True, "invalid_reason": "pass"}
    for minutes in FEATURE_WINDOWS:
        start = decision_time - pd.Timedelta(minutes=minutes)
        end = decision_time - pd.Timedelta(minutes=1)
        block = premium[(premium["date"] >= start) & (premium["date"] < decision_time)]
        prefix = f"p{minutes}m"
        if len(block) != minutes or block["date"].nunique() != minutes or block["date"].min() != start or block["date"].max() != end:
            row["source_valid"] = False
            row["invalid_reason"] = f"{prefix}_grid"
            continue
        vals = block[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        if (not np.isfinite(vals.to_numpy(dtype=float)).all()) or (vals["high"] < vals[["open", "close", "low"]].max(axis=1)).any() or (vals["low"] > vals[["open", "close", "high"]].min(axis=1)).any():
            row["source_valid"] = False
            row["invalid_reason"] = f"{prefix}_finite_or_ohlc"
            continue
        close = vals["close"].to_numpy(dtype=float)
        op = vals["open"].to_numpy(dtype=float)
        high = vals["high"].to_numpy(dtype=float)
        low = vals["low"].to_numpy(dtype=float)
        diffs = np.diff(close)
        std = float(np.std(close, ddof=0))
        half = minutes // 2
        body_high = np.maximum(op, close)
        body_low = np.minimum(op, close)
        feats = {
            f"{prefix}_close_mean": float(np.mean(close)),
            f"{prefix}_close_std": std,
            f"{prefix}_close_min": float(np.min(close)),
            f"{prefix}_close_max": float(np.max(close)),
            f"{prefix}_close_range": float(np.max(close) - np.min(close)),
            f"{prefix}_close_last_minus_first": float(close[-1] - close[0]),
            f"{prefix}_close_second_half_minus_first_half": float(np.mean(close[half:]) - np.mean(close[:half])),
            f"{prefix}_close_abs_diff_sum": float(np.sum(np.abs(diffs))) if len(diffs) else 0.0,
            f"{prefix}_positive_close_share": float(np.mean(close > 0.0)),
            f"{prefix}_negative_close_share": float(np.mean(close < 0.0)),
            f"{prefix}_end_zscore": float((close[-1] - np.mean(close)) / (std + EPS)),
            f"{prefix}_ohlc_high_low_range": float(np.max(high) - np.min(low)),
            f"{prefix}_upper_wick_mean": float(np.mean(high - body_high)),
            f"{prefix}_lower_wick_mean": float(np.mean(body_low - low)),
        }
        if not _finite(feats.values()):
            row["source_valid"] = False
            row["invalid_reason"] = f"{prefix}_nonfinite_feature"
        row.update(feats)
    return row


def build_feature_frame(labels: pd.DataFrame, premium: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame([feature_row(pd.Timestamp(t).tz_convert("UTC"), premium) for t in labels["decision_time"]])
    merged = labels.merge(features, on="decision_time", validate="one_to_one")
    feature_cols = [c for c in merged.columns if c.startswith("p") and c not in {"position"}]
    merged["source_valid"] = merged["source_valid"].astype(bool) & np.isfinite(merged[feature_cols].to_numpy(dtype=float)).all(axis=1)
    return merged


def standardize(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd < EPS, 1.0, sd)
    return (train_x - mu) / sd, (val_x - mu) / sd


def fit_logistic(x: np.ndarray, y: np.ndarray, *, steps: int = 1200, lr: float = 0.05, l2: float = 0.05) -> np.ndarray:
    n, p = x.shape
    xb = np.column_stack([np.ones(n), x])
    w = np.zeros(p + 1, dtype=float)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    weights = np.where(y > 0, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    for _ in range(steps):
        z = np.clip(xb @ w, -35, 35)
        pred = 1.0 / (1.0 + np.exp(-z))
        grad = xb.T @ ((pred - y) * weights) / n
        grad[1:] += l2 * w[1:] / n
        w -= lr * grad
    return w


def predict_logistic(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.column_stack([np.ones(len(x)), x])
    return 1.0 / (1.0 + np.exp(-np.clip(xb @ w, -35, 35)))


def auc_score(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    i = 0
    sorted_scores = score[order]
    while i < len(score):
        j = i + 1
        while j < len(score) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def balanced_accuracy(y: np.ndarray, score: np.ndarray) -> float | None:
    yhat = (score >= 0.5).astype(int)
    pos = y == 1
    neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    return float((((yhat[pos] == 1).mean()) + ((yhat[neg] == 0).mean())) / 2.0)


def bootstrap_lower_auc(y: np.ndarray, score: np.ndarray, *, seed: int = 20260905, samples: int = 2000) -> float | None:
    base = auc_score(y, score)
    if base is None:
        return None
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    idx = np.arange(len(y))
    for _ in range(samples):
        take = rng.choice(idx, size=len(idx), replace=True)
        a = auc_score(y[take], score[take])
        if a is not None:
            aucs.append(a)
    if len(aucs) < samples // 2:
        return None
    return float(np.quantile(aucs, 0.05))


def evaluate_candidate(frame: pd.DataFrame, candidate: str, feature_cols: list[str]) -> dict[str, Any]:
    rows = []
    scores = np.full(len(frame), np.nan)
    evaluated_mask = np.zeros(len(frame), dtype=bool)
    years = prereg.VALIDATION_YEARS
    for year in years:
        train_mask = (frame["decision_time"] < pd.Timestamp(f"{year}-01-01T00:00:00Z")) & frame["source_valid"]
        val_mask = (frame["decision_time"] >= pd.Timestamp(f"{year}-01-01T00:00:00Z")) & (frame["decision_time"] < pd.Timestamp(f"{year+1}-01-01T00:00:00Z")) & frame["source_valid"]
        y_train = frame.loc[train_mask, candidate].to_numpy(dtype=int)
        y_val = frame.loc[val_mask, candidate].to_numpy(dtype=int)
        fold = {"year": year, "train_signals": int(train_mask.sum()), "validation_signals": int(val_mask.sum()), "positives": int(y_val.sum()), "negatives": int(len(y_val) - y_val.sum()), "evaluated": False}
        if train_mask.sum() < prereg.ACCEPTANCE["minimum_training_signals_before_fold"] or val_mask.sum() < prereg.ACCEPTANCE["minimum_validation_signals_per_fold"] or len(set(y_train.tolist())) < 2 or len(set(y_val.tolist())) < 2:
            fold["reason"] = "insufficient_train_or_validation_class_support"
            rows.append(fold)
            continue
        x_train = frame.loc[train_mask, feature_cols].to_numpy(dtype=float)
        x_val = frame.loc[val_mask, feature_cols].to_numpy(dtype=float)
        x_train, x_val = standardize(x_train, x_val)
        w = fit_logistic(x_train, y_train)
        pred = predict_logistic(w, x_val)
        idx = np.flatnonzero(val_mask.to_numpy())
        scores[idx] = pred
        evaluated_mask[idx] = True
        fold.update({"evaluated": True, "auc": auc_score(y_val, pred), "balanced_accuracy": balanced_accuracy(y_val, pred)})
        rows.append(fold)
    pooled_y = frame.loc[evaluated_mask, candidate].to_numpy(dtype=int)
    pooled_score = scores[evaluated_mask]
    pooled_auc = auc_score(pooled_y, pooled_score) if len(pooled_y) else None
    lower = bootstrap_lower_auc(pooled_y, pooled_score, seed=20260905 + (0 if candidate == "SKIP" else 12)) if len(pooled_y) else None
    bacc = balanced_accuracy(pooled_y, pooled_score) if len(pooled_y) else None
    direction = all((not r.get("evaluated")) or (r.get("auc") is not None and float(r["auc"]) >= 0.50) for r in rows)
    checks = {
        "pooled_auc": pooled_auc is not None and pooled_auc >= prereg.ACCEPTANCE["pooled_auc_min_each_candidate"],
        "lower95_auc": lower is not None and lower > prereg.ACCEPTANCE["bootstrap_lower95_auc_min_each_candidate"],
        "balanced_accuracy": bacc is not None and bacc >= prereg.ACCEPTANCE["balanced_accuracy_min_each_candidate"],
        "consistent_year_direction": direction and any(r.get("evaluated") for r in rows),
    }
    return {
        "candidate": candidate,
        "folds": rows,
        "pooled": {"signals": int(len(pooled_y)), "positives": int(pooled_y.sum()) if len(pooled_y) else 0, "negatives": int(len(pooled_y) - pooled_y.sum()) if len(pooled_y) else 0, "auc": pooled_auc, "bootstrap_lower95_auc": lower, "balanced_accuracy": bacc},
        "checks": checks,
        "passed": all(checks.values()),
    }


def source_hash_frame(frame: pd.DataFrame, columns: list[str]) -> str:
    payload = frame[columns].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return sha256_bytes(payload)


def build(cfg: Config) -> dict[str, Any]:
    reg = load_preregistration(cfg.preregistration)
    labels = pre2024_label_frame(cfg.manifest)
    premium = query_premium(cfg.env_file)
    frame = build_feature_frame(labels, premium)
    feature_cols = [c for c in frame.columns if c.startswith("p")]
    evaluations = {candidate: evaluate_candidate(frame, candidate, feature_cols) for candidate in CANDIDATES}
    acceptance_checks = {f"{candidate}_{name}": bool(value) for candidate, ev in evaluations.items() for name, value in ev["checks"].items()}
    source_valid = frame["source_valid"].astype(bool)
    core = {
        "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(cfg.preregistration), "sha256": sha256_path(cfg.preregistration), "preregistration_hash": reg["preregistration_hash"]},
        "query_hashes": {"premium": sha256_bytes(prereg.PREMIUM_QUERY.encode("utf-8"))},
        "source_rows": {"premium_1m": int(len(premium)), "labels_pre2024_signals": int(len(labels)), "feature_rows": int(len(frame)), "source_valid_feature_rows": int(source_valid.sum())},
        "source_window": {"first": premium["date"].min().isoformat(), "last": premium["date"].max().isoformat(), "query_start": prereg.QUERY_START, "query_end_exclusive": prereg.QUERY_END_EXCLUSIVE},
        "source_hashes": {"premium_1m": source_hash_frame(premium, ["date", "open", "high", "low", "close"]), "labels_pre2024": source_hash_frame(labels, ["signal_position", "decision_time", "SKIP", "TP12", "tp4_utility", "skip_advantage", "tp12_advantage"]), "features": source_hash_frame(frame.loc[:, ["decision_time", "source_valid", *feature_cols]], ["decision_time", "source_valid", *feature_cols])},
        "label_distribution": {candidate: {"switch": int(frame[candidate].sum()), "keep": int(len(frame) - frame[candidate].sum())} for candidate in CANDIDATES},
        "feature_columns": feature_cols,
        "evaluations": evaluations,
        "acceptance_checks": acceptance_checks,
        "support_passed": all(acceptance_checks.values()),
        "decision": "pass_to_separate_oos_preregistration" if all(acceptance_checks.values()) else "terminal_pre2024_source_support_reject",
        "evidence_boundary": {"premium_intrahour_pre2024_opened": True, "pre2024_labels_opened": True, "oos_labels_opened": False, "fresh_outcomes_opened": False, "trained_adapter_scores_opened": False},
    }
    core["result_hash"] = sha256_bytes(canonical_json(core).encode("utf-8"))
    immutable_write_json(cfg.output, core)
    return core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=prereg.DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--manifest", type=Path, default=numeric.DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    result = build(Config(**vars(parse_args())))
    print(json.dumps({"support_passed": result["support_passed"], "decision": result["decision"], "evaluations": {k: v["pooled"] for k, v in result["evaluations"].items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
