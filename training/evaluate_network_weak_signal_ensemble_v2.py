"""One-shot strict pre-2024 evaluator for frozen NWE-8.

The module refuses to parse market prices, construct labels, fit forecasts, or
simulate execution until its exact source has been committed and frozen by
``freeze_network_weak_signal_ensemble_v2_evaluator``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import preregister_network_weak_signal_ensemble_v2 as prereg


SUPPORT_COMMIT = "76184e9fe6c4727f100c126aff8a9d4bd66da8db"
PREREG_SOURCE = Path("training/preregister_network_weak_signal_ensemble_v2.py")
PREREG_SOURCE_SHA256 = "fab8093f5edd2b9d5361d2337833a1956b220b0b8e336ac0b9e0cfbcb7fdcee8"
NWE7_PREREG_SOURCE = Path("training/preregister_network_weak_signal_ensemble.py")
NWE7_PREREG_SOURCE_SHA256 = "92f7b8a92647bb9f74b5859248d32a11a0a8cf70f7bf938e7cb709be4583e966"
SUPPORT_SOURCE = Path("training/build_network_weak_signal_ensemble_v2_support.py")
SUPPORT_SOURCE_SHA256 = "dbba76ddb2f93f6427f1a25270bb37773c8994e530ebca16ede6eda6cb2d3702"
SUPPORT_DOCUMENT = Path("docs/network-weak-signal-ensemble-v2-support-freeze-2026-07-17.md")
SUPPORT_DOCUMENT_SHA256 = "f18319f43f3f74f409a36edf366779b0a1f91a135869deda02a25923fd89048d"
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = "177fdb55aee96dd3849f45f71d7a845a09071aea357aaf21ac0de9b2538ed4c0"
SUPPORT_RESULT = Path("results/network_weak_signal_ensemble_v2_support_2026-07-17.json")
SUPPORT_RESULT_SHA256 = "f02377d7496751a2243384f73582cc189a4a7c2d5bc4184172424b959af39de7"
FEATURE_CLOCK = Path("results/network_weak_signal_ensemble_v2_feature_clock_2026-07-17.csv")
FEATURE_CLOCK_SHA256 = "3cc7eaa3b80944580651bf36541f0fde8edf4c66fd881d659f32396d1dda1c36"
MARKET_DATA = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_DATA_SHA256 = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
FUNDING_DATA = Path("results/binance_um_btcusdt_realized_funding_2020_2023.csv")
FUNDING_DATA_SHA256 = "c19829fa085a50f29c13762373a2b6db1c62025d657be1f5a3fbb9ce254482f7"
FUNDING_MANIFEST = Path("results/binance_um_btcusdt_realized_funding_2020_2023_manifest.json")
FUNDING_MANIFEST_SHA256 = "c70280e46bcbc2410cc59c2bcc93780c40997dbc5d0edb82d82127b59593250c"
EVALUATION_SOURCE = Path("training/evaluate_network_weak_signal_ensemble_v2.py")
EVALUATION_FREEZE = Path("results/network_weak_signal_ensemble_v2_evaluator_freeze_2026-07-17.json")
DEFAULT_OUTPUT = "results/network_weak_signal_ensemble_v2_selection_2026-07-17.json"
SELECTION_END = pd.Timestamp("2024-01-01")
WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2021-06-07", "2023-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
}
FEE_FEATURES = prereg.FEATURE_COLUMNS[:4]
TOPOLOGY_FEATURES = prereg.FEATURE_COLUMNS[4:]
POLICY_NAMES = (
    "primary",
    "direction_flip",
    "fee_family_only",
    "topology_family_only",
    "no_abstention",
    "stale_features_7d",
    "year_stratified_feature_permutation",
    "constant_long",
    "one_bar_delayed_entry",
)
MECHANISM_REJECTION_CONTROLS = (
    "stale_features_7d",
    "year_stratified_feature_permutation",
)


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_717
    feature_permutation_seed: int = 8_170_701
    minimum_mean_gross_underlying_bp: float = 20.0


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal_result(core: dict[str, Any]) -> dict[str, Any]:
    return {**core, "manifest_hash": _canonical_hash(core)}


def validate_result_hash(payload: dict[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if _canonical_hash(core) != payload.get("manifest_hash"):
        raise ValueError("NWE-8 result manifest hash mismatch")


def verify_support_and_clock() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    frozen_files = (
        (PREREG_SOURCE, PREREG_SOURCE_SHA256),
        (NWE7_PREREG_SOURCE, NWE7_PREREG_SOURCE_SHA256),
        (SUPPORT_SOURCE, SUPPORT_SOURCE_SHA256),
        (SUPPORT_DOCUMENT, SUPPORT_DOCUMENT_SHA256),
        (PREREGISTRATION, PREREGISTRATION_SHA256),
        (SUPPORT_RESULT, SUPPORT_RESULT_SHA256),
        (FEATURE_CLOCK, FEATURE_CLOCK_SHA256),
    )
    for path, expected in frozen_files:
        if _sha256(path) != expected:
            raise ValueError(f"frozen NWE-8 dependency changed: {path}")
    preregistration = json.loads(PREREGISTRATION.read_text())
    prereg.validate_manifest(preregistration)
    support = json.loads(SUPPORT_RESULT.read_text())
    if support.get("outcomes_opened") is not False:
        raise ValueError("NWE-8 support artifact opened outcomes")
    if support.get("policy") != asdict(prereg.Policy()):
        raise ValueError("NWE-8 support policy changed")
    if support.get("support_gate", {}).get("passed") is not True:
        raise ValueError("NWE-8 support gate did not pass")
    if support.get("source", {}).get("market_or_return_rows_loaded") != 0:
        raise ValueError("NWE-8 support loaded a market or return row")
    if support.get("feature_clock", {}).get("sha256") != FEATURE_CLOCK_SHA256:
        raise ValueError("NWE-8 support clock hash changed")

    clock = pd.read_csv(FEATURE_CLOCK)
    required = {
        "policy_id",
        "decision_date",
        "entry_date",
        "exit_date",
        "source_observation_date",
        "feature_available_at",
        "all_features_finite",
        "prediction_eligible",
        *prereg.FEATURE_COLUMNS,
    }
    missing = required.difference(clock.columns)
    if missing:
        raise ValueError(f"NWE-8 feature clock lacks columns: {sorted(missing)}")
    for column in (
        "decision_date",
        "entry_date",
        "exit_date",
        "source_observation_date",
        "feature_available_at",
    ):
        clock[column] = pd.to_datetime(clock[column], errors="raise")
    if not clock["policy_id"].eq(prereg.Policy().policy_id).all():
        raise ValueError("NWE-8 feature clock policy id changed")
    if not clock["decision_date"].is_monotonic_increasing or clock[
        "decision_date"
    ].duplicated().any():
        raise ValueError("NWE-8 feature clock decisions are invalid")
    eligible = clock["prediction_eligible"].astype(bool)
    if clock.loc[eligible, "exit_date"].max() >= SELECTION_END:
        raise ValueError("NWE-8 eligible feature clock crosses sealed 2024")
    values = clock.loc[eligible, list(prereg.FEATURE_COLUMNS)].to_numpy(float)
    if not np.isfinite(values).all() or not clock.loc[
        eligible, "all_features_finite"
    ].astype(bool).all():
        raise ValueError("NWE-8 prediction feature clock is not finite")
    if int(eligible.sum()) != support["feature_clock"]["prediction_eligible_rows"]:
        raise ValueError("NWE-8 prediction clock count changed")
    # The frozen support artifact deliberately retains a final ineligible
    # December row whose seven-day exit is exactly 2024-01-01.  Drop it before
    # any market mapping so no sealed price can become a label.
    evaluation_clock = clock.loc[clock["exit_date"] < SELECTION_END].reset_index(drop=True)
    return evaluation_clock, preregistration, support


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.is_file():
        raise ValueError("NWE-8 evaluator freeze is missing")
    payload = json.loads(EVALUATION_FREEZE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if _canonical_hash(core) != payload.get("manifest_hash"):
        raise ValueError("NWE-8 evaluator freeze manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("NWE-8 evaluator was not frozen before outcomes")
    if payload.get("evaluation_source") != str(EVALUATION_SOURCE):
        raise ValueError("NWE-8 evaluator freeze source path changed")
    if payload.get("evaluation_source_sha256") != _sha256(EVALUATION_SOURCE):
        raise ValueError("NWE-8 evaluator differs from its pre-outcome freeze")
    if payload.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("NWE-8 evaluator support commit changed")
    if payload.get("feature_clock_sha256") != FEATURE_CLOCK_SHA256:
        raise ValueError("NWE-8 evaluator feature clock changed")
    if payload.get("opened_windows") != []:
        raise ValueError("NWE-8 evaluator freeze already opened a window")
    if payload.get("sealed_windows") != [*WINDOWS, "2024", "2025", "2026_ytd"]:
        raise ValueError("NWE-8 evaluator sealed windows changed")
    if payload.get("mutable_parameters") != []:
        raise ValueError("NWE-8 evaluator freeze permits mutable parameters")
    if payload.get("labels_constructed_during_freeze") is not False:
        raise ValueError("NWE-8 evaluator freeze constructed labels")
    if payload.get("market_rows_parsed_during_freeze") != 0:
        raise ValueError("NWE-8 evaluator freeze parsed market rows")
    if payload.get("funding_rows_loaded_during_freeze") != 0:
        raise ValueError("NWE-8 evaluator freeze loaded funding rows")
    if payload.get("execution_simulation_run_during_freeze") is not False:
        raise ValueError("NWE-8 evaluator freeze simulated execution")
    if payload.get("evaluation_config") != asdict(EvaluationConfig()):
        raise ValueError("NWE-8 evaluator configuration changed")
    if payload.get("policy_names") != list(POLICY_NAMES):
        raise ValueError("NWE-8 evaluator control set changed")
    return payload


def _parse_pre2024_market(path: Path) -> pd.DataFrame:
    rows: list[tuple[str, float, float, float, float]] = []
    boundary_seen = False
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        expected = ["date", "open", "high", "low", "close"]
        positions = {column: header.index(column) for column in expected}
        if positions["date"] != 0:
            raise ValueError("NWE-8 market date must be the first physical column")
        for line in handle:
            date_text = line.split(",", 1)[0]
            if date_text >= "2024-01-01":
                boundary_seen = True
                break
            fields = line.rstrip("\r\n").split(",")
            rows.append(
                (
                    date_text,
                    float(fields[positions["open"]]),
                    float(fields[positions["high"]]),
                    float(fields[positions["low"]]),
                    float(fields[positions["close"]]),
                )
            )
    if not boundary_seen:
        raise ValueError("NWE-8 market source did not reach sealed 2024 boundary")
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])


def load_execution_market() -> tuple[pd.DataFrame, dict[str, Any]]:
    verify_evaluation_freeze()
    if _sha256(MARKET_DATA) != MARKET_DATA_SHA256:
        raise ValueError("NWE-8 market data differs from frozen hash")
    market = _parse_pre2024_market(MARKET_DATA)
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    if market.empty or market["date"].max() >= SELECTION_END:
        raise ValueError("NWE-8 market interval is invalid")
    if market["date"].duplicated().any() or not market["date"].is_monotonic_increasing:
        raise ValueError("NWE-8 market timestamps are invalid")
    prices = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("NWE-8 market contains invalid prices")
    opens, highs, lows, closes = (market[column].to_numpy(float) for column in ("open", "high", "low", "close"))
    if (
        (highs < np.maximum(opens, closes)).any()
        or (lows > np.minimum(opens, closes)).any()
        or (highs < lows).any()
    ):
        raise ValueError("NWE-8 market violates OHLC invariants")
    return market, {
        "market_sha256": MARKET_DATA_SHA256,
        "market_rows": int(len(market)),
        "columns_parsed": ["date", "open", "high", "low", "close"],
        "physical_parse_boundary": "stop before parsing the first date >= 2024-01-01",
        "first_date": str(market["date"].iloc[0]),
        "last_date": str(market["date"].iloc[-1]),
    }


def load_realized_funding() -> tuple[pd.DataFrame, dict[str, Any]]:
    verify_evaluation_freeze()
    if _sha256(FUNDING_MANIFEST) != FUNDING_MANIFEST_SHA256:
        raise ValueError("NWE-8 funding manifest differs from frozen hash")
    if _sha256(FUNDING_DATA) != FUNDING_DATA_SHA256:
        raise ValueError("NWE-8 funding data differs from frozen hash")
    manifest = json.loads(FUNDING_MANIFEST.read_text())
    if manifest.get("protocol", {}).get("luri_outcomes_opened") is not False:
        raise ValueError("NWE-8 funding source lacks unopened-source provenance")
    if manifest.get("protocol", {}).get("stage") != "pre_outcome_funding_source_freeze":
        raise ValueError("NWE-8 funding source stage changed")
    if manifest.get("data", {}).get("sha256") != FUNDING_DATA_SHA256:
        raise ValueError("NWE-8 funding manifest data hash differs")
    funding = pd.read_csv(
        FUNDING_DATA,
        usecols=["funding_time_ms", "funding_time_utc", "symbol", "funding_rate"],
        dtype={"symbol": str, "funding_rate": str},
    )
    if len(funding) != manifest["data"]["rows"]:
        raise ValueError("NWE-8 funding row count differs from manifest")
    funding["funding_time_ms"] = pd.to_numeric(
        funding["funding_time_ms"], errors="raise"
    ).astype(np.int64)
    utc = pd.to_datetime(funding["funding_time_utc"], utc=True, errors="raise").dt.tz_convert(None)
    epoch = pd.to_datetime(
        funding["funding_time_ms"], unit="ms", utc=True, errors="raise"
    ).dt.tz_convert(None)
    if not utc.equals(epoch):
        raise ValueError("NWE-8 funding timestamps disagree")
    if funding["funding_time_ms"].duplicated().any() or not funding[
        "funding_time_ms"
    ].is_monotonic_increasing:
        raise ValueError("NWE-8 funding timestamps are invalid")
    if not funding["symbol"].eq("BTCUSDT").all():
        raise ValueError("NWE-8 funding contains another symbol")
    rates = pd.to_numeric(funding["funding_rate"], errors="raise").to_numpy(float)
    if not np.isfinite(rates).all() or utc.max() >= SELECTION_END:
        raise ValueError("NWE-8 funding values or interval are invalid")
    normalized = pd.DataFrame(
        {
            "funding_time_ms": funding["funding_time_ms"].to_numpy(np.int64),
            "funding_time": utc,
            "funding_rate": rates,
        }
    )
    return normalized, {
        "funding_manifest_sha256": FUNDING_MANIFEST_SHA256,
        "funding_data_sha256": FUNDING_DATA_SHA256,
        "funding_rows": int(len(normalized)),
        "first_funding_time": str(normalized["funding_time"].iloc[0]),
        "last_funding_time": str(normalized["funding_time"].iloc[-1]),
    }


def attach_return_labels(clock: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    verify_evaluation_freeze()
    labelled = clock.copy()
    positions = pd.Series(np.arange(len(market), dtype=np.int64), index=market["date"])
    for label in ("decision", "entry", "exit"):
        mapped = labelled[f"{label}_date"].map(positions)
        if mapped.isna().any():
            missing = labelled.loc[mapped.isna(), f"{label}_date"].head().tolist()
            raise ValueError(f"NWE-8 {label} timestamps missing from market: {missing}")
        labelled[f"{label}_position"] = mapped.astype(np.int64)
    if not (
        (labelled["decision_position"] < labelled["entry_position"])
        & (labelled["entry_position"] < labelled["exit_position"])
    ).all():
        raise ValueError("NWE-8 market positions violate the frozen clock")
    opens = market["open"].to_numpy(float)
    entry = opens[labelled["entry_position"].to_numpy(np.int64)]
    exit_ = opens[labelled["exit_position"].to_numpy(np.int64)]
    labelled["target_log_return"] = np.log(exit_ / entry)
    if not np.isfinite(labelled["target_log_return"].to_numpy(float)).all():
        raise ValueError("NWE-8 return labels are invalid")
    return labelled


def _standardize_train_and_row(
    train: np.ndarray, row: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    valid = np.isfinite(std) & (std > 1e-12)
    train_z = np.zeros_like(train, dtype=float)
    row_z = np.zeros_like(row, dtype=float)
    train_z[:, valid] = (train[:, valid] - mean[valid]) / std[valid]
    row_z[valid] = (row[valid] - mean[valid]) / std[valid]
    return train_z, row_z


def _permute_training_rows_within_year(
    values: np.ndarray,
    years: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    result = values.copy()
    rng = np.random.default_rng(seed)
    for year in np.unique(years):
        positions = np.flatnonzero(years == year)
        if len(positions) > 1:
            result[positions] = values[rng.permutation(positions)]
    return result


def online_forecasts(
    labelled: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    cfg: EvaluationConfig,
    abstain: bool = True,
    stale_seven_days: bool = False,
    permute_within_year: bool = False,
) -> pd.DataFrame:
    verify_evaluation_freeze()
    policy = prereg.Policy()
    features = list(feature_columns)
    working = labelled.copy()
    if stale_seven_days:
        working[features] = working[features].shift(1)
    rows: list[dict[str, Any]] = []
    prediction_rows = working.loc[working["prediction_eligible"].astype(bool)]
    for current_index, current in prediction_rows.iterrows():
        decision = pd.Timestamp(current["decision_date"])
        pool = working.loc[
            (working["decision_date"] < decision)
            & (working["feature_available_at"] <= decision)
            & (working["exit_date"] <= decision)
            & working["all_features_finite"].astype(bool)
        ].copy()
        finite = np.isfinite(pool[features].to_numpy(float)).all(axis=1)
        pool = pool.loc[finite].tail(policy.maximum_train_samples)
        if len(pool) < policy.minimum_train_samples:
            raise ValueError(
                f"NWE-8 has only {len(pool)} causal training rows at {decision}"
            )
        x_train = pool[features].to_numpy(float)
        if permute_within_year:
            decision_seed = cfg.feature_permutation_seed + int(
                decision.value // 1_000_000_000 // 86_400
            )
            x_train = _permute_training_rows_within_year(
                x_train,
                pool["decision_date"].dt.year.to_numpy(np.int64),
                seed=decision_seed,
            )
        x_row = current[features].to_numpy(float)
        if not np.isfinite(x_row).all():
            raise ValueError("NWE-8 prediction row contains stale/non-finite features")
        x_train_z, x_row_z = _standardize_train_and_row(x_train, x_row)
        target = pool["target_log_return"].to_numpy(float)
        centered_target = target - target.mean()
        gram = x_train_z.T @ x_train_z
        beta = np.linalg.solve(
            gram + policy.ridge_alpha * np.eye(gram.shape[0]),
            x_train_z.T @ centered_target,
        )
        fitted = x_train_z @ beta
        forecast = float(x_row_z @ beta)
        threshold = float(np.quantile(np.abs(fitted), policy.abstain_quantile))
        raw_side = int(np.sign(forecast))
        side = raw_side if (not abstain or abs(forecast) >= threshold) else 0
        rows.append(
            {
                "decision_date": decision,
                "entry_date": pd.Timestamp(current["entry_date"]),
                "exit_date": pd.Timestamp(current["exit_date"]),
                "decision_position": int(current["decision_position"]),
                "entry_position": int(current["entry_position"]),
                "exit_position": int(current["exit_position"]),
                "forecast": forecast,
                "abstain_threshold": threshold,
                "side": side,
                "raw_side": raw_side,
                "train_count": int(len(pool)),
                "oldest_train_decision": pd.Timestamp(pool["decision_date"].iloc[0]),
                "latest_train_exit": pd.Timestamp(pool["exit_date"].iloc[-1]),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("NWE-8 produced no prediction rows")
    if (result["latest_train_exit"] > result["decision_date"]).any():
        raise ValueError("NWE-8 online model used an unavailable label")
    return result


def _trade_rows(predictions: pd.DataFrame, *, policy_name: str) -> pd.DataFrame:
    schedule = predictions.loc[predictions["side"].astype(int).ne(0)].copy()
    schedule.insert(0, "policy_name", policy_name)
    schedule["side"] = schedule["side"].astype(int)
    return schedule.reset_index(drop=True)


def build_schedules(
    labelled: pd.DataFrame, cfg: EvaluationConfig
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    primary_predictions = online_forecasts(
        labelled, feature_columns=prereg.FEATURE_COLUMNS, cfg=cfg
    )
    schedules: dict[str, pd.DataFrame] = {
        "primary": _trade_rows(primary_predictions, policy_name="primary")
    }

    flipped = primary_predictions.copy()
    flipped["side"] = -flipped["side"]
    flipped["raw_side"] = -flipped["raw_side"]
    schedules["direction_flip"] = _trade_rows(flipped, policy_name="direction_flip")

    fee = online_forecasts(labelled, feature_columns=FEE_FEATURES, cfg=cfg)
    schedules["fee_family_only"] = _trade_rows(fee, policy_name="fee_family_only")
    topology = online_forecasts(labelled, feature_columns=TOPOLOGY_FEATURES, cfg=cfg)
    schedules["topology_family_only"] = _trade_rows(
        topology, policy_name="topology_family_only"
    )

    no_abstention = primary_predictions.copy()
    no_abstention["side"] = no_abstention["raw_side"]
    schedules["no_abstention"] = _trade_rows(
        no_abstention, policy_name="no_abstention"
    )

    stale = online_forecasts(
        labelled,
        feature_columns=prereg.FEATURE_COLUMNS,
        cfg=cfg,
        stale_seven_days=True,
    )
    schedules["stale_features_7d"] = _trade_rows(
        stale, policy_name="stale_features_7d"
    )
    permuted = online_forecasts(
        labelled,
        feature_columns=prereg.FEATURE_COLUMNS,
        cfg=cfg,
        permute_within_year=True,
    )
    schedules["year_stratified_feature_permutation"] = _trade_rows(
        permuted, policy_name="year_stratified_feature_permutation"
    )

    constant = primary_predictions.copy()
    constant["forecast"] = 1.0
    constant["abstain_threshold"] = 0.0
    constant["side"] = 1
    constant["raw_side"] = 1
    schedules["constant_long"] = _trade_rows(constant, policy_name="constant_long")

    delayed = schedules["primary"].copy()
    delayed["policy_name"] = "one_bar_delayed_entry"
    delayed["entry_position"] = delayed["entry_position"] + 1
    delayed["entry_date"] = delayed["entry_date"] + pd.Timedelta(minutes=5)
    if (delayed["entry_position"] >= delayed["exit_position"]).any():
        raise ValueError("NWE-8 delayed control has no holding interval")
    schedules["one_bar_delayed_entry"] = delayed

    if set(schedules) != set(POLICY_NAMES):
        raise ValueError("NWE-8 control schedule set is incomplete")
    return schedules, primary_predictions


def _schedule_hash(schedule: pd.DataFrame) -> str:
    columns = [
        "policy_name",
        "decision_date",
        "entry_date",
        "exit_date",
        "decision_position",
        "entry_position",
        "exit_position",
        "forecast",
        "abstain_threshold",
        "side",
        "raw_side",
        "train_count",
        "oldest_train_decision",
        "latest_train_exit",
    ]
    content = schedule[columns].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def weekly_cluster_sign_flip(
    trade_returns: list[float],
    entry_dates: list[str],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    if permutations < 1 or len(trade_returns) != len(entry_dates):
        raise ValueError("NWE-8 cluster inputs are invalid")
    if not trade_returns:
        return {
            "p_value_one_sided": 1.0,
            "observed_mean_return": 0.0,
            "cluster_count": 0,
            "permutations": int(permutations),
            "seed": int(seed),
        }
    values = np.asarray(trade_returns, dtype=float)
    dates = pd.to_datetime(pd.Series(entry_dates), utc=True, errors="raise")
    monday = (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.floor("D")
    clusters = (
        pd.DataFrame({"week": monday, "return": values})
        .groupby("week", sort=True, observed=True)["return"]
        .sum()
        .to_numpy(float)
    )
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    completed = 0
    while completed < permutations:
        batch = min(4_096, permutations - completed)
        signs = rng.integers(0, 2, size=(batch, len(clusters)), dtype=np.int8)
        signs = signs.astype(float) * 2.0 - 1.0
        exceedances += int(
            np.count_nonzero(signs.dot(clusters) / len(values) >= observed)
        )
        completed += batch
    return {
        "p_value_one_sided": float((1 + exceedances) / (permutations + 1)),
        "observed_mean_return": observed,
        "cluster_count": int(len(clusters)),
        "permutations": int(permutations),
        "seed": int(seed),
    }


def _trade_statistics(values: list[float]) -> dict[str, Any]:
    count = len(values)
    if not count:
        return {
            "n_trades": 0,
            "mean_trade_ret_pct": 0.0,
            "std_trade_ret_pct": 0.0,
            "t_stat_like": 0.0,
            "ci95_mean_trade_ret_pct": [0.0, 0.0],
        }
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    se = std / math.sqrt(count)
    return {
        "n_trades": count,
        "mean_trade_ret_pct": mean * 100.0,
        "std_trade_ret_pct": std * 100.0,
        "t_stat_like": mean / se if se > 0.0 else 0.0,
        "ci95_mean_trade_ret_pct": [
            (mean - 1.96 * se) * 100.0,
            (mean + 1.96 * se) * 100.0,
        ],
    }


def _slice_schedule(schedule: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    inside = (
        schedule["decision_date"].ge(start_timestamp)
        & schedule["entry_date"].ge(start_timestamp)
        & schedule["exit_date"].ge(start_timestamp)
        & schedule["decision_date"].lt(end_timestamp)
        & schedule["entry_date"].lt(end_timestamp)
        & schedule["exit_date"].lt(end_timestamp)
    )
    return schedule.loc[inside].reset_index(drop=True)


def simulate_schedule(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
    cost_notional_per_side: float,
    cfg: EvaluationConfig,
    compute_cluster: bool,
) -> dict[str, Any]:
    verify_evaluation_freeze()
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp or cfg.leverage <= 0.0:
        raise ValueError("NWE-8 simulation parameters are invalid")
    per_side_cost = cost_notional_per_side * cfg.leverage
    if not 0.0 <= per_side_cost < 1.0:
        raise ValueError("NWE-8 per-side cost is invalid")
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    dates = market["date"]
    funding_times = funding["funding_time_ms"].to_numpy(np.int64)
    funding_rates = funding["funding_rate"].to_numpy(float)
    equity = 1.0
    peak = 1.0
    strict_mdd = 0.0
    previous_exit = -1
    trade_returns: list[float] = []
    gross_returns: list[float] = []
    entry_dates: list[str] = []
    sides: list[int] = []
    settlement_count = 0
    trades_with_funding = 0

    for row in schedule.itertuples(index=False):
        decision_position = int(row.decision_position)
        entry_position = int(row.entry_position)
        exit_position = int(row.exit_position)
        side = int(row.side)
        if side not in (-1, 1):
            raise ValueError("NWE-8 side must be long or short")
        if not 0 <= decision_position < entry_position < exit_position:
            raise ValueError("NWE-8 scheduled positions are invalid")
        if entry_position < previous_exit:
            raise ValueError("NWE-8 schedules overlap")
        if exit_position >= len(market):
            raise ValueError("NWE-8 exit exceeds market frame")
        for label, position in (
            ("decision", decision_position),
            ("entry", entry_position),
            ("exit", exit_position),
        ):
            if pd.Timestamp(getattr(row, f"{label}_date")) != dates.iloc[position]:
                raise ValueError(f"NWE-8 {label} timestamp differs from market")
            timestamp = dates.iloc[position]
            if not start_timestamp <= timestamp < end_timestamp:
                raise ValueError("NWE-8 trade crosses simulation split")

        entry_price = float(opens[entry_position])
        exit_price = float(opens[exit_position])
        held_high = float(np.max(highs[entry_position:exit_position]))
        held_low = float(np.min(lows[entry_position:exit_position]))
        if min(entry_price, exit_price, held_high, held_low) <= 0.0:
            raise ValueError("NWE-8 scheduled trade has invalid price")

        entry_ms = int(pd.Timestamp(dates.iloc[entry_position]).value // 1_000_000)
        exit_ms = int(pd.Timestamp(dates.iloc[exit_position]).value // 1_000_000)
        left = int(np.searchsorted(funding_times, entry_ms, side="left"))
        right = int(np.searchsorted(funding_times, exit_ms, side="left"))
        rates = funding_rates[left:right]
        funding_contributions = -cfg.leverage * side * rates
        funding_return = float(np.sum(funding_contributions, dtype=float))
        funding_credit = float(
            np.sum(np.maximum(funding_contributions, 0.0), dtype=float)
        )
        funding_debit = float(
            np.sum(np.minimum(funding_contributions, 0.0), dtype=float)
        )

        entry_equity = equity
        peak = max(peak, equity)
        equity = entry_equity * (1.0 - per_side_cost)
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)

        favorable_price = held_high if side > 0 else held_low
        adverse_price = held_low if side > 0 else held_high
        favorable_equity = entry_equity * max(
            0.0,
            1.0
            - per_side_cost
            + cfg.leverage * side * (favorable_price / entry_price - 1.0)
            + funding_credit,
        )
        intratrade_peak = max(peak, favorable_equity)
        adverse_liquidation_factor = max(
            0.0,
            1.0
            - per_side_cost
            + cfg.leverage * side * (adverse_price / entry_price - 1.0)
            + funding_debit
            - per_side_cost * (adverse_price / entry_price),
        )
        adverse_equity = entry_equity * adverse_liquidation_factor
        strict_mdd = max(
            strict_mdd, 1.0 - max(0.0, adverse_equity) / intratrade_peak
        )
        peak = intratrade_peak

        gross_return = side * (exit_price / entry_price - 1.0)
        exit_factor = max(
            0.0,
            1.0
            - per_side_cost
            + cfg.leverage * gross_return
            + funding_return
            - per_side_cost * (exit_price / entry_price),
        )
        equity = entry_equity * exit_factor
        strict_mdd = max(strict_mdd, 1.0 - max(0.0, equity) / peak)
        peak = max(peak, equity)

        trade_returns.append(equity / entry_equity - 1.0)
        gross_returns.append(gross_return)
        entry_dates.append(str(dates.iloc[entry_position]))
        sides.append(side)
        settlement_count += int(len(rates))
        trades_with_funding += int(len(rates) > 0)
        previous_exit = exit_position

    years = (end_timestamp - start_timestamp).total_seconds() / (365.25 * 86_400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    strict_mdd_pct = strict_mdd * 100.0
    if strict_mdd_pct > 1e-12:
        cagr_to_strict_mdd = float(cagr / strict_mdd_pct)
        zero_mdd_ratio_cap_applied = False
    elif cagr > 0.0:
        # JSON forbids infinity. A finite cap preserves the mathematical pass
        # semantics of positive return with zero drawdown without NaN/Inf.
        cagr_to_strict_mdd = 1.0e12
        zero_mdd_ratio_cap_applied = True
    else:
        cagr_to_strict_mdd = 0.0
        zero_mdd_ratio_cap_applied = False
    cluster = (
        weekly_cluster_sign_flip(
            trade_returns,
            entry_dates,
            permutations=cfg.cluster_permutations,
            seed=cfg.cluster_seed,
        )
        if compute_cluster
        else None
    )
    count = len(sides)
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_mdd_pct),
        "cagr_to_strict_mdd": cagr_to_strict_mdd,
        "zero_mdd_ratio_cap_applied": zero_mdd_ratio_cap_applied,
        "trade_count": int(count),
        "long_count": int(sum(side > 0 for side in sides)),
        "short_count": int(sum(side < 0 for side in sides)),
        "long_share": float(sum(side > 0 for side in sides) / count) if count else 0.0,
        "wall_clock_years": float(years),
        "mean_gross_underlying_move_bp": (
            float(np.mean(gross_returns) * 10_000.0) if gross_returns else 0.0
        ),
        "funding_settlement_count": int(settlement_count),
        "trades_with_funding": int(trades_with_funding),
        "execution_cost_notional_per_side_bp": float(cost_notional_per_side * 10_000.0),
        "trade_statistics": _trade_statistics(trade_returns),
        "weekly_cluster_sign_flip": cluster,
    }


def _evaluate_policy_windows(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (start, end) in WINDOWS.items():
        sliced = _slice_schedule(schedule, start=start, end=end)
        result[name] = {
            "base": simulate_schedule(
                market,
                funding,
                sliced,
                start=start,
                end=end,
                cost_notional_per_side=cfg.base_cost_notional_per_side,
                cfg=cfg,
                compute_cluster=name in {"train", "select2023"},
            ),
            "stress_10bp": simulate_schedule(
                market,
                funding,
                sliced,
                start=start,
                end=end,
                cost_notional_per_side=cfg.stress_cost_notional_per_side,
                cfg=cfg,
                compute_cluster=False,
            ),
        }
    return result


def _primary_gate_failures(windows: dict[str, Any], cfg: EvaluationConfig) -> list[str]:
    gates = prereg.build_manifest()["selection_protocol"]["performance_gates"]
    failures: list[str] = []
    for name in ("train", "select2023"):
        base = windows[name]["base"]
        stress = windows[name]["stress_10bp"]
        minimum_count = (
            gates["train_trade_count_min"]
            if name == "train"
            else gates["selection_trade_count_min"]
        )
        if base["trade_count"] < minimum_count:
            failures.append(f"{name}: trade count below {minimum_count}")
        low, high = gates["each_side_share_range"]
        if not low <= base["long_share"] <= high:
            failures.append(f"{name}: side share outside [{low}, {high}]")
        if base["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if base["cagr_to_strict_mdd"] < gates["train_and_2023_cagr_to_strict_mdd_min"]:
            failures.append(f"{name}: CAGR/strict-MDD below 3")
        if base["strict_mdd_pct"] > gates["train_and_2023_strict_mdd_pct_max"]:
            failures.append(f"{name}: strict MDD above 15%")
        cluster = base["weekly_cluster_sign_flip"]
        if cluster is None or cluster["p_value_one_sided"] > gates[
            "train_and_2023_weekly_cluster_signflip_p_max"
        ]:
            failures.append(f"{name}: weekly-cluster p-value above 0.10")
        if base["mean_gross_underlying_move_bp"] < cfg.minimum_mean_gross_underlying_bp:
            failures.append(f"{name}: mean gross edge below 20 bp")
        if stress["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: 10bp stress non-positive")
    for name in ("select2023_h1", "select2023_h2"):
        base = windows[name]["base"]
        if base["trade_count"] < gates["each_selection_half_trade_count_min"]:
            failures.append(f"{name}: trade count below 8")
        if base["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
    return failures


def qualification(policy_windows: dict[str, dict[str, Any]], cfg: EvaluationConfig) -> dict[str, Any]:
    primary_failures = _primary_gate_failures(policy_windows["primary"], cfg)
    delayed_failures = [
        f"{name}: one-bar-delayed entry non-positive absolute return"
        for name in ("train", "select2023")
        if policy_windows["one_bar_delayed_entry"][name]["base"][
            "absolute_return_pct"
        ]
        <= 0.0
    ]
    mechanism_failures = {
        name: _primary_gate_failures(policy_windows[name], cfg)
        for name in MECHANISM_REJECTION_CONTROLS
    }
    passing_nulls = [name for name, failures in mechanism_failures.items() if not failures]
    failures = [*primary_failures, *delayed_failures]
    failures.extend(
        f"mechanism-null control independently passed every gate: {name}"
        for name in passing_nulls
    )
    return {
        "qualifies": not failures,
        "scope": "pre-orthogonality performance and mechanism gates",
        "final_promotion_allowed": False,
        "failures": failures,
        "primary_performance_gate_failures": primary_failures,
        "delayed_entry_gate_failures": delayed_failures,
        "mechanism_control_gate_failures": mechanism_failures,
        "passing_mechanism_controls": passing_nulls,
        "direction_flip_component_and_constant_long_are_diagnostic_only": True,
    }


def _selection_decision(verdict: dict[str, Any]) -> dict[str, Any]:
    if verdict["qualifies"]:
        return {
            "selected_alpha": None,
            "performance_candidate": "NWE-8",
            "rejected": False,
            "status": "pending_preregistered_orthogonality_and_portfolio_gates",
            "orthogonality_evaluated": False,
            "promotion_ready": False,
        }
    return {
        "selected_alpha": None,
        "performance_candidate": None,
        "rejected": True,
        "status": "rejected_before_orthogonality",
        "orthogonality_evaluated": False,
        "promotion_ready": False,
    }


def _canonical_primary_schedule(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "decision_date",
        "entry_date",
        "exit_date",
        "side",
        "forecast",
        "abstain_threshold",
        "train_count",
        "latest_train_exit",
    ]
    rows: list[dict[str, Any]] = []
    for row in schedule[columns].itertuples(index=False):
        rows.append(
            {
                "decision_date": str(row.decision_date),
                "entry_date": str(row.entry_date),
                "exit_date": str(row.exit_date),
                "side": int(row.side),
                "forecast": float(row.forecast),
                "abstain_threshold": float(row.abstain_threshold),
                "train_count": int(row.train_count),
                "latest_train_exit": str(row.latest_train_exit),
            }
        )
    return rows


def run_evaluation(cfg: EvaluationConfig | None = None) -> dict[str, Any]:
    frozen_cfg = EvaluationConfig() if cfg is None else cfg
    if frozen_cfg != EvaluationConfig():
        raise ValueError("NWE-8 evaluation parameters are frozen")
    evaluator_freeze = verify_evaluation_freeze()
    clock, preregistration, support = verify_support_and_clock()
    market, market_source = load_execution_market()
    funding, funding_source = load_realized_funding()
    labelled = attach_return_labels(clock, market)
    schedules, prediction_rows = build_schedules(labelled, frozen_cfg)
    windows = {
        name: _evaluate_policy_windows(market, funding, schedule, frozen_cfg)
        for name, schedule in schedules.items()
    }
    verdict = qualification(windows, frozen_cfg)
    core = {
        "protocol": {
            "name": "NWE-8 frozen pre-2024 selection evaluation",
            "support_commit": SUPPORT_COMMIT,
            "evaluation_source_commit": evaluator_freeze["evaluation_source_commit"],
            "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
            "evaluation_freeze_sha256": _sha256(EVALUATION_FREEZE),
            "outcomes_opened": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["2024", "2025", "2026_ytd"],
            "parameters_mutable": False,
            "target": "weekly unlevered entry-open to exit-open log return",
            "label_availability": "feature_available_at and label exit <= refit decision",
            "target_centering": "training mean removed and never added back",
            "stale_control": "previous weekly snapshot, exactly seven calendar days stale",
            "permutation_control": "training features permuted causally within year per refit",
            "funding_interval": "entry_time <= funding_time < exit_time",
            "strict_mdd": (
                "global/pre-entry HWM; favorable then adverse held OHLC; funding "
                "credits raise HWM while debits lower the adverse envelope; "
                "hypothetical adverse liquidation cost"
            ),
            "cagr": "full wall-clock split including abstained cash weeks",
        },
        "policy": preregistration["policy"],
        "evaluation_config": asdict(frozen_cfg),
        "source": {
            "support_result_sha256": SUPPORT_RESULT_SHA256,
            "feature_clock_sha256": FEATURE_CLOCK_SHA256,
            "market": market_source,
            "funding": funding_source,
            "support": support["source"],
        },
        "model": {
            "prediction_row_count": int(len(prediction_rows)),
            "first_prediction": str(prediction_rows["decision_date"].min()),
            "last_prediction": str(prediction_rows["decision_date"].max()),
            "minimum_causal_train_count": int(prediction_rows["train_count"].min()),
            "maximum_causal_train_count": int(prediction_rows["train_count"].max()),
            "schedule_counts": {name: int(len(value)) for name, value in schedules.items()},
            "schedule_hashes": {name: _schedule_hash(value) for name, value in schedules.items()},
        },
        "primary_schedule": _canonical_primary_schedule(schedules["primary"]),
        "windows": windows,
        "qualification": verdict,
        "selection": _selection_decision(verdict),
    }
    return _seal_result(core)


def _headline(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "absolute_return_pct",
            "cagr_pct",
            "strict_mdd_pct",
            "cagr_to_strict_mdd",
            "mean_gross_underlying_move_bp",
            "trade_count",
            "long_count",
            "short_count",
        )
    }


def main() -> None:
    output = Path(DEFAULT_OUTPUT)
    if output.exists():
        raise RuntimeError("NWE-8 outcome result already exists")
    result = run_evaluation()
    validate_result_hash(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "qualification": result["qualification"],
                "primary": {
                    name: {
                        cost: _headline(metrics)
                        for cost, metrics in result["windows"]["primary"][name].items()
                    }
                    for name in WINDOWS
                },
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
