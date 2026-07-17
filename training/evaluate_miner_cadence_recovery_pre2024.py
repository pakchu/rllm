"""One-shot strict pre-2024 evaluator for frozen MCR-7.

The exact outcome-blind event clock, control algorithms, execution accounting,
costs, funding treatment, gates, and random seeds are frozen before this module
is allowed to parse a post-entry market row.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training import build_miner_cadence_recovery_support as support_builder
from training import preregister_miner_cadence_recovery as prereg


SUPPORT_COMMIT = "3bf6a7bca31c6a13045a17884f6ddc6f6d96228d"
PREREG_SOURCE = Path("training/preregister_miner_cadence_recovery.py")
PREREG_SOURCE_SHA256 = "695cba08bb2f479b4f416bc8b131cf902ef6174795f89d3d1b2d3eb5d8025a92"
SUPPORT_SOURCE = Path("training/build_miner_cadence_recovery_support.py")
SUPPORT_SOURCE_SHA256 = "b5818d7761f08150f0214547ef6b8387df2c868c180487eb5735bbfeac19aab3"
SUPPORT_DOCUMENT = Path("docs/miner-cadence-recovery-support-2026-07-17.md")
SUPPORT_DOCUMENT_SHA256 = "5fd9585104c60bcff9a8ba824f50d9b2ab1c0e69591a2647a353b3f05956673b"
PREREGISTRATION = Path(prereg.DEFAULT_OUTPUT)
PREREGISTRATION_SHA256 = "243b3a26316addefe1429fda0506366fad87962642ebdbeb24e0b37dd16afec4"
SUPPORT_RESULT = Path("results/miner_cadence_recovery_support_2026-07-17.json")
SUPPORT_RESULT_SHA256 = "817081407607a3f495c93c31c31d2ce18f8c5652f3b8a83a0811bc082ff62df5"
PRIMARY_CLOCK = Path("results/miner_cadence_recovery_clock_2026-07-17.csv")
PRIMARY_CLOCK_SHA256 = "2535244889b046ff00c369ee854973a91c23429dff82a6dd3c1a293a01352b0b"
MARKET_DATA = Path("data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz")
MARKET_DATA_SHA256 = "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
FUNDING_DATA = Path("data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz")
FUNDING_DATA_SHA256 = "3284bbb6bb67946acb673c6b67459543e217f752589e1d47b6c7c3b659f733e6"
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)
FUNDING_MANIFEST_SHA256 = "a0b2d27e1aa8cf2d9ab8cb659b598ee0a6d7bd25401c9e10ae92d1a74415845b"
EVALUATION_SOURCE = Path("training/evaluate_miner_cadence_recovery_pre2024.py")
EVALUATION_FREEZE = Path("results/miner_cadence_recovery_evaluator_freeze_2026-07-17.json")
DEFAULT_OUTPUT = "results/miner_cadence_recovery_pre2024_evaluation_2026-07-17.json"
SELECTION_END = pd.Timestamp("2024-01-01")
WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2021-03-01", "2023-01-01"),
    "train2021": ("2021-03-01", "2022-01-01"),
    "train2022": ("2022-01-01", "2023-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
}
POLICY_NAMES = (
    "primary",
    "direction_flip",
    "cadence_confirmation_removed",
    "stale_hash_state_7d",
    "random_clock",
    "constant_long",
    "one_bar_delayed_entry",
)
MECHANISM_REJECTION_CONTROLS = (
    "cadence_confirmation_removed",
    "stale_hash_state_7d",
)


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 0.5
    base_cost_notional_per_side: float = 0.0006
    stress_cost_notional_per_side: float = 0.0010
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_717
    random_clock_seed: int = 7_310_077
    stale_observations: int = 7
    additional_delay_bars: int = 1
    minimum_mean_gross_underlying_bp: float = 40.0


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
        raise ValueError("MCR-7 result manifest hash mismatch")


def _parse_clock(path: Path) -> pd.DataFrame:
    clock = pd.read_csv(
        path,
        parse_dates=[
            "observation_date",
            "available_at",
            "earliest_tradable_open",
            "entry_date",
            "exit_date",
        ],
    )
    required = {
        "policy_id",
        "side",
        "observation_date",
        "available_at",
        "earliest_tradable_open",
        "entry_date",
        "exit_date",
    }
    missing = required.difference(clock.columns)
    if missing:
        raise ValueError(f"MCR-7 primary clock lacks columns: {sorted(missing)}")
    return clock


def _normalize_control_clock(
    clock: pd.DataFrame,
    *,
    name: str,
    side: int | None = None,
) -> pd.DataFrame:
    normalized = clock.copy()
    if side is not None:
        normalized["side"] = side
    normalized.insert(0, "control", name)
    normalized = normalized.loc[normalized["exit_date"] < SELECTION_END].copy()
    normalized = normalized.sort_values("entry_date").reset_index(drop=True)
    return normalized


def _clock_hash(clock: pd.DataFrame) -> str:
    rows = [
        {
            "control": str(row.control),
            "side": int(row.side),
            "available_at": str(row.available_at),
            "entry_date": str(row.entry_date),
            "exit_date": str(row.exit_date),
        }
        for row in clock[
            ["control", "side", "available_at", "entry_date", "exit_date"]
        ].itertuples(index=False)
    ]
    return _canonical_hash(rows)


def _validate_clock(clock: pd.DataFrame, *, name: str, policy: prereg.Policy) -> None:
    if clock.empty:
        raise ValueError(f"MCR-7 {name} clock is empty")
    if not clock["entry_date"].is_monotonic_increasing:
        raise ValueError(f"MCR-7 {name} clock is not sorted")
    if clock["entry_date"].duplicated().any():
        raise ValueError(f"MCR-7 {name} clock has duplicate entries")
    if not clock["side"].isin([-1, 1]).all():
        raise ValueError(f"MCR-7 {name} clock has an invalid side")
    expected_hold = pd.to_timedelta(policy.hold_bars * 5, unit="min")
    if not (clock["exit_date"] - clock["entry_date"]).eq(expected_hold).all():
        raise ValueError(f"MCR-7 {name} hold changed")
    if clock["entry_date"].min() < support_builder.TRAIN_START:
        raise ValueError(f"MCR-7 {name} starts before the frozen interval")
    if clock["exit_date"].max() >= SELECTION_END:
        raise ValueError(f"MCR-7 {name} crosses sealed 2024")
    if len(clock) > 1:
        current = clock["entry_date"].iloc[1:].reset_index(drop=True)
        previous_exit = clock["exit_date"].iloc[:-1].reset_index(drop=True)
        if not current.ge(previous_exit).all():
            raise ValueError(f"MCR-7 {name} clock overlaps")


def _schedule_feature_control(
    features: pd.DataFrame,
    event: Iterable[bool],
    policy: prereg.Policy,
    *,
    name: str,
    side: int = 1,
) -> pd.DataFrame:
    candidate = features.copy()
    candidate["event"] = np.asarray(event, dtype=bool)
    clock = support_builder.schedule_clock(candidate, policy)
    return _normalize_control_clock(clock, name=name, side=side)


def build_control_clocks(
    primary: pd.DataFrame,
    features: pd.DataFrame,
    *,
    cfg: EvaluationConfig,
    policy: prereg.Policy,
) -> dict[str, pd.DataFrame]:
    controls: dict[str, pd.DataFrame] = {
        "primary": _normalize_control_clock(primary, name="primary", side=1),
        "direction_flip": _normalize_control_clock(
            primary, name="direction_flip", side=-1
        ),
    }

    valid_lag = features["source_lag_days"].le(policy.maximum_source_lag_days)
    cadence_removed_event = (
        features["recovery_cross"].astype(bool)
        & features["recent_stress"].astype(bool)
        & valid_lag
    )
    controls["cadence_confirmation_removed"] = _schedule_feature_control(
        features,
        cadence_removed_event,
        policy,
        name="cadence_confirmation_removed",
    )

    stale_z = features["hash_change_z"].shift(cfg.stale_observations)
    stale_prior_z = features["prior_hash_change_z"].shift(cfg.stale_observations)
    stale_recent_stress = features["recent_stress"].shift(
        cfg.stale_observations, fill_value=False
    )
    stale_source_lag = features["source_lag_days"].shift(cfg.stale_observations)
    stale_event = (
        stale_z.ge(policy.recovery_z_min)
        & stale_prior_z.lt(policy.recovery_z_min)
        & stale_recent_stress.astype(bool)
        & stale_source_lag.le(policy.maximum_source_lag_days)
        & features["cadence_gap"].ge(0.0)
    )
    controls["stale_hash_state_7d"] = _schedule_feature_control(
        features,
        stale_event,
        policy,
        name="stale_hash_state_7d",
    )

    all_source_days = features["source_lag_days"].le(
        policy.maximum_source_lag_days
    )
    nonoverlap_universe = _schedule_feature_control(
        features,
        all_source_days,
        policy,
        name="constant_long",
    )
    controls["constant_long"] = nonoverlap_universe

    rng = np.random.default_rng(cfg.random_clock_seed)
    random_rows: list[pd.DataFrame] = []
    primary_year_counts = controls["primary"]["entry_date"].dt.year.value_counts()
    for year, count in sorted(primary_year_counts.items()):
        pool = nonoverlap_universe.loc[
            nonoverlap_universe["entry_date"].dt.year.eq(year)
        ]
        if len(pool) < count:
            raise ValueError(f"MCR-7 random clock lacks {year} support")
        selected = np.sort(rng.choice(pool.index.to_numpy(), size=int(count), replace=False))
        random_rows.append(pool.loc[selected])
    random_clock = pd.concat(random_rows, ignore_index=True).sort_values("entry_date")
    random_clock = random_clock.reset_index(drop=True)
    random_clock["control"] = "random_clock"
    controls["random_clock"] = random_clock

    delayed = controls["primary"].copy()
    delay = pd.to_timedelta(cfg.additional_delay_bars * 5, unit="min")
    delayed["entry_date"] += delay
    delayed["exit_date"] += delay
    delayed["control"] = "one_bar_delayed_entry"
    controls["one_bar_delayed_entry"] = delayed

    for name in POLICY_NAMES:
        _validate_clock(controls[name], name=name, policy=policy)
    return controls


def verify_support_and_control_clocks(
    cfg: EvaluationConfig | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any]]:
    frozen_cfg = EvaluationConfig() if cfg is None else cfg
    if frozen_cfg != EvaluationConfig():
        raise ValueError("MCR-7 evaluation parameters are frozen")
    frozen_files = (
        (PREREG_SOURCE, PREREG_SOURCE_SHA256),
        (SUPPORT_SOURCE, SUPPORT_SOURCE_SHA256),
        (SUPPORT_DOCUMENT, SUPPORT_DOCUMENT_SHA256),
        (PREREGISTRATION, PREREGISTRATION_SHA256),
        (SUPPORT_RESULT, SUPPORT_RESULT_SHA256),
        (PRIMARY_CLOCK, PRIMARY_CLOCK_SHA256),
    )
    for path, expected in frozen_files:
        if _sha256(path) != expected:
            raise ValueError(f"frozen MCR-7 dependency changed: {path}")
    preregistration = json.loads(PREREGISTRATION.read_text())
    prereg.validate_manifest(preregistration)
    support = json.loads(SUPPORT_RESULT.read_text())
    if support.get("outcomes_opened") is not False:
        raise ValueError("MCR-7 support artifact opened outcomes")
    if support.get("policy") != asdict(prereg.Policy()):
        raise ValueError("MCR-7 support policy changed")
    if support.get("support_gate", {}).get("passed") is not True:
        raise ValueError("MCR-7 support gate did not pass")
    if support.get("source", {}).get("market_or_funding_rows_loaded") != 0:
        raise ValueError("MCR-7 support loaded market or funding rows")
    if support.get("clock", {}).get("sha256") != PRIMARY_CLOCK_SHA256:
        raise ValueError("MCR-7 support clock hash changed")

    primary = _parse_clock(PRIMARY_CLOCK)
    policy = prereg.Policy()
    if not primary["policy_id"].eq(policy.policy_id).all():
        raise ValueError("MCR-7 primary policy id changed")
    if not primary["side"].eq(1).all():
        raise ValueError("MCR-7 primary side changed")
    if not primary["earliest_tradable_open"].ge(primary["available_at"]).all():
        raise ValueError("MCR-7 primary clock predates source availability")
    if not primary["entry_date"].eq(
        primary["earliest_tradable_open"] + pd.Timedelta(minutes=5)
    ).all():
        raise ValueError("MCR-7 primary latency changed")
    source = support_builder.load_miner_security(preregistration)
    features = support_builder.build_features(source, policy)
    controls = build_control_clocks(primary, features, cfg=frozen_cfg, policy=policy)
    if len(controls["primary"]) != support["clock"]["rows"]:
        raise ValueError("MCR-7 primary clock count changed")
    return controls, preregistration, support


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.is_file():
        raise ValueError("MCR-7 evaluator freeze is missing")
    payload = json.loads(EVALUATION_FREEZE.read_text())
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if _canonical_hash(core) != payload.get("manifest_hash"):
        raise ValueError("MCR-7 evaluator freeze manifest hash mismatch")
    if payload.get("outcomes_opened") is not False:
        raise ValueError("MCR-7 evaluator was not frozen before outcomes")
    if payload.get("evaluation_source") != str(EVALUATION_SOURCE):
        raise ValueError("MCR-7 evaluator freeze source path changed")
    if payload.get("evaluation_source_sha256") != _sha256(EVALUATION_SOURCE):
        raise ValueError("MCR-7 evaluator differs from its pre-outcome freeze")
    if payload.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("MCR-7 evaluator support commit changed")
    if payload.get("primary_clock_sha256") != PRIMARY_CLOCK_SHA256:
        raise ValueError("MCR-7 evaluator primary clock changed")
    if payload.get("opened_windows") != []:
        raise ValueError("MCR-7 evaluator freeze already opened a window")
    if payload.get("sealed_windows") != [*WINDOWS, "2024", "2025", "2026_ytd"]:
        raise ValueError("MCR-7 evaluator sealed windows changed")
    if payload.get("mutable_parameters") != []:
        raise ValueError("MCR-7 evaluator freeze permits mutable parameters")
    if payload.get("market_rows_parsed_during_freeze") != 0:
        raise ValueError("MCR-7 evaluator freeze parsed market rows")
    if payload.get("funding_rows_loaded_during_freeze") != 0:
        raise ValueError("MCR-7 evaluator freeze loaded funding rows")
    if payload.get("execution_simulation_run_during_freeze") is not False:
        raise ValueError("MCR-7 evaluator freeze simulated execution")
    if payload.get("evaluation_config") != asdict(EvaluationConfig()):
        raise ValueError("MCR-7 evaluator configuration changed")
    if payload.get("policy_names") != list(POLICY_NAMES):
        raise ValueError("MCR-7 evaluator control set changed")
    return payload


def _parse_pre2024_market(path: Path) -> pd.DataFrame:
    rows: list[tuple[str, float, float, float, float]] = []
    boundary_seen = False
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split(",")
        expected = ["date", "open", "high", "low", "close"]
        positions = {column: header.index(column) for column in expected}
        if positions["date"] != 0:
            raise ValueError("MCR-7 market date must be the first physical column")
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
        raise ValueError("MCR-7 market source did not reach sealed 2024 boundary")
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])


def load_execution_market() -> tuple[pd.DataFrame, dict[str, Any]]:
    verify_evaluation_freeze()
    if _sha256(MARKET_DATA) != MARKET_DATA_SHA256:
        raise ValueError("MCR-7 market data differs from its frozen hash")
    market = _parse_pre2024_market(MARKET_DATA)
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    if market.empty or market["date"].max() >= SELECTION_END:
        raise ValueError("MCR-7 market interval is invalid")
    if market["date"].duplicated().any() or not market["date"].is_monotonic_increasing:
        raise ValueError("MCR-7 market timestamps are invalid")
    prices = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("MCR-7 market contains invalid prices")
    opens, highs, lows, closes = (
        market[column].to_numpy(float) for column in ("open", "high", "low", "close")
    )
    if (
        (highs < np.maximum(opens, closes)).any()
        or (lows > np.minimum(opens, closes)).any()
        or (highs < lows).any()
    ):
        raise ValueError("MCR-7 market violates OHLC invariants")
    return market, {
        "sha256": MARKET_DATA_SHA256,
        "rows": int(len(market)),
        "columns_parsed": ["date", "open", "high", "low", "close"],
        "physical_parse_boundary": "stop before parsing first date >= 2024-01-01",
        "first_date": str(market["date"].iloc[0]),
        "last_date": str(market["date"].iloc[-1]),
    }


def load_realized_funding() -> tuple[pd.DataFrame, dict[str, Any]]:
    verify_evaluation_freeze()
    if _sha256(FUNDING_MANIFEST) != FUNDING_MANIFEST_SHA256:
        raise ValueError("MCR-7 funding manifest differs from its frozen hash")
    if _sha256(FUNDING_DATA) != FUNDING_DATA_SHA256:
        raise ValueError("MCR-7 funding data differs from its frozen hash")
    manifest = json.loads(FUNDING_MANIFEST.read_text())
    if manifest.get("outcomes_opened") is not False:
        raise ValueError("MCR-7 funding source lacks unopened-source provenance")
    if manifest.get("strategy_outcomes_calculated") != []:
        raise ValueError("MCR-7 funding source calculated a strategy outcome")
    if manifest.get("data", {}).get("sha256") != FUNDING_DATA_SHA256:
        raise ValueError("MCR-7 funding manifest data hash differs")
    if manifest.get("quality", {}).get("events") != manifest["data"]["rows"]:
        raise ValueError("MCR-7 funding-mark event count differs from manifest")
    if (
        manifest.get("quality", {}).get(
            "maximum_proxy_funding_cash_error_bp_notional", float("inf")
        )
        > manifest.get("mapping", {}).get(
            "maximum_allowed_proxy_funding_cash_error_bp_notional", -1.0
        )
    ):
        raise ValueError("MCR-7 funding-mark proxy error exceeds frozen limit")
    funding = pd.read_csv(
        FUNDING_DATA,
        usecols=[
            "funding_time_ms",
            "funding_time_utc",
            "symbol",
            "funding_rate",
            "settlement_mark_price",
            "funding_time_offset_ms",
            "mark_source",
        ],
        dtype={
            "symbol": str,
            "funding_rate": str,
            "settlement_mark_price": str,
            "mark_source": str,
        },
    )
    if len(funding) != manifest["data"]["rows"]:
        raise ValueError("MCR-7 funding row count differs from manifest")
    funding["funding_time_ms"] = pd.to_numeric(
        funding["funding_time_ms"], errors="raise"
    ).astype(np.int64)
    utc = pd.to_datetime(funding["funding_time_utc"], utc=True, errors="raise").dt.tz_convert(None)
    epoch = pd.to_datetime(
        funding["funding_time_ms"], unit="ms", utc=True, errors="raise"
    ).dt.tz_convert(None)
    if not utc.equals(epoch):
        raise ValueError("MCR-7 funding timestamps disagree")
    if funding["funding_time_ms"].duplicated().any() or not funding[
        "funding_time_ms"
    ].is_monotonic_increasing:
        raise ValueError("MCR-7 funding timestamps are invalid")
    if not funding["symbol"].eq("BTCUSDT").all():
        raise ValueError("MCR-7 funding contains another symbol")
    if not funding["mark_source"].eq("binance_8h_mark_price_kline_open").all():
        raise ValueError("MCR-7 funding uses another mark-price source")
    offsets = pd.to_numeric(
        funding["funding_time_offset_ms"], errors="raise"
    ).to_numpy(np.int64)
    maximum_offset = manifest["mapping"]["maximum_allowed_timestamp_offset_ms"]
    if (offsets < 0).any() or (offsets > maximum_offset).any():
        raise ValueError("MCR-7 funding timestamps exceed the mark mapping tolerance")
    rates = pd.to_numeric(funding["funding_rate"], errors="raise").to_numpy(float)
    mark_prices = pd.to_numeric(
        funding["settlement_mark_price"], errors="raise"
    ).to_numpy(float)
    if (
        not np.isfinite(rates).all()
        or not np.isfinite(mark_prices).all()
        or (mark_prices <= 0.0).any()
        or utc.max() >= SELECTION_END
    ):
        raise ValueError("MCR-7 funding values or interval are invalid")
    normalized = pd.DataFrame(
        {
            "funding_time_ms": funding["funding_time_ms"].to_numpy(np.int64),
            "funding_time": utc,
            "funding_rate": rates,
            "settlement_mark_price": mark_prices,
        }
    )
    return normalized, {
        "manifest_sha256": FUNDING_MANIFEST_SHA256,
        "data_sha256": FUNDING_DATA_SHA256,
        "rows": int(len(normalized)),
        "first_funding_time": str(normalized["funding_time"].iloc[0]),
        "last_funding_time": str(normalized["funding_time"].iloc[-1]),
    }


def attach_market_positions(schedule: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    attached = schedule.copy()
    positions = pd.Series(np.arange(len(market), dtype=np.int64), index=market["date"])
    for label in ("entry", "exit"):
        mapped = attached[f"{label}_date"].map(positions)
        if mapped.isna().any():
            missing = attached.loc[mapped.isna(), f"{label}_date"].head().tolist()
            raise ValueError(f"MCR-7 {label} timestamps missing from market: {missing}")
        attached[f"{label}_position"] = mapped.astype(np.int64)
    if not (attached["entry_position"] < attached["exit_position"]).all():
        raise ValueError("MCR-7 market positions violate the frozen clock")
    return attached


def monthly_cluster_sign_flip(
    values: Iterable[float],
    entry_dates: Iterable[pd.Timestamp | str],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    returns = np.asarray(list(values), dtype=float)
    dates = pd.to_datetime(list(entry_dates))
    if len(returns) == 0 or len(returns) != len(dates):
        return {
            "p_value_one_sided": 1.0,
            "observed_mean_return": 0.0,
            "cluster_count": 0,
            "method": "empty",
            "permutations": 0,
            "seed": int(seed),
        }
    frame = pd.DataFrame({"month": dates.to_period("M"), "return": returns})
    clusters = frame.groupby("month", sort=True)["return"].sum().to_numpy(float)
    observed = float(np.sum(clusters) / len(returns))
    if len(clusters) <= 18:
        outcomes = np.fromiter(
            (
                np.dot(signs, clusters) / len(returns)
                for signs in product((-1.0, 1.0), repeat=len(clusters))
            ),
            dtype=float,
        )
        p_value = float(np.mean(outcomes >= observed - 1e-15))
        method = "exact"
        completed = int(len(outcomes))
    else:
        rng = np.random.default_rng(seed)
        exceedances = 0
        completed = 0
        while completed < permutations:
            batch = min(10_000, permutations - completed)
            signs = rng.integers(0, 2, size=(batch, len(clusters)), dtype=np.int8)
            signs = signs.astype(float) * 2.0 - 1.0
            randomized = signs.dot(clusters) / len(returns)
            exceedances += int(np.count_nonzero(randomized >= observed - 1e-15))
            completed += batch
        p_value = float((1 + exceedances) / (permutations + 1))
        method = "monte_carlo"
    return {
        "p_value_one_sided": p_value,
        "observed_mean_return": observed,
        "cluster_count": int(len(clusters)),
        "method": method,
        "permutations": completed,
        "seed": int(seed),
    }


def _trade_statistics(values: list[float]) -> dict[str, Any]:
    count = len(values)
    if not count:
        return {
            "n_trades": 0,
            "mean_trade_return_pct": 0.0,
            "std_trade_return_pct": 0.0,
            "t_stat_like": 0.0,
            "ci95_mean_trade_return_pct": [0.0, 0.0],
        }
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    standard_error = std / math.sqrt(count)
    return {
        "n_trades": count,
        "mean_trade_return_pct": mean * 100.0,
        "std_trade_return_pct": std * 100.0,
        "t_stat_like": mean / standard_error if standard_error > 0.0 else 0.0,
        "ci95_mean_trade_return_pct": [
            (mean - 1.96 * standard_error) * 100.0,
            (mean + 1.96 * standard_error) * 100.0,
        ],
    }


def _slice_schedule(schedule: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    inside = (
        schedule["entry_date"].ge(start_timestamp)
        & schedule["exit_date"].gt(start_timestamp)
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
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp or cfg.leverage <= 0.0:
        raise ValueError("MCR-7 simulation parameters are invalid")
    per_side_cost = cost_notional_per_side * cfg.leverage
    if not 0.0 <= per_side_cost < 1.0:
        raise ValueError("MCR-7 per-side cost is invalid")
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    dates = market["date"]
    market_ms = (dates.astype("int64") // 1_000_000).to_numpy(np.int64)
    funding_times = funding["funding_time_ms"].to_numpy(np.int64)
    funding_rates = funding["funding_rate"].to_numpy(float)
    funding_mark_prices = funding["settlement_mark_price"].to_numpy(float)

    equity = 1.0
    peak = 1.0
    strict_mdd = 0.0
    previous_exit = -1
    trade_returns: list[float] = []
    gross_returns: list[float] = []
    entry_dates: list[pd.Timestamp] = []
    sides: list[int] = []
    settlement_count = 0
    trades_with_funding = 0
    total_funding_return = 0.0

    for row in schedule.itertuples(index=False):
        entry_position = int(row.entry_position)
        exit_position = int(row.exit_position)
        side = int(row.side)
        if side not in (-1, 1):
            raise ValueError("MCR-7 side must be long or short")
        if not 0 <= entry_position < exit_position < len(market):
            raise ValueError("MCR-7 scheduled positions are invalid")
        if entry_position < previous_exit:
            raise ValueError("MCR-7 schedules overlap")
        entry_time = pd.Timestamp(row.entry_date)
        exit_time = pd.Timestamp(row.exit_date)
        if entry_time != dates.iloc[entry_position] or exit_time != dates.iloc[exit_position]:
            raise ValueError("MCR-7 schedule timestamp differs from market")
        if not (
            start_timestamp <= entry_time < end_timestamp
            and start_timestamp < exit_time < end_timestamp
        ):
            raise ValueError("MCR-7 trade crosses a simulation split")

        entry_price = float(opens[entry_position])
        exit_price = float(opens[exit_position])
        held_high = float(np.max(highs[entry_position:exit_position]))
        held_low = float(np.min(lows[entry_position:exit_position]))
        if min(entry_price, exit_price, held_high, held_low) <= 0.0:
            raise ValueError("MCR-7 scheduled trade has an invalid price")

        entry_ms = int(market_ms[entry_position])
        exit_ms = int(market_ms[exit_position])
        left = int(np.searchsorted(funding_times, entry_ms, side="left"))
        right = int(np.searchsorted(funding_times, exit_ms, side="left"))
        rates = funding_rates[left:right]
        marks = funding_mark_prices[left:right]
        funding_contributions = -cfg.leverage * side * rates * (marks / entry_price)
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
        strict_mdd = max(strict_mdd, 1.0 - max(0.0, equity) / peak)

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
        adverse_factor = max(
            0.0,
            1.0
            - per_side_cost
            + cfg.leverage * side * (adverse_price / entry_price - 1.0)
            + funding_debit
            - per_side_cost * (adverse_price / entry_price),
        )
        adverse_equity = entry_equity * adverse_factor
        strict_mdd = max(
            strict_mdd,
            1.0 - max(0.0, adverse_equity) / intratrade_peak,
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
        entry_dates.append(entry_time)
        sides.append(side)
        settlement_count += int(len(rates))
        trades_with_funding += int(len(rates) > 0)
        total_funding_return += funding_return
        previous_exit = exit_position

    years = (end_timestamp - start_timestamp).total_seconds() / (365.25 * 86_400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    strict_mdd_pct = strict_mdd * 100.0
    if strict_mdd_pct > 1e-12:
        ratio = float(cagr / strict_mdd_pct)
        zero_mdd_ratio_cap_applied = False
    elif cagr > 0.0:
        ratio = 1.0e12
        zero_mdd_ratio_cap_applied = True
    else:
        ratio = 0.0
        zero_mdd_ratio_cap_applied = False
    cluster = (
        monthly_cluster_sign_flip(
            trade_returns,
            entry_dates,
            permutations=cfg.cluster_permutations,
            seed=cfg.cluster_seed,
        )
        if compute_cluster
        else None
    )
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_mdd_pct),
        "cagr_to_strict_mdd": ratio,
        "zero_mdd_ratio_cap_applied": zero_mdd_ratio_cap_applied,
        "trade_count": int(len(sides)),
        "long_count": int(sum(side > 0 for side in sides)),
        "short_count": int(sum(side < 0 for side in sides)),
        "wall_clock_years": float(years),
        "mean_gross_underlying_move_bp": (
            float(np.mean(gross_returns) * 10_000.0) if gross_returns else 0.0
        ),
        "funding_settlement_count": int(settlement_count),
        "trades_with_funding": int(trades_with_funding),
        "total_funding_return_pct_of_entry_equity_sum": float(
            total_funding_return * 100.0
        ),
        "execution_cost_notional_per_side_bp": float(
            cost_notional_per_side * 10_000.0
        ),
        "trade_statistics": _trade_statistics(trade_returns),
        "monthly_cluster_sign_flip": cluster,
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
            "base_6bp": simulate_schedule(
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
    gates = prereg.protocol()["selection_protocol"]["gates"]
    failures: list[str] = []
    for name in ("train", "select2023"):
        base = windows[name]["base_6bp"]
        stress = windows[name]["stress_10bp"]
        if base["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if base["cagr_to_strict_mdd"] < gates["train_and_2023_cagr_to_strict_mdd_min"]:
            failures.append(f"{name}: CAGR/strict-MDD below 3")
        if base["strict_mdd_pct"] > gates["train_and_2023_strict_mdd_pct_max"]:
            failures.append(f"{name}: strict MDD above 15%")
        if stress["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: 10bp stress non-positive")
        if (
            base["mean_gross_underlying_move_bp"]
            < cfg.minimum_mean_gross_underlying_bp
        ):
            failures.append(f"{name}: mean gross edge below 40 bp")
        cluster = base["monthly_cluster_sign_flip"]
        if cluster is None or cluster["p_value_one_sided"] > gates[
            "train_and_2023_monthly_cluster_signflip_p_max"
        ]:
            failures.append(f"{name}: monthly-cluster p-value above 0.10")
    for name in ("train2021", "train2022", "select2023_h1", "select2023_h2"):
        if windows[name]["base_6bp"]["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
    return failures


def qualification(
    policy_windows: dict[str, dict[str, Any]], cfg: EvaluationConfig
) -> dict[str, Any]:
    primary_failures = _primary_gate_failures(policy_windows["primary"], cfg)
    delayed_failures = [
        f"{name}: one-bar-delayed entry non-positive absolute return"
        for name in ("train", "select2023")
        if policy_windows["one_bar_delayed_entry"][name]["base_6bp"][
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
        "direction_random_and_constant_controls_are_diagnostic_only": True,
    }


def _selection_decision(verdict: dict[str, Any]) -> dict[str, Any]:
    if verdict["qualifies"]:
        return {
            "selected_alpha": None,
            "performance_candidate": "MCR-7",
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


def run_evaluation(cfg: EvaluationConfig | None = None) -> dict[str, Any]:
    frozen_cfg = EvaluationConfig() if cfg is None else cfg
    if frozen_cfg != EvaluationConfig():
        raise ValueError("MCR-7 evaluation parameters are frozen")
    evaluator_freeze = verify_evaluation_freeze()
    clocks, preregistration, support = verify_support_and_control_clocks(frozen_cfg)
    expected_hashes = evaluator_freeze["control_clock_hashes"]
    actual_hashes = {name: _clock_hash(clock) for name, clock in clocks.items()}
    if actual_hashes != expected_hashes:
        raise ValueError("MCR-7 control clock differs from pre-outcome freeze")
    market, market_source = load_execution_market()
    funding, funding_source = load_realized_funding()
    schedules = {
        name: attach_market_positions(clock, market) for name, clock in clocks.items()
    }
    windows = {
        name: _evaluate_policy_windows(market, funding, schedule, frozen_cfg)
        for name, schedule in schedules.items()
    }
    verdict = qualification(windows, frozen_cfg)
    core = {
        "protocol": {
            "name": "MCR-7 frozen pre-2024 evaluation",
            "support_commit": SUPPORT_COMMIT,
            "evaluation_source_commit": evaluator_freeze["evaluation_source_commit"],
            "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
            "evaluation_freeze_sha256": _sha256(EVALUATION_FREEZE),
            "outcomes_opened": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["2024", "2025", "2026_ytd"],
            "parameters_mutable": False,
            "funding_interval": "entry_time <= funding_time < exit_time",
            "funding_notional": (
                "fixed entry quantity times the exact frozen realized funding rate "
                "and its returned settlement mark_price"
            ),
            "strict_mdd": (
                "global/pre-entry HWM; entry cost; favorable then adverse held OHLC; "
                "funding credits raise HWM and debits lower the adverse envelope; "
                "hypothetical adverse liquidation cost; exit cost"
            ),
            "cagr": "full declared wall-clock split including idle cash",
            "absolute_return_always_reported": True,
        },
        "policy": preregistration["policy"],
        "evaluation_config": asdict(frozen_cfg),
        "source": {
            "support_result_sha256": SUPPORT_RESULT_SHA256,
            "primary_clock_sha256": PRIMARY_CLOCK_SHA256,
            "market": market_source,
            "funding": funding_source,
            "support": support["source"],
        },
        "control_clock_counts": {name: int(len(clock)) for name, clock in clocks.items()},
        "control_clock_hashes": actual_hashes,
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
        )
    }


def main() -> None:
    output = Path(DEFAULT_OUTPUT)
    if output.exists():
        raise RuntimeError("MCR-7 outcome result already exists")
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
