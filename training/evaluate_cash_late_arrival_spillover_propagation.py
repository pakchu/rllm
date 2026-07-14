"""Frozen pre-2024 return evaluator for CLASP-24.

The evaluator must be committed and hash-recorded before it can load USD-M
execution prices or realized funding. It reserves the primary and every control
clock from causal features first, then opens only 2020-2023 outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_cash_late_arrival_spillover_propagation as clasp
from training.evaluate_metaorder_fragmentation_impact_curvature import (
    weekly_cluster_sign_flip,
)
from training.strict_bar_backtest import _trade_stats


PREREGISTRATION_COMMIT = "29e3983"
SUPPORT_COMMIT = "aa6fab4"
CLOCK_COMMIT = "e5758d4"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cash_late_arrival_spillover_propagation.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "76059910f2aa7eca4f47103f9fbe1494b8f779ce152c619c2cde7627c18529a4"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cash-late-arrival-spillover-propagation-preregistration-2026-07-14.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "9fa551a639247d0b1c3a172d2afe654d84922e3187b7d17f94b59250f7e8c188"
)
SUPPORT_DOCUMENT = Path(
    "docs/cash-late-arrival-spillover-propagation-support-decision-2026-07-14.md"
)
SUPPORT_DOCUMENT_SHA256 = (
    "a402842fa8645d90158ce2b51a7f4b6dcccb108608fca9f78e7968331a47c2b2"
)
CLOCK_DOCUMENT = Path(
    "docs/cash-late-arrival-spillover-propagation-clock-freeze-2026-07-14.md"
)
CLOCK_DOCUMENT_SHA256 = (
    "4b2e2b0965a3d7186641e0a882aa7906031a671584cfc8943c908c8bf2abe1bc"
)
SUPPORT_RESULT = Path(
    "results/cash_late_arrival_spillover_propagation_support_2026-07-14.json"
)
SUPPORT_RESULT_SHA256 = (
    "bd26905f7c33360a62c9eb14cef23ba917612e64fc5d83e47e25b50b56db8930"
)
EVENT_CLOCK = Path(
    "results/cash_late_arrival_spillover_propagation_clock_2026-07-14.csv"
)
EVENT_CLOCK_SHA256 = "e166f4bd24afd5a2f129bcc26393ad4293ad0bc5792686b3b0fc4a805d53f9d5"
EVENT_CLOCK_MANIFEST = Path(
    "results/cash_late_arrival_spillover_propagation_clock_manifest_2026-07-14.json"
)
EVENT_CLOCK_MANIFEST_SHA256 = (
    "ba32d90eacc3ce63d7723eeb0b3c5b078e81f9b8a5a5dbc6cdd9f6fceb5b6d02"
)
CLOCK_FREEZE_SOURCE = Path(
    "training/freeze_cash_late_arrival_spillover_propagation_clock.py"
)
CLOCK_FREEZE_SOURCE_SHA256 = (
    "29021bb7ea7bc0e3d4aa87c7b5ceaa867b4302a066a233f50ac81c4ade2ec4d5"
)
FEATURE_BUILDER_SOURCE = Path("training/build_binance_cross_venue_minute_leadership.py")
FEATURE_BUILDER_SOURCE_SHA256 = (
    "a3b18b35cd7fb0a7230a720cc50de79d84312543638bd70732b35dd36209085f"
)
FEATURE_MANIFEST = Path(
    "data/binance_cross_venue_minute_leadership_btc_2020_2023/build_manifest.json"
)
FEATURE_MANIFEST_SHA256 = (
    "544c2945a2b56be478a1edc4abbb93b762bda5afc32cbd0658dd6822ff6b70fa"
)
SOURCE_AUDIT = Path(
    "results/binance_cross_venue_minute_leadership_audit_2026-07-14.json"
)
SOURCE_AUDIT_SHA256 = "ffe0124ac9c5c0c3f1d1c284b672618cf910dc16cae36e65c1efe79710f039af"
CLUSTER_SOURCE = Path("training/evaluate_metaorder_fragmentation_impact_curvature.py")
CLUSTER_SOURCE_SHA256 = (
    "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3"
)
TRADE_STATS_SOURCE = Path("training/strict_bar_backtest.py")
TRADE_STATS_SOURCE_SHA256 = (
    "3e95ad320d8869755afa1f4907d2d478200a3ebfc015e4eaeace0be0b15f9682"
)
MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
MARKET_DATA = Path(
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
MARKET_DATA_SHA256 = "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
FUNDING_DATA = Path("results/binance_um_btcusdt_realized_funding_2020_2023.csv")
FUNDING_DATA_SHA256 = "c19829fa085a50f29c13762373a2b6db1c62025d657be1f5a3fbb9ce254482f7"
FUNDING_MANIFEST = Path(
    "results/binance_um_btcusdt_realized_funding_2020_2023_manifest.json"
)
FUNDING_MANIFEST_SHA256 = (
    "c70280e46bcbc2410cc59c2bcc93780c40997dbc5d0edb82d82127b59593250c"
)
EVALUATION_SOURCE = Path("training/evaluate_cash_late_arrival_spillover_propagation.py")
EVALUATION_FREEZE = Path(
    "results/cash_late_arrival_spillover_propagation_evaluator_freeze_2026-07-14.json"
)

SELECTED_QUANTILE = 0.75
WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2023-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
}
POLICY_NAMES = tuple(clasp.CONTROL_SIDE_RULES)
SCORE_BEARING_CONTROLS = clasp.SCORE_BEARING_CONTROLS
FALSIFICATION_CONTROLS = ("direction_flip", "signal_delay_1bar")
CLOCK_COLUMNS = clasp.SCHEDULE_COLUMNS


@dataclass(frozen=True)
class EvaluationConfig:
    output: str = (
        "results/cash_late_arrival_spillover_propagation_selection_2026-07-14.json"
    )
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_714
    minimum_mean_gross_underlying_bp: float = 12.0
    minimum_2023_half_trades: int = 65


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_evaluation_config(cfg: EvaluationConfig) -> None:
    expected = {
        "leverage": 0.5,
        "fee_rate": 0.0005,
        "slippage_rate": 0.0001,
        "cluster_permutations": 100_000,
        "cluster_seed": 20_260_714,
        "minimum_mean_gross_underlying_bp": 12.0,
        "minimum_2023_half_trades": 65,
    }
    changed = {
        name: {"expected": value, "observed": getattr(cfg, name)}
        for name, value in expected.items()
        if getattr(cfg, name) != value
    }
    if changed:
        raise ValueError(f"CLASP evaluation config is frozen: {changed}")


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.is_file():
        raise ValueError("CLASP evaluator freeze manifest is missing")
    freeze = _read_json(EVALUATION_FREEZE)
    if freeze.get("outcomes_opened_for_clasp24") is not False:
        raise ValueError("CLASP evaluator was not frozen before outcomes")
    if freeze.get("evaluation_source") != str(EVALUATION_SOURCE):
        raise ValueError("CLASP evaluator freeze path changed")
    if freeze.get("evaluation_source_sha256") != _sha256(EVALUATION_SOURCE):
        raise ValueError("CLASP evaluator differs from pre-outcome freeze")
    if len(str(freeze.get("evaluation_source_commit", ""))) != 40:
        raise ValueError("CLASP evaluator source commit is not full length")
    expected_values = {
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "support_commit": SUPPORT_COMMIT,
        "clock_commit": CLOCK_COMMIT,
        "support_result_sha256": SUPPORT_RESULT_SHA256,
        "event_clock_sha256": EVENT_CLOCK_SHA256,
        "market_data_sha256": MARKET_DATA_SHA256,
        "funding_data_sha256": FUNDING_DATA_SHA256,
    }
    for key, expected in expected_values.items():
        if freeze.get(key) != expected:
            raise ValueError(f"CLASP evaluator freeze {key} changed")
    if freeze.get("opened_windows") != []:
        raise ValueError("CLASP evaluator freeze already opened a window")
    if freeze.get("returns_prices_or_funding_loaded_during_freeze") is not False:
        raise ValueError("CLASP evaluator freeze loaded outcomes")
    if freeze.get("mutable_parameters") != []:
        raise ValueError("CLASP evaluator freeze permits mutable parameters")
    expected_sealed = [*WINDOWS, "test2024", "eval2025", "ytd2026"]
    if freeze.get("sealed_windows") != expected_sealed:
        raise ValueError("CLASP evaluator freeze sealed windows changed")
    return freeze


def verify_preregistration() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frozen_files = (
        (PREREGISTRATION_SOURCE, PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (SUPPORT_DOCUMENT, SUPPORT_DOCUMENT_SHA256),
        (CLOCK_DOCUMENT, CLOCK_DOCUMENT_SHA256),
        (SUPPORT_RESULT, SUPPORT_RESULT_SHA256),
        (EVENT_CLOCK, EVENT_CLOCK_SHA256),
        (EVENT_CLOCK_MANIFEST, EVENT_CLOCK_MANIFEST_SHA256),
        (CLOCK_FREEZE_SOURCE, CLOCK_FREEZE_SOURCE_SHA256),
        (FEATURE_BUILDER_SOURCE, FEATURE_BUILDER_SOURCE_SHA256),
        (FEATURE_MANIFEST, FEATURE_MANIFEST_SHA256),
        (SOURCE_AUDIT, SOURCE_AUDIT_SHA256),
        (CLUSTER_SOURCE, CLUSTER_SOURCE_SHA256),
        (TRADE_STATS_SOURCE, TRADE_STATS_SOURCE_SHA256),
        (MARKET_MANIFEST, MARKET_MANIFEST_SHA256),
        (FUNDING_MANIFEST, FUNDING_MANIFEST_SHA256),
    )
    for path, expected in frozen_files:
        if _sha256(path) != expected:
            raise ValueError(f"frozen CLASP dependency changed: {path}")

    result = _read_json(SUPPORT_RESULT)
    protocol = result.get("protocol", {})
    if protocol.get("clasp_outcomes_opened") is not False:
        raise ValueError("CLASP support artifact opened outcomes")
    if protocol.get("support_only") is not True:
        raise ValueError("CLASP support artifact is not support-only")
    if protocol.get("control_side_rules") != clasp.CONTROL_SIDE_RULES:
        raise ValueError("CLASP control side rules changed")
    if result.get("support_decision") != "pass":
        raise ValueError("CLASP support gate did not pass")
    if result.get("selected_quantile") != SELECTED_QUANTILE:
        raise ValueError("CLASP selected quantile changed")
    if result.get("config") != asdict(clasp.Config()):
        raise ValueError("CLASP signal config differs from frozen support")
    selected = result.get("selected_support", {})
    if selected.get("passes_support") is not True:
        raise ValueError("CLASP selected support row is not passing")
    if selected.get("support", {}).get("nonoverlap_total") != 615:
        raise ValueError("CLASP selected event count changed")

    clock_manifest = _read_json(EVENT_CLOCK_MANIFEST)
    frozen_protocol = clock_manifest.get("protocol", {})
    clock = clock_manifest.get("clock", {})
    if frozen_protocol.get("outcomes_opened") is not False:
        raise ValueError("CLASP event clock opened outcomes")
    if frozen_protocol.get("preregistration_commit") != PREREGISTRATION_COMMIT:
        raise ValueError("CLASP event clock preregistration changed")
    if frozen_protocol.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("CLASP event clock support commit changed")
    if frozen_protocol.get("selected_quantile") != SELECTED_QUANTILE:
        raise ValueError("CLASP event clock quantile changed")
    if clock.get("sha256") != EVENT_CLOCK_SHA256 or clock.get("rows") != 615:
        raise ValueError("CLASP event clock record changed")
    if clock_manifest.get("support", {}).get("sha256") != SUPPORT_RESULT_SHA256:
        raise ValueError("CLASP event clock support hash changed")

    market_manifest = _read_json(MARKET_MANIFEST)
    market_protocol = market_manifest.get("protocol", {})
    if market_protocol.get("archive_checksums_verified") is not True:
        raise ValueError("CLASP execution archives were not checksum-verified")
    if market_protocol.get("end_is_exclusive") is not True:
        raise ValueError("CLASP execution manifest end contract changed")
    if market_manifest.get("combined_output") != str(MARKET_DATA):
        raise ValueError("CLASP execution market path changed")
    if market_manifest.get("combined_sha256") != MARKET_DATA_SHA256:
        raise ValueError("CLASP execution market hash changed")
    if market_manifest.get("rows") != 420_768:
        raise ValueError("CLASP execution market row count changed")

    funding_manifest = _read_json(FUNDING_MANIFEST)
    funding_protocol = funding_manifest.get("protocol", {})
    funding_data = funding_manifest.get("data", {})
    if funding_protocol.get("luri_outcomes_opened") is not False:
        raise ValueError("CLASP funding source opened outcomes")
    if funding_data.get("path") != str(FUNDING_DATA):
        raise ValueError("CLASP funding data path changed")
    if funding_data.get("sha256") != FUNDING_DATA_SHA256:
        raise ValueError("CLASP funding data hash changed")
    if funding_data.get("rows") != 4_383:
        raise ValueError("CLASP funding row count changed")
    return result, market_manifest, funding_manifest


def _canonical_clock_records(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    missing = set(CLOCK_COLUMNS).difference(schedule.columns)
    if missing:
        raise ValueError(f"CLASP clock is missing columns: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    for row in schedule[list(CLOCK_COLUMNS)].itertuples(index=False):
        records.append(
            {
                "signal_position": int(row.signal_position),
                "entry_position": int(row.entry_position),
                "exit_position": int(row.exit_position),
                "signal_date": str(row.signal_date),
                "entry_date": str(row.entry_date),
                "exit_date": str(row.exit_date),
                "side": int(row.side),
                "branch": str(row.branch),
                "hold_bars": int(row.hold_bars),
            }
        )
    return records


def verify_signal_replay(
    frame: pd.DataFrame,
    cfg: clasp.Config,
    support_result: dict[str, Any],
    source: dict[str, Any],
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], pd.DataFrame]:
    signal, controls, control_sides, _ = clasp.classify_events(
        frame, cfg, quantile=SELECTED_QUANTILE
    )
    replayed = clasp.nonoverlapping_schedule(signal, frame)
    frozen = pd.read_csv(EVENT_CLOCK)
    if _canonical_clock_records(replayed) != _canonical_clock_records(frozen):
        raise ValueError("CLASP primary event-clock replay differs from freeze")
    if not replayed["entry_position"].eq(replayed["signal_position"] + 1).all():
        raise ValueError("CLASP replay is not next-open entry")
    if not replayed["hold_bars"].eq(cfg.hold_bars).all():
        raise ValueError("CLASP replay hold changed")
    if (
        not replayed["exit_position"]
        .eq(replayed["entry_position"] + cfg.hold_bars)
        .all()
    ):
        raise ValueError("CLASP replay exit changed")

    selected = support_result["selected_support"]
    if clasp._support(replayed, cfg) != selected["support"]:
        raise ValueError("CLASP selected support replay differs from freeze")
    raw_counts = {
        name: int(mask.sum()) for name, mask in controls.items() if name != "primary"
    }
    if raw_counts != selected["control_raw_counts"]:
        raise ValueError("CLASP control raw clocks differ from freeze")
    if int(controls["primary"].sum()) != selected["raw_primary"]:
        raise ValueError("CLASP raw primary clock differs from freeze")
    clock_manifest = _read_json(EVENT_CLOCK_MANIFEST)
    if source != clock_manifest.get("source"):
        raise ValueError("CLASP source replay differs from clock freeze")
    return controls, control_sides, replayed.reset_index(drop=True)


def _schedule_from_control(
    frame: pd.DataFrame,
    cfg: clasp.Config,
    *,
    mask: pd.Series,
    side: pd.Series,
    branch: str,
) -> pd.DataFrame:
    active = mask.fillna(False).astype(bool)
    numeric_side = pd.to_numeric(side, errors="coerce").fillna(0.0)
    if not numeric_side.loc[active].isin((-1.0, 1.0)).all():
        raise ValueError(f"CLASP control {branch} has an invalid active side")
    signal = pd.DataFrame(
        {
            "side": np.where(active, numeric_side, 0).astype(np.int8),
            "branch": np.where(active, branch, "none"),
            "hold_bars": np.where(active, cfg.hold_bars, 0).astype(np.int16),
        }
    )
    return (
        clasp.nonoverlapping_schedule(signal, frame)
        .reindex(columns=CLOCK_COLUMNS)
        .reset_index(drop=True)
    )


def build_control_schedules(
    frame: pd.DataFrame,
    controls: dict[str, pd.Series],
    control_sides: dict[str, pd.Series],
    primary_schedule: pd.DataFrame,
    cfg: clasp.Config,
    support_result: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    frozen_primary = primary_schedule.reindex(columns=CLOCK_COLUMNS).reset_index(
        drop=True
    )
    schedules: dict[str, pd.DataFrame] = {"primary": frozen_primary}
    flipped = frozen_primary.copy()
    flipped["side"] = -flipped["side"].astype(np.int8)
    flipped["branch"] = "direction_flip"
    schedules["direction_flip"] = flipped
    for name in POLICY_NAMES[2:]:
        schedules[name] = _schedule_from_control(
            frame,
            cfg,
            mask=controls[name],
            side=control_sides[name],
            branch=name,
        )
    if tuple(schedules) != POLICY_NAMES:
        raise ValueError("CLASP evaluator control set differs from preregistration")

    if support_result is not None:
        expected = support_result["selected_support"]["control_scheduled_counts"]
        observed = {
            name: int(len(schedules[name]))
            for name in POLICY_NAMES
            if name not in {"primary", "direction_flip"}
        }
        if observed != expected:
            raise ValueError("CLASP control scheduled clocks differ from freeze")
    return schedules


def slice_schedule(schedule: pd.DataFrame, *, start: str, end: str) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp:
        raise ValueError("CLASP split start must precede end")
    if schedule.empty:
        return schedule.copy()
    signal_date = pd.to_datetime(schedule["signal_date"], errors="raise")
    entry_date = pd.to_datetime(schedule["entry_date"], errors="raise")
    exit_date = pd.to_datetime(schedule["exit_date"], errors="raise")
    inside = (
        signal_date.ge(start_timestamp)
        & signal_date.lt(end_timestamp)
        & entry_date.ge(start_timestamp)
        & entry_date.lt(end_timestamp)
        & exit_date.ge(start_timestamp)
        & exit_date.lt(end_timestamp)
    )
    return schedule.loc[inside].reset_index(drop=True)


def load_execution_market(
    signal_frame: pd.DataFrame, market_manifest: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if _sha256(MARKET_DATA) != MARKET_DATA_SHA256:
        raise ValueError("CLASP execution market differs from frozen SHA-256")
    market = pd.read_csv(
        MARKET_DATA,
        compression="gzip",
        usecols=["date", "open", "high", "low", "close"],
        parse_dates=["date"],
    )
    if len(market) != market_manifest["rows"]:
        raise ValueError("CLASP execution market row count differs from manifest")
    if market["date"].duplicated().any() or not market["date"].is_monotonic_increasing:
        raise ValueError("CLASP execution timestamps are duplicate or unordered")
    if not market["date"].equals(signal_frame["date"]):
        raise ValueError("CLASP execution market does not align with signal clock")
    prices = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("CLASP execution market contains invalid prices")
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    closes = market["close"].to_numpy(float)
    if (
        (highs < np.maximum(opens, closes)).any()
        or (lows > np.minimum(opens, closes)).any()
        or (highs < lows).any()
    ):
        raise ValueError("CLASP execution market violates OHLC invariants")
    frame = signal_frame.assign(open=opens, high=highs, low=lows, close=closes)
    source = {
        "market_manifest_sha256": MARKET_MANIFEST_SHA256,
        "market_data_sha256": MARKET_DATA_SHA256,
        "market_rows": int(len(market)),
        "first_date": str(market["date"].iloc[0]),
        "last_date": str(market["date"].iloc[-1]),
    }
    return frame, source


def load_realized_funding(
    funding_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if _sha256(FUNDING_DATA) != FUNDING_DATA_SHA256:
        raise ValueError("CLASP realized funding differs from frozen SHA-256")
    funding = pd.read_csv(
        FUNDING_DATA,
        dtype={"symbol": str, "funding_rate": str, "mark_price": str},
    )
    expected_columns = [
        "funding_time_ms",
        "funding_time_utc",
        "symbol",
        "funding_rate",
        "mark_price",
    ]
    if list(funding.columns) != expected_columns:
        raise ValueError("CLASP funding columns differ from freeze")
    if len(funding) != funding_manifest["data"]["rows"]:
        raise ValueError("CLASP funding row count differs from manifest")
    funding["funding_time_ms"] = pd.to_numeric(
        funding["funding_time_ms"], errors="raise"
    ).astype(np.int64)
    funding_time = pd.to_datetime(
        funding["funding_time_utc"], utc=True, errors="raise"
    ).dt.tz_convert(None)
    funding_from_ms = pd.to_datetime(
        funding["funding_time_ms"], unit="ms", utc=True, errors="raise"
    ).dt.tz_convert(None)
    if not funding_time.equals(funding_from_ms):
        raise ValueError("CLASP funding UTC strings differ from epoch milliseconds")
    if (
        funding["funding_time_ms"].duplicated().any()
        or not funding["funding_time_ms"].is_monotonic_increasing
    ):
        raise ValueError("CLASP funding timestamps are duplicate or unordered")
    if not funding["symbol"].eq("BTCUSDT").all():
        raise ValueError("CLASP funding source contains a non-BTCUSDT symbol")
    rates = pd.to_numeric(funding["funding_rate"], errors="raise").to_numpy(float)
    if not np.isfinite(rates).all():
        raise ValueError("CLASP funding source contains an invalid rate")
    if funding_time.max() >= clasp.SELECTION_END:
        raise ValueError("CLASP funding source opens the sealed interval")
    normalized = pd.DataFrame(
        {
            "funding_time_ms": funding["funding_time_ms"].to_numpy(np.int64),
            "funding_time": funding_time,
            "funding_rate": rates,
        }
    )
    source = {
        "funding_manifest_sha256": FUNDING_MANIFEST_SHA256,
        "funding_data_sha256": FUNDING_DATA_SHA256,
        "funding_rows": int(len(normalized)),
        "first_funding_time": str(normalized["funding_time"].iloc[0]),
        "last_funding_time": str(normalized["funding_time"].iloc[-1]),
    }
    return normalized, source


def simulate_funding_schedule(
    frame: pd.DataFrame,
    funding: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp:
        raise ValueError("simulation start must be before end")
    if cfg.leverage <= 0.0:
        raise ValueError("leverage must be positive")
    per_side_cost = (cfg.fee_rate + cfg.slippage_rate) * cfg.leverage
    if not 0.0 <= per_side_cost < 1.0:
        raise ValueError("per-side execution cost is invalid")

    opens = frame["open"].to_numpy(float)
    highs = frame["high"].to_numpy(float)
    lows = frame["low"].to_numpy(float)
    dates = frame["date"]
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
    branches: list[str] = []
    settlement_counts: list[int] = []
    funding_factors: list[float] = []

    for row in schedule.itertuples(index=False):
        signal_position = int(row.signal_position)
        entry_position = int(row.entry_position)
        exit_position = int(row.exit_position)
        side = int(row.side)
        if side not in (-1, 1):
            raise ValueError("scheduled side must be long or short")
        if not signal_position < entry_position < exit_position:
            raise ValueError("scheduled positions are not strictly ordered")
        if entry_position != signal_position + 1:
            raise ValueError("scheduled entry is not the next five-minute open")
        if exit_position != entry_position + 24:
            raise ValueError("scheduled exit does not match the frozen 24-bar hold")
        if int(row.hold_bars) != 24:
            raise ValueError("scheduled hold_bars does not match the frozen hold")
        if entry_position < previous_exit:
            raise ValueError("scheduled trades overlap")
        if exit_position >= len(frame):
            raise ValueError("scheduled exit exceeds market frame")
        if not (
            start_timestamp <= dates.iloc[signal_position] < end_timestamp
            and start_timestamp <= dates.iloc[entry_position] < end_timestamp
            and start_timestamp <= dates.iloc[exit_position] < end_timestamp
        ):
            raise ValueError("scheduled trade crosses the simulation split")

        entry_price = float(opens[entry_position])
        exit_price = float(opens[exit_position])
        if entry_price <= 0.0 or exit_price <= 0.0:
            raise ValueError("scheduled trade has non-positive open price")
        held_high = float(np.max(highs[entry_position:exit_position]))
        held_low = float(np.min(lows[entry_position:exit_position]))

        entry_ms = int(pd.Timestamp(dates.iloc[entry_position]).value // 1_000_000)
        exit_ms = int(pd.Timestamp(dates.iloc[exit_position]).value // 1_000_000)
        funding_left = int(np.searchsorted(funding_times, entry_ms, side="left"))
        funding_right = int(np.searchsorted(funding_times, exit_ms, side="right"))
        trade_funding_rates = funding_rates[funding_left:funding_right]
        factors = 1.0 - cfg.leverage * side * trade_funding_rates
        if not np.isfinite(factors).all() or (factors <= 0.0).any():
            raise ValueError("scheduled trade has an invalid funding factor")
        funding_factor = float(np.prod(factors, dtype=float))
        funding_debit_factor = float(np.prod(np.minimum(factors, 1.0), dtype=float))

        entry_equity = equity
        equity *= 1.0 - per_side_cost
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        post_entry_equity = equity

        favorable_price = held_high if side > 0 else held_low
        adverse_price = held_low if side > 0 else held_high
        favorable_equity = max(
            0.0,
            post_entry_equity
            * (1.0 + cfg.leverage * side * (favorable_price / entry_price - 1.0)),
        )
        intratrade_peak = max(peak, favorable_equity)
        adverse_equity = max(
            0.0,
            post_entry_equity
            * funding_debit_factor
            * (1.0 + cfg.leverage * side * (adverse_price / entry_price - 1.0)),
        )
        strict_mdd = max(strict_mdd, 1.0 - adverse_equity / intratrade_peak)
        peak = max(peak, intratrade_peak)

        raw_return = side * (exit_price / entry_price - 1.0)
        price_factor = max(0.0, 1.0 + cfg.leverage * raw_return)
        equity = post_entry_equity * price_factor * funding_factor
        equity *= 1.0 - per_side_cost
        strict_mdd = max(strict_mdd, 1.0 - equity / peak)
        peak = max(peak, equity)

        trade_returns.append(equity / entry_equity - 1.0)
        gross_returns.append(raw_return)
        entry_dates.append(str(dates.iloc[entry_position]))
        sides.append(side)
        branches.append(str(row.branch))
        settlement_counts.append(int(len(trade_funding_rates)))
        funding_factors.append(funding_factor)
        previous_exit = exit_position

    years = (end_timestamp - start_timestamp).total_seconds() / (365.25 * 86_400.0)
    absolute_return = (equity - 1.0) * 100.0
    cagr = (equity ** (1.0 / years) - 1.0) * 100.0 if equity > 0.0 else -100.0
    strict_mdd_pct = strict_mdd * 100.0
    trade_stats = _trade_stats(trade_returns)
    cluster = weekly_cluster_sign_flip(
        trade_returns,
        entry_dates,
        permutations=cfg.cluster_permutations,
        seed=cfg.cluster_seed,
    )
    return {
        "absolute_return_pct": float(absolute_return),
        "cagr_pct": float(cagr),
        "strict_mdd_pct": float(strict_mdd_pct),
        "cagr_to_strict_mdd": (
            float(cagr / strict_mdd_pct) if strict_mdd_pct > 1e-12 else 0.0
        ),
        "trade_count": int(len(trade_returns)),
        "long_count": int(sum(side > 0 for side in sides)),
        "short_count": int(sum(side < 0 for side in sides)),
        "wall_clock_years": float(years),
        "mean_gross_underlying_move_bp": (
            float(np.mean(gross_returns) * 10_000.0) if gross_returns else 0.0
        ),
        "funding_settlement_count": int(sum(settlement_counts)),
        "trades_with_funding": int(sum(count > 0 for count in settlement_counts)),
        "mean_funding_factor": float(np.mean(funding_factors))
        if funding_factors
        else 1.0,
        "branch_counts": {
            branch: int(branches.count(branch)) for branch in sorted(set(branches))
        },
        "trade_statistics": trade_stats,
        "weekly_cluster_sign_flip": cluster,
    }


def evaluate_policy(
    frame: pd.DataFrame,
    funding: pd.DataFrame,
    global_schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    schedule = slice_schedule(global_schedule, start=start, end=end)
    metrics = simulate_funding_schedule(
        frame, funding, schedule, start=start, end=end, cfg=cfg
    )
    metrics["global_clock_count"] = int(len(global_schedule))
    metrics["split_clock_count"] = int(len(schedule))
    return metrics


def _qualification(windows: dict[str, Any], cfg: EvaluationConfig) -> dict[str, Any]:
    train = windows["train"]["primary"]
    select = windows["select2023"]["primary"]
    h1 = windows["select2023_h1"]["primary"]
    h2 = windows["select2023_h2"]["primary"]
    failures: list[str] = []
    for name, metrics in (("train", train), ("select2023", select)):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["cagr_to_strict_mdd"] < 3.0:
            failures.append(f"{name}: CAGR/strict-MDD below 3")
        if metrics["strict_mdd_pct"] > 15.0:
            failures.append(f"{name}: strict MDD above 15%")
        if metrics["weekly_cluster_sign_flip"]["p_value_one_sided"] >= 0.10:
            failures.append(f"{name}: weekly-cluster p-value not below 0.10")
        if (
            metrics["mean_gross_underlying_move_bp"]
            <= cfg.minimum_mean_gross_underlying_bp
        ):
            failures.append(f"{name}: mean gross underlying move not above 12 bp")
    for name, metrics in (("select2023_h1", h1), ("select2023_h2", h2)):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["trade_count"] < cfg.minimum_2023_half_trades:
            failures.append(f"{name}: fewer than 65 trades")

    primary_min_ratio = min(train["cagr_to_strict_mdd"], select["cagr_to_strict_mdd"])
    control_min_ratios: dict[str, float] = {}
    for control in SCORE_BEARING_CONTROLS:
        control_min = min(
            windows["train"][control]["cagr_to_strict_mdd"],
            windows["select2023"][control]["cagr_to_strict_mdd"],
        )
        control_min_ratios[control] = float(control_min)
        if primary_min_ratio <= control_min:
            failures.append(
                "primary: minimum train/select ratio does not beat " + control
            )
    return {
        "qualifies": not failures,
        "failures": failures,
        "primary_min_train_select_ratio": float(primary_min_ratio),
        "score_bearing_control_min_train_select_ratios": control_min_ratios,
    }


def run_evaluation(cfg: EvaluationConfig) -> dict[str, Any]:
    _validate_evaluation_config(cfg)
    evaluator_freeze = verify_evaluation_freeze()
    support_result, market_manifest, funding_manifest = verify_preregistration()
    signal_cfg = clasp.Config()
    signal_frame, signal_source = clasp.load_causal_frame()
    controls, control_sides, primary_schedule = verify_signal_replay(
        signal_frame, signal_cfg, support_result, signal_source
    )
    schedules = build_control_schedules(
        signal_frame,
        controls,
        control_sides,
        primary_schedule,
        signal_cfg,
        support_result,
    )
    frame, market_source = load_execution_market(signal_frame, market_manifest)
    funding, funding_source = load_realized_funding(funding_manifest)

    windows: dict[str, Any] = {}
    for window_name, (start, end) in WINDOWS.items():
        windows[window_name] = {
            policy: evaluate_policy(
                frame,
                funding,
                schedules[policy],
                start=start,
                end=end,
                cfg=cfg,
            )
            for policy in POLICY_NAMES
        }
    qualification = _qualification(windows, cfg)
    return {
        "protocol": {
            "name": "CLASP-24 frozen pre-2024 selection evaluation",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "support_commit": SUPPORT_COMMIT,
            "clock_commit": CLOCK_COMMIT,
            "evaluation_source_commit": evaluator_freeze["evaluation_source_commit"],
            "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
            "evaluation_freeze_manifest_sha256": _sha256(EVALUATION_FREEZE),
            "outcomes_opened_for_clasp24": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
            "signal_parameters_mutable": False,
            "primary_clock_replayed_from_freeze": True,
            "control_clocks_reserved_before_outcomes": True,
            "control_clocks_reserved_before_return_slicing": True,
            "control_side_rules": clasp.CONTROL_SIDE_RULES,
            "score_bearing_controls": list(SCORE_BEARING_CONTROLS),
            "falsification_controls": list(FALSIFICATION_CONTROLS),
            "entry": "next Binance USD-M 5m open after completed signal bar",
            "exit": "scheduled USD-M open 24 completed 5m bars after entry",
            "funding_window": "entry_time <= funding_time <= exit_time",
            "trade_multiplier": (
                "(1-0.0003)*(1+0.5*r)*product(1-0.5*side*funding_rate)*(1-0.0003)"
            ),
            "strict_mdd": (
                "held path, favorable first; funding debits before adverse; "
                "credits do not raise intratrade peak; exit-bar high/low excluded"
            ),
            "cagr": "full wall-clock split including idle cash",
        },
        "evaluation_config": asdict(cfg),
        "signal_config": support_result["config"],
        "source": {
            "signal": signal_source,
            "execution": market_source,
            "funding": funding_source,
        },
        "global_clock_counts": {
            policy: int(len(schedule)) for policy, schedule in schedules.items()
        },
        "windows": windows,
        "qualification": qualification,
        "selection": {
            "selected_alpha": "clasp24" if qualification["qualifies"] else None,
            "rejected": not qualification["qualifies"],
            "reason": (
                "passed every frozen pre-2024 gate"
                if qualification["qualifies"]
                else "CLASP-24 failed at least one frozen pre-2024 gate"
            ),
        },
    }


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
            "funding_settlement_count",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=EvaluationConfig.output)
    cfg = EvaluationConfig(output=parser.parse_args().output)
    result = run_evaluation(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "qualification": result["qualification"],
                "primary": {
                    name: _headline(policies["primary"])
                    for name, policies in result["windows"].items()
                },
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
