"""Frozen pre-2024 return evaluator for CATCH-12.

The evaluator must be committed and hash-recorded before it can load USD-M
execution prices. It reserves the primary and every control clock from causal
features first, then opens only 2020-2023 outcomes under the preregistered gate.
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

from training import preregister_cash_auction_transfer_catchup_handoff as catch
from training.evaluate_metaorder_fragmentation_impact_curvature import (
    simulate_schedule,
)


PREREGISTRATION_COMMIT = "5d1270bf0d3c453ecfc3a5e02a193db8db59f1e5"
SUPPORT_COMMIT = "49bf3f7b17c39b9c2abc56ef64e1a6fa0bad4279"
CLOCK_COMMIT = "2172ae04a6989702d3c095b9982226628232e445"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cash_auction_transfer_catchup_handoff.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "5af869f788466aa263f3c00d0d04b0042c759aacd5e91385dd74eb87e951a56e"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cash-auction-transfer-catchup-handoff-preregistration-2026-07-14.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "b7e62ce7d4f8ece9dfcb1434a4271587ce6fae358fc49594bd617a77c877579b"
)
SUPPORT_DOCUMENT = Path(
    "docs/cash-auction-transfer-catchup-handoff-support-freeze-2026-07-14.md"
)
SUPPORT_DOCUMENT_SHA256 = (
    "b6858bdecfbf179389e3246b1d37441424cb0642c53b9a76ca26e6f355f68a6e"
)
EVALUATOR_DOCUMENT = Path(
    "docs/cash-auction-transfer-catchup-handoff-evaluator-freeze-2026-07-14.md"
)
EVALUATOR_DOCUMENT_SHA256 = (
    "5be7d6061a8184cd7b3fe339cbe870bf7c7d30af9a6c700511b44eeb10559aa1"
)
SUPPORT_RESULT = Path(
    "results/cash_auction_transfer_catchup_handoff_support_2026-07-14.json"
)
SUPPORT_RESULT_SHA256 = (
    "454c54cb234a34b51fca12a810332039d7b21e4395f47f4e6b8ad6375370be02"
)
EVENT_CLOCK = Path("results/cash_auction_transfer_catchup_handoff_clock_2026-07-14.csv")
EVENT_CLOCK_SHA256 = "066bf8e08267a043cc191eb436f0aa33105ab948de9f9f1edfde4d9c30de46d1"
EVENT_CLOCK_MANIFEST = Path(
    "results/cash_auction_transfer_catchup_handoff_clock_manifest_2026-07-14.json"
)
EVENT_CLOCK_MANIFEST_SHA256 = (
    "f461529e14539ea4aa6e4b498ef8738f918dfb095ebbfcefb43ed127fe272ca6"
)
CLOCK_FREEZE_SOURCE = Path(
    "training/freeze_cash_auction_transfer_catchup_handoff_clock.py"
)
CLOCK_FREEZE_SOURCE_SHA256 = (
    "91bd6734d357d1546c1b270881757ee17ee9cd8aef3a554aec288bedb3d539f5"
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
EXECUTION_SOURCE = Path("training/evaluate_metaorder_fragmentation_impact_curvature.py")
EXECUTION_SOURCE_SHA256 = (
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
EVALUATION_SOURCE = Path("training/evaluate_cash_auction_transfer_catchup_handoff.py")
EVALUATION_FREEZE = Path(
    "results/cash_auction_transfer_catchup_handoff_evaluator_freeze_2026-07-14.json"
)

SELECTED_QUANTILE = 0.975
WINDOWS: dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2023-01-01"),
    "select2023": ("2023-01-01", "2024-01-01"),
    "select2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
}
POLICY_NAMES = tuple(catch.CONTROL_SIDE_RULES)
SCORE_BEARING_CONTROLS = (
    "venue_swap",
    "reverse_time",
    "simultaneous_only",
    "aggregate_only",
    "basis_only",
    "asymmetry_only",
    "no_basis_lag",
    "no_activity_order",
    "stale_1h",
    "stale_24h",
)
FALSIFICATION_CONTROLS = ("direction_flip", "signal_delay_1bar")
CLOCK_COLUMNS = catch.SCHEDULE_COLUMNS


@dataclass(frozen=True)
class EvaluationConfig:
    output: str = (
        "results/cash_auction_transfer_catchup_handoff_selection_2026-07-14.json"
    )
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_714
    minimum_mean_gross_underlying_bp: float = 12.0
    minimum_2023_half_trades: int = 200


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
        "minimum_2023_half_trades": 200,
    }
    changed = {
        name: {"expected": value, "observed": getattr(cfg, name)}
        for name, value in expected.items()
        if getattr(cfg, name) != value
    }
    if changed:
        raise ValueError(f"CATCH evaluation config is frozen: {changed}")


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.is_file():
        raise ValueError("CATCH evaluator freeze manifest is missing")
    freeze = _read_json(EVALUATION_FREEZE)
    if freeze.get("outcomes_opened_for_catch12") is not False:
        raise ValueError("CATCH evaluator was not frozen before outcomes")
    if freeze.get("evaluation_source") != str(EVALUATION_SOURCE):
        raise ValueError("CATCH evaluator freeze path changed")
    if freeze.get("evaluation_source_sha256") != _sha256(EVALUATION_SOURCE):
        raise ValueError("CATCH evaluator differs from pre-outcome freeze")
    if len(str(freeze.get("evaluation_source_commit", ""))) != 40:
        raise ValueError("CATCH evaluator source commit is not full length")
    if freeze.get("preregistration_commit") != PREREGISTRATION_COMMIT:
        raise ValueError("CATCH evaluator freeze preregistration changed")
    if freeze.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("CATCH evaluator freeze support commit changed")
    if freeze.get("clock_commit") != CLOCK_COMMIT:
        raise ValueError("CATCH evaluator freeze clock commit changed")
    if freeze.get("support_result_sha256") != SUPPORT_RESULT_SHA256:
        raise ValueError("CATCH evaluator freeze support hash changed")
    if freeze.get("event_clock_sha256") != EVENT_CLOCK_SHA256:
        raise ValueError("CATCH evaluator freeze event clock changed")
    if freeze.get("market_data_sha256") != MARKET_DATA_SHA256:
        raise ValueError("CATCH evaluator freeze market hash changed")
    if freeze.get("opened_windows") != []:
        raise ValueError("CATCH evaluator freeze already opened a window")
    if freeze.get("returns_or_prices_loaded_during_freeze") is not False:
        raise ValueError("CATCH evaluator freeze loaded returns or prices")
    if freeze.get("mutable_parameters") != []:
        raise ValueError("CATCH evaluator freeze permits mutable parameters")
    expected_sealed = [*WINDOWS, "test2024", "eval2025", "ytd2026"]
    if freeze.get("sealed_windows") != expected_sealed:
        raise ValueError("CATCH evaluator freeze sealed windows changed")
    return freeze


def verify_preregistration() -> tuple[dict[str, Any], dict[str, Any]]:
    frozen_files = (
        (PREREGISTRATION_SOURCE, PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (SUPPORT_DOCUMENT, SUPPORT_DOCUMENT_SHA256),
        (EVALUATOR_DOCUMENT, EVALUATOR_DOCUMENT_SHA256),
        (SUPPORT_RESULT, SUPPORT_RESULT_SHA256),
        (EVENT_CLOCK, EVENT_CLOCK_SHA256),
        (EVENT_CLOCK_MANIFEST, EVENT_CLOCK_MANIFEST_SHA256),
        (CLOCK_FREEZE_SOURCE, CLOCK_FREEZE_SOURCE_SHA256),
        (FEATURE_BUILDER_SOURCE, FEATURE_BUILDER_SOURCE_SHA256),
        (FEATURE_MANIFEST, FEATURE_MANIFEST_SHA256),
        (SOURCE_AUDIT, SOURCE_AUDIT_SHA256),
        (EXECUTION_SOURCE, EXECUTION_SOURCE_SHA256),
        (TRADE_STATS_SOURCE, TRADE_STATS_SOURCE_SHA256),
        (MARKET_MANIFEST, MARKET_MANIFEST_SHA256),
    )
    for path, expected in frozen_files:
        if _sha256(path) != expected:
            raise ValueError(f"frozen CATCH dependency changed: {path}")

    result = _read_json(SUPPORT_RESULT)
    protocol = result.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("CATCH support artifact opened outcomes")
    if protocol.get("support_only") is not True:
        raise ValueError("CATCH support artifact is not support-only")
    if protocol.get("control_side_rules") != catch.CONTROL_SIDE_RULES:
        raise ValueError("CATCH control side rules changed")
    if result.get("support_decision") != "pass":
        raise ValueError("CATCH support gate did not pass")
    if result.get("selected_quantile") != SELECTED_QUANTILE:
        raise ValueError("CATCH selected quantile changed")
    if result.get("config") != asdict(catch.Config()):
        raise ValueError("CATCH signal config differs from frozen support")
    selected = result.get("selected_support", {})
    if selected.get("passes_support") is not True:
        raise ValueError("CATCH selected support row is not passing")
    if selected.get("support", {}).get("nonoverlap_total") != 3_957:
        raise ValueError("CATCH selected event count changed")

    clock_manifest = _read_json(EVENT_CLOCK_MANIFEST)
    frozen_protocol = clock_manifest.get("protocol", {})
    clock = clock_manifest.get("clock", {})
    if frozen_protocol.get("outcomes_opened") is not False:
        raise ValueError("CATCH event clock opened outcomes")
    if frozen_protocol.get("preregistration_commit") != PREREGISTRATION_COMMIT:
        raise ValueError("CATCH event clock preregistration changed")
    if frozen_protocol.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("CATCH event clock support commit changed")
    if frozen_protocol.get("selected_quantile") != SELECTED_QUANTILE:
        raise ValueError("CATCH event clock quantile changed")
    if clock.get("sha256") != EVENT_CLOCK_SHA256 or clock.get("rows") != 3_957:
        raise ValueError("CATCH event clock record changed")
    if clock_manifest.get("support", {}).get("sha256") != SUPPORT_RESULT_SHA256:
        raise ValueError("CATCH event clock support hash changed")

    market_manifest = _read_json(MARKET_MANIFEST)
    market_protocol = market_manifest.get("protocol", {})
    if market_protocol.get("archive_checksums_verified") is not True:
        raise ValueError("CATCH execution archives were not checksum-verified")
    if market_protocol.get("end_is_exclusive") is not True:
        raise ValueError("CATCH execution manifest end contract changed")
    if market_manifest.get("combined_output") != str(MARKET_DATA):
        raise ValueError("CATCH execution market path changed")
    if market_manifest.get("combined_sha256") != MARKET_DATA_SHA256:
        raise ValueError("CATCH execution market hash changed")
    if market_manifest.get("rows") != 420_768:
        raise ValueError("CATCH execution market row count changed")
    return result, market_manifest


def _canonical_clock_records(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    missing = set(CLOCK_COLUMNS).difference(schedule.columns)
    if missing:
        raise ValueError(f"CATCH clock is missing columns: {sorted(missing)}")
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
    cfg: catch.Config,
    support_result: dict[str, Any],
    source: dict[str, Any],
) -> tuple[
    dict[str, pd.Series],
    dict[str, pd.Series],
    pd.DataFrame,
]:
    signal, controls, control_sides, _ = catch.classify_events(
        frame, cfg, quantile=SELECTED_QUANTILE
    )
    replayed = catch.nonoverlapping_schedule(signal, frame)
    frozen = pd.read_csv(EVENT_CLOCK)
    if _canonical_clock_records(replayed) != _canonical_clock_records(frozen):
        raise ValueError("CATCH primary event-clock replay differs from freeze")
    if not replayed["entry_position"].eq(replayed["signal_position"] + 1).all():
        raise ValueError("CATCH replay is not next-open entry")
    if not replayed["hold_bars"].eq(cfg.hold_bars).all():
        raise ValueError("CATCH replay hold changed")
    if (
        not replayed["exit_position"]
        .eq(replayed["entry_position"] + cfg.hold_bars)
        .all()
    ):
        raise ValueError("CATCH replay exit changed")

    selected = support_result["selected_support"]
    if catch._support(replayed, cfg) != selected["support"]:
        raise ValueError("CATCH selected support replay differs from freeze")
    raw_counts = {
        name: int(mask.sum()) for name, mask in controls.items() if name != "primary"
    }
    if raw_counts != selected["control_raw_counts"]:
        raise ValueError("CATCH control raw clocks differ from freeze")
    if int(controls["primary"].sum()) != selected["raw_primary"]:
        raise ValueError("CATCH raw primary clock differs from freeze")
    clock_manifest = _read_json(EVENT_CLOCK_MANIFEST)
    if source != clock_manifest.get("source"):
        raise ValueError("CATCH source replay differs from clock freeze")
    return controls, control_sides, replayed.reset_index(drop=True)


def _schedule_from_control(
    frame: pd.DataFrame,
    cfg: catch.Config,
    *,
    mask: pd.Series,
    side: pd.Series,
    branch: str,
) -> pd.DataFrame:
    active = mask.fillna(False).astype(bool)
    numeric_side = pd.to_numeric(side, errors="coerce").fillna(0.0)
    if not numeric_side.loc[active].isin((-1.0, 1.0)).all():
        raise ValueError(f"CATCH control {branch} has an invalid active side")
    signal = pd.DataFrame(
        {
            "side": np.where(active, numeric_side, 0).astype(np.int8),
            "branch": np.where(active, branch, "none"),
            "hold_bars": np.where(active, cfg.hold_bars, 0).astype(np.int16),
        }
    )
    return (
        catch.nonoverlapping_schedule(signal, frame)
        .reindex(columns=CLOCK_COLUMNS)
        .reset_index(drop=True)
    )


def build_control_schedules(
    frame: pd.DataFrame,
    controls: dict[str, pd.Series],
    control_sides: dict[str, pd.Series],
    primary_schedule: pd.DataFrame,
    cfg: catch.Config,
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
        raise ValueError("CATCH evaluator control set differs from preregistration")

    if support_result is not None:
        expected = support_result["selected_support"]["control_scheduled_counts"]
        observed = {
            name: int(len(schedules[name]))
            for name in POLICY_NAMES
            if name not in {"primary", "direction_flip"}
        }
        if observed != expected:
            raise ValueError("CATCH control scheduled clocks differ from freeze")
    return schedules


def slice_schedule(
    schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp:
        raise ValueError("CATCH split start must precede end")
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
    signal_frame: pd.DataFrame,
    market_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if _sha256(MARKET_DATA) != MARKET_DATA_SHA256:
        raise ValueError("CATCH execution market differs from frozen SHA-256")
    market = pd.read_csv(
        MARKET_DATA,
        compression="gzip",
        usecols=["date", "open", "high", "low", "close"],
        parse_dates=["date"],
    )
    if len(market) != market_manifest["rows"]:
        raise ValueError("CATCH execution market row count differs from manifest")
    if market["date"].duplicated().any() or not market["date"].is_monotonic_increasing:
        raise ValueError("CATCH execution timestamps are duplicate or unordered")
    if not market["date"].equals(signal_frame["date"]):
        raise ValueError("CATCH execution market does not align with signal clock")
    prices = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("CATCH execution market contains invalid prices")
    opens = market["open"].to_numpy(float)
    highs = market["high"].to_numpy(float)
    lows = market["low"].to_numpy(float)
    closes = market["close"].to_numpy(float)
    if (
        (highs < np.maximum(opens, closes)).any()
        or (lows > np.minimum(opens, closes)).any()
        or (highs < lows).any()
    ):
        raise ValueError("CATCH execution market violates OHLC invariants")
    frame = signal_frame.assign(
        open=opens,
        high=highs,
        low=lows,
        close=closes,
    )
    source = {
        "market_manifest_sha256": MARKET_MANIFEST_SHA256,
        "market_data_sha256": MARKET_DATA_SHA256,
        "market_rows": int(len(market)),
        "first_date": str(market["date"].iloc[0]),
        "last_date": str(market["date"].iloc[-1]),
    }
    return frame, source


def mean_gross_underlying_move_bp(
    mean_net_account_return_pct: float,
    cfg: EvaluationConfig,
) -> float:
    per_side_cost = (cfg.fee_rate + cfg.slippage_rate) * cfg.leverage
    mean_net = float(mean_net_account_return_pct) / 100.0
    gross = ((1.0 + mean_net) / (1.0 - per_side_cost) ** 2 - 1.0) / (cfg.leverage)
    return float(gross * 10_000.0)


def evaluate_policy(
    frame: pd.DataFrame,
    global_schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    schedule = slice_schedule(global_schedule, start=start, end=end)
    metrics = simulate_schedule(frame, schedule, start=start, end=end, cfg=cfg)
    metrics.pop("continuation_count", None)
    metrics.pop("fade_count", None)
    mean_net = metrics["trade_statistics"]["mean_trade_ret_pct"]
    metrics["mean_gross_underlying_move_bp"] = (
        mean_gross_underlying_move_bp(mean_net, cfg) if metrics["trade_count"] else 0.0
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
            failures.append(f"{name}: fewer than 200 trades")

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
    support_result, market_manifest = verify_preregistration()
    signal_cfg = catch.Config()
    signal_frame, signal_source = catch.load_causal_frame(signal_cfg)
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

    windows: dict[str, Any] = {}
    for window_name, (start, end) in WINDOWS.items():
        windows[window_name] = {
            policy: evaluate_policy(
                frame,
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
            "name": "CATCH-12 frozen pre-2024 selection evaluation",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "support_commit": SUPPORT_COMMIT,
            "clock_commit": CLOCK_COMMIT,
            "evaluation_source_commit": evaluator_freeze["evaluation_source_commit"],
            "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
            "evaluation_freeze_manifest_sha256": _sha256(EVALUATION_FREEZE),
            "outcomes_opened_for_catch12": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
            "signal_parameters_mutable": False,
            "primary_clock_replayed_from_freeze": True,
            "control_clocks_reserved_before_outcomes": True,
            "control_clocks_reserved_before_return_slicing": True,
            "control_side_rules": catch.CONTROL_SIDE_RULES,
            "score_bearing_controls": list(SCORE_BEARING_CONTROLS),
            "falsification_controls": list(FALSIFICATION_CONTROLS),
            "entry": "next Binance USD-M 5m open after completed signal bar",
            "exit": "scheduled USD-M open 12 completed 5m bars after entry",
            "cost_multiplier": "(1-0.0003)*(1+0.5*r)*(1-0.0003)",
            "strict_mdd": (
                "held path, favorable extreme first then adverse; exit-bar "
                "high/low excluded"
            ),
            "cagr": "full wall-clock split including idle cash",
        },
        "evaluation_config": asdict(cfg),
        "signal_config": support_result["config"],
        "source": {"signal": signal_source, "execution": market_source},
        "global_clock_counts": {
            policy: int(len(schedule)) for policy, schedule in schedules.items()
        },
        "windows": windows,
        "qualification": qualification,
        "selection": {
            "selected_alpha": "catch12" if qualification["qualifies"] else None,
            "rejected": not qualification["qualifies"],
            "reason": (
                "passed every frozen pre-2024 gate"
                if qualification["qualifies"]
                else "CATCH-12 failed at least one frozen pre-2024 gate"
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
