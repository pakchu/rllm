"""One-shot calendar-2023 return evaluator for frozen PDF-10.

The preregistration owns every signal and scheduling parameter.  This module
must itself be committed and hash-frozen before ``run_evaluation`` can load
the calendar-2023 execution prices.
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

from training.evaluate_metaorder_fragmentation_impact_curvature import (
    simulate_schedule,
)
from training.preregister_cross_collateral_liquidity_credibility_fracture import (
    Config as SignalConfig,
    _independence_summary,
    _quarterly_schedule,
    build_signal,
    load_credibility,
    support_summary,
)


PREREGISTRATION_COMMIT = "2737ab2620b1b1d6acc42bae8289ebf14ab8a50e"
SUPPORT_COMMIT = "11f40b8bac682a5e01ef450e81155c68372b225b"
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_credibility_fracture.py"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "8947050c990b5638f6d8b2e952f252289ddef6c92f85fb13f75001fe721e6e28"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cross-collateral-liquidity-credibility-fracture-"
    "preregistration-2026-07-14.md"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "e7bf6dc9b2c7bf1ec2d560ea4e1dff8018cb6c28177fa012b729d2e0a2ca1dfe"
)
PREREGISTRATION_RESULT = Path(
    "results/cross_collateral_liquidity_credibility_fracture_"
    "support_2026-07-14.json"
)
PREREGISTRATION_RESULT_SHA256 = (
    "9a3001db640ec8041d885645d33f11dd6075276685eb22f8ae3c618363d3099a"
)
EVENT_CLOCK_FREEZE = Path(
    "results/cross_collateral_liquidity_credibility_fracture_"
    "event_clock_2026-07-14.json"
)
EVENT_CLOCK_FREEZE_SHA256 = (
    "ab8209308619b97880277b95fcc1a2f825b050a603e24b3e2125ddd5bfb226f8"
)
EVENT_CLOCK_SHA256 = (
    "ce1c6ec42434874d97c6b6034f51a73771b27e314da6d37a4f44b0563e6972e2"
)
STANDARDIZER_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_void_refill.py"
)
STANDARDIZER_SOURCE_SHA256 = (
    "8465af153f4e5a19299c7ee2b6104e7ea009feb8da80ad10d50cf49bddd7ad51"
)
SCHEDULER_SOURCE = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)
SCHEDULER_SOURCE_SHA256 = (
    "51e99dbdc5ba13e6b4ac15e3915ec5b30e36dff89c1e5b31a5f3f7f272f01a59"
)
EXECUTION_SOURCE = Path(
    "training/evaluate_metaorder_fragmentation_impact_curvature.py"
)
EXECUTION_SOURCE_SHA256 = (
    "1589a52605386570485a7e6be3b8f3aa9439a498abb60eaa42272ac62d4cbed3"
)
TRADE_STATS_SOURCE = Path("training/strict_bar_backtest.py")
TRADE_STATS_SOURCE_SHA256 = (
    "3e95ad320d8869755afa1f4907d2d478200a3ebfc015e4eaeace0be0b15f9682"
)
CREDIBILITY_MANIFEST = Path(
    "results/binance_cross_collateral_book_credibility_btc_2023_manifest.json"
)
CREDIBILITY_MANIFEST_SHA256 = (
    "f530f472765c8cb56bf564efd346c734e2404e072b87e2ff8dc3b84e303c30f7"
)
MARKET_MANIFEST = Path(
    "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
)
MARKET_MANIFEST_SHA256 = (
    "c04fbbd299cc748a6745c0ef030787da4d560833c744c81c98dd8840efc7913e"
)
MARKET_DATA_SHA256 = (
    "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
)
EVALUATION_SOURCE = Path(
    "training/evaluate_cross_collateral_liquidity_credibility_fracture.py"
)
EVALUATION_FREEZE = Path(
    "results/cross_collateral_liquidity_credibility_fracture_"
    "evaluator_freeze_2026-07-14.json"
)

WINDOWS: dict[str, tuple[str, str]] = {
    "train2023_h1": ("2023-01-01", "2023-07-01"),
    "select2023_h2": ("2023-07-01", "2024-01-01"),
    "q1": ("2023-01-01", "2023-04-01"),
    "q2": ("2023-04-01", "2023-07-01"),
    "q3": ("2023-07-01", "2023-10-01"),
    "q4": ("2023-10-01", "2024-01-01"),
}
POLICY_NAMES = (
    "pdf10",
    "reverse",
    "always_long",
    "always_short",
    "permuted_sign",
    "price_momentum",
)
QUALIFICATION_CONTROLS = POLICY_NAMES[1:]


@dataclass(frozen=True)
class EvaluationConfig:
    output: str = (
        "results/cross_collateral_liquidity_credibility_fracture_"
        "selection_2026-07-14.json"
    )
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    cluster_permutations: int = 100_000
    cluster_seed: int = 20_260_714
    sign_permutation_seed: int = 20_260_714
    price_momentum_bars: int = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_evaluation_config(cfg: EvaluationConfig) -> None:
    expected = {
        "leverage": 0.5,
        "fee_rate": 0.0005,
        "slippage_rate": 0.0001,
        "cluster_permutations": 100_000,
        "cluster_seed": 20_260_714,
        "sign_permutation_seed": 20_260_714,
        "price_momentum_bars": 1,
    }
    changed = {
        name: {"expected": value, "observed": getattr(cfg, name)}
        for name, value in expected.items()
        if getattr(cfg, name) != value
    }
    if changed:
        raise ValueError(f"PDF-10 evaluation config is frozen: {changed}")


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.is_file():
        raise ValueError("PDF-10 evaluator freeze manifest is missing")
    freeze = json.loads(EVALUATION_FREEZE.read_text())
    if freeze.get("outcomes_opened_for_pdf10") is not False:
        raise ValueError("PDF-10 evaluator was not frozen before outcomes")
    if freeze.get("evaluation_source") != str(EVALUATION_SOURCE):
        raise ValueError("PDF-10 evaluator freeze path changed")
    if freeze.get("evaluation_source_sha256") != _sha256(EVALUATION_SOURCE):
        raise ValueError("PDF-10 evaluator differs from pre-outcome freeze")
    if not freeze.get("evaluation_source_commit"):
        raise ValueError("PDF-10 evaluator source commit is missing")
    if len(str(freeze["evaluation_source_commit"])) != 40:
        raise ValueError("PDF-10 evaluator source commit is not full length")
    if freeze.get("preregistration_commit") != PREREGISTRATION_COMMIT:
        raise ValueError("PDF-10 evaluator freeze preregistration changed")
    if freeze.get("support_commit") != SUPPORT_COMMIT:
        raise ValueError("PDF-10 evaluator freeze support commit changed")
    if freeze.get("preregistration_result_sha256") != (
        PREREGISTRATION_RESULT_SHA256
    ):
        raise ValueError("PDF-10 evaluator freeze support hash changed")
    if freeze.get("opened_windows") != []:
        raise ValueError("PDF-10 evaluator freeze already opened a window")
    expected_sealed = [*WINDOWS, "test2024", "eval2025", "ytd2026"]
    if freeze.get("sealed_windows") != expected_sealed:
        raise ValueError("PDF-10 evaluator freeze sealed windows changed")
    return freeze


def verify_preregistration() -> dict[str, Any]:
    for path, expected in (
        (PREREGISTRATION_SOURCE, PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (PREREGISTRATION_RESULT, PREREGISTRATION_RESULT_SHA256),
        (EVENT_CLOCK_FREEZE, EVENT_CLOCK_FREEZE_SHA256),
        (STANDARDIZER_SOURCE, STANDARDIZER_SOURCE_SHA256),
        (SCHEDULER_SOURCE, SCHEDULER_SOURCE_SHA256),
        (EXECUTION_SOURCE, EXECUTION_SOURCE_SHA256),
        (TRADE_STATS_SOURCE, TRADE_STATS_SOURCE_SHA256),
        (CREDIBILITY_MANIFEST, CREDIBILITY_MANIFEST_SHA256),
        (MARKET_MANIFEST, MARKET_MANIFEST_SHA256),
    ):
        if _sha256(path) != expected:
            raise ValueError(f"frozen PDF-10 dependency changed: {path}")

    result = json.loads(PREREGISTRATION_RESULT.read_text())
    protocol = result.get("protocol", {})
    calibration = result.get("support_calibration", {})
    if protocol.get("outcomes_opened_for_pdf10") is not False:
        raise ValueError("PDF-10 support artifact opened outcomes")
    if protocol.get("price_or_return_loaded") is not False:
        raise ValueError("PDF-10 support artifact loaded a return source")
    if protocol.get("support_rejected") is not False:
        raise ValueError("PDF-10 was support-rejected")
    if result.get("all_support_gates_pass") is not True:
        raise ValueError("PDF-10 support/independence gates are not passing")
    if result.get("config") != asdict(SignalConfig()):
        raise ValueError("PDF-10 config differs from frozen support")
    if calibration != {
        "outcomes_opened_for_pdf10": False,
        "parameters_searched": False,
        "all_parameters_fixed": True,
        "further_support_repairs_allowed": False,
    }:
        raise ValueError("PDF-10 support-calibration contract changed")
    if protocol.get("sealed_windows") != [
        "test2024",
        "eval2025",
        "ytd2026",
    ]:
        raise ValueError("PDF-10 sealed-window contract changed")
    event_clock = json.loads(EVENT_CLOCK_FREEZE.read_text())
    if event_clock.get("outcomes_opened_for_pdf10") is not False:
        raise ValueError("PDF-10 event clock opened outcomes")
    if event_clock.get("price_or_return_loaded") is not False:
        raise ValueError("PDF-10 event clock loaded returns")
    if event_clock.get("preregistration_result_sha256") != (
        PREREGISTRATION_RESULT_SHA256
    ):
        raise ValueError("PDF-10 event clock support hash changed")
    if event_clock.get("event_clock_sha256") != EVENT_CLOCK_SHA256:
        raise ValueError("PDF-10 event clock hash record changed")
    return result


def _event_clock_sha256(schedule: pd.DataFrame) -> str:
    columns = (
        "signal_position",
        "entry_position",
        "exit_position",
        "side",
        "branch",
        "hold_bars",
    )
    records = [
        {
            "signal_position": int(row.signal_position),
            "entry_position": int(row.entry_position),
            "exit_position": int(row.exit_position),
            "side": int(row.side),
            "branch": str(row.branch),
            "hold_bars": int(row.hold_bars),
        }
        for row in schedule[list(columns)].itertuples(index=False)
    ]
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_signal_replay(
    frame: pd.DataFrame,
    signal: pd.DataFrame,
    cfg: SignalConfig,
    preregistration: dict[str, Any],
) -> pd.DataFrame:
    schedule = _quarterly_schedule(signal, frame)
    if not (
        schedule["entry_position"].to_numpy(int)
        == schedule["signal_position"].to_numpy(int) + 1
    ).all():
        raise ValueError("PDF-10 replay is not next-open entry")
    if not schedule["hold_bars"].astype(int).eq(2).all():
        raise ValueError("PDF-10 replay hold differs from two bars")
    if not (
        schedule["exit_position"].to_numpy(int)
        == schedule["entry_position"].to_numpy(int) + 2
    ).all():
        raise ValueError("PDF-10 replay is not t+3 scheduled-open exit")
    event_clock_sha256 = _event_clock_sha256(schedule)
    if event_clock_sha256 != EVENT_CLOCK_SHA256:
        raise ValueError("PDF-10 canonical event clock differs from freeze")
    event_clock = json.loads(EVENT_CLOCK_FREEZE.read_text())
    if len(schedule) != event_clock.get("event_count"):
        raise ValueError("PDF-10 event-clock count differs from freeze")
    support = support_summary(signal, frame, cfg, schedule=schedule)
    if support != preregistration.get("support"):
        raise ValueError("PDF-10 support replay differs from frozen artifact")
    if int(signal["candidate"].sum()) != preregistration.get(
        "raw_candidate_count"
    ):
        raise ValueError("PDF-10 candidate replay differs from frozen artifact")
    feature = preregistration.get("feature", {})
    replay_feature = {
        "raw_bullish_rows": int(signal["raw_state"].gt(0).sum()),
        "raw_bearish_rows": int(signal["raw_state"].lt(0).sum()),
        "confirmed_bullish_rows": int(signal["confirmed_state"].gt(0).sum()),
        "confirmed_bearish_rows": int(signal["confirmed_state"].lt(0).sum()),
    }
    for name, observed in replay_feature.items():
        if observed != feature.get(name):
            raise ValueError(f"PDF-10 feature replay differs: {name}")
    side_counts = {
        "long": int(schedule["side"].gt(0).sum()),
        "short": int(schedule["side"].lt(0).sum()),
    }
    if side_counts != preregistration.get("scheduled_side_counts"):
        raise ValueError("PDF-10 side-count replay differs")
    branch_counts = {
        name: int(count)
        for name, count in schedule["branch"].value_counts().items()
    }
    if branch_counts != preregistration.get("scheduled_branch_counts"):
        raise ValueError("PDF-10 branch-count replay differs")
    independence = _independence_summary(
        signal,
        frame,
        cfg,
        schedule=schedule,
    )
    if independence != preregistration.get("independence"):
        raise ValueError("PDF-10 independence replay differs")
    return schedule


def _load_execution_frame(signal_cfg: SignalConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    credibility, source = load_credibility(signal_cfg)
    manifest = json.loads(MARKET_MANIFEST.read_text())
    if manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("market manifest was not built outcome-blind")
    path = Path(manifest.get("combined_output", ""))
    if manifest.get("combined_sha256") != MARKET_DATA_SHA256:
        raise ValueError("market manifest contains a non-frozen data hash")
    if not path.is_file() or _sha256(path) != MARKET_DATA_SHA256:
        raise ValueError("execution market hash mismatch")
    market = pd.read_csv(path, compression="gzip", parse_dates=["date"])
    market = market.loc[
        market["date"].ge("2023-01-01") & market["date"].lt("2024-01-01")
    ].reset_index(drop=True)
    clvr_columns = ["date", "open", "high", "low", "close"]
    if not set(clvr_columns).issubset(market.columns):
        raise ValueError("execution market columns are incomplete")
    market = market[clvr_columns]
    expected_dates = pd.date_range(
        "2023-01-01", "2024-01-01", freq="5min", inclusive="left"
    )
    if not pd.DatetimeIndex(market["date"]).equals(expected_dates):
        raise ValueError("execution market is not a complete 2023 grid")
    prices = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ValueError("execution OHLC contains an invalid price")
    if np.any(market["high"] < market[["open", "low", "close"]].max(axis=1)):
        raise ValueError("execution high does not envelope the bar")
    if np.any(market["low"] > market[["open", "high", "close"]].min(axis=1)):
        raise ValueError("execution low does not envelope the bar")
    frame = credibility.merge(market, on="date", validate="one_to_one")
    source.update(
        {
            "market_manifest_sha256": _sha256(MARKET_MANIFEST),
            "market_data_sha256": _sha256(path),
            "execution_price_rows": int(len(market)),
        }
    )
    return frame, source


def policy_schedule(
    reserved_schedule: pd.DataFrame,
    market: pd.DataFrame,
    policy: str,
    *,
    permutation_seed: int,
    price_momentum_bars: int,
) -> pd.DataFrame:
    """Assign actions once on the frozen annual candidate clock."""
    if policy not in POLICY_NAMES:
        raise ValueError(f"unknown PDF-10 control policy: {policy}")
    output = reserved_schedule.copy()
    if output.empty:
        return output
    if not output["branch"].isin(
        [
            "bullish_display_firmness_divergence",
            "bearish_display_firmness_divergence",
        ]
    ).all():
        raise ValueError("PDF-10 schedule contains an unknown branch")
    sides = output["side"].to_numpy(np.int8)
    if not np.isin(sides, [-1, 1]).all():
        raise ValueError("PDF-10 schedule contains an invalid side")

    if policy == "reverse":
        output["side"] = -sides
    elif policy == "always_long":
        output["side"] = np.ones(len(output), dtype=np.int8)
    elif policy == "always_short":
        output["side"] = -np.ones(len(output), dtype=np.int8)
    elif policy == "permuted_sign":
        rng = np.random.default_rng(permutation_seed)
        output["side"] = rng.permutation(sides)
    elif policy == "price_momentum":
        if price_momentum_bars != 1:
            raise ValueError("PDF-10 freezes one-bar close momentum")
        positions = output["signal_position"].to_numpy(int)
        if np.any(positions < price_momentum_bars):
            raise ValueError("price momentum lacks causal lookback")
        closes = market["close"].to_numpy(float)
        momentum = closes[positions] / closes[positions - 1] - 1.0
        output["side"] = np.where(momentum >= 0.0, 1, -1).astype(np.int8)
    return output.reset_index(drop=True)


def _schedule_for_window(
    annual_schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp:
        raise ValueError("window start must be before end")
    signal_dates = pd.to_datetime(annual_schedule["signal_date"])
    output = annual_schedule.loc[
        signal_dates.ge(start_timestamp) & signal_dates.lt(end_timestamp)
    ].copy()
    for column in ("entry_date", "exit_date"):
        dates = pd.to_datetime(output[column])
        if not (dates.ge(start_timestamp) & dates.lt(end_timestamp)).all():
            raise ValueError("frozen PDF-10 trade crosses an evaluation window")
    return output.reset_index(drop=True)


def evaluate_policy(
    market: pd.DataFrame,
    annual_policy_schedule: pd.DataFrame,
    *,
    start: str,
    end: str,
    cfg: EvaluationConfig,
) -> dict[str, Any]:
    schedule = _schedule_for_window(
        annual_policy_schedule,
        start=start,
        end=end,
    )
    metrics = simulate_schedule(
        market,
        schedule,
        start=start,
        end=end,
        cfg=cfg,
    )
    metrics.pop("continuation_count", None)
    metrics.pop("fade_count", None)
    metrics["reserved_candidate_count"] = int(len(schedule))
    metrics["executed_candidate_count"] = int(len(schedule))
    return metrics


def _qualification(windows: dict[str, Any]) -> dict[str, Any]:
    train = windows["train2023_h1"]["pdf10"]
    select = windows["select2023_h2"]["pdf10"]
    failures: list[str] = []
    for name, metrics in (
        ("train2023_h1", train),
        ("select2023_h2", select),
    ):
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["cagr_to_strict_mdd"] < 3.0:
            failures.append(f"{name}: CAGR/strict-MDD below 3")
        if metrics["strict_mdd_pct"] > 15.0:
            failures.append(f"{name}: strict MDD above 15%")
        if metrics["trade_count"] < 180:
            failures.append(f"{name}: fewer than 180 trades")
        if metrics["weekly_cluster_sign_flip"]["p_value_one_sided"] >= 0.10:
            failures.append(f"{name}: weekly-cluster p-value not below 0.10")

    for name in ("q1", "q2", "q3", "q4"):
        metrics = windows[name]["pdf10"]
        if metrics["absolute_return_pct"] <= 0.0:
            failures.append(f"{name}: non-positive absolute return")
        if metrics["trade_count"] < 75:
            failures.append(f"{name}: fewer than 75 trades")

    pdf_min_ratio = min(
        train["cagr_to_strict_mdd"],
        select["cagr_to_strict_mdd"],
    )
    for control in QUALIFICATION_CONTROLS:
        control_min_ratio = min(
            windows["train2023_h1"][control]["cagr_to_strict_mdd"],
            windows["select2023_h2"][control]["cagr_to_strict_mdd"],
        )
        if pdf_min_ratio <= control_min_ratio:
            failures.append(
                "pdf10: minimum train/select ratio does not beat " + control
            )
    return {
        "qualifies": not failures,
        "failures": failures,
        "pdf10_min_train_select_ratio": float(pdf_min_ratio),
    }


def run_evaluation(cfg: EvaluationConfig) -> dict[str, Any]:
    _validate_evaluation_config(cfg)
    evaluation_freeze = verify_evaluation_freeze()
    preregistration = verify_preregistration()
    signal_cfg = SignalConfig()

    # Replay all signal/support facts before opening execution prices.
    credibility, _ = load_credibility(signal_cfg)
    signal = build_signal(credibility, signal_cfg)
    annual_schedule = verify_signal_replay(
        credibility,
        signal,
        signal_cfg,
        preregistration,
    )

    market, source = _load_execution_frame(signal_cfg)
    annual_policies = {
        policy: policy_schedule(
            annual_schedule,
            market,
            policy,
            permutation_seed=cfg.sign_permutation_seed,
            price_momentum_bars=cfg.price_momentum_bars,
        )
        for policy in POLICY_NAMES
    }
    windows = {
        name: {
            policy: evaluate_policy(
                market,
                schedule,
                start=start,
                end=end,
                cfg=cfg,
            )
            for policy, schedule in annual_policies.items()
        }
        for name, (start, end) in WINDOWS.items()
    }
    qualification = _qualification(windows)
    return {
        "protocol": {
            "name": "PDF-10 frozen calendar-2023 research evaluation",
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "support_commit": SUPPORT_COMMIT,
            "preregistration_source_sha256": PREREGISTRATION_SOURCE_SHA256,
            "preregistration_document_sha256": PREREGISTRATION_DOCUMENT_SHA256,
            "preregistration_result_sha256": PREREGISTRATION_RESULT_SHA256,
            "event_clock_freeze_sha256": EVENT_CLOCK_FREEZE_SHA256,
            "event_clock_sha256": EVENT_CLOCK_SHA256,
            "evaluation_source_sha256": _sha256(EVALUATION_SOURCE),
            "evaluation_freeze_manifest_sha256": _sha256(EVALUATION_FREEZE),
            "evaluation_source_commit": evaluation_freeze[
                "evaluation_source_commit"
            ],
            "outcomes_opened_for_pdf10": True,
            "opened_windows": list(WINDOWS),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
            "signal_parameters_mutable": False,
            "annual_actions_assigned_before_window_slicing": True,
            "candidate_clock_reserved_before_action_control": True,
            "entry": "next 5m open after the second same-side evidence bar",
            "exit": "scheduled open after two completed 5m held bars",
            "strict_mdd": (
                "held path, favorable extreme first then adverse extreme; "
                "exit-bar later high/low excluded"
            ),
            "cagr": "full wall-clock split including idle cash",
        },
        "evaluation_config": asdict(cfg),
        "signal_config": preregistration["config"],
        "source": source,
        "windows": windows,
        "qualification": qualification,
        "selection": {
            "selected_alpha": "pdf10" if qualification["qualifies"] else None,
            "rejected": not qualification["qualifies"],
            "reason": (
                "passed every frozen calendar-2023 development gate"
                if qualification["qualifies"]
                else "PDF-10 failed at least one frozen calendar-2023 gate"
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
            "trade_count",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=EvaluationConfig.output)
    args = parser.parse_args()
    cfg = EvaluationConfig(output=args.output)
    result = run_evaluation(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "qualification": result["qualification"],
                "pdf10": {
                    name: _headline(policies["pdf10"])
                    for name, policies in result["windows"].items()
                },
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
