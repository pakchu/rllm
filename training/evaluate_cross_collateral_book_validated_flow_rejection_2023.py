"""One-shot strict calendar-2023 outcome evaluator for frozen CBFR-72.

The evaluator and its freeze manifest must be committed while CBFR outcomes
remain sealed.  A failed calendar-2023 gate retires this exact policy and keeps
calendar 2024 onward unopened.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import evaluate_crrc_2023 as strict
from training import preregister_cross_collateral_book_validated_flow_rejection as cbfr
from training.evaluate_metaorder_fragmentation_impact_curvature import (
    weekly_cluster_sign_flip,
)


START = pd.Timestamp("2023-01-01 00:00:00")
END = pd.Timestamp("2024-01-01 00:00:00")
WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "2023": (START, END),
    "h1": (START, pd.Timestamp("2023-07-01")),
    "h2": (pd.Timestamp("2023-07-01"), END),
    "q1": (START, pd.Timestamp("2023-04-01")),
    "q2": (pd.Timestamp("2023-04-01"), pd.Timestamp("2023-07-01")),
    "q3": (pd.Timestamp("2023-07-01"), pd.Timestamp("2023-10-01")),
    "q4": (pd.Timestamp("2023-10-01"), END),
}

EVALUATION_SOURCE = Path(
    "training/evaluate_cross_collateral_book_validated_flow_rejection_2023.py"
)
TEST_PATH = Path(
    "tests/test_evaluate_cross_collateral_book_validated_flow_rejection_2023.py"
)
FREEZE_SOURCE = Path(
    "training/freeze_cross_collateral_book_validated_flow_rejection_2023_evaluator.py"
)
FREEZE_TEST_PATH = Path(
    "tests/test_freeze_cross_collateral_book_validated_flow_rejection_2023_evaluator.py"
)
EVALUATION_FREEZE = Path(
    "results/cross_collateral_book_validated_flow_rejection_"
    "evaluator_freeze_2026-07-18.json"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_cross_collateral_book_validated_flow_rejection.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/cross-collateral-book-validated-flow-rejection-"
    "preregistration-2026-07-18.md"
)
SUPPORT = Path(
    "results/cross_collateral_book_validated_flow_rejection_"
    "support_2026-07-18.json"
)
PRIMARY_CLOCK = Path(
    "results/cross_collateral_book_validated_flow_rejection_"
    "event_clock_2026-07-18.json"
)
MARKET_MANIFEST = cbfr.MARKET_MANIFEST
MARKET_DATA = cbfr.MARKET_DATA
FUNDING_MANIFEST = Path("results/binance_um_aux_btc_2021_2023_manifest.json")
FUNDING_DATA = Path(
    "data/binance_um_aux_btc_2021_2023/"
    "BTCUSDT_funding_2021-01-01_2023-12-31.csv.gz"
)
DEFAULT_OUTPUT = Path(
    "results/cross_collateral_book_validated_flow_rejection_"
    "selection_2023_2026-07-18.json"
)
DEFAULT_DOCS = Path(
    "docs/cross-collateral-book-validated-flow-rejection-"
    "selection-2023-2026-07-18.md"
)

PREREGISTRATION_SOURCE_SHA256 = (
    "004fa71b1951eff58eca592863cf7ad09e0e36e4749a3e611ce299e1ac3d601f"
)
PREREGISTRATION_DOCUMENT_SHA256 = (
    "d926afd4e7fd6ad8da1e2dfb54344aa664d5357c30a8511b55d05f3a895d1561"
)
SUPPORT_SHA256 = "048a8723494a91b082bdd07d466e1741a13a974c3c3c25c8ec81e081f27cc444"
PRIMARY_CLOCK_SHA256 = (
    "79b4838ae634efcff705e028a0ddff8b75d28d79180e3ac89f54b9cab7e5005f"
)
PRIMARY_EVENT_CLOCK_HASH = (
    "d2cdcad8f57867722c220e32029d0ccbf1f1aa511e5ae590cf43411a588af4bd"
)
MARKET_MANIFEST_SHA256 = cbfr.MARKET_MANIFEST_SHA256
MARKET_DATA_SHA256 = cbfr.MARKET_DATA_SHA256
FUNDING_MANIFEST_SHA256 = (
    "80c77f461be54b77c7554837a304a187321a052dd05cb39b4e0a3c80de5d2bdc"
)
FUNDING_DATA_SHA256 = (
    "654c668e3aea344d5906465cbbd090f2e4ff0c47e9d4bd8cf3856c24549cfc97"
)
STRICT_LEDGER_SOURCE = Path("training/evaluate_crrc_2023.py")
STRICT_LEDGER_SOURCE_SHA256 = (
    "89eb3396263689fd0a8332ffa5f1e59a88f9928a6aa8bf29032c0479b0f6afac"
)


@dataclass(frozen=True)
class EvaluationConfig:
    leverage: float = 0.5
    base_cost_bp_per_notional_side: float = 6.0
    stress_cost_bp_per_notional_side: float = 10.0
    delay_control_minutes: int = 5
    weekly_signflip_permutations: int = 100_000
    weekly_signflip_seed: int = 20_260_718


CONFIG = EvaluationConfig()


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clock_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(events)
    required = {
        "quarter",
        "signal_position",
        "entry_position",
        "exit_position",
        "signal_date",
        "entry_date",
        "exit_date",
        "side",
        "branch",
        "hold_bars",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"CBFR event clock misses columns: {sorted(missing)}")
    for column in ("signal_date", "entry_date", "exit_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    for column in (
        "signal_position",
        "entry_position",
        "exit_position",
        "side",
        "hold_bars",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    return frame.sort_values("entry_position").reset_index(drop=True)


def validate_clock(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) != 144:
        raise RuntimeError("CBFR event count changed")
    if not frame["side"].isin([-1, 1]).all():
        raise RuntimeError("CBFR event clock contains invalid side")
    if not frame["hold_bars"].eq(72).all():
        raise RuntimeError("CBFR hold changed")
    if not frame["entry_position"].eq(frame["signal_position"] + 1).all():
        raise RuntimeError("CBFR next-open entry changed")
    if not frame["exit_position"].eq(frame["entry_position"] + 72).all():
        raise RuntimeError("CBFR six-hour exit changed")
    if not (frame["signal_date"] < frame["entry_date"]).all():
        raise RuntimeError("CBFR signal crossed entry")
    if not (frame["entry_date"] < frame["exit_date"]).all():
        raise RuntimeError("CBFR entry crossed exit")
    if not frame["entry_date"].dt.quarter.eq(frame["exit_date"].dt.quarter).all():
        raise RuntimeError("CBFR event crossed a quarter")
    entries = frame["entry_position"].to_numpy(int)
    exits = frame["exit_position"].to_numpy(int)
    if np.any(entries[1:] < exits[:-1]):
        raise RuntimeError("CBFR event clock overlaps")
    return frame


def verify_preoutcome_artifacts() -> tuple[dict[str, Any], pd.DataFrame]:
    dependencies = (
        (PREREGISTRATION_SOURCE, PREREGISTRATION_SOURCE_SHA256),
        (PREREGISTRATION_DOCUMENT, PREREGISTRATION_DOCUMENT_SHA256),
        (SUPPORT, SUPPORT_SHA256),
        (PRIMARY_CLOCK, PRIMARY_CLOCK_SHA256),
        (MARKET_MANIFEST, MARKET_MANIFEST_SHA256),
        (FUNDING_MANIFEST, FUNDING_MANIFEST_SHA256),
        (STRICT_LEDGER_SOURCE, STRICT_LEDGER_SOURCE_SHA256),
    )
    for path, expected in dependencies:
        if sha256(path) != expected:
            raise RuntimeError(f"frozen CBFR dependency changed: {path}")
    support = json.loads(SUPPORT.read_text())
    if support.get("all_support_gates_pass") is not True:
        raise RuntimeError("CBFR support did not pass")
    if support.get("protocol", {}).get("evidence_boundary", {}).get(
        "post_entry_outcomes_opened"
    ) is not False:
        raise RuntimeError("CBFR support opened an outcome")
    if support.get("event_clock_sha256") != PRIMARY_EVENT_CLOCK_HASH:
        raise RuntimeError("CBFR support clock hash changed")
    selected = support.get("support_selection", {}).get("selected_cell", {})
    if (selected.get("flow_quantile"), selected.get("defense_threshold")) != (
        0.75,
        0.50,
    ):
        raise RuntimeError("CBFR selected support cell changed")

    clock_payload = json.loads(PRIMARY_CLOCK.read_text())
    if clock_payload.get("post_entry_outcomes_opened") is not False:
        raise RuntimeError("CBFR event clock opened an outcome")
    if clock_payload.get("event_clock_sha256") != PRIMARY_EVENT_CLOCK_HASH:
        raise RuntimeError("CBFR event clock hash changed")
    clock = validate_clock(_clock_frame(clock_payload["events"]))
    canonical_events = json.loads(PRIMARY_CLOCK.read_text())["events"]
    if cbfr.canonical_hash(canonical_events) != PRIMARY_EVENT_CLOCK_HASH:
        raise RuntimeError("CBFR canonical event records changed")
    return support, clock


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.is_file():
        raise RuntimeError("CBFR evaluator freeze is missing")
    payload = json.loads(EVALUATION_FREEZE.read_text())
    body = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if cbfr.canonical_hash(body) != payload.get("manifest_hash"):
        raise RuntimeError("CBFR evaluator freeze manifest hash mismatch")
    expected = {
        "protocol": "CBFR-72 strict 2023 evaluator pre-outcome freeze v1",
        "outcomes_opened": False,
        "evaluation_source": str(EVALUATION_SOURCE),
        "evaluation_source_sha256": sha256(EVALUATION_SOURCE),
        "test_path": str(TEST_PATH),
        "test_sha256": sha256(TEST_PATH),
        "freeze_source": str(FREEZE_SOURCE),
        "freeze_source_sha256": sha256(FREEZE_SOURCE),
        "freeze_test_path": str(FREEZE_TEST_PATH),
        "freeze_test_sha256": sha256(FREEZE_TEST_PATH),
        "support_sha256": SUPPORT_SHA256,
        "primary_clock_sha256": PRIMARY_CLOCK_SHA256,
        "primary_event_clock_hash": PRIMARY_EVENT_CLOCK_HASH,
        "market_data_sha256": MARKET_DATA_SHA256,
        "funding_data_sha256": FUNDING_DATA_SHA256,
        "strict_ledger_source_sha256": STRICT_LEDGER_SOURCE_SHA256,
        "evaluation_config": asdict(CONFIG),
        "mutable_parameters": [],
        "market_rows_parsed_during_freeze": 0,
        "funding_rows_loaded_during_freeze": 0,
        "execution_simulation_run_during_freeze": False,
        "opened_windows": [],
        "sealed_windows": ["2023", "2024", "2025", "2026"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"CBFR evaluator freeze changed: {key}")
    commit = str(payload.get("evaluation_source_commit", ""))
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RuntimeError("CBFR evaluator freeze commit is invalid")
    for path in (EVALUATION_SOURCE, TEST_PATH, FREEZE_SOURCE, FREEZE_TEST_PATH):
        committed = subprocess.check_output(["git", "show", f"{commit}:{path}"])
        if hashlib.sha256(committed).hexdigest() != sha256(path):
            raise RuntimeError(f"CBFR freeze commit does not bind {path}")
    return payload


def replay_primary_and_controls(
    frozen_clock: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Replay signal inputs before any held OHLC or funding is opened."""
    frame, source = cbfr.load_support_inputs(cbfr.Config())
    if source["post_entry_return_funding_pnl_or_equity_loaded"] is not False:
        raise RuntimeError("CBFR signal replay opened an outcome")
    panel = cbfr.build_feature_panel(frame, cbfr.Config())
    primary_signal = cbfr.build_signal(
        panel,
        frame,
        cbfr.Config(),
        flow_quantile=0.75,
        defense_threshold=0.50,
    )
    primary = cbfr.quarterly_schedule(primary_signal, frame)
    comparable = [
        "signal_position",
        "entry_position",
        "exit_position",
        "side",
        "branch",
        "hold_bars",
    ]
    if primary[comparable].to_dict("records") != frozen_clock[comparable].to_dict(
        "records"
    ):
        raise RuntimeError("CBFR primary signal no longer replays frozen clock")

    threshold = cbfr.lagged_flow_threshold(
        panel["flow"].abs(),
        quantile=0.75,
        window=cbfr.Config.robust_baseline_bars,
        minimum=cbfr.Config.robust_min_periods,
    )
    base = (
        panel["clean"]
        & threshold.notna()
        & panel["flow"].abs().ge(threshold)
        & (panel["direction"] * panel["completed_bar_return"]).le(0.0)
    )

    def make_control(name: str, mask: pd.Series) -> pd.DataFrame:
        side = pd.Series(0, index=panel.index, dtype=np.int8)
        side.loc[mask] = -panel.loc[mask, "direction"].astype(np.int8)
        signal = pd.DataFrame(
            {
                "date": panel["date"],
                "side": side,
                "branch": np.where(side.ne(0), name, "none"),
                "hold_bars": np.where(side.ne(0), 72, 0).astype(np.int16),
                "quarantined": False,
            }
        )
        return cbfr.quarterly_schedule(signal, frame)

    return {
        "primary": primary,
        "without_book_confirmation": make_control("without_book_confirmation", base),
        "um_only_confirmation": make_control(
            "um_only_confirmation",
            base & panel["um_defense"].ge(0.50),
        ),
        "cm_only_confirmation": make_control(
            "cm_only_confirmation",
            base & panel["cm_defense"].ge(0.50),
        ),
    }


def load_bundle_2023() -> strict.MarketBundle:
    """Load physically pre-2024 execution rows only, after evaluator freeze."""
    if sha256(MARKET_DATA) != MARKET_DATA_SHA256:
        raise RuntimeError("CBFR execution market bytes changed")
    if sha256(FUNDING_DATA) != FUNDING_DATA_SHA256:
        raise RuntimeError("CBFR realized funding bytes changed")
    market_manifest = json.loads(MARKET_MANIFEST.read_text())
    if market_manifest.get("combined_sha256") != MARKET_DATA_SHA256:
        raise RuntimeError("CBFR market manifest changed source")
    funding_manifest = json.loads(FUNDING_MANIFEST.read_text())
    funding_file = funding_manifest.get("files", {}).get("funding", {})
    if funding_file.get("sha256") != FUNDING_DATA_SHA256:
        raise RuntimeError("CBFR funding manifest changed source")
    if funding_manifest.get("protocol", {}).get("post_2023_rows_written") is not False:
        raise RuntimeError("CBFR funding source includes a later prefix")

    market = pd.read_csv(
        MARKET_DATA,
        compression="gzip",
        usecols=["date", "open", "high", "low", "close"],
        parse_dates=["date"],
    )
    market = market.loc[market["date"].ge(START) & market["date"].lt(END)].reset_index(
        drop=True
    )
    expected = pd.date_range(START, END, freq="5min", inclusive="left")
    if not pd.DatetimeIndex(market["date"]).equals(expected):
        raise RuntimeError("CBFR execution market is not the complete 2023 grid")
    prices = market[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise RuntimeError("CBFR execution OHLC is invalid")
    if np.any(market["high"] < market[["open", "low", "close"]].max(axis=1)):
        raise RuntimeError("CBFR high does not envelope its bar")
    if np.any(market["low"] > market[["open", "high", "close"]].min(axis=1)):
        raise RuntimeError("CBFR low does not envelope its bar")

    funding = pd.read_csv(
        FUNDING_DATA,
        compression="gzip",
        usecols=["funding_time", "funding_rate"],
    )
    funding["event_time"] = pd.to_datetime(
        funding["funding_time"], unit="ms", utc=True, errors="raise"
    ).dt.tz_localize(None)
    funding = funding.loc[
        funding["event_time"].ge(START) & funding["event_time"].lt(END),
        ["event_time", "funding_rate"],
    ].reset_index(drop=True)
    if len(funding) != 1_095:
        raise RuntimeError("CBFR 2023 funding event count changed")
    if not np.isfinite(funding["funding_rate"].to_numpy(float)).all():
        raise RuntimeError("CBFR funding rate is invalid")
    return strict.MarketBundle(
        dates=pd.DatetimeIndex(market["date"]),
        open=market["open"].to_numpy(float),
        high=market["high"].to_numpy(float),
        low=market["low"].to_numpy(float),
        close=market["close"].to_numpy(float),
        funding=funding,
        source_hashes={"market": sha256(MARKET_DATA), "funding": sha256(FUNDING_DATA)},
    )


def transform_clock(clock: pd.DataFrame, kind: str) -> pd.DataFrame:
    output = clock.copy()
    if kind == "direction_flip":
        output["side"] *= -1
    elif kind == "delay_five_minutes":
        output["entry_position"] += 1
        output["exit_position"] += 1
        output["entry_date"] += pd.Timedelta(minutes=CONFIG.delay_control_minutes)
        output["exit_date"] += pd.Timedelta(minutes=CONFIG.delay_control_minutes)
        if output["exit_date"].ge(END).any():
            raise RuntimeError("CBFR delay control escaped the sealed prefix")
    elif kind == "long_only":
        output = output.loc[output["side"].gt(0)].copy()
    elif kind == "short_only":
        output = output.loc[output["side"].lt(0)].copy()
    else:
        raise ValueError(kind)
    return output.sort_values("entry_position").reset_index(drop=True)


def _slim(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key != "trade_rows"}


def selection_checks(
    primary: dict[str, dict[str, Any]],
    *,
    long_only: dict[str, Any],
    short_only: dict[str, Any],
    stress: dict[str, Any],
    delayed: dict[str, Any],
    flipped: dict[str, Any],
    without_book: dict[str, Any],
    signflip: dict[str, Any],
) -> dict[str, bool]:
    annual = primary["2023"]
    return {
        "annual_absolute_return_positive": annual["absolute_return_pct"] > 0.0,
        "annual_cagr_to_strict_mdd_at_least_3": annual["cagr_to_strict_mdd"] >= 3.0,
        "annual_strict_mdd_at_most_15_pct": annual["strict_mdd_pct"] <= 15.0,
        "annual_trades_at_least_120": annual["trades"] >= 120,
        "both_halves_absolute_return_positive": all(
            primary[name]["absolute_return_pct"] > 0.0 for name in ("h1", "h2")
        ),
        "every_quarter_absolute_return_positive": all(
            primary[name]["absolute_return_pct"] > 0.0
            for name in ("q1", "q2", "q3", "q4")
        ),
        "long_only_absolute_return_positive": long_only["absolute_return_pct"] > 0.0,
        "short_only_absolute_return_positive": short_only["absolute_return_pct"] > 0.0,
        "ten_bp_stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "delay_plus_5m_absolute_return_positive": delayed["absolute_return_pct"] > 0.0,
        "direction_flip_cagr_lower": flipped["cagr_pct"] < annual["cagr_pct"],
        "book_confirmation_improves_mean_net_bps": (
            annual["mean_net_bps"] > without_book["mean_net_bps"]
        ),
        "weekly_cluster_signflip_p_at_most_0_10": (
            signflip["p_value_one_sided"] <= 0.10
        ),
    }


def evaluate(
    bundle: strict.MarketBundle,
    clocks: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    if strict.CONFIG.leverage != CONFIG.leverage:
        raise RuntimeError("shared strict ledger leverage changed")
    primary_full = {
        name: strict.simulate(
            bundle,
            clocks["primary"],
            start=start,
            end=end,
            cost_bp=CONFIG.base_cost_bp_per_notional_side,
        )
        for name, (start, end) in WINDOWS.items()
    }
    long_only = strict.simulate(
        bundle,
        transform_clock(clocks["primary"], "long_only"),
        start=START,
        end=END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    short_only = strict.simulate(
        bundle,
        transform_clock(clocks["primary"], "short_only"),
        start=START,
        end=END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    stress = strict.simulate(
        bundle,
        clocks["primary"],
        start=START,
        end=END,
        cost_bp=CONFIG.stress_cost_bp_per_notional_side,
    )
    delayed = strict.simulate(
        bundle,
        transform_clock(clocks["primary"], "delay_five_minutes"),
        start=START,
        end=END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    flipped = strict.simulate(
        bundle,
        transform_clock(clocks["primary"], "direction_flip"),
        start=START,
        end=END,
        cost_bp=CONFIG.base_cost_bp_per_notional_side,
    )
    mechanism = {
        name: strict.simulate(
            bundle,
            clock,
            start=START,
            end=END,
            cost_bp=CONFIG.base_cost_bp_per_notional_side,
        )
        for name, clock in clocks.items()
        if name != "primary"
    }
    annual_rows = primary_full["2023"]["trade_rows"]
    signflip = weekly_cluster_sign_flip(
        [float(row["net_return"]) for row in annual_rows],
        [str(row["entry_time"]) for row in annual_rows],
        permutations=CONFIG.weekly_signflip_permutations,
        seed=CONFIG.weekly_signflip_seed,
    )
    checks = selection_checks(
        primary_full,
        long_only=long_only,
        short_only=short_only,
        stress=stress,
        delayed=delayed,
        flipped=flipped,
        without_book=mechanism["without_book_confirmation"],
        signflip=signflip,
    )
    return {
        "primary": {name: _slim(stats) for name, stats in primary_full.items()},
        "long_only": _slim(long_only),
        "short_only": _slim(short_only),
        "ten_bp_notional_side_cost_stress": _slim(stress),
        "entry_and_exit_delay_plus_5m": _slim(delayed),
        "direction_flip": _slim(flipped),
        "mechanism_controls": {name: _slim(stats) for name, stats in mechanism.items()},
        "weekly_cluster_signflip": signflip,
        "selection_gates": checks,
        "passes_2023_selection": bool(all(checks.values())),
    }


def _markdown(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    lines = [
        "# CBFR-72 frozen 2023 selection outcome — 2026-07-18",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| Window / control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows = [
        (name.upper(), evaluation["primary"][name])
        for name in ("2023", "h1", "h2", "q1", "q2", "q3", "q4")
    ]
    rows.extend(
        [
            ("Long only", evaluation["long_only"]),
            ("Short only", evaluation["short_only"]),
            ("10bp stress", evaluation["ten_bp_notional_side_cost_stress"]),
            ("+5m delay", evaluation["entry_and_exit_delay_plus_5m"]),
            ("Direction flip", evaluation["direction_flip"]),
        ]
    )
    rows.extend(
        (f"Control: {name}", stats)
        for name, stats in evaluation["mechanism_controls"].items()
    )
    for label, stats in rows:
        lines.append(
            f"| {label} | {stats['absolute_return_pct']:+.3f}% | "
            f"{stats['cagr_pct']:+.3f}% | {stats['strict_mdd_pct']:.3f}% | "
            f"{stats['cagr_to_strict_mdd']:.3f} | {stats['trades']} |"
        )
    failed = [
        name for name, passed in evaluation["selection_gates"].items() if not passed
    ]
    lines.extend(
        [
            "",
            f"- Weekly-cluster sign-flip p: `{evaluation['weekly_cluster_signflip']['p_value_one_sided']:.6f}`",
            f"- Failed gates: `{failed}`",
            "- CAGR spans the full declared calendar, including every idle interval.",
            "- Strict MDD uses global/pre-entry HWM, entry cost, realized funding, favorable-before-adverse held OHLC, hypothetical liquidation cost, and exit cost.",
            "- Mechanism controls are diagnostics and cannot replace or rerank the frozen primary clock.",
            "- Failure keeps 2024 onward sealed and forbids threshold, sign, hold, or feature repair.",
            "",
        ]
    )
    return "\n".join(lines)


def _clean_repository_head() -> str:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if status:
        raise RuntimeError("repository must be clean before CBFR outcomes open")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def run(
    output: str | Path | None = None,
    docs_output: str | Path | None = None,
) -> dict[str, Any]:
    output_path = DEFAULT_OUTPUT if output is None else Path(output)
    docs_path = DEFAULT_DOCS if docs_output is None else Path(docs_output)
    if output_path != DEFAULT_OUTPUT or docs_path != DEFAULT_DOCS:
        raise ValueError("CBFR outcome output paths are immutable")
    if output_path.exists() or docs_path.exists():
        raise RuntimeError("CBFR 2023 outcome result already exists")
    support, primary = verify_preoutcome_artifacts()
    freeze = verify_evaluation_freeze()
    clocks = replay_primary_and_controls(primary)
    outcome_opening_head = _clean_repository_head()
    bundle = load_bundle_2023()
    evaluated = evaluate(bundle, clocks)
    passed = bool(evaluated["passes_2023_selection"])
    result: dict[str, Any] = {
        "protocol_version": "cbfr72_v1_2023_selection_2026-07-18",
        "outcomes_opened": True,
        "opened_window": [str(START), str(END)],
        "2024_test_opened": False,
        "2025_eval_opened": False,
        "2026_holdout_opened": False,
        "support_sha256": SUPPORT_SHA256,
        "primary_clock_sha256": PRIMARY_CLOCK_SHA256,
        "primary_event_clock_hash": PRIMARY_EVENT_CLOCK_HASH,
        "market_data_sha256": MARKET_DATA_SHA256,
        "funding_data_sha256": FUNDING_DATA_SHA256,
        "evaluation_config": asdict(CONFIG),
        "support_snapshot": support["support_selection"]["selected_cell"]["support"],
        "evaluation_freeze_hash": freeze["manifest_hash"],
        "outcome_opening_head": outcome_opening_head,
        "evaluation": evaluated,
        "decision": "2023_pass_open_2024_next" if passed else "rejected_before_2024",
        "anti_repair": (
            "A rejection seals 2024+ for CBFR-72; no sign, threshold, hold, scale, "
            "feature, side subset, or mechanism-control substitution is permitted."
        ),
    }
    result["manifest_hash"] = cbfr.canonical_hash(result)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(_markdown(result))
    return result


def main() -> None:
    result = run()
    evaluation = result["evaluation"]
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "primary": evaluation["primary"],
                "controls": {
                    "10bp": evaluation["ten_bp_notional_side_cost_stress"],
                    "+5m": evaluation["entry_and_exit_delay_plus_5m"],
                    "flip": evaluation["direction_flip"],
                    "mechanisms": evaluation["mechanism_controls"],
                },
                "signflip": evaluation["weekly_cluster_signflip"],
                "failed_gates": [
                    key
                    for key, passed in evaluation["selection_gates"].items()
                    if not passed
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
