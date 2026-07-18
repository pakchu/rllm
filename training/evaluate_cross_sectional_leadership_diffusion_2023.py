"""One-shot strict 2023 evaluator for the frozen CLD-72 event clock."""
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
from training import export_crrc_2023_execution_sources as source_export
from training import preregister_cross_sectional_leadership_diffusion as cld
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
    "training/evaluate_cross_sectional_leadership_diffusion_2023.py"
)
TEST_PATH = Path(
    "tests/test_evaluate_cross_sectional_leadership_diffusion_2023.py"
)
FREEZE_SOURCE = Path(
    "training/freeze_cross_sectional_leadership_diffusion_2023_evaluator.py"
)
FREEZE_TEST_PATH = Path(
    "tests/test_freeze_cross_sectional_leadership_diffusion_2023_evaluator.py"
)
EVALUATION_FREEZE = Path(
    "results/cross_sectional_leadership_diffusion_evaluator_freeze_2026-07-18.json"
)
SUPPORT = Path("results/cross_sectional_leadership_diffusion_support_2026-07-18.json")
PRIMARY_CLOCK = Path(
    "results/cross_sectional_leadership_diffusion_event_clock_2026-07-18.json"
)
CONTROL_CLOCKS = Path(
    "results/cross_sectional_leadership_diffusion_control_clocks_2026-07-18.json"
)
DEFAULT_OUTPUT = Path(
    "results/cross_sectional_leadership_diffusion_selection_2023_2026-07-18.json"
)
DEFAULT_DOCS = Path(
    "docs/cross-sectional-leadership-diffusion-selection-2023-2026-07-18.md"
)

SUPPORT_SHA256 = "e2e23be7504473edc0d5df44b5a25d2fa2ec6f82770206cf35bdf9ca66e020dc"
PRIMARY_CLOCK_SHA256 = "089ae3f854459a76bade4e3fd6682d1b1a9a6d600dc990a367840c179c0e623d"
PRIMARY_EVENT_CLOCK_HASH = "dcbed47f339ff8f602008ed4cdad482f2b9fcc73dc522ac3411014ca1420396e"
CONTROL_CLOCKS_SHA256 = "cbde0ff03543bc2d89d6010236f281f9b0cf8013b3082fbb36bb0c73a1c93218"
CONTROL_CLOCKS_MANIFEST_HASH = (
    "533f0079da2b6ecf767464e3edd78dffdb6ea4080a5e81aa629f2268e265feb6"
)
PREREGISTRATION_SOURCE_SHA256 = (
    "c85201b86da38a28f79885b60b4cfa0c132f6bc892e12f34b8d33118dde871c5"
)
CONTROL_FREEZER_SOURCE_SHA256 = (
    "9f56e01d8f1395bba52049841e93559f90753f496bd1dc8a0c3aaba94a490788"
)
STRICT_LEDGER_SOURCE = Path("training/evaluate_crrc_2023.py")
STRICT_LEDGER_SOURCE_SHA256 = (
    "89eb3396263689fd0a8332ffa5f1e59a88f9928a6aa8bf29032c0479b0f6afac"
)
SOURCE_EXPORTER = Path("training/export_crrc_2023_execution_sources.py")
SOURCE_EXPORTER_SHA256 = (
    "feef4d10af07771a5b79bc2c3d6e2a34d16f278fd952d18064aa186532eee32b"
)
MARKET_MANIFEST = cld.BTC_MANIFEST
MARKET_DATA = cld.BTC_DATA
MARKET_MANIFEST_SHA256 = cld.BTC_MANIFEST_SHA256
MARKET_DATA_SHA256 = cld.BTC_DATA_SHA256
FUNDING_MANIFEST = Path("results/binance_um_aux_btc_2021_2023_manifest.json")
FUNDING_DATA = Path(
    "data/binance_um_aux_btc_2021_2023/"
    "BTCUSDT_funding_2021-01-01_2023-12-31.csv.gz"
)
FUNDING_MANIFEST_SHA256 = (
    "80c77f461be54b77c7554837a304a187321a052dd05cb39b4e0a3c80de5d2bdc"
)
FUNDING_DATA_SHA256 = (
    "654c668e3aea344d5906465cbbd090f2e4ff0c47e9d4bd8cf3856c24549cfc97"
)
CONTROL_COUNTS = {
    "static_alt_breadth": 599,
    "transition_without_flow": 159,
    "transition_without_btc_lag": 136,
    "btc_momentum_at_primary_opportunities": 106,
}


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
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clock_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(events)
    required = {
        "quarter",
        "signal_position",
        "entry_position",
        "exit_position",
        "signal_date",
        "feature_boundary",
        "entry_date",
        "exit_date",
        "side",
        "branch",
        "hold_bars",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"CLD event clock misses columns: {sorted(missing)}")
    for column in ("signal_date", "feature_boundary", "entry_date", "exit_date"):
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


def validate_clock(
    frame: pd.DataFrame,
    *,
    expected_count: int | None = None,
    primary: bool = False,
) -> pd.DataFrame:
    if frame.empty:
        raise RuntimeError("CLD event clock is empty")
    if expected_count is not None and len(frame) != expected_count:
        raise RuntimeError("CLD event count changed")
    if not frame["side"].isin([-1, 1]).all():
        raise RuntimeError("CLD event clock contains an invalid side")
    if not frame["hold_bars"].eq(72).all():
        raise RuntimeError("CLD hold changed")
    if not frame["entry_position"].eq(frame["signal_position"] + 2).all():
        raise RuntimeError("CLD entry delay changed")
    if not frame["exit_position"].eq(frame["entry_position"] + 72).all():
        raise RuntimeError("CLD exit clock changed")
    if not (
        (frame["signal_date"] < frame["feature_boundary"])
        & (frame["feature_boundary"] < frame["entry_date"])
        & (frame["entry_date"] < frame["exit_date"])
    ).all():
        raise RuntimeError("CLD causal clock ordering changed")
    if not frame["entry_date"].dt.quarter.eq(frame["exit_date"].dt.quarter).all():
        raise RuntimeError("CLD clock crossed a quarter")
    entries = frame["entry_position"].to_numpy(int)
    exits = frame["exit_position"].to_numpy(int)
    if len(frame) > 1 and np.any(entries[1:] < exits[:-1]):
        raise RuntimeError("CLD event clock overlaps")
    if primary and not frame["branch"].isin(
        ["alt_leadership_diffusion_long", "alt_leadership_diffusion_short"]
    ).all():
        raise RuntimeError("CLD primary branch changed")
    return frame


def verify_preoutcome_artifacts() -> tuple[dict[str, Any], pd.DataFrame, dict[str, pd.DataFrame]]:
    dependencies = (
        (SUPPORT, SUPPORT_SHA256),
        (PRIMARY_CLOCK, PRIMARY_CLOCK_SHA256),
        (CONTROL_CLOCKS, CONTROL_CLOCKS_SHA256),
        (cld.PREREGISTRATION_SOURCE, PREREGISTRATION_SOURCE_SHA256),
        (
            Path("training/freeze_cross_sectional_leadership_diffusion_control_clocks.py"),
            CONTROL_FREEZER_SOURCE_SHA256,
        ),
        (STRICT_LEDGER_SOURCE, STRICT_LEDGER_SOURCE_SHA256),
        (SOURCE_EXPORTER, SOURCE_EXPORTER_SHA256),
        (MARKET_MANIFEST, MARKET_MANIFEST_SHA256),
        (FUNDING_MANIFEST, FUNDING_MANIFEST_SHA256),
    )
    for path, expected in dependencies:
        if sha256(path) != expected:
            raise RuntimeError(f"frozen CLD dependency changed: {path}")
    support = json.loads(SUPPORT.read_text())
    support_body = {key: value for key, value in support.items() if key != "manifest_hash"}
    if cld.canonical_hash(support_body) != support.get("manifest_hash"):
        raise RuntimeError("CLD support manifest changed")
    if support.get("all_support_gates_pass") is not True:
        raise RuntimeError("CLD support did not pass")
    if support.get("protocol", {}).get("evidence_boundary", {}).get(
        "post_entry_outcomes_opened"
    ) is not False:
        raise RuntimeError("CLD support opened an outcome")

    primary_payload = json.loads(PRIMARY_CLOCK.read_text())
    if primary_payload.get("post_entry_outcomes_opened") is not False:
        raise RuntimeError("CLD primary clock opened an outcome")
    if primary_payload.get("event_clock_sha256") != PRIMARY_EVENT_CLOCK_HASH:
        raise RuntimeError("CLD primary event hash changed")
    if cld.canonical_hash(primary_payload["events"]) != PRIMARY_EVENT_CLOCK_HASH:
        raise RuntimeError("CLD primary event records changed")
    primary = validate_clock(
        _clock_frame(primary_payload["events"]), expected_count=106, primary=True
    )

    control_payload = json.loads(CONTROL_CLOCKS.read_text())
    control_body = {
        key: value for key, value in control_payload.items() if key != "manifest_hash"
    }
    if (
        control_payload.get("manifest_hash") != CONTROL_CLOCKS_MANIFEST_HASH
        or cld.canonical_hash(control_body) != CONTROL_CLOCKS_MANIFEST_HASH
    ):
        raise RuntimeError("CLD control-clock manifest changed")
    if control_payload.get("post_entry_outcomes_opened") is not False:
        raise RuntimeError("CLD control clocks opened an outcome")
    if control_payload.get("controls_are_diagnostics_not_repair_candidates") is not True:
        raise RuntimeError("CLD anti-repair control contract changed")
    controls: dict[str, pd.DataFrame] = {}
    if set(control_payload["controls"]) != set(CONTROL_COUNTS):
        raise RuntimeError("CLD control set changed")
    for name, expected_count in CONTROL_COUNTS.items():
        item = control_payload["controls"][name]
        if item["event_count"] != expected_count:
            raise RuntimeError(f"CLD control count changed: {name}")
        if cld.canonical_hash(item["events"]) != item["event_clock_sha256"]:
            raise RuntimeError(f"CLD control event hash changed: {name}")
        controls[name] = validate_clock(
            _clock_frame(item["events"]), expected_count=expected_count
        )
    return support, primary, controls


def verify_evaluation_freeze() -> dict[str, Any]:
    if not EVALUATION_FREEZE.is_file():
        raise RuntimeError("CLD evaluator freeze is missing")
    payload = json.loads(EVALUATION_FREEZE.read_text())
    body = {key: value for key, value in payload.items() if key != "manifest_hash"}
    if cld.canonical_hash(body) != payload.get("manifest_hash"):
        raise RuntimeError("CLD evaluator freeze manifest changed")
    expected = {
        "protocol": "CLD-72 strict 2023 evaluator pre-outcome freeze v1",
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
        "control_clocks_sha256": CONTROL_CLOCKS_SHA256,
        "control_clocks_manifest_hash": CONTROL_CLOCKS_MANIFEST_HASH,
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
        "primary_clock_rows": 106,
        "control_clock_rows": CONTROL_COUNTS,
        "source_prefix_contract": {
            "market_rows": source_export.MARKET_ROWS,
            "funding_rows": source_export.FUNDING_ROWS,
            "maximum_timestamp_exclusive": str(END),
            "2024_rows_permitted": 0,
        },
        "decision_rule": (
            "open the singleton CLD-72 2023 clock once under all frozen gates; "
            "failure retires CLD-72 without opening 2024 or repairing from controls"
        ),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"CLD evaluator freeze changed: {key}")
    commit = str(payload.get("evaluation_source_commit", ""))
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RuntimeError("CLD evaluator freeze commit is invalid")
    for path in (EVALUATION_SOURCE, TEST_PATH, FREEZE_SOURCE, FREEZE_TEST_PATH):
        committed = subprocess.check_output(["git", "show", f"{commit}:{path}"])
        if hashlib.sha256(committed).hexdigest() != sha256(path):
            raise RuntimeError(f"CLD freeze commit does not bind {path}")
    return payload


def load_bundle_2023() -> strict.MarketBundle:
    """Parse only the frozen calendar-2023 execution prefix after freeze."""
    verify_evaluation_freeze()
    if sha256(MARKET_DATA) != MARKET_DATA_SHA256:
        raise RuntimeError("CLD market source changed")
    if sha256(FUNDING_DATA) != FUNDING_DATA_SHA256:
        raise RuntimeError("CLD funding source changed")
    market = source_export.validate_market_2023(
        pd.read_csv(
            MARKET_DATA,
            usecols=source_export.MARKET_COLUMNS,
            skiprows=range(
                1,
                source_export.SOURCE_MARKET_ROWS - source_export.MARKET_ROWS + 1,
            ),
            nrows=source_export.MARKET_ROWS,
        )
    )
    funding = source_export.validate_funding_2023(
        pd.read_csv(
            FUNDING_DATA,
            usecols=source_export.FUNDING_INPUT_COLUMNS,
            skiprows=range(
                1,
                source_export.SOURCE_FUNDING_ROWS - source_export.FUNDING_ROWS + 1,
            ),
            nrows=source_export.FUNDING_ROWS,
        )
    )
    arrays = {
        column: market[column].to_numpy(float)
        for column in ("open", "high", "low", "close")
    }
    return strict.MarketBundle(
        dates=pd.DatetimeIndex(market["date"]),
        open=arrays["open"],
        high=arrays["high"],
        low=arrays["low"],
        close=arrays["close"],
        funding=funding,
        source_hashes={"market": MARKET_DATA_SHA256, "funding": FUNDING_DATA_SHA256},
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
        output = output.loc[
            output["exit_date"].lt(END)
            & output["entry_date"].dt.quarter.eq(output["exit_date"].dt.quarter)
        ].copy()
    elif kind == "long_only":
        output = output.loc[output["side"].gt(0)].copy()
    elif kind == "short_only":
        output = output.loc[output["side"].lt(0)].copy()
    else:
        raise ValueError(kind)
    return output.sort_values("entry_position").reset_index(drop=True)


def _simulate(
    bundle: strict.MarketBundle,
    clock: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bp: float,
) -> dict[str, Any]:
    if strict.CONFIG.leverage != CONFIG.leverage:
        raise RuntimeError("CLD strict ledger leverage changed")
    return strict.simulate(bundle, clock, start=start, end=end, cost_bp=cost_bp)


def selection_checks(
    primary: dict[str, dict[str, Any]],
    long_only: dict[str, Any],
    short_only: dict[str, Any],
    stress: dict[str, Any],
    delayed: dict[str, Any],
    flipped: dict[str, Any],
    controls: dict[str, dict[str, Any]],
    signflip: dict[str, Any],
) -> dict[str, bool]:
    annual = primary["2023"]
    positive_quarters = sum(
        primary[name]["absolute_return_pct"] > 0.0 for name in ("q1", "q2", "q3", "q4")
    )
    primary_ratio = annual["cagr_to_strict_mdd"]
    return {
        "annual_absolute_return_positive": annual["absolute_return_pct"] > 0.0,
        "annual_cagr_to_strict_mdd_at_least_3": primary_ratio >= 3.0,
        "annual_strict_mdd_at_most_15_pct": annual["strict_mdd_pct"] <= 15.0,
        "annual_trades_at_least_80": annual["trades"] >= 80,
        "both_halves_absolute_return_positive": all(
            primary[name]["absolute_return_pct"] > 0.0 for name in ("h1", "h2")
        ),
        "at_least_three_quarters_absolute_return_positive": positive_quarters >= 3,
        "long_only_absolute_return_positive": long_only["absolute_return_pct"] > 0.0,
        "short_only_absolute_return_positive": short_only["absolute_return_pct"] > 0.0,
        "ten_bp_stress_absolute_return_positive": stress["absolute_return_pct"] > 0.0,
        "delay_plus_5m_absolute_return_positive": delayed["absolute_return_pct"] > 0.0,
        "direction_flip_cagr_lower": flipped["cagr_pct"] < annual["cagr_pct"],
        "transition_beats_static_breadth_ratio": primary_ratio
        > controls["static_alt_breadth"]["cagr_to_strict_mdd"],
        "flow_gate_improves_ratio": primary_ratio
        > controls["transition_without_flow"]["cagr_to_strict_mdd"],
        "btc_lag_gate_improves_ratio": primary_ratio
        > controls["transition_without_btc_lag"]["cagr_to_strict_mdd"],
        "alt_direction_beats_btc_momentum_ratio": primary_ratio
        > controls["btc_momentum_at_primary_opportunities"]["cagr_to_strict_mdd"],
        "weekly_cluster_signflip_p_at_most_0_10": signflip["p_value_one_sided"] <= 0.10,
    }


def evaluate(
    bundle: strict.MarketBundle,
    primary_clock: pd.DataFrame,
    control_clocks: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    primary_full = {
        name: _simulate(
            bundle,
            primary_clock,
            start,
            end,
            CONFIG.base_cost_bp_per_notional_side,
        )
        for name, (start, end) in WINDOWS.items()
    }
    long_only = _simulate(
        bundle, transform_clock(primary_clock, "long_only"), START, END, 6.0
    )
    short_only = _simulate(
        bundle, transform_clock(primary_clock, "short_only"), START, END, 6.0
    )
    stress = _simulate(
        bundle,
        primary_clock,
        START,
        END,
        CONFIG.stress_cost_bp_per_notional_side,
    )
    delayed = _simulate(
        bundle, transform_clock(primary_clock, "delay_five_minutes"), START, END, 6.0
    )
    flipped = _simulate(
        bundle, transform_clock(primary_clock, "direction_flip"), START, END, 6.0
    )
    controls = {
        name: _simulate(bundle, clock, START, END, 6.0)
        for name, clock in control_clocks.items()
    }
    annual_rows = primary_full["2023"]["trade_rows"]
    signflip = weekly_cluster_sign_flip(
        [row["net_return"] for row in annual_rows],
        [row["entry_time"] for row in annual_rows],
        permutations=CONFIG.weekly_signflip_permutations,
        seed=CONFIG.weekly_signflip_seed,
    )
    checks = selection_checks(
        primary_full,
        long_only,
        short_only,
        stress,
        delayed,
        flipped,
        controls,
        signflip,
    )
    return {
        "primary": {name: strict._slim(stats) for name, stats in primary_full.items()},
        "long_only": strict._slim(long_only),
        "short_only": strict._slim(short_only),
        "ten_bp_notional_side_cost_stress": strict._slim(stress),
        "entry_and_exit_delay_plus_5m": strict._slim(delayed),
        "direction_flip": strict._slim(flipped),
        "mechanism_controls": {
            name: strict._slim(stats) for name, stats in controls.items()
        },
        "weekly_cluster_signflip": signflip,
        "selection_gates": checks,
        "passes_2023_selection": bool(all(checks.values())),
    }


def _markdown(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    rows = [(name.upper(), evaluation["primary"][name]) for name in WINDOWS]
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
    lines = [
        "# CLD-72 frozen 2023 selection outcome — 2026-07-18",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| Window / control | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, stats in rows:
        lines.append(
            f"| {label} | {stats['absolute_return_pct']:+.3f}% | "
            f"{stats['cagr_pct']:+.3f}% | {stats['strict_mdd_pct']:.3f}% | "
            f"{stats['cagr_to_strict_mdd']:.3f} | {stats['trades']} |"
        )
    failed = [name for name, passed in evaluation["selection_gates"].items() if not passed]
    lines.extend(
        [
            "",
            f"- Weekly-cluster sign-flip p: `{evaluation['weekly_cluster_signflip']['p_value_one_sided']:.6f}`",
            f"- Failed gates: `{failed}`",
            "- CAGR uses the full declared calendar, including warm-up and idle cash.",
            "- Strict MDD uses global/pre-entry HWM, held OHLC, realized funding, and entry/liquidation/exit costs.",
            "- Controls are diagnostics only and cannot replace or repair the frozen primary clock.",
            "",
        ]
    )
    return "\n".join(lines)


def _clean_repository_head() -> str:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if status:
        raise RuntimeError("repository must be clean before CLD 2023 outcomes open")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def run(
    output: str | Path = DEFAULT_OUTPUT,
    docs_output: str | Path = DEFAULT_DOCS,
) -> dict[str, Any]:
    output_path, docs_path = Path(output), Path(docs_output)
    if output_path != DEFAULT_OUTPUT or docs_path != DEFAULT_DOCS:
        raise ValueError("CLD outcome output paths are immutable")
    if output_path.exists() or docs_path.exists():
        raise RuntimeError("CLD 2023 outcome result already exists")
    support, primary, controls = verify_preoutcome_artifacts()
    freeze = verify_evaluation_freeze()
    outcome_opening_head = _clean_repository_head()
    bundle = load_bundle_2023()
    evaluated = evaluate(bundle, primary, controls)
    passed = bool(evaluated["passes_2023_selection"])
    result = {
        "protocol_version": "cld72_v1_2023_selection_2026-07-18",
        "outcomes_opened": True,
        "opened_window": [str(START), str(END)],
        "2024_test_opened": False,
        "2025_eval_opened": False,
        "2026_holdout_opened": False,
        "support_sha256": SUPPORT_SHA256,
        "primary_clock_sha256": PRIMARY_CLOCK_SHA256,
        "control_clocks_sha256": CONTROL_CLOCKS_SHA256,
        "market_rows_loaded": source_export.MARKET_ROWS,
        "funding_rows_loaded": source_export.FUNDING_ROWS,
        "maximum_loaded_timestamp_exclusive": str(END),
        "evaluation_config": asdict(CONFIG),
        "support_snapshot": support["support_selection"]["selected_cell"]["support"],
        "evaluation_freeze_hash": freeze["manifest_hash"],
        "outcome_opening_head": outcome_opening_head,
        "evaluation": evaluated,
        "decision": "2023_pass_open_2024_next" if passed else "rejected_before_2024",
        "anti_repair": (
            "A rejection seals 2024+ for CLD-72; no sign, threshold, hold, scale, "
            "feature, or mechanism-control substitution is permitted."
        ),
    }
    result["manifest_hash"] = cld.canonical_hash(result)
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    with docs_path.open("x", encoding="utf-8") as handle:
        handle.write(_markdown(result))
    return result


def main() -> None:
    result = run()
    evaluation = result["evaluation"]
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "2024_test_opened": result["2024_test_opened"],
                "primary": evaluation["primary"],
                "controls": {
                    "10bp": evaluation["ten_bp_notional_side_cost_stress"],
                    "+5m": evaluation["entry_and_exit_delay_plus_5m"],
                    "flip": evaluation["direction_flip"],
                    "mechanisms": evaluation["mechanism_controls"],
                },
                "signflip": evaluation["weekly_cluster_signflip"],
                "failed_gates": [
                    name
                    for name, passed in evaluation["selection_gates"].items()
                    if not passed
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
