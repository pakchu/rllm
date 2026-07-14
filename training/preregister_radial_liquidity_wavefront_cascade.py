"""Outcome-blind support preregistration for RLWC-144.

RLWC-144 detects ordered outer-to-inner flow propagation in non-overlapping
book-depth shells.  It uses only the frozen calendar-2023 shell panel and
contains no market-price, return, PnL, CAGR, or drawdown calculation.
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

from training import preregister_cross_collateral_liquidity_credibility_fracture as pdf
from training import preregister_cross_collateral_liquidity_hysteresis as cclh
from training import preregister_cross_collateral_liquidity_void_refill as clvr


FROZEN_SHELL_MANIFEST = Path(
    "results/binance_cross_collateral_book_shells_btc_2023_manifest.json"
)
FROZEN_SHELL_MANIFEST_SHA256 = (
    "1b5519143d58f62ef3e8b6d9e22f012f80197a59903509041aca24252ed04521"
)
FROZEN_SHELL_DATA = Path(
    "data/binance_cross_collateral_book_shells_btc_2023/"
    "BTC_cross_collateral_book_shells_5m_2023.csv.gz"
)
FROZEN_SHELL_DATA_SHA256 = (
    "ead931ec8ce2bbd73c946b8660e16d7750ce73051e60ce4989467a7c5bc68342"
)
PDF_EVENT_CLOCK_SHA256 = (
    "ce1c6ec42434874d97c6b6034f51a73771b27e314da6d37a4f44b0563e6972e2"
)
PREREGISTRATION_SOURCE = Path(
    "training/preregister_radial_liquidity_wavefront_cascade.py"
)
PREREGISTRATION_DOCUMENT = Path(
    "docs/radial-liquidity-wavefront-cascade-preregistration-2026-07-14.md"
)
SHELL_BUILDER_SOURCE = Path(
    "training/build_binance_cross_collateral_book_shells_2023.py"
)
STANDARDIZER_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_void_refill.py"
)
SCHEDULER_SOURCE = Path(
    "training/preregister_metaorder_fragmentation_impact_curvature.py"
)
CCLH_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_hysteresis.py"
)
CCLH_SUPPORT = Path(
    "results/cross_collateral_liquidity_hysteresis_support_2026-07-14.json"
)
PDF_SOURCE = Path(
    "training/preregister_cross_collateral_liquidity_credibility_fracture.py"
)
PDF_SUPPORT = Path(
    "results/cross_collateral_liquidity_credibility_fracture_"
    "support_2026-07-14.json"
)
PDF_EVENT_CLOCK = Path(
    "results/cross_collateral_liquidity_credibility_fracture_"
    "event_clock_2026-07-14.json"
)


@dataclass(frozen=True)
class Config:
    shell_manifest: str = str(FROZEN_SHELL_MANIFEST)
    output: str = (
        "results/radial_liquidity_wavefront_cascade_support_2026-07-14.json"
    )
    robust_baseline_bars: int = 8_640
    robust_min_periods: int = 2_016
    wave_lookback_bars: int = 6
    outer_z: float = 1.25
    middle_z: float = 1.00
    inner_z: float = 1.25
    early_inner_veto_z: float = 1.00
    terminal_outer_veto_z: float = 1.00
    minimum_stage_efficiency: float = 0.35
    recent_wave_bars: int = 2
    hold_bars: int = 144
    minimum_nonoverlap_total: int = 120
    minimum_nonoverlap_per_half: int = 45
    minimum_nonoverlap_per_quarter: int = 20
    minimum_side_share: float = 0.35
    maximum_quarter_share: float = 0.40
    overlap_tolerance_bars: int = 12
    maximum_prior_event_jaccard: float = 0.35


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_frozen_config(cfg: Config) -> None:
    expected = {
        "shell_manifest": str(FROZEN_SHELL_MANIFEST),
        "robust_baseline_bars": 8_640,
        "robust_min_periods": 2_016,
        "wave_lookback_bars": 6,
        "outer_z": 1.25,
        "middle_z": 1.00,
        "inner_z": 1.25,
        "early_inner_veto_z": 1.00,
        "terminal_outer_veto_z": 1.00,
        "minimum_stage_efficiency": 0.35,
        "recent_wave_bars": 2,
        "hold_bars": 144,
        "minimum_nonoverlap_total": 120,
        "minimum_nonoverlap_per_half": 45,
        "minimum_nonoverlap_per_quarter": 20,
        "minimum_side_share": 0.35,
        "maximum_quarter_share": 0.40,
        "overlap_tolerance_bars": 12,
        "maximum_prior_event_jaccard": 0.35,
    }
    changed = {
        name: {"expected": value, "observed": getattr(cfg, name)}
        for name, value in expected.items()
        if getattr(cfg, name) != value
    }
    if changed:
        raise ValueError(f"RLWC-144 v1 config is frozen: {changed}")


def _required_columns() -> list[str]:
    depth = [
        f"{venue}_depth_{side}{distance}"
        for venue in ("um", "cm")
        for side in ("m", "p")
        for distance in range(1, 6)
    ]
    shell = [
        f"{venue}_shell_{statistic}_{side}{distance}"
        for venue in ("um", "cm")
        for statistic in (
            "share_median",
            "flow_net",
            "flow_add",
            "flow_withdraw",
            "flow_churn",
            "flow_efficiency",
        )
        for side in ("m", "p")
        for distance in range(1, 6)
    ]
    return depth + shell


def load_shells(cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = Path(cfg.shell_manifest)
    if _sha256(manifest_path) != FROZEN_SHELL_MANIFEST_SHA256:
        raise ValueError("radial-shell manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest.get("protocol", {})
    if protocol.get("outcomes_opened") is not False:
        raise ValueError("radial-shell manifest opened outcomes")
    if protocol.get("post_2023_rows_requested") is not False:
        raise ValueError("radial-shell manifest requested post-2023 rows")
    if protocol.get("base_depth_replayed_exactly") is not True:
        raise ValueError("radial-shell manifest did not replay base depth")
    if protocol.get("raw_archives_retained") is not False:
        raise ValueError("radial-shell manifest retained raw archives")
    item = manifest.get("file", {})
    path = Path(item.get("path", ""))
    if path != FROZEN_SHELL_DATA:
        raise ValueError("radial-shell manifest points to a non-frozen panel")
    if item.get("sha256") != FROZEN_SHELL_DATA_SHA256:
        raise ValueError("radial-shell manifest contains a non-frozen data hash")
    if not path.is_file() or _sha256(path) != FROZEN_SHELL_DATA_SHA256:
        raise ValueError("radial-shell data hash mismatch")

    frame = pd.read_csv(path, compression="gzip", parse_dates=["date"])
    clvr._validate_grid(
        frame,
        start="2023-01-01",
        end="2024-01-01",
        label="cross-collateral radial shells",
    )
    if len(frame) != item.get("rows") or len(frame.columns) != item.get("columns"):
        raise ValueError("radial-shell dimensions differ from manifest")
    complete = frame["source_complete"]
    if complete.dtype != bool:
        complete = complete.astype("string").str.lower().map(
            {"true": True, "false": False}
        )
    if complete.isna().any():
        raise ValueError("source_complete contains an unknown value")
    frame["source_complete"] = complete.astype(bool)
    required = _required_columns()
    if not set(required).issubset(frame.columns):
        raise ValueError("radial-shell columns are incomplete")
    if not np.isfinite(
        frame.loc[frame["source_complete"], required].to_numpy(float)
    ).all():
        raise ValueError("complete radial-shell row contains non-finite data")
    frame["quarantined"] = False
    return frame, {
        "shell_manifest_sha256": _sha256(manifest_path),
        "shell_data_sha256": _sha256(path),
        "range_start": "2023-01-01 00:00:00",
        "range_end": "2023-12-31 23:55:00",
        "rows": int(len(frame)),
        "source_complete_rows": int(frame["source_complete"].sum()),
    }


def _lagged_z(values: pd.Series, clean: pd.Series, cfg: Config) -> pd.Series:
    return clvr.lagged_robust_zscore(
        values.where(clean),
        window=cfg.robust_baseline_bars,
        minimum=cfg.robust_min_periods,
    )


def detect_wavefront(
    outer: pd.Series,
    middle: pd.Series,
    inner: pd.Series,
    outer_efficiency: pd.Series,
    middle_efficiency: pd.Series,
    inner_efficiency: pd.Series,
    outer_raw_valid: pd.Series,
    middle_raw_valid: pd.Series,
    inner_raw_valid: pd.Series,
    clean: pd.Series,
    cfg: Config,
) -> pd.Series:
    """Confirm a causal outer→middle→inner wave at each completed bar."""
    series = [
        outer,
        middle,
        inner,
        outer_efficiency,
        middle_efficiency,
        inner_efficiency,
    ]
    if any(len(values) != len(clean) for values in series):
        raise ValueError("wavefront inputs must have equal length")
    if cfg.wave_lookback_bars != 6:
        raise ValueError("RLWC-144 v1 freezes a six-bar wave window")
    numeric = np.column_stack([values.to_numpy(float) for values in series])
    observed = clean.astype(bool).to_numpy() & np.isfinite(numeric).all(axis=1)
    raw_valid = np.column_stack(
        [outer_raw_valid, middle_raw_valid, inner_raw_valid]
    ).astype(bool)
    outer_values = outer.to_numpy(float)
    middle_values = middle.to_numpy(float)
    inner_values = inner.to_numpy(float)
    efficiencies = np.column_stack(
        [outer_efficiency, middle_efficiency, inner_efficiency]
    ).astype(float)
    waves = np.zeros(len(clean), dtype=bool)
    for terminal in range(cfg.wave_lookback_bars - 1, len(clean)):
        start = terminal - cfg.wave_lookback_bars + 1
        if not observed[start : terminal + 1].all():
            continue
        outer_positions = np.arange(start, terminal - 2)
        outer_position = int(
            outer_positions[np.argmax(outer_values[outer_positions])]
        )
        if outer_values[outer_position] < cfg.outer_z:
            continue
        middle_positions = np.arange(outer_position + 1, terminal)
        if len(middle_positions) == 0:
            continue
        middle_position = int(
            middle_positions[np.argmax(middle_values[middle_positions])]
        )
        if middle_values[middle_position] < cfg.middle_z:
            continue
        if inner_values[terminal] < cfg.inner_z:
            continue
        if not (
            raw_valid[outer_position, 0]
            and raw_valid[middle_position, 1]
            and raw_valid[terminal, 2]
        ):
            continue
        if min(
            efficiencies[outer_position, 0],
            efficiencies[middle_position, 1],
            efficiencies[terminal, 2],
        ) < cfg.minimum_stage_efficiency:
            continue
        if np.any(
            inner_values[start:middle_position] >= cfg.early_inner_veto_z
        ):
            continue
        if outer_values[terminal] >= cfg.terminal_outer_veto_z:
            continue
        waves[terminal] = True
    return pd.Series(waves, index=clean.index)


def _wave_for(
    frame: pd.DataFrame,
    venue: str,
    side: str,
    kind: str,
    cfg: Config,
) -> pd.Series:
    if venue not in ("um", "cm") or side not in ("m", "p"):
        raise ValueError("unknown RLWC venue or side")
    if kind not in ("add", "withdraw"):
        raise ValueError("unknown RLWC wave kind")
    clean = frame["source_complete"].astype(bool)
    raw = [
        frame[f"{venue}_shell_flow_net_{side}{shell}"].astype(float)
        for shell in range(1, 6)
    ]
    zscore = [_lagged_z(values, clean, cfg) for values in raw]
    sign = 1.0 if kind == "add" else -1.0
    signed_z = [sign * values for values in zscore]
    signed_raw = [sign * values for values in raw]
    efficiency = [
        frame[f"{venue}_shell_flow_efficiency_{side}{shell}"].astype(float)
        for shell in range(1, 6)
    ]
    outer = 0.5 * (signed_z[3] + signed_z[4])
    middle = signed_z[2]
    inner = 0.5 * (signed_z[0] + signed_z[1])
    outer_efficiency = pd.concat(efficiency[3:5], axis=1).min(axis=1)
    middle_efficiency = efficiency[2]
    inner_efficiency = pd.concat(efficiency[0:2], axis=1).min(axis=1)
    return detect_wavefront(
        outer,
        middle,
        inner,
        outer_efficiency,
        middle_efficiency,
        inner_efficiency,
        (signed_raw[3] > 0.0) & (signed_raw[4] > 0.0),
        signed_raw[2] > 0.0,
        (signed_raw[0] > 0.0) & (signed_raw[1] > 0.0),
        clean,
        cfg,
    )


def _recent(wave: pd.Series, bars: int) -> pd.Series:
    if bars != 2:
        raise ValueError("RLWC-144 v1 freezes two-bar cross-venue tolerance")
    return wave.astype(bool) | wave.shift(1, fill_value=False).astype(bool)


def build_signal(
    frame: pd.DataFrame,
    cfg: Config,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    waves = {
        f"{venue}_{side}_{kind}": _wave_for(
            frame, venue, side, kind, cfg
        )
        for venue in ("um", "cm")
        for side in ("m", "p")
        for kind in ("add", "withdraw")
    }
    recent = {
        name: _recent(values, cfg.recent_wave_bars)
        for name, values in waves.items()
    }
    ask_withdraw_both = recent["um_p_withdraw"] & recent["cm_p_withdraw"]
    bid_withdraw_both = recent["um_m_withdraw"] & recent["cm_m_withdraw"]
    bid_add_any = recent["um_m_add"] | recent["cm_m_add"]
    ask_add_any = recent["um_p_add"] | recent["cm_p_add"]
    ask_add_both = recent["um_p_add"] & recent["cm_p_add"]
    bid_add_both = recent["um_m_add"] & recent["cm_m_add"]
    decision_clean = frame["source_complete"].astype(bool)

    bullish = (
        decision_clean
        & ask_withdraw_both
        & bid_add_any
        & ~bid_withdraw_both
        & ~ask_add_both
    )
    bearish = (
        decision_clean
        & bid_withdraw_both
        & ask_add_any
        & ~ask_withdraw_both
        & ~bid_add_both
    )
    conflict = bullish & bearish
    side = pd.Series(0, index=frame.index, dtype=np.int8)
    side.loc[bullish & ~conflict] = 1
    side.loc[bearish & ~conflict] = -1
    branch = pd.Series("none", index=frame.index, dtype="string")
    branch.loc[side.gt(0)] = "ask_withdrawal_wave_bid_addition"
    branch.loc[side.lt(0)] = "bid_withdrawal_wave_ask_addition"
    signal = pd.DataFrame(
        {
            "date": frame["date"],
            "candidate": side.ne(0),
            "side": side,
            "branch": branch,
            "hold_bars": np.where(side.ne(0), cfg.hold_bars, 0).astype(
                np.int16
            ),
        }
    )
    return signal, waves


def support_summary(
    schedule: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    total = len(schedule)
    if total:
        quarter = pd.to_datetime(schedule["signal_date"]).dt.quarter
    else:
        quarter = pd.Series(dtype=np.int8)
    by_quarter = {
        f"q{number}": int(quarter.eq(number).sum()) for number in range(1, 5)
    }
    h1 = by_quarter["q1"] + by_quarter["q2"]
    h2 = by_quarter["q3"] + by_quarter["q4"]
    long_share = float(schedule["side"].gt(0).mean()) if total else 0.0
    short_share = float(schedule["side"].lt(0).mean()) if total else 0.0
    maximum_quarter_share = max(by_quarter.values()) / total if total else 1.0
    passes = (
        total >= cfg.minimum_nonoverlap_total
        and h1 >= cfg.minimum_nonoverlap_per_half
        and h2 >= cfg.minimum_nonoverlap_per_half
        and all(
            count >= cfg.minimum_nonoverlap_per_quarter
            for count in by_quarter.values()
        )
        and min(long_share, short_share) >= cfg.minimum_side_share
        and maximum_quarter_share <= cfg.maximum_quarter_share
    )
    return {
        "nonoverlap_total": int(total),
        "by_quarter": by_quarter,
        "h1": int(h1),
        "h2": int(h2),
        "long_share": long_share,
        "short_share": short_share,
        "maximum_observed_quarter_share": float(maximum_quarter_share),
        "passes_support": bool(passes),
    }


def _pdf_event_clock_sha256(schedule: pd.DataFrame) -> str:
    records = [
        {
            "signal_position": int(row.signal_position),
            "entry_position": int(row.entry_position),
            "exit_position": int(row.exit_position),
            "side": int(row.side),
            "branch": str(row.branch),
            "hold_bars": int(row.hold_bars),
        }
        for row in schedule[
            [
                "signal_position",
                "entry_position",
                "exit_position",
                "side",
                "branch",
                "hold_bars",
            ]
        ].itertuples(index=False)
    ]
    payload = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _independence_summary(
    schedule: pd.DataFrame,
    frame: pd.DataFrame,
    cfg: Config,
) -> dict[str, Any]:
    cclh_signal = cclh.build_signal(frame, cclh.Config())
    cclh_schedule = pdf._quarterly_schedule(cclh_signal, frame)
    if pdf._event_clock_sha256(cclh_schedule) != pdf.CCLH_EVENT_CLOCK_SHA256:
        raise ValueError("CCLH event clock did not replay from shell panel")

    credibility, _ = pdf.load_credibility(pdf.Config())
    pdf_signal = pdf.build_signal(credibility, pdf.Config())
    pdf_schedule = pdf._quarterly_schedule(pdf_signal, credibility)
    if _pdf_event_clock_sha256(pdf_schedule) != PDF_EVENT_CLOCK_SHA256:
        raise ValueError("PDF-10 event clock did not replay exactly")

    current_positions = schedule["signal_position"].astype(int).tolist()
    output: dict[str, Any] = {}
    for name, prior in (("cclh", cclh_schedule), ("pdf10", pdf_schedule)):
        overlap = cclh.tolerant_event_jaccard(
            current_positions,
            prior["signal_position"].astype(int).tolist(),
            tolerance_bars=cfg.overlap_tolerance_bars,
        )
        overlap["maximum_allowed_jaccard"] = cfg.maximum_prior_event_jaccard
        overlap["passes"] = bool(
            overlap["jaccard"] <= cfg.maximum_prior_event_jaccard
        )
        output[name] = overlap
    output["passes_independence"] = bool(
        all(item["passes"] for item in output.values())
    )
    return output


def run_support(cfg: Config) -> dict[str, Any]:
    _validate_frozen_config(cfg)
    frame, source = load_shells(cfg)
    signal, waves = build_signal(frame, cfg)
    schedule = pdf._quarterly_schedule(signal, frame)
    support = support_summary(schedule, cfg)
    independence = _independence_summary(schedule, frame, cfg)
    passes_all = support["passes_support"] and independence["passes_independence"]
    wave_counts = {
        name: int(values.sum()) for name, values in waves.items()
    }
    return {
        "protocol": {
            "name": "RLWC-144 — Radial Liquidity Wavefront Cascade",
            "support_only": True,
            "outcomes_opened_for_rlwc": False,
            "price_or_return_loaded": False,
            "support_rejected": not passes_all,
            "selection_end_exclusive": "2024-01-01 00:00:00",
            "event_clock": (
                "six completed 5m bars with ordered outer-to-middle-to-inner "
                "shell flow propagation"
            ),
            "signal_availability": "terminal wave bar complete; enter next 5m open",
            "action_rule": (
                "ask withdrawal plus bid addition long; exact side symmetry short"
            ),
            "holding_rule": "144 completed 5m bars; scheduled-open exit",
            "quarter_boundary_policy": (
                "four quarter-contained schedules concatenated; reset flat"
            ),
            "support_parameters_searched": False,
            "source_gap_policy": (
                "decision bar and all six participating wave bars are "
                "source-complete; future shell gaps do not cancel an "
                "entered trade"
            ),
            "sealed_windows": ["test2024", "eval2025", "ytd2026"],
        },
        "config": asdict(cfg),
        "frozen_artifacts": {
            "preregistration_source": str(PREREGISTRATION_SOURCE),
            "preregistration_source_sha256": _sha256(PREREGISTRATION_SOURCE),
            "preregistration_document": str(PREREGISTRATION_DOCUMENT),
            "preregistration_document_sha256": _sha256(
                PREREGISTRATION_DOCUMENT
            ),
            "shell_builder_source": str(SHELL_BUILDER_SOURCE),
            "shell_builder_source_sha256": _sha256(SHELL_BUILDER_SOURCE),
            "standardizer_source": str(STANDARDIZER_SOURCE),
            "standardizer_source_sha256": _sha256(STANDARDIZER_SOURCE),
            "scheduler_source": str(SCHEDULER_SOURCE),
            "scheduler_source_sha256": _sha256(SCHEDULER_SOURCE),
            "cclh_source": str(CCLH_SOURCE),
            "cclh_source_sha256": _sha256(CCLH_SOURCE),
            "cclh_support": str(CCLH_SUPPORT),
            "cclh_support_sha256": _sha256(CCLH_SUPPORT),
            "pdf_source": str(PDF_SOURCE),
            "pdf_source_sha256": _sha256(PDF_SOURCE),
            "pdf_support": str(PDF_SUPPORT),
            "pdf_support_sha256": _sha256(PDF_SUPPORT),
            "pdf_event_clock": str(PDF_EVENT_CLOCK),
            "pdf_event_clock_sha256": _sha256(PDF_EVENT_CLOCK),
            "shell_manifest_sha256": _sha256(cfg.shell_manifest),
        },
        "source": source,
        "feature": {
            "wave_counts": wave_counts,
            "raw_candidate_count": int(signal["candidate"].sum()),
            "standardization": (
                "each venue/side/shell flow_net uses strictly lagged rolling "
                "median and recursive MAD; clip [-12, 12]"
            ),
        },
        "support_calibration": {
            "outcomes_opened_for_rlwc": False,
            "parameters_searched": False,
            "all_parameters_fixed": True,
            "further_support_repairs_allowed": False,
        },
        "scheduled_side_counts": {
            "long": int(schedule["side"].gt(0).sum()),
            "short": int(schedule["side"].lt(0).sum()),
        },
        "scheduled_branch_counts": {
            name: int(count)
            for name, count in schedule["branch"].value_counts().items()
        },
        "independence": independence,
        "support": support,
        "all_support_gates_pass": bool(passes_all),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=Config.output)
    args = parser.parse_args()
    result = run_support(Config(output=args.output))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "outcomes_opened_for_rlwc": False,
                "support_rejected": result["protocol"]["support_rejected"],
                "support": result["support"],
                "independence": result["independence"],
                "wave_counts": result["feature"]["wave_counts"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
