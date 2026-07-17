"""Preregister CITA-1 without loading any BTC execution outcome."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from training import cftc_institutional_transfer_absorption_clock as clock


AS_OF_DATE = "2026-07-17"
SOURCE_COMMIT = "27823e7"
DEFAULT_OUTPUT = (
    "results/cftc_institutional_transfer_absorption_preregistration_2026-07-17.json"
)
DEFAULT_CLOCK = (
    "results/cftc_institutional_transfer_absorption_preregistered_clock_"
    "2026-07-17.csv.gz"
)
DEFAULT_DOCS = (
    "docs/cftc-institutional-transfer-absorption-preregistration-2026-07-17.md"
)
MARKET_PATH = (
    "data/binance_um_kline_reference_btc_2020_2023/"
    "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
)
FUNDING_PATH = "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz"
MARKET_MANIFEST = "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
FUNDING_MANIFEST = (
    "results/binance_um_btcusdt_funding_marks_2020_2023_manifest_2026-07-17.json"
)


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_gzip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))


def _distribution(frame: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    opened = frame.loc[
        frame["entry_time"].ge(pd.Timestamp(start, tz="UTC"))
        & frame["exit_time"].le(pd.Timestamp(end, tz="UTC"))
    ]
    months = opened["entry_time"].dt.strftime("%Y-%m").value_counts()
    return {
        "trades": len(opened),
        "longs": int(opened["side"].eq(1).sum()),
        "shorts": int(opened["side"].eq(-1).sum()),
        "special_publication_overrides": int(
            opened["special_publication_override"].astype(bool).sum()
        ),
        "max_single_month_count": int(months.max()) if not months.empty else 0,
        "max_single_month_share": (
            float(months.max() / len(opened)) if len(opened) else 0.0
        ),
    }


def build_report(clock_output: Path) -> dict[str, Any]:
    rows = clock.load_source()
    frame = clock.events_frame(clock.build_events(rows))
    _write_gzip(
        clock_output,
        frame.to_csv(index=False, lineterminator="\n").encode(),
    )
    distributions = {
        "2019_source_history": _distribution(frame, "2019-01-01", "2020-01-01"),
        "2020": _distribution(frame, "2020-01-01", "2021-01-01"),
        "2021": _distribution(frame, "2021-01-01", "2022-01-01"),
        "2022": _distribution(frame, "2022-01-01", "2023-01-01"),
        "stage1_2020_2022": _distribution(frame, "2020-01-01", "2023-01-01"),
        "2023_h1": _distribution(frame, "2023-01-01", "2023-07-01"),
        "2023_h2": _distribution(frame, "2023-07-01", "2024-01-01"),
        "stage2_2023": _distribution(frame, "2023-01-01", "2024-01-01"),
    }
    report: dict[str, Any] = {
        "protocol_version": "cftc_institutional_transfer_absorption_preregistration_v1",
        "policy_id": "CITA-1",
        "as_of_date": AS_OF_DATE,
        "source_commit": SOURCE_COMMIT,
        "hypothesis": (
            "Opposite weekly net-position changes between asset managers and "
            "leveraged funds identify an ownership transfer. Institutional "
            "accumulation against leveraged selling is persistent absorption and "
            "supports BTC; institutional distribution into leveraged buying "
            "opposes BTC."
        ),
        "source_contract": {
            "panel": str(clock.SOURCE),
            "panel_sha256": clock.SOURCE_SHA256,
            "build_manifest": str(clock.BUILD_MANIFEST),
            "build_manifest_sha256": clock.BUILD_MANIFEST_SHA256,
            "source_manifest": str(clock.SOURCE_MANIFEST),
            "source_manifest_sha256": clock.SOURCE_MANIFEST_SHA256,
            "official_report": (
                "CFTC TFF futures-only, CME Bitcoin contract 133741"
            ),
            "availability": (
                "report+8d 00:00 UTC, with exact conservative overrides for "
                "the seven documented 2023 ION catch-up publications"
            ),
            "source_rows": len(rows),
            "source_complete_rows": sum(row.source_complete for row in rows),
            "source_quarantined_rows": sum(not row.source_complete for row in rows),
            "market_or_funding_rows_loaded": 0,
        },
        "policy": {
            "features": {
                "asset_manager_net_change": (
                    "published asset-manager long change minus short change"
                ),
                "leveraged_money_net_change": (
                    "published leveraged-money long change minus short change"
                ),
            },
            "action": (
                "asset change >0 and leveraged change <0 -> LONG; asset change "
                "<0 and leveraged change >0 -> SHORT; otherwise abstain"
            ),
            "overlap_rule": (
                "source-time first admissible signal; skip a signal whose entry "
                "would overlap the prior accepted seven-day hold"
            ),
            "signal_time": "frozen conservative source availability clock",
            "entry": "signal+5m exact five-minute boundary",
            "hold": "7 days",
            "leverage": 0.5,
            "base_cost_notional_per_side": 0.0006,
            "stress_cost_notional_per_side": 0.001,
            "funding": "exact realized funding on [entry, exit)",
            "strict_mdd": (
                "full-calendar/global HWM plus favorable-before-adverse held OHLC and costs"
            ),
            "mutable_parameters": [],
        },
        "controls": {
            "mechanism": [
                "asset_manager_only",
                "leveraged_contrarian_only",
            ],
            "falsification": [
                "direction_flip",
                "one_complete_report_delay",
                "deterministic_random_side",
            ],
        },
        "stage_contract": {
            "stage1": "[2020-01-01, 2023-01-01)",
            "stage2": (
                "[2023-01-01, 2024-01-01), opened only after exact Stage1 pass replay"
            ),
            "2024_plus": "sealed",
            "stage1_gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_at_least": 3.0,
                "strict_mdd_pct_at_most": 15.0,
                "weekly_cluster_signflip_p_at_most": 0.10,
                "minimum_trades": 75,
                "mean_gross_underlying_bp_at_least": 35.0,
                "stress_cost_absolute_return_positive": True,
                "each_calendar_year_absolute_return_positive": True,
                "each_calendar_year_minimum_trades": 20,
                "mechanism_control_ratio_margin_at_least": 0.25,
            },
            "stage2_gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_at_least": 3.0,
                "strict_mdd_pct_at_most": 15.0,
                "weekly_cluster_signflip_p_at_most": 0.20,
                "minimum_trades": 15,
                "h1_and_h2_absolute_return_nonnegative": True,
                "each_half_minimum_trades": 5,
                "stress_cost_absolute_return_positive": True,
            },
        },
        "source_only_distributions": distributions,
        "clock_output": str(clock_output),
        "clock_sha256": sha256_file(clock_output),
        "opened_outcome_windows": [],
        "sealed_outcome_windows": ["stage1_2020_2022", "stage2_2023", "2024_plus"],
        "simulation_run": False,
    }
    report["manifest_hash"] = sha256_bytes(canonical_json(report))
    return report


def render_docs(report: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {name} | {item['trades']} | {item['longs']} | "
        f"{item['shorts']} | {item['special_publication_overrides']} |"
        for name, item in report["source_only_distributions"].items()
    )
    return f"""# CITA-1 CFTC institutional transfer absorption preregistration

## Mechanism

CFTC classifies asset managers/institutions and leveraged funds as distinct
buy-side groups. CITA-1 uses only their published weekly net-position changes.
When institutional net exposure rises while leveraged-fund net exposure falls,
the policy treats the transfer as persistent absorption and goes long BTC. The
reverse transfer goes short. Same-sign or zero changes abstain.

This is source-orthogonal to price action, Binance positioning, funding,
premium, FX, network, and existing portfolio scores.

## Frozen execution

- conservative CFTC availability, including the 2023 ION backlog;
- entry: availability + 5 minutes;
- hold: 7 days; source-time overlapping catch-up reports are skipped;
- exposure: 0.5x BTCUSDT perpetual;
- cost: 6 bp/notional/side, 10 bp stress;
- exact realized funding and strict intratrade MDD;
- no threshold, z-score, fitted direction, or mutable parameter.

## Source-only density

| Window | Trades | Long | Short | ION overrides |
|---|---:|---:|---:|---:|
{rows}

## Controls

- asset-manager-only and leveraged-money-contrarian-only mechanisms;
- exact direction flip;
- one complete report delay;
- deterministic hash-random side on the primary clock.

Every control receives the same cost, funding, strict-MDD, subperiod,
significance, and trade-count battery.

## Sequential boundary

Stage1 may physically parse only 2020–2022. Stage2 2023 opens only after a
hash-bound exact replay of a passing Stage1 result. Any Stage1 failure rejects
CITA-1 unchanged and leaves 2023 sealed. 2024+ remains sealed.

## Frozen identity

- source commit: {report['source_commit']}
- source panel SHA-256: {report['source_contract']['panel_sha256']}
- clock SHA-256: {report['clock_sha256']}
- preregistration manifest: {report['manifest_hash']}
"""


def run(output: Path, clock_output: Path, docs: Path) -> dict[str, Any]:
    report = build_report(clock_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(report))
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(render_docs(report))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clock-output", default=DEFAULT_CLOCK)
    parser.add_argument("--docs", default=DEFAULT_DOCS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            run(Path(args.output), Path(args.clock_output), Path(args.docs)),
            indent=2,
        )
    )

