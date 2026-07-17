"""Preregister EPSB-1 without loading any BTC execution outcome."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from training import eia_petroleum_stock_breadth_clock as clock


AS_OF_DATE = "2026-07-17"
SOURCE_COMMIT = "c45dda1"
DEFAULT_OUTPUT = "results/eia_petroleum_stock_breadth_preregistration_2026-07-17.json"
DEFAULT_CLOCK = "results/eia_petroleum_stock_breadth_preregistered_clock_2026-07-17.csv.gz"
DEFAULT_DOCS = "docs/eia-petroleum-stock-breadth-preregistration-2026-07-17.md"
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
        & frame["entry_time"].lt(pd.Timestamp(end, tz="UTC"))
    ]
    months = opened["entry_time"].dt.strftime("%Y-%m").value_counts()
    return {
        "trades": len(opened),
        "longs": int(opened["side"].eq(1).sum()),
        "shorts": int(opened["side"].eq(-1).sum()),
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
        "stage2_2023": _distribution(frame, "2023-01-01", "2024-01-01"),
    }
    report: dict[str, Any] = {
        "protocol_version": "eia_petroleum_stock_breadth_preregistration_v1",
        "policy_id": "EPSB-1",
        "as_of_date": AS_OF_DATE,
        "source_commit": SOURCE_COMMIT,
        "hypothesis": (
            "Concordant builds across commercial crude, gasoline, and distillate "
            "stocks are a broad physical-glut/disinflation impulse that supports "
            "BTC; concordant draws are a scarcity/reflation impulse that opposes BTC."
        ),
        "source_contract": {
            "panel": str(clock.SOURCE),
            "panel_sha256": clock.SOURCE_SHA256,
            "build_manifest": str(clock.BUILD_MANIFEST),
            "build_manifest_sha256": clock.BUILD_MANIFEST_SHA256,
            "source_manifest": str(clock.SOURCE_MANIFEST),
            "source_manifest_sha256": clock.SOURCE_MANIFEST_SHA256,
            "point_in_time_values": "each archived EIA WPSR issue's Table 1",
            "availability": "next UTC day 13:00 after official release date",
            "source_rows": len(rows),
            "source_complete_rows": sum(row.source_complete for row in rows),
            "source_quarantined_rows": sum(not row.source_complete for row in rows),
            "market_or_funding_rows_loaded": 0,
        },
        "policy": {
            "features": {
                "commercial_crude_change": "published issue-local Difference",
                "gasoline_change": "published issue-local Difference",
                "distillate_change": "published issue-local Difference",
            },
            "action": (
                "all three >0 -> LONG; all three <0 -> SHORT; mixed, zero, or "
                "quarantined -> no trade"
            ),
            "signal_time": "conservative source availability clock",
            "entry": "signal+5m exact five-minute boundary",
            "hold": "72 hours",
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
            "mechanism": ["crude_only", "refined_products_only"],
            "falsification": [
                "direction_flip",
                "one_complete_release_delay",
                "deterministic_random_side",
            ],
        },
        "stage_contract": {
            "stage1": "[2020-01-01, 2023-01-01)",
            "stage2": "[2023-01-01, 2024-01-01), opened only after exact Stage1 pass replay",
            "2024_plus": "sealed",
            "stage1_gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_at_least": 3.0,
                "strict_mdd_pct_at_most": 15.0,
                "weekly_cluster_signflip_p_at_most": 0.10,
                "minimum_trades": 30,
                "mean_gross_underlying_bp_at_least": 35.0,
                "stress_cost_absolute_return_positive": True,
                "each_calendar_year_absolute_return_positive": True,
                "each_calendar_year_minimum_trades": 8,
                "mechanism_control_ratio_margin_at_least": 0.25,
            },
            "stage2_gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_at_least": 3.0,
                "strict_mdd_pct_at_most": 15.0,
                "weekly_cluster_signflip_p_at_most": 0.20,
                "minimum_trades": 10,
                "h1_and_h2_absolute_return_nonnegative": True,
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
    distributions = report["source_only_distributions"]
    rows = "\n".join(
        f"| {name} | {item['trades']} | {item['longs']} | {item['shorts']} |"
        for name, item in distributions.items()
    )
    return f"""# EPSB-1 EIA petroleum stock breadth preregistration

## Mechanism

EPSB-1 combines three weak, source-orthogonal physical inventory signals from
the same point-in-time WPSR issue. Concordant commercial-crude, gasoline, and
distillate builds represent a broad glut/disinflation impulse and go long BTC;
concordant draws represent a scarcity/reflation impulse and go short. Mixed,
zero, or quarantined releases abstain.

## Frozen execution

- source availability: next UTC day 13:00 after the official release date;
- entry: availability + 5 minutes;
- hold: 72 hours;
- exposure: 0.5x BTCUSDT perpetual;
- cost: 6 bp/notional/side, 10 bp stress;
- exact realized funding and strict intratrade MDD;
- no grid and no mutable parameter.

## Source-only density

| Window | Trades | Long | Short |
|---|---:|---:|---:|
{rows}

## Controls

- commercial-crude-only and refined-products-only mechanism controls;
- exact direction flip;
- one complete WPSR release delay;
- deterministic hash-random side on the primary clock.

Every control receives the same cost, funding, strict-MDD, subperiod,
significance, and trade-count battery.

## Sequential boundary

Stage1 may physically parse only 2020–2022. Stage2 2023 opens only after a
hash-bound exact replay of a passing Stage1 result. Any Stage1 failure rejects
EPSB-1 unchanged and leaves 2023 sealed. 2024+ remains sealed.

## Frozen identity

- source commit: `{report['source_commit']}`
- source panel SHA-256: `{report['source_contract']['panel_sha256']}`
- clock SHA-256: `{report['clock_sha256']}`
- preregistration manifest: `{report['manifest_hash']}`
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
