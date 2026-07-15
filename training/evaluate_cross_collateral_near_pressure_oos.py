#!/usr/bin/env python3
"""Open frozen 2024-2026H1 OOS for cross-collateral near pressure."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_fresh_kimchi_orthogonal_alpha import (
    Config as Rank7AuditConfig,
    build_rank7_context,
    daily_marked_returns,
    pnl_correlation,
    rank7_schedule,
    trade_timing_overlap,
)
from training.evaluate_expanding_extratrees_top10_oos import FULL_CUTOFF
from training.preregister_cross_collateral_liquidity_void_refill import lagged_robust_zscore
from training.search_inventory_purge_reclaim_alpha import (
    Config as ExecutionConfig,
    ExecutionEngine,
    Trade,
    _schedule_hash,
    equity_stats,
)
from training.select_cross_collateral_near_pressure_pre2024 import (
    Config as SelectionConfig,
    EXPECTED_SELECTED,
    WINDOWS as SELECTION_WINDOWS,
    json_hash,
    load_sources as load_selection_sources,
    raw_pressure,
    resolve_existing,
    schedule,
    sha256,
)


SELECTION_MANIFEST = "results/cross_collateral_near_pressure_pre2024_manifest_2026-07-16.json"
EXPECTED_SELECTION_MANIFEST_SHA256 = "808a2e042af461e79c9c37102cea0e8b9faa483a24dabd581a32730ff49224bd"
EXPECTED_SELECTION_MANIFEST_HASH = "c7840f17e06804d8ec8bc8e50cfb880696505146d0f81a9f21fd9607e5b5d4d7"
FUTURE_BOOK_MANIFEST = "results/binance_cross_collateral_near_pressure_btc_2024_2026_manifest.json"
EXPECTED_FUTURE_BOOK_MANIFEST_SHA256 = "f16a82d45749f77961e987151805884c2911c6d8013b841e3e4120b941f5921f"
EXPECTED_BUILDER_SHA256 = "f681dbc0d55403c077e70ba44447a7422b634ad38a7a3be35c8837cefcf2f21a"
EXPECTED_BUILDER_DEPENDENCIES = {
    "book_depth_parser": "ea828e4d58e2b57b6fab363144a04575bb731b30eaf925ecb03dbc8d5706e06f",
    "shell_aggregator": "8d343830e4d51596e7b369f303a7ba3fc807dbecb5f19193028dcf43c8c67a1c",
}
DEFAULT_OUTPUT = "results/cross_collateral_near_pressure_oos_2026-07-16.json"
DEFAULT_DOCS = "docs/cross-collateral-near-pressure-oos-2026-07-16.md"

WINDOWS: dict[str, tuple[str, str]] = {
    "test_2024": ("2024-01-01", "2025-01-01"),
    "eval_2025": ("2025-01-01", "2026-01-01"),
    "holdout_2026h1": ("2026-01-01", FULL_CUTOFF),
    "future_2025_2026h1": ("2025-01-01", FULL_CUTOFF),
    "oos_2024_2026h1": ("2024-01-01", FULL_CUTOFF),
    "all_2023_2026h1": ("2023-01-01", FULL_CUTOFF),
}

ORTHOGONALITY_LIMITS = {
    "exact_entry_jaccard": 0.02,
    "candidate_entries_near_6h_fraction": 0.10,
    "position_jaccard": 0.15,
    "absolute_daily_pnl_pearson": 0.30,
    "minimum_nonzero_pnl_days": 10,
}


@dataclass(frozen=True)
class Config:
    selection_manifest: str = SELECTION_MANIFEST
    future_book_manifest: str = FUTURE_BOOK_MANIFEST
    output: str = DEFAULT_OUTPUT
    docs_output: str = DEFAULT_DOCS


def validate_selection_manifest(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(path)
    if sha256(manifest_path) != EXPECTED_SELECTION_MANIFEST_SHA256:
        raise RuntimeError("selection manifest file hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedded = manifest.pop("manifest_hash", None)
    calculated = json_hash(manifest)
    manifest["manifest_hash"] = embedded
    if embedded != calculated or embedded != EXPECTED_SELECTION_MANIFEST_HASH:
        raise RuntimeError("selection manifest self-hash mismatch")
    if manifest["future_outcomes_opened"] is not False:
        raise RuntimeError("selection manifest already opened future outcomes")
    if manifest["selection_cutoff_exclusive"] != "2024-01-01":
        raise RuntimeError("selection cutoff drifted")
    if manifest["selected_spec"] != EXPECTED_SELECTED:
        raise RuntimeError("frozen selected spec drifted")
    result_path = Path(manifest["selection_result"])
    if sha256(result_path) != manifest["selection_result_sha256"]:
        raise RuntimeError("selection result hash mismatch")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["post_2023_rows_opened"] is not False:
        raise RuntimeError("selection result opened future rows")
    return manifest, result


def validate_future_book_manifest(path: str) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = Path(path)
    if sha256(manifest_path) != EXPECTED_FUTURE_BOOK_MANIFEST_SHA256:
        raise RuntimeError("future book manifest file hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    expected_protocol = {
        "outcomes_opened": False,
        "price_or_return_loaded": False,
        "raw_archives_retained": False,
        "checksums_verified": True,
        "start_inclusive": "2024-01-01",
        "end_exclusive": FULL_CUTOFF,
    }
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise RuntimeError(f"future book protocol drifted: {key}")
    if manifest["builder_sha256"] != EXPECTED_BUILDER_SHA256:
        raise RuntimeError("future book builder hash mismatch")
    if manifest.get("dependency_sha256") != EXPECTED_BUILDER_DEPENDENCIES:
        raise RuntimeError("future book builder dependency hash mismatch")
    item = manifest["file"]
    data_path = resolve_existing(item["path"])
    if sha256(data_path) != item["sha256"]:
        raise RuntimeError("future book panel hash mismatch")
    frame = pd.read_csv(data_path, compression="infer", parse_dates=["date"])
    expected_columns = [
        "date",
        "um_snapshot_count",
        "um_first_offset_seconds",
        "um_last_offset_seconds",
        "um_near_pressure",
        "cm_snapshot_count",
        "cm_first_offset_seconds",
        "cm_last_offset_seconds",
        "cm_near_pressure",
        "source_complete",
    ]
    if frame.columns.tolist() != expected_columns:
        raise RuntimeError("future book panel contains an unexpected or outcome-bearing column")
    expected_dates = pd.date_range("2024-01-01", FULL_CUTOFF, freq="5min", inclusive="left")
    if not frame["date"].equals(pd.Series(expected_dates, name="date")):
        raise RuntimeError("future book panel is not a complete 5m grid")
    if len(frame) != item["rows"] or int(frame["source_complete"].sum()) != item["source_complete_rows"]:
        raise RuntimeError("future book panel counts differ from manifest")
    return manifest, frame


def build_book_score(cfg: SelectionConfig, future: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    shells, credibility, selection_market, selection_funding, source = load_selection_sources(cfg)
    weights = (1.0, 0.5, 0.0, 0.0, 0.0)
    history = pd.DataFrame(
        {
            "date": shells["date"],
            "um_near_pressure": raw_pressure(
                shells, credibility, venue="um", weights=weights, credibility_weighted=False
            ),
            "cm_near_pressure": raw_pressure(
                shells, credibility, venue="cm", weights=weights, credibility_weighted=False
            ),
        }
    )
    history["source_complete"] = history[["um_near_pressure", "cm_near_pressure"]].notna().all(axis=1)
    columns = ["date", "um_near_pressure", "cm_near_pressure", "source_complete"]
    book = pd.concat([history[columns], future[columns]], ignore_index=True)
    expected_dates = pd.date_range("2023-01-01", FULL_CUTOFF, freq="5min", inclusive="left")
    if not book["date"].equals(pd.Series(expected_dates, name="date")):
        raise RuntimeError("combined book panel is not a complete 5m grid")
    venue_scores = []
    for venue in ("um", "cm"):
        raw = book[f"{venue}_near_pressure"].where(book["source_complete"].astype(bool))
        venue_scores.append(
            lagged_robust_zscore(
                raw,
                window=cfg.robust_window_bars,
                minimum=cfg.robust_min_periods,
            )
        )
    score = (venue_scores[0] + venue_scores[1]) / np.sqrt(2.0)
    score = score.where(book["source_complete"].astype(bool) & score.notna())
    return book, score, {
        "selection_market": selection_market,
        "selection_funding": selection_funding,
        "selection_source": source,
    }


def assert_selection_replay(
    manifest: dict[str, Any],
    result: dict[str, Any],
    book: pd.DataFrame,
    score: pd.Series,
    context: dict[str, Any],
    selection_cfg: SelectionConfig,
) -> None:
    market = context["selection_market"]
    if not np.array_equal(book.loc[: len(market) - 1, "date"].to_numpy(), market["date"].to_numpy()):
        raise RuntimeError("selection book/market dates differ")
    fit = market["date"].lt("2023-07-01")
    threshold = float(score.iloc[: len(market)].loc[fit].dropna().abs().quantile(0.985))
    if not np.isclose(threshold, EXPECTED_SELECTED["threshold"], rtol=0.0, atol=1e-15):
        raise RuntimeError("selection threshold did not replay")
    execution_cfg = context["selection_execution_cfg"]
    engine = ExecutionEngine(market, context["selection_funding"], execution_cfg)
    hashes = {}
    for name, (start, end) in SELECTION_WINDOWS.items():
        trades = schedule(
            market,
            engine,
            score.iloc[: len(market)].reset_index(drop=True),
            threshold=threshold,
            hold_bars=288,
            start=start,
            end=end,
        )
        hashes[name] = _schedule_hash(trades)
        expected_stats = result["selected"]["stats"][name]
        actual_stats = equity_stats(trades, start=start, end=end, cfg=execution_cfg)
        if actual_stats != expected_stats:
            raise RuntimeError(f"selection stats did not replay: {name}")
    if hashes != manifest["selected_schedule_hashes"]:
        raise RuntimeError("selection schedule hashes did not replay")


def align_score(score_frame: pd.DataFrame, score: pd.Series, full_dates: pd.Series) -> pd.Series:
    mapped = pd.Series(score.to_numpy(float), index=pd.DatetimeIndex(score_frame["date"]), name="score")
    aligned = mapped.reindex(pd.DatetimeIndex(full_dates))
    if not np.array_equal(
        aligned.loc[(aligned.index >= "2023-01-01") & (aligned.index < FULL_CUTOFF)].index.to_numpy(),
        score_frame["date"].to_numpy(),
    ):
        raise RuntimeError("score alignment lost a book timestamp")
    return aligned.reset_index(drop=True)


def assert_execution_parity(selection_cfg: ExecutionConfig, future_cfg: ExecutionConfig) -> None:
    for field in ("leverage", "fee_rate", "slippage_rate"):
        if not np.isclose(float(getattr(selection_cfg, field)), float(getattr(future_cfg, field))):
            raise RuntimeError(f"candidate execution economics drifted: {field}")
    if resolve_existing(selection_cfg.funding_csv) != resolve_existing(future_cfg.funding_csv):
        raise RuntimeError("candidate funding source drifted")


def correlation_diagnostics(candidate: pd.Series, baseline: pd.Series) -> dict[str, Any]:
    left, right = candidate.align(baseline, join="outer", fill_value=0.0)
    minimum = int(ORTHOGONALITY_LIMITS["minimum_nonzero_pnl_days"])
    candidate_nonzero = int(left.abs().gt(1e-15).sum())
    baseline_nonzero = int(right.abs().gt(1e-15).sum())
    defined = (
        candidate_nonzero >= minimum
        and baseline_nonzero >= minimum
        and float(left.std(ddof=0)) > 0.0
        and float(right.std(ddof=0)) > 0.0
    )
    correlation = pnl_correlation(left, right) if defined else {"pearson": None, "spearman": None}
    return {
        **correlation,
        "defined": defined,
        "candidate_nonzero_days": candidate_nonzero,
        "baseline_nonzero_days": baseline_nonzero,
    }


def compact_stats(trades: list[Trade], *, start: str, end: str, cfg: Any) -> dict[str, Any]:
    return {**equity_stats(trades, start=start, end=end, cfg=cfg), "schedule_hash": _schedule_hash(trades)}


def render_docs(payload: dict[str, Any]) -> str:
    lines = [
        "# Cross-collateral near-pressure frozen OOS",
        "",
        "The 104-cell 2023 search, q0.985 threshold, 288-bar hold, and event clock were hash-frozen "
        "before this evaluator joined the outcome-blind 2024+ book panel to execution prices.",
        "",
        "## Performance",
        "",
        "| Window | Absolute return | CAGR | Strict MDD | CAGR/MDD | Trades | Long/short |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in WINDOWS:
        stats = payload["candidate_stats"][name]
        lines.append(
            f"| {name} | {stats['absolute_return_pct']:.4f}% | {stats['cagr_pct']:.4f}% | "
            f"{stats['strict_mdd_pct']:.4f}% | {stats['cagr_to_strict_mdd']:.4f} | "
            f"{stats['trades']} | {stats['longs']}/{stats['shorts']} |"
        )
    lines.extend(
        [
            "",
            "## Independence from frozen rank-7",
            "",
            "| Window | Exact entry Jaccard | Candidate entries within 6h | Position Jaccard | Daily PnL Pearson | Spearman | Pass |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name in WINDOWS:
        row = payload["orthogonality"][name]
        pearson = row["daily_pnl_correlation"]["pearson"]
        spearman = row["daily_pnl_correlation"]["spearman"]
        pearson_text = f"{pearson:.4f}" if pearson is not None else "undefined"
        spearman_text = f"{spearman:.4f}" if spearman is not None else "undefined"
        lines.append(
            f"| {name} | {row['exact_entry_jaccard']:.4f} | "
            f"{row['candidate_entries_near_6h_fraction']:.4f} | {row['position_jaccard']:.4f} | "
            f"{pearson_text} | {spearman_text} | {row['passes_limits']} |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            "- Feature inputs contain only checksum-verified USD-M/COIN-M bookDepth paths.",
            "- Every robust baseline excludes the current bar; missing source bars fail closed.",
            "- Entry is next-open, cost is 6 bp/side, realized funding is included, and strict MDD includes intratrade extremes.",
            "- CAGR uses the full calendar window, including idle periods.",
            "- Rank-7 frozen hashes/stats and the candidate's complete 2023 schedule replay before OOS is accepted.",
            "",
            "## Verdict",
            "",
            payload["verdict"],
            "",
        ]
    )
    return "\n".join(lines)


def run(cfg: Config) -> dict[str, Any]:
    selection_manifest, selection_result = validate_selection_manifest(cfg.selection_manifest)
    future_manifest, future_book = validate_future_book_manifest(cfg.future_book_manifest)
    selection_cfg = SelectionConfig(output="/tmp/no_write_selection.json", manifest_output="/tmp/no_write_manifest.json", docs_output="")
    book, score, selection_context = build_book_score(selection_cfg, future_book)

    rank7 = build_rank7_context(
        Rank7AuditConfig(
            input_csv="data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz",
            funding_csv=selection_cfg.funding_csv,
            premium_csv=(
                "data/binance_um_aux_btc_2020_2026/"
                "BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
            ),
            output="/tmp/no_write_rank7_audit.json",
            docs_output="",
            exclude_from=FULL_CUTOFF,
        )
    )
    full_market = rank7["base"]["context"]["market"]
    full_dates = pd.to_datetime(full_market["date"])
    score_full = align_score(book, score, full_dates)
    selection_context["selection_execution_cfg"] = ExecutionConfig(
        input_csv="",
        metrics_csv="",
        funding_csv=str(resolve_existing(selection_cfg.funding_csv)),
        output="/tmp/no_write_selection_replay.json",
        manifest_output="/tmp/no_write_selection_replay_manifest.json",
        exclude_from="2024-01-01",
        leverage=0.5,
        fee_rate=0.0005,
        slippage_rate=0.0001,
    )
    assert_execution_parity(
        selection_context["selection_execution_cfg"], rank7["base"]["execution_cfg"]
    )
    assert_selection_replay(
        selection_manifest,
        selection_result,
        book,
        score,
        selection_context,
        selection_cfg,
    )

    candidate_engine = ExecutionEngine(
        full_market,
        rank7["base"]["context"]["funding"],
        rank7["base"]["execution_cfg"],
    )
    candidate_stats: dict[str, dict[str, Any]] = {}
    rank7_stats: dict[str, dict[str, Any]] = {}
    orthogonality: dict[str, dict[str, Any]] = {}
    for name, (start, end) in WINDOWS.items():
        candidate_trades = schedule(
            full_market,
            candidate_engine,
            score_full,
            threshold=float(EXPECTED_SELECTED["threshold"]),
            hold_bars=int(EXPECTED_SELECTED["hold_bars"]),
            start=start,
            end=end,
        )
        baseline_trades = rank7_schedule(rank7, start=start, end=end)
        candidate_stats[name] = compact_stats(
            candidate_trades, start=start, end=end, cfg=rank7["base"]["execution_cfg"]
        )
        rank7_stats[name] = compact_stats(
            baseline_trades, start=start, end=end, cfg=rank7["base"]["execution_cfg"]
        )
        overlap = trade_timing_overlap(
            candidate_trades, baseline_trades, total_bars=len(full_market), near_bars=72
        )
        candidate_daily = daily_marked_returns(
            full_market,
            rank7["base"]["context"]["funding"],
            candidate_trades,
            rank7["base"]["execution_cfg"],
            start=start,
            end=end,
        )
        baseline_daily = daily_marked_returns(
            full_market,
            rank7["base"]["context"]["funding"],
            baseline_trades,
            rank7["base"]["execution_cfg"],
            start=start,
            end=end,
        )
        correlation = correlation_diagnostics(candidate_daily, baseline_daily)
        checks = {
            "exact_entry_jaccard": overlap["exact_entry_jaccard"] <= ORTHOGONALITY_LIMITS["exact_entry_jaccard"],
            "candidate_entries_near_6h_fraction": overlap["candidate_entries_near_6h_fraction"] <= ORTHOGONALITY_LIMITS["candidate_entries_near_6h_fraction"],
            "position_jaccard": overlap["position_jaccard"] <= ORTHOGONALITY_LIMITS["position_jaccard"],
            "daily_pnl_correlation_defined": correlation["defined"],
            "absolute_daily_pnl_pearson": (
                correlation["defined"]
                and abs(float(correlation["pearson"]))
                <= ORTHOGONALITY_LIMITS["absolute_daily_pnl_pearson"]
            ),
        }
        orthogonality[name] = {
            **overlap,
            "daily_pnl_correlation": correlation,
            "checks": checks,
            "passes_limits": all(checks.values()),
        }

    combined = candidate_stats["oos_2024_2026h1"]
    performance_pass = (
        combined["cagr_to_strict_mdd"] >= 3.0
        and combined["trades"] >= 60
        and all(candidate_stats[name]["absolute_return_pct"] > 0.0 for name in ("test_2024", "eval_2025", "holdout_2026h1"))
    )
    independence_pass = all(row["passes_limits"] for row in orthogonality.values())
    verdict = (
        "Promoted as a low-correlation alpha candidate; retain frozen parameters and require forward shadow before live sizing."
        if performance_pass and independence_pass
        else "Rejected for live promotion; retain only as research evidence because the frozen OOS performance or independence gate failed."
    )
    payload = {
        "schema_version": 1,
        "mode": "cross_collateral_near_pressure_frozen_oos",
        "selection_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
        "future_book_manifest_sha256": EXPECTED_FUTURE_BOOK_MANIFEST_SHA256,
        "future_policy_oos": True,
        "future_did_not_rerank": True,
        "selected_spec": EXPECTED_SELECTED,
        "source": {"future_book": future_manifest["file"], "selection": selection_context["selection_source"]},
        "orthogonality_limits": ORTHOGONALITY_LIMITS,
        "candidate_stats": candidate_stats,
        "rank7_stats": rank7_stats,
        "orthogonality": orthogonality,
        "performance_pass": performance_pass,
        "independence_pass": independence_pass,
        "verdict": verdict,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if cfg.docs_output:
        docs = Path(cfg.docs_output)
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text(render_docs(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", default=SELECTION_MANIFEST)
    parser.add_argument("--future-book-manifest", default=FUTURE_BOOK_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-output", default=DEFAULT_DOCS)
    return parser.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(
        json.dumps(
            {
                "candidate_stats": payload["candidate_stats"],
                "orthogonality": payload["orthogonality"],
                "verdict": payload["verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
