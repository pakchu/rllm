#!/usr/bin/env python3
"""Materialize historical barrier-clock trades for Fresh Kimchi and annual Rank7.

This exporter is intentionally narrow: it does not optimize weights and does not
open post-2026-06-01 sources.  It rebuilds the already audited Fresh Kimchi/FX
candidate plus the frozen annual-refit Rank7 context, then writes exact scheduled
trade clocks for 2024, 2025, and the physically available 2026 prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.audit_fresh_kimchi_orthogonal_alpha import (  # noqa: E402
    CANDIDATE_SPEC as FRESH_CANDIDATE_SPEC,
    DEFAULT_FUNDING,
    DEFAULT_INPUT,
    DEFAULT_PREMIUM,
    Config as FreshAuditConfig,
    build_candidate_context,
    build_rank7_context,
    candidate_schedule,
    compact_stats,
    rank7_schedule,
)
from training.select_expanding_extratrees_top10_pre2025 import _action as rank7_action  # noqa: E402
from training.evaluate_expanding_extratrees_top10_oos import FULL_CUTOFF  # noqa: E402
from training.search_funding_premium_external_state_gate_alpha import _frame_hash  # noqa: E402
from training.search_inventory_purge_reclaim_alpha import Trade  # noqa: E402

DEFAULT_OUTPUT = "research/g9_historical_barriers/report.json"
DEFAULT_TRADES_DIR = "research/g9_historical_barriers/trades"
DEFAULT_DOCS = "research/g9_historical_barriers/README.md"

STATIC_WINDOWS: dict[str, tuple[str, str]] = {
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
}
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "2024": {"fresh_kimchi_fx": 30, "frozen_annual_rank7": 22},
    "2025": {"fresh_kimchi_fx": 17, "frozen_annual_rank7": 21},
    "2026_prefix": {"fresh_kimchi_fx": 28, "frozen_annual_rank7": 12},
}


@dataclass(frozen=True)
class Config:
    input_csv: str = DEFAULT_INPUT
    funding_csv: str = DEFAULT_FUNDING
    premium_csv: str = DEFAULT_PREMIUM
    output: str = DEFAULT_OUTPUT
    trades_dir: str = DEFAULT_TRADES_DIR
    docs_output: str = DEFAULT_DOCS
    exclude_from: str = FULL_CUTOFF


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _market_coverage(market: pd.DataFrame) -> dict[str, Any]:
    dates = pd.DatetimeIndex(pd.to_datetime(market["date"]))
    if dates.empty:
        raise RuntimeError("empty market source")
    deltas = dates.to_series().diff().dropna()
    complete_5m = bool(len(deltas) == 0 or deltas.eq(pd.Timedelta(minutes=5)).all())
    data_end_exclusive = dates.max() + pd.Timedelta(minutes=5)
    return {
        "first_bar": str(dates.min()),
        "last_bar": str(dates.max()),
        "data_end_exclusive": str(data_end_exclusive),
        "rows": int(len(dates)),
        "complete_5m_grid": complete_5m,
        "bar_interval_minutes": 5,
    }


def _source_snapshot(cfg: Config, candidate: dict[str, Any], rank7: dict[str, Any]) -> dict[str, Any]:
    market = candidate["market"]
    rank7_market = rank7["base"]["context"]["market"]
    funding = candidate["funding"]
    coverage = _market_coverage(market)
    paths = {
        "market_csv": str(Path(cfg.input_csv)),
        "funding_csv": str(Path(cfg.funding_csv)),
        "premium_csv": str(Path(cfg.premium_csv)),
    }
    resolved = {k: str(Path(v).resolve()) for k, v in paths.items() if Path(v).exists()}
    source_files = {
        k: {"path": v, "sha256": _file_sha256(v), "bytes": int(Path(v).stat().st_size)}
        for k, v in resolved.items()
    }
    frame_identity = {
        "candidate_market_hash": _frame_hash(market),
        "rank7_market_hash": _frame_hash(rank7_market),
        "funding_hash": _frame_hash(funding),
        "market_grids_equal": bool(np.array_equal(pd.to_datetime(market["date"]), pd.to_datetime(rank7_market["date"]))),
        "ohlc_equal": all(np.array_equal(market[col].to_numpy(float), rank7_market[col].to_numpy(float)) for col in ("open", "high", "low", "close")),
    }
    if not frame_identity["market_grids_equal"] or not frame_identity["ohlc_equal"]:
        raise RuntimeError("Fresh Kimchi and Rank7 contexts do not share the same market grid")
    return {
        "paths_requested": paths,
        "source_files": source_files,
        "coverage": coverage,
        "frame_identity": frame_identity,
        "exclude_from_requested": cfg.exclude_from,
        "physical_2026_prefix_end_exclusive": coverage["data_end_exclusive"],
    }


def _exit_kind(trade: Trade, hold_bars: int) -> str:
    return "open" if int(trade.exit_position) == int(trade.entry_position) + int(hold_bars) else "barrier"


def _exit_price(market: pd.DataFrame, trade: Trade, *, hold_bars: int, take_bps: int, stop_bps: int) -> float:
    opens = pd.to_numeric(market["open"], errors="raise").to_numpy(float)
    entry_price = float(opens[int(trade.entry_position)])
    if _exit_kind(trade, hold_bars) == "open":
        return float(opens[int(trade.exit_position)])
    if float(trade.gross_return) < 0.0:
        stop = float(stop_bps) / 10_000.0
        return float(entry_price * (1.0 - stop)) if int(trade.side) > 0 else float(entry_price * (1.0 + stop))
    take = float(take_bps) / 10_000.0
    return float(entry_price * (1.0 + take)) if int(trade.side) > 0 else float(entry_price * (1.0 - take))


def _trade_records(
    market: pd.DataFrame,
    trades: Iterable[Trade],
    *,
    sleeve: str,
    window: str,
    hold_bars: int | None = None,
    take_bps: int | None = None,
    stop_bps: int | None = None,
    funding_leg: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    dates = pd.DatetimeIndex(pd.to_datetime(market["date"]))
    opens = pd.to_numeric(market["open"], errors="raise").to_numpy(float)
    out: list[dict[str, Any]] = []
    for ordinal, trade in enumerate(trades, start=1):
        side = int(trade.side)
        if side not in (-1, 1):
            raise RuntimeError(f"{sleeve} trade has non-normalized side: {trade.side!r}")
        if funding_leg is not None:
            hold, take, stop = rank7_action(bool(funding_leg[int(trade.signal_position)]))
            rank7_source = "funding_exit" if bool(funding_leg[int(trade.signal_position)]) else "premium_exit"
        else:
            if hold_bars is None or take_bps is None or stop_bps is None:
                raise ValueError("static hold/take/stop required without funding_leg")
            hold, take, stop = int(hold_bars), int(take_bps), int(stop_bps)
            rank7_source = None
        record = {
            "window": window,
            "sleeve": sleeve,
            "ordinal": ordinal,
            "signal_position": int(trade.signal_position),
            "entry_position": int(trade.entry_position),
            "exit_position": int(trade.exit_position),
            "signal_date": str(dates[int(trade.signal_position)]),
            "entry_date": str(dates[int(trade.entry_position)]),
            "exit_date": str(dates[int(trade.exit_position)]),
            "side": side,
            "side_label": "long" if side > 0 else "short",
            "entry_price": float(opens[int(trade.entry_position)]),
            "exit_price": _exit_price(market, trade, hold_bars=hold, take_bps=take, stop_bps=stop),
            "exit_kind": _exit_kind(trade, hold),
            "hold_bars": int(hold),
            "take_bps": int(take),
            "stop_bps": int(stop),
            "actual_hold_bars": int(trade.exit_position) - int(trade.entry_position),
            "gross_return": float(trade.gross_return),
            "price_factor": float(trade.price_factor),
            "funding_factor": float(trade.funding_factor),
            "funding_debit_factor": float(trade.funding_debit_factor),
            "favorable_price_factor": float(trade.favorable_price_factor),
            "adverse_price_factor": float(trade.adverse_price_factor),
        }
        if rank7_source is not None:
            record["rank7_action_source"] = rank7_source
        out.append(record)
    return out


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    return {"path": str(path), "rows": len(rows), "sha256": _file_sha256(path)}


def _fit_receipts(rank7: dict[str, Any]) -> list[dict[str, Any]]:
    receipts = []
    for fold in rank7["folds"]:
        receipts.append(
            {
                "name": str(fold["name"]),
                "start": str(fold["start"]),
                "end": str(fold["end"]),
                "fit_examples": int(fold["fit_examples"]),
                "predict_events": int(fold["predict_events"]),
                "latest_fit_exit": str(fold["latest_fit_exit"]),
                "fit_mask_hash": _safe_payload_hash(np.flatnonzero(np.asarray(fold["fit"], dtype=bool)).tolist()),
                "predict_mask_hash": _safe_payload_hash(np.flatnonzero(np.asarray(fold["predict"], dtype=bool)).tolist()),
            }
        )
    return receipts


def _window_map(candidate: dict[str, Any]) -> dict[str, tuple[str, str]]:
    coverage = _market_coverage(candidate["market"])
    return {
        **STATIC_WINDOWS,
        "2026_prefix": ("2026-01-01", coverage["data_end_exclusive"]),
    }


def _render_docs(payload: dict[str, Any]) -> str:
    lines = [
        "# G9 historical barrier clocks",
        "",
        "Frozen historical trade-clock export for Fresh Kimchi/FX and annual-refit Rank7.",
        "No portfolio weights are optimized in this artifact.",
        "",
        f"Market coverage: `{payload['source_snapshot']['coverage']['first_bar']}` through "
        f"`{payload['source_snapshot']['coverage']['last_bar']}` "
        f"(end-exclusive `{payload['source_snapshot']['coverage']['data_end_exclusive']}`).",
        "",
        "| Window | Sleeve | Trades | Long/Short | Return | MDD | Hash | JSONL |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for window, row in payload["windows"].items():
        for sleeve in ("fresh_kimchi_fx", "frozen_annual_rank7"):
            stats = row["sleeves"][sleeve]["stats"]
            artifact = row["sleeves"][sleeve]["artifact"]
            lines.append(
                f"| {window} | {sleeve} | {stats['trades']} | {stats['longs']}/{stats['shorts']} | "
                f"{stats['absolute_return_pct']:.4f}% | {stats['strict_mdd_pct']:.4f}% | "
                f"`{stats['schedule_hash'][:12]}` | `{artifact['path']}` |"
            )
    lines.extend([
        "",
        "## Integrity",
        "",
        f"- Annual Rank7 frozen-prefix verification passed: `{payload['integrity']['rank7_frozen_prefix_verification_passed']}`.",
        f"- Rank7 annual-reference verification passed: `{payload['integrity']['rank7_annual_reference_verification_passed']}`.",
        f"- Market frame identity passed: `{payload['source_snapshot']['frame_identity']['market_grids_equal'] and payload['source_snapshot']['frame_identity']['ohlc_equal']}`.",
        "- Trade `side` is numeric normalized (`1` long, `-1` short); `side_label` is auxiliary only.",
        "- `exit_kind=open` means max-hold cap at next open; `exit_kind=barrier` means take/stop barrier fill with stop-first execution convention.",
        "",
    ])
    return "\n".join(lines)


def run(cfg: Config) -> dict[str, Any]:
    if cfg.exclude_from != FULL_CUTOFF:
        raise ValueError(f"exclude_from must equal frozen cutoff {FULL_CUTOFF}")
    audit_cfg = FreshAuditConfig(
        input_csv=cfg.input_csv,
        funding_csv=cfg.funding_csv,
        premium_csv=cfg.premium_csv,
        output="/tmp/no_write_g9_historical_barriers.json",
        docs_output="",
        exclude_from=cfg.exclude_from,
    )
    candidate = build_candidate_context(audit_cfg)
    rank7 = build_rank7_context(audit_cfg)
    market = candidate["market"]
    rank7_market = rank7["base"]["context"]["market"]
    source_snapshot = _source_snapshot(cfg, candidate, rank7)
    windows_def = _window_map(candidate)
    trades_dir = Path(cfg.trades_dir)
    funding_leg = np.asarray(rank7["base"]["context"]["funding_leg"], dtype=bool)
    windows: dict[str, Any] = {}
    all_schedule_hash_records: dict[str, Any] = {}
    for name, (start, end) in windows_def.items():
        fresh_trades = candidate_schedule(candidate, start=start, end=end)
        rank7_trades = rank7_schedule(rank7, start=start, end=end)
        fresh_rows = _trade_records(
            market,
            fresh_trades,
            sleeve="fresh_kimchi_fx",
            window=name,
            hold_bars=int(FRESH_CANDIDATE_SPEC["hold_bars"]),
            take_bps=int(FRESH_CANDIDATE_SPEC["take_bps"]),
            stop_bps=int(FRESH_CANDIDATE_SPEC["stop_bps"]),
        )
        rank7_rows = _trade_records(
            rank7_market,
            rank7_trades,
            sleeve="frozen_annual_rank7",
            window=name,
            funding_leg=funding_leg,
        )
        expected = EXPECTED_COUNTS.get(name)
        if expected is not None:
            actual_counts = {"fresh_kimchi_fx": len(fresh_rows), "frozen_annual_rank7": len(rank7_rows)}
            if actual_counts != expected:
                raise RuntimeError(f"{name} trade counts drifted: {actual_counts} != {expected}")
        fresh_artifact = _write_jsonl(trades_dir / f"{name}__fresh_kimchi_fx.jsonl", fresh_rows)
        rank7_artifact = _write_jsonl(trades_dir / f"{name}__frozen_annual_rank7.jsonl", rank7_rows)
        fresh_stats = compact_stats(fresh_trades, start=start, end=end, cfg=candidate["execution_cfg"])
        rank7_stats = compact_stats(rank7_trades, start=start, end=end, cfg=rank7["base"]["execution_cfg"])
        windows[name] = {
            "start": start,
            "end_exclusive": end,
            "sleeves": {
                "fresh_kimchi_fx": {"stats": fresh_stats, "artifact": fresh_artifact, "trades": fresh_rows},
                "frozen_annual_rank7": {"stats": rank7_stats, "artifact": rank7_artifact, "trades": rank7_rows},
            },
        }
        all_schedule_hash_records[name] = {
            "fresh_kimchi_fx": fresh_stats["schedule_hash"],
            "frozen_annual_rank7": rank7_stats["schedule_hash"],
        }
    payload = {
        "schema_version": 1,
        "mode": "g9_historical_barrier_clock_export",
        "config": asdict(cfg),
        "windows": windows,
        "source_snapshot": source_snapshot,
        "fresh_kimchi_candidate_spec": FRESH_CANDIDATE_SPEC,
        "rank7": {
            "refit_cadence": "annual",
            "fit_receipts": _fit_receipts(rank7),
            "reference_stats": rank7["reference_stats"],
            "reference_schedule_hashes": rank7["reference_hashes"],
        },
        "integrity": {
            "rank7_frozen_prefix_verification_passed": True,
            "rank7_annual_reference_verification_passed": True,
            "freshness_fail_closed_diagnostics": candidate["freshness"],
            "expected_counts": EXPECTED_COUNTS,
            "expected_counts_verification_passed": True,
            "trade_side_numeric_normalized": True,
            "schedule_hashes": all_schedule_hash_records,
            "report_hash_excluding_self": "",
        },
    }
    payload["integrity"]["report_hash_excluding_self"] = _safe_payload_hash(payload)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    if cfg.docs_output:
        docs = Path(cfg.docs_output)
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text(_render_docs(payload), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", default=DEFAULT_INPUT)
    p.add_argument("--funding-csv", default=DEFAULT_FUNDING)
    p.add_argument("--premium-csv", default=DEFAULT_PREMIUM)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--trades-dir", default=DEFAULT_TRADES_DIR)
    p.add_argument("--docs-output", default=DEFAULT_DOCS)
    p.add_argument("--exclude-from", default=FULL_CUTOFF)
    return p.parse_args()


def main() -> None:
    payload = run(Config(**vars(parse_args())))
    print(json.dumps({
        "output": payload["config"]["output"],
        "coverage": payload["source_snapshot"]["coverage"],
        "schedule_hashes": payload["integrity"]["schedule_hashes"],
        "passed": True,
    }, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
