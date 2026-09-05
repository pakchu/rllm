"""Distill the fixed OI-price divergence replay into a disabled shadow config."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import search_meaningful_alpha_combinations as base

ROOT = base.ROOT
SOURCE_CONFIG = ROOT / "configs/live/oi_divergence_pullback_range_rsi_h96_s6_candidate.json"
FRESH = ROOT / "research/oi_divergence_fresh_v2/report.json"
CONFIG = ROOT / "configs/shadow/oi_divergence_pullback_fresh_candidate_2026-09-06.json"
DOC = ROOT / "docs/oi-divergence-pullback-fresh-alpha-2026-09-06.md"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    source = json.loads(SOURCE_CONFIG.read_text())
    fresh = json.loads(FRESH.read_text())
    base_report = fresh["reports"]["0.0006"]
    stress_report = fresh["reports"]["0.001"]
    if fresh["live_enabled"] or source["allow_live_orders"]:
        raise RuntimeError("Source candidate must remain live-disabled")
    if fresh["scheduled_nonoverlap_trades"] != 8:
        raise RuntimeError("Frozen recent schedule changed")
    if min(base_report["absolute_return_pct"], stress_report["absolute_return_pct"]) <= 0:
        raise RuntimeError("Candidate no longer passes recent cost stress")

    signal = source["signal"]
    config = {
        "id": "oi_divergence_pullback_range_rsi_h96_s6_fresh_v1",
        "status": "shadow_candidate",
        "enabled": False,
        "live_authorized": False,
        "research_only": True,
        "symbol": source["symbol"],
        "side": "long",
        "leverage": signal["leverage"],
        "position_overlap_allowed": False,
        "execution": {
            "decision_stride_5m_bars": signal["stride_bars_5m"],
            "entry_delay_5m_bars": signal["entry_delay_bars"],
            "hold_5m_bars": signal["hold_bars_5m"],
            "hold_hours": signal["hold_hours"],
            "entry_price": "next 5m open",
            "exit": "fixed 8h or evaluation boundary",
        },
        "signal": {
            "gates": signal["gates"],
            "feature_definition": signal["feature_definition"],
            "required_current_oi": True,
            "excluded_feature_families": signal["excluded_feature_families"],
        },
        "accounting": {
            "base_cost_per_notional_side": 0.0006,
            "stress_cost_per_notional_side": 0.001,
            "funding": "realized funding events during each position",
            "mdd": "strict in-position 5m adverse excursion",
        },
        "evidence": {
            "historical_validation_from_source": source["validation"],
            "fresh_window": fresh["registration"]["design"]["evaluation_window"],
            "fresh_source_receipt": fresh["source_receipt"],
            "fresh_oi": fresh["oi"],
            "signal_candidates": fresh["signal_candidates"],
            "scheduled_nonoverlap_trades": fresh["scheduled_nonoverlap_trades"],
            "schedule_hash": fresh["schedule_hash"],
            "fresh_base": base_report,
            "fresh_stress": stress_report,
        },
        "risks": [
            "Only eight recent non-overlapping trades; uncertainty is large despite positive cost stress.",
            "The source candidate has weak pre-2024 training performance and a weak original 2026 YTD result.",
            "Historical test/eval periods and thresholds were exposed before this distillation; they are not new OOS evidence.",
            "The fresh authoritative OI source ends at 2026-08-03 13:35 UTC, so later behavior is untested.",
            "Long-only regime concentration can overlap economically with other BTC dip-buying sleeves.",
            "No liquidation, capacity, market-impact, or tick-order model.",
        ],
        "artifacts": {
            str(SOURCE_CONFIG.relative_to(ROOT)): file_hash(SOURCE_CONFIG),
            str(FRESH.relative_to(ROOT)): file_hash(FRESH),
        },
        "implementation": "training/evaluate_oi_divergence_fresh_v2.py:run",
    }
    config["result_hash"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return config


def run() -> None:
    config = build()
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    history = config["evidence"]["historical_validation_from_source"]
    recent = config["evidence"]["fresh_base"]
    stress = config["evidence"]["fresh_stress"]
    DOC.write_text(
        f"""# OI divergence pullback alpha — 2026-09-06

## Decision

**SHADOW/PAPER CANDIDATE; live disabled.** This is a distinct second alpha candidate, but the fresh replay contains only eight trades and does not justify live promotion.

## Frozen formula

Enter long at the next 5-minute open when all four pre-existing gates pass:

- 4-hour OI growth minus price return z-score >= {config['signal']['gates'][0]['threshold']:.9f}
- 48-bar return z-score <= {config['signal']['gates'][1]['threshold']:.9f}
- range volatility >= {config['signal']['gates'][2]['threshold']:.9f}
- normalized RSI <= {config['signal']['gates'][3]['threshold']:.9f}

Evaluate every 30 minutes, prohibit overlapping positions, and hold for 8 hours. The recent replay additionally requires current OI availability.

## Evidence

| Window | Return | strict MDD | CAGR/MDD | Trades | Win rate |
|---|---:|---:|---:|---:|---:|
| Historical 2024 | {history['test_2024']['return_pct']:.2f}% | {history['test_2024']['strict_mdd_pct']:.2f}% | {history['test_2024']['cagr_to_strict_mdd']:.2f} | {history['test_2024']['trades']} | — |
| Historical 2025 | {history['eval_2025']['return_pct']:.2f}% | {history['eval_2025']['strict_mdd_pct']:.2f}% | {history['eval_2025']['cagr_to_strict_mdd']:.2f} | {history['eval_2025']['trades']} | — |
| Original 2026 YTD | {history['ytd_2026']['return_pct']:.2f}% | {history['ytd_2026']['strict_mdd_pct']:.2f}% | {history['ytd_2026']['cagr_to_strict_mdd']:.2f} | {history['ytd_2026']['trades']} | — |
| Fresh Jun 1–Aug 3, 6 bp/side | {recent['absolute_return_pct']:.2f}% | {recent['strict_mdd_pct']:.2f}% | {recent['cagr_to_strict_mdd']:.2f} | {recent['trades']} | {recent['win_rate']:.0%} |
| Fresh Jun 1–Aug 3, 10 bp/side | {stress['absolute_return_pct']:.2f}% | {stress['strict_mdd_pct']:.2f}% | {stress['cagr_to_strict_mdd']:.2f} | {stress['trades']} | {stress['win_rate']:.0%} |

The fresh report reproduced byte-for-byte with SHA-256 `{config['artifacts']['research/oi_divergence_fresh_v2/report.json']}`. Annualized recent ratios are descriptive only because the window and trade count are small.

## Interpretation

The signal buys price pullbacks when open interest remains unusually strong, only in elevated range volatility and weak RSI conditions. It is structurally different from the hourly dollar-flow/regime-switch sleeve, but both can acquire long BTC exposure during stress regimes.

## Risks and next gate

- Eight fresh trades are insufficient for a robust standalone claim.
- Pre-2024 training CAGR/MDD was only {history['train_pre2024']['cagr_to_strict_mdd']:.2f}; the strong 2024/2025 results may reflect selection luck.
- The authoritative OI sample stops on 2026-08-03, so there is no September confirmation.
- Keep disabled and collect forward paper trades without changing thresholds. Reassess only after materially more independent events.
""",
        encoding="utf-8",
    )
    print(json.dumps({"config": str(CONFIG), "doc": str(DOC), "result_hash": config["result_hash"]}, indent=2))


if __name__ == "__main__":
    run()
