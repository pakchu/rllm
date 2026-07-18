"""Freeze the QQQ/KODEX200/GLD five-minute transfer battery before outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OUTPUT = "results/cross_asset_5m_transfer_preregistration_2026-07-19.json"
DOCS_OUTPUT = "docs/cross-asset-5m-transfer-preregistration-2026-07-19.md"


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "research_id": "CAT-XA-5M-1",
        "title": "QQQ/KODEX200/GLD five-minute alpha-transfer battery",
        "status": "preregistered_before_policy_outcomes",
        "claim_boundary": (
            "This is a five-minute OHLCV projection of three previously researched mechanisms. "
            "It is not an exact port of crypto-native gross-8 sleeves and cannot validate inputs "
            "such as taker flow, open interest, funding, premium, FX, or Kimchi premium."
        ),
        "instruments": {
            "QQQ": {
                "name": "Invesco QQQ Trust",
                "investing_id": 651,
                "exchange_timezone": "America/New_York",
                "regular_session": ["09:30", "16:00"],
                "currency": "USD",
            },
            "069500": {
                "name": "KODEX 200",
                "investing_id": 953500,
                "exchange_timezone": "Asia/Seoul",
                "regular_session": ["09:00", "15:20"],
                "currency": "KRW",
            },
            "GLD": {
                "name": "SPDR Gold Shares",
                "investing_id": 9227,
                "exchange_timezone": "America/New_York",
                "regular_session": ["09:30", "16:00"],
                "currency": "USD",
            },
        },
        "source_contract": {
            "primary_provider": "Investing.com TVC chart service",
            "provider_status": "unofficial and unsupported for production",
            "endpoint_template": "https://tvc6.investing.com/{nonce}/0/0/0/0/history",
            "transport": (
                "direct HTTPS first; r.jina.ai read-through transport may be used only when the "
                "provider rate-limits direct research access; archive the extracted provider JSON"
            ),
            "resolution_minutes": 5,
            "period_start_utc": "2024-03-05T00:00:00Z",
            "period_end_exclusive_utc": "2026-07-19T00:00:00Z",
            "chunk_calendar_days": 45,
            "session_filter": "regular exchange session only; timestamp denotes bar open",
            "corporate_action_adjustment": (
                "multiply intraday OHLC by the same-session Yahoo daily adjclose/close factor; "
                "volume is unchanged"
            ),
            "cross_source_control": {
                "provider": "Yahoo Finance chart API",
                "overlap": "latest available 59 calendar days",
                "comparison": "unadjusted close on matching UTC timestamps",
                "minimum_matching_bars": 1000,
                "median_absolute_difference_bps_at_most": 5.0,
                "p95_absolute_difference_bps_at_most": 25.0,
            },
            "cache_policy": "raw/extracted payloads remain local; committed result records hashes",
        },
        "calendar_contract": {
            "warmup": ["2024-03-05", "2024-09-01"],
            "train": ["2024-09-01", "2025-07-01"],
            "test": ["2025-07-01", "2026-01-01"],
            "eval": ["2026-01-01", "2026-07-19"],
            "threshold_fit": "per instrument, train only",
            "test_and_eval_use": "report only; no selection, reranking, sign repair, or substitution",
            "cagr_denominator": "full wall-clock split including idle periods",
        },
        "policies": {
            "rex_htf_pullback_reclaim_5m": {
                "source": "training/event_candidate_pool_probe.py::_feature_candidates/rex_htf_pullback_reclaim",
                "feature_builder": "preprocessing.market_features.build_market_feature_frame",
                "rex_windows_5m": [36, 144, 576, 2016, 8640],
                "missing_crypto_inputs": "taker imbalance is fixed to zero; volume z-score remains",
                "threshold_quantile": 0.75,
                "decision_stride_bars": 12,
                "hold_bars": 144,
                "side": "sign of completed higher-timeframe trend",
            },
            "rex_multiscale_extreme_fade_5m": {
                "source": "training/event_candidate_pool_probe.py::_feature_candidates/rex_multiscale_extreme_fade",
                "feature_builder": "preprocessing.market_features.build_market_feature_frame",
                "rex_windows_5m": [36, 144, 576, 2016, 8640],
                "threshold_quantile": 0.75,
                "decision_stride_bars": 12,
                "hold_bars": 144,
                "side": "fade mean multiscale range location",
            },
            "persistent_barrier_mass_density_fade_5m": {
                "source": "training/search_persistent_barrier_annihilation_alpha.py",
                "horizon_bars": 2016,
                "scale_window_bars": 2016,
                "scale_min_periods": 1008,
                "minimum_prominence_z": 0.5,
                "threshold_quantile": 0.975,
                "decision_clock": "completed bars whose local minute is 00",
                "hold_bars": 288,
                "side": "fade completed traversal through frozen persistence barriers",
            },
        },
        "execution": {
            "signal_information": "completed five-minute bar t and strictly prior rolling state only",
            "entry": "next available regular-session five-minute bar open",
            "exit": "open after fixed tradable hold_bars",
            "positioning": "one position per policy/instrument; skip overlapping signals",
            "leverage": 1.0,
            "base_cost_bps_per_side": 5.0,
            "stress_cost_bps_per_side": 10.0,
            "strict_mdd": (
                "entry cost, every held five-minute high/low, conservative favorable-then-adverse "
                "same-bar ordering, exit cost, idle history, and split-contained exits"
            ),
        },
        "integrity_gates": {
            "duplicate_utc_timestamps": "fail closed",
            "invalid_ohlcv": "fail closed",
            "non_five_minute_alignment": "fail closed",
            "missing_interior_regular_session_bar": (
                "fail closed except provider-stable KRX no-trade/trading-halt intervals; never synthesize bars"
            ),
            "prefix_invariance": True,
            "direction_flip_same_trade_clock": True,
        },
        "support_gates": {
            "minimum_train_positive_strength_rows": 100,
            "minimum_eval_trades": 20,
            "minimum_eval_months_with_trade": 4,
        },
        "transfer_decision": {
            "primary_instruments": ["QQQ", "069500", "GLD"],
            "all_required": True,
            "per_instrument_eval": {
                "absolute_return_pct": "> 0",
                "cagr_to_strict_mdd": ">= 3.0",
                "strict_mdd_pct": "<= 15.0",
                "trades": ">= 20",
                "stress_10bp_absolute_return_pct": "> 0",
                "positive_eval_calendar_month_share": ">= 0.60",
                "weekly_cluster_signflip_pvalue": "<= 0.10",
            },
            "mechanism_control": "same-clock direction flip has lower eval CAGR/MDD on every instrument",
            "decision": "one frozen policy must pass every gate on all three instruments",
        },
        "sealed_outcome_fields": [
            "threshold",
            "entry_price",
            "exit_price",
            "trade_return",
            "equity",
            "absolute_return",
            "cagr",
            "strict_mdd",
            "win_rate",
            "test_outcomes",
            "eval_outcomes",
        ],
    }
    payload["manifest_hash"] = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    return payload


def render_docs(payload: dict[str, Any]) -> str:
    source = payload["source_contract"]
    calendar = payload["calendar_contract"]
    lines = [
        "# Cross-asset five-minute transfer preregistration — 2026-07-19",
        "",
        f"Manifest: `{payload['manifest_hash']}`",
        "",
        "## Scope",
        "",
        payload["claim_boundary"],
        "",
        "- Instruments: QQQ, KODEX 200 (`069500`), and GLD. No KOSPI strategy is evaluated.",
        "- Bar clock: completed regular-session five-minute bars; next-bar-open execution.",
        "- Source is frozen before policy outcomes are computed.",
        "",
        "## Frozen data and splits",
        "",
        f"- Primary research source: {source['primary_provider']} ({source['provider_status']}).",
        f"- Raw period: {source['period_start_utc']} through {source['period_end_exclusive_utc']} exclusive.",
        f"- Train: {calendar['train'][0]} to {calendar['train'][1]} exclusive.",
        f"- Test: {calendar['test'][0]} to {calendar['test'][1]} exclusive.",
        f"- Eval: {calendar['eval'][0]} to {calendar['eval'][1]} exclusive.",
        "- Recent matching timestamps must pass the frozen Yahoo close-parity control.",
        "",
        "## Frozen policies",
        "",
    ]
    for name, spec in payload["policies"].items():
        lines.append(
            f"- `{name}`: q={spec['threshold_quantile']}, hold={spec['hold_bars']} five-minute bars."
        )
    lines += [
        "",
        "## Admission",
        "",
        "A policy transfers only if it independently passes every frozen eval gate on QQQ, KODEX 200, and GLD.",
        "Test/eval cannot tune thresholds, select a row, repair direction, or substitute a policy.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(output: str = OUTPUT, docs_output: str = DOCS_OUTPUT) -> dict[str, Any]:
    payload = manifest()
    out = Path(output)
    docs = Path(docs_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    docs.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    docs.write_text(render_docs(payload))
    return payload


def main() -> None:
    payload = write_outputs()
    print(json.dumps({"output": OUTPUT, "manifest_hash": payload["manifest_hash"]}, indent=2))


if __name__ == "__main__":
    main()
