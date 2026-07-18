"""Freeze the QQQ/KODEX200/GLD alpha-transfer battery before outcomes.

The current gross-8 sleeves are crypto-native.  This preregistration therefore
does not claim an exact strategy port.  It freezes three daily OHLCV-only
translations of previously researched price-action mechanisms and the strict
rules used to decide whether any mechanism transfers across markets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OUTPUT = "results/cross_asset_alpha_transfer_preregistration_2026-07-19.json"
DOCS_OUTPUT = "docs/cross-asset-alpha-transfer-preregistration-2026-07-19.md"


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "research_id": "CAT-XA-1",
        "title": "QQQ/KODEX200/GLD daily alpha-transfer battery",
        "status": "preregistered_before_outcome_download",
        "scope": {
            "exact_port_of_gross8": False,
            "claim_boundary": (
                "The test concerns translated price/volume mechanisms only. A positive result does not "
                "validate the crypto-native gross-8 sleeve or its BTC-fitted thresholds."
            ),
            "instruments": {
                "QQQ": {
                    "name": "Invesco QQQ Trust",
                    "role": "tradable Nasdaq-100 proxy",
                    "currency": "USD",
                },
                "069500.KS": {
                    "name": "KODEX 200",
                    "role": "tradable KOSPI 200 proxy",
                    "currency": "KRW",
                },
                "GLD": {
                    "name": "SPDR Gold Shares",
                    "role": "tradable gold proxy",
                    "currency": "USD",
                },
            },
        },
        "gross8_portability_audit": {
            "fresh_kimchi_fx": {
                "exact_portable": False,
                "blocking_inputs": ["BTC funding", "Kimchi premium", "USDKRW", "BTC flow"],
            },
            "frozen_annual_rank7": {
                "exact_portable": False,
                "blocking_inputs": ["BTC funding/premium event clock", "BTC-fitted 40-feature model"],
            },
            "rex_taker_low_range_position": {
                "exact_portable": False,
                "blocking_inputs": ["crypto aggressor-side taker imbalance", "BTC-fitted threshold"],
            },
            "cand_rex_veto_7": {
                "exact_portable": False,
                "blocking_inputs": ["BTC open interest", "BTC-fitted threshold"],
            },
            "markov_transition_long": {
                "exact_portable": False,
                "blocking_inputs": ["BTC funding/premium event clock", "BTC-fitted transition states"],
            },
        },
        "source_contract": {
            "provider": "Yahoo Finance chart API",
            "endpoint_template": "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            "period_start_utc": "2000-01-01T00:00:00Z",
            "period_end_exclusive_utc": "2026-07-19T00:00:00Z",
            "interval": "1d",
            "events": "div,splits",
            "include_adjusted_close": True,
            "adjustment": (
                "For each row multiply raw open/high/low/close by adjclose/close. Drop rows with missing "
                "or nonpositive OHLC; retain provider volume and record source SHA256."
            ),
            "cache_policy": "raw response is local-only; commit hashes and normalized row metadata, not provider payload",
        },
        "calendar_contract": {
            "train": ["2007-01-29", "2017-01-01"],
            "test": ["2017-01-01", "2022-01-01"],
            "eval": ["2022-01-01", "2026-07-19"],
            "threshold_fit": "per instrument, train only",
            "test_and_eval_use": "report only; no parameter selection, reranking, sign repair, or substitution",
            "cagr_denominator": "full wall-clock split including idle periods",
        },
        "cadence_translation": {
            "source_bar": "BTC 5-minute continuous bar",
            "target_bar": "one completed exchange session",
            "minutes_per_target_session": 390,
            "rounding": "max(2, round(source_bars*5/390)) for lookbacks; max(1, round(...)) for holds",
            "frozen_session_windows": [2, 7, 26, 111],
            "frozen_trend_sessions": [4, 12, 20],
        },
        "policies": {
            "rex_pullback_reclaim_session": {
                "source": "training/event_candidate_pool_probe.py::_feature_candidates/rex_htf_pullback_reclaim",
                "lookbacks": [2, 7, 26, 111],
                "trend_sessions": [4, 12, 20],
                "local_trend_sessions": 1,
                "volume_zscore_sessions": 20,
                "strength_quantile": 0.75,
                "formula": (
                    "same multiscale location/pullback/reclaim algebra as REX, but vol_confirm uses only "
                    "max(0, volume_zscore); no taker, OI, funding, premium, FX, or Kimchi input"
                ),
                "side": "sign(r4+r12+r20)",
                "hold_sessions": 2,
            },
            "rex_multiscale_extreme_fade_session": {
                "source": "training/event_candidate_pool_probe.py::_feature_candidates/rex_multiscale_extreme_fade",
                "lookbacks": [2, 7, 26, 111],
                "strength_quantile": 0.75,
                "formula": "max(0,abs(mean_range_pos)-0.55)*(1+abs(short_location-long_location))",
                "side": "-sign(mean_range_pos)",
                "hold_sessions": 2,
            },
            "persistent_barrier_mass_density_fade_session": {
                "source": "training/search_persistent_barrier_annihilation_alpha.py",
                "source_frozen_row": {
                    "horizon_5m_bars": 2016,
                    "variant": "mass_density",
                    "tail_quantile": 0.975,
                    "hold_5m_bars": 288,
                    "direction_mode": "fade",
                },
                "lookback_sessions": 26,
                "scale_sessions": 26,
                "scale_min_periods": 13,
                "minimum_prominence_z": 0.5,
                "strength_quantile": 0.975,
                "side": "fade direction of completed-session barrier traversal",
                "hold_sessions": 4,
            },
        },
        "execution": {
            "signal_information": "completed session t and strictly prior rolling state only",
            "entry": "next available session open",
            "exit": "open after fixed hold_sessions",
            "positioning": "one position per policy/instrument; skip overlapping signals",
            "leverage": 1.0,
            "base_cost_bps_per_side": 5.0,
            "stress_cost_bps_per_side": 10.0,
            "strict_mdd": (
                "includes entry cost, every held session high/low, conservative favorable-then-adverse "
                "same-session ordering, exit cost, cash/idle history, and forced split-contained exits"
            ),
        },
        "support_gates": {
            "minimum_train_events_before_nonoverlap": 40,
            "minimum_eval_trades": 20,
            "minimum_years_with_eval_trade": 3,
            "fail_closed_on_missing_or_duplicate_sessions": True,
        },
        "transfer_decision": {
            "primary_instruments": ["QQQ", "069500.KS", "GLD"],
            "all_required": True,
            "per_instrument_eval": {
                "absolute_return_pct": "> 0",
                "cagr_to_strict_mdd": ">= 3.0",
                "strict_mdd_pct": "<= 15.0",
                "trades": ">= 20",
                "stress_10bp_absolute_return_pct": "> 0",
                "positive_eval_calendar_year_share": ">= 0.60",
            },
            "mechanism_control": "same-clock direction flip must have lower eval CAGR/MDD on every instrument",
            "decision": (
                "A policy transfers only if one frozen policy satisfies every gate on all three instruments. "
                "No best-of-three promotion after outcomes."
            ),
        },
        "sealed_outcome_fields": [
            "entry_price",
            "exit_price",
            "post_signal_return",
            "trade_pnl",
            "equity",
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
    lines = [
        "# Cross-asset alpha transfer preregistration — 2026-07-19",
        "",
        f"Manifest: `{payload['manifest_hash']}`",
        "",
        "## Claim boundary",
        "",
        "The gross-8 sleeves cannot be ported exactly because every sleeve uses at least one crypto-native input. ",
        "This battery tests three fixed daily OHLCV translations only; it does not relabel them as the original BTC strategies.",
        "",
        "## Instruments and splits",
        "",
        "- QQQ (Nasdaq-100 ETF), 069500.KS (KODEX 200), GLD (gold ETF).",
        "- Train: 2007-01-29 through 2016-12-31.",
        "- Test: 2017-01-01 through 2021-12-31.",
        "- Eval: 2022-01-01 through 2026-07-18.",
        "- Thresholds are fit per instrument on train only. Test/eval cannot select or repair a policy.",
        "",
        "## Frozen translated policies",
        "",
    ]
    for name, spec in payload["policies"].items():
        lines.append(f"- `{name}`: {spec['formula'] if 'formula' in spec else spec['source_frozen_row']}")
    lines += [
        "",
        "## Admission",
        "",
        "One policy must independently pass all QQQ/KODEX200/GLD eval gates: positive base and 10 bp/side stress return, ",
        "CAGR/strict-MDD >= 3, strict MDD <= 15%, at least 20 trades, >=60% positive eval years, and a weaker direction flip.",
        "",
        "No outcome field was downloaded or inspected when this manifest was generated.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(output: str = OUTPUT, docs_output: str = DOCS_OUTPUT) -> dict[str, Any]:
    payload = manifest()
    out = Path(output)
    docs = Path(docs_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    docs.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    rendered = render_docs(payload)
    for path, content in ((out, encoded), (docs, rendered)):
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f"refusing to overwrite mismatched frozen artifact: {path}")
        path.write_text(content)
    return payload


def main() -> None:
    payload = write_outputs()
    print(json.dumps({"output": OUTPUT, "docs": DOCS_OUTPUT, "manifest_hash": payload["manifest_hash"]}, indent=2))


if __name__ == "__main__":
    main()
