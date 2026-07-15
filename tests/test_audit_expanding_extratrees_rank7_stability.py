from __future__ import annotations

from training.audit_expanding_extratrees_rank7_stability import (
    FOLDS,
    pass_fail,
    render_md,
    validate_frozen_spec,
)
from training.evaluate_expanding_extratrees_top10_oos import EXPECTED_MANIFEST_HASH


def _stats() -> dict[str, dict[str, float | int]]:
    stats = {
        name: {
            "absolute_return_pct": 10.0,
            "cagr_pct": 12.0,
            "strict_mdd_pct": 4.0,
            "cagr_to_strict_mdd": 3.0,
            "trades": 8 if name == "2026h1" else 15,
        }
        for name, _, _ in FOLDS
    }
    stats["all"] = {
        "absolute_return_pct": 50.0,
        "cagr_pct": 14.0,
        "strict_mdd_pct": 4.0,
        "cagr_to_strict_mdd": 3.5,
        "trades": 60,
    }
    return stats


def test_rank7_spec_is_pinned_to_frozen_manifest() -> None:
    assert validate_frozen_spec() == EXPECTED_MANIFEST_HASH


def test_stability_pass_rule_covers_every_period() -> None:
    stats = _stats()
    assert pass_fail(stats) == (True, [])
    stats["2025"]["cagr_to_strict_mdd"] = 2.99
    passed, reasons = pass_fail(stats)
    assert not passed
    assert reasons == ["2025:ratio_lt_3"]


def test_docs_report_frozen_risk_quantile() -> None:
    payload = {
        "manifest_hash": EXPECTED_MANIFEST_HASH,
        "individual_pass_count": 0,
        "individuals": [],
        "ensembles": {
            str(trees): {
                "pass": True,
                "full_result_hash": "a" * 64,
                "label": f"ensemble5_{trees}",
                "stats": {
                    period: _stats()[period]
                    for period in ("2023", "2024", "2025", "2026h1", "all")
                },
            }
            for trees in (300, 1000, 2000)
        },
        "determinism_checks": [],
    }
    docs = render_md(payload)
    assert "risk_q=.75" in docs
    assert "risk_q=.80" not in docs
