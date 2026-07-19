from __future__ import annotations

import copy

import pytest

from training import preregister_coinm_next_maturity_shock_relay as p


def test_manifest_is_frozen_and_outcome_blind() -> None:
    report = p.build_manifest()
    p.validate_manifest(report)
    assert report["outcomes_opened"] is False
    assert report["policy"]["policy_id"] == "CMSR-36"
    assert report["policy"]["hold_bars"] == 36
    assert "BTCUSDT_execution_OHLC" in report["causal_feature_contract"][
        "forbidden_feature_columns"
    ]


def test_selected_cell_is_incidence_only_and_singleton() -> None:
    report = p.build_manifest()
    grid = report["research_history_boundary"]["incidence_grid"]
    assert grid["cells"] == 27
    assert grid["selected"] == [0.90, 0.80, 0.80]
    assert report["selection_protocol"]["candidate_count"] == 1


def test_policy_mutation_is_rejected() -> None:
    report = p.build_manifest()
    changed = copy.deepcopy(report)
    changed["policy"]["hold_bars"] = 12
    changed["manifest_hash"] = p.canonical_hash(
        {key: value for key, value in changed.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="policy changed"):
        p.validate_manifest(changed, verify_sources=False)


def test_open_outcome_is_rejected() -> None:
    report = p.build_manifest()
    changed = copy.deepcopy(report)
    changed["outcomes_opened"] = True
    changed["manifest_hash"] = p.canonical_hash(
        {key: value for key, value in changed.items() if key != "manifest_hash"}
    )
    with pytest.raises(ValueError, match="opened outcomes"):
        p.validate_manifest(changed, verify_sources=False)
