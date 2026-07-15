from __future__ import annotations

import json

from training import sync_feature_tier_pools as sync


def test_orphaned_view_entries_detects_history_that_sync_would_delete(tmp_path, monkeypatch):
    alpha = tmp_path / "alpha.json"
    beta = tmp_path / "beta.json"
    gamma = tmp_path / "gamma.json"
    alpha.write_text(json.dumps({"entries": [{"id": "kept"}]}))
    beta.write_text(json.dumps({"entries": [{"id": "view-only"}]}))
    gamma.write_text(json.dumps({"entries": []}))
    monkeypatch.setattr(
        sync,
        "TIER_FILES",
        {"alpha_feature": alpha, "beta_feature": beta, "gamma_feature": gamma},
    )

    orphaned = sync.orphaned_view_entries({"entries": [{"id": "kept"}]})

    assert orphaned == {"beta_feature": ["view-only"]}


def test_orphaned_view_entries_accepts_materialized_subset(tmp_path, monkeypatch):
    beta = tmp_path / "beta.json"
    beta.write_text(json.dumps({"entries": [{"id": "beta"}]}))
    monkeypatch.setattr(sync, "TIER_FILES", {"beta_feature": beta})

    assert sync.orphaned_view_entries(
        {"entries": [{"id": "alpha"}, {"id": "beta"}]}
    ) == {}
