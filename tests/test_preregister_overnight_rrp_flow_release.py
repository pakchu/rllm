from __future__ import annotations

from training import preregister_overnight_rrp_flow_release as prereg


def test_manifest_is_outcome_blind_and_hash_valid() -> None:
    manifest = prereg.build_manifest()
    prereg.validate_manifest(manifest)
    assert manifest["outcomes_opened"] is False
    assert manifest["causal_feature_contract"]["price_signal_columns"] == []
    assert manifest["policy"]["lower_tail_rank"] == 0.125


def test_source_only_density_satisfies_frozen_support_rule() -> None:
    manifest = prereg.build_manifest()
    counts = manifest["research_history_boundary"]["disclosure"]["primary"]
    assert counts["train"]["events"] >= 100
    assert counts["2021"]["events"] >= 45
    assert counts["2022"]["events"] >= 45
    assert counts["2023"]["events"] >= 60
    assert counts["2023_h1"]["events"] >= 20
    assert counts["2023_h2"]["events"] >= 20
    assert min(counts["train"]["side_counts"].values()) >= 35
    assert min(counts["2023"]["side_counts"].values()) >= 15
    assert counts["train"]["events"] == 112
    assert counts["train"]["side_counts"] == {"LONG": 63, "SHORT": 49}
    assert counts["2023"]["events"] == 74
    assert counts["2023"]["side_counts"] == {"LONG": 50, "SHORT": 24}


def test_stage2_is_conditional_and_orthogonality_is_deferred() -> None:
    manifest = prereg.build_manifest()
    assert manifest["selection_protocol"][
        "stage2_requires_exact_unchanged_stage1_pass"
    ] is True
    assert manifest["orthogonality_after_standalone_pass"][
        "not_allowed_before_pass"
    ] is True
    assert manifest["selection_protocol"]["stage2_gates"]["minimum_trades"] == 60
    assert (
        manifest["selection_protocol"]["stage2_gates"][
            "each_subperiod_minimum_trades"
        ]
        == 20
    )
