from training import preregister_cross_asset_alpha_transfer as prereg


def test_manifest_is_deterministic_and_outcomes_are_sealed() -> None:
    first = prereg.manifest()
    second = prereg.manifest()
    assert first == second
    assert first["manifest_hash"] == second["manifest_hash"]
    assert first["status"] == "preregistered_before_outcome_download"
    assert "post_signal_return" in first["sealed_outcome_fields"]
    assert "eval_outcomes" in first["sealed_outcome_fields"]


def test_scope_uses_only_requested_tradable_proxies() -> None:
    instruments = prereg.manifest()["scope"]["instruments"]
    assert tuple(instruments) == ("QQQ", "069500.KS", "GLD")
    assert "^KS11" not in instruments


def test_every_gross8_sleeve_is_explicitly_nonportable() -> None:
    audit = prereg.manifest()["gross8_portability_audit"]
    assert len(audit) == 5
    assert all(row["exact_portable"] is False for row in audit.values())
    assert all(row["blocking_inputs"] for row in audit.values())


def test_transfer_rule_cannot_promote_a_best_of_three_result() -> None:
    rule = prereg.manifest()["transfer_decision"]
    assert rule["all_required"] is True
    assert rule["primary_instruments"] == ["QQQ", "069500.KS", "GLD"]
    assert rule["per_instrument_eval"]["cagr_to_strict_mdd"] == ">= 3.0"
    assert "No best-of-three promotion" in rule["decision"]


def test_execution_is_next_open_nonoverlapping_and_cost_stressed() -> None:
    execution = prereg.manifest()["execution"]
    assert execution["entry"] == "next available session open"
    assert "skip overlapping" in execution["positioning"]
    assert execution["base_cost_bps_per_side"] == 5.0
    assert execution["stress_cost_bps_per_side"] == 10.0
