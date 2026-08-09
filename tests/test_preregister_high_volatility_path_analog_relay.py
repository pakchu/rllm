from training import preregister_high_volatility_path_analog_relay as subject


def test_hvpar_preregistration_is_oos_blind_singleton() -> None:
    payload = subject.build()
    assert payload["policy_id"] == "HVPAR-8"
    assert payload["singleton"] is True
    assert payload["oos_outcomes_opened"] is False
    assert payload["oos_source_incidence_opened"] is False
    assert payload["gross9_rows_opened"] is False
    assert payload["research_boundary"]["repair_of_prior_candidate"] is False
    assert payload["training_contract"]["hyperparameter_grid"] is False


def test_hvpar_feature_and_analog_contract_is_exact() -> None:
    payload = subject.build()
    assert payload["feature_contract"]["ordered_features"] == list(subject.FEATURES)
    assert len(subject.FEATURES) == 33
    assert payload["training_contract"]["estimator"]["neighbors"] == 64
    assert payload["policy"]["prediction_strength_quantile"] == 0.75
    assert payload["oos_clock"]["hold"] == "8 elapsed hours"
    assert payload["diagnostic_controls"]["names"][-1] == "forced_long"


def test_hvpar_manifest_hash_binds_payload() -> None:
    payload = subject.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == subject.canonical_hash(core)

