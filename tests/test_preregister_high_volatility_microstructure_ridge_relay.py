from training import preregister_high_volatility_microstructure_ridge_relay as prereg


def test_hvmrr_separates_pretraining_authority_from_oos_seal():
    result = prereg.build()
    assert result["policy_id"] == "HVMRR-6"
    assert result["pretraining_outcomes_authorized_after_preregistration"] is True
    assert result["oos_outcomes_opened"] is False
    assert result["oos_source_incidence_opened"] is False
    assert result["training_contract"]["label_end_exclusive"] == "2023-07-01T00:00:00Z"
    assert result["training_contract"]["refit_after_2023_06_30"] is False


def test_hvmrr_freezes_single_model_without_grid_or_feature_selection():
    result = prereg.build()
    training = result["training_contract"]
    assert training["estimator"] == "sklearn.linear_model.Ridge(alpha=10.0, fit_intercept=True, solver='svd')"
    assert training["hyperparameter_grid"] is False
    assert training["feature_selection"] is False
    assert len(result["feature_contract"]["ordered_features"]) == 10
    assert result["policy"]["prediction_strength_quantile"] == 0.80
    assert result["policy"]["variation_rank_min"] == 0.65


def test_hvmrr_preserves_strict_economics_and_no_repair_boundary():
    result = prereg.build()
    assert result["economic_gates"]["cagr_to_strict_mdd_min"] == 3.0
    assert result["economic_gates"]["mean_gross_underlying_min_bp"] == 20.0
    assert result["research_boundary"]["candidate_count"] == 1
    assert result["research_boundary"]["grid"] is False
    assert result["research_boundary"]["repair_of_prior_candidate"] is False
    assert result["research_boundary"]["promoted_prior_control"] is False


def test_hvmrr_hash_binds_core():
    result = prereg.build()
    assert result["manifest_hash"] == prereg.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
