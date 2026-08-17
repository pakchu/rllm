import json
from training import preregister_high_volatility_btc_factor_residual_flip_continuation as p

def test_preregistration_freezes_singleton_before_incidence():
    value=p.build();p.validate(value)
    assert value["policy_id"]=="HVBFRRFC-8"
    assert value["candidate_family"]==["HVBFRRFC-8"]
    assert value["policy"]["prior_standardized_residual_min"]==2.0
    assert value["policy"]["variation_rank_min"]==0.65
    assert value["research_boundary"]["full_candidate_incidence_opened"] is False
    assert value["diagnostic_controls"]["cannot_be_promoted"] is True
    json.dumps(value,allow_nan=False)

def test_source_is_hash_pinned():
    assert p.sha256(p.SOURCE)==p.SOURCE_SHA
