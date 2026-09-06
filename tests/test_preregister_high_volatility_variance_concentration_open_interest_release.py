from __future__ import annotations
import copy
import pytest
from training import preregister_high_volatility_variance_concentration_open_interest_release as p

def test_singleton_mechanism_and_frozen_gates() -> None:
    x=p.build();p.validate(x)
    assert x['policy_id']=='HVCVCOIR-12' and x['singleton'] is True
    assert x['clock']['hold']=='12 elapsed hours'
    assert 'D-4h and D' in x['clock']['open_interest_confirmation']
    assert x['source_support_gates']['minimum_events']=={'train':8,'test':12,'eval':12,'final':8}
    assert x['economic_gates']['cagr_to_strict_mdd_min']==3.0
    assert x['economic_gates']['weekly_signflip_one_sided_p_max']==.1

def test_boundary_is_honest_and_outcome_blind() -> None:
    b=p.build()['research_boundary']
    assert b['original_HVVCR_source_incidence_and_novelty_failure_known'] is True
    assert b['original_HVVCR_economic_outcomes_opened'] is False
    assert b['exact_composite_incidence_or_outcomes_known'] is False
    assert b['postentry_return_or_pnl_opened'] is False
    assert b['repair_of_prior_candidate'] is False

def test_manifest_and_source_binding_drift_rejected(monkeypatch) -> None:
    x=copy.deepcopy(p.build());x['clock']['hold']='8 elapsed hours'
    with pytest.raises(RuntimeError,match='preregistration hash mismatch'):p.validate(x)
