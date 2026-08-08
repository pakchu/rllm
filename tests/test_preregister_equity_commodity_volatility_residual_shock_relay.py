import json
from training import preregister_equity_commodity_volatility_residual_shock_relay as prereg


def test_ecvrs_is_singleton_and_outcome_blind():
    result=prereg.build();prereg.validate(result)
    assert result["policy_id"]=="ECVRS-12"
    assert result["singleton"] is True
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["policy"]["variation_rank_min"]==.65
    assert result["policy"]["absolute_residual_rank_min"]==.70


def test_ecvrs_round_trip(tmp_path):
    result=prereg.build();path=tmp_path/"registration.json"
    path.write_text(json.dumps(result,allow_nan=False));prereg.validate(json.loads(path.read_text()))
