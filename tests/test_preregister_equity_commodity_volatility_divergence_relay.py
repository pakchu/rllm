import json
from training import preregister_equity_commodity_volatility_divergence_relay as prereg


def test_ecvdr_is_singleton_and_outcome_blind():
    result=prereg.build();prereg.validate(result)
    assert result["policy_id"]=="ECVDR-12"
    assert result["singleton"] is True
    assert result["outcomes_opened"] is False
    assert result["source_incidence_opened"] is False
    assert result["policy"]["variation_rank_min"]==.65


def test_ecvdr_round_trip(tmp_path):
    result=prereg.build();path=tmp_path/"registration.json"
    path.write_text(json.dumps(result,allow_nan=False));prereg.validate(json.loads(path.read_text()))
