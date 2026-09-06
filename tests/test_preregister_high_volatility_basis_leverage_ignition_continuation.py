from training import preregister_high_volatility_basis_leverage_ignition_continuation as p


def test_frozen_contract():
    value=p.build();p.validate(value)
    assert value["policy_id"]=="HVBLIC-6"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["clock"]["side"]=="premium displacement sign"
    assert value["policy"]["hold_hours"]==6
    assert value["diagnostic_controls"]["cannot_be_promoted"] is True
