from training import preregister_funding_cash_transfer_basis_inventory_reanchoring_reversal as p


def test_frozen_contract():
    value=p.build();p.validate(value)
    assert value["policy_id"]=="FCBIRR-8"
    assert value["outcomes_opened"] is False and value["source_incidence_opened"] is False
    assert value["clock"]["side"]=="opposite common funding and premium sign"
    assert value["clock"]["signal_settlement"].startswith("excluded")
    assert value["diagnostic_controls"]["cannot_be_promoted"] is True
