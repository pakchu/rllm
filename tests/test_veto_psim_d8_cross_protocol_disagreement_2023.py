from __future__ import annotations

from training import veto_psim_d8_cross_protocol_disagreement_2023 as veto


def test_future_veto_gate_is_conjunctive() -> None:
    selection = {"base": {"strict_mdd": 0.20}}
    passing = {
        "base": {
            "absolute_return": 0.01,
            "strict_mdd": 0.30,
            "cagr_to_strict_mdd": 0.60,
            "closed_trades": 15,
        },
        "stress": {"absolute_return": 0.0},
    }
    assert veto.veto_checks(passing, selection) == {
        "base_net_return_strictly_positive": True,
        "stress_net_return_not_negative": True,
        "base_return_over_strict_mdd_minimum": True,
        "strict_mdd_not_more_than_selection_multiple": True,
        "minimum_closed_trades": True,
    }
    failing = {
        **passing,
        "stress": {"absolute_return": -1e-9},
    }
    assert not all(veto.veto_checks(failing, selection).values())
