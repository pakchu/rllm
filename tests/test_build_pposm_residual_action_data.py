import pytest

from training import build_pposm_residual_action_data as residual


def _base_row():
    return {
        "split": "train",
        "prompt": "causal_predicates: {}\nsignal_time_state: {}",
        "target": "TP12",
        "metadata": {
            "identity": "pposm-counterfactual-action|pre_2024|7",
            "window": "pre_2024",
            "signal_position": 7,
            "signal_time": "2023-01-01 00:00:00+00:00",
            "action_utilities": {"SKIP": 0.0, "TP4": 0.01, "TP12": 0.025},
            "action_take_profit_bps": {"TP4": 400, "TP12": 1200},
            "executable_positions": {"TP4": {"entry_position": 8, "exit_position": 10}},
            "entry_rule": "next_5m_open",
        },
    }


def test_pair_rows_are_candidate_minus_tp4_and_causal():
    rows = residual.build_pair_rows(_base_row())
    assert [row["metadata"]["candidate_action"] for row in rows] == ["SKIP", "TP12"]
    assert [row["target"] for row in rows] == ["KEEP", "SWITCH"]
    assert rows[0]["metadata"]["residual_utilities"] == {"KEEP": 0.0, "SWITCH": -0.01}
    assert rows[1]["metadata"]["residual_utilities"]["KEEP"] == 0.0
    assert rows[1]["metadata"]["residual_utilities"]["SWITCH"] == pytest.approx(0.015)
    assert "action_utilities" not in rows[0]["prompt"]
    assert rows[0]["metadata"]["identity"] == "pposm-residual-action|SKIP|pposm-counterfactual-action|pre_2024|7"


def test_rows_from_counterfactual_rejects_duplicate_pair_identity():
    rows = [_base_row(), _base_row()]
    try:
        residual.rows_from_counterfactual(rows)
    except RuntimeError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate identity accepted")


def test_residual_prompt_removes_counterfactual_route_instruction():
    prompt = residual.residual_prompt(
        "Frozen PPOSM counterfactual action choice.\n"
        "Return exactly one token: SKIP, TP4, or TP12.\n"
        "causal_predicates: {}\n"
        "signal_time_state: {}\n"
        "predicate_priority: capitulation predicates before premium-overheat predicates.",
        candidate="SKIP",
    )
    assert "Return exactly one token: KEEP or SWITCH." in prompt
    assert "Return exactly one token: SKIP, TP4, or TP12." not in prompt
    assert "causal_predicates" in prompt and "signal_time_state" in prompt
