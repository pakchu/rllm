from __future__ import annotations

import json

import pytest

from training import backtest_pposm_conditional_residual as backtest
from training import build_pposm_counterfactual_action_data as counterfactual
from training import build_pposm_residual_action_data as residual


def test_parse_route_requires_exactly_one_route() -> None:
    assert backtest.parse_route("TP4") == "TP4"
    assert backtest.parse_route('{"prediction":"SKIP"}') == "SKIP"
    with pytest.raises(ValueError, match="exactly one"):
        backtest.parse_route("TP4 TP12")


def test_lock_pair_rows_and_load_routes_preserve_frozen_identity(tmp_path) -> None:
    positions = {"test_2024": (1, 2), "eval_2025": (), "holdout_2026": ()}
    pair_rows = []
    base_ids = []
    for signal in positions["test_2024"]:
        base = counterfactual.signal_identity("test_2024", signal)
        base_ids.append(base)
        for candidate in residual.CANDIDATE_ACTIONS:
            pair_rows.append(
                {
                    "metadata": {
                        "identity": residual.residual_identity(base, candidate),
                        "base_identity": base,
                        "candidate_action": candidate,
                    }
                }
            )
    signals, observed_bases = backtest.lock_pair_rows(pair_rows, positions)
    assert signals == (1, 2)
    assert observed_bases == tuple(base_ids)

    predictions = tmp_path / "routes.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps({"identity": identity, "prediction": route})
            for identity, route in zip(base_ids, ("TP4", "SKIP"), strict=True)
        )
        + "\n",
        encoding="utf-8",
    )
    assert backtest.load_routes(predictions, expected_base_ids=base_ids) == (
        "TP4",
        "SKIP",
    )


def test_load_routes_rejects_identity_drift(tmp_path) -> None:
    predictions = tmp_path / "routes.jsonl"
    predictions.write_text(
        json.dumps({"identity": "wrong", "prediction": "TP4"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity"):
        backtest.load_routes(predictions, expected_base_ids=("expected",))
