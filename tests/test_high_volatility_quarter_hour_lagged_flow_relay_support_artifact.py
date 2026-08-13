import json

from training import build_high_volatility_quarter_hour_lagged_flow_relay_support as support


EXPECTED_CLOCK_SHA = "22432585e6ce24ebfcde66438035a61fbd23238824bc2e0fff17d83ac2ac9292"
EXPECTED_SOURCE_SHA = "b16bc89caf258fc86a3f6fb68190bf46767ebbe2e50619e95dd9453871b564ac"
EXPECTED_RESULT_SHA = "09b000b4f1a678956bdd605e9e00fa35337d63e92ad37731b43c55f560e29a86"


def test_reproduced_source_support_artifact_is_sealed() -> None:
    value = json.loads(support.RESULT.read_text())
    assert support.sha256(support.CLOCK) == EXPECTED_CLOCK_SHA
    assert support.sha256(support.FEATURES) == EXPECTED_SOURCE_SHA
    assert support.sha256(support.RESULT) == EXPECTED_RESULT_SHA
    assert value["support_passed"] is True
    assert value["advance_to_gross9_novelty"] is True
    assert value["advance_to_economic_outcomes"] is False
    assert value["gross9_rows_opened"] is False
    assert {name: row["events"] for name, row in value["support"].items()} == {
        "train": 40,
        "test": 130,
        "eval": 206,
        "final": 84,
    }
    assert all(value["support_checks"].values())
    assert all(
        control["promotion_authorized"] is False
        for control in value["controls"].values()
    )
