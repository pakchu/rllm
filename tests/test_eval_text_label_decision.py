from training.eval_text_label import parse_label


def test_parse_decision_json_targets():
    assert parse_label('{"decision":"TRADE"}', key="decision") == "TRADE"
    assert parse_label('{"decision":"ABSTAIN"}', key="decision") == "ABSTAIN"


def test_parse_decision_defaults_to_abstain_when_unparseable():
    assert parse_label('nonsense', key="decision") == "ABSTAIN"
