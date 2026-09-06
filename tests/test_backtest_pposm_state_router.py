from training import backtest_pposm_state_router as backtest


def test_parse_route_requires_exactly_one_route_token():
    assert backtest.parse_route("TP4") == "TP4"
    assert backtest.parse_route('{"route":"SKIP"}') == "SKIP"
    try:
        backtest.parse_route("TP4 TP12")
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("ambiguous route accepted")


def test_agreement_reports_nonconstant_confusion():
    report = backtest._agreement(
        ["SKIP", "TP4", "TP12"], ["SKIP", "TP12", "TP12"]
    )
    assert report["decisions"] == 3
    assert report["matching_decisions"] == 2
    assert report["decision_agreement_rate"] == 2 / 3
