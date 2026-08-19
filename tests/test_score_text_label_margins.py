from training import score_text_label_margins as scorer


def test_score_rows_preserves_identity_and_margin(monkeypatch):
    monkeypatch.setattr(
        scorer,
        "candidate_mean_logprobs",
        lambda **_: {"NO_TRADE": -2.0, "TRADE": -0.5},
    )
    rows = [{"prompt": "p", "target": "TRADE", "metadata": {"identity": "id-1"}}]
    scored = scorer.score_rows(
        rows, labels=("NO_TRADE", "TRADE"), tokenizer=object(), model=object()
    )
    assert scored[0]["identity"] == "id-1"
    assert scored[0]["prediction"] == "TRADE"
    assert scored[0]["margin"] == 1.5


def test_candidate_labels_must_be_unique():
    try:
        scorer._validate_labels(("A", "A"))
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate labels were accepted")
