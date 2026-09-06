from training import calibrate_pposm_symbolic_confidence as calibration


def test_score_info_returns_top_margin():
    label, margin = calibration.score_info(
        {"label_mean_logprobs": {"SKIP": -1.0, "TP4": -2.0, "TP12": -3.0}}
    )
    assert label == "SKIP" and margin == 1.0


def test_train_threshold_prefers_most_authoritative_accurate_candidate():
    data = [{"target": "SKIP", "metadata": {"identity": str(i)}} for i in range(40)]
    scores = []
    for i in range(40):
        # Ten low-confidence mistakes, thirty high-confidence correct rows.
        values = {"SKIP": -0.1, "TP4": -2.0, "TP12": -3.0}
        if i < 10:
            values = {"SKIP": -0.2, "TP4": -0.1, "TP12": -3.0}
        scores.append({"index": i, "identity": str(i), "label_mean_logprobs": values})
    result = calibration.choose_threshold(data, scores)
    assert result["chosen"]["accuracy"] == 1.0
    assert result["chosen"]["model_authoritative"] == 30
