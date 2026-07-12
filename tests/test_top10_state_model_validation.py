from training.validate_top10_state_model_candidates import _jaccard, _ranked_live_rows


def test_ranked_live_rows_never_admits_rank11():
    selected = [{"signal_hash": f"h{i}"} for i in range(1, 12)]
    result = {"selected": selected, "live_grade": [selected[3], selected[10]]}

    assert _ranked_live_rows(result) == [(4, selected[3])]


def test_jaccard_reports_entry_overlap():
    assert _jaccard({1, 2}, {2, 3}) == 1 / 3
    assert _jaccard(set(), set()) == 0.0
