from training import search_ml_long_complement_shorts as m


def test_ml_selection_and_cost_threshold_frozen():
    assert m.DESIGN['selection'].startswith('2024H2')
    assert 'mature strictly' in m.DESIGN['fit']
    assert m.DESIGN['standalone_profit_gate'] is False
    assert m.DESIGN['weights']==[.25,.5]
