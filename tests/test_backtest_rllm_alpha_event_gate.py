from training.backtest_rllm_alpha_event_gate import attach_predictions, select_clock, veto_frozen_clock

def rows():
    return [
        {"entry_time":"2024-01-01T00:05:00+00:00","exit_time":"2024-01-01T08:05:00+00:00","side":1,"slug":"a","policy_id":"A","stage":"test"},
        {"entry_time":"2024-01-01T00:05:00+00:00","exit_time":"2024-01-01T04:05:00+00:00","side":-1,"slug":"b","policy_id":"B","stage":"test"},
        {"entry_time":"2024-01-01T02:05:00+00:00","exit_time":"2024-01-01T10:05:00+00:00","side":1,"slug":"c","policy_id":"C","stage":"test"},
    ]
def test_same_entry_uses_highest_train_score_and_skips_overlap():
    attached=attach_predictions(rows(),[{"prediction":"TRADE"}]*3);clock=select_clock(attached,{"a":1.,"b":2.,"c":3.},gated=True);assert len(clock)==1 and clock.iloc[0]["candidate"]=="B"
def test_gate_removes_no_trade_events():
    preds=[{"prediction":"NO_TRADE"},{"prediction":"TRADE"},{"prediction":"NO_TRADE"}];clock=select_clock(attach_predictions(rows(),preds),{"a":3.,"b":2.,"c":1.},gated=True);assert len(clock)==1 and clock.iloc[0]["candidate"]=="B"

def test_matched_veto_does_not_repack_with_later_trade():
    attached=attach_predictions(rows(),[{"prediction":"NO_TRADE"},{"prediction":"TRADE"},{"prediction":"TRADE"}])
    frozen=select_clock(attached,{"a":3.,"b":2.,"c":1.},gated=False)
    matched=veto_frozen_clock(frozen,attached)
    assert frozen["candidate"].tolist()==["A"]
    assert matched.empty
