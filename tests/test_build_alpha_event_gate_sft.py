import pandas as pd
from training.build_alpha_event_gate_sft import _event_context, _prompt, _validate_funding_for_labels

def test_event_prompt_excludes_future_label_metadata():
    event=pd.Series({"entry_time":pd.Timestamp("2023-07-01T00:05Z"),"exit_time":pd.Timestamp("2023-07-01T08:05Z"),"side":1,"variation_rank":.8,"future_return":.5})
    candidate={"policy_id":"P","formula":{"features":{"x":"prior"}},"slug":"s"}
    prompt=_prompt(candidate,event)
    assert "variation_rank" in prompt
    assert "future_return" not in prompt and "net_return" not in prompt and "funding_cash" not in prompt

def test_event_context_keeps_only_compact_signal_fields():
    event=pd.Series({"candidate":"x","entry_time":"t","side":1,"score":1.25,"note":"ok"})
    assert _event_context(event)=={"score":1.25,"note":"ok"}

def test_funding_boundary_allows_exchange_millisecond_jitter():
    start=pd.Timestamp("2025-01-01T00:00Z");end=pd.Timestamp("2025-01-02T00:00Z")
    funding=pd.DataFrame({"date":[start+pd.Timedelta(milliseconds=15),start+pd.Timedelta(hours=8,milliseconds=5),start+pd.Timedelta(hours=16,milliseconds=1)],"funding_rate":[.001]*3,"mark_price":[100.]*3})
    _validate_funding_for_labels(funding,start,end)
