import json
from training import seal_high_volatility_equity_breadth_participation_relay_source_reproduction_failure as s


def test_terminal_reproduction_failure_is_canonical_and_blocks_downstream():
    x=s.build()
    assert x["decision"]=="terminal_source_reproduction_reject_no_repair"
    assert x["advance_to_gross9_novelty"] is x["advance_to_economic_outcomes"] is False
    assert x["postentry_return_pnl_execution_price_opened"] is x["gross9_rows_opened"] is False
    assert x["manifest_hash"]==s.canonical_hash({k:v for k,v in x.items() if k!="manifest_hash"})
    assert json.loads(s.OUTPUT.read_text())==x
