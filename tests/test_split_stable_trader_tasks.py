from training.split_stable_trader_tasks import to_gate_row, to_side_row


def _row(action="LONG"):
    return {"prompt": 'Return exactly one JSON object: {"action": <LONG|SHORT|NO_TRADE>, "risk": <LOW|MEDIUM|HIGH>}\nctx', "target": f'{{"action":"{action}","risk":"MEDIUM"}}'}


def test_to_gate_row_maps_trade():
    assert '"gate": "TRADE"' in to_gate_row(_row("LONG"))["target"]


def test_to_side_row_skips_no_trade():
    assert to_side_row(_row("NO_TRADE")) is None


def test_to_side_row_keeps_side():
    assert '"side": "SHORT"' in to_side_row(_row("SHORT"))["target"]
