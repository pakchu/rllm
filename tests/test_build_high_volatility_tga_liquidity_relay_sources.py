import json

from training import build_high_volatility_tga_liquidity_relay_sources as builder


def test_parse_selects_only_tga_closing_balance():
    raw = json.dumps({"data": [
        {"record_date": "2022-04-18", "account_type": builder.ACCOUNT_TYPE, "open_today_bal": "400000"},
        {"record_date": "2022-04-18", "account_type": "Total TGA Deposits (Table II)", "open_today_bal": "10"},
        {"record_date": "2026-07-24", "account_type": builder.ACCOUNT_TYPE, "open_today_bal": "500000"},
    ], "meta": {"total-count": 3}}).encode()
    frame, metadata = builder.parse(raw)
    assert frame.tga_close_millions.tolist() == [400000, 500000]
    assert metadata["selected_rows"] == 2
