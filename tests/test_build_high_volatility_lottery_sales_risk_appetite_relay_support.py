import hashlib

import pandas as pd

from training import build_high_volatility_lottery_sales_risk_appetite_relay_support as b


def report(draw_date: str, header_date: str, generated: str, sales: str) -> dict:
    text = (
        "winner_summary   1.00  T e x a s  Page: 1\n"
        f"CDC:12237 / {header_date} POWERBALL WINNER SUMMARY REPORT FOR DRAW 1498 {generated}\n"
        "---\n"
        f"  NET SALES       :           $ {sales}\n"
    )
    return {"draw_date": draw_date, "url": "https://example.invalid", "text": text, "sha256": hashlib.sha256(text.encode("latin-1")).hexdigest()}


def test_parse_report_uses_explicit_generation_and_net_sales():
    value = b.parse_report(report("2023-07-03", "Mon Jul-03-2023", "Tue Jul-04-2023 00:01:14", "3,555,211.00"))
    assert value["draw_date"].isoformat() == "2023-07-03"
    assert value["draw_number"] == 1498
    assert value["net_sales"] == 3555211.0
    assert value["report_generated_time"] == pd.Timestamp("2023-07-04T05:01:14Z")
    assert value["decision_time"] == pd.Timestamp("2023-07-04T12:00:00Z")
    assert value["available_by_decision"] is True


def test_strict_prior_midrank_excludes_current():
    values = pd.Series(range(61), dtype=float)
    ranks = b.strict_prior_midrank(values, lookback=180, minimum=60)
    assert ranks.iloc[:60].isna().all()
    assert ranks.iloc[60] == 1.0


def test_primary_clock_holds_24_hours():
    features = pd.DataFrame(
        {
            "decision_time": [pd.Timestamp("2023-07-03T12:00:00Z")],
            "result_side": [-1],
            "weekday_balanced_side": [1],
            "net_sales": [2_000_000.0],
            "sales_change": [-0.1],
            "btc_realized_variation": [0.02],
            "btc_variation_rank": [0.8],
        }
    )
    clock = b.build_clock(features)
    assert len(clock) == 1
    assert clock.iloc[0].side == -1
    assert clock.iloc[0].exit_time - clock.iloc[0].entry_time == pd.Timedelta(hours=24)
