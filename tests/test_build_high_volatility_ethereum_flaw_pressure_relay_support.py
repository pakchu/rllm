import pandas as pd
from training import build_high_volatility_ethereum_flaw_pressure_relay_support as b


def test_flaw_grammar_uses_exact_ascii_tokens():
    assert b.FLAW_PATTERN.search("core: fix trie corruption")
    assert b.FLAW_PATTERN.search("Security: prevent crash")
    assert not b.FLAW_PATTERN.search("fixture cleanup")
    assert not b.FLAW_PATTERN.search("prefix handling")


def test_daily_panel_uses_weekly_change_and_inverse_side():
    commits = [
        {"commit": f"{i:040x}", "committer_unix": int(pd.Timestamp(day, tz="UTC").timestamp()), "parents": [], "subject": subject}
        for i, (day, subject) in enumerate([
            ("2022-01-01", "fix bug"),
            ("2022-01-08", "fix bug"),
            ("2022-01-08", "security fix"),
            ("2022-01-09", "docs only"),
        ], 1)
    ]
    frame = b.build_daily_panel(commits)
    jan8 = frame.loc[frame.source_day.eq(pd.Timestamp("2022-01-08", tz="UTC"))].iloc[0]
    jan9 = frame.loc[frame.source_day.eq(pd.Timestamp("2022-01-09", tz="UTC"))].iloc[0]
    assert jan8.daily_count == 2 and jan8.pressure_change == 1 and jan8.result_side == -1
    assert jan9.daily_count == 0 and jan9.pressure_change == 0 and jan9.result_side == 0
    assert jan8.decision_time == pd.Timestamp("2022-01-10T12:00:00Z")


def test_support_stats_enforce_side_and_month_visibility():
    clock = pd.DataFrame({"split": ["train"] * 4, "side": [1, 1, -1, -1], "entry_time": pd.to_datetime(["2023-07-01", "2023-07-02", "2023-08-01", "2023-08-02"], utc=True)})
    value = b.stats(clock, "train")
    assert value == {"events": 4, "longs": 2, "shorts": 2, "minority_side_share": .5, "max_month_share": .5}
