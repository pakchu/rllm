from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

RESULT = Path(
    "results/portfolio_added_alpha_july_2026_performance_2026-07-27.json"
)


def test_july_performance_artifact_is_frozen_and_internally_consistent() -> None:
    report = json.loads(RESULT.read_text())
    assert report["mode"] == "promoted_added_alpha_completed_bar_monthly_replay"
    assert report["accounting_version"] == "same_btc_low_high_v1"
    assert report["retrospective_not_pristine_oos"] is True
    assert report["config"]["env_path"] == "<redacted>"
    assert report["window"] == {
        "requested_start": "2026-07-01 00:00:00",
        "requested_end_exclusive": "2026-08-01 00:00:00",
        "start": "2026-07-01 00:00:00",
        "end_exclusive": "2026-07-27 15:00:00",
        "last_completed_bar": "2026-07-27 14:55:00",
        "bars": 7668,
        "calendar_days": 26.625,
    }
    assert (
        report["data_quality"]["window_market_hash"]
        == "d4dcd50aeabe600eb8a1bb37f5348e019ebc1c469db074ae11c3176bb88f70e2"
    )

    portfolio = report["full_window"]["portfolio"]
    assert np.isclose(portfolio["absolute_return_pct"], -1.0403933929998055)
    assert np.isclose(portfolio["strict_mdd_pct"], 2.0359030824293245)
    assert portfolio["trades"] == 6
    assert portfolio["longs"] == 0
    assert portfolio["shorts"] == 6
    assert portfolio["trades_by_sleeve"] == {
        "fresh_kimchi_fx": 0,
        "frozen_annual_rank7": 0,
        "rex_taker_low_range_position": 5,
        "cand_rex_veto_7": 1,
        "markov_transition_long": 0,
    }
    assert report["post_promotion"]["portfolio"]["trades"] == 0
    assert report["post_promotion"]["portfolio"]["absolute_return_pct"] == 0.0
    assert report["live_ledger"]["post_promotion_execution_rows"] == 0
    assert report["live_ledger"]["july14_cand_rex_ledger_rows"] == [
        {
            "action": "CLOSE",
            "status": "FILLED",
            "rows": 2,
            "net_realized_pnl": "4.85892684",
        },
        {
            "action": "OPEN",
            "status": "FILLED",
            "rows": 2,
            "net_realized_pnl": "0",
        },
    ]


def test_july_artifact_pins_every_executable_source() -> None:
    report = json.loads(RESULT.read_text())
    for raw_path, expected in report["source_sha256"].items():
        path = Path(raw_path)
        assert path.is_file(), raw_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
