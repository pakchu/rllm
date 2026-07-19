from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.preregister_packet_churn_persistence import canonical_hash, sha256_file


RESULT = Path("results/packet_churn_persistence_support_2026-07-19.json")
CLOCK = Path("results/packet_churn_persistence_clock_2026-07-19.csv")
RESULT_SHA256 = "c8bdbf3d16dac7623f8f291c98a11d9e40cc14defdadb4d6206364bdbc64dc4f"
CLOCK_SHA256 = "e50be3f0744617e3797581ea762546e7f21c32dd0707b9bdcbd264686d4f9acb"


def test_support_artifacts_are_hash_frozen() -> None:
    assert sha256_file(RESULT) == RESULT_SHA256
    assert sha256_file(CLOCK) == CLOCK_SHA256
    payload = json.loads(RESULT.read_text())
    assert payload["implementation"]["preregistration_source_sha256"] == sha256_file(
        "training/preregister_packet_churn_persistence.py"
    )
    assert payload["implementation"]["predecessor_source_sha256"] == sha256_file(
        "training/preregister_minute_packet_topology_alpha.py"
    )


def test_support_artifact_keeps_every_outcome_window_sealed() -> None:
    payload = json.loads(RESULT.read_text())
    protocol = payload["protocol"]
    assert protocol["outcomes_opened"] is False
    assert protocol["post_entry_prices_loaded"] is False
    assert protocol["funding_loaded"] is False
    assert protocol["successor_train_outcomes_opened"] is False
    assert protocol["successor_2023_outcomes_opened"] is False
    assert protocol["support_selection_uses_2023_incidence"] is False
    assert protocol["2023_incidence_disclosed_but_profitability_unopened"] is True
    assert set(protocol["sealed_windows"]) == {
        "train_2020_2022",
        "selection_2023",
        "test_2024",
        "eval_2025",
        "holdout_2026",
    }
    assert payload["source"]["columns_used"] == [
        "date",
        "feature_available_time_utc",
        "minute_dispersion_feature_valid",
        "um_net_flow_fraction",
        "spot_net_flow_fraction",
        "um_flow_sign_switch_rate",
        "um_ticket_log_std",
        "um_signed_impact_bp",
        "spot_signed_impact_bp",
    ]


def test_unique_support_winner_and_clock_contract_are_frozen() -> None:
    payload = json.loads(RESULT.read_text())
    trials = payload["support_stopping_rule"]["trials"]
    passing = [trial for trial in trials if trial["passes_support"]]
    assert [trial["name"] for trial in passing] == [
        "pcp_cross_venue_churn_breakout_p70_s35_h96_confirm6"
    ]
    assert payload["selected"]["support"] == {
        "by_year": {"2020": 48, "2021": 54, "2022": 45, "2023": 45},
        "longs": 104,
        "selection_2023": {
            "h1": 25,
            "h2": 20,
            "longs": 26,
            "max_month_fraction": 0.13333333333333333,
            "shorts": 19,
            "total": 45,
        },
        "shorts": 88,
        "total": 192,
        "train_2020_2022": {
            "longs": 78,
            "max_month_fraction": 0.061224489795918366,
            "shorts": 69,
            "total": 147,
        },
    }
    clock = pd.read_csv(CLOCK)
    assert len(clock) == 192
    assert np.all(
        clock["confirmation_end_position"].to_numpy()
        - clock["setup_position"].to_numpy()
        == 6
    )
    assert np.all(
        clock["entry_position"].to_numpy()
        - clock["confirmation_end_position"].to_numpy()
        == 2
    )
    assert np.all(
        clock["exit_position"].to_numpy() - clock["entry_position"].to_numpy() == 96
    )
    signal_available = pd.to_datetime(clock["signal_available_at"])
    confirmation_end = pd.to_datetime(clock["confirmation_end_bar_date"])
    entry = pd.to_datetime(clock["entry_date"])
    assert (signal_available == confirmation_end + pd.Timedelta("5min")).all()
    assert (entry == signal_available + pd.Timedelta("5min")).all()
    forbidden = {"return", "pnl", "open", "high", "low", "close", "funding"}
    assert not any(
        any(word in column.lower() for word in forbidden) for column in clock.columns
    )


def test_embedded_result_hash_replays() -> None:
    payload = json.loads(RESULT.read_text())
    expected = canonical_hash(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_at", "result_hash"}
        }
    )
    assert payload["result_hash"] == expected
