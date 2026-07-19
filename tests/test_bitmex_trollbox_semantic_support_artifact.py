from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from training.download_bitmex_trollbox_attention import canonical_hash, sha256_file


RESULT = Path("results/bitmex_trollbox_semantic_support_2026-07-20.json")
CLOCK = Path("results/bitmex_trollbox_semantic_clock_2026-07-20.json")
RESULT_FILE_SHA256 = (
    "2b89f710d59a5c0708d400541defb43d5e292f6d9bdedbe66d6bdcf614d09e94"
)
CLOCK_FILE_SHA256 = (
    "af8687564614ec5a1cbd7a1438c908f687af7bd99ceede9539016e5c1b111bd4"
)


def test_semantic_support_and_clock_are_hash_bound_private_safe() -> None:
    assert sha256_file(RESULT) == RESULT_FILE_SHA256
    assert sha256_file(CLOCK) == CLOCK_FILE_SHA256
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    clock = json.loads(CLOCK.read_text(encoding="utf-8"))
    result_core = {
        key: value
        for key, value in result.items()
        if key not in {"result_hash", "created_at"}
    }
    clock_core = {
        key: value
        for key, value in clock.items()
        if key not in {"manifest_hash", "created_at"}
    }

    assert result["result_hash"] == canonical_hash(result_core)
    assert result["result_hash"] == (
        "5996b7d7497d6bf5e96343f7ceca766363d58aa34280aea0fdb7b8653a8b1725"
    )
    assert clock["manifest_hash"] == canonical_hash(clock_core)
    assert clock["manifest_hash"] == (
        "fdcd9c7c376b18df2799acf24af04a421ca679e27009e6a539888defc7438aa8"
    )
    assert clock["support_result_hash"] == result["result_hash"]
    assert result["protocol"]["private_text_opened"] is True
    assert result["protocol"]["private_text_committed"] is False
    assert result["market_or_outcomes_opened"] is False
    assert result["protocol"]["market_rows_loaded"] == 0
    assert result["protocol"]["funding_rows_loaded"] == 0
    assert result["protocol"]["outcome_rows_loaded"] == 0
    assert result["source_audit"]["pages"] == 13_610
    assert result["source_audit"]["messages"] == 6_791_328
    assert result["semantic_jobs"] == 67_497
    assert result["model_generated_jobs"] == 67_497
    assert result["meta_instruction_guarded_jobs"] == 0

    gate = result["support_gate"]
    assert gate["passed"] is True
    assert len(gate["checks"]) == 17
    assert all(gate["checks"].values())
    assert gate["counts"] == {
        "total": 2_718,
        "train": 1_728,
        "train_2020h2": 527,
        "train_2021": 1_201,
        "test_2022": 990,
        "test_2022_h1": 498,
        "test_2022_h2": 492,
    }
    assert gate["active_weeks"] == {"all": 131, "train": 79, "test": 53}
    assert gate["meta_instruction_guard_share"] == 0.0
    assert gate["parse_success"] == 0.9990518097100611

    events = clock["events"]
    assert len(events) == 5_417
    expected_keys = {
        "observation_start",
        "observation_end",
        "entry_earliest",
        "exit_time",
        "crowd_label",
        "contrarian_side",
        "bullish_participants",
        "bearish_participants",
        "unclear_participants",
        "selected_participants",
        "selected_messages",
        "meta_instruction_guarded_messages",
    }
    assert all(set(event) == expected_keys for event in events)
    assert Counter(event["crowd_label"] for event in events) == {
        "BULLISH": 1_106,
        "BEARISH": 1_612,
        "UNCLEAR": 2_699,
    }
    side = {"BULLISH": -1, "BEARISH": 1, "UNCLEAR": 0}
    assert all(
        event["contrarian_side"] == side[event["crowd_label"]]
        for event in events
    )
    assert all(
        event["bullish_participants"]
        + event["bearish_participants"]
        + event["unclear_participants"]
        == event["selected_participants"]
        for event in events
    )
    assert all(3 <= event["selected_participants"] <= 8 for event in events)
    assert all(
        0 <= event["selected_messages"] <= 2 * event["selected_participants"]
        for event in events
    )
    assert all(event["meta_instruction_guarded_messages"] == 0 for event in events)
    starts = pd.to_datetime([event["observation_start"] for event in events])
    ends = pd.to_datetime([event["observation_end"] for event in events])
    entries = pd.to_datetime([event["entry_earliest"] for event in events])
    assert starts.is_monotonic_increasing
    assert (ends - starts == pd.Timedelta(minutes=5)).all()
    assert (entries - ends == pd.Timedelta(minutes=5)).all()
