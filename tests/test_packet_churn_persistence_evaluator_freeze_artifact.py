from __future__ import annotations

import json
from pathlib import Path

from training.evaluate_packet_churn_persistence_pre2024 import (
    EvaluationConfig,
    _result_hash,
    sha256_file,
    verify_evaluator_freeze,
)


FREEZE = Path("results/packet_churn_persistence_evaluator_freeze_2026-07-19.json")
FREEZE_SHA256 = "8d64046f83c5960689c259fbcd27a971e741c728cb5af17d71afd4f96038829d"


def test_evaluator_freeze_is_hash_frozen_and_replays() -> None:
    assert sha256_file(FREEZE) == FREEZE_SHA256
    payload = json.loads(FREEZE.read_text())
    assert payload["result_hash"] == _result_hash(payload)
    assert payload["evaluation_source_sha256"] == sha256_file(
        payload["evaluation_source"]
    )
    assert payload["evaluation_document_sha256"] == sha256_file(
        payload["evaluation_document"]
    )
    assert verify_evaluator_freeze(EvaluationConfig()) == payload


def test_evaluator_freeze_has_not_opened_any_outcome_value() -> None:
    payload = json.loads(FREEZE.read_text())
    protocol = payload["protocol"]
    assert protocol["outcomes_opened"] is False
    assert protocol["outcome_value_columns_parsed"] is False
    assert protocol["opened_windows"] == []
    assert protocol["selection_loader_requires_exact_train_replay"] is True
    assert set(protocol["sealed_windows"]) == {
        "train_2020_2022",
        "selection_2023",
        "test_2024",
        "eval_2025",
        "holdout_2026",
    }


def test_evaluator_freezes_physical_prefixes_sources_and_controls() -> None:
    payload = json.loads(FREEZE.read_text())
    assert payload["outcome_boundaries"] == {
        "market": {
            "rows_pre2023": 315648,
            "rows_pre2024": 420768,
            "first_date": "2020-01-01 00:00:00",
            "last_date": "2023-12-31 23:55:00",
        },
        "funding": {
            "rows_pre2023": 3288,
            "rows_pre2024": 4383,
            "first_date": "2020-01-01 00:00:00",
            "last_date": "2023-12-31 16:00:00",
        },
    }
    dependencies = payload["static_dependencies"]
    assert dependencies["feature_source_sha256"] == (
        "5ea9f5075171c255732cc6eed003736c1beed211a0e6fd7797ab02f31a917aaa"
    )
    assert dependencies["market_source_sha256"] == (
        "e7a987ac662601bff445a23bb3c9aea736d14b8f7ef88d7e69794cdaf9d6c28d"
    )
    assert dependencies["funding_source_sha256"] == (
        "c19829fa085a50f29c13762373a2b6db1c62025d657be1f5a3fbb9ce254482f7"
    )
    clocks = payload["frozen_clocks"]
    assert clocks["primary"] == {
        "rows": 192,
        "hash": "df54c6114244c2676c90e9fe0e7698dd96db04537802893f7edc5318e67d39a0",
        "entry_delay_bars_from_confirmation_end": 2,
    }
    assert (
        clocks["immediate_entry_control"]["entry_delay_bars_from_confirmation_end"] == 1
    )
    assert (
        clocks["extra_latency_control"]["entry_delay_bars_from_confirmation_end"] == 3
    )
