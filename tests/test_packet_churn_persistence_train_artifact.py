from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.evaluate_packet_churn_persistence_pre2024 import (
    _result_hash,
    sha256_file,
)


TRAIN = Path("results/packet_churn_persistence_train_2020_2022_2026-07-19.json")
SELECTION = Path("results/packet_churn_persistence_selection_2023_2026-07-19.json")
TRAIN_SHA256 = "0bcb93a483d6e87f92fbbdeae60274e0b4c65ac4e78f6da7fd45456b3054437c"


def test_train_rejection_artifact_is_hash_frozen() -> None:
    assert sha256_file(TRAIN) == TRAIN_SHA256
    payload = json.loads(TRAIN.read_text())
    assert payload["result_hash"] == _result_hash(payload)
    assert payload["freeze_sha256"] == sha256_file(
        "results/packet_churn_persistence_evaluator_freeze_2026-07-19.json"
    )


def test_train_rejects_primary_and_keeps_selection_sealed() -> None:
    payload = json.loads(TRAIN.read_text())
    assert payload["decision"] == "reject_before_selection"
    assert payload["primary"]["passes"] is False
    assert payload["protocol"]["opened_windows"] == ["train_2020_2022"]
    assert payload["protocol"]["selection_2023_opened"] is False
    assert not SELECTION.exists()
    assert payload["primary"]["gates"] == {
        "absolute_return_positive": False,
        "cagr_to_strict_mdd_at_least_1_5": False,
        "strict_mdd_at_most_15": False,
        "trades_at_least_100": True,
        "ten_bp_per_side_stress_positive": False,
        "weekly_cluster_p_below_0_10": False,
    }


def test_train_primary_statistics_are_frozen() -> None:
    stats = json.loads(TRAIN.read_text())["primary"]["train"]
    assert stats["absolute_return_pct"] == pytest.approx(-8.224605034155918)
    assert stats["cagr_pct"] == pytest.approx(-2.819695659733379)
    assert stats["strict_mdd_pct"] == pytest.approx(20.58123450718111)
    assert stats["cagr_to_strict_mdd"] == pytest.approx(-0.1370032326656374)
    assert stats["trades"] == 147
    assert stats["longs"] == 78
    assert stats["shorts"] == 69
    assert stats["weekly_cluster_sign_flip"]["p_value_one_sided"] == pytest.approx(
        0.7300026999730003
    )


def test_all_frozen_timing_and_direction_controls_are_negative() -> None:
    controls = json.loads(TRAIN.read_text())["train_controls"]
    assert set(controls) == {
        "same_clock_side_flip",
        "immediate_entry",
        "extra_latency",
    }
    assert all(stats["absolute_return_pct"] < 0.0 for stats in controls.values())
