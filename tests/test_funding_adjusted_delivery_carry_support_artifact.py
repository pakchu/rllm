from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import training.build_funding_adjusted_delivery_carry_support as support
from training.preregister_funding_adjusted_delivery_carry import canonical_hash, file_sha256


ARTIFACT = Path("results/funding_adjusted_delivery_carry_support_2026-07-17.json")
EXPECTED_CONTENT_HASH = "dab72f4495f54b91ccccf052e1a6c9d0dccadd8a90130336516d7dd3b8276785"
EXPECTED_BUILDER_SHA256 = "b0282b5f4f1efb98834c875be9046573cc2589f40c139a460560cf15c48bae74"


def _load() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_support_artifact_is_bound_to_frozen_code_and_protocol() -> None:
    report = _load()
    body = {key: value for key, value in report.items() if key not in {"as_of", "content_hash"}}
    assert canonical_hash(body) == EXPECTED_CONTENT_HASH
    assert report["content_hash"] == EXPECTED_CONTENT_HASH
    assert report["protocol_hash"] == support.PREREGISTRATION_HASH
    assert report["support_builder_sha256"] == EXPECTED_BUILDER_SHA256
    assert file_sha256(support.__file__) == EXPECTED_BUILDER_SHA256


def test_support_counts_and_outcome_boundary_are_frozen() -> None:
    report = _load()
    assert report["disposition"] == "PASS_SUPPORT_OPEN_2021_2022_PNL"
    assert report["support"]["passed"] is True
    assert report["support"]["failures"] == []
    assert report["support"]["pre2023"]["events"] == 30
    assert report["support"]["pre2023"]["by_year"] == {"2021": 16, "2022": 14}
    assert report["support"]["pre2023"]["by_half"] == {
        "2021H1": 8,
        "2021H2": 8,
        "2022H1": 9,
        "2022H2": 5,
    }
    assert report["support"]["holdout_2023_diagnostic"]["events"] == 9
    assert report["forbidden_outcome_columns_loaded"] == []
    assert report["pnl_opened"] is False
    assert report["oos_2024_plus_opened"] is False
    assert "settlement_mark_price" not in report["loaded_columns"]["funding"]
    assert set(report["loaded_columns"]["quarterly"]).isdisjoint(
        {"um_open", "um_high", "um_low"}
    )


def test_support_schedule_has_causal_delays_no_overlap_and_no_2024() -> None:
    events = pd.DataFrame(_load()["events_2021_2023"])
    for column in (
        "funding_time",
        "signal_time",
        "entry_time",
        "exit_time",
        "mandatory_exit_time",
        "delivery_time",
    ):
        events[column] = pd.to_datetime(events[column], utc=True)
    assert (events["signal_time"] - events["funding_time"]).eq(pd.Timedelta(minutes=5)).all()
    assert (events["entry_time"] - events["funding_time"]).eq(pd.Timedelta(minutes=10)).all()
    assert events["entry_time"].lt(events["exit_time"]).all()
    assert events["exit_time"].le(events["mandatory_exit_time"]).all()
    assert events["mandatory_exit_time"].lt(events["delivery_time"]).all()
    assert events["exit_time"].lt(pd.Timestamp("2024-01-01", tz="UTC")).all()
    assert (
        events["entry_time"].iloc[1:].reset_index(drop=True)
        >= events["exit_time"].iloc[:-1].reset_index(drop=True) + pd.Timedelta(hours=24)
    ).all()
