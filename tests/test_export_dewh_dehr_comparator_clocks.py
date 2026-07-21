from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from training.export_dewh_dehr_comparator_clocks import (
    EXPECTED_FROZEN_EVENT_CLOCK_HASH,
    build_outputs,
    canonical_hash,
    normalized_entry,
    publish,
)


def test_normalized_entry_waits_a_complete_bucket() -> None:
    exact = cast(pd.Timestamp, pd.Timestamp("2022-01-01 09:05:00", tz="UTC"))
    offset = cast(
        pd.Timestamp,
        pd.Timestamp("2022-01-01 09:05:00.125", tz="UTC"),
    )

    assert normalized_entry(exact) == pd.Timestamp("2022-01-01 09:10:00", tz="UTC")
    assert normalized_entry(offset) == pd.Timestamp("2022-01-01 09:15:00", tz="UTC")
    with pytest.raises(ValueError, match="lacks timezone"):
        normalized_entry(cast(pd.Timestamp, pd.Timestamp("2022-01-01 09:05:00")))


def test_real_dehr_selection_is_unchanged_and_outcomes_remain_closed() -> None:
    manifest, clock_bytes = build_outputs()

    assert manifest["frozen_dehr_bindings"]["frozen_event_clock_hash"] == (
        EXPECTED_FROZEN_EVENT_CLOCK_HASH
    )
    assert manifest["normalization"]["selection_changed"] is False
    assert manifest["normalization"]["side_changed"] is False
    assert manifest["normalization"]["rows"] == 159
    assert manifest["normalization"]["original_off_grid_rows"] == 159
    assert manifest["normalization"]["normalized_off_grid_rows"] == 0
    assert manifest["clock"]["rows"] == 159
    assert manifest["clock"]["longs"] == 62
    assert manifest["clock"]["shorts"] == 97
    assert manifest["outcome_boundary"] == {
        "dehr_source_rows_read": 1119,
        "dehr_candidate_rows_reconstructed": 159,
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "performance_artifacts_parsed": 0,
        "return_or_pnl_fields_read": 0,
        "post_2022_source_rows_loaded": 0,
        "network_calls": 0,
        "economic_outcomes_computed": False,
    }
    assert manifest["dehr_reopened_or_repaired"] is False
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    assert manifest["manifest_hash"] == canonical_hash(core)
    assert len(clock_bytes) > 0


def test_export_is_deterministic_and_create_only(tmp_path: Path) -> None:
    first_manifest, first_clock = build_outputs()
    second_manifest, second_clock = build_outputs()
    assert first_manifest == second_manifest
    assert first_clock == second_clock

    manifest_path = tmp_path / "manifest.json"
    clock_path = tmp_path / "clock.csv.gz"
    publish(manifest_path, clock_path, first_manifest, first_clock)
    assert json.loads(manifest_path.read_text())["clock"]["rows"] == 159
    assert clock_path.read_bytes() == first_clock
    with pytest.raises(FileExistsError):
        publish(manifest_path, clock_path, first_manifest, first_clock)
