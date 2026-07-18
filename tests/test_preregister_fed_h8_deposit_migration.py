from __future__ import annotations

import gzip
import json

import pandas as pd

from training import fed_h8_deposit_migration_clock as clock
from training import preregister_fed_h8_deposit_migration as prereg


def test_source_only_selector_chooses_q50() -> None:
    selected, grid = prereg.select_tail_quantile(clock.load_source())
    assert selected == 0.50
    assert grid["0.55"]["support_passed"] is False
    assert grid["0.50"]["support_passed"] is True
    assert grid["0.50"]["counts"]["stage1"]["events"] == 75
    assert grid["0.50"]["counts"]["2023"]["events"] == 24


def test_preregistration_is_outcome_blind_and_clock_replays(tmp_path) -> None:
    clock_path = tmp_path / "clock.csv.gz"
    payload = prereg.build_manifest(clock_output=clock_path)
    prereg.validate_manifest(payload)
    assert payload["outcomes_opened"] is False
    assert payload["opened_outcome_windows"] == []
    disclosure = payload["research_history_boundary"]["disclosure"]
    assert disclosure["market_rows_loaded"] == 0
    assert disclosure["funding_rows_loaded"] == 0
    assert payload["policy"]["tail_quantile"] == 0.50
    assert payload["primary_clock"]["events"] == 99
    with gzip.open(clock_path, "rt") as handle:
        frame = pd.read_csv(handle)
    assert len(frame) == 99
    assert not set(clock.STRUCTURAL_EXCLUSION_RELEASES) & set(frame["release_date"])
    local_entry = pd.to_datetime(frame["entry_time"], utc=True).dt.tz_convert(
        clock.NEW_YORK
    )
    assert local_entry.dt.hour.eq(17).all()
    assert local_entry.dt.minute.eq(0).all()


def test_frozen_preregistration_manifest_is_self_sealed() -> None:
    payload = json.loads(open(prereg.DEFAULT_OUTPUT).read())
    prereg.validate_manifest(payload)
    assert payload["primary_clock"]["sha256"] == prereg.sha256_file(
        prereg.DEFAULT_CLOCK
    )
