from __future__ import annotations

import json

import pandas as pd
import pytest

from training import preregister_coinm_roll_migration_alpha as v1
from training import preregister_coinm_roll_migration_alpha_v2 as v2


def test_v2_reuses_exact_frozen_candidate_definitions() -> None:
    frozen = json.loads(
        open("results/coinm_roll_migration_support_2026-07-19.json").read()
    )
    assert [item["candidate"] for item in frozen["candidates"]] == [
        v2.asdict(candidate) for candidate in v1.CANDIDATES
    ]
    assert v2.verify_fixed_logic()["sha256"] == v2.BASE_PREREGISTRATION_SHA256


def test_v2_source_and_manifest_are_exactly_sealed() -> None:
    seal = v2.verify_source_seal(v2.Config())
    assert seal["output_sha256"] == v2.EXPECTED_SOURCE_SHA256
    assert seal["monthly_rows_added"] == 3_718
    assert seal["monthly_overlap_diagnostics"]["conflict_rows"] == 2


def test_v2_support_loader_never_reads_high_or_low() -> None:
    source = v1.load_source(v2.Config.input_csv)
    assert "front_high" not in source.columns
    assert "next_low" not in source.columns
    assert source["feature_valid"].sum() == 368_180
    critical = source.loc[
        source["signal_bar_open_utc"].between(
            pd.Timestamp("2023-12-21 00:00"),
            pd.Timestamp("2023-12-21 00:15"),
        )
    ]
    assert critical["feature_valid"].all()


def test_v2_paths_are_not_mutable() -> None:
    with pytest.raises(ValueError, match="paths are frozen"):
        v2.build_report(v2.Config(output="results/alternate.json"))


def test_v2_implementation_path_is_cwd_independent(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        v2.implementation_path()
        == "training/preregister_coinm_roll_migration_alpha_v2.py"
    )
