from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import pytest

from training import freeze_cdltr_prior_comparator_views as freeze


EXPECTED_CLOCK_SHA256 = (
    "bffdcf158d7d4e38db5794fb4761de528fb73b0b772ae950f3a087a93ab63f1a"
)
EXPECTED_COUNTS = {
    "CVTR-1": 661,
    "DFFB-601": 112,
    "FLCC-1:FLCC-H4-Q60": 136,
    "FLCC-1:FLCC-H4-Q65": 122,
    "FLCC-1:FLCC-H8-Q60": 125,
    "FLCC-1:FLCC-H8-Q65": 116,
    "NTB-7": 41,
    "NWE-7": 147,
    "NWE-8": 81,
    "ORFR-1": 328,
    "chain_activity_impulse_momentum": 66,
    "live_anchor_2023": 136,
    "prior_microstructure:cbfr72": 144,
    "prior_microstructure:mfic_fast": 1_566,
    "prior_microstructure:mfic_slow": 1_635,
    "prior_microstructure:mfic_union": 3_019,
    "prior_microstructure:netf_fast": 319,
    "prior_microstructure:netf_slow": 267,
    "prior_microstructure:netf_union": 586,
    "prior_microstructure:terminal_absorption_wait72_h72": 100,
    "prior_microstructure:wfrs_l288_q90_h144": 278,
}


@pytest.fixture(scope="module")
def bundle() -> tuple[pd.DataFrame, dict[str, Any]]:
    return freeze.build_bundle()


def test_repository_paths_never_fall_back_to_another_checkout() -> None:
    relative = freeze._repository_path("missing/local-only.csv")
    assert relative == freeze.REPOSITORY_ROOT / "missing/local-only.csv"
    assert not str(relative).startswith("/home/pakchu/rllm/missing")
    for unsafe in ("/tmp/outside.csv", "~/outside.csv", "../outside.csv"):
        with pytest.raises(RuntimeError, match="repository-relative"):
            freeze._repository_path(unsafe)


def test_streams_only_the_exact_top_level_json_subtree(tmp_path: Path) -> None:
    path = tmp_path / "fixture.json"
    path.write_text(
        "{\n"
        '  "outcomes": {"pnl": [999.0]},\n'
        '  "events": [{"time": "2023-01-01T00:00:00Z"}],\n'
        '  "later_outcome": THIS_IS_INTENTIONALLY_INVALID_JSON\n'
        "}\n",
        encoding="utf-8",
    )
    assert freeze._stream_top_level_json_value(path, "events") == [
        {"time": "2023-01-01T00:00:00Z"}
    ]


def test_capability_rows_enforce_directional_and_timestamp_boundaries() -> None:
    timestamp = freeze._timestamp_row(
        "timestamp",
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:05:00Z",
        "source",
    )
    assert timestamp["capability"] == freeze.TIMESTAMP_ONLY
    assert timestamp["exit_time"] == ""
    assert timestamp["side"] == ""

    directional = freeze._directional_row(
        "directional",
        "2023-01-01T00:00:00Z",
        "2023-01-01T00:05:00Z",
        "2023-01-01T01:05:00Z",
        "SHORT",
        "source",
    )
    assert directional["capability"] == freeze.DIRECTIONAL
    assert directional["side"] == -1
    with pytest.raises(RuntimeError, match="interval is invalid"):
        freeze._directional_row(
            "directional",
            "2023-01-01T00:00:00Z",
            "2023-01-01T00:05:00Z",
            "2023-01-01T00:05:00Z",
            1,
            "source",
        )


def test_actual_bundle_is_complete_schema_clean_and_deterministic(
    bundle: tuple[pd.DataFrame, dict[str, Any]],
) -> None:
    frame, manifest = bundle
    assert tuple(frame.columns) == freeze.CLOCK_COLUMNS
    assert len(frame) == 9_985
    assert manifest["clock"]["sha256"] == EXPECTED_CLOCK_SHA256
    assert manifest["clock"]["counts"] == EXPECTED_COUNTS
    assert manifest["clock"]["capability_counts"] == {
        "directional_interval": 1_788,
        "timestamp_only": 8_197,
    }
    assert set(manifest["clock"]["directional_comparators"]) == (
        freeze.EXPECTED_DIRECTIONAL_COMPARATORS
    )
    assert set(manifest["clock"]["timestamp_only_comparators"]) == (
        freeze.EXPECTED_TIMESTAMP_COMPARATORS
    )
    forbidden = {
        "return",
        "pnl",
        "equity",
        "cagr",
        "mdd",
        "price",
        "funding",
        "forecast",
    }
    assert forbidden.isdisjoint(frame.columns)
    timestamp = frame["capability"].eq(freeze.TIMESTAMP_ONLY)
    assert bool(frame.loc[timestamp, "side"].eq("").all())
    assert bool(frame.loc[timestamp, "exit_time"].eq("").all())
    assert not bool(
        frame.loc[timestamp, "source_clock"]
        .str.contains(r"long|short", case=False, regex=True)
        .any()
    )
    directional = ~timestamp
    assert bool(frame.loc[directional, "side"].astype(int).isin((-1, 1)).all())
    assert bool(frame.loc[directional, "exit_time"].ne("").all())


def test_chain_is_consumed_only_from_the_committed_clock(
    bundle: tuple[pd.DataFrame, dict[str, Any]],
) -> None:
    frame, manifest = bundle
    chain = frame.loc[frame["comparator"].eq("chain_activity_impulse_momentum")]
    assert len(chain) == 66
    assert set(chain["source_clock"]) == {
        "chain_activity_impulse_momentum:fit_2021",
        "chain_activity_impulse_momentum:fit_2022",
        "chain_activity_impulse_momentum:select_2023",
    }
    assert manifest["inputs"]["chain_clock"] == {
        "path": str(freeze.CHAIN_CLOCK),
        "sha256": freeze.CHAIN_CLOCK_SHA256,
    }
    assert manifest["inputs"]["chain_manifest"] == {
        "path": str(freeze.CHAIN_MANIFEST),
        "sha256": freeze.CHAIN_MANIFEST_SHA256,
    }


def test_manifest_explicitly_keeps_cdltr_and_prior_outcomes_closed(
    bundle: tuple[pd.DataFrame, dict[str, Any]],
) -> None:
    _, manifest = bundle
    assert manifest["protocol"] == {
        "cdltr_source_rows_read": 0,
        "cdltr_incidence_rows_derived": 0,
        "chain_raw_market_network_funding_rows_read": 0,
        "chain_execution_or_research_code_imported": False,
        "json_sibling_outcomes_decoded": False,
        "prior_return_fields_retained": 0,
        "prior_pnl_fields_retained": 0,
        "prior_equity_cagr_mdd_computed": False,
        "timestamp_only_side_and_exit_forced_empty": True,
        "complete_comparator_identity_set_enforced": True,
        "output_columns": list(freeze.CLOCK_COLUMNS),
    }
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    assert manifest["manifest_hash"] == freeze.canonical_hash(core)


def test_write_is_deterministic_and_immutable(tmp_path: Path) -> None:
    del tmp_path  # The protocol intentionally forbids outputs outside the repository.
    with tempfile.TemporaryDirectory(dir=freeze.REPOSITORY_ROOT / "results") as raw:
        directory = Path(raw)
        relative = directory.relative_to(freeze.REPOSITORY_ROOT)
        cfg = freeze.Config(
            output_clock=str(relative / "clocks.csv.gz"),
            output_manifest=str(relative / "manifest.json"),
        )
        manifest = freeze.write_bundle(cfg)
        clock = freeze._repository_path(cfg.output_clock)
        result = freeze._repository_path(cfg.output_manifest)
        assert hashlib.sha256(clock.read_bytes()).hexdigest() == EXPECTED_CLOCK_SHA256
        assert json.loads(result.read_text(encoding="utf-8")) == manifest
        with gzip.open(clock, "rt", encoding="utf-8", newline="") as handle:
            assert tuple(pd.read_csv(handle, nrows=0).columns) == freeze.CLOCK_COLUMNS
        with pytest.raises(FileExistsError, match="immutable"):
            freeze.write_bundle(cfg)
