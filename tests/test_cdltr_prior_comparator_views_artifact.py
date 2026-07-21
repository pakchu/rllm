from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


AMENDMENT = Path("docs/cdltr72a-preincidence-comparator-amendment-2026-07-21.md")
BUILDER = Path("training/freeze_cdltr_prior_comparator_views.py")
CLOCK = Path("results/cdltr_prior_comparator_views_2026-07-21.csv.gz")
MANIFEST = Path("results/cdltr_prior_comparator_views_manifest_2026-07-21.json")

EXPECTED_HASHES = {
    AMENDMENT: "fba002d78e0c29d5824d2bfd922d74c1d5477f2eb63f55959f14aafd88661064",
    BUILDER: "d0ef6a6f084c086b7355c18dbb13f2fb7739019fb31add903e5aad92932653b6",
    CLOCK: "bffdcf158d7d4e38db5794fb4761de528fb73b0b772ae950f3a087a93ab63f1a",
    MANIFEST: "a795f384287f24200e00d2cc5a5721610bb5282d1b044b3a653a053190c44261",
}
EXPECTED_COLUMNS = (
    "comparator",
    "capability",
    "decision_time",
    "entry_time",
    "exit_time",
    "side",
    "source_clock",
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_clock() -> pd.DataFrame:
    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        return pd.read_csv(handle, keep_default_na=False)


def test_cdltr72a_comparator_artifacts_are_hash_frozen() -> None:
    for path, expected in EXPECTED_HASHES.items():
        assert _sha256(path) == expected


def test_cdltr72a_manifest_binds_the_sanitized_clock_and_builder() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    assert manifest["candidate"] == "CDLTR-72A"
    assert manifest["manifest_hash"] == _canonical_hash(core)
    assert manifest["amendment"] == {
        "path": str(AMENDMENT),
        "sha256": EXPECTED_HASHES[AMENDMENT],
    }
    assert manifest["builder"] == {
        "path": str(BUILDER),
        "sha256": EXPECTED_HASHES[BUILDER],
    }
    assert manifest["clock"] == {
        "path": str(CLOCK),
        "sha256": EXPECTED_HASHES[CLOCK],
        "rows": 9_985,
        "columns": list(EXPECTED_COLUMNS),
        "counts": EXPECTED_COUNTS,
        "capability_counts": {
            "directional_interval": 1_788,
            "timestamp_only": 8_197,
        },
        "directional_comparators": [
            "CVTR-1",
            "DFFB-601",
            "FLCC-1:FLCC-H4-Q60",
            "FLCC-1:FLCC-H4-Q65",
            "FLCC-1:FLCC-H8-Q60",
            "FLCC-1:FLCC-H8-Q65",
            "NTB-7",
            "NWE-8",
            "ORFR-1",
            "chain_activity_impulse_momentum",
        ],
        "timestamp_only_comparators": [
            "NWE-7",
            "live_anchor_2023",
            "prior_microstructure:cbfr72",
            "prior_microstructure:mfic_fast",
            "prior_microstructure:mfic_slow",
            "prior_microstructure:mfic_union",
            "prior_microstructure:netf_fast",
            "prior_microstructure:netf_slow",
            "prior_microstructure:netf_union",
            "prior_microstructure:terminal_absorption_wait72_h72",
            "prior_microstructure:wfrs_l288_q90_h144",
        ],
    }


def test_cdltr72a_clock_contains_only_capability_safe_prior_views() -> None:
    frame = _load_clock()
    assert tuple(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 9_985
    assert frame.groupby("comparator", sort=True).size().to_dict() == EXPECTED_COUNTS
    assert frame.groupby("capability", sort=True).size().to_dict() == {
        "directional_interval": 1_788,
        "timestamp_only": 8_197,
    }

    timestamp_only = frame["capability"].eq("timestamp_only")
    assert bool(frame.loc[timestamp_only, "side"].eq("").all())
    assert bool(frame.loc[timestamp_only, "exit_time"].eq("").all())
    assert not bool(
        frame.loc[timestamp_only, "source_clock"]
        .str.contains(r"long|short", case=False, regex=True)
        .any()
    )

    directional = frame["capability"].eq("directional_interval")
    assert bool(frame.loc[directional, "side"].astype(int).isin((-1, 1)).all())
    assert bool(frame.loc[directional, "exit_time"].ne("").all())
    decision = pd.to_datetime(frame.loc[directional, "decision_time"], utc=True)
    entry = pd.to_datetime(frame.loc[directional, "entry_time"], utc=True)
    exit_time = pd.to_datetime(frame.loc[directional, "exit_time"], utc=True)
    assert bool(decision.le(entry).all())
    assert bool(entry.lt(exit_time).all())

    forbidden_fragments = (
        "return",
        "pnl",
        "equity",
        "cagr",
        "mdd",
        "price",
        "funding",
        "forecast",
    )
    assert not any(
        fragment in column.lower()
        for column in frame.columns
        for fragment in forbidden_fragments
    )


def test_cdltr72a_manifest_proves_no_cdltr_or_prior_outcomes_were_opened() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["inputs"]["chain_clock"] == {
        "path": "results/chain_activity_impulse_momentum_pre2024_comparator_clock_2026-07-21.csv.gz",
        "sha256": "e50cc154e23950a381aa456180970140882083734128bd7f902257738633f320",
    }
    assert manifest["inputs"]["chain_manifest"] == {
        "path": "results/chain_activity_impulse_momentum_pre2024_comparator_clock_manifest_2026-07-21.json",
        "sha256": "899704a0e998d818fd09735ca90af3c82aecfce94a288eec2bbc77c0c3df8441",
    }
    assert manifest["protocol"] == {
        "cdltr_incidence_rows_derived": 0,
        "cdltr_source_rows_read": 0,
        "chain_execution_or_research_code_imported": False,
        "chain_raw_market_network_funding_rows_read": 0,
        "complete_comparator_identity_set_enforced": True,
        "json_sibling_outcomes_decoded": False,
        "output_columns": list(EXPECTED_COLUMNS),
        "prior_equity_cagr_mdd_computed": False,
        "prior_pnl_fields_retained": 0,
        "prior_return_fields_retained": 0,
        "timestamp_only_side_and_exit_forced_empty": True,
    }
