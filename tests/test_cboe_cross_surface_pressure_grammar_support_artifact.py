from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import build_cboe_cross_surface_pressure_grammar_support as s


REPORT = Path(
    "results/cboe_cross_surface_pressure_grammar_support_2026-07-24.json"
)
CLOCK = Path(
    "data/cboe_cross_surface_pressure_grammar_clocks_2020_2023.csv.gz"
)
REPORT_SHA256 = (
    "ee52aa25d9a1a870eb5f52974e4ab5720fd34d5abac4bd5819900ce2c0db9a1c"
)
CLOCK_SHA256 = (
    "33a87fe5cb9c99e9a8b4ecc0699f5746387fabfc5ea434f4adf96615374aacdb"
)
MANIFEST_HASH = (
    "07d8ecf98bf78221d6385648c8b551a860cc7dbc3911d15e091fe75b401267ac"
)


def _payload() -> dict[str, object]:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_frozen_cspg_support_artifacts_record_outcome_blind_retirement() -> None:
    assert s.sha256_file(REPORT) == REPORT_SHA256
    assert s.sha256_file(CLOCK) == CLOCK_SHA256
    payload = _payload()
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["manifest_hash"] == s.canonical_hash(core)
    assert payload["source_support_passed"] is True
    assert payload["token_support_passed"] is False
    assert payload["first_failing_stage"] == "token_support"
    assert payload["first_failing_check"] == "train:tail_level:LOW_share_min"
    assert payload["decision"] == (
        "retire_CSPG_288_unchanged_before_market_outcomes"
    )
    assert payload["authorized_next_stage"] is None
    assert payload["clock"]["sha256"] == hashlib.sha256(
        CLOCK.read_bytes()
    ).hexdigest()
    assert payload["implementation"]["source_sha256"] == s.sha256_file(
        s.SCRIPT_PATH
    )
    assert payload["implementation"]["tests_sha256"] == s.sha256_file(
        s.TEST_PATH
    )
    assert payload["implementation"]["contract_sha256"] == (
        s.IMPLEMENTATION_CONTRACT_SHA256
    )


def test_cspg_support_and_failure_diagnostics_are_frozen() -> None:
    payload = _payload()
    funnel = payload["feature_funnel"]
    assert funnel["exact_common_dates"] == 1006
    assert funnel["rank_complete_common_states"] == 879
    assert funnel["token_ready_common_states"] == 878
    assert payload["reservation_funnel"] == {
        "globally_reserved": 878,
        "raw_candidates": 878,
        "suppressed_overlap": 0,
    }
    statistics = payload["clock_statistics"]
    assert statistics["global"]["events"] == 878
    assert statistics["train"]["events"] == 375
    assert statistics["2020"]["events"] == 123
    assert statistics["2021"]["events"] == 250
    assert statistics["2022"]["events"] == 250
    assert statistics["2023"]["events"] == 250
    assert all(payload["source_support_checks"].values())
    failed = {
        key
        for key, passed in payload["token_support_checks"].items()
        if not passed
    }
    assert len(failed) == 24
    assert {
        "train:tail_level:LOW_share_min",
        "train:tail_change:max_share",
        "train:maximum_exact_signature_share",
        "2022:tail_level:HIGH_share_min",
        "2022:tail_change:max_share",
        "2022:maximum_exact_signature_share",
        "2023:tail_level:LOW_share_min",
        "2023:option_level:max_share",
        "2023:maximum_exact_signature_share",
    } <= failed
    report = payload["token_report"]
    assert report["train"]["token_shares"]["tail_level"]["LOW"] == 0.072
    assert report["2022"]["token_shares"]["tail_level"]["LOW"] == 0.904
    assert report["2023"]["token_shares"]["tail_level"]["HIGH"] == 0.776
    assert report["train"]["maximum_exact_signature_share"] == 0.176
    assert report["2022"]["maximum_exact_signature_share"] == 0.328
    assert report["2023"]["maximum_exact_signature_share"] == 0.26


def test_cspg_support_outcome_boundary_never_opened_market_or_model() -> None:
    payload = _payload()
    assert payload["outcomes_opened"] is False
    assert payload["market_loaded"] is False
    assert payload["funding_loaded"] is False
    assert payload["comparators_opened"] is False
    assert payload["post_2023_loaded"] is False
    boundary = payload["outcome_boundary"]
    for field in (
        "BTC_market_rows_decoded",
        "funding_rows_decoded",
        "comparator_rows_decoded",
        "future_return_rows_decoded",
        "return_or_PnL_fields_decoded",
        "PnL_CAGR_MDD_values_decoded",
        "post_2023_rows_decoded",
        "model_labels_created",
        "model_training_runs",
        "network_calls",
    ):
        assert boundary[field] == 0
