from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import build_fee_endpoint_topology_disagreement_support as support


ARTIFACT = Path(
    "results/fee_endpoint_topology_disagreement_support_2026-07-20.json"
)
EXPECTED_FILE_SHA256 = (
    "03ba910a314ba6efb647f6588dff603261d414e5114680ca33bdc27d59aed035"
)
EXPECTED_MANIFEST_HASH = (
    "24902cbe9869d2c5dc3443047d31c7ad0a1650d23822da6158d7e4b5ee758c27"
)


def _artifact() -> dict[str, object]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_fetd_support_artifact_is_frozen_and_outcome_blind() -> None:
    payload = _artifact()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == EXPECTED_FILE_SHA256
    assert support.canonical_hash(core) == payload["manifest_hash"]
    assert payload["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert payload["event_rows_published"] == 0
    assert payload["feature_values_published"] == 0
    assert payload["performance_values_opened"] is False
    boundary = payload["outcome_boundary"]
    assert boundary["market_rows_loaded"] == 0
    assert boundary["funding_rows_loaded"] == 0
    assert boundary["return_rows_loaded"] == 0
    assert boundary["return_or_pnl_fields"] == 0
    assert boundary["post_2023_source_rows_loaded"] == 0


def test_fetd_support_failure_and_stopping_decision_are_frozen() -> None:
    payload = _artifact()
    gate = payload["support_gates"]
    assert gate["passed"] is False
    assert gate["counts"] == {
        "clock_total": 119,
        "train": 82,
        "selection": 37,
        "2021": 21,
        "2022": 61,
        "2021H1": 0,
        "2021H2": 21,
        "2022H1": 35,
        "2022H2": 26,
        "2023H1": 16,
        "2023H2": 21,
        "2023Q1": 13,
        "2023Q2": 3,
        "2023Q3": 9,
        "2023Q4": 12,
    }
    failed = {name for name, passed in gate["checks"].items() if not passed}
    assert failed == {
        "train_each_year_minimum",
        "train_each_half_year_minimum",
        "train_maximum_month_share",
        "selection_each_quarter_minimum",
        "selection_maximum_month_share",
    }
    assert gate["side_counts"] == {
        "train": {"long": 41, "short": 41},
        "selection": {"long": 16, "short": 21},
    }
    assert gate["delayed_entry_dropped_split_edges"] == {
        "train": 0,
        "selection": 0,
    }
    assert payload["next_action"] == "reject FETD-288 without opening outcomes"


def test_fetd_support_publishes_only_sealed_clock_commitments() -> None:
    payload = _artifact()
    commitments = payload["sealed_clock_commitments"]
    assert set(commitments) == {"primary", "controls"}
    assert set(commitments["primary"]) == {"columns", "frame_hash", "rows"}
    assert set(commitments["controls"]) == {
        "clock_counts",
        "columns",
        "frame_hash",
        "rows",
    }
    assert commitments["primary"]["rows"] == 119
    assert "artifacts" not in payload
