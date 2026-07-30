from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPORT = Path(
    "results/blockspace_fee_witness_concordance_support_2026-07-30.json"
)
PRIMARY = Path(
    "results/blockspace_fee_witness_concordance_primary_clock_2026-07-30.csv.gz"
)
CONTROLS = Path(
    "results/blockspace_fee_witness_concordance_control_clocks_2026-07-30.csv.gz"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bfwc_support_artifacts_are_exact() -> None:
    assert _sha256(REPORT) == (
        "1d7af687d4f0469ff1688d123e9e83ea957a5d9b51fa3617ab16c4c43978e22c"
    )
    assert _sha256(PRIMARY) == (
        "b125046a1a3defda960e51b42e03ee1c3bb72a0799c646d3ae16a3e692735ed1"
    )
    assert _sha256(CONTROLS) == (
        "de09979da981c91f91c8c0c57270df72bdc1d5fb6d344b7a783280774b5e3a9d"
    )


def test_bfwc_retires_before_novelty_or_outcomes() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["manifest_hash"] == (
        "0557d542597a7dcc5d195c1bd51f8a8ba8828dbce6015d2be7b07542dae9d56d"
    )
    assert payload["support_passed"] is False
    assert payload["novelty_status"] == "not_opened"
    assert payload["novelty_passed"] is False
    assert payload["decision"] == "retire_BFWC_288_unchanged"
    assert payload["next_action"] is None
    assert payload["rows_loaded"]["comparator_total"] == 0
    assert payload["rows_loaded"]["market"] == 0
    assert payload["rows_loaded"]["funding"] == 0
    assert payload["rows_loaded"]["premium"] == 0
    assert payload["rows_loaded"]["returns"] == 0
    assert payload["outcomes_opened"] is False
    assert payload["outcome_boundary"]["outcomes_computed"] is False


def test_bfwc_exact_failed_support_checks_are_locked() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    failed = {
        key for key, passed in payload["support_checks"].items() if not passed
    }
    assert failed == {
        "selection_2023_NovDec_min",
        "future_2025_total_min",
        "future_2026_maximum_month_share",
    }
    stats = payload["support_statistics"]
    assert stats["selection"]["total"] == 49
    assert stats["selection"]["LONG"] == 27
    assert stats["selection"]["SHORT"] == 22
    assert (
        stats["selection"]["monthly_counts"]["2023-11"]
        + stats["selection"]["monthly_counts"]["2023-12"]
        == 3
    )
    assert stats["future_2025"]["total"] == 28
    assert stats["future_2026"]["total"] == 17
    assert stats["future_2026"]["maximum_month_share"] == 6 / 17
