from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from training import build_intrinsic_volume_price_lag_handoff_support as s


REPORT = Path(s.DEFAULT_REPORT_OUTPUT)
CLOCK = Path(s.DEFAULT_CLOCK_OUTPUT)
REPORT_SHA256 = "ef0b187c4de29c27583bfe7bef85c7a55db95eb193954fe587cf5cce23a17103"
CLOCK_SHA256 = "2efca3b44b0512a9423da90171f43babcadec2316dc6148796f3e61f98138e80"
MANIFEST_HASH = "77ed74b30e7941eb5bfe47671f7d9e679ff56c70afea94c56abdec34bf4c8ba3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ivplh_source_support_artifacts_are_exact_and_reproducible() -> None:
    assert _sha256(REPORT) == REPORT_SHA256
    assert _sha256(CLOCK) == CLOCK_SHA256
    payload = json.loads(REPORT.read_text("utf-8"))
    rebuilt, clock_bytes = s.build_real_support_payload()

    assert payload == rebuilt
    assert CLOCK.read_bytes() == clock_bytes
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["clock"]["sha256"] == CLOCK_SHA256
    assert payload["clock"]["rows"] == 1399
    assert payload["clock"]["control_counts"]["primary"] == 66

    with gzip.open(CLOCK, "rt", encoding="utf-8", newline="") as handle:
        assert handle.readline() == ",".join(s.CLOCK_COLUMNS) + "\n"
        first = dict(
            zip(
                s.CLOCK_COLUMNS,
                handle.readline().rstrip("\n").split(","),
                strict=True,
            )
        )
    assert len(first["source_day"]) == 10
    assert first["decision_time"].endswith("Z")
    assert first["entry_time"].endswith("Z")
    assert first["exit_time"].endswith("Z")


def test_ivplh_is_terminally_rejected_before_comparators_and_outcomes() -> None:
    payload = json.loads(REPORT.read_text("utf-8"))
    failed = {
        name for name, passed in payload["support_checks"].items() if not passed
    }

    assert failed == {"selection_side_support", "maximum_split_month_share"}
    assert all(payload["identity_checks"].values())
    assert payload["source_support_passed"] is False
    assert payload["advance_to_comparator_novelty_freeze"] is False
    assert (
        payload["decision"]
        == "retire_IVPLH_72_unchanged_before_comparators_and_outcomes"
    )
    assert payload["outcomes_opened"] is False
    assert payload["post_entry_return_computed"] is False
    assert payload["funding_loaded"] is False
    assert payload["comparator_rows_decoded"] is False

    train = payload["primary_statistics"]["train"]
    selection = payload["primary_statistics"]["selection"]
    assert (train["events"], train["long"], train["short"]) == (33, 20, 13)
    assert (selection["events"], selection["long"], selection["short"]) == (
        18,
        3,
        15,
    )
    assert train["maximum_month_share"] == 3 / 33
    assert selection["maximum_month_share"] == 4 / 18
    assert selection["long_share"] == 3 / 18

    boundary = payload["outcome_boundary"]
    for field in (
        "comparator_rows_decoded",
        "post_entry_price_rows_decoded",
        "funding_rows_decoded",
        "future_return_rows_decoded",
        "return_or_pnl_fields_decoded",
        "pnl_cagr_mdd_values_decoded",
        "network_calls",
    ):
        assert boundary[field] == 0
