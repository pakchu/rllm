from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import evaluate_exact_maturity_fee_cadence_polarity_support as support


RESULT = Path(
    "results/exact_maturity_fee_cadence_polarity_support_2026-07-20.json"
)
CLOCK = Path(
    "results/exact_maturity_fee_cadence_polarity_clocks_2026-07-20.csv"
)
RESULT_SHA256 = "1cfd359de4412972ed133e56523d3c713c372cb307304ce3bcd42c338b9e045d"
CLOCK_SHA256 = "31af41f42ffe4dc73f0ff35ccf278e38c856d224184e802e46b370650d35951d"
RESULT_HASH = "b32ec03347bc4ef3a80fc293ca1b122101286bbbe76e67d6c29801b1ac932880"
SUPPORT_SOURCE_SHA256 = (
    "c58327d32432ac07a8072cbca371a32fb849ca083afefba2f1cdd7b42fc1df3d"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_emfc_source_result_rejects_without_opening_outcomes() -> None:
    assert _sha256(RESULT) == RESULT_SHA256
    assert _sha256(CLOCK) == CLOCK_SHA256
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "result_hash"}
    assert payload["result_hash"] == RESULT_HASH == support.canonical_hash(core)
    assert payload["outcomes_opened"] is False
    assert payload["outcome_boundary"] == support.OUTCOME_BOUNDARY
    assert all(value == 0 for value in payload["outcome_boundary"].values())
    assert payload["source_integrity"]["passed"] is True
    assert payload["event_support"]["passed"] is True
    assert payload["feature_novelty"]["passed"] is True
    assert payload["exposure_novelty"]["passed"] is False
    assert payload["support_gate"]["passed"] is False
    failed = {
        name
        for name, passed in payload["support_gate"]["checks"].items()
        if not passed
    }
    assert failed == {
        "exposure_novelty_pseudo_maturity_99",
        "exposure_novelty_pseudo_maturity_101",
    }
    assert payload["support_source"]["sha256"] == SUPPORT_SOURCE_SHA256
    assert payload["clock"]["sha256"] == CLOCK_SHA256
    assert payload["clock"]["rows_by_clock"]["primary"] == 175
