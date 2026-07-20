from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import evaluate_bank_deposit_secured_repo_concordance_support as support


RESULT = Path(
    "results/bank_deposit_secured_repo_concordance_support_2026-07-20.json"
)
CLOCK = Path(
    "results/bank_deposit_secured_repo_concordance_clocks_2026-07-20.csv.gz"
)
EVALUATOR = Path(
    "training/evaluate_bank_deposit_secured_repo_concordance_support.py"
)
RESULT_SHA256 = "74fea33f6b65eed01710824d57e339fbbec9245686ef13a58510a98cbdd1217c"
CLOCK_SHA256 = "1ff3a6075e3ceff928e1dd19d05880dbe9dbab0e07d79b853146d7b4c8f6cabc"
MANIFEST_HASH = "5cfe9968c579a64e04879e3b47d811e6797da93eaed073255c9ab7d2b5f66f5a"
EVALUATOR_SHA256 = "bb7042b6ab621bc8cda05c05129c0cc80380cb974eceaf9abbd55ca664f9267b"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_bdrc_source_result_rejects_without_opening_outcomes() -> None:
    assert _sha256(RESULT) == RESULT_SHA256
    assert _sha256(CLOCK) == CLOCK_SHA256
    assert _sha256(EVALUATOR) == EVALUATOR_SHA256

    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == MANIFEST_HASH == support.canonical_hash(core)
    assert payload["decision"] == "REJECT_SOURCE_SUPPORT_NO_REPAIR"
    assert payload["support_passed"] is False
    assert payload["evaluator_authorized"] is False
    assert payload["no_repair"] is True
    assert set(payload["failed_gates"]) == {
        "train_total_at_least_45",
        "each_train_year_at_least_10",
        "train_each_side_at_least_15",
    }

    train = payload["primary"]["train_2020_2022"]
    assert (train["events"], train["long"], train["short"]) == (37, 11, 26)
    assert train["repo_abs_le_3bp"] == 26
    assert payload["primary"]["2021"]["events"] == 3
    selection = payload["primary"]["selection_2023"]
    assert (selection["events"], selection["long"], selection["short"]) == (
        17,
        9,
        8,
    )

    assert payload["outcome_boundary"] == {
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "return_rows_loaded": 0,
        "market_values_read": 0,
        "funding_values_read": 0,
        "return_or_pnl_fields": 0,
        "post_2023_source_rows_loaded": 0,
        "outcomes_opened": False,
    }
    assert payload["clocks"] == {
        "path": str(CLOCK),
        "sha256": CLOCK_SHA256,
        "rows": 905,
        "primary_rows": 75,
        "control_rows": 830,
    }
    controls = payload["control_clock_diagnostics"]
    assert controls["direction_flip"]["exact_entry_overlap"]["jaccard"] == 1.0
    assert controls["direction_flip"][
        "signed_5m_occupied_exposure_correlation"
    ] == -0.9999999999999998
