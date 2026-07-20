from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import evaluate_btcdom_leverage_polarity_decomposition as evaluator


MANIFEST = Path(
    "results/btcdom_leverage_polarity_decomposition_train_execution_source_2026-07-20.json"
)
MARKET = Path(
    "data/btcdom_leverage_polarity_decomposition_execution/train/BTCUSDT_5m.csv.gz"
)
FUNDING = Path(
    "data/btcdom_leverage_polarity_decomposition_execution/train/"
    "BTCUSDT_funding_marks.csv.gz"
)
MANIFEST_SHA256 = "a47beeb9822c4319101a378e305bf0c28770dc191338467e0ff2ad309dfbd209"
MARKET_SHA256 = "7e1aab436a96c83680be45047f4dd36a62fa72cb9b8f3991d32e16d4fe1a4be3"
FUNDING_SHA256 = "2c3eb607ae343201ee2d12b29fd07fb25dad4baa059340eef821155a6c3ea2c8"
MANIFEST_HASH = "d688bc802b60c6e3667ab4fcb2e92540c3c7cf064b79f44bfa2a666824811b0a"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dlpd_train_source_is_exact_and_outcome_free() -> None:
    assert _sha256(MANIFEST) == MANIFEST_SHA256
    assert _sha256(MARKET) == MARKET_SHA256
    assert _sha256(FUNDING) == FUNDING_SHA256
    payload = json.loads(MANIFEST.read_text())
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert payload["physical_window"] == [
        "2022-01-01T00:00:00+00:00",
        "2023-01-01T00:00:00+00:00",
    ]
    assert payload["market"]["rows"] == 105_120
    assert payload["funding"]["rows"] == 1_095
    assert payload["strategy_outcomes_calculated"] is False
    assert payload["post_stage_numeric_rows_parsed"] == 0
    assert payload["market"]["diagnostics"]["rows"] == 105_120
    assert payload["funding"]["diagnostics"]["rows"] == 1_095
    assert payload["funding"]["diagnostics"]["maximum_absolute_grid_offset_ms"] == 31.0


def test_dlpd_train_source_reconstructs_from_frozen_parent() -> None:
    freeze = evaluator.verify_evaluator_freeze()
    payload = evaluator._load_stage_source("train", freeze=freeze)
    assert payload["market"]["sha256"] == MARKET_SHA256
    assert payload["funding"]["sha256"] == FUNDING_SHA256
