from __future__ import annotations

import hashlib
import json
from pathlib import Path


TRAIN = Path(
    "results/coinm_next_maturity_shock_relay_train_2020_2022_2026-07-19.json"
)
TEST = Path("results/coinm_next_maturity_shock_relay_test_2023_2026-07-19.json")


def test_cmsr_train_rejection_keeps_2023_physically_sealed() -> None:
    assert hashlib.sha256(TRAIN.read_bytes()).hexdigest() == (
        "602569b5c0bc60a320c956cbb850423780050eb03b2944c386fc15bff55fa9e8"
    )
    report = json.loads(TRAIN.read_text(encoding="utf-8"))
    assert report["manifest_hash"] == (
        "19147ff20d5decef5297f91b4be633e193bb956f751a74b14b48730152e3c66a"
    )
    assert report["train_passed"] is False
    assert report["opened_windows"] == ["train_2020_2022"]
    assert report["sealed_windows"] == ["test_2023", "2024_plus"]
    assert report["disposition"] == "REJECT_KEEP_2023_SEALED"
    assert report["execution_diagnostics"]["market"]["other_stage_files_opened"] == 0
    assert report["execution_diagnostics"]["funding"]["boundary_row_read"] is False
    assert report["result"]["primary_headline"]["trades"] == 93
    assert report["result"]["primary_headline"]["absolute_return_pct"] < 0.0
    assert not TEST.exists()
