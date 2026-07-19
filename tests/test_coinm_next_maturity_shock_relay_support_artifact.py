from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import build_coinm_next_maturity_shock_relay_support as s


SUPPORT = Path("results/coinm_next_maturity_shock_relay_support_2026-07-19.json")
CLOCK = Path("data/coinm_next_maturity_shock_relay_clocks_2020_2023.csv.gz")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_support_passes_without_execution_outcomes() -> None:
    report = json.loads(SUPPORT.read_text(encoding="utf-8"))
    assert _sha256(SUPPORT) == "ca055f162eaee47efc6500ba4f178b0cfe2a0e701f337c6ce306cdc3e5ae368a"
    assert _sha256(CLOCK) == "e81450d4e76ffd0ce2ae96edf97106f2f4c473da233be0db18dc2530c8da8e87"
    assert report["manifest_hash"] == s._canonical_hash(
        {key: value for key, value in report.items() if key != "manifest_hash"}
    )
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["support_passed"] is True
    assert report["failed_checks"] == []
    assert report["advance_to_train_outcomes"] is True
