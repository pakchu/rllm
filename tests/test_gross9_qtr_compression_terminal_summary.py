from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/gross9_qtr_compression_terminal_summary_2026-09-03.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_terminal_summary_is_hash_bound_and_stops_after_test2024() -> None:
    report = json.loads(SUMMARY.read_text(encoding="utf-8"))
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == _canonical_hash(core)
    for artifact in report["artifacts"].values():
        assert _sha(ROOT / artifact["path"]) == artifact["sha256"]
    assert report["trade_counts"]["train"]["sleeve_intervals"] == 150
    assert report["trade_counts"]["test2024"]["sleeve_intervals"] == 348
    assert report["trade_counts"]["test2024"]["net_transitions"] == 507
    assert report["sequence"]["test2024_passed"] is False
    assert report["sequence"]["eval2025_opened"] is False
    assert report["sequence"]["final2026_opened"] is False
    assert report["operational_status"]["live_capital_authorized"] is False
    assert report["decision"] == "terminal_reject_no_repair"
