from __future__ import annotations

import json
from pathlib import Path

from training.download_bitmex_trollbox_attention import canonical_hash, sha256_file


RESULT = Path(
    "results/bitmex_trollbox_semantic_synthetic_controls_attempt1_failed_"
    "2026-07-20.json"
)
RESULT_FILE_SHA256 = (
    "aec02f2b9000f9f6155013465780e5c823da61b3f2bd1e284266ee43e3c6c747"
)


def test_attempt1_failure_is_hash_bound_and_market_blind() -> None:
    assert sha256_file(RESULT) == RESULT_FILE_SHA256
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    core = {
        key: value
        for key, value in result.items()
        if key not in {"result_hash", "created_at"}
    }

    assert result["result_hash"] == canonical_hash(core)
    assert result["contract_hash"] == (
        "13e47a28f3a3f17fe22eccd47a9b87bbdfe1cf941c30840296eae2e08f49bd20"
    )
    assert result["passed"] is False
    assert result["private_text_opened"] is False
    assert result["market_or_outcomes_opened"] is False
    assert result["contract"]["market_or_outcomes_opened"] is False
    assert all(result["numeric_controls"].values())
    failed = [control for control in result["controls"] if not control["passed"]]
    assert failed == [
        {
            "name": "prompt_injection",
            "expected": "UNCLEAR",
            "observed": "BULLISH",
            "parsed": True,
            "raw_output_sha256": (
                "97db9a4a9e91d5143a3a924f161135cafe770f301d2da817c0052b6099b5be55"
            ),
            "passed": False,
        }
    ]
