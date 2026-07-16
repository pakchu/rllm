from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from training.build_causal_residual_expert_switcher_2026_support import (
    DEFAULT_MANIFEST,
    FORBIDDEN_OUTCOME_TOKENS,
    assert_clock_contract,
)
from training.export_leave_one_out_residual_exhaustion_sources import sha256_file
from training.preregister_causal_residual_expert_switcher_2026 import canonical_hash


def test_cres_2026_support_artifact_is_outcome_blind_and_hash_bound() -> None:
    manifest = json.loads(Path(DEFAULT_MANIFEST).read_text())
    body = {key: value for key, value in manifest.items() if key not in {"manifest_hash", "created_at"}}
    assert canonical_hash(body) == manifest["manifest_hash"]
    assert manifest["post_entry_2026_strategy_returns_calculated"] is False
    assert manifest["outcome_like_columns_present"] is False
    assert manifest["support"]["passes_support"] is True
    assert manifest["support"]["events"] == 68
    assert sha256_file(Path(manifest["clock_path"])) == manifest["clock_sha256"]
    assert sha256_file(Path(manifest["historical_seed_path"])) == manifest[
        "historical_seed_sha256"
    ]
    clock = pd.read_csv(manifest["clock_path"])
    assert_clock_contract(clock)
    for column in clock.columns:
        assert not (set(column.lower().split("_")) & FORBIDDEN_OUTCOME_TOKENS)
