import hashlib
import json
from pathlib import Path

import pandas as pd

from training import build_korean_cash_leadership_relay_support as support


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_kclr_source_rejection_is_outcome_sealed():
    assert sha(support.RESULT) == "fffaec7282ab5c02338003412bba4563b737d07cbe0aeaba79235b88c10353d5"
    result = json.loads(support.RESULT.read_text())
    assert result["policy_id"] == "KCLR-12"
    assert result["support_passed"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [result["support"][stage]["events"] for stage in ("train", "test", "eval", "final")] == [0, 0, 0, 8]


def test_kclr_hashes_bind_sparse_exact_path_and_nonpromotable_controls():
    result = json.loads(support.RESULT.read_text())
    assert result["manifest_hash"] == support.canonical_hash({key: value for key, value in result.items() if key != "manifest_hash"})
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
    features = pd.read_csv(support.FEATURE_PANEL, compression="gzip")
    assert len(features) == 277
    assert features.session_date.min() == "2025-10-20"
    assert features.session_date.max() == "2026-07-31"
    assert features.variation_rank.notna().sum() == 97
    for control in result["controls"].values():
        assert control["promotion_authorized"] is False
        assert control["sha256"] == sha(Path(control["path"]))
