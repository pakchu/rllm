from __future__ import annotations

import pandas as pd

from training import evaluate_high_volatility_exchange_reserve_pressure_relay_gross9_novelty as n


def clock(entries: list[str], sides: list[int]) -> pd.DataFrame:
    entry = pd.to_datetime(entries, utc=True)
    return pd.DataFrame({
        "entry_time": entry,
        "exit_time": entry + pd.Timedelta(hours=24),
        "side": sides,
    })


def test_frozen_predecessor_hashes_limits_and_policy():
    assert n.POLICY == "HVEXRP-24"
    assert n.sha(n.PREREG) == n.PREREG_SHA
    assert n.sha(n.SUPPORT) == n.SUPPORT_SHA
    assert n.sha(n.CLOCK) == n.CLOCK_SHA
    assert n.LIMITS == {
        "exact_entry_jaccard": 0.10,
        "one_to_one_6h_max_matched_share": 0.35,
        "occupied_5m_bar_jaccard": 0.25,
        "absolute_signed_exposure_pearson": 0.35,
    }


def test_pair_wrapper_uses_frozen_limits_without_opening_outcomes():
    candidate = clock(
        ["2023-07-02T01:00:00Z", "2023-07-05T07:00:00Z"],
        [1, -1],
    )
    comparator = clock(
        ["2023-07-03T12:00:00Z", "2023-07-07T18:00:00Z"],
        [-1, 1],
    )
    result = n.evaluate_pair(candidate, comparator)
    assert result["metrics"]["exact_entry_jaccard"] == 0.0
    assert result["metrics"]["one_to_one_6h_max_matched_share"] == 0.0
    assert result["checks"] == {
        key: result["metrics"][key] <= limit for key, limit in n.LIMITS.items()
    }


def test_manifest_loader_rejects_hash_drift(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"manifest_hash":"bad","label":"한글"}\n', encoding="utf-8")
    try:
        n.load_manifest(path)
    except RuntimeError as exc:
        assert "manifest drift" in str(exc)
    else:
        raise AssertionError("manifest drift was accepted")
