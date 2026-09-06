import pandas as pd
import pytest

from training import evaluate_high_volatility_cross_alt_rank_persistence_relay_gross9_novelty as novelty


def test_frozen_predecessor_hashes_and_limits():
    assert novelty.sha(novelty.PREREG) == novelty.PREREG_SHA
    assert novelty.sha(novelty.SUPPORT) == novelty.SUPPORT_SHA
    assert novelty.sha(novelty.CLOCK) == novelty.CLOCK_SHA
    assert novelty.CLOCK_ROWS == 819
    assert novelty.POLICY == "HVCARP-8"
    assert novelty.LIMITS == {
        "exact_entry_jaccard": 0.10,
        "one_to_one_6h_max_matched_share": 0.35,
        "occupied_5m_bar_jaccard": 0.25,
        "absolute_signed_exposure_pearson": 0.35,
    }


def _clock(entries, hold_hours=6, sides=None):
    entries = pd.to_datetime(entries, utc=True)
    if sides is None:
        sides = [1] * len(entries)
    return pd.DataFrame(
        {
            "entry_time": entries,
            "exit_time": entries + pd.Timedelta(hours=hold_hours),
            "side": sides,
        }
    )


def test_disjoint_pair_passes_all_limits():
    candidate = _clock(["2024-01-01T00:05:00Z", "2024-01-11T00:05:00Z"], sides=[1, -1])
    comparator = _clock(["2024-01-06T12:00:00Z", "2024-01-16T12:00:00Z"])
    result = novelty.evaluate_pair(candidate, comparator)
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_exact_overlap_fails():
    candidate = _clock(["2024-01-01T00:05:00Z", "2024-01-11T00:05:00Z"])
    comparator = candidate.copy()
    result = novelty.evaluate_pair(candidate, comparator)
    assert result["passed"] is False
    assert result["checks"]["exact_entry_jaccard"] is False


def test_result_write_is_deterministic_and_immutable(tmp_path):
    output = tmp_path / "novelty.json"
    content = b'{"manifest_hash":"frozen"}\n'
    novelty.write_immutable(output, content)
    novelty.write_immutable(output, content)
    assert output.read_bytes() == content
    with pytest.raises(RuntimeError, match="immutable HVCARP novelty result"):
        novelty.write_immutable(output, b"drift\n")
