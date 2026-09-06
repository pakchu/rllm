from __future__ import annotations

import pandas as pd

from training import evaluate_high_volatility_credit_quality_transition_relay_gross9_novelty as novelty


def _clock(entries: list[str], sides: list[int]) -> pd.DataFrame:
    entry = pd.to_datetime(entries, utc=True)
    return pd.DataFrame({"entry_time": entry, "exit_time": entry + pd.Timedelta(hours=8), "side": sides})


def test_reuses_every_hvsof_authority_metric_and_limit() -> None:
    assert novelty.gross9 is novelty.hvsof.gross9
    assert novelty.metric is novelty.hvsof.metric
    assert novelty.LIMITS is novelty.hvsof.LIMITS
    assert novelty.evaluate_pair is novelty.hvsof.evaluate_pair
    result = novelty.evaluate_pair(
        _clock(["2023-07-01T00:05:00Z"], [1]),
        _clock(["2023-07-03T00:05:00Z"], [-1]),
    )
    assert set(result["checks"]) == set(novelty.LIMITS)
    assert result["passed"] is True


def test_frozen_controls_verify_all_hashes() -> None:
    assert novelty.sha(novelty.PREREG) == novelty.PREREG_SHA
    assert novelty.sha(novelty.SUPPORT) == novelty.SUPPORT_SHA
    registration, support, manifest = novelty.load_frozen_controls()
    assert registration["candidate_family"] == [novelty.POLICY]
    assert support["clock"] == novelty.CLOCK
    assert novelty.sha(novelty.CLOCK["path"]) == novelty.CLOCK["sha256"]
    assert set(manifest["clocks"]) == set(novelty.gross9.EXPECTED_WEIGHTS)


def test_exact_entry_collision_fails() -> None:
    candidate = _clock(["2023-07-01T00:05:00Z"], [1])
    result = novelty.evaluate_pair(candidate, candidate.copy())
    assert result["checks"]["exact_entry_jaccard"] is False
    assert result["passed"] is False


def test_run_twice_is_deterministic_and_economics_blind(tmp_path) -> None:
    output = tmp_path / "gross9.json"
    first = novelty.run(output)
    first_bytes = output.read_bytes()
    second = novelty.run(output)
    assert output.read_bytes() == first_bytes
    assert second == first
    assert tuple(first["gross9_sleeves"]) == tuple(novelty.gross9.EXPECTED_WEIGHTS)
    assert first["advance_to_economic_outcomes"] == all(
        sleeve["passed"] for sleeve in first["gross9_sleeves"].values()
    )
    assert first["evidence_boundary"]["price_or_return_rows_opened"] == 0
    assert first["evidence_boundary"]["outcomes_opened"] is False
    assert first["manifest_hash"] == novelty.canonical_hash(
        {key: value for key, value in first.items() if key != "manifest_hash"}
    )
