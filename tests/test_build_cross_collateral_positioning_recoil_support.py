from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_cross_collateral_positioning_recoil_support as support


def test_prior_midrank_excludes_current_and_fails_closed() -> None:
    index = pd.date_range("2023-01-01", periods=6, freq="h", tz="UTC")
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, 5.0], index=index)
    valid = pd.Series(True, index=index)
    rank = support.prior_midrank(values, valid, 4)
    assert np.isnan(rank.iloc[3])
    assert rank.iloc[4] == 1.0
    assert rank.iloc[5] == 0.75

    invalid = valid.copy()
    invalid.iloc[2] = False
    failed = support.prior_midrank(values, invalid, 4)
    assert np.isnan(failed.iloc[4])
    assert np.isnan(failed.iloc[5])


def test_false_to_true_reserves_one_clock_per_episode() -> None:
    flag = pd.Series([False, True, True, False, True, True, False])
    assert support.false_to_true(flag).tolist() == [
        False,
        True,
        False,
        False,
        True,
        False,
        False,
    ]


def test_build_features_requires_exact_contiguous_history() -> None:
    manifest = support.load_preregistration()
    index = pd.date_range("2022-01-01", periods=2500, freq="5min", tz="UTC")
    source = pd.DataFrame(
        {
            "um_sum_open_interest_value": np.exp(np.linspace(20.0, 20.2, len(index))),
            "cm_sum_open_interest": np.exp(np.linspace(15.0, 15.1, len(index))),
            "um_sum_taker_long_short_vol_ratio": 1.0
            + 0.1 * np.sin(np.arange(len(index))),
            "cm_sum_taker_long_short_vol_ratio": 1.0
            + 0.1 * np.cos(np.arange(len(index))),
            "source_complete": True,
        },
        index=index,
    )
    feature = support.build_features(source, manifest)
    assert feature["feature_complete"].any()
    source.loc[index[2200], "source_complete"] = False
    broken = support.build_features(source, manifest)
    later = broken.loc[broken.index >= index[2200]]
    assert not later["feature_complete"].any()


def test_support_loader_does_not_open_market_or_funding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = support.load_preregistration()
    original = pd.read_csv
    opened: list[str] = []

    def guarded(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        opened.append(str(path))
        forbidden = ("kline_reference", "funding_marks")
        assert not any(token in str(path) for token in forbidden)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded)
    frame = support.load_source(manifest)
    assert len(frame) == 261_216
    assert opened == [manifest["source_contract"]["positioning"]]


def test_real_support_selects_q_085_without_outcomes() -> None:
    report = json.loads(Path(support.DEFAULT_OUTPUT).read_text())
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["support_passed"] is True
    assert report["advance_to_stage1_outcomes"] is True
    assert report["selected_q"] == 0.85
    assert report["sealed"]["market_rows_parsed"] == 0
    assert report["sealed"]["funding_rows_parsed"] == 0
    assert report["sealed"]["simulations_run"] == 0
    by_q = {item["q"]: item for item in report["q_evaluations"]}
    assert by_q[0.85]["counts"] == {
        "train": 113,
        "2021_partial": 35,
        "2022": 78,
        "2023": 49,
        "2023_H1": 37,
        "2023_H2": 12,
    }
    assert by_q[0.85]["passed"] is True
    assert by_q[0.90]["passed"] is False
    assert by_q[0.90]["floors"]["2023_halves"] is False


def test_written_clock_hash_and_report_manifest_replay() -> None:
    report = json.loads(Path(support.DEFAULT_OUTPUT).read_text())
    clock_path = Path(report["clocks"]["path"])
    assert (
        hashlib.sha256(clock_path.read_bytes()).hexdigest()
        == report["clocks"]["sha256"]
    )
    assert report["manifest_hash"] == support._canonical_hash(
        {key: value for key, value in report.items() if key != "manifest_hash"}
    )
    clocks = pd.read_csv(clock_path)
    assert len(clocks) == report["clocks"]["rows"]
    assert set(clocks["control"]) == set(support.CONTROL_ORDER)
    assert set(clocks["side"]) == {-1, 1}


def test_tampered_preregistration_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(Path(support.PREREGISTRATION).read_text())
    payload["policy"]["taker_rank_floor"] = 0.0
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        support.load_preregistration(path)
