from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_block_arrival_throughput_elasticity_support as support


def _hash(height: int) -> str:
    return f"{height:064x}"


def _blocks(rows: int = 40, *, first_height: int = 700_000) -> pd.DataFrame:
    height = np.arange(first_height, first_height + rows, dtype=np.int64)
    timestamp = 1_600_000_000 + np.arange(rows, dtype=np.int64) * 600
    return pd.DataFrame(
        {
            "height": height,
            "id": [_hash(int(value)) for value in height],
            "previousblockhash": [_hash(int(value - 1)) for value in height],
            "timestamp": timestamp,
            "mediantime": timestamp - 600,
            "tx_count": 1_000 + np.arange(rows, dtype=np.int64) * 3,
            "size": np.full(rows, 1_000, dtype=np.int64),
            "weight": 2_000 + np.arange(rows, dtype=np.int64) * 5,
        }
    )


def _test_policy(**changes: object) -> support.Policy:
    return replace(
        support.Policy(),
        reference_packets=4,
        z_threshold=0.5,
        **changes,
    )


def test_strict_prior_robust_z_excludes_current_future_and_nan() -> None:
    values = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 100.0])
    actual = support._strict_prior_robust_z(
        values,
        reference=4,
        consistency_scale=1.0,
        batch_size=2,
    )
    assert np.isnan(actual[:5]).all()
    assert actual[5] == pytest.approx((5.0 - 2.5) / 1.0)
    changed = values.copy()
    changed[-1] = -100_000.0
    changed_z = support._strict_prior_robust_z(
        changed,
        reference=4,
        consistency_scale=1.0,
        batch_size=1,
    )
    assert changed_z[5] == actual[5]


def test_build_features_uses_six_past_blocks_and_six_confirmations() -> None:
    blocks = _blocks()
    blocks.loc[8, "timestamp"] = int(blocks["timestamp"].max()) + 3_600
    policy = _test_policy()
    features = support.build_features(blocks, policy)
    first = features.iloc[0]
    assert first["packet_end_height"] == blocks.iloc[6]["height"]
    assert first["confirmation_height"] == blocks.iloc[12]["height"]
    assert first["elapsed_seconds"] == 3_600
    expected_weight = blocks.iloc[1:7]["weight"].sum()
    expected_tx = blocks.iloc[1:7]["tx_count"].sum()
    assert first["weight_log"] == pytest.approx(np.log(expected_weight / 3_600))
    assert first["tx_log"] == pytest.approx(np.log(expected_tx / 3_600))
    assert blocks.loc[8, "timestamp"] > blocks.loc[12, "timestamp"]
    raw_available = int(blocks.iloc[:13]["timestamp"].max()) + 7_200
    boundary = ((raw_available + 299) // 300) * 300
    assert first["entry_time"] == pd.to_datetime(boundary + 300, unit="s", utc=True)
    assert features.iloc[-1]["confirmation_height"] == blocks.iloc[-1]["height"]


def test_future_throughput_change_cannot_rewrite_prior_feature() -> None:
    blocks = _blocks(60)
    policy = _test_policy()
    baseline = support.build_features(blocks, policy)
    changed = blocks.copy()
    changed.loc[changed.index >= 40, "weight"] *= 100
    changed.loc[changed.index >= 40, "tx_count"] *= 100
    rewritten = support.build_features(changed, policy)
    prior = baseline["packet_end_height"] < int(blocks.iloc[40]["height"])
    columns = ["weight_log", "tx_log", "weight_z", "tx_z", "state", "onset"]
    pd.testing.assert_frame_equal(
        baseline.loc[prior, columns].reset_index(drop=True),
        rewritten.loc[prior, columns].reset_index(drop=True),
    )


def test_nonpositive_elapsed_span_is_invalid_and_not_clipped() -> None:
    blocks = _blocks()
    blocks.loc[6, "timestamp"] = blocks.loc[0, "timestamp"]
    features = support.build_features(blocks, _test_policy())
    first = features.iloc[0]
    assert first["elapsed_seconds"] == 0
    assert not bool(first["raw_valid"])
    assert np.isnan(first["weight_log"])
    assert np.isnan(first["tx_log"])
    assert not bool(first["onset"])


def test_invalid_run_does_not_manufacture_same_side_transition() -> None:
    same_side = support._state_onsets(
        np.array([1, 0, 1], dtype=np.int8),
        np.array([True, False, True]),
    )
    assert same_side.tolist() == [True, False, False]
    opposite_side = support._state_onsets(
        np.array([1, 0, -1], dtype=np.int8),
        np.array([True, False, True]),
    )
    assert opposite_side.tolist() == [True, False, True]


def test_build_features_rejects_price_or_noncontiguous_height_input() -> None:
    with_price = _blocks()
    with_price["price"] = 1.0
    with pytest.raises(RuntimeError, match="block-only frozen columns"):
        support.build_features(with_price, _test_policy())
    gap = _blocks().drop(index=10).reset_index(drop=True)
    with pytest.raises(RuntimeError, match="heights must be contiguous"):
        support.build_features(gap, _test_policy())


def test_schedule_uses_height_order_and_fixed_24_hour_nonoverlap() -> None:
    policy = _test_policy()
    features = pd.DataFrame(
        {
            "packet_end_height": [10, 11, 12, 13],
            "confirmation_height": [16, 17, 18, 19],
            "elapsed_seconds": [3_600] * 4,
            "weight_log": [1.0] * 4,
            "tx_log": [1.0] * 4,
            "weight_z": [2.0, -2.0, -2.0, 2.0],
            "tx_z": [2.0, -2.0, -2.0, 2.0],
            "state": [1, -1, -1, 1],
            "onset": [True, True, True, True],
            "entry_time": pd.to_datetime(
                [
                    "2021-01-01T00:00:00Z",
                    "2021-01-01T12:00:00Z",
                    "2021-01-02T00:00:00Z",
                    "2021-01-03T00:00:00Z",
                ],
                utc=True,
            ),
        }
    )
    clock = support.schedule_clock(features, policy)
    assert clock["packet_end_height"].tolist() == [10, 12, 13]
    assert clock["side"].tolist() == [1, -1, 1]
    assert clock["state"].tolist() == ["HIGH", "LOW", "HIGH"]
    assert (
        clock["exit_time"] - clock["entry_time"]
    ).eq(pd.Timedelta(hours=24)).all()


def _dense_clock() -> pd.DataFrame:
    entry = pd.date_range(
        "2021-01-02T00:00:00Z",
        "2023-12-29T00:00:00Z",
        freq="3D",
    )
    state = np.where(np.arange(len(entry)) % 2 == 0, "HIGH", "LOW")
    return pd.DataFrame({"entry_time": entry, "state": state})


def test_support_summary_enforces_both_sides_and_calendar_dispersion() -> None:
    passing = support.support_summary(_dense_clock())
    assert passing["passed"] is True
    assert passing["side_counts"]["selection"]["HIGH"] >= 12
    one_sided = _dense_clock()
    one_sided["state"] = "HIGH"
    failed = support.support_summary(one_sided)
    assert failed["passed"] is False
    assert failed["checks"]["selection_each_side"] is False


def test_maximum_invalid_run() -> None:
    assert support._maximum_true_run(np.array([False, True, True, False, True])) == 2
    assert support._maximum_true_run(np.array([], dtype=bool)) == 0


def test_source_support_checks_each_frozen_integrity_gate() -> None:
    policy = support.Policy()
    passing = pd.DataFrame(
        {
            "raw_valid": np.ones(3_000, dtype=bool),
            "confirmation_height": np.full(3_000, 823_785),
        }
    )
    assert support.source_support_summary(passing, policy)["passed"] is True

    low_ratio = passing.copy()
    low_ratio.loc[:15, "raw_valid"] = False
    ratio_result = support.source_support_summary(low_ratio, policy)
    assert ratio_result["checks"]["positive_span_ratio"] is False

    long_run = passing.copy()
    long_run.loc[:12, "raw_valid"] = False
    run_result = support.source_support_summary(long_run, policy)
    assert run_result["checks"]["positive_span_ratio"] is True
    assert run_result["checks"]["maximum_invalid_span_run"] is False

    escaped = passing.copy()
    escaped.loc[0, "confirmation_height"] = 823_786
    containment = support.source_support_summary(escaped, policy)
    assert containment["checks"]["confirmation_containment"] is False


def test_load_source_rejects_manifest_hash_before_reading_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv.gz"
    source_path.write_bytes(b"not read")
    manifest = {
        "protocol_version": support.block_source.PROTOCOL_VERSION,
        "manifest_hash": "bad",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        support.load_source(str(source_path), str(manifest_path))


def test_frozen_source_loader_identity_is_not_self_attested(tmp_path: Path) -> None:
    forged_loader = tmp_path / "download_bitcoin_block_summaries.py"
    forged_loader.write_text("# structurally plausible but not frozen\n")
    with pytest.raises(RuntimeError, match="frozen SHA-256"):
        support._validate_frozen_loader(forged_loader)
    support._validate_frozen_loader(Path(support.block_source.__file__))


def test_source_config_and_cross_host_anchors_are_frozen() -> None:
    config = {
        "start_height": support.block_source.FROZEN_START_HEIGHT,
        "end_height": support.block_source.FROZEN_END_HEIGHT,
        "end_timestamp_exclusive": support.block_source.FIRST_2024_TIMESTAMP,
        "base_url": support.FROZEN_RESEARCH_BASE_URL,
    }
    support._validate_source_config(config)
    with pytest.raises(RuntimeError, match="prefix and host"):
        support._validate_source_config(
            {**config, "base_url": "https://blockstream.info/api"}
        )

    anchors = pd.DataFrame(
        {
            "height": list(support.FROZEN_BLOCK_ANCHORS),
            "id": list(support.FROZEN_BLOCK_ANCHORS.values()),
        }
    )
    support._validate_block_anchors(anchors)
    anchors.loc[0, "id"] = "0" * 64
    with pytest.raises(RuntimeError, match="anchor mismatch"):
        support._validate_block_anchors(anchors)


def test_frozen_policy_rejects_any_repair() -> None:
    support._validate_policy(support.Policy())
    with pytest.raises(RuntimeError, match="differs from preregistration"):
        support._validate_policy(replace(support.Policy(), z_threshold=1.0))


def test_run_rejects_artifact_alias_before_reading_source(tmp_path: Path) -> None:
    shared = tmp_path / "shared.csv"
    with pytest.raises(ValueError, match="distinct from inputs"):
        support.run(
            source_csv=str(shared),
            source_manifest=str(tmp_path / "manifest.json"),
            output=str(shared),
            clock_output=str(tmp_path / "clock.csv"),
        )
