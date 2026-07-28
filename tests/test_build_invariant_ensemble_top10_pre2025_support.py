from __future__ import annotations

import gzip
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import training.build_invariant_ensemble_top10_pre2025_support as builder
from training.search_river_contextual_utility_alpha import effective_selection_signal_hash


def test_support_schema_rejects_outcome_like_columns_and_future_rows() -> None:
    good = pd.DataFrame(
        [
            {
                "pre_evaluation_rank": 1,
                "signal_position": 143,
                "signal_date": "2023-01-01 11:55:00",
                "side": "long",
            }
        ],
        columns=builder.SUPPORT_COLUMNS,
    )
    builder.verify_support_frame(good)

    bad_column = good.assign(forward_return=0.0)
    with pytest.raises(ValueError, match="outcome-like"):
        builder.verify_support_frame(bad_column)

    future = good.copy()
    future.loc[0, "signal_date"] = "2025-01-01 00:00:00"
    with pytest.raises(ValueError, match="outside"):
        builder.verify_support_frame(future)


def test_deterministic_csv_gzip_is_byte_stable_and_header_bound(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [{"pre_evaluation_rank": 1, "signal_position": 143, "signal_date": "2023-01-01 11:55:00", "side": "long"}],
        columns=builder.SUPPORT_COLUMNS,
    )
    a = tmp_path / "a.csv.gz"
    b = tmp_path / "b.csv.gz"
    builder.deterministic_csv_gzip(frame, a)
    builder.deterministic_csv_gzip(frame, b)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes()[4:8] == b"\x00\x00\x00\x00"
    with gzip.open(a, "rt", encoding="utf-8") as handle:
        assert handle.read().splitlines()[0] == ",".join(builder.SUPPORT_COLUMNS)


def test_processed_frame_hash_is_order_and_value_sensitive() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "value": [1.0, np.nan],
        }
    )
    observed = builder.dataframe_sha256(frame)
    assert observed == builder.dataframe_sha256(frame.copy())
    assert observed != builder.dataframe_sha256(frame.iloc[::-1].reset_index(drop=True))
    changed = frame.copy()
    changed.loc[0, "value"] = 2.0
    assert observed != builder.dataframe_sha256(changed)


def test_effective_rows_match_frozen_hash_algorithm_and_detect_drift() -> None:
    n = 5000
    dates = pd.Series(pd.date_range("2023-01-01", periods=n, freq="5min"))
    market = pd.DataFrame({"date": dates})
    long_active = np.zeros(n, dtype=bool)
    short_active = np.zeros(n, dtype=bool)
    long_active[[143, 791, 1511]] = True
    short_active[[2303]] = True

    rows = builder.effective_signal_rows(
        dates,
        long_active,
        short_active,
        rank=4,
        window=("2023-01-01", "2023-02-01"),
        market_size=n,
    )
    expected = effective_selection_signal_hash(
        market,
        dates,
        long_active,
        short_active,
        window=("2023-01-01", "2023-02-01"),
    )
    assert builder.effective_hash_from_rows(
        rows,
        dates=dates,
        window=("2023-01-01", "2023-02-01"),
    ) == expected
    assert {row["side"] for row in rows} == {"long", "short"}
    assert {row["pre_evaluation_rank"] for row in rows} == {4}
    assert builder.verify_frozen_signal_hash(
        rank=4,
        expected_hash=expected,
        market=market,
        dates=dates,
        long_active=long_active,
        short_active=short_active,
        window=("2023-01-01", "2023-02-01"),
    ) == expected
    with pytest.raises(ValueError, match="hash drift"):
        builder.verify_frozen_signal_hash(
            rank=4,
            expected_hash="deadbeefdeadbeef",
            market=market,
            dates=dates,
            long_active=long_active,
            short_active=short_active,
            window=("2023-01-01", "2023-02-01"),
        )


def test_source_manifest_contract_validation(tmp_path: Path) -> None:
    frozen_rows = [
        {
            "signal_hash": f"{rank:016x}",
            "hold_bars": builder.HOLD_BARS,
            "anchor_stride_bars": builder.ANCHOR_STRIDE,
            "side_policy": "long",
        }
        for rank in builder.EXPECTED_TOP10_RANKS
    ]
    group = {
        "later_metrics_included": False,
        "selection_window": list(builder.SELECTION_WINDOW),
        "target": {"name": "tail"},
        "feature_sets": {"stable8": ["a"]},
        "data_sha256": {"market": "m", "funding": "f", "premium": "p"},
        "top10": frozen_rows,
    }
    ensemble = {
        **group,
        "source_manifest": "results/invariant_groupdro_top10_manifest_2026-07-13.json",
    }
    group_path = tmp_path / "invariant_groupdro_top10_manifest_2026-07-13.json"
    ensemble_path = tmp_path / "invariant_ensemble_uncertainty_top10_manifest_2026-07-13.json"
    group_path.write_text(json.dumps(group))
    ensemble_path.write_text(json.dumps(ensemble))
    loaded_group, loaded_ensemble = builder.load_and_validate_source_manifests(group_path, ensemble_path)
    assert loaded_group["feature_sets"] == loaded_ensemble["feature_sets"]

    ensemble["later_metrics_included"] = True
    ensemble_path.write_text(json.dumps(ensemble))
    with pytest.raises(ValueError, match="later_metrics_included=false"):
        builder.load_and_validate_source_manifests(group_path, ensemble_path)


def test_frozen_manifest_identity_constants_match_repository() -> None:
    observed = builder.validate_frozen_manifest_identities(
        builder.GROUPDRO_MANIFEST,
        builder.ENSEMBLE_MANIFEST,
    )
    assert observed == {
        "groupdro": builder.EXPECTED_GROUPDRO_MANIFEST_SHA256,
        "ensemble_uncertainty": builder.EXPECTED_ENSEMBLE_MANIFEST_SHA256,
    }


def test_strip_metric_fields_is_recursive() -> None:
    assert builder.strip_metric_fields(
        {
            "stream_id": "stable",
            "holdout2023": {"return_pct": 1.0},
            "nested": {"ratio": 2.0, "policy": "long"},
        }
    ) == {"stream_id": "stable", "nested": {"policy": "long"}}


def test_support_alignment_requires_all_ranks_and_non_overlapping_holds() -> None:
    dates = pd.Series(pd.date_range("2023-01-01", periods=150_000, freq="5min"))
    rows = []
    for rank in builder.EXPECTED_TOP10_RANKS:
        for pos in (143, 143 + 72 * 1800):
            rows.append(
                {
                    "pre_evaluation_rank": rank,
                    "signal_position": pos,
                    "signal_date": dates.iloc[pos].strftime("%Y-%m-%d %H:%M:%S"),
                    "side": "long",
                }
            )
    frame = pd.DataFrame(rows, columns=builder.SUPPORT_COLUMNS)
    counts = builder.verify_support_alignment(frame, dates=dates)
    assert set(counts) == {str(rank) for rank in builder.EXPECTED_TOP10_RANKS}

    missing = frame[frame["pre_evaluation_rank"] != 10]
    with pytest.raises(ValueError, match="every frozen"):
        builder.verify_support_alignment(missing, dates=dates)

    overlap = frame.copy()
    overlap.loc[len(overlap)] = {
        "pre_evaluation_rank": 1,
        "signal_position": 143 + builder.ANCHOR_STRIDE,
        "signal_date": dates.iloc[
            143 + builder.ANCHOR_STRIDE
        ].strftime("%Y-%m-%d %H:%M:%S"),
        "side": "long",
    }
    with pytest.raises(ValueError, match="overlapping"):
        builder.verify_support_alignment(overlap, dates=dates)


def test_build_support_enforces_cutoff_and_emitted_manifest_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future_args = Namespace(
        output=tmp_path / "future.csv.gz",
        manifest_output=tmp_path / "future.json",
        exclude_from="2026-01-01",
    )
    with pytest.raises(ValueError, match="frozen pre-2025 cutoff"):
        builder.build_support(future_args)

    dates = pd.Series(pd.date_range("2023-01-01", "2024-12-31 23:55", freq="5min"))
    market = pd.DataFrame(
        {
            "date": dates,
            "open": np.ones(len(dates)),
            "high": np.ones(len(dates)),
            "low": np.ones(len(dates)),
            "close": np.ones(len(dates)),
        }
    )
    anchor_positions = np.arange(
        143,
        len(market) - builder.HOLD_BARS - 2,
        builder.ANCHOR_STRIDE,
        dtype=np.int64,
    )
    long_active = np.zeros(len(market), dtype=bool)
    for year in (2023, 2024):
        eligible = anchor_positions[dates.iloc[anchor_positions].dt.year.to_numpy() == year]
        long_active[eligible[[200, 1000]]] = True
    short_active = np.zeros(len(market), dtype=bool)
    expected_hash = effective_selection_signal_hash(
        market,
        dates,
        long_active,
        short_active,
        window=builder.SELECTION_WINDOW,
    )
    frozen_rows = [
        {
            "stream_id": "mock_stream",
            "feature_set": "stable8",
            "transform": "mean",
            "member_count": 6,
            "rolling_score_window_anchors": 720,
            "score_quantile": 0.9,
            "side_policy": "long",
            "hold_bars": builder.HOLD_BARS,
            "anchor_stride_bars": builder.ANCHOR_STRIDE,
            "holdout2023": {"return_pct": 999.0},
            "signal_hash": expected_hash,
        }
        for _ in builder.EXPECTED_TOP10_RANKS
    ]
    source_manifest = {
        "later_metrics_included": False,
        "selection_window": list(builder.SELECTION_WINDOW),
        "target": {"name": "tail"},
        "feature_sets": {"stable8": ["mock"]},
        "data_sha256": {"market": "m", "funding": "f", "premium": "p"},
        "top10": frozen_rows,
    }
    prepared = {
        "market": market,
        "dates": dates,
        "positions": anchor_positions,
        "reproduced_feature_sets": {"stable8": ["mock"]},
        "masks": {
            "fit2020_2022": np.zeros(len(anchor_positions), dtype=bool),
            "holdout2023": dates.iloc[anchor_positions].dt.year.to_numpy() == 2023,
            "test2024": dates.iloc[anchor_positions].dt.year.to_numpy() == 2024,
        },
    }
    monkeypatch.setattr(
        builder,
        "validate_frozen_manifest_identities",
        lambda *_: {"groupdro": "g", "ensemble_uncertainty": "e"},
    )
    monkeypatch.setattr(
        builder,
        "load_and_validate_source_manifests",
        lambda *_: (source_manifest, source_manifest),
    )
    monkeypatch.setattr(
        builder,
        "validate_input_hashes",
        lambda *_: {"market": "m", "funding": "f", "premium": "p"},
    )
    monkeypatch.setattr(builder, "prepare_invariant_inputs", lambda *_: prepared)
    monkeypatch.setattr(
        builder,
        "train_transformed_streams",
        lambda *_: ({"mock_stream": np.zeros(len(anchor_positions))}, {"mock": {}}),
    )
    monkeypatch.setattr(
        builder,
        "policy_masks_for_frozen_row",
        lambda *_: (long_active, short_active),
    )
    output = tmp_path / "support.csv.gz"
    manifest_output = tmp_path / "support.json"
    args = Namespace(
        groupdro_manifest=tmp_path / "group.json",
        ensemble_manifest=tmp_path / "ensemble.json",
        input_csv="market.csv.gz",
        funding_csv="funding.csv.gz",
        premium_csv="premium.csv.gz",
        output=output,
        manifest_output=manifest_output,
        exclude_from=builder.SUPPORT_WINDOW[1],
        force=False,
    )
    manifest = builder.build_support(args)
    assert manifest["processed_market"]["last_timestamp"].startswith("2024-12-31")
    assert manifest["processed_pre2025_market_frame_sha256"] == builder.dataframe_sha256(
        market
    )
    assert not nested_metric_keys(manifest)
    assert all(
        row["verified_2023_signal_hash"] == row["emitted_2023_signal_hash"]
        for row in manifest["top10"]
    )
    assert all("holdout2023" not in row for row in manifest["top10"])
    assert output.exists()
    assert json.loads(manifest_output.read_text())["support_csv_sha256"] == builder.sha256_file(
        output
    )


def nested_metric_keys(value: object) -> set[str]:
    return builder.nested_keys(value).intersection(builder.METRIC_KEYS)
