from __future__ import annotations

# Pandas test fixtures intentionally exercise broad scalar/index unions.
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# pyright: reportCallIssue=false, reportOperatorIssue=false

import gzip
import hashlib
import math
from typing import Any

import pandas as pd
import pytest

from training import evaluate_blockspace_load_settlement_relay_support as evaluate


def _packet_frame(rows: int = 123) -> pd.DataFrame:
    start = 9_000
    available = pd.date_range("2020-01-01T00:00:00Z", periods=rows, freq="12h")
    fee_pressure = [float(index) for index in range(rows)]
    fee_pressure[-1] += 2.0
    endpoint_density = [-float(index) for index in range(rows)]
    endpoint_density[-1] -= 2.0
    return pd.DataFrame(
        {
            "packet_id": range(start, start + rows),
            "packet_start_height": [72 * value for value in range(start, start + rows)],
            "packet_end_height": [
                72 * value + 71 for value in range(start, start + rows)
            ],
            "confirmation_end_height": [
                72 * value + 77 for value in range(start, start + rows)
            ],
            "packet_valid": True,
            "fee_pressure": fee_pressure,
            "endpoint_density": endpoint_density,
            "source_available_at_utc": available,
            "entry_time_utc": available.ceil("5min") + pd.Timedelta(minutes=5),
            "exit_time_utc": (
                available.ceil("5min")
                + pd.Timedelta(minutes=5)
                + pd.Timedelta(hours=24)
            ),
        }
    )


def _allowed_source_frame(rows: int = 150, start_height: int = 7_200) -> pd.DataFrame:
    ids = [f"{index + 1:064x}" for index in range(rows)]
    previous = ["0" * 64, *ids[:-1]]
    return pd.DataFrame(
        {
            "height": range(start_height, start_height + rows),
            "id": ids,
            "previousblockhash": previous,
            "timestamp": [1_600_000_000 + 600 * index for index in range(rows)],
            "weight": 1_000,
            "total_fees": 100,
            "total_inputs": 5,
            "total_outputs": 6,
        },
        columns=evaluate.BLSR_SOURCE_COLUMNS,
    )


def _feature_frame(specs: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    base = pd.Timestamp("2021-02-01T00:00:00Z")
    for offset, spec in enumerate(specs):
        packet_id = 10_000 + offset
        records.append(
            {
                "packet_id": packet_id,
                "packet_start_height": 72 * packet_id,
                "packet_end_height": 72 * packet_id + 71,
                "confirmation_end_height": 72 * packet_id + 77,
                "source_available_at_utc": base + pd.Timedelta(hours=12 * offset),
                "feature_valid": True,
                "rank_ready": True,
                "fee_change": spec.get("fee", 0.0),
                "endpoint_change": spec.get("endpoint", 0.0),
                "fee_magnitude_rank": spec.get(
                    "fee_rank", 0.9 if spec.get("fee", 0.0) else 0.0
                ),
                "endpoint_magnitude_rank": spec.get(
                    "endpoint_rank", 0.9 if spec.get("endpoint", 0.0) else 0.0
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=evaluate.FEATURE_COLUMNS)


def _raw_candidate(
    *, entry: str, onset: int, confirmation: int, side: int = 1
) -> dict[str, Any]:
    timestamp = pd.Timestamp(entry)
    return {
        "clock": "candidate",
        "onset_packet_id": onset,
        "confirmation_packet_id": confirmation,
        "decision_time_utc": timestamp - pd.Timedelta(minutes=5),
        "entry_time_utc": timestamp,
        "exit_time_utc": timestamp + pd.Timedelta(hours=24),
        "side": side,
    }


def _clock(records: list[dict[str, Any]], *, name: str = "primary") -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        entry = pd.Timestamp(record["entry"])
        out.append(
            {
                "policy_id": evaluate.POLICY_ID,
                "clock": name,
                "window": "train" if entry < evaluate.TRAIN_END else "selection",
                "onset_packet_id": index,
                "confirmation_packet_id": index + 1,
                "decision_time_utc": entry - pd.Timedelta(minutes=5),
                "entry_time_utc": entry,
                "exit_time_utc": entry + pd.Timedelta(hours=24),
                "side": int(record.get("side", 1)),
            }
        )
    return pd.DataFrame.from_records(out, columns=evaluate.CLOCK_COLUMNS)


def test_build_features_uses_one_packet_change_and_strict_prior_midrank() -> None:
    features, audit = evaluate.build_features(_packet_frame())

    assert math.isnan(features.loc[0, "fee_change"])
    assert features.loc[1, "fee_change"] == 1.0
    assert features.loc[1, "endpoint_change"] == -1.0
    assert not bool(features.loc[120, "rank_ready"])
    assert bool(features.loc[121, "rank_ready"])
    assert features.loc[121, "fee_magnitude_rank"] == 0.5
    assert features.loc[122, "fee_change"] == 3.0
    assert features.loc[122, "fee_magnitude_rank"] == 1.0
    assert features.loc[122, "endpoint_magnitude_rank"] == 1.0
    assert audit["base_valid_feature_rows"] == 122
    assert audit["rank_ready_rows"] == 2


def test_build_features_rejects_availability_tie_instead_of_sorting() -> None:
    packets = _packet_frame()
    packets.loc[10, "source_available_at_utc"] = packets.loc[
        9, "source_available_at_utc"
    ]
    with pytest.raises(RuntimeError, match="not strictly increasing"):
        evaluate.build_features(packets)


def test_source_loader_materializes_only_allowed_signal_columns(tmp_path: Any) -> None:
    allowed = _allowed_source_frame(rows=2)
    full = allowed.copy()
    full.insert(4, "mediantime", "DO_NOT_PARSE")
    full.insert(5, "tx_count", "DO_NOT_PARSE")
    full.insert(6, "size", "DO_NOT_PARSE")
    full["utxo_set_change"] = "DO_NOT_PARSE"
    compressed = gzip.compress(
        full.to_csv(index=False, lineterminator="\n").encode("utf-8"), mtime=0
    )
    source = tmp_path / "source.csv.gz"
    source.write_bytes(compressed)
    registration = {
        "source_manifest": {
            "source_output": {
                "path": str(source),
                "sha256": hashlib.sha256(compressed).hexdigest(),
                "bytes": len(compressed),
            }
        }
    }

    loaded = evaluate.load_source_frame(registration)
    assert tuple(loaded.columns) == evaluate.BLSR_SOURCE_COLUMNS
    assert not set(evaluate.FORBIDDEN_SOURCE_VALUE_COLUMNS).intersection(loaded.columns)


def test_packet_builder_uses_six_successors_and_frozen_availability_lag() -> None:
    source = _allowed_source_frame()
    packets, audit = evaluate.build_packets(source)

    assert len(packets) == 2
    assert packets.loc[0, "packet_start_height"] == 7_200
    assert packets.loc[0, "packet_end_height"] == 7_271
    assert packets.loc[0, "confirmation_end_height"] == 7_277
    expected_available = pd.Timestamp(
        int(source.loc[77, "timestamp"]) + 172_800, unit="s", tz="UTC"
    )
    assert packets.loc[0, "source_available_at_utc"] == expected_available
    assert packets.loc[0, "entry_time_utc"] == (
        expected_available.ceil("5min") + pd.Timedelta(minutes=5)
    )
    assert audit["all_confirmation_blocks_contained"] is True


def test_relay_uses_first_response_ignores_active_shocks_and_never_restarts_row() -> (
    None
):
    features = _feature_frame(
        [
            {"fee": 1.0},
            {"fee": -2.0},
            {"endpoint": 1.0},
            {"fee": -1.0},
            {"endpoint": 1.0, "fee": 2.0},
            {"fee": 1.0},
            {},
            {},
            {},
            {"fee": -1.0},
        ]
    )
    candidates, audit = evaluate.relay_candidates(
        features,
        clock="primary",
        onset_family="fee",
        response_family="endpoint",
        confirm_same_sign=True,
    )

    assert candidates["side"].tolist() == [1]
    assert candidates["onset_packet_id"].tolist() == [10_000]
    assert candidates["confirmation_packet_id"].tolist() == [10_002]
    assert audit == {
        "onsets": 4,
        "same_sign_resolutions": 1,
        "opposite_sign_resolutions": 1,
        "emitted_candidates": 1,
        "expired": 1,
        "open_at_source_end": 1,
        "active_fee_or_endpoint_shocks_ignored": 2,
    }

    opposite, opposite_audit = evaluate.relay_candidates(
        features,
        clock="opposite_response_relay",
        onset_family="fee",
        response_family="endpoint",
        confirm_same_sign=False,
    )
    assert opposite["side"].tolist() == [-1]
    assert opposite["confirmation_packet_id"].tolist() == [10_004]
    assert opposite_audit["emitted_candidates"] == 1


def test_stale_response_applies_prior_packet_state_at_later_availability() -> None:
    features = _feature_frame(
        [
            {"fee": 1.0, "endpoint": 1.0},
            {"endpoint": 1.0},
            {},
            {},
        ]
    )
    stale, audit = evaluate.relay_candidates(
        features,
        clock="one_packet_stale_response",
        onset_family="fee",
        response_family="endpoint",
        confirm_same_sign=True,
        stale_response_packets=1,
    )
    assert audit["emitted_candidates"] == 1
    assert stale.loc[0, "confirmation_packet_id"] == 10_002
    assert (
        stale.loc[0, "decision_time_utc"] == features.loc[2, "source_available_at_utc"]
    )


def test_scheduler_is_global_stable_and_allows_equal_exit_boundary() -> None:
    candidates = pd.DataFrame.from_records(
        [
            _raw_candidate(entry="2021-01-02T00:05:00Z", onset=2, confirmation=3),
            _raw_candidate(entry="2021-01-02T00:05:00Z", onset=1, confirmation=2),
            _raw_candidate(entry="2021-01-03T00:05:00Z", onset=4, confirmation=5),
            _raw_candidate(entry="2022-12-31T12:00:00Z", onset=6, confirmation=7),
        ],
        columns=[
            column
            for column in evaluate.CLOCK_COLUMNS
            if column not in ("policy_id", "window")
        ],
    )
    clock, drops = evaluate.schedule_candidates(candidates, clock="primary")

    assert clock["onset_packet_id"].tolist() == [1, 4]
    assert clock.loc[1, "entry_time_utc"] == clock.loc[0, "exit_time_utc"]
    assert drops == {"split_containment": 1, "global_overlap": 1}


def test_controls_are_independent_and_exact_primary_controls_reuse_clock() -> None:
    features = _feature_frame(
        [
            {"fee": 1.0},
            {"endpoint": 1.0},
            {},
            {"fee": -1.0},
            {"endpoint": -1.0},
            {},
        ]
    )
    clocks, _ = evaluate.build_clocks(features)
    primary = clocks.loc[clocks["clock"].eq("primary")]
    direction = clocks.loc[clocks["clock"].eq("direction_flip")]
    random = clocks.loc[clocks["clock"].eq("deterministic_random_side")]

    assert primary["side"].tolist() == [1, -1]
    assert direction["entry_time_utc"].tolist() == primary["entry_time_utc"].tolist()
    assert direction["side"].tolist() == [-1, 1]
    assert random["entry_time_utc"].tolist() == primary["entry_time_utc"].tolist()
    assert clocks.loc[clocks["clock"].eq("fee_only")].shape[0] == 2
    assert clocks.loc[clocks["clock"].eq("endpoint_only")].shape[0] == 2
    assert evaluate.control_structure_summary(clocks)["passed"] is True


def test_support_summary_passes_only_complete_balanced_calendar() -> None:
    records: list[dict[str, Any]] = []
    side = 1
    for year in (2021, 2022, 2023):
        for month in range(1, 13):
            for day in (1, 8, 15, 22):
                records.append(
                    {
                        "entry": pd.Timestamp(
                            year=year, month=month, day=day, tz="UTC"
                        ).isoformat(),
                        "side": side,
                    }
                )
                side *= -1
    primary = _clock(records)
    packet_audit = {
        "complete_packets": 2_959,
        "first_complete_packet_start_height": 610_704,
        "last_complete_packet_end_height": 823_751,
        "all_complete_packets_have_72_blocks": True,
        "complete_packet_ids_consecutive": True,
        "all_confirmation_blocks_contained": True,
    }
    feature_audit = {"availability_strictly_increasing": True}

    summary = evaluate.support_summary(primary, packet_audit, feature_audit)
    assert summary["passed"] is True
    assert summary["counts"]["train"] == 96
    assert summary["counts"]["selection"] == 48

    one_sided = primary.copy()
    one_sided.loc[one_sided["window"].eq("selection"), "side"] = 1
    failed = evaluate.support_summary(one_sided, packet_audit, feature_audit)
    assert failed["passed"] is False
    assert failed["checks"]["selection_each_side_minimum"] is False


def test_exact_and_one_to_one_tolerant_overlap_do_not_double_match() -> None:
    candidate = pd.Series(
        pd.to_datetime(["2021-01-01T00:00:00Z", "2021-01-01T12:00:00Z"], utc=True)
    )
    one_reference = pd.Series(pd.to_datetime(["2021-01-01T06:00:00Z"], utc=True))
    tolerant = evaluate.one_to_one_tolerant_overlap(candidate, one_reference)
    assert tolerant["one_to_one_matches"] == 1
    assert tolerant["candidate_one_to_one_within_six_hours_fraction"] == 0.5

    exact = evaluate.exact_entry_jaccard(
        candidate,
        pd.Series(
            pd.to_datetime(["2021-01-01T00:00:00Z", "2021-01-02T00:00:00Z"], utc=True)
        ),
    )
    assert exact["intersection_entries"] == 1
    assert exact["exact_entry_timestamp_jaccard"] == pytest.approx(1 / 3)


def test_signed_exposure_correlation_is_full_grid_and_rejects_overlap() -> None:
    candidate = _clock(
        [
            {"entry": "2021-01-01T00:00:00Z", "side": 1},
            {"entry": "2021-01-03T00:00:00Z", "side": -1},
        ]
    )
    comparator = candidate.rename(
        columns={"entry_time_utc": "entry_time", "exit_time_utc": "exit_time"}
    )[["entry_time", "exit_time", "side"]]
    result = evaluate.signed_occupied_exposure_correlation(candidate, comparator)
    assert result["defined"] is True
    assert result["absolute_signed_occupied_exposure_pearson"] == pytest.approx(1.0)

    overlapping = pd.concat([comparator, comparator.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="intervals overlap"):
        evaluate.signed_occupied_exposure_correlation(candidate, overlapping)


def test_novelty_timestamp_only_omits_only_exposure_check() -> None:
    primary = _clock(
        [
            {"entry": "2021-01-01T00:00:00Z", "side": 1},
            {"entry": "2021-03-01T00:00:00Z", "side": -1},
            {"entry": "2021-05-01T00:00:00Z", "side": 1},
        ]
    )
    comparator = pd.DataFrame(
        {
            "comparator": "timestamp-only",
            "capability": "timestamp_only",
            "entry_time": pd.to_datetime(
                ["2021-01-20T00:00:00Z", "2021-03-20T00:00:00Z"], utc=True
            ),
            "exit_time": [math.nan, math.nan],
            "side": [math.nan, math.nan],
            "source_clock": ["a", "b"],
        }
    )
    result = evaluate.novelty_summary(primary, comparator)
    summary = result["comparators"]["timestamp-only"]
    assert result["passed"] is True
    assert summary["signed_occupied_exposure"] is None
    assert set(summary["checks"]) == {
        "exact_entry_timestamp_jaccard",
        "candidate_one_to_one_within_six_hours_fraction",
    }

    leaking = comparator.copy()
    leaking.loc[0, "side"] = 1
    with pytest.raises(RuntimeError, match="leaks direction"):
        evaluate.novelty_summary(primary, leaking)


def test_novelty_exposure_keeps_pre_start_interval_crossing_grid_boundary() -> None:
    primary = _clock(
        [
            {"entry": "2021-01-01T00:00:00Z", "side": 1},
            {"entry": "2021-09-01T00:00:00Z", "side": -1},
        ]
    )
    comparator = pd.DataFrame(
        {
            "comparator": ["directional", "directional"],
            "capability": ["directional_interval", "directional_interval"],
            "entry_time": pd.to_datetime(
                ["2020-12-31T12:00:00Z", "2021-08-01T00:00:00Z"], utc=True
            ),
            "exit_time": pd.to_datetime(
                ["2021-01-01T12:00:00Z", "2021-08-02T00:00:00Z"], utc=True
            ),
            "side": [1, -1],
            "source_clock": ["before-boundary", "inside-grid"],
        },
        columns=evaluate.COMPARATOR_COLUMNS,
    )

    result = evaluate.novelty_summary(primary, comparator)
    summary = result["comparators"]["directional"]
    assert summary["rows"] == 1
    assert summary["interval_overlap_rows"] == 2
    assert summary["signed_occupied_exposure"]["comparator_nonflat_rows"] == 432


def test_exact_comparator_identity_and_capability_set_is_fail_closed() -> None:
    rows: list[dict[str, Any]] = []
    base = pd.Timestamp("2021-01-01T00:00:00Z")
    for index, (name, capability) in enumerate(
        evaluate.EXPECTED_COMPARATOR_CAPABILITIES.items()
    ):
        entry = base + pd.Timedelta(days=2 * index)
        directional = capability == "directional_interval"
        rows.append(
            {
                "comparator": name,
                "capability": capability,
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=24)
                if directional
                else math.nan,
                "side": 1 if directional else math.nan,
                "source_clock": f"{name}:synthetic",
            }
        )
    complete = pd.DataFrame.from_records(rows, columns=evaluate.COMPARATOR_COLUMNS)
    evaluate._validate_comparator_identities(complete)

    missing = complete.iloc[:-1].copy()
    with pytest.raises(RuntimeError, match="exact comparator identity"):
        evaluate._validate_comparator_identities(missing)

    replacement = complete.copy()
    replacement.loc[0, "comparator"] = "replacement-with-same-count"
    with pytest.raises(RuntimeError, match="exact comparator identity"):
        evaluate._validate_comparator_identities(replacement)

    mixed = pd.concat(
        [
            complete,
            complete.iloc[[0]].assign(capability="timestamp_only"),
        ],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="capability drift"):
        evaluate._validate_comparator_identities(mixed)


def test_novelty_rejects_comparator_without_in_window_entry() -> None:
    primary = _clock([{"entry": "2021-01-01T00:00:00Z", "side": 1}])
    outside = pd.DataFrame(
        {
            "comparator": ["outside"],
            "capability": ["timestamp_only"],
            "entry_time": pd.to_datetime(["2020-01-01T00:00:00Z"], utc=True),
            "exit_time": [math.nan],
            "side": [math.nan],
            "source_clock": ["outside"],
        },
        columns=evaluate.COMPARATOR_COLUMNS,
    )
    with pytest.raises(RuntimeError, match="no evaluation entries"):
        evaluate.novelty_summary(primary, outside)


def test_evaluation_skips_comparators_after_support_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = pd.DataFrame({"height": [1, 2]})
    features = _feature_frame([{}, {}])
    empty_clocks = pd.DataFrame(columns=evaluate.CLOCK_COLUMNS)
    registration = {
        "source_manifest": {
            "source_output": {"path": "source.csv.gz", "sha256": "a" * 64}
        }
    }
    monkeypatch.setattr(
        evaluate,
        "_exact_source_audit",
        lambda _source: {"passed": True, "checks": {}},
    )
    monkeypatch.setattr(
        evaluate,
        "build_packets",
        lambda _source: (pd.DataFrame(), {"complete_packets": 0}),
    )
    monkeypatch.setattr(
        evaluate,
        "build_features",
        lambda _packets: (
            features,
            {"availability_strictly_increasing": True},
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "build_clocks",
        lambda _features: (empty_clocks, {"primary_candidates": 0}),
    )
    monkeypatch.setattr(
        evaluate,
        "load_comparators",
        lambda _packets: pytest.fail("support failure must not open comparators"),
    )

    report = evaluate.evaluate_source_only(source, registration)
    assert report["verdict"]["status"] == "REJECT"
    assert report["novelty"]["evaluated"] is False
    assert report["outcome_boundary"]["comparator_event_rows_read"] == 0
    assert report["outcome_boundary"]["btc_market_rows_loaded"] == 0
    assert report["event_rows_published"] == 0
    assert report["feature_values_published"] == 0


def test_control_names_and_support_limits_match_frozen_preregistration() -> None:
    policy = evaluate.prereg.policy()
    assert policy["controls"] == evaluate.prereg.CONTROL_DEFINITIONS
    assert tuple(evaluate.prereg.CONTROL_DEFINITIONS) == evaluate.CONTROL_NAMES
    gates = policy["support_gates"]
    expected_support = {
        "train_total_minimum": gates["train_total_minimum"],
        "train_each_year_minimum": gates["train_each_year_minimum"],
        "train_each_half_year_minimum": gates["train_each_half_year_minimum"],
        "train_each_side_minimum": gates["train_long_minimum"],
        "train_each_side_each_year_minimum": gates["train_each_side_each_year_minimum"],
        "train_maximum_month_share": gates["train_maximum_month_share"],
        "train_maximum_weekday_share": gates["train_maximum_weekday_share"],
        "selection_total_minimum": gates["selection_total_minimum"],
        "selection_each_half_minimum": gates["selection_each_half_minimum"],
        "selection_each_quarter_minimum": gates["selection_each_quarter_minimum"],
        "selection_each_side_minimum": gates["selection_long_minimum"],
        "selection_each_side_each_half_minimum": gates[
            "selection_each_side_each_half_minimum"
        ],
        "selection_maximum_month_share": gates["selection_maximum_month_share"],
        "selection_maximum_weekday_share": gates["selection_maximum_weekday_share"],
    }
    assert gates["train_long_minimum"] == gates["train_short_minimum"]
    assert gates["selection_long_minimum"] == gates["selection_short_minimum"]
    assert evaluate.SUPPORT_LIMITS == expected_support

    novelty = policy["novelty_gates"]
    assert evaluate.NOVELTY_LIMITS == {
        "exact_entry_timestamp_jaccard_maximum": novelty[
            "exact_entry_timestamp_jaccard_maximum"
        ],
        "candidate_one_to_one_within_six_hours_fraction_maximum": novelty[
            "candidate_one_to_one_within_six_hours_fraction_maximum"
        ],
        "signed_occupied_exposure_absolute_pearson_maximum": novelty[
            "signed_occupied_exposure_absolute_pearson_maximum"
        ],
    }
    assert novelty["timestamp_only_rule"] == "omit only signed exposure correlation"
    assert set(evaluate.EXPECTED_COMPARATOR_CAPABILITIES) >= {
        "FETD-288",
        "BATE-288",
        "UFCP-1",
        "WCTR-288",
    }
    assert policy["normalization"]["parameter_grid"] == []
