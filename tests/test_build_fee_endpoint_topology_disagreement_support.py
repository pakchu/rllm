from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from training import build_fee_endpoint_topology_disagreement_support as support


def _source_frame(
    *,
    complete_packets: int = 6,
    first_complete_packet_id: int = 10,
    start: str = "2020-01-01T00:00:00Z",
) -> pd.DataFrame:
    first_height = first_complete_packet_id * 72 - 1
    last_complete_end = (
        (first_complete_packet_id + complete_packets - 1) * 72 + 71
    )
    last_height = last_complete_end + 6
    base_timestamp = int(pd.Timestamp(start).timestamp())
    rows: list[dict[str, Any]] = []
    for offset, height in enumerate(range(first_height, last_height + 1)):
        packet_id = height // 72
        wave = packet_id - first_complete_packet_id
        size = 900_000 + (height % 17) * 100
        weight = 2_000_000 + (height % 19) * 100
        total_inputs = 2_000 + 40 * ((wave % 5) + 1)
        total_outputs = 2_100 + 35 * (((wave + 2) % 5) + 1)
        total_fees = 100_000 + 5_000 * ((wave % 7) + 1)
        rows.append(
            {
                "height": height,
                "id": f"{height:064x}",
                "previousblockhash": f"{height - 1:064x}",
                "timestamp": base_timestamp + offset * 600,
                "mediantime": base_timestamp + offset * 600 - 600,
                "tx_count": 2_500,
                "size": size,
                "weight": weight,
                "total_fees": total_fees,
                "total_inputs": total_inputs,
                "total_outputs": total_outputs,
                "utxo_set_change": total_outputs - total_inputs,
            }
        )
    return pd.DataFrame.from_records(rows, columns=support.SOURCE_COLUMNS)


def _packet_frame(rows: int = 12, *, start_id: int = 100) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    start_time = pd.Timestamp("2020-01-01T00:00:00Z")
    for index in range(rows):
        packet_id = start_id + index
        available = start_time + pd.Timedelta(hours=12 * index)
        fee = 0.1 * math.sin(index * 0.9) + index * 0.001
        endpoint = 0.1 * math.cos(index * 0.7) - index * 0.001
        records.append(
            {
                "packet_id": packet_id,
                "packet_start_height": packet_id * 72,
                "packet_end_height": packet_id * 72 + 71,
                "confirmation_end_height": packet_id * 72 + 77,
                "packet_valid": True,
                "fee_pressure": fee,
                "endpoint_density": endpoint,
                "source_available_at_utc": available,
                "entry_time_utc": available + pd.Timedelta(minutes=5),
                "exit_time_utc": available + pd.Timedelta(hours=24, minutes=5),
            }
        )
    return pd.DataFrame.from_records(records)


def _feature_row(
    entry: str,
    *,
    packet_id: int,
    primary: bool = True,
    side: int = 1,
) -> dict[str, Any]:
    entry_time = pd.Timestamp(entry)
    magnitude = 0.02 + packet_id * 1e-7
    if primary:
        endpoint = magnitude if side > 0 else -magnitude
        fee = -endpoint
    else:
        endpoint = magnitude if side > 0 else -magnitude
        fee = endpoint
    return {
        "packet_id": packet_id,
        "packet_start_height": packet_id * 72,
        "packet_end_height": packet_id * 72 + 71,
        "confirmation_end_height": packet_id * 72 + 77,
        "source_available_at_utc": entry_time - pd.Timedelta(minutes=5),
        "entry_time_utc": entry_time,
        "exit_time_utc": entry_time + pd.Timedelta(hours=24),
        "packet_valid": True,
        "feature_valid": True,
        "rank_ready": True,
        "fee_pressure": -2.0,
        "endpoint_density": -6.0,
        "fee_transport": fee,
        "endpoint_transport": endpoint,
        "strain_magnitude": abs(fee * endpoint),
        "strain_rank": 0.80,
        "fee_magnitude_rank": 0.80,
        "endpoint_magnitude_rank": 0.80,
    }


def _cfg(tmp_path: Path, **changes: Any) -> support.Config:
    root = tmp_path / "artifacts"
    cfg = support.Config(
        preregistration=str(support.DEFAULT_PREREGISTRATION),
        output=str(root / "support.json"),
        artifact_root=str(root),
    )
    return replace(cfg, **changes)


def _passing_packet_audit() -> dict[str, Any]:
    return {
        "complete_packets": 2_959,
        "first_complete_packet_start_height": 610_704,
        "last_complete_packet_end_height": 823_751,
        "all_complete_packets_have_72_blocks": True,
        "complete_packet_ids_consecutive": True,
        "all_confirmation_blocks_contained": True,
    }


def _passing_clock() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    packet_id = 20_000
    for year in (2021, 2022, 2023):
        for month in range(1, 13):
            for day in (2, 8, 15, 22):
                entry = pd.Timestamp(
                    year=year, month=month, day=day, tz="UTC"
                ) + pd.Timedelta(minutes=5)
                rows.append(
                    {
                        "window": "selection" if year == 2023 else "train",
                        "entry_time_utc": entry,
                        "exit_time_utc": entry + pd.Timedelta(hours=24),
                        "side": 1 if packet_id % 2 == 0 else -1,
                    }
                )
                packet_id += 1
    return pd.DataFrame.from_records(rows)


def test_source_validation_and_absolute_packet_alignment() -> None:
    source = _source_frame(complete_packets=4)
    validated = support.validate_source_frame(source)
    packets, audit = support.build_packets(validated)
    assert packets["packet_id"].tolist() == [10, 11, 12, 13]
    assert packets["packet_start_height"].tolist() == [720, 792, 864, 936]
    assert packets["packet_end_height"].tolist() == [791, 863, 935, 1007]
    assert packets["confirmation_end_height"].tolist()[-1] == 1013
    assert audit["partial_edge_packet_ids"] == [9, 14]
    assert audit["all_complete_packets_have_72_blocks"] is True
    assert audit["complete_packet_ids_consecutive"] is True


def test_packet_availability_uses_packet_through_h_plus_6_max_and_latency() -> None:
    source = _source_frame(complete_packets=2)
    target_end = 10 * 72 + 71
    successor_height = target_end + 6
    aligned_max = int(pd.Timestamp("2020-02-01T00:00:00Z").timestamp())
    source.loc[source["height"] == successor_height, "timestamp"] = aligned_max
    source.loc[source["height"] == successor_height, "mediantime"] = aligned_max - 1
    packets, _ = support.build_packets(source)
    first = packets.iloc[0]
    expected_available = pd.Timestamp("2020-02-03T00:00:00Z")
    assert first["source_available_at_utc"] == expected_available
    assert first["entry_time_utc"] == expected_available + pd.Timedelta(minutes=5)
    assert first["exit_time_utc"] == expected_available + pd.Timedelta(
        hours=24, minutes=5
    )


def test_source_contract_rejects_chain_schema_numeric_and_identity_drift() -> None:
    source = _source_frame(complete_packets=2)
    assert len(support.validate_source_frame(source)) == len(source)

    with pytest.raises(RuntimeError, match="schema drift"):
        support.validate_source_frame(source.assign(close=1.0))

    gap = source.drop(index=10).reset_index(drop=True)
    with pytest.raises(RuntimeError, match="contiguous heights"):
        support.validate_source_frame(gap)

    broken = source.copy()
    broken.loc[10, "previousblockhash"] = "f" * 64
    with pytest.raises(RuntimeError, match="hash-chain"):
        support.validate_source_frame(broken)

    fractional = source.copy()
    fractional["weight"] = fractional["weight"].astype(float)
    fractional.loc[10, "weight"] += 0.5
    with pytest.raises(RuntimeError, match="exact integers"):
        support.validate_source_frame(fractional)

    identity = source.copy()
    identity.loc[10, "utxo_set_change"] += 1
    with pytest.raises(RuntimeError, match="UTXO identity"):
        support.validate_source_frame(identity)

    bip141 = source.copy()
    bip141.loc[10, "weight"] = bip141.loc[10, "size"] - 1
    with pytest.raises(RuntimeError, match="BIP141"):
        support.validate_source_frame(bip141)


def test_strict_prior_midrank_and_causal_availability_filter() -> None:
    assert support.strict_prior_midrank(2.0, [1.0, 2.0, 2.0, 3.0]) == 0.5
    with pytest.raises(ValueError, match="requires prior"):
        support.strict_prior_midrank(1.0, [])

    packets = _packet_frame(8)
    features = support.build_features(packets, lookback=3, minimum_prior=2)
    row = features.iloc[4]
    expected_fee = packets.iloc[4]["fee_pressure"] - packets.iloc[2]["fee_pressure"]
    expected_endpoint = (
        packets.iloc[4]["endpoint_density"]
        - packets.iloc[2]["endpoint_density"]
    )
    assert row["fee_transport"] == pytest.approx(expected_fee, abs=1e-15)
    assert row["endpoint_transport"] == pytest.approx(
        expected_endpoint, abs=1e-15
    )
    assert bool(row["rank_ready"]) is True

    simultaneous = packets.copy()
    simultaneous.loc[2:4, "source_available_at_utc"] = packets.loc[
        4, "source_available_at_utc"
    ]
    causal = support.build_features(simultaneous, lookback=3, minimum_prior=2)
    assert bool(causal.iloc[4]["rank_ready"]) is False

    unavailable = packets.copy()
    unavailable.loc[3, "source_available_at_utc"] = (
        unavailable.loc[4, "source_available_at_utc"] + pd.Timedelta(hours=1)
    )
    with pytest.raises(RuntimeError, match="ingredient is unavailable"):
        support.build_features(unavailable, lookback=3, minimum_prior=2)


def test_candidate_thresholds_zero_and_side_mapping() -> None:
    long_row = SimpleNamespace(
        **_feature_row("2021-02-01T00:05:00Z", packet_id=1, side=1)
    )
    short_row = SimpleNamespace(
        **_feature_row("2021-02-02T00:05:00Z", packet_id=2, side=-1)
    )
    assert support._candidate_side(long_row, "primary") == 1
    assert support._candidate_side(short_row, "primary") == -1

    boundary = vars(long_row).copy()
    boundary["strain_rank"] = 0.75
    assert support._candidate_side(SimpleNamespace(**boundary), "primary") == 1
    boundary["strain_rank"] = np.nextafter(0.75, 0.0)
    assert support._candidate_side(SimpleNamespace(**boundary), "primary") == 0
    boundary["fee_transport"] = -0.0
    boundary["strain_rank"] = 1.0
    assert support._candidate_side(SimpleNamespace(**boundary), "primary") == 0

    same = SimpleNamespace(
        **_feature_row(
            "2021-02-03T00:05:00Z", packet_id=3, primary=False, side=1
        )
    )
    assert support._candidate_side(same, "same_direction") == 1
    assert support._candidate_side(same, "fee_only") == -1
    assert support._candidate_side(same, "endpoint_only") == 1
    with pytest.raises(ValueError, match="unknown FETD"):
        support._candidate_side(same, "unknown")


def test_clock_nonoverlap_tie_break_and_split_containment() -> None:
    rows = [
        _feature_row("2021-02-01T00:05:00Z", packet_id=101, side=1),
        _feature_row("2021-02-01T00:05:00Z", packet_id=100, side=-1),
        _feature_row("2021-02-01T12:05:00Z", packet_id=102, side=1),
        _feature_row("2021-02-02T00:05:00Z", packet_id=103, side=-1),
        _feature_row("2022-12-31T12:05:00Z", packet_id=104, side=1),
    ]
    clock = support.build_clock(pd.DataFrame(rows), mode="primary", clock="primary")
    assert clock["packet_id"].tolist() == [100, 103]
    assert clock["entry_time_utc"].tolist() == [
        pd.Timestamp("2021-02-01T00:05:00Z"),
        pd.Timestamp("2021-02-02T00:05:00Z"),
    ]
    assert clock["window"].eq("train").all()


def test_controls_are_deterministic_independent_and_report_delayed_drops() -> None:
    rows: list[dict[str, Any]] = []
    start = pd.Timestamp("2021-02-01T00:05:00Z")
    for index in range(180):
        rows.append(
            _feature_row(
                str(start + pd.Timedelta(hours=12 * index)),
                packet_id=1_000 + index,
                primary=index % 3 != 0,
                side=1 if index % 4 < 2 else -1,
            )
        )
    features = pd.DataFrame(rows)
    primary = support.build_clock(features, mode="primary", clock="primary")
    first, first_drops = support.build_control_clocks(features, primary)
    second, second_drops = support.build_control_clocks(features, primary)
    pd.testing.assert_frame_equal(first, second)
    assert first_drops == second_drops == {"train": 0, "selection": 0}
    assert set(first["clock"].unique()) == set(support.CONTROL_NAMES)

    flipped = first[first["clock"] == "direction_flip"].reset_index(drop=True)
    assert flipped["entry_time_utc"].tolist() == primary["entry_time_utc"].tolist()
    assert flipped["side"].tolist() == (-primary["side"]).tolist()
    assert first[first["clock"] == "constant_long_same_clock"]["side"].eq(1).all()
    assert first[first["clock"] == "constant_short_same_clock"]["side"].eq(-1).all()

    stale = first[first["clock"] == "stale_14_packets"]
    assert not stale.empty
    stale_row = stale.iloc[0]
    current_index = int(
        features.index[features["packet_id"] == stale_row["packet_id"]][0]
    )
    lagged = features.iloc[current_index - 14]
    for column in support.FEATURE_COLUMNS:
        assert stale_row[column] == pytest.approx(lagged[column], abs=1e-15)

    boundary = primary.iloc[[0]].copy()
    boundary["entry_time_utc"] = pd.Timestamp("2022-12-31T00:00:00Z")
    boundary["exit_time_utc"] = pd.Timestamp("2023-01-01T00:00:00Z")
    boundary["window"] = "train"
    delayed, drops = support._delayed_clock(boundary)
    assert delayed.empty
    assert drops == {"train": 1, "selection": 0}


def test_random_clock_preserves_month_side_counts_and_is_shuffle_stable() -> None:
    rows = [
        _feature_row(
            str(pd.Timestamp("2021-03-01T00:05:00Z") + pd.Timedelta(hours=12 * i)),
            packet_id=3_000 + i,
            primary=i % 4 == 0,
            side=1 if i % 8 < 4 else -1,
        )
        for i in range(120)
    ]
    features = pd.DataFrame(rows)
    primary = support.build_clock(features, mode="primary", clock="primary")
    first = support._random_clock(features, primary)
    shuffled = support._random_clock(
        features.sample(frac=1.0, random_state=7).reset_index(drop=True), primary
    )
    pd.testing.assert_frame_equal(first, shuffled)
    first_month = first.assign(
        month=first["entry_time_utc"].dt.strftime("%Y-%m")
    )
    primary_month = primary.assign(
        month=primary["entry_time_utc"].dt.strftime("%Y-%m")
    )
    for key, target in primary_month.groupby(["window", "month"]):
        sample = first_month[
            (first_month["window"] == key[0]) & (first_month["month"] == key[1])
        ]
        candidates = features[
            features["entry_time_utc"].dt.strftime("%Y-%m").eq(key[1])
        ].copy()
        candidates["key"] = [
            support._random_key(key[0], key[1], entry)
            for entry in candidates["entry_time_utc"]
        ]
        expected_ids = set(
            candidates.sort_values(
                ["key", "entry_time_utc", "packet_id"], kind="stable"
            )
            .head(len(target))["packet_id"]
            .tolist()
        )
        assert len(sample) == len(target)
        assert set(sample["packet_id"]) == expected_ids
        assert sample["side"].value_counts().to_dict() == (
            target["side"].value_counts().to_dict()
        )


def test_support_gate_exact_contract_and_no_side_waiver() -> None:
    clock = _passing_clock()
    summary = support.support_gate_summary(
        clock,
        _passing_packet_audit(),
        delayed_dropped={"train": 0, "selection": 0},
    )
    assert summary["passed"] is True
    assert summary["counts"]["train"] == 96
    assert summary["counts"]["selection"] == 48
    assert summary["checks"]["selection_each_quarter_minimum"] is True
    assert summary["maximum_month_share"]["train"] == pytest.approx(4 / 96)

    all_long = clock.copy()
    all_long["side"] = 1
    failed = support.support_gate_summary(
        all_long,
        _passing_packet_audit(),
        delayed_dropped={"train": 0, "selection": 0},
    )
    assert failed["passed"] is False
    assert failed["checks"]["train_short_minimum"] is False
    control = support.support_gate_summary(
        all_long,
        _passing_packet_audit(),
        control_name="constant_long_same_clock",
    )
    assert control["waived_checks"] == []
    assert control["passed"] is False


def test_source_parser_uses_exact_bytes_it_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_frame(complete_packets=2)
    compressed = gzip.compress(
        source.to_csv(index=False, lineterminator="\n").encode("utf-8"), mtime=0
    )
    path = tmp_path / "source.csv.gz"
    path.write_bytes(compressed)
    registration = {
        "source_manifest": {
            "source_output": {
                "path": str(path),
                "bytes": len(compressed),
                "sha256": hashlib.sha256(compressed).hexdigest(),
            }
        }
    }
    original_read_bytes = Path.read_bytes

    def replace_after_read(candidate: Path) -> bytes:
        payload = original_read_bytes(candidate)
        if candidate.resolve() == path.resolve():
            candidate.write_bytes(b"replacement after exact read")
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    loaded = support.load_source_frame(registration)
    assert len(loaded) == len(source)
    assert path.read_text() == "replacement after exact read"


def test_build_artifacts_is_source_only_deterministic_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_frame(complete_packets=140)
    compressed = gzip.compress(
        source.to_csv(index=False, lineterminator="\n").encode("utf-8"), mtime=0
    )
    source_path = tmp_path / "synthetic-source.csv.gz"
    source_path.write_bytes(compressed)
    source_binding = {
        "path": str(source_path),
        "bytes": len(compressed),
        "sha256": hashlib.sha256(compressed).hexdigest(),
    }
    reads: list[Path] = []
    original_read_bytes = Path.read_bytes

    def track_reads(path: Path) -> bytes:
        reads.append(path.resolve())
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", track_reads)
    monkeypatch.setattr(support, "_source_binding", lambda _: source_binding)
    monkeypatch.setattr(
        support,
        "_exact_source_audit",
        lambda _: {"passed": True, "checks": {"synthetic": True}},
    )
    cfg = _cfg(tmp_path / "first")
    artifact = support.build_support_artifacts(cfg)
    assert reads == [
        support._repository_path(support.DEFAULT_PREREGISTRATION),
        source_path.resolve(),
    ]
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    assert artifact["manifest_hash"] == support.canonical_hash(core)
    assert artifact["outcome_boundary"] == support.OUTCOME_BOUNDARY
    assert artifact["performance_values_opened"] is False
    assert artifact["feature_audit"]["source_values_summarized"] is False
    assert artifact["event_rows_published"] == 0
    assert artifact["feature_values_published"] == 0
    assert set(artifact["control_support_gates"]) == set(support.CONTROL_NAMES)
    assert Path(cfg.output).is_file()
    assert not list(Path(cfg.artifact_root).glob("*.csv*"))

    reads.clear()
    cfg2 = _cfg(tmp_path / "second")
    artifact2 = support.build_support_artifacts(cfg2)
    assert artifact["sealed_clock_commitments"]["primary"]["frame_hash"] == (
        artifact2["sealed_clock_commitments"]["primary"]["frame_hash"]
    )
    assert artifact["sealed_clock_commitments"]["controls"]["frame_hash"] == (
        artifact2["sealed_clock_commitments"]["controls"]["frame_hash"]
    )
    with pytest.raises(FileExistsError, match="immutable"):
        support.build_support_artifacts(cfg)


def test_single_artifact_publish_failure_leaves_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_frame(complete_packets=140)
    monkeypatch.setattr(support, "load_source_frame", lambda _: source.copy())
    monkeypatch.setattr(
        support,
        "_exact_source_audit",
        lambda _: {"passed": True, "checks": {"synthetic": True}},
    )
    cfg = _cfg(tmp_path)
    def fail_publish(temporary: Path, final: Path) -> None:
        raise OSError("injected publish failure")

    monkeypatch.setattr(support, "_publish_new", fail_publish)
    with pytest.raises(OSError, match="injected"):
        support.build_support_artifacts(cfg)
    assert not Path(cfg.output).exists()


def test_frozen_preregistration_and_output_path_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = support.validate_frozen_preregistration(
        support.DEFAULT_PREREGISTRATION
    )
    assert artifact["policy_hash"] == support.EXPECTED_POLICY_HASH

    monkeypatch.setattr(support, "EXPECTED_POLICY_HASH", "0" * 64)
    with pytest.raises(RuntimeError, match="policy hash drift"):
        support.validate_frozen_preregistration(support.DEFAULT_PREREGISTRATION)
    monkeypatch.undo()

    cfg = _cfg(tmp_path)
    registration = support.validate_frozen_preregistration(cfg.preregistration)
    with pytest.raises(ValueError, match="must be JSON"):
        support._validate_config(
            replace(cfg, output=str(Path(cfg.artifact_root) / "support.txt")),
            registration,
        )
    with pytest.raises(ValueError, match="artifact root"):
        support._validate_config(
            replace(cfg, output=str(tmp_path / "outside.json")), registration
        )
    with pytest.raises(ValueError, match="aliases"):
        support._validate_config(
            replace(cfg, output=str(support.DEFAULT_PREREGISTRATION)), registration
        )
