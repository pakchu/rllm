from __future__ import annotations

from collections import OrderedDict
from datetime import timedelta
import gzip
import hashlib
import io
import itertools
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from training import build_block_clearing_relational_topology_support as s
from training import preregister_block_clearing_relational_topology as p


START = s.SOURCE_START_SECONDS


def _synthetic_source(
    rows: int,
    *,
    start_seconds: int = START,
    interval_seconds: int = 600,
) -> pd.DataFrame:
    ids = [f"{index + 1:064x}" for index in range(rows)]
    previous = ["0" * 64, *ids[:-1]]
    timestamp = np.asarray(
        [
            start_seconds + (index + 1) * interval_seconds
            for index in range(rows)
        ],
        dtype=np.int64,
    )
    tx_count = np.asarray(
        [100 + index % 11 for index in range(rows)],
        dtype=np.int64,
    )
    size = np.asarray(
        [1_000 + index % 17 for index in range(rows)],
        dtype=np.int64,
    )
    weight = np.asarray(
        [3_000 + index % 31 for index in range(rows)],
        dtype=np.int64,
    )
    inputs = np.asarray(
        [120 + index % 13 for index in range(rows)],
        dtype=np.int64,
    )
    outputs = inputs + np.asarray(
        [(index % 7) - 3 for index in range(rows)],
        dtype=np.int64,
    )
    return pd.DataFrame(
        {
            "height": np.arange(1_000, 1_000 + rows, dtype=np.int64),
            "id": ids,
            "previousblockhash": previous,
            "timestamp": timestamp,
            "mediantime": timestamp.copy(),
            "tx_count": tx_count,
            "size": size,
            "weight": weight,
            "total_fees": np.asarray(
                [2_000 + index % 29 for index in range(rows)],
                dtype=np.int64,
            ),
            "total_inputs": inputs,
            "total_outputs": outputs,
            "utxo_set_change": outputs - inputs,
        },
        columns=p.SOURCE_ALLOWLIST,
    )


def _valid_tokens(index: int = 0) -> OrderedDict[str, str]:
    return OrderedDict(
        (
            (column, vocabulary[index % len(vocabulary)])
            for column, vocabulary in p.TOKEN_SCHEMA
        )
    )


def _candidate(
    *,
    bucket_start: str,
    signal: str,
    entry: str,
    exit_time: str,
    index: int = 0,
) -> dict[str, object]:
    start = pd.Timestamp(bucket_start)
    record: dict[str, object] = {
        "bucket_start": start,
        "bucket_end": start + pd.Timedelta(hours=12),
        "anchor_timestamp": start + pd.Timedelta(hours=12),
        "anchor_mediantime": start + pd.Timedelta(hours=12),
        "confirmation_timestamp": start + pd.Timedelta(hours=13),
        "confirmation_mediantime": start + pd.Timedelta(hours=13),
        "signal_available_time": pd.Timestamp(signal),
        "entry_time": pd.Timestamp(entry),
        "exit_time": pd.Timestamp(exit_time),
        **_valid_tokens(index),
    }
    record["signal_id"] = s.signal_id(record)
    return record


def _clock_row(entry: pd.Timestamp, index: int) -> dict[str, object]:
    signal = entry - pd.Timedelta(minutes=5)
    bucket = signal - pd.Timedelta(days=2)
    record: dict[str, object] = {
        "bucket_start": bucket.floor("12h"),
        "signal_available_time": signal,
        "entry_time": entry,
        "exit_time": entry + pd.Timedelta(hours=6),
        **_valid_tokens(index),
    }
    record["signal_id"] = s.signal_id(record)
    return record


def _support_clock(include_2023: bool = True) -> pd.DataFrame:
    entries: list[pd.Timestamp] = []
    entries.extend(
        pd.date_range(
            "2020-03-01T06:00:00Z",
            "2020-12-31T18:00:00Z",
            freq="12h",
        )
    )
    entries.extend(
        pd.date_range(
            "2021-01-01T06:00:00Z",
            "2022-12-31T18:00:00Z",
            freq="12h",
        )
    )
    if include_2023:
        entries.extend(
            pd.date_range(
                "2023-01-01T06:00:00Z",
                "2023-12-31T18:00:00Z",
                freq="12h",
            )
        )
    return pd.DataFrame(
        [_clock_row(entry, index) for index, entry in enumerate(entries)],
        columns=s.CLOCK_COLUMNS,
    )


def test_preregistration_and_pre_source_bindings_are_exact() -> None:
    payload = s.validate_preregistration()
    assert payload["manifest_hash"] == s.PREREGISTRATION_MANIFEST_HASH
    assert s.sha256_file(s.PREREGISTRATION) == s.PREREGISTRATION_SHA256
    bindings = s.verify_pre_source_bindings(payload)
    assert bindings[p.SOURCE]["sha256"] == p.SOURCE_SHA256
    assert bindings[p.REFERENCE]["sha256"] == p.REFERENCE_SHA256
    assert bindings[str(s.IMPLEMENTATION_CONTRACT)]["sha256"] == (
        s.IMPLEMENTATION_CONTRACT_SHA256
    )


def test_validate_source_frame_checks_chain_identity_weight_and_reference() -> None:
    frame = _synthetic_source(12)
    reference = frame.loc[:, list(p.REFERENCE_ALLOWLIST)].copy()
    validated = s.validate_source_frame(
        frame,
        reference=reference,
        require_frozen_range=False,
    )
    assert validated["height"].tolist() == list(range(1_000, 1_012))
    assert validated["utxo_set_change"].tolist() == (
        validated["total_outputs"] - validated["total_inputs"]
    ).tolist()

    broken = frame.copy()
    broken.loc[5, "previousblockhash"] = "f" * 64
    with pytest.raises(RuntimeError, match="parent linkage"):
        s.validate_source_frame(broken, require_frozen_range=False)

    broken = frame.copy()
    broken.loc[3, "utxo_set_change"] += 1
    with pytest.raises(RuntimeError, match="UTXO identity"):
        s.validate_source_frame(broken, require_frozen_range=False)

    broken = frame.copy()
    broken.loc[3, "weight"] = broken.loc[3, "size"] - 1
    with pytest.raises(RuntimeError, match="size/weight"):
        s.validate_source_frame(broken, require_frozen_range=False)

    broken = frame.copy()
    broken.loc[3, "mediantime"] = broken.loc[2, "mediantime"] - 1
    with pytest.raises(RuntimeError, match="nondecreasing"):
        s.validate_source_frame(broken, require_frozen_range=False)

    broken_reference = reference.copy()
    broken_reference.loc[4, "weight"] += 1
    with pytest.raises(RuntimeError, match="reference basic fields"):
        s.validate_source_frame(
            frame,
            reference=broken_reference,
            require_frozen_range=False,
        )


def test_validate_source_frame_rejects_noninteger_and_2024() -> None:
    frame = _synthetic_source(4)
    frame["tx_count"] = frame["tx_count"].astype(object)
    frame.loc[1, "tx_count"] = "1.5"
    with pytest.raises(RuntimeError, match="exact integers"):
        s.validate_source_frame(frame, require_frozen_range=False)

    frame = _synthetic_source(4)
    frame.loc[3, "timestamp"] = s.SOURCE_END_SECONDS
    with pytest.raises(RuntimeError, match="frozen cutoff"):
        s.validate_source_frame(frame, require_frozen_range=False)


def test_anchor_confirmation_and_late_backdated_member_are_prefix_closed() -> None:
    full = _synthetic_source(362)
    full.loc[360, "timestamp"] = START + 1_200
    validated = s.validate_source_frame(
        full,
        require_frozen_range=False,
    )
    buckets, audit = s.build_causal_buckets(
        validated,
        start_seconds=START,
        end_seconds=START + 43_200,
    )
    assert len(buckets) == 1
    row = buckets.iloc[0]
    assert row["anchor_height"] == 1_071
    assert row["confirmation_height"] == 1_359
    assert row["member_count"] == 71
    assert row["late_member_count"] == 1
    assert audit["late_backdated_members_excluded"] == 1
    assert audit["prefix_replay_buckets_checked"] == 1
    assert audit["prefix_replay_passed"] is True

    prefix = validated.iloc[:360].reset_index(drop=True)
    prefix_buckets, _ = s.build_causal_buckets(
        prefix,
        start_seconds=START,
        end_seconds=START + 43_200,
    )
    assert prefix_buckets.iloc[0]["state_digest"] == row["state_digest"]
    for column in (
        *s.PRIMITIVE_COLUMNS,
        "signal_available_time",
        "entry_time",
        "exit_time",
    ):
        assert prefix_buckets.iloc[0][column] == row[column]


def test_prefix_max_clock_precedes_embargo_ceiling_latency_and_hold() -> None:
    frame = _synthetic_source(362)
    forward_clock = START + 250_001
    frame.loc[200, "timestamp"] = forward_clock
    validated = s.validate_source_frame(
        frame,
        require_frozen_range=False,
        cutoff_seconds=START + 1_000_000,
    )
    buckets, _ = s.build_causal_buckets(
        validated,
        start_seconds=START,
        end_seconds=START + 43_200,
    )
    row = buckets.iloc[0]
    raw = forward_clock + p.Policy().minimum_embargo_seconds
    expected_signal = pd.Timestamp(p.ceil_5m(raw), unit="s", tz="UTC")
    assert row["prefix_max_timestamp"] == forward_clock
    assert row["signal_available_time"] == expected_signal
    assert row["entry_time"] - row["signal_available_time"] == timedelta(
        minutes=5
    )
    assert row["exit_time"] - row["entry_time"] == timedelta(hours=6)


def test_primitive_equations_are_exact() -> None:
    members = _synthetic_source(2)
    values = s.primitive_values(members)
    weight = members["weight"].to_numpy(dtype=float)
    size = members["size"].to_numpy(dtype=float)
    tx = members["tx_count"].to_numpy(dtype=float)
    fees = members["total_fees"].to_numpy(dtype=float)
    utxo = members["utxo_set_change"].to_numpy(dtype=float)
    assert values["CADENCE"] == pytest.approx(np.log(2.0))
    assert values["UTILIZATION"] == pytest.approx(
        np.log((weight.sum() + 1.0) / (8_000_000.0 + 1.0))
    )
    assert values["PACKING"] == pytest.approx(
        np.log((tx.sum() + 1.0) / (weight.sum() + 1.0))
    )
    assert values["FEE"] == pytest.approx(
        np.log((fees.sum() + 1.0) / (weight.sum() + 1.0))
    )
    assert values["UTXO"] == pytest.approx(
        utxo.sum() / (tx.sum() + 1.0)
    )
    assert values["WITNESS"] == pytest.approx(
        (4.0 * size.sum() - weight.sum()) / (4.0 * size.sum())
    )
    utilization = weight / 4_000_000.0
    assert values["LOAD_DISPERSION"] == pytest.approx(
        np.median(np.abs(utilization - np.median(utilization)))
    )
    density = np.log((fees + 1.0) / (weight + 1.0))
    assert values["FEE_DISPERSION"] == pytest.approx(
        np.median(np.abs(density - np.median(density)))
    )


def test_rank_history_is_strictly_prior_capped_and_first_state_is_predecessor() -> None:
    rows: list[dict[str, object]] = []
    for index in range(300):
        record: dict[str, object] = {"bucket_start_seconds": index * 43_200}
        for primitive in p.PRIMITIVES:
            record[primitive.lower()] = float(index)
        rows.append(record)
    ranked = s.attach_strict_prior_ranks(pd.DataFrame(rows))
    assert ranked.loc[:125, "rank_complete"].eq(False).all()
    assert ranked.loc[126, "rank_complete"]
    assert ranked.loc[126, "cadence_rank"] == 1.0
    expected = p.strict_prior_midrank(
        299.0,
        [float(value) for value in range(47, 299)],
    )
    assert ranked.loc[299, "cadence_rank"] == expected

    complete = ranked.loc[ranked["rank_complete"]].copy()
    base = pd.Timestamp("2020-01-01T00:00:00Z")
    for offset, column in enumerate(s.INTERNAL_TIME_COLUMNS):
        complete[column] = [
            base + pd.Timedelta(hours=12 * index + offset)
            for index in range(len(complete))
        ]
    candidates, funnel = s.build_token_candidates(complete)
    assert funnel["first_rank_complete_predecessor_only"] == 1
    assert len(candidates) == len(complete) - 1


def test_relational_tokens_cover_pair_leader_breadth_and_transitions() -> None:
    current = OrderedDict(
        zip(
            p.PRIMITIVES,
            (0.9, 0.5, 0.8, 0.1, 0.7, 0.3, 0.6, 0.2),
            strict=True,
        )
    )
    previous = OrderedDict(
        zip(
            p.PRIMITIVES,
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
            strict=True,
        )
    )
    tokens = s.relational_tokens(current, previous)
    assert list(tokens) == list(p.TOKEN_COLUMNS)
    assert tokens["cadence_utilization"] == "CADENCE_LEADS"
    assert tokens["utilization_fee"] == "UTILIZATION_LEADS"
    assert tokens["packing_witness"] == "PACKING_LEADS"
    assert tokens["utxo_fee"] == "UTXO_LEADS"
    assert tokens["load_fee_dispersion"] == "LOAD_WIDER"
    assert tokens["high_leader"] == "CADENCE"
    assert tokens["low_leader"] == "FEE"
    assert tokens["relation_breadth"] == "LEFT_BROAD"

    boundary = current.copy()
    boundary["UTILIZATION"] = 0.0
    boundary["CADENCE"] = 1.0 / 6.0
    assert s.relational_tokens(boundary, previous)[
        "cadence_utilization"
    ] == "BALANCED"


def test_reservation_is_action_independent_half_open_and_before_split_filter() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(
                bucket_start="2020-12-31T12:00:00Z",
                signal="2020-12-31T22:55:00Z",
                entry="2020-12-31T23:00:00Z",
                exit_time="2021-01-01T05:00:00Z",
            ),
            _candidate(
                bucket_start="2021-01-01T00:00:00Z",
                signal="2021-01-01T00:55:00Z",
                entry="2021-01-01T01:00:00Z",
                exit_time="2021-01-01T07:00:00Z",
                index=1,
            ),
            _candidate(
                bucket_start="2021-01-01T12:00:00Z",
                signal="2021-01-01T12:55:00Z",
                entry="2021-01-01T13:00:00Z",
                exit_time="2021-01-01T19:00:00Z",
                index=2,
            ),
        ]
    )
    reserved = s.reserve_candidates(candidates)
    assert reserved["reserved"].tolist() == [True, False, True]
    emitted, funnel = s.eligible_clock(reserved)
    assert funnel == {
        "token_ready": 3,
        "globally_reserved": 2,
        "overlap_suppressed": 1,
        "split_suppressed_after_reservation": 1,
        "emitted": 1,
    }
    assert emitted["entry_time"].tolist() == [
        pd.Timestamp("2021-01-01T13:00:00Z")
    ]

    exact_boundary = _candidate(
        bucket_start="2021-12-31T00:00:00Z",
        signal="2021-12-31T17:55:00Z",
        entry="2021-12-31T18:00:00Z",
        exit_time="2022-01-01T00:00:00Z",
    )
    assert s.split_contained(exact_boundary) == (False, "2021")


def test_development_gates_ignore_2023_incidence_and_token_distribution() -> None:
    without_eval = _support_clock(include_2023=False)
    with_eval = _support_clock(include_2023=True)
    base = s.support_checks(
        without_eval,
        development_replay_passed=True,
    )
    extended = s.support_checks(
        with_eval,
        development_replay_passed=True,
    )
    assert base[3] == extended[3]
    assert base[4] == extended[4]
    assert all(base[3].values())
    assert all(base[4].values())
    assert base[5]["events"] == 0
    assert extended[5]["events"] == 730
    assert extended[5]["boolean_gate"] is False
    assert extended[5][
        "may_authorize_continue_retire_repair_or_selection"
    ] is False
    assert not any(name.startswith("2023:") for name in extended[4])

    malformed_eval = with_eval.copy()
    eval_index = malformed_eval.index[
        malformed_eval["entry_time"].ge(
            pd.Timestamp("2023-01-01T00:00:00Z")
        )
    ][0]
    malformed_eval.loc[eval_index, "exit_time"] = (
        malformed_eval.loc[eval_index, "entry_time"]
        + pd.Timedelta(hours=7)
    )
    malformed = s.support_checks(
        malformed_eval,
        development_replay_passed=True,
        eval_replay_report_only_passed=False,
    )
    assert malformed[3] == extended[3]
    assert malformed[4] == extended[4]
    operational = malformed[5]["operational_validity_report"]
    assert operational["prefix_replay_passed"] is False
    assert operational["timing_integrity"] is False


def test_token_support_reports_exact_signature_and_train_vocabulary() -> None:
    rows = _support_clock(include_2023=True)
    _, _, reports, _, token_checks, eval_report = s.support_checks(
        rows,
        development_replay_passed=True,
    )
    assert reports["train"]["maximum_exact_signature_share"] < 0.05
    assert reports["2022"]["maximum_exact_signature_share"] < 0.05
    assert all(token_checks.values())
    assert all(
        not values
        for values in eval_report["train_vocabulary_coverage"].values()
    )


def test_clock_serialization_is_deterministic_and_rejects_extra_fields() -> None:
    rows = _support_clock(include_2023=False).iloc[:5].copy()
    first = s.deterministic_clock_bytes(rows)
    second = s.deterministic_clock_bytes(rows.sample(frac=1, random_state=7))
    assert first == second
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as zipped:
        text = zipped.read().decode("utf-8")
    assert text.splitlines()[0].split(",") == list(s.CLOCK_COLUMNS)
    assert "raw" not in text.splitlines()[0].lower()
    assert "action" not in text.splitlines()[0].lower()
    assert "side" not in text.splitlines()[0].lower()
    broken = rows.assign(raw_rank=0.5)
    with pytest.raises(RuntimeError, match="schema drift"):
        s.deterministic_clock_bytes(broken)


def test_synthetic_end_to_end_build_cannot_authorize_outcomes() -> None:
    bucket_count = 130
    frame = _synthetic_source(bucket_count * 72 + 320)
    report, clock_bytes = s.build_support_from_frame(
        frame,
        start_seconds=START,
        end_seconds=START + bucket_count * 43_200,
    )
    assert report["artifact_eligible"] is False
    assert report["outcomes_opened"] is False
    assert report["source_incidence_opened"] is False
    assert report["authorized_next_stage"] is None
    assert report["feature_funnel"]["rank_complete_states"] == 4
    assert report["feature_funnel"]["token_ready_states"] == 3
    assert report["clock"]["sha256"] == hashlib.sha256(clock_bytes).hexdigest()
    assert report["outcome_boundary"]["BTC_market_rows_decoded"] == 0
    assert report["outcome_boundary"]["funding_rows_decoded"] == 0
    assert report["outcome_boundary"]["future_return_rows_decoded"] == 0
    assert report["outcome_boundary"]["model_training_runs"] == 0


def test_write_once_rejects_existing_drift(tmp_path: Path) -> None:
    output = tmp_path / "artifact.bin"
    assert s._write_once(output, b"first") == "created"
    assert s._write_once(output, b"first") == "verified_existing"
    with pytest.raises(RuntimeError, match="noncanonical"):
        s._write_once(output, b"second")


def test_protocol_commit_guard_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = itertools.count()

    def fake_git(*_args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=1 if next(calls) == 0 else 0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(s, "_git_check", fake_git)
    with pytest.raises(RuntimeError, match="not committed"):
        s._assert_protocol_committed()

    monkeypatch.setattr(
        s,
        "_git_check",
        lambda *_args: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    s._assert_protocol_committed()


def test_first_failure_never_uses_eval_report() -> None:
    assert s.first_failure(
        {"source": True},
        {"train": True, "2022": True},
        artifact_eligible=True,
    ) == ("none", None)
    assert s.first_failure(
        {"source": False},
        {"train": True},
        artifact_eligible=True,
    ) == ("source_support", "source")
    assert s.first_failure(
        {"source": True},
        {"train": False},
        artifact_eligible=True,
    ) == ("token_support", "train")
