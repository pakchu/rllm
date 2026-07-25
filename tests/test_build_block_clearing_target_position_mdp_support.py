from __future__ import annotations

from collections import OrderedDict
import copy
import gzip
import hashlib
import io
import inspect
import itertools
import json
import random
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from training import preregister_block_clearing_relational_topology as bcrt_p
from training import preregister_block_clearing_target_position_mdp as prereg
from training import build_block_clearing_target_position_mdp_support as s


def _tokens(offset: int = 0) -> OrderedDict[str, str]:
    return OrderedDict(
        (
            name,
            vocabulary[(index + offset) % len(vocabulary)],
        )
        for index, (name, vocabulary) in enumerate(bcrt_p.TOKEN_SCHEMA)
    )


def _release(
    index: int,
    *,
    entry_time: pd.Timestamp | None = None,
    token_offset: int | None = None,
) -> dict[str, object]:
    entry = entry_time or (
        pd.Timestamp("2020-01-01T00:00:00Z")
        + index * pd.Timedelta(hours=12)
    )
    return {
        "signal_id": f"BCRT-synthetic-{index:08d}",
        "bucket_start": entry - pd.Timedelta(days=4),
        "confirmation_height": 700_000 + index,
        "entry_time": entry,
        **_tokens(index if token_offset is None else token_offset),
    }


@pytest.fixture(scope="module")
def full_release_rows() -> list[dict[str, object]]:
    entries = pd.date_range(
        "2020-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
        freq="12h",
        inclusive="left",
    )
    return [
        _release(index, entry_time=entry)
        for index, entry in enumerate(entries)
    ]


@pytest.fixture(scope="module")
def full_sequence_bundle(
    full_release_rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    return s.build_sequence_artifacts(full_release_rows)


def _ranked_frame() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    base = pd.Timestamp("2020-03-01T00:00:00Z")
    for index in range(4):
        bucket = base + index * pd.Timedelta(hours=12)
        ranks = {
            f"{name.lower()}_rank": (
                (index + primitive_index + 1) % 10
            )
            / 10
            for primitive_index, name in enumerate(bcrt_p.PRIMITIVES)
        }
        records.append(
            {
                "bucket_start": bucket,
                "bucket_end": bucket + pd.Timedelta(hours=12),
                "anchor_timestamp": bucket + pd.Timedelta(hours=13),
                "anchor_mediantime": bucket + pd.Timedelta(hours=12),
                "confirmation_height": 700_100 + index,
                "confirmation_timestamp": bucket + pd.Timedelta(days=2),
                "confirmation_mediantime": bucket + pd.Timedelta(days=2),
                "signal_available_time": bucket + pd.Timedelta(days=4),
                "entry_time": (
                    bucket + pd.Timedelta(days=4, minutes=5)
                ),
                "exit_time": (
                    bucket + pd.Timedelta(days=4, hours=6, minutes=5)
                ),
                "rank_complete": True,
                **ranks,
            }
        )
    return pd.DataFrame(records)


def test_preregistration_and_implementation_bindings_without_source_decode() -> None:
    payload = s.validate_preregistration()

    assert payload["policy"]["policy_id"] == "BCTP-12H"
    assert payload["manifest_hash"] == s.PREREGISTRATION_MANIFEST_HASH
    assert s.sha256_file(s.IMPLEMENTATION_CONTRACT) == (
        s.IMPLEMENTATION_CONTRACT_SHA256
    )
    assert payload["source_support_gates"]["calendar_boundary_gap"] == (
        "report_only_non_boolean"
    )
    assert payload["report_only_2023"]["may_change_support_boolean"] is False


def test_enrichment_preserves_exact_bcrt_common_rows() -> None:
    ranked = _ranked_frame()
    enriched, funnel, audit = s.enrich_token_candidates(ranked)

    assert list(enriched.columns) == list(s.BCTP_TOKEN_CANDIDATE_COLUMNS)
    assert enriched["confirmation_height"].tolist() == [700_101, 700_102, 700_103]
    assert funnel == {
        "formed_buckets": 4,
        "rank_complete_states": 4,
        "first_rank_complete_predecessor_only": 1,
        "token_ready_states": 3,
    }
    assert audit["development_gate"]["common_projection_identical"] is True
    assert audit["full_source_report_only"][
        "common_projection_identical"
    ] is True
    common = enriched.drop(columns=["confirmation_height"])
    assert audit["full_source_report_only"][
        "common_projection_sha256"
    ] == hashlib.sha256(
        s._candidate_bytes(common)
    ).hexdigest()


def test_same_release_batching_warmup_and_strict_clock() -> None:
    first = pd.Timestamp("2021-05-01T12:00:00Z")
    rows = [
        {
            **_release(0, entry_time=first),
            "signal_id": "z-older-bucket",
            "bucket_start": first - pd.Timedelta(days=5),
            "confirmation_height": 999_999,
        },
        {
            **_release(1, entry_time=first),
            "signal_id": "zz-latest-low-confirmation",
            "bucket_start": first - pd.Timedelta(days=3),
            "confirmation_height": 800_000,
        },
        {
            **_release(2, entry_time=first),
            "signal_id": "a-latest-high-confirmation",
            "bucket_start": first - pd.Timedelta(days=3),
            "confirmation_height": 800_001,
        },
        {
            **_release(3, entry_time=first),
            "signal_id": "b-latest-high-confirmation",
            "bucket_start": first - pd.Timedelta(days=3),
            "confirmation_height": 800_001,
        },
        _release(4, entry_time=first + pd.Timedelta(hours=12)),
        _release(5, entry_time=first + pd.Timedelta(hours=24)),
    ]
    sequences, batching, append = s.build_sequence_artifacts(rows)
    development = batching["development_boolean"]

    assert development["token_ready_source_states"] == 6
    assert development["actionable_releases"] == 3
    assert development["same_release_suppressed"] == 3
    assert development["actionable_entries_strictly_increasing"] is True
    assert development["warmup_exact"] is True
    assert batching["full_source_report_only"][
        "development_sequences_match_full_prefix"
    ] is True
    assert len(sequences) == 1
    assert sequences.iloc[0]["source_signal_id_m2"] == (
        "b-latest-high-confirmation"
    )
    assert append["development_boolean"]["future_append_invariance_passed"] is True

    duplicate = dict(rows[3])
    duplicate[bcrt_p.TOKEN_COLUMNS[0]] = _tokens(7)[
        bcrt_p.TOKEN_COLUMNS[0]
    ]
    with pytest.raises(ValueError, match="duplicate same-release"):
        s.build_sequence_artifacts([*rows, duplicate])


def test_sequence_build_is_append_and_input_order_invariant() -> None:
    rows = [_release(index) for index in range(30)]
    baseline, _, append = s.build_sequence_artifacts(rows)
    extended, _, _ = s.build_sequence_artifacts(
        [*rows, _release(30), _release(31)]
    )
    shuffled = rows.copy()
    random.Random(20260725).shuffle(shuffled)
    reordered, _, _ = s.build_sequence_artifacts(shuffled)

    assert append["development_boolean"]["future_append_invariance_passed"] is True
    assert extended.iloc[: len(baseline)].to_dict("records") == (
        baseline.to_dict("records")
    )
    assert s.deterministic_sequence_bytes(reordered) == (
        s.deterministic_sequence_bytes(baseline)
    )


def test_development_support_passes_and_gap_is_report_only(
    full_sequence_bundle: tuple[
        pd.DataFrame,
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    sequences, batching, append = full_sequence_bundle
    reports, checks, report_only_2023 = s.support_checks(
        sequences,
        batching_audit=batching,
        append_audit=append,
        replay_audit=s._synthetic_replay_audit(),
    )

    assert all(checks.values())
    assert reports["2020"]["events"] >= 500
    assert reports["2021"]["events"] >= 500
    assert reports["2022"]["events"] >= 500
    assert reports["2021"]["active_months"] == 12
    assert reports["2022"]["active_months"] == 12
    assert reports["development"][
        "maximum_exact_source_signature_share"
    ] <= 0.05
    assert reports["development"]["maximum_gap_days_report_only"] is not None
    assert not any("gap" in name or "2023" in name for name in checks)
    assert report_only_2023["boolean_gate"] is False


def test_adversarial_2023_cannot_change_support_boolean(
    full_sequence_bundle: tuple[
        pd.DataFrame,
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    sequences, batching, append = full_sequence_bundle
    base = s.support_checks(
        sequences,
        batching_audit=batching,
        append_audit=append,
        replay_audit=s._synthetic_replay_audit(),
    )
    adversarial = sequences.copy()
    eval_mask = pd.to_datetime(
        adversarial["entry_time"],
        utc=True,
    ).ge(pd.Timestamp("2023-01-01T00:00:00Z"))
    adversarial.loc[eval_mask, "source_signature"] = "one-eval-signature"
    first_eval = adversarial.index[eval_mask][0]
    token_column = prereg.SOURCE_TOKEN_COLUMNS[0]
    adversarial.loc[first_eval, token_column] = "UNKNOWN_2023"
    adversarial.loc[eval_mask, "entry_time"] = (
        pd.Timestamp("2023-01-01T00:00:00Z")
    )
    adversarial.loc[adversarial.index[eval_mask][-1], "entry_time"] = (
        pd.Timestamp("2023-12-31T00:00:00Z")
    )
    changed = s.support_checks(
        adversarial,
        batching_audit=batching,
        append_audit=append,
        replay_audit=s._synthetic_replay_audit(),
    )

    assert changed[1] == base[1]
    assert changed[2]["maximum_exact_source_signature_share"] == 1.0
    assert changed[2]["maximum_gap_days_report_only"] == 364
    assert changed[2]["unknown_vocabulary"][token_column] == ["UNKNOWN_2023"]
    assert changed[2]["may_authorize_continue_retire_repair_or_selection"] is False


def test_unknown_2023_vocabulary_is_contained_during_sequence_build() -> None:
    entries = pd.date_range(
        "2022-12-30T00:00:00Z",
        periods=10,
        freq="12h",
    )
    rows = [
        _release(index, entry_time=entry)
        for index, entry in enumerate(entries)
    ]
    unknown_row = next(
        row
        for row in rows
        if pd.Timestamp(row["entry_time"]).year == 2023
    )
    unknown_row[bcrt_p.TOKEN_COLUMNS[0]] = "UNKNOWN_2023"
    sequences, batching, _ = s.build_sequence_artifacts(rows)
    report = s.sequence_report(s._window(sequences, "2023"))

    assert report["unknown_vocabulary"]
    assert batching["full_source_report_only"][
        "unknown_2023_vocabulary_present"
    ] is True
    assert batching["full_source_report_only"][
        "strict_full_replay_when_vocabulary_known"
    ] is None


def test_report_only_batching_repeats_all_ties_and_duplicate_rejection() -> None:
    entry = pd.Timestamp("2023-01-01T00:00:00Z")
    development = [
        _release(
            90,
            entry_time=entry - pd.Timedelta(hours=24),
        ),
        _release(
            91,
            entry_time=entry - pd.Timedelta(hours=12),
        ),
    ]
    report_only = [
        {
            **_release(92, entry_time=entry),
            "signal_id": "zz-older-bucket",
            "bucket_start": entry - pd.Timedelta(days=5),
            "confirmation_height": 999_999,
        },
        {
            **_release(93, entry_time=entry),
            "signal_id": "zz-latest-low-confirmation",
            "bucket_start": entry - pd.Timedelta(days=3),
            "confirmation_height": 800_000,
        },
        {
            **_release(94, entry_time=entry),
            "signal_id": "a-latest-high-confirmation",
            "bucket_start": entry - pd.Timedelta(days=3),
            "confirmation_height": 800_001,
        },
        {
            **_release(95, entry_time=entry),
            "signal_id": "b-latest-high-confirmation",
            "bucket_start": entry - pd.Timedelta(days=3),
            "confirmation_height": 800_001,
        },
    ]
    sequences, _, _ = s.build_sequence_artifacts(
        [*development, *report_only]
    )

    assert len(sequences) == 1
    assert sequences.iloc[0]["source_signal_id_s0"] == (
        "b-latest-high-confirmation"
    )
    duplicate = dict(report_only[-1])
    duplicate[bcrt_p.TOKEN_COLUMNS[0]] = "UNKNOWN_DUPLICATE"
    with pytest.raises(ValueError, match="duplicate same-release"):
        s.build_sequence_artifacts(
            [*development, *report_only, duplicate]
        )


def test_full_clock_and_full_sequence_diagnostics_cannot_gate(
    full_sequence_bundle: tuple[
        pd.DataFrame,
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    sequences, batching, append = full_sequence_bundle
    replay = s._synthetic_replay_audit()
    baseline = s.support_checks(
        sequences,
        batching_audit=batching,
        append_audit=append,
        replay_audit=replay,
    )[1]
    changed_batching = copy.deepcopy(batching)
    changed_append = copy.deepcopy(append)
    changed_replay = copy.deepcopy(replay)
    changed_batching["full_source_report_only"] = {"corrupt": True}
    changed_append["full_source_report_only"] = {
        "future_append_invariance_passed": False
    }
    changed_replay["full_clock_report_only"] = {
        key: False for key in replay["full_clock_report_only"]
    }
    changed_replay["full_source_report_only"] = {
        "expected_replay_counts_exact": False,
        "common_projection_identical": False,
    }
    changed = s.support_checks(
        sequences,
        batching_audit=changed_batching,
        append_audit=changed_append,
        replay_audit=changed_replay,
    )[1]

    assert changed == baseline


def test_shared_payload_cannot_authorize_self_asserted_real_evidence(
    full_sequence_bundle: tuple[
        pd.DataFrame,
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    parameters = inspect.signature(s._core_payload).parameters

    assert "artifact_eligible" not in parameters
    assert "real_source_evidence" not in parameters
    assert not hasattr(s, "_authorize_real_source")
    assert not hasattr(s, "_artifact_eligible")
    assert not hasattr(s, "_RealSourceEvidence")
    assert not hasattr(s, "_REAL_SOURCE_AUTHORITY")
    assert not hasattr(s, "_evidence_hash")
    sequences, batching, append = full_sequence_bundle
    sequence_bytes = s.deterministic_sequence_bytes(sequences)
    report = s._core_payload(
        sequences,
        source_audit={
            "source_rows_decoded": 213_095,
            "reference_rows_decoded": 213_095,
            "synthetic_or_injected": False,
            "pre_source_bindings": {"self_asserted": True},
        },
        bucket_audit={"formed_buckets": 2_918},
        feature_funnel={
            "rank_complete_states": 2_792,
            "token_ready_states": 2_791,
        },
        replay_audit=s._synthetic_replay_audit(),
        batching_audit=batching,
        append_audit=append,
        preregistration=s.validate_preregistration(),
        sequence_bytes=sequence_bytes,
    )

    assert report["source_support_passed"] is True
    assert report["artifact_eligible"] is False
    assert report["authorized_next_stage"] is None
    assert report["decision"] == "synthetic_build_cannot_authorize_market_access"


def test_replay_mismatch_fails_before_artifact_eligibility(
    full_sequence_bundle: tuple[
        pd.DataFrame,
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    sequences, batching, append = full_sequence_bundle
    replay = s._synthetic_replay_audit()
    replay["train_2022_gate"]["checks_exact"] = False
    _, checks, _ = s.support_checks(
        sequences,
        batching_audit=batching,
        append_audit=append,
        replay_audit=replay,
    )

    assert checks["bcrt_train_2022_checks_exact"] is False
    assert s.first_failure(checks, artifact_eligible=True) == (
        "source_sequence_support",
        "bcrt_train_2022_checks_exact",
    )


def test_sequence_serialization_is_deterministic_and_rejects_schema_drift(
    full_sequence_bundle: tuple[
        pd.DataFrame,
        dict[str, object],
        dict[str, object],
    ],
) -> None:
    sequences = full_sequence_bundle[0].iloc[:10].copy()
    first = s.deterministic_sequence_bytes(sequences)
    second = s.deterministic_sequence_bytes(
        sequences.sample(frac=1, random_state=7)
    )

    assert first == second
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as zipped:
        text = zipped.read().decode("utf-8")
    header = text.splitlines()[0].split(",")
    assert header == list(prereg.SOURCE_SEQUENCE_COLUMNS)
    assert all(
        fragment not in column.lower()
        for column in header
        for fragment in s.FORBIDDEN_COLUMN_FRAGMENTS
    )
    with pytest.raises(RuntimeError, match="schema drift"):
        s.deterministic_sequence_bytes(sequences.assign(raw_rank=0.5))


def test_synthetic_build_cannot_authorize_market_access(
    full_release_rows: list[dict[str, object]],
) -> None:
    report, sequence_bytes = s.build_support_from_release_rows(
        full_release_rows
    )

    assert report["artifact_eligible"] is False
    assert report["source_support_passed"] is True
    assert report["decision"] == "synthetic_build_cannot_authorize_market_access"
    assert report["authorized_next_stage"] is None
    assert report["source_sequence_incidence_opened"] is False
    assert report["sequence_artifact"]["sha256"] == hashlib.sha256(
        sequence_bytes
    ).hexdigest()
    boundary = report["outcome_boundary"]
    assert boundary["new_raw_source_rows_decoded"] == 0
    assert boundary["market_rows_decoded"] == 0
    assert boundary["funding_rows_decoded"] == 0
    assert boundary["future_return_rows_decoded"] == 0
    assert boundary["actions_or_labels_created"] == 0
    assert boundary["model_training_runs"] == 0


def test_report_serialization_is_canonical(
    full_release_rows: list[dict[str, object]],
) -> None:
    report, _ = s.build_support_from_release_rows(full_release_rows)
    reordered = dict(reversed(tuple(report.items())))
    first = s.deterministic_report_bytes(report)
    second = s.deterministic_report_bytes(reordered)

    assert first == second
    assert first.endswith(b"\n")
    assert json.loads(first) == report
    changed = dict(report)
    changed["decision"] = "drift"
    with pytest.raises(RuntimeError, match="manifest hash"):
        s.deterministic_report_bytes(changed)


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

    loader_calls = 0

    def forbidden_loader() -> None:
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("source loader must remain unopened")

    dirty_calls = iter((0, 1))
    monkeypatch.setattr(
        s,
        "_git_check",
        lambda *_args: subprocess.CompletedProcess(
            args=[],
            returncode=next(dirty_calls),
            stdout="",
            stderr="",
        ),
    )
    monkeypatch.setattr(s.bcrt_s, "load_source_frames", forbidden_loader)
    with pytest.raises(RuntimeError, match="differs from HEAD"):
        s.build_real_support_payload()
    assert loader_calls == 0

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


def test_real_output_paths_are_frozen() -> None:
    with pytest.raises(RuntimeError, match="report output path"):
        s.write_support(report_output="results/not-bctp.json")
    with pytest.raises(RuntimeError, match="sequence output path"):
        s.write_support(sequence_output="data/not-bctp.csv.gz")
