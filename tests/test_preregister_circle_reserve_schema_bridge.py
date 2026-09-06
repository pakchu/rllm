from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, cast

import pytest

from training import preregister_circle_reserve_schema_bridge as p


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha1(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@pytest.fixture
def synthetic_esdi(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], bytes, dict[Path, str]]:
    comparators = {
        f"synthetic_{index:02d}": {
            "path": f"results/synthetic_comparator_{index:02d}.json",
            "sha256": _sha256(f"comparator-{index}"),
        }
        for index in range(p.ESDI_COMPARATOR_COUNT)
    }
    closure_hashes = {
        "training/synthetic_runtime_a.py": _sha256("runtime-a"),
        "training/synthetic_runtime_b.py": _sha256("runtime-b"),
    }
    closure = {
        "paths": list(closure_hashes),
        "sha256": closure_hashes,
    }
    authority = {
        "runtime_code_closure": closure,
        "environment_lock": {
            "path": "config/synthetic-runtime.lock",
            "sha256": _sha256("runtime-lock"),
        },
    }
    gross9 = {
        "authority": authority,
        "weights": copy.deepcopy(p.GROSS9_WEIGHTS),
        "baseline_gross": 9.0,
        "synthetic_baseline": {"configured_gross": 9.0},
    }
    core = {
        "novelty": {"frozen_comparator_artifacts": comparators},
        "gross9": gross9,
    }
    payload = {**core, "manifest_hash": p.canonical_hash(core)}
    raw = _canonical_json_bytes(payload)

    monkeypatch.setattr(p, "ESDI_MANIFEST_HASH", payload["manifest_hash"])
    monkeypatch.setattr(p, "ESDI_PREREGISTRATION_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(p, "ESDI_COMPARATOR_SUBTREE_SHA256", p.canonical_hash(comparators))
    monkeypatch.setattr(p, "ESDI_GROSS9_SUBTREE_SHA256", p.canonical_hash(gross9))
    monkeypatch.setattr(p, "ESDI_GROSS9_AUTHORITY_SHA256", p.canonical_hash(authority))
    monkeypatch.setattr(p, "ESDI_RUNTIME_CLOSURE_SHA256", p.canonical_hash(closure))
    monkeypatch.setattr(
        p,
        "_read_regular",
        lambda path: raw
        if Path(path) == p.ESDI_PREREGISTRATION_PATH
        else pytest.fail(f"unexpected synthetic read: {path}"),
    )
    expected_paths = {
        Path("config/synthetic-runtime.lock"): _sha256("runtime-lock"),
        **{Path(path): digest for path, digest in closure_hashes.items()},
    }
    return payload, raw, expected_paths


@pytest.fixture
def manifest_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    comparators = {
        f"synthetic_{index:02d}": {"synthetic": True}
        for index in range(p.ESDI_COMPARATOR_COUNT)
    }
    authority = {
        "frozen_comparator_artifacts": comparators,
        "gross9": {
            "authority": {"runtime_code_closure": {"paths": []}},
            "weights": copy.deepcopy(p.GROSS9_WEIGHTS),
            "baseline_gross": 9.0,
        },
    }
    identity_paths = (
        p.SOURCE_DECISION_PATH,
        p.MECHANISM_DECISION_PATH,
        p.PRODUCER_PATH,
        p.TEST_PATH,
        p.ESDI_PREREGISTRATION_PATH,
        p.ESDI_NOVELTY_HELPER_PATH,
        p.ESDI_ECONOMICS_HELPER_PATH,
    )
    monkeypatch.setattr(p, "load_esdi_authority", lambda: copy.deepcopy(authority))
    monkeypatch.setattr(p, "committed_identity_paths", lambda: identity_paths)

    paths = sorted(str(path) for path in identity_paths)
    blobs = {path: _sha1(path) for path in paths}
    blobs[str(p.SOURCE_DECISION_PATH)] = p.SOURCE_DECISION_GIT_BLOB
    blobs[str(p.MECHANISM_DECISION_PATH)] = p.MECHANISM_DECISION_GIT_BLOB
    blobs[str(p.ESDI_PREREGISTRATION_PATH)] = p.ESDI_PREREGISTRATION_GIT_BLOB
    blobs[str(p.ESDI_NOVELTY_HELPER_PATH)] = p.ESDI_NOVELTY_HELPER_GIT_BLOB
    blobs[str(p.ESDI_ECONOMICS_HELPER_PATH)] = p.ESDI_ECONOMICS_HELPER_GIT_BLOB
    sha256 = {path: _sha256(path) for path in paths}
    sha256[str(p.SOURCE_DECISION_PATH)] = p.SOURCE_DECISION_SHA256
    sha256[str(p.MECHANISM_DECISION_PATH)] = p.MECHANISM_DECISION_SHA256
    sha256[str(p.ESDI_PREREGISTRATION_PATH)] = p.ESDI_PREREGISTRATION_SHA256
    sha256[str(p.ESDI_NOVELTY_HELPER_PATH)] = p.ESDI_NOVELTY_HELPER_SHA256
    sha256[str(p.ESDI_ECONOMICS_HELPER_PATH)] = p.ESDI_ECONOMICS_HELPER_SHA256
    identity: dict[str, Any] = {
        "branch": p.EXPECTED_BRANCH,
        "head_commit": "1" * 40,
        "head_tree": "2" * 40,
        "upstream": f"origin/{p.EXPECTED_BRANCH}",
        "upstream_commit": "1" * 40,
        "git_blobs": blobs,
        "sha256": sha256,
        "whole_worktree_clean_required": True,
        "head_equals_upstream_required": True,
        "protocol_seal_hash": p.canonical_hash(
            {"git_blobs": blobs, "sha256": sha256}
        ),
    }
    return identity, p.build_manifest(identity)


@pytest.fixture
def write_once_environment(
    manifest_environment: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    identity, payload = manifest_environment
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(p, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        p,
        "frozen_repository_identity",
        lambda: copy.deepcopy(identity),
    )
    monkeypatch.setattr(
        p,
        "build_manifest",
        lambda observed: copy.deepcopy(payload)
        if dict(observed) == identity
        else pytest.fail("write_once used an unexpected repository identity"),
    )
    monkeypatch.setattr(
        p,
        "_validate_creation_publish_state",
        lambda observed, _temporary: (
            None
            if dict(observed) == identity
            else pytest.fail("write_once publish identity drift")
        ),
    )
    monkeypatch.setattr(p, "validate_recorded_repository", lambda _identity: None)
    return identity, payload, tmp_path / p.DEFAULT_OUTPUT


def test_report_month_envelope_is_exact_and_consecutive() -> None:
    assert p.REPORT_MONTHS == (
        "2022-11",
        "2022-12",
        "2023-01",
        "2023-02",
        "2023-03",
        "2023-04",
        "2023-05",
        "2023-06",
        "2023-07",
        "2023-08",
        "2023-09",
        "2023-10",
        "2023-11",
        "2023-12",
        "2024-01",
        "2024-02",
        "2024-03",
        "2024-04",
        "2024-05",
        "2024-06",
        "2024-07",
        "2024-08",
        "2024-09",
        "2024-10",
        "2024-11",
        "2024-12",
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
        "2025-06",
        "2025-07",
        "2025-08",
        "2025-09",
        "2025-10",
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
    )


def test_discovery_month_envelope_and_daily_index_count_are_exact() -> None:
    assert p.DISCOVERY_MONTHS == p.REPORT_MONTHS[1:] + ("2026-05",)
    assert len(p.DISCOVERY_MONTHS) == 42
    assert p.DISCOVERY_DAY_COUNT == len(p.DISCOVERY_MONTHS) * 15 == 630


def test_committed_documents_and_esdi_authorities_match_frozen_bytes() -> None:
    p.validate_frozen_dependencies()
    assert p.sha256_file(p.SOURCE_DECISION_PATH) == p.SOURCE_DECISION_SHA256
    assert p.sha256_file(p.MECHANISM_DECISION_PATH) == p.MECHANISM_DECISION_SHA256
    authority = p.load_esdi_authority()
    assert len(authority["frozen_comparator_artifacts"]) == 18
    assert p.canonical_hash(authority["gross9"]) == p.ESDI_GROSS9_SUBTREE_SHA256
    assert p.esdi_bound_path_hashes()


@pytest.mark.parametrize(
    ("constant", "message"),
    [
        ("SOURCE_DECISION_GIT_BLOB", "source-axis commit blob drift"),
        ("MECHANISM_DECISION_GIT_BLOB", "mechanism commit blob drift"),
    ],
)
def test_frozen_decision_commit_rejects_blob_drift(
    constant: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, constant, "0" * 40)
    with pytest.raises(RuntimeError, match=message):
        p.validate_frozen_decision_commits()


def test_synthetic_esdi_authority_accepts_exact_nested_hashes_and_closure(
    synthetic_esdi: tuple[dict[str, Any], bytes, dict[Path, str]],
) -> None:
    payload, _raw, expected_paths = synthetic_esdi
    loaded = p.load_esdi_authority()
    assert loaded["frozen_comparator_artifacts"] == payload["novelty"][
        "frozen_comparator_artifacts"
    ]
    assert loaded["gross9"] == payload["gross9"]
    assert p.esdi_bound_path_hashes() == expected_paths


@pytest.mark.parametrize(
    ("constant", "message"),
    [
        ("ESDI_MANIFEST_HASH", "manifest hash drift"),
        ("ESDI_COMPARATOR_SUBTREE_SHA256", "comparator registry hash drift"),
        ("ESDI_GROSS9_SUBTREE_SHA256", "Gross9 subtree hash drift"),
        ("ESDI_GROSS9_AUTHORITY_SHA256", "Gross9 authority hash drift"),
        ("ESDI_RUNTIME_CLOSURE_SHA256", "runtime closure hash drift"),
    ],
)
def test_synthetic_esdi_authority_rejects_each_authority_hash_drift(
    constant: str,
    message: str,
    synthetic_esdi: tuple[dict[str, Any], bytes, dict[Path, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, constant, "0" * 64)
    with pytest.raises(RuntimeError, match=message):
        p.load_esdi_authority()


def test_synthetic_esdi_authority_rejects_noncanonical_bytes(
    synthetic_esdi: tuple[dict[str, Any], bytes, dict[Path, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _raw, _paths = synthetic_esdi
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    monkeypatch.setattr(p, "ESDI_PREREGISTRATION_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(p, "_read_regular", lambda _path: raw)
    with pytest.raises(RuntimeError, match="bytes are not canonical"):
        p.load_esdi_authority()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", Fraction(0)),
        ("0.10", Fraction(1, 10)),
        ("42.125", Fraction(337, 8)),
    ],
)
def test_parse_exact_decimal_preserves_exact_rational_value(
    raw: str, expected: Fraction
) -> None:
    assert p.parse_exact_decimal(raw) == expected


@pytest.mark.parametrize("raw", [".5", "01", "1.", "-1", "1e2", "NaN", "inf", 1])
def test_parse_exact_decimal_rejects_noncanonical_input(raw: Any) -> None:
    with pytest.raises(ValueError, match="decimal"):
        p.parse_exact_decimal(raw)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (("1.00", "2.00", "3.00"), 1),
        ((Fraction(3), Fraction(2), Fraction(1)), -1),
        (("1", "3", "2"), 0),
    ],
)
def test_path_vote_uses_exact_end_min_max_balance(
    values: tuple[str | Fraction, ...], expected: int
) -> None:
    assert p.path_vote(values) == expected


def test_path_vote_rejects_empty_negative_or_inexact_values() -> None:
    invalid_values: tuple[tuple[Any, ...], ...] = (
        (),
        (Fraction(-1),),
        (0.5,),
    )
    for values in invalid_values:
        with pytest.raises(ValueError, match="path"):
            p.path_vote(cast(Any, values))


@pytest.mark.parametrize(
    ("current", "previous", "expected"),
    [(9, 10, 1), (11, 10, -1), (10, 10, 0)],
)
def test_maturity_vote_has_exact_lower_is_positive_direction(
    current: int, previous: int, expected: int
) -> None:
    assert p.maturity_vote(current, previous) == expected


def test_maturity_vote_rejects_boolean_negative_and_noninteger_values() -> None:
    invalid_values: tuple[tuple[Any, Any], ...] = (
        (True, 1),
        (-1, 1),
        (1.0, 1),
    )
    for current, previous in invalid_values:
        with pytest.raises(ValueError, match="nonnegative integers"):
            p.maturity_vote(cast(Any, current), cast(Any, previous))


@pytest.mark.parametrize(
    ("components", "expected"),
    [
        ((1, 1, -1, 0), 1),
        ((-1, -1, 1, 0), -1),
        ((1, -1, 0, 0), 0),
    ],
)
def test_primary_vote_uses_sign_of_four_component_sum(
    components: tuple[int, ...], expected: int
) -> None:
    assert p.primary_vote(components) == expected


def test_primary_vote_requires_four_ternary_integer_components() -> None:
    for components in ((1, 0, -1), (1, 0, -1, 2), (True, 0, -1, 0)):
        with pytest.raises(ValueError, match="four exact component votes"):
            p.primary_vote(components)


def test_canonical_timestamp_normalizes_aware_whole_second_datetime() -> None:
    value = datetime(
        2025,
        1,
        2,
        12,
        34,
        56,
        tzinfo=timezone(timedelta(hours=9)),
    )
    assert p.canonical_utc_timestamp(value) == "2025-01-02T03:34:56Z"
    assert p.canonical_utc_timestamp("2025-01-02T03:34:56Z") == (
        "2025-01-02T03:34:56Z"
    )


@pytest.mark.parametrize(
    "value",
    [
        "2025-01-02T03:34:56+00:00",
        "2025-01-02t03:34:56Z",
        "2025-02-30T03:34:56Z",
        datetime(2025, 1, 2, 3, 4, 5),
        datetime(2025, 1, 2, 3, 4, 5, 1, tzinfo=timezone.utc),
    ],
)
def test_canonical_timestamp_rejects_noncanonical_or_ambiguous_values(
    value: str | datetime,
) -> None:
    with pytest.raises(ValueError, match="timestamp|datetime"):
        p.canonical_utc_timestamp(value)


def test_primary_signal_id_hashes_exact_canonical_identity_bytes() -> None:
    accession = "0000844779-25-000123"
    timestamp = "2025-02-03T04:05:06Z"
    raw = f"CRSB-336|primary|{accession}|{timestamp}|-2".encode("utf-8")
    assert p.primary_signal_id(accession, timestamp, -2) == hashlib.sha256(
        raw
    ).hexdigest()


@pytest.mark.parametrize("vote_sum", [0, 5, -5, True, 1.0])
def test_primary_signal_id_rejects_impossible_vote_sum(vote_sum: Any) -> None:
    with pytest.raises(ValueError, match="vote sum"):
        p.primary_signal_id(
            "0000844779-25-000123",
            "2025-02-03T04:05:06Z",
            vote_sum,
        )


@pytest.mark.parametrize(
    "accession",
    ["844779-25-000123", "0000844779/25/000123", "0000844779-5-000123"],
)
def test_primary_signal_id_rejects_noncanonical_accession(accession: str) -> None:
    with pytest.raises(ValueError, match="accession grammar"):
        p.primary_signal_id(accession, "2025-02-03T04:05:06Z", 1)


@pytest.mark.parametrize(
    ("control", "vote_sum"),
    [
        ("daily_path_only", -1),
        ("path_pair", 2),
        ("maturity_pair", -2),
    ],
)
def test_source_control_signal_id_hashes_exact_identity(
    control: str, vote_sum: int
) -> None:
    accession = "0000844779-25-000123"
    timestamp = "2025-02-03T04:05:06Z"
    raw = (
        f"CRSB-336|control|{control}|{accession}|{timestamp}|{vote_sum}"
    ).encode("utf-8")
    assert p.source_control_signal_id(
        control, accession, timestamp, vote_sum
    ) == hashlib.sha256(raw).hexdigest()


def test_source_control_signal_id_rejects_zero_or_wrong_vote_domain() -> None:
    accession = "0000844779-25-000123"
    timestamp = "2025-02-03T04:05:06Z"
    invalid_values: tuple[tuple[str, Any], ...] = (
        ("daily_path_only", 0),
        ("daily_path_only", 2),
        ("daily_path_only", True),
        ("path_pair", 1.0),
        ("unknown", 1),
    )
    for control, vote_sum in invalid_values:
        with pytest.raises(ValueError, match="source-control vote"):
            p.source_control_signal_id(
                control,
                accession,
                timestamp,
                cast(Any, vote_sum),
            )


def test_deterministic_random_side_uses_first_raw_digest_byte() -> None:
    primary_ids = ("0" * 64, "0" * 63 + "1")
    observed = set()
    for primary_id in primary_ids:
        digest = hashlib.sha256(
            f"CRSB-336|{primary_id}|RANDOM_SIDE".encode("utf-8")
        ).digest()
        expected = 1 if digest[0] < 128 else -1
        observed.add(expected)
        assert p.deterministic_random_side(primary_id) == expected
    assert observed == {-1, 1}


def test_same_parent_control_signal_id_hashes_exact_interval_and_side() -> None:
    primary_id = "a" * 64
    entry = "2025-02-03T04:10:06Z"
    exit_ = "2025-02-17T04:10:06Z"
    raw = (
        f"CRSB-336|control|constant_short|{primary_id}|{entry}|{exit_}|SHORT"
    ).encode("utf-8")
    assert p.same_parent_control_signal_id(
        "constant_short", primary_id, entry, exit_, 1
    ) == hashlib.sha256(raw).hexdigest()


def test_same_parent_controls_are_derived_from_the_primary() -> None:
    primary_id = "a" * 64
    entry = "2025-02-03T04:10:06Z"
    exit_ = "2025-02-17T04:10:06Z"
    delayed_entry = "2025-02-03T04:15:06Z"
    delayed_exit = "2025-02-17T04:15:06Z"
    expected = {
        "exact_direction_flip": (entry, exit_, -1),
        "deterministic_random_side": (
            entry,
            exit_,
            p.deterministic_random_side(primary_id),
        ),
        "constant_long": (entry, exit_, 1),
        "constant_short": (entry, exit_, -1),
        "one_bar_delayed_entry": (delayed_entry, delayed_exit, 1),
    }
    assert {
        control: p.same_parent_control_fields(
            control, primary_id, entry, exit_, 1
        )
        for control in p.SAME_PARENT_CONTROL_NAMES
    } == expected
    assert p.same_parent_control_fields(
        "one_bar_delayed_entry", primary_id, entry, exit_, -1
    ) == (delayed_entry, delayed_exit, -1)


@pytest.mark.parametrize(
    ("exit_time", "primary_side", "message"),
    [
        ("2025-02-03T04:10:06Z", 1, "336 hours"),
        ("2025-02-17T04:09:06Z", 1, "336 hours"),
        ("2025-02-17T04:11:06Z", 1, "336 hours"),
        ("2025-02-17T04:10:06Z", 0, "primary side"),
        ("2025-02-17T04:10:06Z", True, "primary side"),
        ("2025-02-17T04:10:06Z", 1.0, "primary side"),
    ],
)
def test_same_parent_control_rejects_noncanonical_primary(
    exit_time: str,
    primary_side: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        p.same_parent_control_signal_id(
            "exact_direction_flip",
            "a" * 64,
            "2025-02-03T04:10:06Z",
            exit_time,
            primary_side,
        )


def test_schedule_adds_five_minutes_then_exactly_336_elapsed_hours() -> None:
    assert p.schedule("2024-02-29T23:58:00Z") == (
        "2024-03-01T00:03:00Z",
        "2024-03-15T00:03:00Z",
    )


@pytest.mark.parametrize("candidate_weight", p.CANDIDATE_WEIGHTS)
def test_same_gross_weights_preserve_nine_configured_gross(
    candidate_weight: float,
) -> None:
    treatment = p.same_gross_weights(candidate_weight)
    scale = (9.0 - candidate_weight) / 9.0
    assert treatment["crsb"] == candidate_weight
    for sleeve, baseline in p.GROSS9_WEIGHTS.items():
        assert treatment[sleeve] == pytest.approx(baseline * scale)
    assert sum(treatment.values()) == pytest.approx(9.0, abs=1e-12)


def test_gross9_weights_equal_authenticated_esdi_authority() -> None:
    authority = p.load_esdi_authority()["gross9"]
    assert authority["weights"] == p.GROSS9_WEIGHTS
    assert authority["baseline_gross"] == 9.0
    assert p.authenticated_gross9_weights() == p.GROSS9_WEIGHTS


def test_gross9_weights_reject_cross_authority_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drifted = {
        "gross9": {
            "weights": {**p.GROSS9_WEIGHTS, "cand_rex_veto_7": 1.5},
            "baseline_gross": 9.0,
        }
    }
    monkeypatch.setattr(p, "load_esdi_authority", lambda: drifted)
    with pytest.raises(RuntimeError, match="differ from ESDI authority"):
        p.authenticated_gross9_weights()


@pytest.mark.parametrize("value", [0.6, True, "0.25", None])
def test_same_gross_weights_reject_nonfrozen_weight(value: Any) -> None:
    with pytest.raises(ValueError, match="frozen grid|frozen numeric"):
        p.same_gross_weights(value)


def test_same_gross_ranking_orders_raw_score_then_lower_weight_and_freezes_one() -> None:
    rows = [
        {"candidate_weight": 1.0, "minimum_improvement": 0.10, "passes": True},
        {"candidate_weight": 0.25, "minimum_improvement": 0.20, "passes": True},
        {"candidate_weight": 0.75, "minimum_improvement": 0.20, "passes": True},
        {"candidate_weight": 0.5, "minimum_improvement": 0.15, "passes": False},
    ]
    original = copy.deepcopy(rows)
    ranked = p.rank_same_gross_rows(rows)
    assert [row["candidate_weight"] for row in ranked] == [0.25, 0.75, 0.5, 1.0]
    assert [row["rank"] for row in ranked] == [1, 2, 3, 4]
    assert [row["frozen"] for row in ranked] == [True, False, False, False]
    assert rows == original


def test_same_gross_ranking_rejects_rank_one_failure_without_substitution() -> None:
    rows = [
        {"candidate_weight": 0.25, "minimum_improvement": 0.40, "passes": False},
        {"candidate_weight": 0.5, "minimum_improvement": 0.30, "passes": True},
        {"candidate_weight": 0.75, "minimum_improvement": 0.20, "passes": True},
        {"candidate_weight": 1.0, "minimum_improvement": 0.10, "passes": True},
    ]
    with pytest.raises(RuntimeError, match="rank one failed; no substitution"):
        p.rank_same_gross_rows(rows)


def test_same_gross_ranking_requires_exact_grid_and_finite_boolean_rows() -> None:
    invalid_cases = [
        [
            {"candidate_weight": weight, "minimum_improvement": 0.1, "passes": True}
            for weight in (0.25, 0.5, 0.75, 0.75)
        ],
        [
            {
                "candidate_weight": weight,
                "minimum_improvement": float("nan") if weight == 0.25 else 0.1,
                "passes": True,
            }
            for weight in p.CANDIDATE_WEIGHTS
        ],
        [
            {
                "candidate_weight": weight,
                "minimum_improvement": 0.1,
                "passes": 1 if weight == 0.25 else True,
            }
            for weight in p.CANDIDATE_WEIGHTS
        ],
        [
            {
                "candidate_weight": True if weight == 1.0 else weight,
                "minimum_improvement": 0.1,
                "passes": True,
            }
            for weight in p.CANDIDATE_WEIGHTS
        ],
        [
            {
                "candidate_weight": "0.25" if weight == 0.25 else weight,
                "minimum_improvement": 0.1,
                "passes": True,
            }
            for weight in p.CANDIDATE_WEIGHTS
        ],
        [
            {
                "candidate_weight": weight,
                "minimum_improvement": "0.1" if weight == 0.25 else 0.1,
                "passes": True,
            }
            for weight in p.CANDIDATE_WEIGHTS
        ],
        [
            {
                "candidate_weight": weight,
                "minimum_improvement": True if weight == 0.25 else 0.1,
                "passes": True,
            }
            for weight in p.CANDIDATE_WEIGHTS
        ],
    ]
    for rows in invalid_cases:
        with pytest.raises(ValueError, match="grid|ranking row|weight type|score type"):
            p.rank_same_gross_rows(rows)


def test_synthetic_repository_identity_is_structurally_valid(
    manifest_environment: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    identity, _payload = manifest_environment
    p.validate_repository_identity(identity)


@pytest.mark.parametrize(
    ("section", "path", "message"),
    [
        ("git_blobs", p.SOURCE_DECISION_PATH, "fixed Git blob"),
        ("git_blobs", p.MECHANISM_DECISION_PATH, "fixed Git blob"),
        ("sha256", p.SOURCE_DECISION_PATH, "fixed SHA-256"),
        ("sha256", p.MECHANISM_DECISION_PATH, "fixed SHA-256"),
    ],
)
def test_repository_identity_rejects_decision_authority_cross_binding(
    section: str,
    path: Path,
    message: str,
    manifest_environment: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    identity, _payload = manifest_environment
    tampered = copy.deepcopy(identity)
    tampered[section][str(path)] = (
        "0" * 40 if section == "git_blobs" else "0" * 64
    )
    tampered["protocol_seal_hash"] = p.canonical_hash(
        {
            "git_blobs": tampered["git_blobs"],
            "sha256": tampered["sha256"],
        }
    )
    with pytest.raises(RuntimeError, match=message):
        p.validate_repository_identity(tampered)


def test_recorded_repository_revalidates_frozen_dependencies_first(
    manifest_environment: tuple[dict[str, Any], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity, _payload = manifest_environment
    monkeypatch.setattr(
        p,
        "validate_frozen_dependencies",
        lambda: (_ for _ in ()).throw(RuntimeError("frozen dependency drift")),
    )
    with pytest.raises(RuntimeError, match="frozen dependency drift"):
        p.validate_recorded_repository(identity)


def test_manifest_has_exact_schema_hash_and_closed_evidence_boundaries(
    manifest_environment: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _identity, payload = manifest_environment
    p.validate_manifest(payload)
    assert set(payload) == set(p.TOP_LEVEL_TYPES)
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    assert payload["manifest_hash"] == p.canonical_hash(core)
    assert all(payload[name] is False for name in p.EVIDENCE_BOUNDARIES)
    assert payload["producer_effects"]["production_source_urls_requested"] == 0
    assert payload["producer_effects"]["production_source_rows_opened"] == 0
    assert payload["producer_effects"]["future_protocol_files_opened_or_hashed"] == 0
    assert payload["source"]["expected_report_months"] == list(p.REPORT_MONTHS)
    assert payload["source"]["discovery_months"] == list(p.DISCOVERY_MONTHS)
    assert payload["source"]["discovery_days_each_month"] == list(range(1, 16))
    assert payload["source"]["daily_index_path_count"] == 630
    assert payload["source"]["source_id"] == "CRF-NMFP-SB"
    assert payload["source"]["warmup_report_months"] == ["2022-11", "2022-12"]
    assert payload["source"]["identity"] == p.SOURCE_IDENTITY
    assert payload["source"]["source_csv_columns"] == list(p.SOURCE_CSV_COLUMNS)
    assert payload["source"]["xml"]["retained"] == list(p.SOURCE_CSV_COLUMNS)
    assert payload["source"]["xml"]["n_mfp2"]["schema_path_kind"] == (
        "nmfp2_friday_slots"
    )
    assert payload["source"]["xml"]["n_mfp2"]["path_length"] == [4, 5]
    assert payload["source"]["xml"]["n_mfp2"]["dollar_container_text_access"] is False
    assert payload["source"]["xml"]["n_mfp3"]["schema_path_kind"] == (
        "nmfp3_dated_details"
    )
    assert payload["source"]["xml"]["n_mfp3"]["detail_count"] == [15, 31]
    assert payload["source"]["xml"]["n_mfp3"]["dollar_child_text_access"] is False
    assert (
        payload["source"]["xml"]["liquidity_path_json"]["inferred_n_mfp2_dates"]
        is False
    )
    assert payload["feature_and_signal"]["schema_bridge"] == {
        "path_kinds": ["nmfp2_friday_slots", "nmfp3_dated_details"],
        "same_path_formula_sign_weight_tie_and_side_rule": True,
        "path_length_or_label_cannot_change_vote_rule": True,
        "first_n_mfp3_maturity_compares_to_prior_n_mfp2": True,
        "transition_reset": False,
        "form_family_is_not_signal_weight_or_regime": True,
    }
    assert payload["support_gates"]["schema_family_support"] == {
        "partition": "schema_path_kind over vote_diversity_population",
        "required_path_kinds": ["nmfp2_friday_slots", "nmfp3_dated_details"],
        "each_candidate_reports_min": 8,
        "each_accepted_full_signals_min": 4,
        "each_accepted_side_min": 1,
        "each_component_positive_min": 1,
        "each_component_negative_min": 1,
        "publish_separate_counts": [
            "candidate_reports",
            "accepted_signals",
            "sides",
            "component_signs",
        ],
        "family_specific_rule_estimation": False,
        "transition_is_not_signal_weight_regime_or_segment": True,
    }
    same_parent = payload["controls"]["same_parent"]
    assert same_parent["fields_are_derived_not_caller_supplied"] is True
    assert same_parent["accepted_primary_interval_hours"] == 336
    assert same_parent["exact_direction_flip_side"] == "opposite primary"
    assert same_parent["one_bar_delayed_entry"] == (
        "primary side; entry and exit each primary endpoint +5 minutes"
    )
    gzip_contract = payload["frozen_preregistration"]["downstream_artifacts"][
        "gzip_csv"
    ]
    assert gzip_contract["header_hex"] == "1f8b08000000000002ff"
    assert gzip_contract["required_zlib_runtime"] == "1.3"
    assert (
        gzip_contract["empty_golden_hex"]
        == "1f8b08000000000002ff03000000000000000000"
    )
    frozen = payload["frozen_preregistration"]
    assert frozen["source_decision"] == {
        "path": str(p.SOURCE_DECISION_PATH),
        "sha256": p.SOURCE_DECISION_SHA256,
        "commit": p.SOURCE_DECISION_COMMIT,
        "git_blob": p.SOURCE_DECISION_GIT_BLOB,
    }
    assert frozen["mechanism_decision"] == {
        "path": str(p.MECHANISM_DECISION_PATH),
        "sha256": p.MECHANISM_DECISION_SHA256,
        "commit": p.MECHANISM_DECISION_COMMIT,
        "git_blob": p.MECHANISM_DECISION_GIT_BLOB,
    }


def test_manifest_independently_freezes_future_stage_and_veto_contract(
    manifest_environment: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _identity, payload = manifest_environment
    assert payload["economic_contract"]["stage_order"] == [
        "2023H2",
        "2024",
        "selection",
        "same_gross",
        "future25",
        "future26",
        "combined_future",
        "stitched_full",
    ]
    gross9 = payload["gross9"]
    assert gross9["candidate_weights"] == [0.25, 0.5, 0.75, 1.0]
    assert gross9["baseline_weights"] == p.GROSS9_WEIGHTS
    assert gross9["baseline_configured_gross"] == 9.0
    assert gross9["treatment_configured_gross"] == 9.0
    assert gross9["future_subperiod_gates"] == {
        "periods": ["future25", "future26"],
        "each_cost_ratio_improvement_min": 0.05,
        "each_cost_return_retention_min": 0.97,
        "each_cost_treatment_return_positive": True,
        "each_cost_liquidation_safe": True,
        "mdd_strict_reduction_in_at_least_one_cost_per_period": True,
    }
    assert gross9["combined_future_gates"] == {
        "fresh_nonstitched_evaluation": True,
        "each_cost_ratio_improvement_min": 0.05,
        "each_cost_return_retention_min": 0.97,
        "each_cost_treatment_return_positive": True,
        "each_cost_liquidation_safe": True,
        "mdd_strict_reduction_in_at_least_one_cost": True,
        "candidate_completed_trades_min": 10,
        "candidate_active_utc_entry_months_min": 10,
        "candidate_signflip_p_value_max": 0.20,
    }
    assert gross9["future_rerank_or_alternate_weight"] is False
    assert gross9["stitched_full_is_confirmation_not_selection"] is True


@pytest.mark.parametrize("boundary", p.EVIDENCE_BOUNDARIES)
def test_manifest_rejects_each_opened_evidence_boundary(
    boundary: str,
    manifest_environment: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _identity, payload = manifest_environment
    tampered = copy.deepcopy(payload)
    tampered[boundary] = True
    with pytest.raises(RuntimeError, match="differs from frozen code|evidence boundary"):
        p.validate_manifest(tampered)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("source"),
        lambda payload: payload.update({"unexpected": False}),
        lambda payload: payload.update({"singleton": 1}),
        lambda payload: payload.update({"strict_sequence": "not-an-array"}),
    ],
)
def test_manifest_rejects_top_level_schema_drift(
    mutation: Any,
    manifest_environment: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _identity, payload = manifest_environment
    tampered = copy.deepcopy(payload)
    mutation(tampered)
    with pytest.raises(RuntimeError, match="top-level schema drift"):
        p.validate_manifest(tampered)


def test_manifest_build_returns_fresh_authority_and_identity_structures(
    manifest_environment: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    identity, first = manifest_environment
    first["gross9"]["authority"]["synthetic_mutation"] = True
    first["frozen_preregistration"]["repository_identity"]["branch"] = "mutated"
    second = p.build_manifest(identity)
    assert "synthetic_mutation" not in second["gross9"]["authority"]
    assert second["frozen_preregistration"]["repository_identity"]["branch"] == (
        p.EXPECTED_BRANCH
    )


def test_canonical_manifest_bytes_are_sorted_ascii_indented_and_single_lf(
    manifest_environment: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _identity, payload = manifest_environment
    raw = p.canonical_manifest_bytes(payload)
    assert raw == _canonical_json_bytes(payload)
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert json.loads(raw.decode("utf-8")) == payload


def test_canonical_manifest_bytes_reject_tampered_payload(
    manifest_environment: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    _identity, payload = manifest_environment
    tampered = copy.deepcopy(payload)
    tampered["status"] = "opened"
    with pytest.raises(RuntimeError, match="differs from frozen code"):
        p.canonical_manifest_bytes(tampered)


@pytest.mark.parametrize(
    "unsafe",
    [
        "/tmp/crlc.json",
        "../results/crlc.json",
        "results/../crlc.json",
        "~/crlc.json",
        "results/not-the-frozen-singleton.json",
    ],
)
def test_output_path_rejects_unsafe_or_noncanonical_destination(unsafe: str) -> None:
    with pytest.raises(RuntimeError, match="unsafe|frozen singleton"):
        p._output_path(unsafe)


def test_output_path_accepts_only_frozen_singleton_destination() -> None:
    assert p._output_path(p.DEFAULT_OUTPUT) == p.DEFAULT_OUTPUT


def test_branch_status_requires_exact_clean_pushed_shape() -> None:
    head = "a" * 40
    clean = b"\n".join(
        [
            f"# branch.oid {head}".encode(),
            f"# branch.head {p.EXPECTED_BRANCH}".encode(),
            f"# branch.upstream origin/{p.EXPECTED_BRANCH}".encode(),
            b"# branch.ab +0 -0",
        ]
    )
    assert p._branch_status_is_clean(clean, head)
    temporary = Path("results/.crlc.tmp")
    assert p._branch_status_is_clean(
        clean + b"\n" + f"? {temporary}".encode(),
        head,
        temporary,
    )
    assert not p._branch_status_is_clean(clean + b"\n1 .M N... tracked", head)
    assert not p._branch_status_is_clean(
        clean.replace(b"# branch.ab +0 -0", b"# branch.ab +1 -0"),
        head,
    )


def test_frozen_repository_identity_rejects_unpushed_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    head = "a" * 40
    tree = "b" * 40
    upstream = "c" * 40
    monkeypatch.setattr(
        p,
        "_git",
        lambda *arguments: (
            f"{p.REPOSITORY_ROOT}\n{head}\n{tree}\n{upstream}\n".encode()
            if arguments[0] == "rev-parse"
            else pytest.fail(f"unexpected Git call: {arguments}")
        ),
    )
    with pytest.raises(RuntimeError, match="requires pushed exact HEAD"):
        p.frozen_repository_identity()


def test_frozen_repository_identity_rejects_dirty_worktree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(p, "validate_frozen_dependencies", lambda: None)
    head = "a" * 40
    tree = "b" * 40
    clean = b"\n".join(
        [
            f"# branch.oid {head}".encode(),
            f"# branch.head {p.EXPECTED_BRANCH}".encode(),
            f"# branch.upstream origin/{p.EXPECTED_BRANCH}".encode(),
            b"# branch.ab +0 -0",
        ]
    )

    def fake_git(*arguments: str) -> bytes:
        if arguments[0] == "rev-parse":
            return f"{p.REPOSITORY_ROOT}\n{head}\n{tree}\n{head}\n".encode()
        if arguments[0] == "status":
            return clean + b"\n1 .M N... 100644 100644 100644 abc abc tracked.py\n"
        return pytest.fail(f"unexpected Git call: {arguments}")

    monkeypatch.setattr(p, "_git", fake_git)
    with pytest.raises(RuntimeError, match="clean pushed branch"):
        p.frozen_repository_identity()


def test_write_once_creates_read_only_canonical_singleton(
    write_once_environment: tuple[dict[str, Any], dict[str, Any], Path],
) -> None:
    _identity, payload, path = write_once_environment
    status, stored = p.write_once()
    assert status == "created"
    assert stored == payload
    assert path.read_bytes() == p.canonical_manifest_bytes(payload)
    assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_write_once_verifies_exact_existing_artifact(
    write_once_environment: tuple[dict[str, Any], dict[str, Any], Path],
) -> None:
    _identity, payload, path = write_once_environment
    path.write_bytes(p.canonical_manifest_bytes(payload))
    status, stored = p.write_once(payload=payload)
    assert status == "verified_existing"
    assert stored == payload


def test_write_once_rejects_existing_payload_mismatch(
    write_once_environment: tuple[dict[str, Any], dict[str, Any], Path],
) -> None:
    _identity, payload, path = write_once_environment
    path.write_bytes(p.canonical_manifest_bytes(payload))
    supplied = copy.deepcopy(payload)
    supplied["status"] = "drifted"
    with pytest.raises(RuntimeError, match="supplied preregistration payload drift"):
        p.write_once(payload=supplied)


def test_write_once_rejects_malformed_existing_artifact(
    write_once_environment: tuple[dict[str, Any], dict[str, Any], Path],
) -> None:
    _identity, _payload, path = write_once_environment
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="top-level schema drift"):
        p.write_once()


def test_write_once_rejects_existing_symlink(
    write_once_environment: tuple[dict[str, Any], dict[str, Any], Path],
) -> None:
    _identity, payload, path = write_once_environment
    target = path.parent / "elsewhere.json"
    target.write_bytes(p.canonical_manifest_bytes(payload))
    path.symlink_to(target)
    with pytest.raises(OSError):
        p.write_once()


def test_write_once_rejects_conflicting_creation_race(
    write_once_environment: tuple[dict[str, Any], dict[str, Any], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _identity, _payload, path = write_once_environment

    def conflicting_link(
        _source: str | bytes | Path,
        destination: str | bytes | Path,
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        Path(os.fsdecode(destination)).write_bytes(b"race\n")
        raise FileExistsError

    monkeypatch.setattr(p.os, "link", conflicting_link)
    with pytest.raises(RuntimeError, match="race drift"):
        p.write_once()
    assert path.read_bytes() == b"race\n"
