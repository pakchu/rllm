from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import preregister_bitcoin_core_immutable_merge_surface as bcims


def _rehash(payload: dict[str, object]) -> None:
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = bcims.canonical_hash(core)


def test_manifest_is_source_only_and_historical_incidence_blind() -> None:
    payload = bcims.build_manifest()
    bcims.validate_manifest(payload)
    assert payload["source_id"] == "BCIMS"
    assert payload["outcomes_opened"] is False
    assert payload["market_clocks_opened"] is False
    assert payload["historical_source_incidence_opened"] is False
    assert payload["semantic_model_opened"] is False
    assert payload["source_only_probe_opened"] is True
    assert payload["source_only_probe"]["excluded_from_source_support"] is True
    assert payload["source_only_probe"]["market_or_outcomes_opened"] is False


def test_probe_disclosure_matches_sealed_out_of_window_observations() -> None:
    probe = bcims.build_manifest()["source_only_probe"]
    assert probe["sealed_tip"] == bcims.PROBE_SEALED_TIP
    assert probe["first_parent_commits"] == 3016
    assert probe["two_parent_merges"] == 3016
    assert probe["bitcoin_subjects"] == 2950
    assert probe["gui_subjects"] == 66
    assert probe["other_two_parent_subjects"] == 0
    assert probe["descending_committer_time_violations"] == 0


@pytest.mark.parametrize(
    ("subject", "repository", "stratum", "pr_number", "title"),
    [
        (
            "Merge bitcoin/bitcoin#35709: depends: Update Qt to 6.8.4",
            "bitcoin/bitcoin",
            "primary_core",
            35709,
            "depends: Update Qt to 6.8.4",
        ),
        (
            "Merge bitcoin-core/gui#949: Fix compiler warning",
            "bitcoin-core/gui",
            "gui_comparator",
            949,
            "Fix compiler warning",
        ),
    ],
)
def test_exact_merge_subject_routing(
    subject: str,
    repository: str,
    stratum: str,
    pr_number: int,
    title: str,
) -> None:
    parsed = bcims.parse_merge_subject(subject)
    assert parsed == {
        "repository": repository,
        "pr_number": pr_number,
        "title": title,
        "stratum": stratum,
    }


@pytest.mark.parametrize(
    "subject",
    [
        "Merge bitcoin/bitcoin#0: invalid zero",
        "Merge Bitcoin/bitcoin#10: wrong case",
        "Merge bitcoin/bitcoin #10: inserted space",
        "Merge bitcoin/bitcoin#10:",
        "Merge bitcoin/bitcoin#10:  leading title space",
        "Merge third-party/repo#10: unrelated",
        "direct commit",
    ],
)
def test_nonexact_subjects_are_audit_only(subject: str) -> None:
    assert bcims.parse_merge_subject(subject) is None


@pytest.mark.parametrize(
    "subject",
    [
        "Merge bitcoin/bitcoin#10: title\nbody",
        "Merge bitcoin/bitcoin#10: title\x00suffix",
    ],
)
def test_control_bearing_subjects_fail_closed(subject: str) -> None:
    with pytest.raises(ValueError, match="control"):
        bcims.parse_merge_subject(subject)


def test_causal_availability_is_utc_monotone_and_delayed_two_days() -> None:
    floors = bcims.causal_availability_floors(
        [
            "2020-01-03T23:30:00-08:00",
            "2020-01-04T01:00:00+00:00",
            "2020-01-06T18:00:00+09:00",
        ]
    )
    assert floors == [
        "2020-01-06T12:00:00Z",
        "2020-01-06T12:00:00Z",
        "2020-01-08T12:00:00Z",
    ]


def test_causal_availability_rejects_naive_or_malformed_times() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        bcims.causal_availability_floors(["2020-01-01T00:00:00"])
    with pytest.raises(ValueError, match="malformed"):
        bcims.causal_availability_floors(["not-a-time"])


def test_causal_availability_pins_a_descending_utc_day_to_running_max() -> None:
    assert bcims.causal_availability_floors(
        ["2020-01-05T01:00:00+00:00", "2020-01-04T23:00:00+00:00"]
    ) == ["2020-01-07T12:00:00Z", "2020-01-07T12:00:00Z"]


@pytest.mark.parametrize(
    ("path", "surface"),
    [
        ("src/net.cpp", "src"),
        ("test/functional/p2p.py", "test"),
        ("CMakeLists.txt", "__root__"),
        ("doc/release-notes.md", "doc"),
    ],
)
def test_path_surface_is_exact_and_root_aware(path: str, surface: str) -> None:
    assert bcims.path_surface(path) == surface


@pytest.mark.parametrize(
    "path",
    ["", "/src/net.cpp", "src/", "src//net.cpp", "src/../net.cpp", "src\\net.cpp"],
)
def test_path_surface_rejects_unsafe_shapes(path: str) -> None:
    with pytest.raises(ValueError):
        bcims.path_surface(path)


def test_path_surface_rejects_controls_and_nonencodable_unicode() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        bcims.path_surface("src/\x7fname")
    with pytest.raises(UnicodeEncodeError):
        bcims.path_surface("src/\udcff")


def test_source_contract_excludes_mutable_metadata_and_blobs() -> None:
    authority = bcims.build_manifest()["source_contract"]["authority"]
    assert authority["remote"] == "https://github.com/bitcoin/bitcoin.git"
    assert authority["branch"] == "master"
    assert authority["sealed_tip"] == bcims.PROBE_SEALED_TIP
    assert authority["mutable_github_pr_metadata_used"] is False
    assert authority["blob_contents_opened_in_source_support"] is False
    forbidden = bcims.build_manifest()["source_contract"]["forbidden_fields"]
    assert "current PR title/body/labels/milestone/state" in forbidden
    assert "market bars/returns/funding/PnL" in forbidden


def test_quality_gates_are_fixed_before_incidence() -> None:
    gates = bcims.build_manifest()["source_quality_gates"]
    assert gates["integrity"]["unknown_first_parent_fraction_max"] == 0.05
    assert gates["integrity"]["quarantine_or_imputation_allowed"] is False
    assert gates["primary_core"] == {
        "minimum_events": 2400,
        "minimum_events_each_year": 500,
        "minimum_events_each_quarter": 100,
        "minimum_unique_availability_days_each_year": 180,
        "maximum_calendar_month_share": 0.12,
        "minimum_distinct_top_level_surfaces_each_year": 6,
        "maximum_fractional_top_level_surface_share": 0.70,
    }
    assert gates["gui_comparator"]["minimum_events_each_year"] == 5
    assert gates["failure_effect"] == "REJECT_NO_REPAIR"


def test_later_llm_boundary_preserves_source_membership() -> None:
    boundary = bcims.build_manifest()["later_semantic_boundary"]
    assert boundary["authorized_now"] is False
    assert boundary["requirements"][0] == "single local LLM"
    assert "LLM event creation/deletion/retiming" in boundary["forbidden"]
    assert "analyzer/trader two-model split" in boundary["forbidden"]


def test_manifest_hash_detects_mutation() -> None:
    payload = bcims.build_manifest()
    payload["source_quality_gates"]["primary_core"]["minimum_events"] = 1
    with pytest.raises(RuntimeError, match="hash mismatch"):
        bcims.validate_manifest(payload)


def test_recomputed_hash_cannot_change_frozen_contract() -> None:
    payload = bcims.build_manifest()
    payload["availability_contract"]["historical"] = "same-day"
    _rehash(payload)
    with pytest.raises(RuntimeError, match="differs from code"):
        bcims.validate_manifest(payload)


def test_builds_do_not_share_mutable_state() -> None:
    first = bcims.build_manifest()
    first["source_contract"]["authority"]["branch"] = "MUTATED"
    second = bcims.build_manifest()
    assert second["source_contract"]["authority"]["branch"] == "master"


def test_repository_bindings_match_bytes() -> None:
    payload = bcims.build_manifest()
    assert (
        bcims.sha256_file(payload["decision_binding"]["path"])
        == payload["decision_binding"]["sha256"]
    )
    assert (
        bcims.sha256_file(payload["implementation_binding"]["path"])
        == payload["implementation_binding"]["sha256"]
    )


def test_write_once_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "bcims.json"
    payload = bcims.build_manifest()
    assert bcims.write_manifest_once(path, payload) == "created"
    assert bcims.write_manifest_once(path, bcims.build_manifest()) == (
        "verified_existing"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_write_once_rejects_mutated_first_write(tmp_path: Path) -> None:
    path = tmp_path / "bcims-mutated.json"
    payload = bcims.build_manifest()
    payload["historical_source_incidence_opened"] = True
    _rehash(payload)
    with pytest.raises(RuntimeError, match="must keep .*false"):
        bcims.write_manifest_once(path, payload)
    assert not path.exists()


def test_repository_artifact_matches_code() -> None:
    artifact = json.loads((bcims.REPO_ROOT / bcims.DEFAULT_OUTPUT).read_text())
    bcims.validate_manifest(artifact)
    assert artifact == bcims.build_manifest()
