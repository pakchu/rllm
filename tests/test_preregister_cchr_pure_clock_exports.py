from __future__ import annotations

import ast
from pathlib import Path

import pytest

from training import cchr_comparator_clock_common as common
from training import preregister_cchr_pure_clock_exports as prereg


def _expected_external_hashes(family: str) -> dict[str, str]:
    expected = {
        str(prereg.MASTER_PREREGISTRATION): prereg.MASTER_PREREGISTRATION_SHA256,
        str(prereg.MASTER_SOURCE): prereg.MASTER_SOURCE_SHA256,
    }
    for binding in prereg._raw_input_bindings(family).values():
        expected[str(binding["path"])] = str(binding["sha256"])
    for binding in prereg._configuration_bindings(family).values():
        expected[str(binding["path"])] = str(binding["sha256"])
    return expected


def _install_hash_stub(monkeypatch: pytest.MonkeyPatch, *families: str) -> None:
    real_hash = prereg.sha256_file
    expected: dict[str, str] = {}
    for family in families:
        expected.update(_expected_external_hashes(family))

    def hash_file(path: str | Path) -> str:
        key = str(path)
        if key in expected:
            return expected[key]
        return real_hash(path)

    monkeypatch.setattr(prereg, "sha256_file", hash_file)


@pytest.mark.parametrize(
    ("family", "member_count"),
    (("pdlh", 16), ("dtv", 24), ("far", 12), ("live", 3)),
)
def test_family_preregistration_binds_exact_source_only_contract(
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    member_count: int,
) -> None:
    _install_hash_stub(monkeypatch, family)
    payload = prereg.build_family_manifest(family)
    prereg.validate_manifest(payload, verify_files=False)

    assert payload["family"] == family
    assert payload["member_count"] == member_count
    assert payload["candidate_map_sha256"] == common.candidate_map_hash(
        payload["candidate_map"]
    )
    assert payload["outcomes_opened"] is False
    assert payload["outcome_boundary"] == prereg.OUTCOME_BOUNDARY
    assert payload["authorization"] == {
        "source_only_clock_export_after_this_artifact": True,
        "outcome_evaluator": False,
        "post_2023_source_access": False,
        "network_access": False,
    }
    assert payload["clock_contract"]["schema"] == list(common.CLOCK_COLUMNS)
    assert payload["clock_contract"]["source_end_exclusive"] == ("2024-01-01T00:00:00Z")
    assert payload["implementation_bindings"]["runner"] == {
        "path": str(prereg.RUNNER_SOURCE),
        "sha256": prereg.sha256_file(prereg.RUNNER_SOURCE),
    }
    assert payload["output_contract"]["export_manifest_contract"] == {
        "protocol_version": prereg.EXPORT_MANIFEST_PROTOCOL_VERSION,
        "required_top_level_keys": list(prereg.EXPORT_MANIFEST_TOP_LEVEL_KEYS),
        "manifest_hash": "sha256 canonical JSON excluding manifest_hash",
    }
    for binding in payload["raw_input_bindings"].values():
        assert binding["path"]
        assert len(binding["sha256"]) == 64
        assert binding["columns"]


def test_write_is_atomic_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_hash_stub(monkeypatch, "pdlh")
    outputs = dict(prereg.PREREGISTRATION_OUTPUTS)
    outputs["pdlh"] = tmp_path / "pdlh.json"
    monkeypatch.setattr(prereg, "PREREGISTRATION_OUTPUTS", outputs)
    monkeypatch.setattr(prereg, "RESULTS_ROOT", tmp_path)

    payload = prereg.write_preregistrations(("pdlh",))["pdlh"]
    assert outputs["pdlh"].is_file()
    prereg.validate_manifest(payload, verify_files=False)
    with pytest.raises(FileExistsError, match="immutable"):
        prereg.write_preregistrations(("pdlh",))


def test_create_only_publish_does_not_overwrite_a_racing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_hash_stub(monkeypatch, "pdlh")
    outputs = dict(prereg.PREREGISTRATION_OUTPUTS)
    target = tmp_path / "pdlh.json"
    outputs["pdlh"] = target
    monkeypatch.setattr(prereg, "PREREGISTRATION_OUTPUTS", outputs)
    monkeypatch.setattr(prereg, "RESULTS_ROOT", tmp_path)
    write_temporary = prereg._write_temporary_json

    def race(payload: dict[str, object], output: Path) -> Path:
        temporary = write_temporary(payload, output)
        output.write_text("racing writer\n", encoding="utf-8")
        return temporary

    monkeypatch.setattr(prereg, "_write_temporary_json", race)
    with pytest.raises(FileExistsError, match="immutable"):
        prereg.write_preregistrations(("pdlh",))
    assert target.read_text(encoding="utf-8") == "racing writer\n"


def test_multi_family_publish_rolls_back_owned_artifacts_on_late_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_hash_stub(monkeypatch, "pdlh", "dtv")
    outputs = dict(prereg.PREREGISTRATION_OUTPUTS)
    outputs["pdlh"] = tmp_path / "pdlh.json"
    outputs["dtv"] = tmp_path / "dtv.json"
    monkeypatch.setattr(prereg, "PREREGISTRATION_OUTPUTS", outputs)
    monkeypatch.setattr(prereg, "RESULTS_ROOT", tmp_path)
    write_temporary = prereg._write_temporary_json

    def race(payload: dict[str, object], output: Path) -> Path:
        temporary = write_temporary(payload, output)
        if output == outputs["dtv"]:
            output.write_text("racing writer\n", encoding="utf-8")
        return temporary

    monkeypatch.setattr(prereg, "_write_temporary_json", race)
    with pytest.raises(FileExistsError, match="immutable"):
        prereg.write_preregistrations(("pdlh", "dtv"))
    assert not outputs["pdlh"].exists()
    assert outputs["dtv"].read_text(encoding="utf-8") == "racing writer\n"


def test_output_must_be_json_under_results_and_not_alias_an_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_hash_stub(monkeypatch, "pdlh")
    outputs = dict(prereg.PREREGISTRATION_OUTPUTS)
    monkeypatch.setattr(prereg, "RESULTS_ROOT", tmp_path / "results")

    outputs["pdlh"] = tmp_path / "outside.json"
    monkeypatch.setattr(prereg, "PREREGISTRATION_OUTPUTS", outputs)
    with pytest.raises(ValueError, match="under results"):
        prereg.write_preregistrations(("pdlh",))

    outputs["pdlh"] = prereg.MASTER_PREREGISTRATION
    monkeypatch.setattr(prereg, "PREREGISTRATION_OUTPUTS", outputs)
    monkeypatch.setattr(prereg, "RESULTS_ROOT", prereg.REPOSITORY_ROOT / "results")
    with pytest.raises(ValueError, match="protected input"):
        prereg.write_preregistrations(("pdlh",))


def test_manifest_build_cannot_reach_transitive_source_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_hash_stub(monkeypatch, *prereg.FAMILIES)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("source row loader was reached during preregistration")

    monkeypatch.setattr(prereg.clock_common, "read_hash_bound_columns", forbidden)
    monkeypatch.setattr(prereg.clock_common, "read_hash_bound_prefix", forbidden)
    monkeypatch.setattr(prereg.pdlh, "load_causal_inputs", forbidden)
    monkeypatch.setattr(prereg.dtv, "load_pre2024", forbidden)
    monkeypatch.setattr(prereg.far, "load_hash_bound_pre2024", forbidden)
    monkeypatch.setattr(prereg.live, "load_causal_inputs", forbidden)

    for family in prereg.FAMILIES:
        payload = prereg.build_family_manifest(family)
        assert payload["outcome_boundary"]["source_csv_values_read"] == 0


@pytest.mark.parametrize("family", prereg.FAMILIES)
def test_frozen_preregistration_artifact_validates_without_source_rows(
    family: str,
) -> None:
    payload = prereg.load_preregistration(family, verify_files=False)
    assert payload["family"] == family
    assert payload["outcomes_opened"] is False


def test_preregistration_module_has_no_row_parser_or_execution_dependency() -> None:
    source = Path(prereg.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert "pandas" not in imports
    assert "subprocess" not in imports
    assert "requests" not in imports
    for forbidden_call in ("read_csv", "read_parquet", "read_pickle"):
        assert forbidden_call not in source
