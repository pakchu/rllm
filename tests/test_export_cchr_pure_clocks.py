from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from training import cchr_comparator_clock_common as common
from training import export_cchr_pure_clocks as runner
from training import preregister_cchr_pure_clock_exports as prereg


def _clock(candidate_ids: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(candidate_ids):
        entry = cast(
            pd.Timestamp,
            pd.Timestamp("2023-02-01T00:05:00Z") + pd.Timedelta(minutes=10 * index),
        )
        exit_time = cast(pd.Timestamp, entry + pd.Timedelta(minutes=5))
        rows.append(
            {
                "candidate_id": candidate_id,
                "split": "selection",
                "decision_time": common.format_utc(entry),
                "entry_time": common.format_utc(entry),
                "exit_time": common.format_utc(exit_time),
                "side": 1 if index % 2 == 0 else -1,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_preregistration(
    tmp_path: Path, candidate_ids: tuple[str, ...]
) -> dict[str, Any]:
    runner_hash = common.sha256_file(prereg.RUNNER_SOURCE)
    payload: dict[str, Any] = {
        "family": "pdlh",
        "manifest_hash": "a" * 64,
        "implementation_bindings": {
            "runner": {"path": str(prereg.RUNNER_SOURCE), "sha256": runner_hash}
        },
        "raw_input_bindings": {},
        "configuration_bindings": {},
        "candidate_map": {candidate_id: {} for candidate_id in candidate_ids},
        "candidate_map_sha256": common.candidate_map_hash(
            {candidate_id: {} for candidate_id in candidate_ids}
        ),
        "member_count": len(candidate_ids),
        "output_contract": {
            "pure_clock": str(tmp_path / "clock.csv.gz"),
            "export_manifest": str(tmp_path / "manifest.json"),
        },
    }
    return payload


def test_preflight_validates_preregistration_before_any_source_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    candidate_ids = ("synthetic:a",)
    payload = _synthetic_preregistration(tmp_path, candidate_ids)
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text("{}\n", encoding="utf-8")
    outputs = dict(prereg.PREREGISTRATION_OUTPUTS)
    outputs["pdlh"] = prereg_path
    monkeypatch.setattr(prereg, "PREREGISTRATION_OUTPUTS", outputs)
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)

    def load_preregistration(
        family: str, *, verify_files: bool
    ) -> tuple[dict[str, Any], str]:
        events.append(f"prereg:{family}:{verify_files}")
        return payload, common.sha256_file(prereg_path)

    def load_source() -> tuple[pd.DataFrame, pd.DataFrame]:
        events.append("source")
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(
        prereg,
        "load_preregistration_with_sha256",
        load_preregistration,
    )
    monkeypatch.setattr(runner.pdlh, "load_causal_inputs", load_source)
    monkeypatch.setattr(
        runner.pdlh,
        "build_pdlh_clock",
        lambda *_args: _clock(candidate_ids),
    )

    [plan] = runner.preflight_exports(("pdlh",))
    assert events == ["prereg:pdlh:True"]
    runner._execute_plan(plan)
    assert events == ["prereg:pdlh:True", "source"]


def test_invalid_preregistration_blocks_source_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise RuntimeError("invalid preregistration")

    def forbidden() -> None:
        raise AssertionError("source loader ran before preregistration validation")

    monkeypatch.setattr(prereg, "load_preregistration_with_sha256", reject)
    monkeypatch.setattr(runner.pdlh, "load_causal_inputs", forbidden)
    with pytest.raises(RuntimeError, match="invalid preregistration"):
        runner.preflight_exports(("pdlh",))


def test_preregistration_mutation_after_preflight_blocks_source_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _synthetic_preregistration(tmp_path, ("synthetic:a",))
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text("{}\n", encoding="utf-8")
    outputs = dict(prereg.PREREGISTRATION_OUTPUTS)
    outputs["pdlh"] = prereg_path
    monkeypatch.setattr(prereg, "PREREGISTRATION_OUTPUTS", outputs)
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        prereg,
        "load_preregistration_with_sha256",
        lambda *_args, **_kwargs: (payload, common.sha256_file(prereg_path)),
    )
    [plan] = runner.preflight_exports(("pdlh",))
    prereg_path.write_text('{"changed":true}\n', encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "_build_family_clock",
        lambda _family: pytest.fail("source loader ran after preregistration mutation"),
    )
    with pytest.raises(RuntimeError, match="changed after preflight"):
        runner._execute_plan(plan)


def test_export_publishes_exact_clock_and_provenance_manifest_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_ids = ("synthetic:a", "synthetic:b")
    payload = _synthetic_preregistration(tmp_path, candidate_ids)
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text("{}\n", encoding="utf-8")
    outputs = dict(prereg.PREREGISTRATION_OUTPUTS)
    outputs["pdlh"] = prereg_path
    monkeypatch.setattr(prereg, "PREREGISTRATION_OUTPUTS", outputs)
    monkeypatch.setattr(runner, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(
        prereg,
        "load_preregistration_with_sha256",
        lambda *_args, **_kwargs: (payload, common.sha256_file(prereg_path)),
    )
    monkeypatch.setattr(
        runner,
        "_build_family_clock",
        lambda _family: _clock(candidate_ids),
    )

    manifest = runner.export_families(("pdlh",))["pdlh"]
    clock_path = tmp_path / "clock.csv.gz"
    manifest_path = tmp_path / "manifest.json"
    assert clock_path.is_file()
    assert manifest_path.is_file()
    assert manifest["clock"]["sha256"] == common.sha256_file(clock_path)
    assert manifest["clock"]["rows"] == 2
    assert manifest["clock"]["rows_by_candidate"] == {
        "synthetic:a": 1,
        "synthetic:b": 1,
    }
    assert manifest["outcomes_opened"] is False
    runner.validate_export_manifest(manifest, payload)
    original_clock = clock_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        runner.export_families(("pdlh",))
    assert clock_path.read_bytes() == original_clock
    assert manifest_path.read_bytes() == original_manifest


def test_pair_publish_rolls_back_owned_clock_if_manifest_target_races(
    tmp_path: Path,
) -> None:
    clock_temporary = tmp_path / "clock.tmp"
    manifest_temporary = tmp_path / "manifest.tmp"
    clock_target = tmp_path / "clock.csv.gz"
    manifest_target = tmp_path / "manifest.json"
    clock_temporary.write_bytes(b"clock")
    manifest_temporary.write_bytes(b"manifest")
    manifest_target.write_bytes(b"racing writer")

    with pytest.raises(FileExistsError, match="immutable"):
        runner._publish_pair_create_only(
            clock_temporary,
            clock_target,
            manifest_temporary,
            manifest_target,
        )
    assert not clock_target.exists()
    assert manifest_target.read_bytes() == b"racing writer"


def test_pair_publish_rolls_back_owned_clock_on_non_race_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock_temporary = tmp_path / "clock.tmp"
    manifest_temporary = tmp_path / "manifest.tmp"
    clock_target = tmp_path / "clock.csv.gz"
    manifest_target = tmp_path / "manifest.json"
    clock_temporary.write_bytes(b"clock")
    manifest_temporary.write_bytes(b"manifest")
    real_link = runner.os.link
    calls = 0

    def fail_second_link(source: str, target: str, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic publication failure")
        real_link(source, target, **kwargs)

    monkeypatch.setattr(runner.os, "link", fail_second_link)
    with pytest.raises(OSError, match="synthetic publication failure"):
        runner._publish_pair_create_only(
            clock_temporary,
            clock_target,
            manifest_temporary,
            manifest_target,
        )
    assert not clock_target.exists()
    assert not manifest_target.exists()


def test_multi_family_publish_rolls_back_every_owned_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files: list[tuple[Path, Path]] = []
    for name in ("a-clock", "a-manifest", "b-clock", "b-manifest"):
        temporary = tmp_path / f"{name}.tmp"
        target = tmp_path / f"{name}.out"
        temporary.write_bytes(name.encode())
        files.append((temporary, target))
    real_link = runner.os.link
    calls = 0

    def fail_third_link(source: str, target: str, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("synthetic second-family failure")
        real_link(source, target, **kwargs)

    monkeypatch.setattr(runner.os, "link", fail_third_link)
    with pytest.raises(OSError, match="second-family failure"):
        runner._publish_files_create_only(files)
    assert all(not target.exists() for _, target in files)


def test_output_paths_must_remain_under_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _synthetic_preregistration(tmp_path, ("synthetic:a",))
    payload["output_contract"]["pure_clock"] = str(tmp_path / "outside.csv.gz")
    results = tmp_path / "results"
    prereg_path = results / "prereg.json"
    prereg_path.parent.mkdir()
    prereg_path.write_text("{}\n", encoding="utf-8")
    outputs = dict(prereg.PREREGISTRATION_OUTPUTS)
    outputs["pdlh"] = prereg_path
    monkeypatch.setattr(prereg, "PREREGISTRATION_OUTPUTS", outputs)
    monkeypatch.setattr(runner, "RESULTS_ROOT", results)
    monkeypatch.setattr(
        prereg,
        "load_preregistration_with_sha256",
        lambda *_args, **_kwargs: (payload, common.sha256_file(prereg_path)),
    )
    with pytest.raises(ValueError, match="under results"):
        runner.preflight_exports(("pdlh",))


def test_runner_has_no_outcome_or_legacy_evaluator_imports() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert all("backtest" not in name for name in imported)
    assert all("evaluator" not in name for name in imported)
    lowered = source.lower()
    for forbidden in ("future_return", "trade_return", "strict_mdd", "cagr"):
        assert forbidden not in lowered
