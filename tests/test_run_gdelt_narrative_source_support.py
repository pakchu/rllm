from __future__ import annotations

import builtins
import io
from pathlib import Path

import pytest

from training import run_gdelt_narrative_source_support as launcher


def test_launcher_frozen_inputs_match_committed_hashes() -> None:
    assert launcher.sha256_file(launcher.SOURCE_ACCESS_SEAL) == (
        launcher.SOURCE_ACCESS_SEAL_SHA256
    )
    assert launcher.sha256_file(launcher.EVALUATOR_SOURCE) == (
        launcher.EVALUATOR_SOURCE_SHA256
    )
    assert launcher.sha256_file(launcher.PROTOCOL_DOCUMENT) == (
        launcher.PROTOCOL_DOCUMENT_SHA256
    )


@pytest.mark.parametrize(
    "tampered_path",
    (
        launcher.SOURCE_ACCESS_SEAL,
        launcher.EVALUATOR_SOURCE,
        launcher.PROTOCOL_DOCUMENT,
    ),
)
def test_launcher_rejects_tampering_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tampered_path: Path
) -> None:
    for path in (
        launcher.SOURCE_ACCESS_SEAL,
        launcher.EVALUATOR_SOURCE,
        launcher.PROTOCOL_DOCUMENT,
    ):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(launcher.repository_path(path).read_bytes())
    (tmp_path / tampered_path).write_bytes(b"tampered\n")
    monkeypatch.setattr(launcher, "REPOSITORY_ROOT", tmp_path)
    imported = False

    def forbidden_import(name: str):
        nonlocal imported
        imported = True
        raise AssertionError(f"import attempted after failed seal check: {name}")

    monkeypatch.setattr(launcher.importlib, "import_module", forbidden_import)
    with pytest.raises(RuntimeError, match="before import"):
        launcher.load_evaluator()
    assert imported is False


def test_launcher_run_is_write_once_and_reads_no_market_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator = launcher.load_evaluator()
    output = tmp_path / "source-support.json"
    allowed = {
        launcher.repository_path(path).resolve()
        for path in (
            launcher.SOURCE_ACCESS_SEAL,
            launcher.EVALUATOR_SOURCE,
            launcher.PROTOCOL_DOCUMENT,
            evaluator.PREREGISTRATION,
            evaluator.TRANSPORT_AMENDMENT,
            evaluator.SOURCE_MANIFEST,
            evaluator.DAILY_SOURCE,
            evaluator.RAW_SOURCE,
        )
    }
    allowed.add(output.resolve())
    opened: set[Path] = set()
    original_path_open = Path.open
    original_builtin_open = builtins.open
    original_io_open = io.open

    def guard(file) -> None:
        resolved = Path(file).resolve()
        if resolved not in allowed:
            raise AssertionError(f"unexpected GNRC launcher file access: {resolved}")
        opened.add(resolved)

    def recording_path_open(path: Path, *args, **kwargs):
        guard(path)
        return original_path_open(path, *args, **kwargs)

    def recording_builtin_open(file, *args, **kwargs):
        guard(file)
        return original_builtin_open(file, *args, **kwargs)

    def recording_io_open(file, *args, **kwargs):
        guard(file)
        return original_io_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_path_open)
    monkeypatch.setattr(builtins, "open", recording_builtin_open)
    monkeypatch.setattr(io, "open", recording_io_open)
    report = launcher.run(output)
    assert report["outcome_boundary"]["outcomes_opened"] is False
    with pytest.raises(FileExistsError, match="write-once"):
        launcher.run(output)
    assert opened <= allowed
