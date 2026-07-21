from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from training import run_gdelt_narrative_economic_selection as launcher


def test_launcher_frozen_inputs_match_committed_hashes() -> None:
    expected = {
        launcher.PREMARKET_ACCESS_SEAL: launcher.PREMARKET_ACCESS_SEAL_SHA256,
        launcher.EVALUATOR_SOURCE: launcher.EVALUATOR_SOURCE_SHA256,
        launcher.PROTOCOL_DOCUMENT: launcher.PROTOCOL_DOCUMENT_SHA256,
        launcher.TEST_SOURCE: launcher.TEST_SOURCE_SHA256,
    }
    for path, expected_hash in expected.items():
        assert launcher.sha256_file(path) == expected_hash


@pytest.mark.parametrize(
    "tampered_path",
    (
        launcher.PREMARKET_ACCESS_SEAL,
        launcher.EVALUATOR_SOURCE,
        launcher.PROTOCOL_DOCUMENT,
        launcher.TEST_SOURCE,
    ),
)
def test_launcher_rejects_tampering_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tampered_path: Path
) -> None:
    for path in (
        launcher.PREMARKET_ACCESS_SEAL,
        launcher.EVALUATOR_SOURCE,
        launcher.PROTOCOL_DOCUMENT,
        launcher.TEST_SOURCE,
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


def test_launcher_delegates_to_write_once_only_after_hash_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "economic.json"
    calls: list[Path] = []

    def write_once(path: str | Path):
        calls.append(Path(path))
        return {"decision": "synthetic"}

    fake = SimpleNamespace(write_once=write_once)
    monkeypatch.setattr(launcher, "load_evaluator", lambda: fake)
    assert launcher.run(output) == {"decision": "synthetic"}
    assert calls == [output]
