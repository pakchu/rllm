"""Run the frozen GNRC source-support evaluator behind its outer hash seal."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ACCESS_SEAL = Path("results/gdelt_gnrc_source_access_seal_2026-07-22.json")
SOURCE_ACCESS_SEAL_SHA256 = (
    "267cbc8c1edd3bbfbbb290f39536a79ff90d51a26eb942c1dabcab5113cc81e8"
)
EVALUATOR_SOURCE = Path("training/evaluate_gdelt_narrative_source_support.py")
EVALUATOR_SOURCE_SHA256 = (
    "b09ae64c831376bce686e55de4bcbe630924faad7acc8cf81bc6cd31ff2b735a"
)
PROTOCOL_DOCUMENT = Path(
    "docs/gdelt-narrative-rotation-clearing-source-support-protocol-2026-07-20.md"
)
PROTOCOL_DOCUMENT_SHA256 = (
    "dfcf20bb5a5191ebe084feb0e9c23bdcc911f89828a78564874b8e03f02dd5ca"
)
DEFAULT_OUTPUT = Path(
    "results/gdelt_narrative_rotation_clearing_source_support_2026-07-20.json"
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bootstrap_inputs() -> None:
    expected_hashes = {
        SOURCE_ACCESS_SEAL: SOURCE_ACCESS_SEAL_SHA256,
        EVALUATOR_SOURCE: EVALUATOR_SOURCE_SHA256,
        PROTOCOL_DOCUMENT: PROTOCOL_DOCUMENT_SHA256,
    }
    for path, expected_hash in expected_hashes.items():
        if sha256_file(path) != expected_hash:
            raise RuntimeError(f"GNRC launcher frozen input changed before import: {path}")


def load_evaluator() -> ModuleType:
    verify_bootstrap_inputs()
    return importlib.import_module("training.evaluate_gdelt_narrative_source_support")


def run(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    evaluator = load_evaluator()
    report = evaluator.write_once(output)
    if not isinstance(report, dict):
        raise TypeError("GNRC source-support evaluator returned a non-object report")
    return report


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
