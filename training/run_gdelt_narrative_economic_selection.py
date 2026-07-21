"""Run the frozen GNRC pre-2024 economic selector behind its market seal."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREMARKET_ACCESS_SEAL = Path(
    "results/gdelt_gnrc_premarket_access_seal_2026-07-22.json"
)
PREMARKET_ACCESS_SEAL_SHA256 = (
    "eef502ab306a074f790593e10f3e7bf52642d7605433a4e5e3cf2b0e07a98478"
)
EVALUATOR_SOURCE = Path(
    "training/evaluate_gdelt_narrative_economic_selection.py"
)
EVALUATOR_SOURCE_SHA256 = (
    "7437fef90dd159d63e56a6226d0a14c8c17133442dabce0bd6a3338c3f5769b6"
)
PROTOCOL_DOCUMENT = Path(
    "docs/gdelt-narrative-rotation-clearing-economic-selection-protocol-"
    "2026-07-20.md"
)
PROTOCOL_DOCUMENT_SHA256 = (
    "bb570db9e18dbf77540af5e1e4ccc2bdeff439295cda031e557b3558dca8af2c"
)
TEST_SOURCE = Path(
    "tests/test_evaluate_gdelt_narrative_economic_selection.py"
)
TEST_SOURCE_SHA256 = (
    "acdc5950248a1d9fbff0950c6091d701c8448cda9a3eb529dc58347f2cecb0b3"
)
DEFAULT_OUTPUT = Path(
    "results/gdelt_narrative_rotation_clearing_economic_selection_2026-07-20.json"
)


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bootstrap_inputs() -> None:
    expected = {
        PREMARKET_ACCESS_SEAL: PREMARKET_ACCESS_SEAL_SHA256,
        EVALUATOR_SOURCE: EVALUATOR_SOURCE_SHA256,
        PROTOCOL_DOCUMENT: PROTOCOL_DOCUMENT_SHA256,
        TEST_SOURCE: TEST_SOURCE_SHA256,
    }
    for path, expected_hash in expected.items():
        if sha256_file(path) != expected_hash:
            raise RuntimeError(
                f"GNRC economic launcher frozen input changed before import: {path}"
            )


def load_evaluator() -> ModuleType:
    verify_bootstrap_inputs()
    return importlib.import_module(
        "training.evaluate_gdelt_narrative_economic_selection"
    )


def run(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    evaluator = load_evaluator()
    report = evaluator.write_once(output)
    if not isinstance(report, dict):
        raise TypeError("GNRC economic evaluator returned a non-object report")
    return report


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
