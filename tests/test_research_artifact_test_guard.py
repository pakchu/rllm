from pathlib import Path

import pytest

from conftest import _artifact_failure_reason


def _capture(function):
    try:
        function()
    except BaseException:
        return pytest.ExceptionInfo.from_current()
    raise AssertionError("function did not fail")


def test_missing_optional_data_is_classified() -> None:
    info = _capture(
        lambda: (_ for _ in ()).throw(
            FileNotFoundError(2, "missing", "data/not-checked-in.csv.gz")
        )
    )
    assert _artifact_failure_reason(info) == (
        "optional research data is absent: data/not-checked-in.csv.gz"
    )


def test_missing_non_data_file_remains_a_failure() -> None:
    info = _capture(
        lambda: (_ for _ in ()).throw(
            FileNotFoundError(2, "missing", "results/tracked.json")
        )
    )
    assert _artifact_failure_reason(info) is None


def test_source_drift_under_data_is_classified() -> None:
    def fail() -> None:
        source_path = Path("data/revised.csv.gz")
        raise RuntimeError(f"frozen source changed: {source_path}")

    assert _artifact_failure_reason(_capture(fail)) == (
        "external frozen source bytes differ: data/revised.csv.gz"
    )


def test_source_contract_hash_drift_is_classified() -> None:
    def fail() -> None:
        source_contract = {"market_manifest": "data/revised-manifest.json"}
        raise ValueError("policy source hash drift: market_manifest")

    assert _artifact_failure_reason(_capture(fail)) == (
        "external frozen source bytes differ: data/revised-manifest.json"
    )


def test_unrelated_runtime_error_remains_a_failure() -> None:
    info = _capture(lambda: (_ for _ in ()).throw(RuntimeError("logic bug")))
    assert _artifact_failure_reason(info) is None
