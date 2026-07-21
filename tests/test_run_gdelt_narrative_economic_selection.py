from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import run_gdelt_narrative_economic_selection as launcher


ECONOMIC_REPORT_SHA256 = (
    "c305c47268649c2e13e37b755ef32af2f626bb71d092a1c7bba15417d6eb90a7"
)
ECONOMIC_REPORT_MANIFEST_HASH = (
    "a68b11e41caf21c130c22ef2a93c72d6735d0f449dea2a7b2bc38204f85091ee"
)


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


def test_committed_economic_result_retires_without_opening_oos() -> None:
    assert launcher.sha256_file(launcher.DEFAULT_OUTPUT) == ECONOMIC_REPORT_SHA256
    with launcher.repository_path(launcher.DEFAULT_OUTPUT).open(
        encoding="utf-8"
    ) as handle:
        report = json.load(handle)
    assert report["manifest_hash"] == ECONOMIC_REPORT_MANIFEST_HASH
    assert report["decision"] == "retire_without_repair"
    assert report["champion_variant_id"] is None
    assert report["champion_policy_hash"] is None
    assert report["selection_candidate_ids"] == []
    assert report["familywise_test"]["ordered_tested_variant_ids"] == []
    assert report["outcome_boundary"] == {
        "oos_opened": False,
        "post_2023_funding_rows_read": 0,
        "post_2023_market_rows_read": 0,
        "post_2023_news_rows_read": 0,
        "pre2024_funding_rows_read": 3_285,
        "pre2024_market_rows_read": 315_360,
    }
    unsupported = [
        variant
        for variant in report["variant_results"].values()
        if not variant["source_support_pass"]
    ]
    assert len(unsupported) == 7
    assert all(
        variant["outcome_status"] == "not_opened_source_unsupported"
        and variant["train"]["market_outcome_opened"] is False
        and variant["selection"]["base_2bps_per_side"] is None
        for variant in unsupported
    )
