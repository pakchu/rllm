from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from training import evaluate_stablecoin_quote_flow_diffusion as evaluator


RESULT = evaluator.STAGE_OUTPUTS["train"]
DOCUMENT = evaluator.STAGE_DOCS["train"]
RESULT_SHA256 = "85d44366bf5e4edcfc9cd9d5abdec268cc20ac8c0ba80ba35d824bfda69e6f6a"
DOCUMENT_SHA256 = "6632d5adfc39042bdfbad8ef301ddfbc21ce9949200ace5e61fd9f614051ae8d"
MANIFEST_HASH = "9333bffca33062a478e7748e8eae2d25142026a78d7e8c24b127dc4cd73465cc"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load() -> dict[str, Any]:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_sqfd_train_rejection_artifact_bytes_and_manifest_are_locked() -> None:
    payload = _load()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}

    assert _sha256(RESULT) == RESULT_SHA256
    assert _sha256(DOCUMENT) == DOCUMENT_SHA256
    assert payload["manifest_hash"] == MANIFEST_HASH
    assert evaluator._canonical_hash(core) == MANIFEST_HASH
    assert payload["evaluator_source_sha256"] == (
        "0ea59a107f05777ba91ab1c8fc5900e724ba48ec6ce647a42c34c34422222e3b"
    )


def test_sqfd_train_failed_without_opening_any_future_window() -> None:
    payload = _load()

    assert payload["stage"] == "train"
    assert payload["stage_passed"] is False
    assert payload["disposition"] == "REJECT_NO_REPAIR"
    assert payload["opened_windows"] == ["train"]
    assert payload["sealed_windows"] == ["test", "eval", "final"]
    assert not evaluator.STAGE_OUTPUTS["test"].exists()
    assert not evaluator.STAGE_OUTPUTS["eval"].exists()
    assert not evaluator.STAGE_OUTPUTS["final"].exists()


def test_sqfd_train_headline_and_failed_gates_are_locked() -> None:
    payload = _load()
    primary = payload["primary"]
    assert isinstance(primary, dict)
    headline = primary["headline"]
    assert isinstance(headline, dict)

    assert headline["absolute_return_pct"] == pytest.approx(-1.720033260895304)
    assert headline["cagr_pct"] == pytest.approx(-3.3854318189226684)
    assert headline["strict_mdd_pct"] == pytest.approx(8.585880438314607)
    assert headline["cagr_to_strict_mdd"] == pytest.approx(-0.39430223181482155)
    assert headline["trades"] == 55
    assert headline["longs"] == 32
    assert headline["shorts"] == 23
    assert headline["mean_gross_underlying_bp"] == pytest.approx(6.16619313871094)
    assert headline["weekly_cluster_signflip_p"] == pytest.approx(0.700225830078125)
    assert payload["failed_gates"] == [
        "absolute_return_positive",
        "cagr_to_strict_mdd_at_least_3",
        "weekly_cluster_signflip_p_at_most_10pct",
        "mean_gross_underlying_at_least_20bp",
        "each_contained_half_absolute_return_positive",
        "stress_absolute_return_positive",
        "stress_cagr_to_strict_mdd_at_least_2_5",
        "mechanism_control_margin_at_least_0_25",
    ]


def test_sqfd_train_physical_source_window_and_checksums_are_locked() -> None:
    payload = _load()
    diagnostics = payload["execution_diagnostics"]
    assert isinstance(diagnostics, dict)

    assert diagnostics["physical_window"] == [
        "2023-07-01T00:00:00+00:00",
        "2024-01-01T00:00:00+00:00",
    ]
    assert diagnostics["market_sha256"] == evaluator.TRAIN_MARKET_SHA256
    assert diagnostics["funding_sha256"] == evaluator.TRAIN_FUNDING_SHA256
    market = diagnostics["market"]
    funding = diagnostics["funding"]
    assert isinstance(market, dict)
    assert isinstance(funding, dict)
    assert market["rows"] == 52_992
    assert funding["rows"] == 552
    assert funding["exit_boundary_events"] == 0
