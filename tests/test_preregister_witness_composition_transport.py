from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from training import preregister_witness_composition_transport as prereg


def _cfg(tmp_path: Path, **changes: Any) -> prereg.Config:
    cfg = prereg.Config(
        source_manifest=str(prereg.SOURCE_MANIFEST),
        preregistration_output=str(tmp_path / "wctr-prereg.json"),
    )
    return replace(cfg, **changes)


def _install_manifest_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    manifest = deepcopy(
        json.loads(prereg._repository_path(prereg.SOURCE_MANIFEST).read_text())
    )
    mutate(manifest)
    core = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = prereg.canonical_hash(core)
    path = tmp_path / "source-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(prereg, "SOURCE_MANIFEST", path)
    monkeypatch.setattr(
        prereg, "EXPECTED_SOURCE_MANIFEST_FILE_SHA256", prereg.sha256_file(path)
    )
    monkeypatch.setattr(
        prereg, "EXPECTED_SOURCE_MANIFEST_HASH", manifest["manifest_hash"]
    )
    return path


def test_writes_exact_outcome_blind_singleton_and_hashes(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    artifact = prereg.write_preregistration(cfg)
    assert artifact == json.loads(Path(cfg.preregistration_output).read_text())
    core = {key: value for key, value in artifact.items() if key != "manifest_hash"}
    assert artifact["manifest_hash"] == prereg.canonical_hash(core)
    assert artifact["policy_hash"] == prereg.canonical_hash(artifact["policy"])
    assert artifact["preregistration_document"] == {
        "path": str(prereg.PREREGISTRATION_DOCUMENT),
        "sha256": prereg.PREREGISTRATION_DOCUMENT_SHA256,
    }
    assert artifact["policy_id"] == "WCTR-288"
    assert artifact["outcomes_opened"] is False
    assert artifact["outcome_boundary"] == {
        "source_manifest_json_read": True,
        "raw_source_artifact_bytes_hashed": True,
        "normalized_source_artifact_bytes_hashed": True,
        "raw_source_values_read": 0,
        "normalized_source_values_read": 0,
        "source_feature_rows_derived": 0,
        "signal_incidence_rows_derived": 0,
        "market_rows_loaded": 0,
        "funding_rows_loaded": 0,
        "return_or_pnl_fields": 0,
    }
    assert artifact["source_manifest"]["manifest_hash"] == (
        prereg.EXPECTED_SOURCE_MANIFEST_HASH
    )
    assert artifact["source_manifest"]["raw_artifact"] == (
        prereg.EXPECTED_RAW_ARTIFACT
    )
    assert artifact["source_manifest"]["normalized_artifact"] == (
        prereg.EXPECTED_NORMALIZED_ARTIFACT
    )
    assert artifact["source_manifest"]["source_semantics"] == (
        prereg.EXPECTED_SOURCE_SEMANTICS
    )

    policy = artifact["policy"]
    assert policy["singleton"] is True
    assert prereg.canonical_hash(policy) == (
        "510cedafde2775d65e3bc77eaefeccb9d526b9d738e503aa7c6c0e277974ddeb"
    )
    assert policy["source_features"]["witness_share"] == (
        "(4.0*avg_size-avg_weight)/(3.0*avg_size)"
    )
    assert policy["source_features"]["transport_horizon_buckets"] == 14
    assert policy["source_features"]["impulse_horizon_buckets"] == 2
    assert policy["source_features"]["fullness"] == "avg_weight/4000000.0"
    assert policy["source_features"]["component_controls"][
        "primary_excludes_component_fields"
    ] is True
    assert policy["normalization"]["lookback_valid_feature_buckets"] == 180
    assert policy["normalization"]["minimum_prior_valid_feature_buckets"] == 120
    assert policy["eligibility"]["common"] == (
        "magnitude_rank>=0.75 and fullness_rank>=0.50 and "
        "sign(transport_7d)==sign(impulse_24h)"
    )
    assert policy["eligibility"]["side"] == "sign(transport_7d)"
    assert policy["eligibility"]["zero_tolerance"] == (
        "none; exact binary64 zero is invalid"
    )
    assert policy["causal_availability"]["entry_latency_seconds"] == 300
    assert policy["causal_availability"]["ceil_5m"] == (
        "((unix_seconds+299)//300)*300"
    )
    assert policy["execution"]["hold_bars"] == 288
    assert policy["execution"]["non_overlap"] is True
    assert policy["calendar"]["test"] == (
        "[2024-01-01T00:00:00Z,2025-01-01T00:00:00Z)"
    )
    assert policy["calendar"]["eval"] == (
        "[2025-01-01T00:00:00Z,2026-01-01T00:00:00Z)"
    )
    assert policy["calendar"]["forward"] == (
        "[2026-01-01T00:00:00Z,2026-07-20T00:00:00Z)"
    )
    assert policy["calendar"]["later_window_use"].startswith("report-only")
    assert policy["support_gates"]["train_total_minimum"] == 45
    assert policy["support_gates"]["train_long_minimum"] == 14
    assert policy["support_gates"]["train_short_minimum"] == 14
    assert policy["support_gates"]["test_total_minimum"] == 35
    assert policy["support_gates"]["test_long_minimum"] == 10
    assert policy["support_gates"]["test_short_minimum"] == 10
    assert policy["support_gates"]["test_each_quarter_minimum"] == 5
    assert policy["support_gates"]["test_maximum_month_share"] == 0.20
    assert policy["support_gates"]["eval_total_minimum"] == 35
    assert policy["support_gates"]["eval_long_minimum"] == 10
    assert policy["support_gates"]["eval_short_minimum"] == 10
    assert policy["support_gates"]["eval_maximum_month_share"] == 0.20
    assert policy["support_gates"]["forward_total_minimum"] == 18
    assert policy["support_gates"]["forward_long_minimum"] == 5
    assert policy["support_gates"]["forward_short_minimum"] == 5
    assert policy["support_gates"]["forward_2026_h1_minimum"] == 16
    assert policy["support_gates"]["forward_maximum_month_share"] == 0.28
    assert policy["performance_gates"][
        "cagr_to_strict_mdd_minimum_each_constituent"
    ] == 3.0
    assert policy["performance_gates"][
        "strict_max_drawdown_maximum_each_constituent"
    ] == 0.15
    assert policy["performance_gates"]["stitched_cagr_to_strict_mdd_minimum"] == 3.0
    assert policy["performance_gates"]["positive_subperiods"] == [
        "train_2023H1",
        "train_2023H2",
        "test_2024H1",
        "test_2024H2",
        "eval_2025H1",
        "eval_2025H2",
        "forward_2026H1",
    ]
    assert policy["performance_gates"][
        "long_and_short_absolute_return_positive"
    ] == ["train", "test", "eval"]
    assert policy["performance_gates"][
        "one_bar_delayed_entry_absolute_return_positive_each_constituent"
    ] is True
    assert policy["performance_gates"][
        "component_control_full_gate_pass_rejects_specific_mechanism"
    ] == [
        "transport_only",
        "impulse_only",
        "low_fullness_complement",
        "serialized_size_only",
        "block_weight_only",
        "constant_long_same_clock",
        "constant_short_same_clock",
        "stale_7d",
        "month_side_stratified_random_clock",
    ]
    assert policy["controls"] == prereg.CONTROL_DEFINITIONS
    assert policy["promotion_boundary"].startswith("snapshot research only")
    assert "no sign, threshold" in policy["stopping_rule"]
    assert "support-floor" in policy["stopping_rule"]
    assert prereg.load_preregistration(cfg.preregistration_output) == artifact


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda m: m["config"].update(timeout_sec=31.0), "config drift"),
        (
            lambda m: m["source_decision"].update(sha256="0" * 64),
            "source_decision drift",
        ),
        (
            lambda m: m["source_builder"].update(sha256="0" * 64),
            "source_builder drift",
        ),
        (
            lambda m: m["source_audit"].update(missing_12h_buckets=1),
            "source_audit drift",
        ),
        (
            lambda m: m["raw_artifact"].update(sha256="0" * 64),
            "raw_artifact drift",
        ),
        (
            lambda m: m["normalized_artifact"].update(rows=2_920),
            "normalized_artifact drift",
        ),
        (
            lambda m: m["source_semantics"].update(derived_features="changed"),
            "source_semantics drift",
        ),
        (
            lambda m: m["causal_availability"].update(
                source_availability_lag_seconds=0
            ),
            "causal_availability drift",
        ),
        (
            lambda m: m["outcome_boundary"].update(btc_market_rows_loaded=1),
            "outcome_boundary drift",
        ),
        (lambda m: m.update(data_use="changed"), "data_use drift"),
    ],
)
def test_rejects_frozen_source_metadata_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    path = _install_manifest_copy(tmp_path, monkeypatch, mutate)
    with pytest.raises(RuntimeError, match=message):
        prereg.write_preregistration(
            _cfg(tmp_path, source_manifest=str(path))
        )


def test_rejects_manifest_canonical_and_file_sha_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = prereg._repository_path(prereg.SOURCE_MANIFEST).read_text()
    path = tmp_path / "source-manifest.json"
    path.write_text(original.replace('"http_status": 200', '"http_status": 201'))
    monkeypatch.setattr(prereg, "SOURCE_MANIFEST", path)
    monkeypatch.setattr(
        prereg, "EXPECTED_SOURCE_MANIFEST_FILE_SHA256", prereg.sha256_file(path)
    )
    with pytest.raises(RuntimeError, match="canonical hash"):
        prereg.write_preregistration(_cfg(tmp_path, source_manifest=str(path)))

    monkeypatch.setattr(prereg, "EXPECTED_SOURCE_MANIFEST_FILE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="file SHA"):
        prereg.write_preregistration(_cfg(tmp_path, source_manifest=str(path)))


def test_output_alias_immutability_and_cwd_independence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="protected source"):
        prereg.write_preregistration(
            _cfg(tmp_path, preregistration_output=str(prereg.SOURCE_MANIFEST))
        )
    with pytest.raises(ValueError, match="must be JSON"):
        prereg.write_preregistration(
            _cfg(tmp_path, preregistration_output=str(tmp_path / "artifact.txt"))
        )

    cfg = _cfg(tmp_path / "cwd-independent")
    monkeypatch.chdir(tmp_path)
    prereg.write_preregistration(cfg)
    assert Path(cfg.preregistration_output).is_file()
    with pytest.raises(FileExistsError, match="immutable"):
        prereg.write_preregistration(cfg)


def test_load_rejects_policy_outcome_and_source_binding_drift(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    artifact = prereg.write_preregistration(cfg)
    path = Path(cfg.preregistration_output)

    def write_drift(mutator: Callable[[dict[str, Any]], None]) -> None:
        drift = deepcopy(artifact)
        mutator(drift)
        core = {
            key: value for key, value in drift.items() if key != "manifest_hash"
        }
        drift["manifest_hash"] = prereg.canonical_hash(core)
        path.write_text(json.dumps(drift), encoding="utf-8")

    write_drift(lambda value: value["policy"].update(singleton=False))
    with pytest.raises(RuntimeError, match="policy drift"):
        prereg.load_preregistration(path)

    write_drift(lambda value: value.update(outcomes_opened=True))
    with pytest.raises(RuntimeError, match="opened outcomes"):
        prereg.load_preregistration(path)

    write_drift(
        lambda value: value["outcome_boundary"].update(market_rows_loaded=1)
    )
    with pytest.raises(RuntimeError, match="outcome boundary drift"):
        prereg.load_preregistration(path)

    write_drift(
        lambda value: value["source_manifest"].update(manifest_hash="0" * 64)
    )
    with pytest.raises(RuntimeError, match="source-manifest binding drift"):
        prereg.load_preregistration(path)

    write_drift(
        lambda value: value["preregistration_document"].update(sha256="0" * 64)
    )
    with pytest.raises(RuntimeError, match="document binding drift"):
        prereg.load_preregistration(path)

    write_drift(
        lambda value: value["config"].update(
            preregistration_output=str(tmp_path / "other.json")
        )
    )
    with pytest.raises(RuntimeError, match="output-path binding drift"):
        prereg.load_preregistration(path)
