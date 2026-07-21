from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterator

import pytest

from training import preregister_cross_domain_liquidity_transmission_relay as prereg


@contextmanager
def _temporary_directory() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir=prereg.REPOSITORY_ROOT / "results") as raw:
        yield Path(raw)


def _relative(path: Path) -> str:
    return str(path.relative_to(prereg.REPOSITORY_ROOT))


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    payload["manifest_hash"] = prereg.canonical_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    return payload


def test_writes_exact_singleton_with_closed_outcome_boundary() -> None:
    with _temporary_directory() as directory:
        cfg = prereg.Config(output=_relative(directory / "cdltr-prereg.json"))
        artifact = prereg.write_preregistration(cfg)
        output = prereg._repository_path(cfg.output)

        assert artifact == json.loads(output.read_text(encoding="utf-8"))
        assert artifact["protocol_version"] == prereg.PROTOCOL_VERSION
        assert artifact["candidate"] == "CDLTR-72A"
        assert artifact["policy"]["singleton"] is True
        assert artifact["outcomes_opened"] is False
        assert artifact["source_incidence_opened"] is False
        assert artifact["comparator_incidence_opened"] is False
        assert artifact["performance_values_opened"] is False
        assert artifact["outcome_boundary"] == prereg.EXPECTED_OUTCOME_BOUNDARY
        assert all(value == 0 for value in artifact["outcome_boundary"].values())
        assert artifact["manifest_hash"] == prereg.canonical_hash(
            {key: value for key, value in artifact.items() if key != "manifest_hash"}
        )
        assert artifact["policy_hash"] == prereg.canonical_hash(artifact["policy"])
        assert prereg.load_preregistration(cfg.output) == artifact


def test_policy_freezes_votes_relay_execution_support_controls_and_llm() -> None:
    policy = prereg.policy_payload()
    assert (
        "CDLTR-72 rejected before preregistration" in policy["predecessor_disposition"]
    )

    rrp = policy["source_votes"]["rrp"]
    assert rrp["lookback_normal_operation_slots"] == 5
    assert rrp["required_complete_slots_inclusive"] == 6
    assert rrp["quarantine_breaks_baseline"] is True
    assert rrp["long"].endswith("< 0")
    assert rrp["short"].endswith("> 0")

    cboe = policy["source_votes"]["cboe"]
    assert cboe["long"] == "VIX9D_close < VIX3M_close"
    assert cboe["forward_fill"] is False
    network = policy["source_votes"]["network"]
    assert network["lookback_calendar_days"] == 7
    assert network["required_consecutive_dates"] == 8
    assert network["metrics"] == ["AdrActCnt", "TxCnt", "TxTfrCnt"]

    relay = policy["relay"]
    assert relay["rrp_vote_expiry_hours"] == 36
    assert relay["cboe_vote_expiry_hours"] == 36
    assert relay["network_deadline_hours_after_onset"] == 36
    assert relay["failed_confirmation_retry"] is False
    execution = policy["execution"]
    assert execution["entry"] == "ceil_to_5m(decision_time) + 5 minutes"
    assert execution["hold_hours"] == 72
    assert execution["notional_exposure"] == 0.5
    assert execution["global_nonoverlap"] is True

    gates = policy["support_gates"]
    assert gates == {
        "train_total_minimum": 60,
        "each_train_year_minimum": 25,
        "each_train_half_year_minimum": 12,
        "selection_total_minimum": 30,
        "each_selection_half_year_minimum": 12,
        "train_each_side_minimum": 18,
        "selection_each_side_minimum": 8,
        "maximum_month_share": 0.20,
        "maximum_weekday_share": 0.35,
        "all_controls_must_pass_calendar_and_containment": True,
    }
    controls = policy["controls"]
    assert tuple(controls) == (
        "macro_only",
        "network_only",
        "reverse_order",
        "one_network_report_delay",
        "direction_flip",
        "deterministic_random_side",
    )
    assert "CDLTR-72|20260721|" in controls["deterministic_random_side"]
    assert "predecessor seed retained" in controls["deterministic_random_side"]

    novelty = policy["novelty_gates"]
    assert novelty["decision_date_jaccard_maximum"] == 0.30
    assert novelty["cdltr_dates_within_one_utc_day_fraction_maximum"] == 0.50
    assert novelty["signed_occupied_exposure_absolute_pearson_maximum"] == 0.40
    assert novelty["timestamp_only_comparators"] == list(
        prereg.TIMESTAMP_ONLY_COMPARATORS
    )
    assert novelty["signed_exposure_applies_to"] == list(prereg.DIRECTIONAL_COMPARATORS)
    assert set(novelty["timestamp_gates_apply_to"]) == {
        *prereg.DIRECTIONAL_COMPARATORS,
        *prereg.TIMESTAMP_ONLY_COMPARATORS,
    }
    assert novelty["flcc_candidates_pass_independently"] is True
    assert novelty["missing_side_exit_or_union_may_not_be_invented"] is True

    llm = policy["llm_boundary"]
    assert llm["authorized_before_deterministic_train_and_selection_pass"] is False
    assert llm["later_role"] == "TRADE/ABSTAIN veto only"
    assert llm["may_change_side_timing_hold_or_relay"] is False


def test_source_hash_header_builder_manifest_and_allowlist_are_bound() -> None:
    artifact = prereg.build_preregistration()
    bindings = artifact["source_bindings"]
    assert set(bindings) == {"rrp", "cboe", "network"}
    for name, frozen in prereg.SOURCE_BINDINGS.items():
        bound = bindings[name]
        assert bound["source"] == str(frozen["source"])
        assert bound["source_sha256"] == frozen["source_sha256"]
        assert bound["manifest"] == str(frozen["manifest"])
        assert bound["manifest_sha256"] == frozen["manifest_sha256"]
        assert bound["builder"] == str(frozen["builder"])
        assert bound["builder_sha256"] == frozen["builder_sha256"]
        assert bound["header"] == list(frozen["header"])
        assert bound["allowed_columns"] == list(frozen["allowed_columns"])


def test_complete_sanitized_comparator_bundle_is_the_only_binding() -> None:
    artifact = prereg.build_preregistration()
    binding = artifact["comparator_binding"]
    assert binding == {
        "clock": str(prereg.COMPARATOR_CLOCK),
        "clock_sha256": prereg.COMPARATOR_CLOCK_SHA256,
        "manifest": str(prereg.COMPARATOR_MANIFEST),
        "manifest_sha256": prereg.COMPARATOR_MANIFEST_SHA256,
        "format": "csv_gzip",
        "header": list(prereg.COMPARATOR_HEADER),
        "rows": 9_985,
        "directional_rows": 1_788,
        "timestamp_only_rows": 8_197,
        "directional_comparators": list(prereg.DIRECTIONAL_COMPARATORS),
        "timestamp_only_comparators": list(prereg.TIMESTAMP_ONLY_COMPARATORS),
        "event_rows_read_during_preregistration": 0,
        "manifest_values_parsed_during_preregistration": 0,
    }
    assert "comparator_bindings" not in artifact


def test_build_does_not_parse_json_or_source_and_comparator_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_loads(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("preregistration must not parse bound JSON artifacts")

    monkeypatch.setattr(prereg.json, "loads", forbidden_loads)
    artifact = prereg.build_preregistration()
    assert artifact["outcome_boundary"] == prereg.EXPECTED_OUTCOME_BOUNDARY
    assert artifact["comparator_binding"]["event_rows_read_during_preregistration"] == 0
    assert (
        artifact["comparator_binding"]["manifest_values_parsed_during_preregistration"]
        == 0
    )


def test_source_and_complete_comparator_hash_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_source = prereg.SOURCE_BINDINGS["network"]
    drifted_source = {**original_source, "source_sha256": "0" * 64}
    monkeypatch.setitem(prereg.SOURCE_BINDINGS, "network", drifted_source)
    with pytest.raises(RuntimeError, match="network source SHA drift"):
        prereg.build_preregistration()

    monkeypatch.setitem(prereg.SOURCE_BINDINGS, "network", original_source)
    monkeypatch.setattr(prereg, "COMPARATOR_CLOCK_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="complete comparator clock SHA drift"):
        prereg.build_preregistration()


def test_bound_input_symlink_and_output_alias_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _temporary_directory() as directory:
        real = prereg._repository_path(prereg.NETWORK_SOURCE)
        link = directory / "network.csv.gz"
        link.symlink_to(real)
        original = prereg.SOURCE_BINDINGS["network"]
        monkeypatch.setitem(
            prereg.SOURCE_BINDINGS,
            "network",
            {**original, "source": Path(_relative(link))},
        )
        with pytest.raises(RuntimeError, match="symlink"):
            prereg.build_preregistration()

        monkeypatch.setitem(prereg.SOURCE_BINDINGS, "network", original)
        with pytest.raises(ValueError, match="aliases a protected input"):
            prereg.write_preregistration(prereg.Config(output=str(prereg.RRP_MANIFEST)))


def test_paths_reject_absolute_tilde_and_parent_traversal() -> None:
    for unsafe in ("/tmp/outside.json", "~/outside.json", "../outside.json"):
        with pytest.raises(RuntimeError, match="repository-relative"):
            prereg._repository_path(unsafe)


def test_symlinked_parent_directory_is_rejected_for_input_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory() as external, _temporary_directory() as holder:
        escape = holder / "escape"
        escape.symlink_to(external, target_is_directory=True)
        escaped_relative = Path(_relative(escape))

        original = prereg.SOURCE_BINDINGS["network"]
        monkeypatch.setitem(
            prereg.SOURCE_BINDINGS,
            "network",
            {**original, "source": escaped_relative / "network.csv.gz"},
        )
        with pytest.raises(RuntimeError, match="contains a symlink"):
            prereg.build_preregistration()

        monkeypatch.setitem(prereg.SOURCE_BINDINGS, "network", original)
        with pytest.raises(RuntimeError, match="contains a symlink"):
            prereg.write_preregistration(
                prereg.Config(output=str(escaped_relative / "outside.json"))
            )


def test_immutable_output_and_tampered_artifact_fail_closed() -> None:
    with _temporary_directory() as directory:
        cfg = prereg.Config(output=_relative(directory / "cdltr-prereg.json"))
        artifact = prereg.write_preregistration(cfg)
        with pytest.raises(FileExistsError, match="immutable"):
            prereg.write_preregistration(cfg)

        tampered = deepcopy(artifact)
        tampered["policy"]["relay"]["rrp_vote_expiry_hours"] = 37
        _rehash(tampered)
        path = directory / "tampered.json"
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(RuntimeError, match="policy drift"):
            prereg.load_preregistration(_relative(path))


def test_cli_writes_once_and_refuses_overwrite() -> None:
    with _temporary_directory() as directory:
        relative_output = _relative(directory / "cli-cdltr.json")
        command = [
            sys.executable,
            "-m",
            "training.preregister_cross_domain_liquidity_transmission_relay",
            "--output",
            relative_output,
        ]
        first = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=prereg.REPOSITORY_ROOT,
        )
        assert json.loads(first.stdout)["candidate"] == "CDLTR-72A"
        assert prereg.load_preregistration(relative_output)["outcomes_opened"] is False

        second = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=prereg.REPOSITORY_ROOT,
        )
        assert second.returncode != 0
        assert "immutable" in second.stderr
