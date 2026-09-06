from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import importlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from training import preregister_cross_venue_volatility_shape_handoff as p


def fake_repository() -> dict[str, Any]:
    seal = {
        relative: {
            "git_blob": "c" * 40,
            "sha256": (
                p.COMMON_WINDOW_POLICY["sha256"]
                if relative == p.COMMON_WINDOW_POLICY["path"]
                else "d" * 64
            ),
        }
        for relative in p.PROTOCOL_PATHS
    }
    payload = {
        "branch": p.EXPECTED_BRANCH,
        "commit": "a" * 40,
        "tree": "b" * 40,
        "upstream": f"origin/{p.EXPECTED_BRANCH}",
        "upstream_ref": f"refs/remotes/origin/{p.EXPECTED_BRANCH}",
        "upstream_remote": "origin",
        "upstream_remote_url": p.EXPECTED_ORIGIN_URL,
        "upstream_fetch_urls": [p.EXPECTED_ORIGIN_URL],
        "upstream_push_urls": [p.EXPECTED_ORIGIN_URL],
        "upstream_commit": "a" * 40,
        "canonical_remote_commit": "a" * 40,
        "tracked_clean": True,
        "upstream_exact": True,
        "protocol_seal": seal,
        "protocol_seal_hash": p.canonical_hash(seal),
    }
    return payload


def fake_bindings() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, spec in {**p.SOURCE_ARTIFACTS, **p.PRIOR_VOLATILITY_COMPARATORS}.items():
        row: dict[str, Any] = {
            "path": spec["path"],
            "sha256": spec["sha256"],
            "bytes": 1,
        }
        if "header" in spec:
            row.update(
                {
                    "header": copy.deepcopy(spec["header"]),
                    "header_line_sha256": spec["header_line_sha256"],
                    "rows_decoded": 0,
                }
            )
        output[name] = row
    return output


def fake_gross9() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    return p.load_gross9_authority(root)


def registration() -> dict[str, Any]:
    return p.build_registration(
        repository=fake_repository(),
        artifact_bindings=fake_bindings(),
        gross9=fake_gross9(),
    )


def test_registration_is_source_blind_and_singleton() -> None:
    result = registration()
    assert result["policy_id"] == "CVVH-432"
    assert result["singleton"] is True
    assert result["research_boundary"]["bvol_or_dvol_rows_decoded"] == 0
    assert result["research_boundary"]["candidate_incidence_opened"] is False
    assert result["research_boundary"]["gross9_clock_rows_opened"] == 0
    assert result["research_boundary"]["btc_execution_rows_opened"] == 0
    assert result["research_boundary"]["funding_rows_opened"] == 0
    assert (
        result["research_boundary"]["return_pnl_cagr_or_drawdown_opened"] is False
    )
    assert result["producer_effects"]["csv_data_rows_decoded"] == 0
    assert result["producer_effects"]["network_calls"] == {
        "source_or_web": 0,
        "read_only_git_ls_remote_attestation": 1,
    }


def test_source_hash_headers_join_and_decimal_contract_are_exact() -> None:
    result = registration()["source_contract"]
    assert set(result["artifacts"]) == set(p.SOURCE_ARTIFACTS)
    assert result["join"] == {
        "type": "exact UTC one-to-one completed-hour inner join",
        "bvol_key": "feature_available_time_utc",
        "dvol_key": "close_time",
        "joint_availability": "max of both exact source clocks",
        "fill_imputation_tolerance_or_nearest": False,
        "duplicate_or_nonmonotonic": "terminal failure",
        "first_post_gap_or_invalid_can_emit": False,
    }
    assert result["decimal_tokens"]["coefficient_digits_max"] == 128
    assert result["decimal_tokens"]["exponent_min"] == -128
    assert result["decimal_tokens"]["exponent_max"] == 128
    assert result["decimal_tokens"]["binary_float_feature_arithmetic"] is False
    for name, spec in p.SOURCE_ARTIFACTS.items():
        assert result["artifacts"][name]["sha256"] == spec["sha256"]
        if "header" in spec:
            assert result["artifacts"][name]["validated_binding"][
                "rows_decoded"
            ] == 0


def test_mechanism_execution_and_control_contracts_are_frozen() -> None:
    result = registration()
    mechanism = result["mechanism"]
    assert mechanism["primary"]["binance_positive"] == "SHORT"
    assert mechanism["primary"]["binance_negative"] == "LONG"
    assert mechanism["primary"]["equality"] == "NONE"
    assert mechanism["onset"]["opposite_side_transition"] == "emit"
    assert mechanism["onset"]["first_after_gap_or_invalid"] == "suppress"
    execution = result["execution"]
    assert execution["hold_bars_5m"] == 432
    assert execution["hold_seconds"] == 129_600
    assert execution["leverage"] == "1/2"
    assert execution["base_cost_bp_per_notional_side"] == 6
    assert execution["stress_cost_bp_per_notional_side"] == 10
    assert execution["reservation"]["suppressed_candidates_queued"] is False
    assert set(result["controls"]["independent_own_clock"]) == {
        "deribit_led",
        "body_lead_only",
        "range_lead_only",
        "stale_deribit",
    }


def test_support_floors_append_invariance_and_terminal_action_are_exact() -> None:
    support = registration()["support_gates"]
    assert support["selection"] == {
        "total_min": 45,
        "2023H2_min": 12,
        "2024H1_min": 12,
        "2024H2_min": 12,
        "each_side_min": 14,
        "maximum_month_share": "1/5",
    }
    assert support["future25"] == {
        "total_min": 30,
        "each_side_min": 8,
        "maximum_month_share": "1/4",
    }
    assert support["future26"] == {
        "total_min": 15,
        "each_side_min": 4,
        "maximum_month_share": "3/10",
    }
    assert support["full"]["maximum_accepted_entry_gap_elapsed_days"] == 90
    assert support["full"]["maximum_same_side_run"] == 12
    assert support["selection_prefix_append_invariance"][
        "later_rows_may_change_prefix"
    ] is False
    assert "retire exact CVVH-432" in support["failure_action"]


def test_prior_volatility_registry_is_hash_header_parser_and_window_bound() -> None:
    novelty = registration()["novelty"]
    assert novelty["common_window_policy"] == p.COMMON_WINDOW_POLICY
    assert set(novelty["prior_volatility_comparators"]) == {
        "OPDR-24",
        "old_dvol_price_follow",
        "PSR-30/6",
        "PCBR-12",
        "CMSR-36",
    }
    for name, spec in p.PRIOR_VOLATILITY_COMPARATORS.items():
        observed = novelty["prior_volatility_comparators"][name]
        for key in (
            "path",
            "sha256",
            "header",
            "header_line_sha256",
            "filters",
            "entry_column",
            "exit_column",
            "side_column",
            "side_parser",
            "common_window",
        ):
            assert observed[key] == spec[key]
    assert novelty["minimum_fully_contained_rows_each_clock"] == 10
    assert novelty["full_containment_only_no_clip_shift_or_split"] is True
    assert all(
        spec["side_parser"] == {"1": 1, "-1": -1}
        for spec in p.PRIOR_VOLATILITY_COMPARATORS.values()
    )


def test_frozen_comparator_producers_serialize_numeric_sides() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = {
        "training/build_options_perpetual_demand_relay_support.py": (
            '"side": side,',
            'side = int(row["side"])',
        ),
        "training/preregister_cross_venue_vol_disagreement_alpha.py": (
            "action = int(side.iloc[position])",
            '"side": action,',
            '"longs": int((schedule["side"] > 0).sum())',
        ),
        "training/preregister_premium_snapback_recenter.py": (
            '"direction": int(row[direction_column])',
        ),
        "training/build_premium_compression_breakout_relay_support.py": (
            'side = int(row["side"])',
            '"side": side,',
        ),
        "training/build_coinm_next_maturity_shock_relay_support.py": (
            '"side": int(sides[control].iloc[position])',
            'isin((-1, 1))',
        ),
    }
    for relative, snippets in evidence.items():
        text = (root / relative).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in text


def test_matching_and_novelty_thresholds_are_unambiguous() -> None:
    novelty = registration()["novelty"]
    assert novelty["matching"] == {
        "tolerance_elapsed_seconds": 21_600,
        "one_to_one": True,
        "objective_order": [
            "maximum matched cardinality",
            "minimum exact total absolute elapsed seconds",
            "lexicographically smallest ordered timestamp-pair list",
        ],
        "reported_pair_list_sha256": True,
    }
    assert novelty["thresholds_each_prior_volatility_and_gross9_sleeve"] == {
        "exact_entry_jaccard_max": "1/10",
        "one_to_one_6h_max_matched_share_max": "7/20",
        "occupied_5m_bar_jaccard_max": "1/4",
        "absolute_signed_exposure_pearson_max": "7/20",
        "pearson_implementation_gate": "exact square<=49/400",
    }
    assert novelty["all_declared_comparators_and_sleeves_must_pass"] is True
    disclosure = novelty["common_window_policy_contamination_disclosure"]
    assert disclosure["disclosed"] is True
    assert "RMSR" in disclosure["fact"]
    assert disclosure["eligibility_rule_frozen_before_cvvh_incidence"] is True
    assert "no CVVH source value" in disclosure["effect"]


def test_complete_gross9_authority_roster_and_boundary_are_frozen() -> None:
    result = registration()
    assert result["gross9"]["weights"] == p.GROSS9_WEIGHTS
    assert sum(result["gross9"]["weights"].values()) == 9.0
    assert result["gross9"]["gross"] == 9.0
    assert result["gross9"]["reference_preregistration"] == p.ESDI_PREREGISTRATION
    assert result["gross9"]["reuse_exact_esdi_runtime_closure_validation"] is True
    boundary = result["gross9_evidence_boundary"]
    assert boundary["future_rows_used_for_structural_candidate_veto"] is True
    assert boundary["future_rows_used_for_economic_weight_ranking"] is False
    assert boundary["portfolio_return_pnl_metrics_at_novelty"] is False


def test_selection_future_and_exact_three_year_calendars_cannot_drift() -> None:
    calendars = registration()["calendars"]
    assert calendars["full"] == [
        "2023-06-01T00:00:00Z",
        "2026-06-01T00:00:00Z",
    ]
    assert calendars["selection_periods"] == {
        "2023H2": [
            "2023-06-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
        ],
        "2024": [
            "2024-01-01T00:00:00Z",
            "2025-01-01T00:00:00Z",
        ],
    }
    assert calendars["future25"] == [
        "2025-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    ]
    assert calendars["future26"] == [
        "2026-01-01T00:00:00Z",
        "2026-06-01T00:00:00Z",
    ]
    assert calendars["full_calendar_years"] == 3


def test_economic_engine_same_gross_and_future_veto_are_strict() -> None:
    result = registration()
    economic = result["economic_contract"]
    assert economic["engine"] == "exact ESDI strict-open fixed-quantity engine"
    assert economic["path_order"][0] == "global/pre-entry high-water mark"
    assert economic["signflip"]["gated_stages"].startswith(
        "exact exhaustive 2^k"
    )
    same = result["same_gross"]
    assert same["candidate_weights"] == ["1/4", "1/2", "3/4"]
    assert same["configured_treatment_gross"] == 9.0
    assert same["selection_periods_only"] == ["2023H2", "2024"]
    assert same["requirements_every_period_base_and_stress"] == {
        "cagr_to_strict_mdd_absolute_improvement_min": "1/20",
        "unscaled_absolute_return_retention_min": "97/100",
        "treatment_absolute_return_positive": True,
        "liquidation_safe": True,
    }
    assert same["freeze_rank"] == 1
    assert same["top1_must_pass_or_terminal"] is True
    assert same["rank2_substitution"] is False
    assert same["future"]["rerank_repair_or_alternate_weight"] is False
    assert (
        same["stitched_exact_three_year_report"]["descriptive_non_gating"] is True
    )
    controls = economic["source_specific_superiority"]
    assert controls == {
        "metric": "cagr_to_strict_mdd",
        "operator": "primary > control",
        "periods": ["2023H2", "2024"],
        "costs": ["base", "stress"],
        "primary_must_strictly_exceed": [
            "body_lead_only",
            "range_lead_only",
        ],
        "diagnostic_only_cannot_replace": [
            "deribit_led",
            "stale_deribit",
            "one_bar_delayed_entry",
        ],
        "cannot_completely_qualify": [
            "direction_flip",
            "deterministic_random_side",
            "constant_long",
            "constant_short",
        ],
    }


def test_claim_and_verification_replay_contract_has_no_retry_authority() -> None:
    result = registration()["attempt_and_reproduction_contract"]
    authoritative = result["each_authoritative_stage"]
    assert authoritative["atomic_write_once_claim_before_first_protected_read"]
    assert authoritative["claim_binds"] == [
        "commit",
        "preregistration",
        "evaluator closure",
        "dependency hashes",
        "prior receipt hashes",
    ]
    assert authoritative["retry_resume_or_fallback_after_claim"] is False
    assert authoritative["stop_on_first_failure"] is True
    replay = result["verification_replay"]
    assert replay["allowed_only_after_successful_committed_authoritative_bytes"]
    assert replay["verification_only"] is True
    assert replay["separate_atomic_write_once_claim"] is True
    assert replay["clean_checkout_and_upstream_exact"] is True
    assert replay["canonical_write_ranking_or_repair_authority"] is False
    assert replay["byte_identical_temp_artifacts_and_receipts_required"] is True
    assert replay["authoritative_failure_permanently_forbids_replay"] is True
    assert replay["mismatch"] == "terminal reproducibility failure"
    sequence = registration()["stage_sequence"]
    assert sequence.index("authoritative_source_support_claim_then_one_run") < (
        sequence.index("authoritative_novelty_claim_then_one_run")
    )
    assert sequence.index("authoritative_novelty_claim_then_one_run") < (
        sequence.index("economics_evaluator_tests_commit_and_push")
    )
    same = registration()["same_gross"]
    assert same["future"] == {
        "weight": "exact frozen rank1 only",
        "rerank_repair_or_alternate_weight": False,
        "future25_then_future26": True,
        "each_requires_standalone_and_same_gross_base_stress": True,
        "each_requires_strict_mdd_reduction_in_at_least_one_cost_cell": True,
        "failure": "terminal veto",
    }
    assert same["stitched_exact_three_year_report"] == {
        "required_after_both_future_pass": True,
        "descriptive_non_gating": True,
        "can_repair_or_rerank": False,
    }
    assert registration()["sequence_rules"] == {
        "stop_at_first_failure": True,
        "parameter_threshold_hold_latency_or_polarity_repair": False,
        "future_can_rank_or_repair": False,
        "control_can_replace_primary": False,
        "ordinary_failure_repair_under_same_identity": False,
    }


def test_manifest_hash_detects_tampering() -> None:
    result = registration()
    p.validate_registration(result)
    tampered = copy.deepcopy(result)
    tampered["execution"]["hold_bars_5m"] = 431
    with pytest.raises(RuntimeError, match="manifest hash drift"):
        p.validate_registration(tampered)


def _rehashed(payload: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(payload)
    core = {key: value for key, value in output.items() if key != "manifest_hash"}
    output["manifest_hash"] = p.canonical_hash(core)
    return output


@pytest.mark.parametrize(
    "mutator",
    [
        lambda row: row["execution"].__setitem__("hold_bars_5m", 431),
        lambda row: row["repository"].__setitem__(
            "protocol_seal_hash", "0" * 64
        ),
        lambda row: row["source_contract"]["artifacts"][
            "binance_btc_bvol_hourly"
        ]["validated_binding"].__setitem__("sha256", "0" * 64),
        lambda row: row["gross9"].pop("authority"),
    ],
)
def test_recomputed_manifest_cannot_authorize_frozen_contract_tampering(
    mutator: Any,
) -> None:
    tampered = copy.deepcopy(registration())
    mutator(tampered)
    tampered = _rehashed(tampered)
    with pytest.raises(RuntimeError):
        p.validate_registration(tampered)


def test_build_registration_rejects_dirty_or_wrong_gross9_identity() -> None:
    dirty = fake_repository()
    dirty["tracked_clean"] = False
    with pytest.raises(RuntimeError, match="repository identity"):
        p.build_registration(
            repository=dirty,
            artifact_bindings=fake_bindings(),
            gross9=fake_gross9(),
        )
    wrong = fake_gross9()
    wrong["weights"] = {**p.GROSS9_WEIGHTS, "extra": 1.0}
    with pytest.raises(RuntimeError, match="roster drift"):
        p.build_registration(
            repository=fake_repository(),
            artifact_bindings=fake_bindings(),
            gross9=wrong,
        )


def test_atomic_write_once_is_complete_and_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "registration.json"
    raw = p.encoded_registration(registration())
    p.atomic_write_once(output, raw)
    assert output.read_bytes() == raw
    assert not list(output.parent.glob(".*.tmp-*"))
    with pytest.raises(FileExistsError):
        p.atomic_write_once(output, b"replacement")
    assert output.read_bytes() == raw


def test_atomic_write_once_has_exactly_one_complete_concurrent_winner(
    tmp_path: Path,
) -> None:
    output = tmp_path / "race" / "registration.json"
    contenders = [
        (f"contender-{index}-" + "x" * 10_000).encode("utf-8")
        for index in range(8)
    ]

    def attempt(raw: bytes) -> bool:
        try:
            p.atomic_write_once(output, raw)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=len(contenders)) as executor:
        winners = list(executor.map(attempt, contenders))
    assert sum(winners) == 1
    assert output.read_bytes() in contenders
    assert not list(output.parent.glob(".*.tmp-*"))


def test_encoded_registration_is_deterministic_and_round_trips() -> None:
    result = registration()
    first = p.encoded_registration(result)
    second = p.encoded_registration(copy.deepcopy(result))
    assert first == second
    assert json.loads(first) == result
    assert first.endswith(b"\n")


def test_hash_bound_artifact_access_is_header_only_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "clock.csv.gz"
    header_raw = b"entry_time,exit_time,side\n"
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as handle:
        handle.write(header_raw)
        handle.write(
            b"2023-01-01T00:00:00Z,2023-01-01T01:00:00Z,1\n"
        )
    spec = {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "header": ["entry_time", "exit_time", "side"],
        "header_line_sha256": hashlib.sha256(header_raw).hexdigest(),
    }
    monkeypatch.setattr(p, "SOURCE_ARTIFACTS", {"fixture": spec})
    monkeypatch.setattr(p, "PRIOR_VOLATILITY_COMPARATORS", {})

    real_open = p.gzip.open
    external_readlines = 0

    class HeaderOnly:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.context = real_open(*args, **kwargs)
            self.inner: Any = None

        def __enter__(self) -> "HeaderOnly":
            self.inner = self.context.__enter__()
            return self

        def readline(self) -> bytes:
            nonlocal external_readlines
            external_readlines += 1
            if external_readlines > 1:
                raise AssertionError("producer attempted a second decoded line")
            return self.inner.readline()

        def read(self, *args: Any, **kwargs: Any) -> bytes:
            raise AssertionError("producer attempted decoded row reads")

        def __exit__(self, *args: Any) -> Any:
            return self.context.__exit__(*args)

    monkeypatch.setattr(p.gzip, "open", HeaderOnly)
    binding = p.validate_hash_bound_artifacts(tmp_path)["fixture"]
    assert external_readlines == 1
    assert binding["rows_decoded"] == 0
    assert binding["sha256"] == spec["sha256"]

    wrong_hash = {**spec, "sha256": "0" * 64}
    monkeypatch.setattr(p, "SOURCE_ARTIFACTS", {"fixture": wrong_hash})
    with pytest.raises(RuntimeError, match="hash drift"):
        p.validate_hash_bound_artifacts(tmp_path)

    wrong_header = {**spec, "header": ["wrong"]}
    monkeypatch.setattr(p, "SOURCE_ARTIFACTS", {"fixture": wrong_header})
    external_readlines = 0
    with pytest.raises(RuntimeError, match="header drift"):
        p.validate_hash_bound_artifacts(tmp_path)

    missing = {**spec, "path": "missing.csv.gz"}
    monkeypatch.setattr(p, "SOURCE_ARTIFACTS", {"fixture": missing})
    with pytest.raises(RuntimeError, match="missing"):
        p.validate_hash_bound_artifacts(tmp_path)


def _write_esdi_reference(
    path: Path,
    *,
    authority: dict[str, Any],
    weights: dict[str, float] | None = None,
    preserve_manifest: str | None = None,
) -> tuple[str, str]:
    core = {
        "gross9": {
            "authority": copy.deepcopy(authority),
            "weights": copy.deepcopy(weights or p.GROSS9_WEIGHTS),
        }
    }
    manifest = preserve_manifest or p.canonical_hash(core)
    payload = {**core, "manifest_hash": manifest}
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest(), manifest


def test_load_gross9_authority_validates_outer_inner_and_roster_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = copy.deepcopy(fake_gross9()["authority"])
    path = tmp_path / "esdi.json"
    file_hash, manifest = _write_esdi_reference(path, authority=authority)
    reference = {
        "path": path.name,
        "sha256": file_hash,
        "manifest_hash": manifest,
    }
    monkeypatch.setattr(p, "ESDI_PREREGISTRATION", reference)
    assert p.load_gross9_authority(tmp_path)["authority_hash"] == (
        p.GROSS9_AUTHORITY_HASH
    )

    monkeypatch.setattr(
        p,
        "ESDI_PREREGISTRATION",
        {**reference, "sha256": "0" * 64},
    )
    with pytest.raises(RuntimeError, match="artifact drift"):
        p.load_gross9_authority(tmp_path)

    tampered_authority = copy.deepcopy(authority)
    tampered_authority["runtime_code_closure"][
        "all_distribution_inventory_count"
    ] = 107
    tampered_hash, _ = _write_esdi_reference(
        path,
        authority=tampered_authority,
        preserve_manifest=manifest,
    )
    monkeypatch.setattr(
        p,
        "ESDI_PREREGISTRATION",
        {**reference, "sha256": tampered_hash},
    )
    with pytest.raises(RuntimeError, match="manifest hash drift"):
        p.load_gross9_authority(tmp_path)

    changed_hash, changed_manifest = _write_esdi_reference(
        path,
        authority=tampered_authority,
    )
    monkeypatch.setattr(
        p,
        "ESDI_PREREGISTRATION",
        {
            **reference,
            "sha256": changed_hash,
            "manifest_hash": manifest,
        },
    )
    with pytest.raises(RuntimeError, match="manifest identity drift"):
        p.load_gross9_authority(tmp_path)

    roster = {**p.GROSS9_WEIGHTS, "extra": 1.0}
    roster_hash, roster_manifest = _write_esdi_reference(
        path,
        authority=authority,
        weights=roster,
    )
    monkeypatch.setattr(
        p,
        "ESDI_PREREGISTRATION",
        {
            "path": path.name,
            "sha256": roster_hash,
            "manifest_hash": roster_manifest,
        },
    )
    with pytest.raises(RuntimeError, match="roster drift"):
        p.load_gross9_authority(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("all_distribution_inventory_count", 107),
        ("ast_import_closure_must_match_before_artifact_creation", False),
        ("runtime_environment_must_match_before_artifact_creation", False),
    ],
)
def test_build_registration_rejects_each_incomplete_gross9_closure_gate(
    field: str,
    value: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gross9 = fake_gross9()
    gross9["authority"]["runtime_code_closure"][field] = value
    altered_hash = p.canonical_hash(gross9["authority"])
    gross9["authority_hash"] = altered_hash
    monkeypatch.setattr(p, "GROSS9_AUTHORITY_HASH", altered_hash)
    with pytest.raises(RuntimeError, match="authority closure drift"):
        p.build_registration(
            repository=fake_repository(),
            artifact_bindings=fake_bindings(),
            gross9=gross9,
        )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sealed_test_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    remote = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", p.EXPECTED_BRANCH)
    _git(repo, "config", "user.email", "cvvh@example.invalid")
    _git(repo, "config", "user.name", "CVVH Test")
    (repo / "protocol.txt").write_text("protocol\n", encoding="utf-8")
    policy = repo / "policy.md"
    policy.write_text("policy\n", encoding="utf-8")
    _git(repo, "add", "protocol.txt", "policy.md")
    _git(repo, "commit", "-m", "seal")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", p.EXPECTED_BRANCH)
    monkeypatch.setattr(p, "EXPECTED_ORIGIN_URL", str(remote))
    monkeypatch.setattr(p, "PROTOCOL_PATHS", ("protocol.txt", "policy.md"))
    monkeypatch.setattr(
        p,
        "COMMON_WINDOW_POLICY",
        {
            "path": "policy.md",
            "sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        },
    )
    return repo, remote


def test_repository_identity_requires_real_remote_clean_exact_complete_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _sealed_test_repository(tmp_path, monkeypatch)
    identity = p.repository_identity(repo)
    assert identity["commit"] == identity["upstream_commit"]
    assert identity["upstream_ref"].startswith("refs/remotes/origin/")
    assert set(identity["protocol_seal"]) == {"protocol.txt", "policy.md"}
    assert identity["protocol_seal_hash"] == p.canonical_hash(
        identity["protocol_seal"]
    )

    subdir = repo / "subdir"
    subdir.mkdir()
    with pytest.raises(RuntimeError, match="outside repository root"):
        p.repository_identity(subdir)

    junk = repo / "untracked.txt"
    junk.write_text("untracked\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="including untracked"):
        p.repository_identity(repo)
    junk.unlink()

    (repo / "protocol.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="including untracked"):
        p.repository_identity(repo)
    _git(repo, "checkout", "--", "protocol.txt")

    (repo / "later.txt").write_text("later\n", encoding="utf-8")
    _git(repo, "add", "later.txt")
    _git(repo, "commit", "-m", "unpushed")
    with pytest.raises(RuntimeError, match="upstream exact HEAD"):
        p.repository_identity(repo)
    _git(repo, "reset", "--hard", "@{u}")

    monkeypatch.setattr(p, "PROTOCOL_PATHS", ("missing.txt", "policy.md"))
    with pytest.raises(RuntimeError, match="protocol path missing"):
        p.repository_identity(repo)


def test_repository_identity_rejects_policy_hash_drift_on_clean_pushed_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _sealed_test_repository(tmp_path, monkeypatch)
    (repo / "policy.md").write_text("changed policy\n", encoding="utf-8")
    _git(repo, "add", "policy.md")
    _git(repo, "commit", "-m", "change policy")
    _git(repo, "push")
    with pytest.raises(RuntimeError, match="common-window policy hash drift"):
        p.repository_identity(repo)


def test_repository_identity_rejects_wrong_origin_url_and_local_remote_spoof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _sealed_test_repository(tmp_path, monkeypatch)
    expected_url = p.EXPECTED_ORIGIN_URL
    monkeypatch.setattr(p, "EXPECTED_ORIGIN_URL", "/tmp/not-the-origin.git")
    with pytest.raises(RuntimeError, match="origin URL set drift"):
        p.repository_identity(repo)
    monkeypatch.setattr(p, "EXPECTED_ORIGIN_URL", expected_url)

    _git(repo, "config", f"branch.{p.EXPECTED_BRANCH}.remote", ".")
    with pytest.raises(RuntimeError, match="branch remote origin"):
        p.repository_identity(repo)


def test_repository_identity_rejects_pushurl_and_forged_tracking_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _ = _sealed_test_repository(tmp_path, monkeypatch)
    alternate = tmp_path / "alternate.git"
    subprocess.run(
        ["git", "init", "--bare", str(alternate)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo, "remote", "set-url", "--add", "--push", "origin", str(alternate))
    with pytest.raises(RuntimeError, match="origin URL set drift"):
        p.repository_identity(repo)
    _git(repo, "config", "--unset-all", "remote.origin.pushurl")

    (repo / "forged.txt").write_text("not pushed\n", encoding="utf-8")
    _git(repo, "add", "forged.txt")
    _git(repo, "commit", "-m", "forge local tracking")
    head = _git(repo, "rev-parse", "HEAD")
    _git(
        repo,
        "update-ref",
        f"refs/remotes/origin/{p.EXPECTED_BRANCH}",
        head,
    )
    with pytest.raises(RuntimeError, match="canonical remote branch"):
        p.repository_identity(repo)


def test_producer_ast_has_no_network_dataframe_or_outcome_dependency() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "training"
        / "preregister_cross_venue_volatility_shape_handoff.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports.isdisjoint(
        {
            "aiohttp",
            "httpx",
            "numpy",
            "pandas",
            "requests",
            "urllib",
            "websockets",
        }
    )
    text = source_path.read_text(encoding="utf-8")
    assert "read_csv(" not in text
    assert "read_parquet(" not in text


def test_module_reload_never_creates_or_changes_default_artifact() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / p.DEFAULT_OUTPUT
    before = output.read_bytes() if output.exists() else None
    importlib.reload(p)
    after = output.read_bytes() if output.exists() else None
    assert after == before
