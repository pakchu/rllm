from __future__ import annotations

import gzip
import json

import pytest

from training import preregister_cboe_cross_surface_risk_transfer as p


def test_manifest_is_composite_incidence_comparator_and_outcome_blind() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["policy"]["policy_id"] == "CXRT-288"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["source_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
    assert payload["research_history_boundary"][
        "exact_cxrt_common_state_or_candidate_incidence_seen"
    ] is False


def test_source_contracts_use_exact_allowlists_and_no_fill() -> None:
    source = p.build_manifest()["source_contracts"]
    assert source["term"]["allowlist"] == list(p.TERM_ALLOWLIST)
    assert source["tail"]["allowlist"] == list(p.TAIL_ALLOWLIST)
    assert source["option"]["allowlist"] == list(p.OPTION_ALLOWLIST)
    assert all(
        contract["loader"].startswith("pandas.read_csv(usecols=")
        for contract in (
            source["term"],
            source["tail"],
            source["option"],
        )
    )
    assert source["exact_date_join"] == (
        "intersection after independent causal features"
    )
    assert "no fill" in source["missing_policy"]
    validation = source["row_validation"]
    assert validation["dates"] == (
        "unique and strictly increasing within each panel"
    )
    assert "finite and strictly positive" in validation[
        "numeric_primitives"
    ]
    assert validation["invalid_primitive_action"].startswith("fail the source")
    assert validation["pre_2024_only"] is True


def test_rank_surface_vote_and_execution_are_frozen() -> None:
    payload = p.build_manifest()
    rank = payload["rank_contract"]
    algebra = payload["surface_algebra"]
    vote = payload["vote_contract"]
    execution = payload["execution_contract"]
    assert rank["lookback"] == 252
    assert rank["minimum"] == 126
    assert rank["current_appended_after_rank"] is True
    assert algebra["tail_vix_subtraction"] is False
    assert algebra["tail_second_layer_rank"] is False
    assert "one-source-observation deltas" in algebra["option_pressure"]
    assert vote["eligible"] == (
        "at least two nonzero votes and vote_sum != 0"
    )
    assert vote["fitted_weights"] is False
    assert vote["tail_threshold"] is None
    assert execution["entry_date"] == (
        "first later exact common CBOE source date D_next"
    )
    assert execution["exit"] == "entry + exactly 288*5m"
    assert execution["global_nonoverlap_before_split"] is True


def test_controls_support_and_composition_gates_are_frozen() -> None:
    payload = p.build_manifest()
    controls = payload["source_only_controls"]
    gate = payload["source_support_gate"]
    composition = gate["composition"]
    assert controls["ordered"] == list(p.CONTROL_ORDER)
    definitions = controls["definitions"]
    assert definitions["primary"] == "three-surface majority"
    assert definitions["term_tail_agreement"].endswith(
        "agree non-neutrally"
    )
    assert "immediately preceding rank-complete common-date votes" in (
        definitions["one_common_date_stale"]
    )
    assert "shift entry and exit exactly 288 bars" in (
        definitions["one_day_execution_delay"]
    )
    assert "recompute global overlap and split containment" in (
        definitions["one_day_execution_delay"]
    )
    assert gate["train_events_min"] == 400
    assert gate["each_train_year_events_min"] == 190
    assert gate["selection_events_min"] == 190
    assert gate["selection_each_quarter_events_min"] == 40
    assert composition["each_surface_each_vote_share_min"] == 0.15
    assert composition["each_surface_unique_minority_share_min"] == 0.08
    assert composition["unanimous_share_range"] == [0.10, 0.80]
    assert composition["single_surface_same_side_reproduction_max"] == 0.80
    assert composition["stale_same_side_reproduction_max"] == 0.85
    assert composition["random_same_side_reproduction_max"] == 0.60


def test_comparator_and_rllm_contracts_are_fail_closed() -> None:
    payload = p.build_manifest()
    novelty = payload["novelty_contract"]
    comparators = {item["id"]: item for item in novelty["comparators"]}
    assert list(comparators) == ["CVTR-1", "CTHD-1", "CIHM-1"]
    assert comparators["CVTR-1"]["selected_groups"] == [
        "primary",
        "deterministic_random_side",
        "constant_long",
    ]
    assert novelty["exact_entry_jaccard_max"] == 0.45
    assert novelty["same_entry_same_side_reproduction_max"] == 0.75
    assert novelty["absolute_signed_occupancy_pearson_max"] == 0.60
    assert "empty required common-coverage extraction" in novelty[
        "failure_conditions"
    ]
    assert "undefined/nonfinite signed-exposure correlation" in novelty[
        "failure_conditions"
    ]
    rllm = payload["rllm_boundary"]
    assert rllm["action_space"] == ["TRADE_FIXED_SIDE", "ABSTAIN"]
    assert "raw_numeric_values_or_ranks" in rllm["forbidden"]
    assert "date_year_month_weekday_timestamp_or_row_identity" in rllm[
        "forbidden"
    ]
    assert "surface_vote_transitions" in rllm["allowed_tokens"]


def test_frozen_hashes_and_headers_match_without_decoding_rows() -> None:
    p.validate_frozen_dependencies()
    assert p.sha256_csv_header(p.TERM_SOURCE) == p.TERM_HEADER_SHA256
    assert p.sha256_csv_header(p.TAIL_SOURCE) == p.TAIL_HEADER_SHA256
    assert p.sha256_csv_header(p.OPTION_SOURCE) == p.OPTION_HEADER_SHA256
    assert set(p.TERM_ALLOWLIST).issubset(p.csv_header(p.TERM_SOURCE))
    assert set(p.TAIL_ALLOWLIST).issubset(p.csv_header(p.TAIL_SOURCE))
    assert set(p.OPTION_ALLOWLIST).issubset(p.csv_header(p.OPTION_SOURCE))
    assert len(p.frozen_dependencies()) == 11


def test_header_reader_does_not_decode_later_rows(tmp_path) -> None:
    plain = tmp_path / "clock.csv"
    plain.write_bytes(b"a,b\n\xff\xfe\x00not-csv")
    assert p.csv_header(plain) == ["a", "b"]

    compressed = tmp_path / "clock.csv.gz"
    with gzip.open(compressed, "wb") as handle:
        handle.write(b"x,y\n\xff\xfe\x00not-csv")
    assert p.csv_header(compressed) == ["x", "y"]


def test_write_once_is_reproducible_and_rejects_drift(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        p,
        "validate_frozen_dependencies",
        lambda: calls.append(True),
    )
    payload = p.build_manifest()
    output = tmp_path / "freeze.json"
    assert p.write_once(output, payload) == "created"
    assert calls == [True]
    assert output.read_text(encoding="utf-8") == p._canonical_manifest_text()
    assert p.write_once(output, p.build_manifest()) == "verified_existing"
    stored = json.loads(output.read_text())
    stored["policy"]["hold_bars"] = 287
    with pytest.raises(RuntimeError, match="core differs from code"):
        p.validate_manifest(stored)


def test_validate_manifest_rejects_self_rehashed_drift() -> None:
    payload = p.build_manifest()
    payload["novelty_contract"]["exact_entry_jaccard_max"] = 0.46
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="core differs from code"):
        p.validate_manifest(payload)
