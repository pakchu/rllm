from __future__ import annotations

import gzip
import json
from datetime import date, datetime, timezone
import hashlib

import pytest

from training import preregister_cboe_option_pressure_rank_rotation as p


def test_manifest_is_candidate_comparator_and_outcome_blind() -> None:
    payload = p.build_manifest()
    p.validate_manifest(payload)
    assert payload["policy"]["policy_id"] == "OPRR-288"
    assert payload["outcomes_opened"] is False
    assert payload["source_incidence_opened"] is False
    assert payload["source_rows_decoded"] is False
    assert payload["comparator_rows_decoded"] is False
    history = payload["research_history_boundary"]
    assert history["oprr_reserved_before_dclb_and_scaf_incidence"] is True
    assert history["dclb_or_scaf_incidence_used_to_define_oprr"] is False
    assert history["exact_oprr_ordinal_state_or_candidate_incidence_seen"] is False


def test_source_rank_and_surface_contracts_are_causal() -> None:
    payload = p.build_manifest()
    source = payload["source_contracts"]
    assert source["term"]["allowlist"] == list(p.TERM_ALLOWLIST)
    assert source["tail"]["allowlist"] == list(p.TAIL_ALLOWLIST)
    assert source["option"]["allowlist"] == list(p.OPTION_ALLOWLIST)
    assert all(
        item["loader"].startswith("pandas.read_csv(usecols=")
        for item in (source["term"], source["tail"], source["option"])
    )
    assert source["exact_date_join"] == (
        "intersection after independent causal features"
    )
    assert "no fill" in source["missing_policy"]
    rank = payload["rank_contract"]
    assert rank["lookback"] == 252
    assert rank["minimum"] == 126
    assert rank["current_appended_after_rank"] is True
    assert rank["future_normalization"] is False
    algebra = payload["surface_algebra"]
    assert algebra["tail_vix_subtraction"] is False
    assert algebra["tail_second_layer_rank"] is False
    assert "immediately previous option-source" in algebra["option_pressure"]


def test_ordinal_transition_side_and_fail_flat_rules_are_exact() -> None:
    payload = p.build_manifest()
    ordinal = payload["ordinal_state_contract"]
    transition = payload["transition_contract"]
    assert ordinal["requires_pairwise_distinct_pressures"] is True
    assert ordinal["option_position"] == (
        "1{term_pressure<option_pressure}+1{tail_pressure<option_pressure}"
    )
    assert ordinal["positions"] == {
        "0": "BELOW", "1": "MIDDLE", "2": "ABOVE"
    }
    assert "never skip" in ordinal["tie_action"]
    assert transition["rotation"] == (
        "option_position[t]-option_position[t-1]"
    )
    assert transition["eligible"] == [
        "rotation != 0",
        "sign(delta_option_pressure) == sign(rotation)",
        "sign(delta_term_pressure) == sign(rotation)",
        "sign(delta_tail_pressure) == sign(rotation)",
    ]
    assert transition["side"] == {
        "rotation_positive": "SHORT", "rotation_negative": "LONG"
    }
    assert transition["numeric_threshold"] is None
    assert transition["btc_or_calendar_regime_gate"] is None


def test_session_calendar_cannot_use_future_source_row_membership() -> None:
    payload = p.build_manifest()
    calendar = payload["session_calendar_contract"]
    execution = payload["execution_contract"]
    assert calendar["future_source_row_membership_used"] is False
    assert execution["entry_date"] == (
        "first later prospective regular CBOE session S_next"
    )
    assert execution["future_session_source_row"] == (
        "cannot create suppress or reschedule entry"
    )
    closures = [date.fromisoformat(value) for value in p.SESSION_CLOSURES]
    assert len(closures) == 47
    assert len(closures) == len(set(closures))
    assert all(value.weekday() < 5 for value in closures)
    assert calendar["full_day_closures"] == list(p.SESSION_CLOSURES)
    assert execution["exposure_interval"] == "[entry,exit)"
    assert execution["split_containment"] == "entry>=start and exit<=end"


def test_controls_random_bytes_and_support_gates_are_frozen() -> None:
    payload = p.build_manifest()
    controls = payload["source_only_controls"]
    support = payload["source_support_gate"]
    composition = payload["rotation_composition_gate"]
    assert controls["ordered"] == list(p.CONTROL_ORDER)
    assert controls["definitions"]["primary"].startswith("exact OPRR")
    assert "no term/tail deltas" in controls["definitions"][
        "option_own_confirmed"
    ]
    random_side = controls["random_side"]
    assert random_side["canonical_entry_utc"] == "YYYY-MM-DDTHH:MM:SSZ"
    assert random_side["message"] == (
        "ASCII bytes b'OPRR-288|'+canonical_entry_utc"
    )
    canonical = datetime(2023, 1, 3, 14, 35, tzinfo=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    digest = hashlib.sha256(b"OPRR-288|" + canonical.encode("ascii")).digest()
    assert digest[0] == 31
    assert support["train_events_min"] == 100
    assert support["each_train_year_events_min"] == 40
    assert support["selection_events_min"] == 45
    assert support["selection_each_quarter_events_min"] == 6
    assert composition["two_step_share_min"] == 0.10
    assert composition["raw_primary_retention_within_option_own_max"] == 0.75
    assert "before reservation" in composition["raw_retention_formula"]
    assert composition["sponsor_exact_entry_jaccard_max"] == 0.65


def test_comparator_novelty_metrics_are_fully_defined_and_fail_closed() -> None:
    novelty = p.build_manifest()["novelty_contract"]
    comparators = {item["id"]: item for item in novelty["comparators"]}
    assert list(comparators) == ["CXRT-288", "CVTR-1", "CTHD-1", "CIHM-1"]
    assert comparators["CXRT-288"]["selected_groups"] == [
        "primary", "term_only", "tail_only", "option_only",
        "term_tail_agreement", "one_common_date_stale",
        "exact_direction_flip", "deterministic_random_side",
        "one_day_execution_delay",
    ]
    assert novelty["entry_inclusion"] == (
        "start<=entry<end after UTC normalization"
    )
    assert novelty["exact_entry_jaccard"] == (
        "|A intersect B|/|A union B|"
    )
    assert novelty["exact_entry_jaccard_max"] == 0.35
    assert novelty["same_entry_same_side_reproduction_max"] == 0.80
    tolerant = novelty["tolerant_entry_jaccard"]
    assert tolerant["status"] == "report_only"
    assert "order-preserving DP maximum cardinality" in tolerant["matching"]
    assert tolerant["formula"] == "m/(len(A)+len(B)-m)"
    occupancy = novelty["signed_occupancy"]
    assert "five-minute left endpoints" in occupancy["grid"]
    assert occupancy["position_interval"] == "[entry,exit)"
    assert occupancy["zero_variance"] == "fail"
    assert novelty["absolute_signed_occupancy_pearson_max"] == 0.55
    assert novelty["duplicate_entry_or_empty_group"] == "fail"


def test_rllm_contract_keeps_llm_on_fixed_side_abstention() -> None:
    rllm = p.build_manifest()["rllm_boundary"]
    assert rllm["action_space"] == ["TRADE_FIXED_SIDE", "ABSTAIN"]
    assert "prior_option_position" in rllm["allowed_tokens"]
    assert "term_confirmation" in rllm["allowed_tokens"]
    assert "raw_numeric_values_or_ranks" in rllm["forbidden"]
    assert "date_year_month_weekday_timestamp_or_row_identity" in rllm[
        "forbidden"
    ]


def test_frozen_hashes_and_headers_match_without_decoding_rows() -> None:
    p.validate_frozen_dependencies()
    assert p.sha256_csv_header(p.TERM_SOURCE) == p.TERM_HEADER_SHA256
    assert p.sha256_csv_header(p.TAIL_SOURCE) == p.TAIL_HEADER_SHA256
    assert p.sha256_csv_header(p.OPTION_SOURCE) == p.OPTION_HEADER_SHA256
    assert set(p.TERM_ALLOWLIST).issubset(p.csv_header(p.TERM_SOURCE))
    assert set(p.TAIL_ALLOWLIST).issubset(p.csv_header(p.TAIL_SOURCE))
    assert set(p.OPTION_ALLOWLIST).issubset(p.csv_header(p.OPTION_SOURCE))
    assert len(p.frozen_dependencies()) == 12


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
        p, "validate_frozen_dependencies", lambda: calls.append(True)
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
    payload["novelty_contract"]["exact_entry_jaccard_max"] = 0.36
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    payload["manifest_hash"] = p.canonical_hash(core)
    with pytest.raises(RuntimeError, match="core differs from code"):
        p.validate_manifest(payload)
