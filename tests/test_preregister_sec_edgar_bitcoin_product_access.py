from __future__ import annotations

from dataclasses import replace

import pytest

import training.preregister_sec_edgar_bitcoin_product_access as bpax

from training.preregister_sec_edgar_bitcoin_product_access import (
    MODEL_ID,
    MODEL_REVISION,
    SYNTHETIC_CASES,
    Config,
    _validate_frozen_config,
    aggregate_window_classes,
    breadth_crossings,
    build_artifact,
    directional_filings,
    parse_model_output,
    redact_excerpt,
    reserve_nonoverlap,
    semantic_contract,
)


def test_redaction_preserves_access_fact_but_removes_identity_time_and_number() -> None:
    source = (
        "Example Payments Inc. (Nasdaq: EXAM) enabled 125 customers to buy Bitcoin "
        "on January 3, 2023; CIK 1234567890."
    )
    redacted = redact_excerpt(
        source,
        issuer_aliases=["Example Payments Inc."],
        issuer_tickers=["EXAM"],
    )
    assert "Example" not in redacted
    assert "EXAM" not in redacted
    assert "1234567890" not in redacted
    assert "2023" not in redacted
    assert "125" not in redacted
    assert "customers to buy Bitcoin" in redacted
    assert "[ENTITY]" in redacted
    assert "[TICKER]" in redacted
    assert "[NUM]" in redacted
    assert "[DATE]" in redacted


def test_model_output_is_two_key_strict_json_and_quote_grounded() -> None:
    excerpt = "[ENTITY] enabled customers to buy and sell Bitcoin."
    output = (
        '{"class":"BTC_ACCESS_EXPANSION",'
        '"quote":"enabled customers to buy and sell Bitcoin"}'
    )
    assert parse_model_output(output, excerpt) == {
        "class": "BTC_ACCESS_EXPANSION",
        "quote": "enabled customers to buy and sell Bitcoin",
    }
    assert (
        parse_model_output(
            '{"quote":"enabled customers to buy and sell Bitcoin",'
            '"class":"BTC_ACCESS_EXPANSION"}',
            excerpt,
        )
        is None
    )
    assert parse_model_output(output.replace("enabled", "invented"), excerpt) is None
    assert (
        parse_model_output(
            '{"class":"UNSUPPORTED","class":"BTC_ACCESS_EXPANSION",'
            '"quote":"enabled customers to buy and sell Bitcoin"}',
            excerpt,
        )
        is None
    )
    assert (
        parse_model_output('{"class":"UNSUPPORTED","quote":"evidence"}', excerpt)
        is None
    )
    assert parse_model_output('{"class":"UNSUPPORTED","quote":""}', excerpt) == {
        "class": "UNSUPPORTED",
        "quote": "",
    }


def test_accession_aggregation_fails_closed_on_mixed_or_empty() -> None:
    assert (
        aggregate_window_classes(["UNSUPPORTED", "BTC_ACCESS_EXPANSION"])
        == "BTC_ACCESS_EXPANSION"
    )
    assert (
        aggregate_window_classes(["BTC_ACCESS_EXPANSION", "BTC_ACCESS_RETRACTION"])
        == "MIXED_OR_UNSUPPORTED"
    )
    assert aggregate_window_classes(["UNSUPPORTED"]) == "MIXED_OR_UNSUPPORTED"


def _row(
    when: str,
    accession: str,
    cik: str,
    label: str,
) -> dict[str, object]:
    return {
        "acceptance_datetime": when,
        "accession": accession,
        "ciks": [cik],
        "filing_class": label,
    }


def test_directional_filings_apply_smallest_cik_and_nonresetting_cooldown() -> None:
    rows = [
        _row("2021-02-01T00:00:00Z", "kept-late", "9", "BTC_ACCESS_RETRACTION"),
        {
            **_row(
                "2021-01-01T00:00:00Z",
                "kept-first",
                "10",
                "BTC_ACCESS_EXPANSION",
            ),
            "ciks": ["10", "9"],
        },
        _row("2021-01-20T00:00:00Z", "skipped", "9", "BTC_ACCESS_RETRACTION"),
        _row("2021-02-20T00:00:00Z", "unsupported", "9", "UNSUPPORTED"),
    ]
    events = directional_filings(rows)
    assert [event["accession"] for event in events] == ["kept-first", "kept-late"]
    assert all(event["issuer_key"] == "0000000009" for event in events)
    assert events[0]["ready_datetime"] == "2021-01-01T01:00:00+00:00"


def _filing(when: str, accession: str, issuer: str, label: str) -> dict[str, str]:
    return {
        "ready_datetime": when,
        "accession": accession,
        "issuer_key": issuer,
        "filing_class": label,
    }


def test_breadth_emits_only_on_signed_crossing_with_four_total_issuers() -> None:
    rows = [
        _filing("2021-01-01T00:01:00Z", "a", "1", "BTC_ACCESS_EXPANSION"),
        _filing("2021-01-02T00:02:00Z", "b", "2", "BTC_ACCESS_EXPANSION"),
        _filing("2021-01-03T00:03:00Z", "c", "3", "BTC_ACCESS_EXPANSION"),
        _filing("2021-01-04T00:04:00Z", "d", "4", "BTC_ACCESS_RETRACTION"),
        _filing("2021-01-05T00:06:00Z", "e", "5", "BTC_ACCESS_EXPANSION"),
        _filing("2021-01-06T00:07:00Z", "f", "6", "BTC_ACCESS_EXPANSION"),
    ]
    signals = breadth_crossings(rows)
    assert signals == [
        {
            "resolved_datetime": "2021-01-05T00:06:00+00:00",
            "entry_earliest": "2021-01-05T00:15:00+00:00",
            "exit_earliest": "2021-01-10T00:15:00+00:00",
            "side": 1,
            "resolved_class": "BTC_ACCESS_EXPANSION",
            "score_before": 2,
            "score_after": 3,
            "expansion_issuers": 4,
            "retraction_issuers": 1,
            "total_issuers": 5,
        }
    ]


def test_four_same_side_issuers_emit_when_minimum_total_becomes_eligible() -> None:
    rows = [
        _filing("2021-01-01T00:00:00Z", "a", "1", "BTC_ACCESS_EXPANSION"),
        _filing("2021-01-02T00:00:00Z", "b", "2", "BTC_ACCESS_EXPANSION"),
        _filing("2021-01-03T00:00:00Z", "c", "3", "BTC_ACCESS_EXPANSION"),
        _filing("2021-01-04T00:00:00Z", "d", "4", "BTC_ACCESS_EXPANSION"),
    ]
    signals = breadth_crossings(rows)
    assert len(signals) == 1
    assert signals[0]["side"] == 1
    assert signals[0]["score_before"] == 3
    assert signals[0]["score_after"] == 4
    assert signals[0]["total_issuers"] == 4


def test_fractional_ready_time_never_enters_before_ready_plus_delay() -> None:
    rows = [
        _filing("2021-01-01T00:00:00Z", "a", "1", "BTC_ACCESS_EXPANSION"),
        _filing("2021-01-02T00:00:00Z", "b", "2", "BTC_ACCESS_EXPANSION"),
        _filing("2021-01-03T00:00:00Z", "c", "3", "BTC_ACCESS_EXPANSION"),
        _filing("2021-01-04T00:00:00.001Z", "d", "4", "BTC_ACCESS_EXPANSION"),
    ]
    signals = breadth_crossings(rows)
    assert signals[0]["resolved_datetime"] == "2021-01-04T00:00:00.001000+00:00"
    assert signals[0]["entry_earliest"] == "2021-01-04T00:10:00+00:00"


def test_breadth_short_crossing_and_expiry_boundary_are_causal() -> None:
    rows = [
        _filing("2021-01-01T00:00:00Z", "a", "1", "BTC_ACCESS_EXPANSION"),
        _filing("2021-01-02T00:00:00Z", "b", "2", "BTC_ACCESS_RETRACTION"),
        _filing("2021-01-03T00:00:00Z", "c", "3", "BTC_ACCESS_RETRACTION"),
        _filing("2021-01-04T00:00:00Z", "d", "4", "BTC_ACCESS_RETRACTION"),
        _filing("2021-01-05T00:00:00Z", "e", "5", "BTC_ACCESS_RETRACTION"),
        _filing("2021-01-15T00:00:00Z", "f", "6", "BTC_ACCESS_EXPANSION"),
    ]
    signals = breadth_crossings(rows)
    assert len(signals) == 1
    assert signals[0]["side"] == -1
    assert signals[0]["score_before"] == -2
    assert signals[0]["score_after"] == -3
    assert signals[0]["resolved_datetime"] == "2021-01-05T00:00:00+00:00"


def test_breadth_rejects_missing_issuer_cooldown() -> None:
    rows = [
        _filing("2021-01-01T00:00:00Z", "a", "1", "BTC_ACCESS_EXPANSION"),
        _filing("2021-01-02T00:00:00Z", "b", "1", "BTC_ACCESS_RETRACTION"),
    ]
    with pytest.raises(ValueError, match="cooldown"):
        breadth_crossings(rows)


def test_nonoverlap_reserves_entry_at_prior_exit() -> None:
    events = [
        {
            "entry_earliest": "2021-01-01T00:00:00Z",
            "exit_earliest": "2021-01-06T00:00:00Z",
            "side": 1,
        },
        {
            "entry_earliest": "2021-01-02T00:00:00Z",
            "exit_earliest": "2021-01-07T00:00:00Z",
            "side": -1,
        },
        {
            "entry_earliest": "2021-01-06T00:00:00Z",
            "exit_earliest": "2021-01-11T00:00:00Z",
            "side": -1,
        },
    ]
    assert [row["side"] for row in reserve_nonoverlap(events)] == [1, -1]


def test_synthetic_cases_are_literal_guarded_and_swap_invariant() -> None:
    contract = semantic_contract(Config())
    frozen = contract["synthetic_controls"]["cases"]
    assert len(frozen) == len(SYNTHETIC_CASES) == 24
    guarded = [row for row in frozen if row["guarded"]]
    assert {row["name"] for row in guarded} == {
        "expansion_prompt_injection",
        "retraction_prompt_injection",
    }
    equivalents = [
        row["redacted_excerpt"]
        for row in frozen
        if row.get("equivalence_group") == "entity_product_date_amount_swap"
    ]
    assert len(equivalents) == 2
    assert equivalents[0] == equivalents[1]


def test_frozen_contract_keeps_model_memory_outcomes_and_future_sealed() -> None:
    cfg = Config()
    _validate_frozen_config(cfg)
    with pytest.raises(ValueError, match="configuration is frozen"):
        _validate_frozen_config(replace(cfg, hold_hours=72))
    contract = semantic_contract(cfg)
    assert contract["model"]["id"] == MODEL_ID
    assert contract["model"]["revision"] == MODEL_REVISION
    assert contract["model"]["fine_tuned"] is False
    assert contract["synthetic_gate"]["maximum_peak_allocated_bytes"] == 7 * 1024**3
    assert contract["splits"]["sealed_eval"].startswith("2024-01-01")
    assert contract["learning_boundary"]["rl_or_lora_authorized"] is False
    prompt_tail = contract["prompt"].lower().split("redacted sec excerpt:")[-1]
    assert "return" not in prompt_tail


def test_preregistration_opens_no_body_model_market_or_future_rows() -> None:
    artifact = build_artifact(Config(), verify_model=False)
    boundary = artifact["outcome_boundary"]
    assert boundary["filing_bodies_opened"] == 0
    assert boundary["semantic_model_calls"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["comparator_anchor_files_hashed"] == 16
    assert boundary["comparator_rows_parsed"] == 0
    assert boundary["comparator_clock_fields_read"] == 0
    assert boundary["2024_or_later_source_rows_read"] == 0
    assert artifact["decision"]["synthetic_model_gate_authorized"] is True
    assert artifact["decision"]["historical_semantic_execution_authorized"] is False
    assert artifact["decision"]["economic_evaluation_authorized"] is False


def test_comparator_preregistration_reads_only_frozen_raw_byte_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        str(path): digest
        for collection in (bpax.COMPARATOR_CLOCKS, bpax.MECHANISM_NEGATIVE_CONTROLS)
        for path, digest in collection.values()
    }
    observed: list[str] = []

    def fake_hash(path: str | bpax.Path) -> str:
        key = str(path)
        observed.append(key)
        return expected[key]

    monkeypatch.setattr(bpax, "sha256_file", fake_hash)
    audit = bpax._validate_comparator_anchors()
    assert observed == list(expected)
    assert len(audit) == 16
    assert all(row["read_mode"] == "raw bytes for SHA-256 only" for row in audit)
    assert all(row["rows_parsed"] == row["fields_read"] == 0 for row in audit)
