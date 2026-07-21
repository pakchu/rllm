from __future__ import annotations

from dataclasses import replace

import pytest

from training.preregister_sec_edgar_bitcoin_constraint_transition_breadth import (
    MODEL_ID,
    MODEL_REVISION,
    SYNTHETIC_CASES,
    Config,
    _validate_frozen_config,
    aggregate_window_labels,
    breadth_events,
    build_artifact,
    parse_model_output,
    raw_state_transitions,
    redact_excerpt,
    reserve_nonoverlap,
    semantic_contract,
)


def test_redaction_removes_identity_time_and_numbers_but_preserves_asset_action() -> None:
    source = (
        "Example Mining Holdings Inc. (Nasdaq: MSTR) sold 125 BTC for $7,500,000 on "
        "January 3, 2023; CIK 1234567890; contact ceo@example.com at "
        "https://example.com/filing."
    )
    redacted = redact_excerpt(
        source,
        issuer_aliases=["Example Mining Holdings Inc."],
        issuer_tickers=["MSTR"],
    )
    assert "Example" not in redacted
    assert "1234567890" not in redacted
    assert "2023" not in redacted
    assert "7,500,000" not in redacted
    assert "example.com" not in redacted
    assert "MSTR" not in redacted
    assert "Nasdaq" not in redacted
    assert "BTC" in redacted
    assert "sold" in redacted
    assert "[ENTITY]" in redacted
    assert "[NUM] BTC" in redacted


def test_model_output_is_strict_json_and_quote_grounded() -> None:
    excerpt = "[ENTITY] sold [NUM] BTC and used the proceeds for working capital."
    output = (
        '{"label":"BTC_CONSTRAINT_DRAW","role":"BTC_SALE",'
        '"quote":"sold [NUM] BTC"}'
    )
    assert parse_model_output(output, excerpt) == {
        "label": "BTC_CONSTRAINT_DRAW",
        "role": "BTC_SALE",
        "quote": "sold [NUM] BTC",
    }
    assert parse_model_output(output.replace("BTC_SALE", "BTC_RETENTION"), excerpt) is None
    assert parse_model_output(output.replace("sold [NUM] BTC", "invented quote"), excerpt) is None
    assert parse_model_output(
        '{"role":"BTC_SALE","label":"BTC_CONSTRAINT_DRAW","quote":"sold [NUM] BTC"}',
        excerpt,
    ) is None
    assert parse_model_output(
        '{"label":"UNSUPPORTED","role":"NONE","quote":""}', excerpt
    ) == {"label": "UNSUPPORTED", "role": "NONE", "quote": ""}


@pytest.mark.parametrize(
    "source",
    [
        "Strategy Inc. (NYSE American: MSTR) sold 100 BTC.",
        "Strategy Inc. (NYSE American under the symbol MSTR) sold 100 BTC.",
    ],
)
def test_exchange_labeled_ticker_redaction_prefers_longest_exchange(source: str) -> None:
    redacted = redact_excerpt(source, issuer_aliases=["Strategy Inc."])
    assert "MSTR" not in redacted
    assert "American" not in redacted
    assert "[TICKER]" in redacted


def test_accession_aggregation_fails_closed_on_mixed_or_empty() -> None:
    assert aggregate_window_labels(["UNSUPPORTED", "BTC_CONSTRAINT_DRAW"]) == (
        "BTC_CONSTRAINT_DRAW"
    )
    assert aggregate_window_labels(
        ["BTC_CONSTRAINT_DRAW", "BTC_CONSTRAINT_BUFFER"]
    ) == "MIXED_OR_UNSUPPORTED"
    assert aggregate_window_labels(["UNSUPPORTED"]) == "MIXED_OR_UNSUPPORTED"


def _row(
    when: str, accession: str, cik: str, label: str
) -> dict[str, object]:
    return {
        "acceptance_datetime": when,
        "accession": accession,
        "ciks": [cik],
        "filing_label": label,
    }


def test_state_transition_then_three_issuer_breadth_resolves_causally() -> None:
    rows = [
        _row("2020-01-01T00:00:00Z", "a0", "1", "BTC_CONSTRAINT_DRAW"),
        _row("2020-01-01T00:01:00Z", "b0", "2", "BTC_CONSTRAINT_DRAW"),
        _row("2020-01-01T00:02:00Z", "c0", "3", "BTC_CONSTRAINT_DRAW"),
        _row("2021-01-01T00:01:10Z", "a1", "1", "BTC_CONSTRAINT_BUFFER"),
        _row("2021-01-02T00:02:20Z", "b1", "2", "BTC_CONSTRAINT_BUFFER"),
        _row("2021-01-03T00:03:30Z", "c1", "3", "BTC_CONSTRAINT_BUFFER"),
    ]
    transitions = raw_state_transitions(rows)
    assert len(transitions) == 3
    events = breadth_events(transitions)
    assert events == [
        {
            "resolved_datetime": "2021-01-03T00:18:30+00:00",
            "entry_earliest": "2021-01-03T00:25:00+00:00",
            "exit_earliest": "2021-01-06T00:25:00+00:00",
            "side": 1,
            "resolved_label": "BTC_CONSTRAINT_BUFFER",
            "supporting_issuers": 3,
            "opposite_issuers": 0,
        }
    ]


def test_state_machine_sorts_input_and_uses_smallest_numeric_cofiler_cik() -> None:
    rows = [
        _row("2021-01-01T00:00:00Z", "later", "9", "BTC_CONSTRAINT_BUFFER"),
        {
            **_row("2020-01-01T00:00:00Z", "initial", "10", "BTC_CONSTRAINT_DRAW"),
            "ciks": ["10", "9"],
        },
    ]
    transitions = raw_state_transitions(rows)
    assert len(transitions) == 1
    assert transitions[0]["issuer_key"] == "0000000009"
    assert transitions[0]["from_label"] == "BTC_CONSTRAINT_DRAW"
    assert transitions[0]["to_label"] == "BTC_CONSTRAINT_BUFFER"


def test_breadth_counts_one_transition_per_issuer_in_an_episode() -> None:
    transitions = [
        {
            "ready_datetime": "2021-01-01T00:00:00Z",
            "issuer_key": "1",
            "to_label": "BTC_CONSTRAINT_DRAW",
        },
        {
            "ready_datetime": "2021-01-02T00:00:00Z",
            "issuer_key": "1",
            "to_label": "BTC_CONSTRAINT_DRAW",
        },
        {
            "ready_datetime": "2021-01-03T00:00:00Z",
            "issuer_key": "2",
            "to_label": "BTC_CONSTRAINT_DRAW",
        },
    ]
    assert breadth_events(transitions) == []


def test_nonoverlap_reservation_uses_entry_and_prior_accepted_exit() -> None:
    events = [
        {"entry_earliest": "2021-01-01T00:00:00Z", "exit_earliest": "2021-01-04T00:00:00Z", "side": 1},
        {"entry_earliest": "2021-01-02T00:00:00Z", "exit_earliest": "2021-01-05T00:00:00Z", "side": -1},
        {"entry_earliest": "2021-01-04T00:00:00Z", "exit_earliest": "2021-01-07T00:00:00Z", "side": -1},
    ]
    accepted = reserve_nonoverlap(events)
    assert [row["side"] for row in accepted] == [1, -1]


def test_synthetic_cases_are_literal_and_redaction_equivalence_is_frozen() -> None:
    contract = semantic_contract(Config())
    frozen = contract["synthetic_controls"]["cases"]
    assert len(frozen) == len(SYNTHETIC_CASES) == 17
    equivalents = [
        row["redacted_excerpt"]
        for row in frozen
        if row.get("equivalence_group") == "entity_date_amount_swap"
    ]
    assert len(equivalents) == 2
    assert equivalents[0] == equivalents[1]
    guarded = [row for row in frozen if row["guarded"]]
    assert {row["name"] for row in guarded} == {
        "draw_prompt_injection",
        "buffer_prompt_injection",
    }


def test_frozen_config_and_contract_keep_model_outcomes_and_future_sealed() -> None:
    cfg = Config()
    _validate_frozen_config(cfg)
    with pytest.raises(ValueError, match="configuration is frozen"):
        _validate_frozen_config(replace(cfg, hold_hours=48))
    contract = semantic_contract(cfg)
    assert contract["model"]["id"] == MODEL_ID
    assert contract["model"]["revision"] == MODEL_REVISION
    assert contract["model"]["fine_tuned"] is False
    assert contract["model"]["batch_size"] == 1
    assert contract["splits"]["sealed"].startswith("2024-01-01")
    assert "return" not in contract["prompt"].lower().split("excerpt:")[-1]
    assert contract["economic_gates"]["failure_action"].startswith(
        "retire exact singleton"
    )


def test_preregistration_build_opens_no_body_model_market_or_future_rows() -> None:
    artifact = build_artifact(Config(), verify_model=False)
    boundary = artifact["outcome_boundary"]
    assert boundary["filing_bodies_opened"] == 0
    assert boundary["semantic_model_calls"] == 0
    assert boundary["btc_market_rows_read"] == 0
    assert boundary["funding_rows_read"] == 0
    assert boundary["future_return_rows_read"] == 0
    assert boundary["2024_or_later_source_rows_read"] == 0
    assert artifact["decision"]["synthetic_model_gate_authorized"] is True
    assert artifact["decision"]["historical_semantic_execution_authorized"] is False
    assert artifact["decision"]["economic_evaluation_authorized"] is False
