from __future__ import annotations

import copy

import pandas as pd
import pytest

from training import evaluate_gross9_overlap_net_position_eval2025_override as evaluator
from training import preregister_gross9_overlap_net_position_eval2025_override as freeze


def test_freeze_binds_terminal_test_without_relabelling_it() -> None:
    value = freeze.build()
    freeze.validate(value)

    assert value["original_chain"]["original_protocol_terminal_reject_preserved"] is True
    assert value["user_override"]["overrides_original_stop_on_first_failure"] is True
    assert value["user_override"]["relabels_test2024_as_pass"] is False
    assert value["user_override"]["advance_beyond_eval2025_authorized"] is False
    assert value["evidence_boundary"]["eval2025_market_or_funding_rows_opened"] == 0
    assert value["evidence_boundary"]["eval2025_outcomes_opened"] is False
    assert value["v1_infrastructure_failure"]["economic_metrics_computed"] is False
    assert value["v1_infrastructure_failure"]["candidate_pass_fail_observed"] is False
    normalization = value["eval2025"]["funding_time_normalization"]
    assert normalization["rate_and_mark_price_changed"] is False
    assert normalization["duplicate_normalized_buckets_allowed"] is False


def test_freeze_keeps_exact_weights_and_no_repair() -> None:
    value = freeze.build()

    assert value["fixed_portfolio"]["sleeve_weights"] == {
        "HVBLIC-6": 0.2,
        "HVCPF17-8__ASYNC_ACTIVE_OPPOSITE_VETO_6H__HVDIMIO-8": 0.25,
        "HVRDBR-6": 0.15,
        "HVUI-12": 0.1,
    }
    assert value["fixed_portfolio"]["weights_changed"] is False
    assert value["fixed_portfolio"]["rerank_repair_or_substitution_authorized"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["user_override"].update(
            {"advance_beyond_eval2025_authorized": True}
        ),
        lambda value: value["original_chain"]["test2024_terminal"].update(
            {"sha256": "0" * 64}
        ),
        lambda value: value["fixed_portfolio"]["sleeve_weights"].update(
            {"HVBLIC-6": 0.3}
        ),
        lambda value: value["eval2025"].update(
            {"window": ["2025-01-01T00:00:00Z", "2026-08-01T00:00:00Z"]}
        ),
        lambda value: value["evidence_boundary"].update(
            {"final2026_outcomes_opened": True}
        ),
    ],
)
def test_freeze_rejects_rehashed_bound_input_tampering(mutation) -> None:
    value = copy.deepcopy(freeze.build())
    mutation(value)
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    value["manifest_hash"] = freeze.canonical_hash(core)

    with pytest.raises(RuntimeError):
        freeze.validate(value)


def test_public_source_receipt_redacts_database_location() -> None:
    public = evaluator.public_source_receipt(
        {
            "database_identity": {
                "configured_host": "db.example",
                "configured_database": "private",
                "server_version_num": "160015",
                "transaction_snapshot": "1:2:",
            },
            "database_environment_source": "/secret/path/.env",
        }
    )

    assert public["database_identity"]["network_and_database_names_redacted"] is True
    assert "configured_host" not in public["database_identity"]
    assert "configured_database" not in public["database_identity"]
    assert public["database_environment_source"] == "redacted_local_env_file"


def test_funding_millisecond_jitter_is_bucketed_without_value_changes() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-01T00:00:00.015Z", "2025-01-01T08:00:00.001Z"],
                utc=True,
            ),
            "funding_rate": [0.0001, -0.0002],
            "mark_price": [100.0, 101.0],
        }
    )

    normalized = evaluator.normalize_funding_clock(raw)

    assert normalized["date"].tolist() == pd.to_datetime(
        ["2025-01-01T00:00:00Z", "2025-01-01T08:00:00Z"], utc=True
    ).tolist()
    assert normalized["funding_rate"].tolist() == raw["funding_rate"].tolist()
    assert normalized["mark_price"].tolist() == raw["mark_price"].tolist()


def test_evaluator_contains_no_search_or_selector_calls() -> None:
    source = freeze.EVALUATOR.read_text(encoding="utf-8")

    for forbidden in (
        "beam_search_portfolios(",
        "select_authoritative_rank1(",
        "optimize_from_manifest(",
        "run_frozen(",
    ):
        assert forbidden not in source
