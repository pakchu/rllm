from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from training import (
    build_protocol_specification_intent_maturity_d8_source_support as runner,
)
from training import (
    probe_protocol_specification_intent_maturity_d8_relation_subcard_mechanism
    as mechanism,
)


RESULT_SHA256 = (
    "0b92b476b654cd76f0cf9dc004690cbcb78e7a5e73917b5d66611c0460d00204"
)
RESULT_HASH = (
    "7104593f0c0aa32e9f1219ab075fa10261058b57460286eaddf3e6764626fba5"
)
EVENTS_SHA256 = (
    "d7308789176af4bfe1bb2f5f13c89d6811bc7f938f3ecec08b1bf8acc5f7e2b2"
)
EVENTS_ROW_SHA256 = (
    "b6f1e1733d423fd0fd88f7008d1e505d3a513c0d2bec692446c6e2cf32196ac0"
)
CARDS_SHA256 = (
    "ce1bd1bd9a24068e6e223efca323db805781e912eadb0d2a8b7d63610fab96c1"
)
CARDS_ROW_SHA256 = (
    "cd73cd6f7f82a02b8662ef4689a721fa32698f73f37aebc1f1041dbfab3fb071"
)
CONTROLS_SHA256 = (
    "6c24b5d6ea693e19a90972a31ae96a24ac28a1f1a6b20be63418d0b5881551b1"
)
CONTROLS_ROW_HASH = (
    "d3c4f4868de128328aa36eda11764914fe3714fb57fd20ddb691822684f712ac"
)
CONTROLS_REPORT_HASH = (
    "abf3e50c750f906948c2e7220f6e724dc076f29f30002c1b04e47c38ffb21f55"
)
SEAL_HASH = (
    "f8f7ac92585227a3430008e4d68c170b48729798e99773866b63a2596059587b"
)


def _bytes(path: str | Path) -> bytes:
    return (runner.REPO_ROOT / path).read_bytes()


def _result() -> dict:
    return json.loads(_bytes(runner.DEFAULT_RESULT_PATH))


def _gzip_rows(path: str | Path) -> tuple[bytes, list[dict]]:
    raw = gzip.decompress(_bytes(path))
    return raw, [
        json.loads(line)
        for line in raw.decode("utf-8").splitlines()
    ]


def test_terminal_pass_is_canonical_hash_bound_and_source_only() -> None:
    raw = _bytes(runner.DEFAULT_RESULT_PATH)
    payload = _result()
    core = {
        key: value for key, value in payload.items() if key != "result_hash"
    }

    assert hashlib.sha256(raw).hexdigest() == RESULT_SHA256
    assert raw == runner.canonical_json_bytes(payload)
    assert payload["result_hash"] == RESULT_HASH
    assert payload["result_hash"] == runner.canonical_hash(core)
    assert payload["protocol_version"] == runner.RESULT_PROTOCOL
    assert payload["policy_id"] == runner.POLICY_ID
    assert payload["decision"] == "pass"
    assert payload["terminal_action"] == runner.PASS_ACTION
    assert payload["first_failure"] is None
    assert payload["error"] is None
    assert payload["profitability_result"] is False
    assert payload["outcomes_opened"] is False


def test_all_thirteen_source_gates_pass_and_forbidden_access_is_zero() -> None:
    payload = _result()
    gates = payload["gates"]

    assert [row["name"] for row in gates] == list(runner.GATE_NAMES)
    assert len(gates) == 13
    assert all(row["passed"] is True for row in gates)
    assert all(row["failure"] == "" for row in gates)
    assert payload["counts"] == {
        "events": 5_356,
        "daily_cards": 6_208,
    }
    assert payload["source_audit"]["source_run_attempt"] == 1
    assert payload["source_audit"]["source_root"] == "/tmp/psim-d8-source"
    assert payload["source_audit"]["source_traversal_ref"] == (
        "refs/psim-d8/sealed-tip"
    )
    assert payload["source_audit"]["repair_or_provider_swap_used"] is False
    assert payload["authority"]["execution_seal"]["seal_hash"] == SEAL_HASH

    relation_gate = gates[6]["metrics"]["relation_subcards"]
    assert relation_gate == {
        "logical_daily_cards": 6_208,
        "subcards": 6_308,
        "maximum_relation_units_per_subcard": 64,
        "frozen_limit": 64,
        "validation_errors": {},
    }
    forbidden = gates[11]["metrics"]["forbidden_fields"]
    ledger = gates[11]["metrics"]["ledger"]
    assert forbidden == list(runner.FORBIDDEN_ACCESS_FIELDS)
    assert all(ledger[name] == 0 for name in forbidden)


def test_frozen_event_card_and_control_artifacts_match_manifest() -> None:
    payload = _result()
    artifacts = payload["artifacts"]

    event_raw, event_rows = _gzip_rows(runner.DEFAULT_EVENTS_PATH)
    assert hashlib.sha256(
        _bytes(runner.DEFAULT_EVENTS_PATH)
    ).hexdigest() == EVENTS_SHA256
    assert hashlib.sha256(event_raw).hexdigest() == EVENTS_ROW_SHA256
    assert len(event_rows) == 5_356
    assert artifacts["events"] == {
        "path": runner.DEFAULT_EVENTS_PATH.as_posix(),
        "sha256": EVENTS_SHA256,
        "rows": 5_356,
        "row_hash": EVENTS_ROW_SHA256,
    }

    card_raw, card_rows = _gzip_rows(runner.DEFAULT_CARDS_PATH)
    assert hashlib.sha256(
        _bytes(runner.DEFAULT_CARDS_PATH)
    ).hexdigest() == CARDS_SHA256
    assert hashlib.sha256(card_raw).hexdigest() == CARDS_ROW_SHA256
    assert len(card_rows) == 6_208
    assert artifacts["daily_cards"] == {
        "path": runner.DEFAULT_CARDS_PATH.as_posix(),
        "sha256": CARDS_SHA256,
        "rows": 6_208,
        "row_hash": CARDS_ROW_SHA256,
    }

    control_raw = _bytes(runner.DEFAULT_CONTROLS_PATH)
    controls = json.loads(control_raw)
    assert hashlib.sha256(control_raw).hexdigest() == CONTROLS_SHA256
    assert control_raw == runner.canonical_json_bytes(controls)
    assert controls["report_hash"] == CONTROLS_REPORT_HASH
    assert controls["report_hash"] == runner.canonical_hash(
        {
            key: value
            for key, value in controls.items()
            if key != "report_hash"
        }
    )
    assert controls["profitability_result"] is False
    assert controls["outcomes_opened"] is False
    assert all(value == 0 for value in controls["forbidden_access"].values())
    assert controls["control_order"] == list(
        runner.core.prereg.RELATION_CONTROLS
    )
    assert all(
        row["passed"] is True
        for row in controls["metrics"].values()
    )
    assert all(
        cell["passed"] is True
        for row in controls["metrics"].values()
        for cell in row["cells"].values()
    )
    assert artifacts["controls"] == {
        "path": runner.DEFAULT_CONTROLS_PATH.as_posix(),
        "sha256": CONTROLS_SHA256,
        "rows": 7,
        "row_hash": CONTROLS_ROW_HASH,
    }


def test_every_logical_card_binds_a_complete_capped_subcard_manifest() -> None:
    _, cards = _gzip_rows(runner.DEFAULT_CARDS_PATH)
    identities: set[tuple[str, str]] = set()
    subcard_count = 0
    maximum_units = 0
    logical_relation_counts: list[int] = []
    subcard_histogram: dict[int, int] = {}

    for card in cards:
        mechanism.validate_logical_daily_card_envelope(card)
        identity = (card["schedule"], card["decision_at"])
        assert identity not in identities
        identities.add(identity)
        manifest = card["local_payload"]["relation_subcard_manifest"]
        subcard_count += manifest["subcard_count"]
        subcard_histogram[manifest["subcard_count"]] = (
            subcard_histogram.get(manifest["subcard_count"], 0) + 1
        )
        logical_relation_counts.append(manifest["relation_unit_count"])
        maximum_units = max(
            maximum_units,
            *(
                row["relation_unit_count"]
                for row in manifest["subcards"]
            ),
        )
        assert manifest["relation_unit_count"] == len(
            card["local_payload"]["relation_units"]
        )
        assert manifest["maximum_relation_units_per_subcard"] == 64

    assert len(identities) == 6_208
    assert subcard_count == 6_308
    assert maximum_units == 64
    assert sum(value > 64 for value in logical_relation_counts) == 24
    assert next(value for value in logical_relation_counts if value > 64) == 143
    assert max(logical_relation_counts) == 1_221
    assert subcard_histogram == {1: 6_184, 2: 16, 3: 4, 20: 4}
    assert {
        schedule: sum(1 for row in cards if row["schedule"] == schedule)
        for schedule in sorted({row["schedule"] for row in cards})
    } == {
        "ARCHIVE_D2": 1_552,
        "ARCHIVE_D7": 1_552,
        "ARCHIVE_D30": 1_552,
        "ARCHIVE_D90": 1_552,
    }


def test_terminal_pass_closes_source_execution_without_economic_claim() -> None:
    payload = runner.terminal_state()
    assert payload == _result()
    assert not (runner.REPO_ROOT / runner.RUN_LOCK_PATH).exists()
    assert not (runner.REPO_ROOT / runner.DEFAULT_REJECTION_PATH).exists()
    assert payload["source_audit"]["disk_below_limit_at_start"] is True
    assert payload["source_audit"]["disk_used_gib_at_start"] < 300
    assert payload["access_ledger"]["models_loaded"] == 0
    assert payload["access_ledger"]["model_outputs_built"] == 0
    assert payload["access_ledger"]["future_return_rows_read"] == 0
    assert payload["access_ledger"]["trade_rows_built"] == 0
    assert payload["access_ledger"]["pnl_rows_built"] == 0
    assert payload["access_ledger"]["cagr_values_built"] == 0
    assert payload["access_ledger"]["strict_mdd_values_built"] == 0
