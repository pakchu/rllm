from __future__ import annotations

import copy
import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from training import build_tron_usdt_supply_events as builder
from training import evaluate_tron_usdt_supply_impulse_source_support as s


def timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if not isinstance(result, pd.Timestamp):
        raise AssertionError("test timestamp unexpectedly resolved to NaT")
    return result


def minutes(value: int) -> pd.Timedelta:
    result = pd.Timedelta(minutes=value)
    if not isinstance(result, pd.Timedelta):
        raise AssertionError("test duration unexpectedly resolved to NaT")
    return result


def frame_records(rows: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], rows.to_dict(orient="records"))


def boundary_record(boundary: Mapping[str, Any], frozen: bool) -> dict[str, Any]:
    number = boundary["number"]
    if isinstance(number, bool) or not isinstance(number, int):
        raise AssertionError("test boundary number is not an integer")
    return {
        "utc": boundary["utc"],
        "previous_block": number - 1,
        "first_block_at_or_after": number,
        "parent_relation_exact": True,
        "timestamp_relation_exact": True,
        "frozen_hash_exact": frozen,
    }


def registration() -> dict[str, Any]:
    core = {
        "policy_id": s.POLICY_ID,
        "feature_and_signal": {"eligible_event_types": ["Issue", "Redeem"]},
        "execution": {"hold_hours": 168},
        "source": {"deprecate_terminates_source_v1": True},
    }
    return {**core, "manifest_hash": s.canonical_hash(core)}


def source_row(
    number: int,
    available: str,
    *,
    event_type: str = "Issue",
    amount: int = 1_000_000,
    transaction_index: int = 0,
    log_index: int = 0,
) -> dict[str, Any]:
    direction = {
        "Issue": 1,
        "Redeem": -1,
        "DestroyedBlackFunds": -1,
    }[event_type]
    event_time = timestamp(available) - minutes(4)
    return {
        "event_type": event_type,
        "supply_direction": direction,
        "actor_address": "0x" + f"{number + 1:040x}"[-40:],
        "amount_raw": amount,
        "block_number": builder.SOURCE_START_BLOCK + number,
        "block_hash": "0x" + f"{number + 101:064x}"[-64:],
        "transaction_hash": "0x" + f"{number + 201:064x}"[-64:],
        "transaction_index": transaction_index,
        "log_index": log_index,
        "paired_transfer_log_index": (
            log_index + 1 if event_type in {"Issue", "Redeem"} else None
        ),
        "event_timestamp_utc": event_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "confirmation_block": (
            builder.SOURCE_START_BLOCK + number + builder.CONFIRMATION_BLOCKS
        ),
        "confirmation_block_hash": "0x" + f"{number + 301:064x}"[-64:],
        "available_at_utc": available,
    }


def frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["block_number"],
            row["transaction_index"],
            row["log_index"],
            row["transaction_hash"],
        ),
    )
    return pd.DataFrame(ordered, columns=pd.Index(builder.CSV_COLUMNS))


def passing_frame() -> pd.DataFrame:
    dates = [
        "2023-06-10T00:00:00Z",
        "2023-09-10T00:00:00Z",
        "2023-12-10T00:00:00Z",
        "2024-02-10T00:00:00Z",
        "2024-05-10T00:00:00Z",
        "2024-08-10T00:00:00Z",
        "2024-11-10T00:00:00Z",
        "2024-12-20T00:00:00Z",
        "2025-02-10T00:00:00Z",
        "2025-05-10T00:00:00Z",
        "2025-08-10T00:00:00Z",
        "2025-11-10T00:00:00Z",
        "2026-02-10T00:00:00Z",
        "2026-04-10T00:00:00Z",
    ]
    return frame(
        source_row(
            index,
            date,
            event_type="Issue" if index % 2 == 0 else "Redeem",
            amount=1_000_000 + index,
        )
        for index, date in enumerate(dates)
    )


def builder_records(rows: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "event_type": str(row["event_type"]),
            "supply_direction": int(row["supply_direction"]),
            "actor_address": str(row["actor_address"]),
            "amount_raw": str(int(row["amount_raw"])),
            "block_number": int(row["block_number"]),
            "block_hash": str(row["block_hash"]),
            "transaction_hash": str(row["transaction_hash"]),
            "transaction_index": int(row["transaction_index"]),
            "log_index": int(row["log_index"]),
            "paired_transfer_log_index": (
                ""
                if row["paired_transfer_log_index"] is None
                or bool(pd.isna(row["paired_transfer_log_index"]))
                else int(row["paired_transfer_log_index"])
            ),
            "event_timestamp_utc": str(row["event_timestamp_utc"]),
            "confirmation_block": int(row["confirmation_block"]),
            "confirmation_block_hash": str(row["confirmation_block_hash"]),
            "available_at_utc": str(row["available_at_utc"]),
        }
        for row in frame_records(rows)
    ]


def manifest_inputs(
    records: list[dict[str, Any]],
) -> tuple[
    dict[str, tuple[builder.CanonicalLog, ...]],
    dict[str, builder.Receipt],
]:
    category_logs: dict[str, list[builder.CanonicalLog]] = {
        category: [] for category in builder.CATEGORIES
    }
    receipts: dict[str, builder.Receipt] = {}
    semantic_topics = {
        "Issue": builder.ISSUE_TOPIC,
        "Redeem": builder.REDEEM_TOPIC,
        "DestroyedBlackFunds": builder.DESTROYED_BLACK_FUNDS_TOPIC,
    }
    for record in records:
        amount_word = f"0x{int(record['amount_raw']):064x}"
        semantic = builder.CanonicalLog(
            block_number=record["block_number"],
            transaction_index=record["transaction_index"],
            log_index=record["log_index"],
            block_hash=record["block_hash"],
            transaction_hash=record["transaction_hash"],
            address=builder.USDT_CONTRACT,
            topics=(semantic_topics[record["event_type"]],),
            data=amount_word,
            removed=False,
        )
        category_logs[builder.CATEGORY_SEMANTIC].append(semantic)
        transaction_logs = [semantic]
        if record["event_type"] in {"Issue", "Redeem"}:
            transfer = builder.CanonicalLog(
                block_number=record["block_number"],
                transaction_index=record["transaction_index"],
                log_index=int(record["paired_transfer_log_index"]),
                block_hash=record["block_hash"],
                transaction_hash=record["transaction_hash"],
                address=builder.USDT_CONTRACT,
                topics=(builder.TRANSFER_TOPIC,),
                data=amount_word,
                removed=False,
            )
            category = (
                builder.CATEGORY_MINT
                if record["event_type"] == "Issue"
                else builder.CATEGORY_BURN
            )
            category_logs[category].append(transfer)
            transaction_logs.append(transfer)
        receipts[record["transaction_hash"]] = builder.Receipt(
            transaction_hash=record["transaction_hash"],
            transaction_index=record["transaction_index"],
            block_hash=record["block_hash"],
            block_number=record["block_number"],
            status=1,
            logs=tuple(transaction_logs),
        )
    return (
        {
            category: tuple(sorted(logs))
            for category, logs in category_logs.items()
        },
        receipts,
    )


def synthetic_manifest(
    csv_bytes: bytes,
    rows: pd.DataFrame,
    *,
    production: bool = False,
) -> dict[str, Any]:
    records = builder_records(rows)
    category_logs, receipts = manifest_inputs(records)
    protocol_parent = "a" * 40
    frozen_boundaries = production
    return builder.build_manifest(
        records,
        csv_bytes,
        category_logs=category_logs,
        replay_chunks=(
            builder.frozen_chunks()
            if production
            else (
                (
                    min(record["block_number"] for record in records),
                    max(record["block_number"] for record in records),
                ),
            )
        ),
        finalized_head=builder.Header(
            number=builder.LAST_CONFIRMATION_BLOCK,
            block_hash="0x" + "99" * 32,
            parent_hash="0x" + "98" * 32,
            timestamp=int(timestamp(builder.END_BOUNDARY_UTC).timestamp()),
        ),
        boundary_evidence={
            "outside_before_count": 0,
            "outside_after_maximum_admissible_count": 0,
            "header_count": 2 * len(builder.FROZEN_BOUNDARIES),
            "canonical_header_set_sha256": (
                builder.BOUNDARY_HEADER_SET_SHA256
                if frozen_boundaries
                else "d" * 64
            ),
            "frozen_header_set_exact": frozen_boundaries,
            "boundaries": [
                boundary_record(boundary, frozen_boundaries)
                for boundary in builder.FROZEN_BOUNDARIES
            ],
        },
        source_integrity=dict(builder.ZERO_SOURCE_INTEGRITY),
        headers={},
        receipts=receipts,
        protocol_seal={"git_head": protocol_parent},
        claim_binding=(
            {
                "claim_commit": "b" * 40,
                "protocol_parent_commit": protocol_parent,
                "sha256": "c" * 64,
            }
            if production
            else None
        ),
        production=production,
    )


def write_source_artifacts(
    tmp_path: Path, rows: pd.DataFrame
) -> tuple[Path, Path, dict[str, Any]]:
    csv_path = tmp_path / "source.csv.gz"
    manifest_path = tmp_path / "manifest.json"
    records = builder_records(rows)
    csv_bytes = builder.serialize_csv(records)
    payload = synthetic_manifest(csv_bytes, rows)
    csv_path.write_bytes(csv_bytes)
    manifest_path.write_bytes(builder.serialize_manifest(payload))
    return csv_path, manifest_path, payload


def test_entry_rounding_always_waits_one_bar() -> None:
    assert s.candidate_entry_time("2024-01-01T00:00:00Z") == timestamp(
        "2024-01-01T00:05:00Z"
    )
    assert s.candidate_entry_time("2024-01-01T00:00:01Z") == timestamp(
        "2024-01-01T00:10:00Z"
    )
    assert s.candidate_entry_time("2024-01-01T00:04:59Z") == timestamp(
        "2024-01-01T00:10:00Z"
    )


def test_source_schema_labels_directions_and_deprecate_are_exact() -> None:
    valid = frame(
        [
            source_row(0, "2024-01-01T00:00:00Z", event_type="Issue"),
            source_row(1, "2024-01-02T00:00:00Z", event_type="Redeem"),
            source_row(
                2,
                "2024-01-03T00:00:00Z",
                event_type="DestroyedBlackFunds",
            ),
        ]
    )
    assert s.validate_source_frame(valid)["event_type"].tolist() == [
        "Issue",
        "Redeem",
        "DestroyedBlackFunds",
    ]
    assert valid.iloc[-1]["block_number"] < builder.LAST_EVENT_BLOCK
    assert len(s.validate_source_frame(valid)) == 3
    huge = valid.iloc[[0]].copy()
    huge["amount_raw"] = huge["amount_raw"].astype("object")
    huge.loc[huge.index[0], "amount_raw"] = 2**200
    assert s.validate_source_frame(huge).iloc[0]["amount_raw"] == 2**200
    largest = valid.iloc[[0]].copy()
    largest["amount_raw"] = largest["amount_raw"].astype("object")
    largest.loc[largest.index[0], "amount_raw"] = 2**256 - 1
    assert s.validate_source_frame(largest).iloc[0]["amount_raw"] == 2**256 - 1
    for amount in (0, 2**256):
        invalid_amount = valid.iloc[[0]].copy()
        invalid_amount["amount_raw"] = invalid_amount["amount_raw"].astype(
            "object"
        )
        invalid_amount.loc[invalid_amount.index[0], "amount_raw"] = amount
        with pytest.raises(RuntimeError, match="nonpositive|below 2\\*\\*256"):
            s.validate_source_frame(invalid_amount)

    wrong_case = valid.copy()
    wrong_case.loc[0, "event_type"] = "issue"
    with pytest.raises(RuntimeError, match="unsupported event type"):
        s.validate_source_frame(wrong_case)
    wrong_direction = valid.copy()
    wrong_direction.loc[1, "supply_direction"] = 1
    with pytest.raises(RuntimeError, match="supply direction"):
        s.validate_source_frame(wrong_direction)
    deprecate = valid.copy()
    deprecate.loc[1, "event_type"] = "Deprecate"
    with pytest.raises(RuntimeError, match="Deprecate"):
        s.validate_source_frame(deprecate)


def test_source_timestamps_require_canonical_strings_and_monotone_block_order() -> None:
    rows = frame(
        [
            source_row(0, "2024-01-02T00:00:00Z"),
            source_row(1, "2024-01-03T00:00:00Z"),
        ]
    )
    for column, replacement in (
        ("event_timestamp_utc", "2024-01-01T23:56:00+00:00"),
        ("available_at_utc", "2024-01-02T00:00:00.000Z"),
    ):
        changed = rows.copy()
        changed.loc[0, column] = replacement
        with pytest.raises(RuntimeError, match="noncanonical whole-second"):
            s.validate_source_frame(changed)

    event_decrease = rows.copy()
    event_decrease.loc[1, "event_timestamp_utc"] = "2024-01-01T23:55:00Z"
    with pytest.raises(RuntimeError, match="timestamps decrease"):
        s.validate_source_frame(event_decrease)

    availability_decrease = rows.copy()
    availability_decrease.loc[0, "event_timestamp_utc"] = "2024-01-01T00:00:00Z"
    availability_decrease.loc[0, "available_at_utc"] = "2024-01-02T00:00:00Z"
    availability_decrease.loc[1, "event_timestamp_utc"] = "2024-01-01T01:00:00Z"
    availability_decrease.loc[1, "available_at_utc"] = "2024-01-01T02:00:00Z"
    with pytest.raises(RuntimeError, match="timestamps decrease"):
        s.validate_source_frame(availability_decrease)

    above_envelope = rows.iloc[[0]].copy()
    above_envelope.loc[above_envelope.index[0], "block_number"] = (
        builder.LAST_EVENT_BLOCK + 1
    )
    above_envelope.loc[above_envelope.index[0], "confirmation_block"] = (
        builder.LAST_EVENT_BLOCK + 1 + builder.CONFIRMATION_BLOCKS
    )
    with pytest.raises(RuntimeError, match="outside frozen source range"):
        s.validate_source_frame(above_envelope)


def test_primary_bucket_identity_amount_decision_sort_and_zero_abstain() -> None:
    rows = frame(
        [
            source_row(
                0,
                "2024-02-01T00:00:01Z",
                event_type="Issue",
                amount=10,
            ),
            source_row(
                1,
                "2024-02-01T00:04:59Z",
                event_type="Redeem",
                amount=3,
            ),
            source_row(
                2,
                "2024-03-01T00:00:00Z",
                event_type="Issue",
                amount=7,
            ),
            source_row(
                3,
                "2024-03-01T00:00:00Z",
                event_type="Redeem",
                amount=7,
            ),
        ]
    )
    candidates = s.raw_candidates(rows)
    assert len(candidates) == 1
    candidate = candidates.iloc[0]
    assert candidate["bucket_amount_raw"] == 7
    assert candidate["side"] == "LONG"
    assert candidate["decision_time_utc"] == timestamp("2024-02-01T00:04:59Z")
    bucket = s.validate_source_frame(rows).iloc[:2]
    assert candidate["source_identity"] == s.source_identity(bucket)
    assert candidate["source_identity"] == s.source_identity(bucket.iloc[::-1])
    expected_arrays = [
        [
            int(row.block_number),
            int(row.transaction_index),
            int(row.log_index),
            str(row.transaction_hash),
            str(row.event_type),
            int(row.amount_raw),
        ]
        for row in bucket.itertuples(index=False)
    ]
    expected_bytes = json.dumps(
        sorted(expected_arrays),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert s.canonical_constituent_bytes(bucket) == expected_bytes
    assert candidate["source_identity"] == hashlib.sha256(expected_bytes).hexdigest()
    assert candidate["constituent_identities_json"] == expected_bytes.decode()


def test_independent_controls_rebuild_own_buckets_and_schedules() -> None:
    rows = frame(
        [
            source_row(
                0,
                "2024-02-01T00:00:00Z",
                event_type="Issue",
                amount=5,
            ),
            source_row(
                1,
                "2024-02-01T00:00:00Z",
                event_type="Issue",
                amount=5,
            ),
            source_row(
                2,
                "2024-02-01T00:00:00Z",
                event_type="Redeem",
                amount=8,
            ),
            source_row(
                3,
                "2024-02-01T00:00:00Z",
                event_type="DestroyedBlackFunds",
                amount=3,
            ),
        ]
    )
    primary = s.raw_candidates(rows, "primary").iloc[0]
    assert (primary["bucket_amount_raw"], primary["side"]) == (2, "LONG")
    issue = s.raw_candidates(rows, "issue_only").iloc[0]
    assert (issue["bucket_amount_raw"], issue["side"]) == (10, "LONG")
    redeem = s.raw_candidates(rows, "redeem_only").iloc[0]
    assert (redeem["bucket_amount_raw"], redeem["side"]) == (-8, "SHORT")
    destroyed = s.raw_candidates(rows, "include_destroyed_black_funds").iloc[0]
    assert (destroyed["bucket_amount_raw"], destroyed["side"]) == (-1, "SHORT")
    count = s.raw_candidates(rows, "count_net_side").iloc[0]
    assert count["bucket_amount_raw"] == 1
    assert count["side"] == "LONG"
    assert (
        len(
            {
                issue["source_identity"],
                redeem["source_identity"],
                destroyed["source_identity"],
            }
        )
        == 3
    )


def test_scheduler_is_global_accepts_exact_exit_and_never_queues_suppressed() -> None:
    rows = frame(
        [
            source_row(0, "2024-01-01T00:00:00Z"),
            source_row(1, "2024-01-02T00:00:00Z"),
            source_row(2, "2024-01-08T00:00:00Z"),
        ]
    )
    raw = s.raw_candidates(rows)
    accepted = s.reserve_nonoverlap(raw)
    assert accepted["entry_time_utc"].tolist() == [
        timestamp("2024-01-01T00:05:00Z"),
        timestamp("2024-01-08T00:05:00Z"),
    ]
    assert accepted.iloc[1]["entry_time_utc"] == accepted.iloc[0]["exit_time_utc"]


def test_parent_controls_use_accepted_primary_without_rescheduling() -> None:
    controls, raw_counts = s.build_controls(passing_frame())
    primary = controls["primary"]
    for name in s.SAME_PARENT_CONTROLS:
        assert len(controls[name]) == len(primary)
        assert raw_counts[name] == len(primary)
        assert (
            controls[name]["source_identity"].tolist()
            == primary["source_identity"].tolist()
        )
    assert controls["exact_direction_flip"]["side"].tolist() == [
        "SHORT" if side == "LONG" else "LONG" for side in primary["side"]
    ]
    assert bool(controls["constant_long"]["side"].eq("LONG").all())
    assert bool(controls["constant_short"]["side"].eq("SHORT").all())
    assert controls["deterministic_random_side"]["side"].tolist() == [
        s.deterministic_random_side(identity) for identity in primary["source_identity"]
    ]
    delayed = controls["one_bar_delayed_entry"]
    assert (
        delayed["entry_time_utc"].reset_index(drop=True)
        == primary["entry_time_utc"].reset_index(drop=True) + s.BAR
    ).all()
    assert (
        delayed["exit_time_utc"].reset_index(drop=True)
        == primary["exit_time_utc"].reset_index(drop=True) + s.BAR
    ).all()


def test_delayed_control_retains_parent_and_reports_shifted_crossers() -> None:
    rows = frame([source_row(0, "2024-12-24T23:55:00Z")])
    primary = s.reserve_nonoverlap(s.raw_candidates(rows))
    delayed = s._parent_control(primary, "one_bar_delayed_entry")
    assert delayed["window"].tolist() == ["selection"]
    assert delayed["source_identity"].tolist() == primary["source_identity"].tolist()
    crossers = s.delayed_control_boundary_crossers(primary, delayed)
    identity = primary.iloc[0]["source_identity"]
    assert crossers["selection"] == {
        "count": 1,
        "source_identities": [identity],
    }
    assert crossers["2024"] == {
        "count": 1,
        "source_identities": [identity],
    }
    assert crossers["2024H2"] == {
        "count": 1,
        "source_identities": [identity],
    }
    assert crossers["full"] == {"count": 0, "source_identities": []}
    assert s.project_clock_to_period(delayed, *s.SPLITS["selection"]).empty
    report, primary_bytes, control_bytes = s.build_support_from_frame(
        rows, registration=registration()
    )
    assert report["support_audit"]["one_bar_delayed_entry_boundary_crossers"][
        "selection"
    ] == {
        "count": 1,
        "source_identities": [identity],
    }
    s._validate_output_generation(
        s._json_bytes(report),
        primary_bytes,
        control_bytes,
        source_frame=rows,
    )


def test_half_open_calendars_skip_crossers_without_truncation() -> None:
    rows = frame(
        [
            source_row(0, "2023-05-31T23:49:59Z"),
            source_row(1, "2024-12-27T00:00:00Z"),
            source_row(2, "2025-01-02T00:00:00Z"),
            source_row(3, "2026-05-25T00:00:00Z"),
        ]
    )
    accepted = s.reserve_nonoverlap(s.raw_candidates(rows))
    assert accepted["window"].tolist() == ["future25"]
    assert accepted.iloc[0]["entry_time_utc"] == timestamp(
        "2025-01-02T00:05:00Z"
    )
    assert accepted.iloc[0]["exit_time_utc"] == timestamp(
        "2025-01-09T00:05:00Z"
    )


def test_diagnostics_project_one_accepted_clock_without_rescheduling() -> None:
    rows = frame(
        [
            source_row(0, "2024-06-27T00:00:00Z"),
            source_row(1, "2024-07-02T00:00:00Z"),
            source_row(2, "2024-07-04T00:00:00Z"),
        ]
    )
    accepted = s.reserve_nonoverlap(s.raw_candidates(rows))
    assert accepted["entry_time_utc"].tolist() == [
        timestamp("2024-06-27T00:05:00Z"),
        timestamp("2024-07-04T00:05:00Z"),
    ]
    audit, _ = s.support_checks(accepted)
    assert audit["period_diagnostics"]["selection"]["accepted"] == 2
    assert audit["period_diagnostics"]["2024H1"]["accepted"] == 0
    assert audit["period_diagnostics"]["2024H2"]["accepted"] == 1
    assert audit["period_diagnostics"]["2024"]["accepted"] == 2


def test_future_append_selection_rebuild_is_exact() -> None:
    rows = passing_frame()
    assert s.FUTURE_APPEND_ROW_KEYS == {
        "accepted",
        "control",
        "constituent_identities",
        "source_identity",
        "constituent_count",
        "signed_bucket_amount_or_count",
        "decision_time_utc",
        "entry_time_utc",
        "exit_time_utc",
        "side",
    }
    passed, report = s.future_append_selection_invariance(rows)
    assert passed is True
    assert report["total_differences"] == 0
    assert set(report["comparisons"]) == set(s.CONTROL_ORDER)
    for control in s.INDEPENDENT_CONTROLS:
        assert set(report["comparisons"][control]) == {"raw", "accepted"}
    for control in s.SAME_PARENT_CONTROLS:
        assert set(report["comparisons"][control]) == {"accepted"}
    for views in report["comparisons"].values():
        for comparison in views.values():
            assert comparison["differences"] == 0
            assert comparison["full_sha256"] == comparison["prefix_sha256"]
    construction = s._selection_construction_views(s.validate_source_frame(rows))
    for control in s.CONTROL_ORDER:
        expected_views = (
            ("raw", "accepted")
            if control in s.INDEPENDENT_CONTROLS
            else ("accepted",)
        )
        for view_name in expected_views:
            payload = s._append_view_payload(
                construction[control][view_name],
                accepted=view_name == "accepted",
            )
            for canonical_row in payload:
                assert set(canonical_row) == s.FUTURE_APPEND_ROW_KEYS
                assert canonical_row["accepted"] is (view_name == "accepted")
                assert canonical_row["control"] == control
                assert isinstance(canonical_row["constituent_identities"], list)


def test_support_floors_month_share_gap_and_long_short_reporting() -> None:
    controls, _ = s.build_controls(passing_frame())
    audit, checks = s.support_checks(controls["primary"])
    assert all(checks.values())
    assert audit["period_diagnostics"]["selection"]["accepted"] == 8
    assert audit["period_diagnostics"]["future25"]["accepted"] == 4
    assert audit["period_diagnostics"]["future26"]["accepted"] == 2
    assert audit["long_short_counts_are_report_only"] is True
    assert audit["maximum_full_entry_gap_seconds"] <= 240 * 86400

    concentrated = controls["primary"].copy()
    selection_index = concentrated.index[concentrated["window"].eq("selection")]
    concentrated.loc[selection_index, "entry_time_utc"] = [
        timestamp("2024-02-01T00:05:00Z") + pd.Timedelta(days=index)
        for index in range(len(selection_index))
    ]
    concentrated.loc[selection_index, "exit_time_utc"] = (
        concentrated.loc[selection_index, "entry_time_utc"] + s.HOLD
    )
    _, failed = s.support_checks(concentrated)
    assert failed["selection_maximum_month_share_at_most_half"] is False


def test_manifest_csv_hash_schema_integrity_and_counts_are_validated(
    tmp_path: Path,
) -> None:
    rows = passing_frame()
    csv_path, manifest_path, payload = write_source_artifacts(tmp_path, rows)
    loaded, audit = s.load_synthetic_source_artifacts(
        csv_path=csv_path,
        manifest_path=manifest_path,
    )
    assert len(loaded) == len(rows)
    assert audit["source_integrity"] == builder.ZERO_SOURCE_INTEGRITY
    assert audit["source_manifest_hash"] == payload["manifest_hash"]
    assert builder.PRODUCTION_GENERATION_COMMIT == {
        "protocol": "manifest_last_v1",
        "mode": "production",
        "full_envelope_integrity": True,
        "canonical_publication_eligible": True,
        "manifest_is_commit_marker": True,
        "posix_multi_path_atomic": False,
    }
    assert builder.SYNTHETIC_GENERATION_COMMIT == {
        "protocol": "computation_only_v1",
        "mode": "synthetic_nonproduction",
        "full_envelope_integrity": False,
        "canonical_publication_eligible": False,
        "manifest_is_commit_marker": False,
        "posix_multi_path_atomic": False,
    }
    assert payload["generation_commit"] == builder.SYNTHETIC_GENERATION_COMMIT
    assert payload["protocol_parent_commit"] == "a" * 40
    assert payload["replay_claim_commit"] is None
    assert payload["replay_claim_sha256"] is None
    assert payload["source_csv_sha256"] == hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()

    changed = json.loads(manifest_path.read_text())
    changed["source_integrity"]["header_differences"] = 1
    changed_core = {
        key: value for key, value in changed.items() if key != "manifest_hash"
    }
    changed["manifest_hash"] = s.canonical_hash(changed_core)
    manifest_path.write_bytes(builder.serialize_manifest(changed))
    with pytest.raises(RuntimeError, match="source integrity"):
        s.load_synthetic_source_artifacts(
            csv_path=csv_path,
            manifest_path=manifest_path,
        )

    manifest_path.write_bytes(builder.serialize_manifest(payload))
    csv_path.write_bytes(csv_path.read_bytes() + b"drift")
    with pytest.raises(RuntimeError, match="not valid gzip|binding drift"):
        s.load_synthetic_source_artifacts(
            csv_path=csv_path,
            manifest_path=manifest_path,
        )


def test_current_builder_production_manifest_schema_and_envelope_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = passing_frame()
    records = builder_records(rows)
    csv_bytes = builder.serialize_csv(records)
    payload = synthetic_manifest(csv_bytes, rows, production=True)
    assert payload["generation_commit"] == builder.PRODUCTION_GENERATION_COMMIT
    assert payload["protocol_parent_commit"] == "a" * 40
    assert payload["replay_claim_commit"] == "b" * 40
    assert payload["replay_claim_sha256"] == "c" * 64
    assert rows["block_number"].max() < builder.LAST_EVENT_BLOCK
    monkeypatch.setattr(s, "_validate_production_claim_binding", lambda _: None)
    audit = s.validate_source_manifest(
        payload,
        csv_bytes=csv_bytes,
        frame=s.validate_source_frame(rows),
        production=True,
    )
    assert audit["source_integrity"] == builder.ZERO_SOURCE_INTEGRITY
    missing_direct_binding = copy.deepcopy(payload)
    missing_direct_binding["replay_claim_commit"] = None
    core = {
        key: value
        for key, value in missing_direct_binding.items()
        if key != "manifest_hash"
    }
    missing_direct_binding["manifest_hash"] = s.canonical_hash(core)
    with pytest.raises(
        RuntimeError,
        match="builder source-manifest|direct claim/seal|binding missing",
    ):
        s.validate_source_manifest(
            missing_direct_binding,
            csv_bytes=csv_bytes,
            frame=s.validate_source_frame(rows),
            production=True,
        )


@pytest.mark.parametrize("marker_mutation", ("missing", "nonexact"))
def test_invalid_generation_commit_marker_is_rejected_before_csv_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_mutation: str,
) -> None:
    rows = passing_frame()
    csv_path, manifest_path, payload = write_source_artifacts(tmp_path, rows)
    changed = copy.deepcopy(payload)
    if marker_mutation == "missing":
        del changed["generation_commit"]
    else:
        changed["generation_commit"]["manifest_is_commit_marker"] = True
    core = {
        key: value for key, value in changed.items() if key != "manifest_hash"
    }
    changed["manifest_hash"] = s.canonical_hash(core)
    manifest_path.write_bytes(builder.serialize_manifest(changed))
    original_read_bytes = Path.read_bytes
    csv_reads = 0

    def tracked_read_bytes(path: Path) -> bytes:
        nonlocal csv_reads
        if path == csv_path:
            csv_reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)
    with pytest.raises(RuntimeError, match="commit-marker"):
        s.load_synthetic_source_artifacts(
            csv_path=csv_path,
            manifest_path=manifest_path,
        )
    assert csv_reads == 0


def test_production_loader_checks_marker_before_csv_and_uses_builder_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = passing_frame()
    records = builder_records(rows)
    csv_bytes = builder.serialize_csv(records)
    payload = synthetic_manifest(csv_bytes, rows, production=True)
    csv_path = tmp_path / "source.csv.gz"
    manifest_path = tmp_path / "manifest.json"
    csv_path.write_bytes(csv_bytes)
    manifest_path.write_bytes(builder.serialize_manifest(payload))
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(s, "_git", lambda *arguments: b"")
    monkeypatch.setattr(s, "_validate_production_claim_binding", lambda _: None)
    real_builder_validator = builder._validate_production_manifest
    builder_validation_calls = 0
    read_order: list[Path] = []
    real_read_bytes = Path.read_bytes

    def tracked_builder_validator(
        manifest: dict[str, Any], source_bytes: bytes
    ) -> dict[str, Any]:
        nonlocal builder_validation_calls
        builder_validation_calls += 1
        return real_builder_validator(manifest, source_bytes)

    def tracked_read_bytes(path: Path) -> bytes:
        if path in {manifest_path, csv_path}:
            read_order.append(path)
        return real_read_bytes(path)

    monkeypatch.setattr(
        builder,
        "_validate_production_manifest",
        tracked_builder_validator,
    )
    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)
    loaded, audit = s.load_source_artifacts(
        csv_path=csv_path,
        manifest_path=manifest_path,
        production=True,
    )
    assert len(loaded) == len(rows)
    assert audit["artifact_eligible"] is True
    assert builder_validation_calls == 1
    assert read_order == [manifest_path, csv_path]

    invalid = copy.deepcopy(payload)
    invalid["generation_commit"] = dict(builder.SYNTHETIC_GENERATION_COMMIT)
    core = {
        key: value for key, value in invalid.items() if key != "manifest_hash"
    }
    invalid["manifest_hash"] = s.canonical_hash(core)
    manifest_path.write_bytes(builder.serialize_manifest(invalid))
    read_order.clear()
    with pytest.raises(RuntimeError, match="commit-marker"):
        s.load_source_artifacts(
            csv_path=csv_path,
            manifest_path=manifest_path,
            production=True,
        )
    assert read_order == [manifest_path]

    missing = copy.deepcopy(payload)
    del missing["generation_commit"]
    core = {
        key: value for key, value in missing.items() if key != "manifest_hash"
    }
    missing["manifest_hash"] = s.canonical_hash(core)
    manifest_path.write_bytes(builder.serialize_manifest(missing))
    read_order.clear()
    with pytest.raises(RuntimeError, match="commit-marker"):
        s.load_source_artifacts(
            csv_path=csv_path,
            manifest_path=manifest_path,
            production=True,
        )
    assert read_order == [manifest_path]


def test_build_is_deterministic_reports_hashes_diagnostics_and_overlap() -> None:
    rows = passing_frame()
    source_audit = {
        "artifact_eligible": False,
        "source_integrity": dict(builder.ZERO_SOURCE_INTEGRITY),
    }
    first, primary_a, controls_a = s.build_support_from_frame(
        rows,
        registration=registration(),
        source_audit=source_audit,
    )
    second, primary_b, controls_b = s.build_support_from_frame(
        rows.copy(),
        registration=registration(),
        source_audit=source_audit,
    )
    assert first == second
    assert primary_a == primary_b
    assert controls_a == controls_b
    assert gzip.decompress(primary_a).startswith(
        b"policy_id,control,window,constituent_identities_json,source_identity"
    )
    assert (
        first["clock_artifacts"]["primary_sha256"]
        == hashlib.sha256(primary_a).hexdigest()
    )
    assert (
        first["clock_artifacts"]["controls_sha256"]
        == hashlib.sha256(controls_a).hexdigest()
    )
    assert set(first["control_overlap"]) == set(s.CONTROL_ORDER[1:])
    assert first["source_support_precedes_novelty"] is True
    assert first["novelty_comparator_market_or_outcome_artifacts_opened"] is False
    assert set(first) == s.REPORT_KEYS
    assert tuple(first["period_diagnostics"]) == s.PERIOD_ORDER
    delayed_report = first["support_audit"]["one_bar_delayed_entry_boundary_crossers"]
    assert set(delayed_report) == set(s.PERIOD_ORDER)
    s._validate_output_generation(
        s._json_bytes(first),
        primary_a,
        controls_a,
        source_frame=rows,
    )
    unknown = {**first, "unknown": False}
    unknown_core = {
        key: value for key, value in unknown.items() if key != "manifest_hash"
    }
    unknown["manifest_hash"] = s.canonical_hash(unknown_core)
    with pytest.raises(RuntimeError, match="report exact schema"):
        s._validate_output_generation(
            s._json_bytes(unknown),
            primary_a,
            controls_a,
            source_frame=rows,
        )
    missing_nested = json.loads(json.dumps(first))
    del missing_nested["registration"]["mode"]
    missing_core = {
        key: value for key, value in missing_nested.items() if key != "manifest_hash"
    }
    missing_nested["manifest_hash"] = s.canonical_hash(missing_core)
    with pytest.raises(RuntimeError, match="registration exact schema"):
        s._validate_output_generation(
            s._json_bytes(missing_nested),
            primary_a,
            controls_a,
            source_frame=rows,
        )
    bad_header = gzip.decompress(primary_a).replace(
        b"policy_id,control,window,",
        b"policy_id,control,window,unknown,",
        1,
    )
    with pytest.raises(RuntimeError, match="exact header"):
        s._validate_output_generation(
            s._json_bytes(first),
            s._deterministic_gzip(bad_header),
            controls_a,
            source_frame=rows,
        )
    core = {key: value for key, value in first.items() if key != "manifest_hash"}
    assert first["manifest_hash"] == s.canonical_hash(core)


@pytest.mark.parametrize(
    ("rows", "expected_passed", "expected_status", "expected_decision"),
    (
        (
            passing_frame(),
            True,
            "source_support_passed",
            "SOURCE_SUPPORT_PASS",
        ),
        (
            frame([source_row(0, "2024-01-01T00:00:00Z")]),
            False,
            "retired_before_novelty",
            "RETIRE_TUSI_168_UNCHANGED_BEFORE_NOVELTY",
        ),
    ),
)
def test_artifact_eligible_terminal_status_decision_mapping(
    rows: pd.DataFrame,
    expected_passed: bool,
    expected_status: str,
    expected_decision: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_audit = {
        "artifact_eligible": True,
        "source_csv_path": str(s.DEFAULT_SOURCE_CSV),
        "source_csv_sha256": "a" * 64,
        "source_csv_bytes": 1,
        "source_manifest_path": str(s.DEFAULT_SOURCE_MANIFEST),
        "source_manifest_sha256": "b" * 64,
        "source_manifest_hash": "c" * 64,
        "source_integrity": dict(builder.ZERO_SOURCE_INTEGRITY),
    }
    with pytest.raises(RuntimeError, match="synthetic source audit"):
        s.build_support_from_frame(
            rows,
            registration=registration(),
            source_audit=source_audit,
        )
    report, primary_bytes, control_bytes = s._build_support_from_frame(
        rows,
        registration=registration(),
        source_audit=source_audit,
        artifact_eligible=True,
    )
    assert report["artifact_eligible"] is True
    assert report["terminal"] is True
    assert report["support_passed"] is expected_passed
    assert report["status"] == expected_status
    assert report["decision"] == expected_decision
    monkeypatch.setattr(
        s,
        "_validate_production_report_provenance",
        lambda report, source_frame: None,
    )
    s._validate_output_generation(
        s._json_bytes(report),
        primary_bytes,
        control_bytes,
        source_frame=rows,
        artifact_eligible_required=True,
    )
    if expected_passed:
        downgraded = copy.deepcopy(report)
        downgraded["artifact_eligible"] = False
        downgraded["terminal"] = False
        downgraded["status"] = "synthetic_only_nonpublishable"
        downgraded["decision"] = "SYNTHETIC_ONLY_NO_SOURCE_SUPPORT_DECISION"
        downgraded["registration"]["mode"] = "injected"
        downgraded["source_contract"]["artifact_eligible"] = False
        core = {
            key: value
            for key, value in downgraded.items()
            if key != "manifest_hash"
        }
        downgraded["manifest_hash"] = s.canonical_hash(core)
        with pytest.raises(RuntimeError, match="authorization drift"):
            s._validate_output_generation(
                s._json_bytes(downgraded),
                primary_bytes,
                control_bytes,
                source_frame=rows,
                artifact_eligible_required=True,
            )


@pytest.mark.parametrize(
    "forgery",
    (
        "status",
        "decision",
        "terminal",
        "artifact_eligible",
        "support_passed",
        "append_gate",
        "append_hash",
        "coordinated_append",
        "coordinated_control_overlap",
        "coordinated_artifact_eligible",
        "source_rows_opened",
        "comparator_rows_opened",
        "novelty_opened",
    ),
)
def test_forged_report_cross_field_invariants_are_rejected(
    forgery: str,
) -> None:
    rows = passing_frame()
    report, primary_bytes, control_bytes = s.build_support_from_frame(
        rows,
        registration=registration(),
    )
    forged = copy.deepcopy(report)
    if forgery == "status":
        forged["status"] = "source_support_passed"
    elif forgery == "decision":
        forged["decision"] = "SOURCE_SUPPORT_PASS"
    elif forgery == "terminal":
        forged["terminal"] = True
    elif forgery == "artifact_eligible":
        forged["artifact_eligible"] = True
    elif forgery == "support_passed":
        forged["support_passed"] = False
    elif forgery == "append_gate":
        forged["support_checks"][
            "future_append_selection_differences_zero"
        ] = False
        forged["support_passed"] = False
    elif forgery == "append_hash":
        forged["future_append_selection_invariance"]["comparisons"]["primary"][
            "raw"
        ]["full_sha256"] = "0" * 64
    elif forgery == "coordinated_append":
        comparison = forged["future_append_selection_invariance"][
            "comparisons"
        ]["primary"]["raw"]
        comparison["full_rows"] = 999
        comparison["prefix_rows"] = 999
        comparison["full_sha256"] = "d" * 64
        comparison["prefix_sha256"] = "d" * 64
    elif forgery == "coordinated_control_overlap":
        overlap = forged["control_overlap"]["issue_only"]
        overlap["exact_entry_intersection"] = 0
        overlap["exact_entry_union"] = (
            forged["accepted_clock_counts"]["primary"]
            + forged["accepted_clock_counts"]["issue_only"]
        )
        overlap["exact_entry_jaccard"] = {
            "numerator": 0,
            "denominator": overlap["exact_entry_union"],
        }
        overlap["exact_parent_identity_intersection"] = 0
    elif forgery == "coordinated_artifact_eligible":
        forged["artifact_eligible"] = True
        forged["terminal"] = True
        forged["status"] = "source_support_passed"
        forged["decision"] = "SOURCE_SUPPORT_PASS"
        forged["registration"]["mode"] = "artifact"
        forged["source_contract"].update(
            {
                "artifact_eligible": True,
                "source_csv_path": str(s.DEFAULT_SOURCE_CSV),
                "source_csv_sha256": "a" * 64,
                "source_csv_bytes": 1,
                "source_manifest_path": str(s.DEFAULT_SOURCE_MANIFEST),
                "source_manifest_sha256": "b" * 64,
                "source_manifest_hash": "c" * 64,
            }
        )
    elif forgery == "source_rows_opened":
        forged["evidence_boundary"]["source_rows_opened"] += 1
    elif forgery == "comparator_rows_opened":
        forged["evidence_boundary"]["comparator_rows_opened"] = 1
    else:
        forged[
            "novelty_comparator_market_or_outcome_artifacts_opened"
        ] = True
    core = {
        key: value for key, value in forged.items() if key != "manifest_hash"
    }
    forged["manifest_hash"] = s.canonical_hash(core)
    with pytest.raises(RuntimeError, match="drift"):
        s._validate_output_generation(
            s._json_bytes(forged),
            primary_bytes,
            control_bytes,
            source_frame=rows,
        )


def test_publication_is_atomic_idempotent_and_rejects_mixed_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = tmp_path / "primary.csv.gz"
    controls = tmp_path / "controls.csv.gz"
    report = tmp_path / "report.json"
    rows = passing_frame()
    report_payload, primary_bytes, control_bytes = s.build_support_from_frame(
        rows,
        registration=registration(),
    )
    report_bytes = s._json_bytes(report_payload)
    real_fsync_directory = s._fsync_directory
    fsynced_directories: list[Path] = []

    def publish() -> dict[str, str]:
        return s.publish_outputs(
            primary_output=primary,
            controls_output=controls,
            report_output=report,
            primary_bytes=primary_bytes,
            control_bytes=control_bytes,
            report_bytes=report_bytes,
            source_frame=rows,
        )

    def tracked_fsync(directory: Path, *, directory_fd: int) -> None:
        fsynced_directories.append(directory)
        real_fsync_directory(directory, directory_fd=directory_fd)

    monkeypatch.setattr(s, "_fsync_directory", tracked_fsync)
    assert publish() == {
        "primary": "created",
        "controls": "created",
        "report": "created",
    }
    assert fsynced_directories == [tmp_path, tmp_path, tmp_path]
    assert publish() == {
        "primary": "verified_existing",
        "controls": "verified_existing",
        "report": "verified_existing",
    }
    report.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="generation drift"):
        publish()


def test_publication_rejects_relative_parent_traversal_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = passing_frame()
    report, primary_bytes, control_bytes = s.build_support_from_frame(
        rows,
        registration=registration(),
    )
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    primary_alias = Path("results") / ".." / s.DEFAULT_PRIMARY_OUTPUT

    with pytest.raises(RuntimeError, match="parent traversal alias"):
        s.publish_outputs(
            primary_output=primary_alias,
            controls_output=Path("synthetic-controls.csv.gz"),
            report_output=Path("synthetic-report.json"),
            primary_bytes=primary_bytes,
            control_bytes=control_bytes,
            report_bytes=s._json_bytes(report),
            source_frame=rows,
        )

    assert not (tmp_path / s.DEFAULT_PRIMARY_OUTPUT).exists()


def test_synthetic_publication_recognizes_absolute_canonical_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = passing_frame()
    report, primary_bytes, control_bytes = s.build_support_from_frame(
        rows,
        registration=registration(),
    )
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    canonical_outputs = (
        tmp_path / s.DEFAULT_PRIMARY_OUTPUT,
        tmp_path / s.DEFAULT_CONTROLS_OUTPUT,
        tmp_path / s.DEFAULT_REPORT_OUTPUT,
    )

    with pytest.raises(RuntimeError, match="eligibility authorization"):
        s.publish_outputs(
            primary_output=canonical_outputs[0],
            controls_output=canonical_outputs[1],
            report_output=canonical_outputs[2],
            primary_bytes=primary_bytes,
            control_bytes=control_bytes,
            report_bytes=s._json_bytes(report),
            source_frame=rows,
        )

    with pytest.raises(RuntimeError, match="mixed canonical output paths"):
        s.publish_outputs(
            primary_output=canonical_outputs[0],
            controls_output=tmp_path / "synthetic-controls.csv.gz",
            report_output=tmp_path / "synthetic-report.json",
            primary_bytes=primary_bytes,
            control_bytes=control_bytes,
            report_bytes=s._json_bytes(report),
            source_frame=rows,
        )

    assert not any(path.exists() for path in canonical_outputs)


def test_publication_rejects_symlinked_ancestor_alias(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    rows = passing_frame()
    report, primary_bytes, control_bytes = s.build_support_from_frame(
        rows,
        registration=registration(),
    )

    with pytest.raises(RuntimeError, match="output ancestor is unsafe"):
        s.publish_outputs(
            primary_output=alias_parent / "primary.csv.gz",
            controls_output=alias_parent / "controls.csv.gz",
            report_output=alias_parent / "report.json",
            primary_bytes=primary_bytes,
            control_bytes=control_bytes,
            report_bytes=s._json_bytes(report),
            source_frame=rows,
        )

    assert not list(real_parent.iterdir())


def test_publication_rejects_symlink_leaf_alias(tmp_path: Path) -> None:
    real_output = tmp_path / "real-primary.csv.gz"
    real_output.write_bytes(b"untouched")
    alias_output = tmp_path / "primary.csv.gz"
    alias_output.symlink_to(real_output)
    rows = passing_frame()
    report, primary_bytes, control_bytes = s.build_support_from_frame(
        rows,
        registration=registration(),
    )

    with pytest.raises(RuntimeError, match="output leaf is unsafe"):
        s.publish_outputs(
            primary_output=alias_output,
            controls_output=tmp_path / "controls.csv.gz",
            report_output=tmp_path / "report.json",
            primary_bytes=primary_bytes,
            control_bytes=control_bytes,
            report_bytes=s._json_bytes(report),
            source_frame=rows,
        )

    assert real_output.read_bytes() == b"untouched"
    assert alias_output.is_symlink()


def test_publication_rejects_duplicate_normalized_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = passing_frame()
    report, primary_bytes, control_bytes = s.build_support_from_frame(
        rows,
        registration=registration(),
    )
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    duplicate_relative = Path("synthetic") / "same.csv.gz"
    duplicate_absolute = tmp_path / duplicate_relative

    with pytest.raises(RuntimeError, match="paths must be distinct"):
        s.publish_outputs(
            primary_output=duplicate_relative,
            controls_output=duplicate_absolute,
            report_output=Path("synthetic") / "report.json",
            primary_bytes=primary_bytes,
            control_bytes=control_bytes,
            report_bytes=s._json_bytes(report),
            source_frame=rows,
        )

    assert not (tmp_path / "synthetic").exists()


def test_canonical_production_publication_uses_secure_normalized_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = passing_frame()
    source_audit = {
        "artifact_eligible": True,
        "source_csv_path": str(s.DEFAULT_SOURCE_CSV),
        "source_csv_sha256": "a" * 64,
        "source_csv_bytes": 1,
        "source_manifest_path": str(s.DEFAULT_SOURCE_MANIFEST),
        "source_manifest_sha256": "b" * 64,
        "source_manifest_hash": "c" * 64,
        "source_integrity": dict(builder.ZERO_SOURCE_INTEGRITY),
    }
    report, primary_bytes, control_bytes = s._build_support_from_frame(
        rows,
        registration=registration(),
        source_audit=source_audit,
        artifact_eligible=True,
    )
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        s,
        "_validate_production_report_provenance",
        lambda report, source_frame: None,
    )

    assert s.publish_outputs(
        primary_output=s.DEFAULT_PRIMARY_OUTPUT,
        controls_output=s.DEFAULT_CONTROLS_OUTPUT,
        report_output=s.DEFAULT_REPORT_OUTPUT,
        primary_bytes=primary_bytes,
        control_bytes=control_bytes,
        report_bytes=s._json_bytes(report),
        source_frame=rows,
    ) == {
        "primary": "created",
        "controls": "created",
        "report": "created",
    }
    assert (tmp_path / s.DEFAULT_PRIMARY_OUTPUT).read_bytes() == primary_bytes
    assert (tmp_path / s.DEFAULT_CONTROLS_OUTPUT).read_bytes() == control_bytes
    assert (tmp_path / s.DEFAULT_REPORT_OUTPUT).read_bytes() == s._json_bytes(
        report
    )


@pytest.mark.parametrize("failure_after_link", (1, 2, 3))
def test_publication_failure_after_each_link_rolls_back_durably(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_after_link: int,
) -> None:
    outputs = (
        tmp_path / "primary.csv.gz",
        tmp_path / "controls.csv.gz",
        tmp_path / "report.json",
    )
    rows = passing_frame()
    report, primary_bytes, control_bytes = s.build_support_from_frame(
        rows,
        registration=registration(),
    )
    real_link = s.os.link
    real_fsync_directory = s._fsync_directory
    link_calls = 0
    fsynced_directories: list[Path] = []

    def link_then_fail(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal link_calls
        link_calls += 1
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        if link_calls == failure_after_link:
            raise OSError(f"injected failure after link {link_calls}")

    def tracked_fsync(directory: Path, *, directory_fd: int) -> None:
        fsynced_directories.append(directory)
        real_fsync_directory(directory, directory_fd=directory_fd)

    monkeypatch.setattr(s.os, "link", link_then_fail)
    monkeypatch.setattr(s, "_fsync_directory", tracked_fsync)
    with pytest.raises(OSError, match="injected failure"):
        s.publish_outputs(
            primary_output=outputs[0],
            controls_output=outputs[1],
            report_output=outputs[2],
            primary_bytes=primary_bytes,
            control_bytes=control_bytes,
            report_bytes=s._json_bytes(report),
            source_frame=rows,
        )
    assert link_calls == failure_after_link
    assert not any(path.exists() for path in outputs)
    assert not list(tmp_path.glob(".*.tmp"))
    assert len(fsynced_directories) >= failure_after_link
    assert set(fsynced_directories) == {tmp_path}


def test_evaluate_failure_writes_no_outputs(tmp_path: Path) -> None:
    rows = passing_frame()
    csv_path, manifest_path, payload = write_source_artifacts(tmp_path, rows)
    changed = dict(payload)
    changed["source_csv_sha256"] = "0" * 64
    changed_core = {
        key: value for key, value in changed.items() if key != "manifest_hash"
    }
    changed["manifest_hash"] = s.canonical_hash(changed_core)
    manifest_path.write_bytes(builder.serialize_manifest(changed))
    prereg_path = tmp_path / "registration.json"
    prereg_path.write_text(json.dumps(registration()))
    outputs = [
        tmp_path / "primary.csv.gz",
        tmp_path / "controls.csv.gz",
        tmp_path / "report.json",
    ]
    with pytest.raises(RuntimeError, match="binding drift"):
        s.evaluate_and_write(
            source_csv=csv_path,
            source_manifest=manifest_path,
            preregistration=prereg_path,
            primary_output=outputs[0],
            controls_output=outputs[1],
            report_output=outputs[2],
            production=False,
            registration=registration(),
        )
    assert not any(path.exists() for path in outputs)


def test_production_preregistration_requires_bound_committed_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "prereg.json"
    artifact.write_text(json.dumps(registration()))
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(s, "PREREGISTRATION_SHA256", None)
    with pytest.raises(RuntimeError, match="SHA-256 is not bound"):
        s.validate_preregistration(artifact, production=True)


def test_production_preregistration_binding_matches_committed_artifact() -> None:
    payload = s.validate_preregistration(production=True)
    artifact = s.REPOSITORY_ROOT / s.DEFAULT_PREREGISTRATION

    assert s.sha256_file(artifact) == s.PREREGISTRATION_SHA256
    assert payload["manifest_hash"] == s.PREREGISTRATION_MANIFEST_HASH
    assert s.PREREGISTRATION_SHA256 == (
        builder.PREREGISTRATION_ARTIFACT_SHA256
    )
    assert s.PREREGISTRATION_MANIFEST_HASH == (
        builder.PREREGISTRATION_MANIFEST_HASH
    )
