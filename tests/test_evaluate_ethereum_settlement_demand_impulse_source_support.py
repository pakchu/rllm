from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from training import build_ethereum_settlement_demand_impulse_source as builder
from training import evaluate_ethereum_settlement_demand_impulse_source_support as s


def _epochs(
    count: int,
    *,
    start_epoch: int = 4_531,
    start_time: str = "2023-06-01T00:00:00Z",
    step: str = "12h",
) -> pd.DataFrame:
    times = pd.date_range(start_time, periods=count, freq=step, tz="UTC")
    medians: list[int] = []
    gas: list[str] = []
    for index in range(count):
        phase = (index // 2) % 2
        medians.append(100 if phase == 0 else 200)
        gas.append("0.2" if phase == 0 else "0.4")
    rows = []
    for index, epoch_id in enumerate(range(start_epoch, start_epoch + count)):
        start_block, end_block, confirmation_block = s.prereg.epoch_blocks(epoch_id)
        rows.append(
            {
                "epoch_id": epoch_id,
                "start_block": start_block,
                "end_block": end_block,
                "end_block_hash": "0x" + f"{epoch_id:064x}",
                "end_block_timestamp_utc": times[index],
                "confirmation_block": confirmation_block,
                "confirmation_block_hash": "0x"
                + f"{confirmation_block:064x}",
                "available_at_utc": times[index] + pd.Timedelta(seconds=61),
                "median_base_fee_wei_x2": medians[index],
                "base_fee_vector_sha256": hashlib.sha256(
                    str(epoch_id).encode()
                ).hexdigest(),
                "mean_gas_used_ratio_decimal": gas[index],
            }
        )
    return pd.DataFrame(rows, columns=s.SOURCE_COLUMNS)


def _clock(
    entry: str | pd.Timestamp,
    *,
    epoch_id: int = 5_000,
    side: str = "LONG",
    hold: pd.Timedelta = s.HOLD,
    control: str = "primary",
) -> dict[str, object]:
    timestamp = pd.Timestamp(entry)
    return {
        "policy_id": s.POLICY_ID,
        "control": control,
        "window": None,
        "signal_id": s.prereg.canonical_signal_id(epoch_id),
        "epoch_id": epoch_id,
        "source_hash": hashlib.sha256(str(epoch_id).encode()).hexdigest(),
        "source_available_at_utc": timestamp - pd.Timedelta(minutes=5),
        "entry_time_utc": timestamp,
        "exit_time_utc": timestamp + hold,
        "side": side,
        "rank_L": 135,
        "rank_E": 0,
        "rank_n": 180,
        "rank_numerator": 270,
        "rank_denominator": 360,
    }


def _clock_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=s.CLOCK_COLUMNS)


def _feature_row(
    *,
    epoch_id: int = 5_000,
    available: str = "2024-01-01T00:00:01Z",
    rank: Fraction = Fraction(3, 4),
) -> pd.DataFrame:
    values: dict[str, object] = {
        "epoch_id": epoch_id,
        "available_at_utc": pd.Timestamp(available),
        "source_hash": hashlib.sha256(str(epoch_id).encode()).hexdigest(),
    }
    for prefix in ("primary", "stale", "gas"):
        values.update(
            {
                f"{prefix}_sign": 1,
                f"{prefix}_ratio_num": 2,
                f"{prefix}_ratio_den": 1,
                f"{prefix}_rank_L": 135,
                f"{prefix}_rank_E": 0,
                f"{prefix}_rank_n": 180,
                f"{prefix}_rank": rank,
            }
        )
    return pd.DataFrame([values], columns=s.FEATURE_COLUMNS)


def test_preregistration_binding_and_explicit_evidence_boundary() -> None:
    assert s.PREREGISTRATION_SHA256 == (
        "2a481fc60044d3d468340457d50f92a91"
        "f2a52184a464e1a91badfb418bbcaba"
    )
    assert s.PREREGISTRATION_MANIFEST_HASH == (
        "d5279f95cc7b92757aa77ecbbc5835d8"
        "b1cc4ce34f5a81d6f279abdcf2fcfe8a"
    )
    assert s.validate_preregistration()["manifest_hash"] == (
        s.PREREGISTRATION_MANIFEST_HASH
    )
    assert s.EVIDENCE_BOUNDARY == {
        "official_ethereum_raw_rows_opened": 0,
        "official_ethereum_epoch_rows_opened": 0,
        "synthetic_epoch_rows_processed": 0,
        "comparator_rows_opened": 0,
        "market_rows_opened": 0,
        "funding_rows_opened": 0,
        "outcome_rows_opened": 0,
        "outcomes_computed": False,
        "network_calls": 0,
    }


def test_source_schema_decimal_and_outcome_columns_fail_closed() -> None:
    rows = _epochs(4)
    validated = s.validate_epoch_frame(rows)
    assert validated["mean_gas_used_ratio_decimal"].tolist() == [
        Decimal("0.2"),
        Decimal("0.2"),
        Decimal("0.4"),
        Decimal("0.4"),
    ]
    with pytest.raises(RuntimeError, match="schema drift"):
        s.validate_epoch_frame(rows.assign(close=1))
    invalid = rows.copy()
    invalid.loc[0, "mean_gas_used_ratio_decimal"] = "NaN"
    with pytest.raises(RuntimeError, match="gas ratio"):
        s.validate_epoch_frame(invalid)


def test_future_builder_manifest_contract_is_consumed_without_other_rows(
    tmp_path: Path,
) -> None:
    seal_core = {
        "protocol_version": (
            "ethereum_settlement_demand_impulse_synthetic_protocol_seal_v1"
        ),
        "policy_id": s.POLICY_ID,
        "mode": "synthetic_only",
        "protocol_paths": [path.as_posix() for path in builder.PROTOCOL_PATHS],
    }
    seal = {**seal_core, "seal_hash": s.canonical_hash(seal_core)}
    raw_path = tmp_path / "raw.ndjson.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        handle.write('{"request_index":0}\n')
    epoch_path = tmp_path / "epochs.csv.gz"
    _epochs(4).to_csv(epoch_path, index=False, compression="gzip")
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    epoch_sha = hashlib.sha256(epoch_path.read_bytes()).hexdigest()
    core = {
        "protocol_version": builder.PROTOCOL_VERSION,
        "policy_id": s.POLICY_ID,
        "status": "complete_outcome_blind_source_replay",
        "claim": {
            "path": None,
            "sha256": s.canonical_hash(
                {
                    "mode": "synthetic_unpublished",
                    "outputs": [
                        str(raw_path),
                        str(epoch_path),
                        str(tmp_path / "manifest.json"),
                    ],
                    "pre_replay_protocol_seal_hash": seal["seal_hash"],
                }
            ),
            "synthetic_unpublished": True,
        },
        "preregistration": {
            "path": str(builder.PREREGISTRATION_PATH),
            "sha256": s.PREREGISTRATION_SHA256,
            "manifest_hash": s.PREREGISTRATION_MANIFEST_HASH,
        },
        "pre_replay_protocol_seal": seal,
        "source_builder": {
            "path": str(s.SOURCE_BUILDER_PATH),
            "sha256": s.sha256_file(s.SOURCE_BUILDER_PATH),
        },
        "transports": list(builder.TRANSPORTS),
        "rpc": {
            "methods": list(builder.RPC_METHODS),
            "attempts_per_request": 1,
            "retry": False,
            "backoff": False,
            "fallback": False,
            "resume": False,
            "request_chunk_blocks": builder.REQUEST_CHUNK_BLOCKS,
            "fee_history_requests_per_transport": builder.REQUEST_COUNT,
            "boundary_header_requests_per_transport": (
                builder.BOUNDARY_HEADER_REQUESTS
            ),
            "finalized_header_requests_per_transport": 1,
            "epoch_and_confirmation_header_requests_per_transport": (
                builder.EPOCH_HEADER_REQUESTS
            ),
            "total_requests_per_transport": (
                builder.TOTAL_RPC_REQUESTS_PER_TRANSPORT
            ),
        },
        "range": {
            "first_requested_block": builder.FIRST_REQUESTED_BLOCK,
            "last_requested_block": builder.LAST_REQUESTED_BLOCK,
            "last_retained_block": builder.LAST_RETAINED_BLOCK,
            "terminal_padding_blocks_requested": builder.TERMINAL_PADDING_BLOCKS,
            "terminal_padding_first_block": builder.LAST_RETAINED_BLOCK + 1,
            "terminal_padding_last_block": builder.LAST_REQUESTED_BLOCK,
            "terminal_padding_disposition": (
                "discarded_before_epoch_normalization"
            ),
            "terminal_padding_entered_normalized_epochs": 0,
            "last_request_before_frozen_2026_06_boundary": True,
        },
        "epochs": {
            "first_epoch_id": builder.FIRST_EPOCH_ID,
            "last_epoch_id": builder.LAST_EPOCH_ID,
            "epoch_size_blocks": builder.EPOCH_SIZE_BLOCKS,
            "rows": 4,
            "confirmation_blocks_after_end": builder.CONFIRMATION_BLOCKS,
            "base_fee_vector_sha256_implementation": (
                "training.preregister_ethereum_settlement_demand_impulse."
                "base_fee_vector_sha256"
            ),
            "gas_ratio_arithmetic": "decimal",
            "gas_ratio_decimal_precision": builder.GAS_RATIO_DECIMAL_PRECISION,
        },
        "validation": {
            "chain_id": builder.CHAIN_ID,
            "boundary_audit": s._expected_boundary_audit(),
            "common_finalized_head": builder.LAST_CONFIRMATION_BLOCK,
            "common_finalized_head_hash": "0x" + "3" * 64,
            "common_finalized_head_at_or_after_last_confirmation": True,
            "dual_provider_response_differences": 0,
            "shortened_responses": 0,
            "next_base_fee_overlap_differences": 0,
            "epoch_end_header_differences": 0,
            "confirmation_header_differences": 0,
        },
        "outputs": {
            "raw_chunks": {
                "path": str(raw_path),
                "format": "deterministic gzip NDJSON",
                "rows": 1,
                "bytes": len(raw_path.read_bytes()),
                "sha256": raw_sha,
            },
            "normalized_epochs": {
                "path": str(epoch_path),
                "format": "deterministic gzip CSV",
                "rows": 4,
                "columns": list(s.SOURCE_COLUMNS),
                "bytes": len(epoch_path.read_bytes()),
                "sha256": epoch_sha,
            },
            "manifest": {"path": str(tmp_path / "manifest.json")},
        },
        "outcome_boundary": {
            "ethereum_source_values_opened": True,
            "btc_market_rows_opened": 0,
            "comparator_rows_opened": 0,
            "funding_rows_opened": 0,
            "return_or_pnl_rows_opened": 0,
            "outcomes_opened": False,
        },
    }
    payload = {**core, "manifest_hash": s.canonical_hash(core)}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(
        (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode()
    )
    frame, audit = s.load_synthetic_source_artifacts(
        manifest_path=manifest_path,
        raw_path=raw_path,
        epoch_path=epoch_path,
        expected_raw_rows=1,
        expected_epoch_rows=4,
    )
    assert list(frame.columns) == list(s.SOURCE_COLUMNS)
    assert audit["epoch_csv_sha256"] == epoch_sha
    assert audit["raw_source_rows_decoded"] == 1
    assert audit["epoch_csv_rows_decoded"] == 4
    assert audit["dual_replay_differences"] == 0
    assert audit["artifact_eligible"] is False

    payload["outputs"]["raw_chunks"]["rows"] = 2
    changed_core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    payload["manifest_hash"] = s.canonical_hash(changed_core)
    manifest_path.write_bytes(
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    )
    with pytest.raises(RuntimeError, match="decoded row count"):
        s.load_synthetic_source_artifacts(
            manifest_path=manifest_path,
            raw_path=raw_path,
            epoch_path=epoch_path,
            expected_raw_rows=2,
            expected_epoch_rows=4,
        )

    del payload["outputs"]["raw_chunks"]["sha256"]
    changed_core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    payload["manifest_hash"] = s.canonical_hash(changed_core)
    manifest_path.write_bytes(
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    )
    with pytest.raises(RuntimeError, match="raw source output schema"):
        s.load_synthetic_source_artifacts(
            manifest_path=manifest_path,
            raw_path=raw_path,
            epoch_path=epoch_path,
            expected_raw_rows=2,
            expected_epoch_rows=4,
        )


def test_real_builder_manifest_round_trips_through_support_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder, "REQUEST_COUNT", 1)
    monkeypatch.setattr(builder, "EPOCH_COUNT", 4)
    monkeypatch.setattr(builder, "LAST_EPOCH_ID", builder.FIRST_EPOCH_ID + 3)
    raw_path = tmp_path / "raw.ndjson.gz"
    epoch_path = tmp_path / "epochs.csv.gz"
    manifest_path = tmp_path / "manifest.json"
    raw_payload = builder._canonical_json_bytes(
        {"request_index": 0},
        trailing_lf=True,
    )
    raw_bytes = builder._deterministic_gzip(raw_payload)
    epoch_bytes = builder._csv_gzip(
        _epochs(4).to_dict(orient="records")
    )
    raw_path.write_bytes(raw_bytes)
    epoch_path.write_bytes(epoch_bytes)
    cfg = builder._SyntheticConfig(
        raw_output=str(raw_path),
        epoch_output=str(epoch_path),
        manifest_output=str(manifest_path),
    )
    seal, claim = builder._synthetic_bindings(cfg)
    manifest = builder._manifest(
        cfg=cfg,
        raw_size=len(raw_bytes),
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        epoch_bytes=epoch_bytes,
        finalized=builder.Header(
            builder.LAST_CONFIRMATION_BLOCK,
            "0x" + "3" * 64,
            "0x" + "2" * 64,
            2_000_000_000,
        ),
        boundary_audit=s._expected_boundary_audit(),
        builder_sha256=s.sha256_file(s.SOURCE_BUILDER_PATH),
        claim_binding=claim,
        pre_replay_protocol_seal=seal,
    )
    manifest_path.write_bytes(
        builder._canonical_json_bytes(manifest, trailing_lf=True)
    )

    frame, audit = s.load_synthetic_source_artifacts(
        manifest_path=manifest_path,
        raw_path=raw_path,
        epoch_path=epoch_path,
        expected_raw_rows=1,
        expected_epoch_rows=4,
    )
    assert list(frame.columns) == list(s.SOURCE_COLUMNS)
    assert len(frame) == 4
    assert audit["artifact_eligible"] is False
    assert audit["raw_source_rows_decoded"] == 1
    assert audit["epoch_csv_rows_decoded"] == 4
    assert audit["source_manifest_hash"] == manifest["manifest_hash"]
    assert audit["pre_replay_protocol_seal"] == seal
    assert audit["replay_claim"] == claim


def test_exact_180_excludes_current_and_ties_have_exact_midrank() -> None:
    features = s.build_features(_epochs(183))
    first_ranked = features.loc[features["epoch_id"].eq(4_713)].iloc[0]
    assert first_ranked["primary_rank_n"] == 180
    assert first_ranked["primary_rank_L"] == 0
    assert first_ranked["primary_rank_E"] == 180
    assert first_ranked["primary_rank"] == Fraction(1, 2)
    assert first_ranked["gas_rank"] == Fraction(1, 2)
    prior = features.loc[features["epoch_id"].lt(4_713)]
    assert prior["primary_rank"].isna().all()


def test_rank_boundary_is_inclusive_and_below_boundary_is_rejected() -> None:
    at_boundary = s.raw_candidates(_feature_row(), "primary")
    assert len(at_boundary) == 1
    assert at_boundary.iloc[0]["rank_numerator"] == 270
    assert at_boundary.iloc[0]["rank_denominator"] == 360
    below = s.raw_candidates(
        _feature_row(rank=Fraction(134, 180)), "primary"
    )
    assert below.empty


def test_entry_wait_nonoverlap_and_split_crossing() -> None:
    candidate = s.raw_candidates(_feature_row(), "primary").iloc[0]
    assert candidate["entry_time_utc"] == pd.Timestamp("2024-01-01T00:10:00Z")
    crossing = _clock_frame(
        [
            _clock(
                "2024-12-31T12:05:00Z",
                hold=pd.Timedelta(hours=13),
                epoch_id=5_001,
            ),
            _clock("2025-01-01T00:05:00Z", epoch_id=5_002),
            _clock("2025-01-01T12:05:00Z", epoch_id=5_003),
            _clock("2025-01-02T00:05:00Z", epoch_id=5_004),
        ]
    )
    accepted = s.reserve_nonoverlap(crossing)
    assert accepted["epoch_id"].tolist() == [5_002, 5_004]
    assert accepted["window"].tolist() == ["future25", "future25"]
    boundary = s.reserve_nonoverlap(
        _clock_frame(
            [
                _clock(
                    "2024-12-31T00:00:00Z",
                    epoch_id=5_005,
                )
            ]
        )
    )
    assert boundary.iloc[0]["window"] == "selection"


def test_independent_and_same_parent_controls_are_frozen() -> None:
    features = _feature_row()
    for control in s.INDEPENDENT_CONTROLS:
        assert len(s.raw_candidates(features, control)) == 1
    primary = s.reserve_nonoverlap(s.raw_candidates(features, "primary"))
    controls, _ = s.build_controls(features)
    assert controls["exact_direction_flip"]["side"].tolist() == ["SHORT"]
    assert controls["constant_long"]["side"].tolist() == ["LONG"]
    assert controls["constant_short"]["side"].tolist() == ["SHORT"]
    assert controls["deterministic_random_side"]["side"].tolist() == [
        s.prereg.deterministic_random_side(5_000)
    ]
    delayed = controls["one_bar_delayed_entry"]
    assert delayed["signal_id"].tolist() == primary["signal_id"].tolist()
    assert delayed["entry_time_utc"].tolist() == (
        primary["entry_time_utc"] + s.BAR
    ).tolist()


def test_delayed_parent_preserves_window_and_rows_across_calendar_boundaries() -> None:
    primary = s.reserve_nonoverlap(
        _clock_frame(
            [
                _clock("2024-12-31T00:00:00Z", epoch_id=5_010),
                _clock("2025-12-31T00:00:00Z", epoch_id=5_011),
                _clock("2026-05-31T00:00:00Z", epoch_id=5_012),
            ]
        )
    )
    assert primary["window"].tolist() == [
        "selection",
        "future25",
        "future26",
    ]
    delayed = s._parent_control(primary, "one_bar_delayed_entry")
    assert delayed["epoch_id"].tolist() == primary["epoch_id"].tolist()
    assert delayed["signal_id"].tolist() == primary["signal_id"].tolist()
    assert delayed["window"].tolist() == primary["window"].tolist()
    assert delayed["entry_time_utc"].tolist() == (
        primary["entry_time_utc"] + s.BAR
    ).tolist()
    assert delayed["exit_time_utc"].tolist() == (
        primary["exit_time_utc"] + s.BAR
    ).tolist()
    assert delayed.iloc[-1]["exit_time_utc"] > s.FULL_END


def test_stale_uses_e_minus_1_e_minus_3_but_not_before_e_availability() -> None:
    features = s.build_features(_epochs(184))
    stale = s.raw_candidates(features, "base_fee_one_epoch_stale")
    # Tied ranks do not trigger, but the feature's availability is always e.
    ranked = features.loc[features["stale_rank_n"].eq(180)].iloc[0]
    assert ranked["available_at_utc"] == _epochs(184).iloc[
        int(ranked["epoch_id"]) - 4_531
    ]["available_at_utc"]
    assert stale.empty


def test_future_append_selection_invariance() -> None:
    rows = _epochs(
        400,
        start_time="2024-06-01T00:00:00Z",
        step="2d",
    )
    passed, report = s.future_append_selection_invariance(rows)
    assert passed
    assert report["full_rebuild_selection_sha256"] == report[
        "prefix_rebuild_selection_sha256"
    ]


def _passing_primary() -> pd.DataFrame:
    timestamps: list[pd.Timestamp] = []
    for month in pd.period_range("2023-06", "2026-05", freq="M"):
        for day in (1, 10, 20):
            timestamps.append(
                pd.Timestamp(f"{month.year}-{month.month:02d}-{day:02d}T00:00:00Z")
            )
    rows = [
        _clock(
            timestamp,
            epoch_id=4_600 + index,
            side="LONG" if index % 2 else "SHORT",
        )
        for index, timestamp in enumerate(timestamps)
    ]
    return s.reserve_nonoverlap(_clock_frame(rows))


def test_support_gates_pass_and_fail_including_exact_control_metrics() -> None:
    primary = _passing_primary()
    empty = pd.DataFrame(columns=s.CLOCK_COLUMNS)
    controls = {
        "primary": primary,
        "base_fee_one_epoch_stale": empty,
        "gas_utilization_only": empty,
        "base_fee_no_tail": empty,
    }
    audit, checks, metrics = s.support_checks(primary, controls=controls)
    assert all(checks.values())
    assert audit["maximum_same_side_run"] == 1
    assert metrics["gas_utilization_only"]["exact_entry_jaccard"][
        "numerator"
    ] == 0

    identical = dict(controls)
    identical["base_fee_no_tail"] = primary.copy()
    _, failed, _ = s.support_checks(primary, controls=identical)
    assert failed["base_fee_no_tail_exact_entry_jaccard_strict"] is False
    assert failed["base_fee_no_tail_candidate_24h_containment_strict"] is False

    concentrated = primary.copy()
    concentrated["side"] = "LONG"
    _, failed, _ = s.support_checks(concentrated, controls=controls)
    assert failed["maximum_same_side_run"] is False
    assert failed["selection_each_side_min"] is False


def test_report_is_terminal_deterministic_outcome_blind_and_gzip_is_stable() -> None:
    rows = _epochs(184)
    first, primary_a, controls_a = s.build_support_from_frame(rows)
    second, primary_b, controls_b = s.build_support_from_frame(rows)
    assert first == second
    assert primary_a == primary_b
    assert controls_a == controls_b
    assert first["status"] == "synthetic_only_nonpublishable"
    assert first["terminal"] is False
    assert first["artifact_eligible"] is False
    assert first["later_stage_artifacts_opened"] is False
    assert first["evidence_boundary"]["comparator_rows_opened"] == 0
    assert first["evidence_boundary"]["official_ethereum_epoch_rows_opened"] == 0
    assert first["evidence_boundary"]["synthetic_epoch_rows_processed"] == 184
    assert gzip.decompress(primary_a).startswith(b"policy_id,control,window")
    core = {key: value for key, value in first.items() if key != "manifest_hash"}
    assert first["manifest_hash"] == s.canonical_hash(core)


def test_write_once_creates_reverifies_and_rejects_drift(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    assert s.write_once(output, b"first") == "created"
    assert s.write_once(output, b"first") == "verified_existing"
    with pytest.raises(RuntimeError, match="noncanonical"):
        s.write_once(output, b"second")


def test_source_support_attempt_claim_precedes_rows_and_forbids_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(s, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(s, "_assert_protocol_committed", lambda: None)
    monkeypatch.setattr(s, "validate_preregistration", lambda: {})

    def fail_after_claim() -> tuple[pd.DataFrame, dict[str, object]]:
        calls.append("source_rows")
        claim_path = tmp_path / s.DEFAULT_ATTEMPT_CLAIM
        assert claim_path.is_file()
        assert s.load_attempt_claim()["path"] == str(s.DEFAULT_ATTEMPT_CLAIM)
        raise RuntimeError("synthetic source-row failure")

    monkeypatch.setattr(s, "load_source_manifest", fail_after_claim)
    with pytest.raises(RuntimeError, match="synthetic source-row failure"):
        s.write_support()

    assert calls == ["source_rows"]
    assert (tmp_path / s.DEFAULT_ATTEMPT_CLAIM).is_file()
    assert not (tmp_path / s.DEFAULT_REPORT_OUTPUT).exists()
    with pytest.raises(RuntimeError, match="lacks a complete generation"):
        s.write_support()
    assert calls == ["source_rows"]


def test_support_publication_is_report_last_and_resumes_canonical_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = tmp_path / "primary.csv.gz"
    controls = tmp_path / "controls.csv.gz"
    report = tmp_path / "report.json"
    primary_bytes = b"primary-generation-a"
    control_bytes = b"control-generation-a"
    report_payload = {
        "clock_artifacts": {
            "primary_sha256": hashlib.sha256(primary_bytes).hexdigest(),
            "controls_sha256": hashlib.sha256(control_bytes).hexdigest(),
        }
    }
    report_core = dict(report_payload)
    report_payload["manifest_hash"] = s.canonical_hash(report_core)
    report_bytes = s._json_bytes(report_payload)

    primary.write_bytes(primary_bytes)
    linked: list[Path] = []
    events: list[tuple[str, Path]] = []
    real_link = s.os.link

    def recording_link(source: Path, target: Path) -> None:
        linked.append(Path(target))
        events.append(("link", Path(target)))
        real_link(source, target)

    monkeypatch.setattr(s.os, "link", recording_link)
    monkeypatch.setattr(
        s,
        "_fsync_directory",
        lambda path: events.append(("fsync", Path(path))),
    )
    statuses = s.publish_support_transaction(
        report_output=report,
        primary_output=primary,
        controls_output=controls,
        report_bytes=report_bytes,
        primary_bytes=primary_bytes,
        control_bytes=control_bytes,
    )
    assert statuses == {
        "primary_clock_status": "verified_existing",
        "control_clocks_status": "created",
        "report_status": "created",
    }
    assert linked == [controls, report]
    assert events == [
        ("link", controls),
        ("fsync", tmp_path),
        ("link", report),
        ("fsync", tmp_path),
    ]
    assert report.read_bytes() == report_bytes
    assert controls.stat().st_mode & 0o777 == 0o444
    assert report.stat().st_mode & 0o777 == 0o444

    assert s.publish_support_transaction(
        report_output=report,
        primary_output=primary,
        controls_output=controls,
        report_bytes=report_bytes,
        primary_bytes=primary_bytes,
        control_bytes=control_bytes,
    ) == {
        "primary_clock_status": "verified_existing",
        "control_clocks_status": "verified_existing",
        "report_status": "verified_existing",
    }


def test_support_publication_rolls_back_attempt_and_rejects_mixed_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = tmp_path / "primary.csv.gz"
    controls = tmp_path / "controls.csv.gz"
    report = tmp_path / "report.json"
    primary_bytes = b"primary"
    control_bytes = b"controls"
    core = {
        "clock_artifacts": {
            "primary_sha256": hashlib.sha256(primary_bytes).hexdigest(),
            "controls_sha256": hashlib.sha256(control_bytes).hexdigest(),
        }
    }
    report_bytes = s._json_bytes({**core, "manifest_hash": s.canonical_hash(core)})
    real_link = s.os.link
    calls = 0

    def fail_second(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic transaction failure")
        real_link(source, target)

    monkeypatch.setattr(s.os, "link", fail_second)
    with pytest.raises(OSError, match="synthetic"):
        s.publish_support_transaction(
            report_output=report,
            primary_output=primary,
            controls_output=controls,
            report_bytes=report_bytes,
            primary_bytes=primary_bytes,
            control_bytes=control_bytes,
        )
    assert not primary.exists()
    assert not controls.exists()
    assert not report.exists()

    monkeypatch.setattr(s.os, "link", real_link)
    primary.write_bytes(b"different-generation")
    with pytest.raises(RuntimeError, match="mixed generation"):
        s.publish_support_transaction(
            report_output=report,
            primary_output=primary,
            controls_output=controls,
            report_bytes=report_bytes,
            primary_bytes=primary_bytes,
            control_bytes=control_bytes,
        )
    assert not controls.exists()
    assert not report.exists()


def test_report_completion_marker_cannot_exist_without_both_clocks(
    tmp_path: Path,
) -> None:
    primary_bytes = b"primary"
    control_bytes = b"controls"
    core = {
        "clock_artifacts": {
            "primary_sha256": hashlib.sha256(primary_bytes).hexdigest(),
            "controls_sha256": hashlib.sha256(control_bytes).hexdigest(),
        }
    }
    report_bytes = s._json_bytes({**core, "manifest_hash": s.canonical_hash(core)})
    report = tmp_path / "report.json"
    report.write_bytes(report_bytes)
    with pytest.raises(RuntimeError, match="completion marker"):
        s.publish_support_transaction(
            report_output=report,
            primary_output=tmp_path / "primary.csv.gz",
            controls_output=tmp_path / "controls.csv.gz",
            report_bytes=report_bytes,
            primary_bytes=primary_bytes,
            control_bytes=control_bytes,
        )


def test_production_loader_has_no_path_fallback_and_synthetic_loader_is_explicit(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError):
        s.load_source_manifest(tmp_path / "manifest.json")
    assert callable(s.load_synthetic_source_artifacts)
    assert s.DEFAULT_SOURCE_MANIFEST == builder.DEFAULT_MANIFEST_OUTPUT
    assert s.DEFAULT_RAW_SOURCE == builder.DEFAULT_RAW_OUTPUT
    assert s.DEFAULT_EPOCH_SOURCE == builder.DEFAULT_EPOCH_OUTPUT
