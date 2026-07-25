from __future__ import annotations

from collections import OrderedDict
import csv
import gzip
import io
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from training import build_tracer4_tri_surface_relational_executor_support as b
from training import preregister_tracer4_tri_surface_relational_executor as p


def _physical_row(header: tuple[str, ...], **updates: object) -> list[str]:
    values = {name: "UNPROJECTED_SENTINEL" for name in header}
    values.update({name: str(value) for name, value in updates.items()})
    return [values[name] for name in header]


def _premium_row(date: str, *, valid: bool = True, poison: bool = False) -> list[str]:
    stamp = pd.Timestamp(date)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    values: dict[str, object] = {
        "date": stamp.isoformat(),
        "source_close_time": (stamp + pd.Timedelta(seconds=59, milliseconds=999)).isoformat(),
        "feature_available_time": (stamp + pd.Timedelta(seconds=61)).isoformat(),
        "source_valid": int(valid),
        "premium_open": "not-a-number" if poison else ("" if not valid else -0.0001),
        "premium_high": "not-a-number" if poison else ("" if not valid else 0.0002),
        "premium_low": "not-a-number" if poison else ("" if not valid else -0.0003),
        "premium_close": "not-a-number" if poison else ("" if not valid else 0.0001),
    }
    return _physical_row(p.PREMIUM_PHYSICAL_HEADER, **values)


def _leadership_row(
    date: str,
    *,
    signed: float = 2.0,
    quote: float = 10.0,
    source_complete: object = 1,
    feature_valid: object = 1,
    blank_numeric: bool = False,
) -> list[str]:
    stamp = pd.Timestamp(date)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    values: dict[str, object] = {
        "date": stamp.isoformat(),
        "feature_available_time_utc": (
            stamp + pd.Timedelta(minutes=5)
        ).isoformat(),
        "spot_quote_notional": quote,
        "um_quote_notional": quote,
        "spot_signed_quote_notional": signed,
        "um_signed_quote_notional": -signed,
        "spot_to_um_lagged_flow_response_bp": 1.0,
        "um_to_spot_lagged_flow_response_bp": -1.0,
        "open_basis_bp": 1.0,
        "close_basis_bp": 2.0,
        "source_complete": source_complete,
        "cross_venue_feature_valid": feature_valid,
    }
    if blank_numeric:
        for field in b.NUMERIC_COLUMNS["leadership"]:
            values[field] = ""
    return _physical_row(
        p.LEADERSHIP_PHYSICAL_HEADER,
        **values,
    )


def _aggtrade_row(date: str, *, first_offset_ms: int = 1, last_offset_ms: int = 299_999) -> list[str]:
    stamp = pd.Timestamp(date)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    epoch_ms = int(stamp.timestamp() * 1000)
    return _physical_row(
        p.AGGTRADE_PHYSICAL_HEADER,
        date=stamp.isoformat(),
        first_transact_time_ms=epoch_ms + first_offset_ms,
        last_transact_time_ms=epoch_ms + last_offset_ms,
        agg_trade_count=10,
        quote_notional=100.0,
        signed_quote_notional=25.0,
        micro_log_return=0.001,
        event_notional_hhi=0.2,
        normalized_effective_event_count=5.0,
        sign_flip_rate=0.3,
        max_same_sign_run_share=0.4,
        interarrival_mean_ms=10.0,
        interarrival_burstiness=0.1,
    )


def _projected_dict(
    header: tuple[str, ...],
    allowlist: tuple[str, ...],
    row: list[str],
) -> dict[str, str]:
    physical = dict(zip(header, row, strict=True))
    return {field: physical[field] for field in allowlist}


def _write_gzip_cut(
    root: Path,
    relative: str,
    columns: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(columns), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _tokens() -> OrderedDict[str, str]:
    return OrderedDict((name, values[0]) for name, values in p.TOKEN_SCHEMA)


def _state(
    boundary: str,
    *,
    sequence_ready: bool = False,
) -> b.PrimitiveState:
    tokens = _tokens()
    return b.PrimitiveState(
        boundary=pd.Timestamp(boundary),
        valid=True,
        invalid_reasons=(),
        primitives={"premium_change": 1.0},
        ranks={"premium_range": "MID"},
        tokens=tokens,
        line=p.canonical_line(tokens),
        rank_ready=True,
        sequence_ready=sequence_ready,
    )


def test_projection_stops_before_poisoned_cutoff_numeric_conversion() -> None:
    before = _premium_row("2023-12-31T23:59:00Z")
    cutoff = _premium_row("2024-01-01T00:00:00Z", poison=True)
    text, audit = b.deterministic_project_rows(
        name="premium",
        header=p.PREMIUM_PHYSICAL_HEADER,
        rows=[before, cutoff],
        allowlist=p.PREMIUM_ALLOWLIST,
    )
    parsed = list(csv.reader(io.StringIO(text)))
    assert len(parsed) == 2
    assert audit["cut_rows_written"] == 1
    assert audit["stopped_at"] == "2024-01-01T00:00:00Z"
    assert audit["post_2023_numeric_source_rows_opened"] == 0


def test_invalid_premium_row_is_preserved_as_invalid_not_numeric_neutral() -> None:
    invalid = _premium_row("2020-01-01T00:00:00Z", valid=False)
    text, audit = b.deterministic_project_rows(
        name="premium",
        header=p.PREMIUM_PHYSICAL_HEADER,
        rows=[invalid],
        allowlist=p.PREMIUM_ALLOWLIST,
    )
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert audit["cut_rows_written"] == 1
    assert parsed[0]["source_valid"] == "0"
    assert parsed[0]["premium_open"] == ""


def test_invalid_leadership_row_preserves_missing_numeric_cells() -> None:
    invalid = _leadership_row(
        "2020-01-01T00:00:00Z",
        source_complete=0,
        blank_numeric=True,
    )
    text, audit = b.deterministic_project_rows(
        name="leadership",
        header=p.LEADERSHIP_PHYSICAL_HEADER,
        rows=[invalid],
        allowlist=p.LEADERSHIP_ALLOWLIST,
    )
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert audit["cut_rows_written"] == 1
    assert parsed[0]["source_complete"] == "0"
    assert parsed[0]["spot_quote_notional"] == ""


def test_projection_rejects_nonbinary_flags_even_on_invalid_rows() -> None:
    malformed = _leadership_row(
        "2020-01-01T00:00:00Z",
        source_complete="yes",
        blank_numeric=True,
    )
    with pytest.raises(RuntimeError, match="non-binary"):
        b.deterministic_project_rows(
            name="leadership",
            header=p.LEADERSHIP_PHYSICAL_HEADER,
            rows=[malformed],
            allowlist=p.LEADERSHIP_ALLOWLIST,
        )


def test_projection_rejects_leadership_signed_notional_overflow() -> None:
    bad = _leadership_row("2020-01-01T00:00:00Z", signed=11.0, quote=10.0)
    with pytest.raises(RuntimeError, match="signed"):
        b.deterministic_project_rows(
            name="leadership",
            header=p.LEADERSHIP_PHYSICAL_HEADER,
            rows=[bad],
            allowlist=p.LEADERSHIP_ALLOWLIST,
        )


def test_projection_rejects_aggtrade_transaction_outside_bar() -> None:
    bad = _aggtrade_row("2020-01-01T00:00:00Z", last_offset_ms=300_000)
    with pytest.raises(RuntimeError, match="transaction"):
        b.deterministic_project_rows(
            name="aggtrade",
            header=p.AGGTRADE_PHYSICAL_HEADER,
            rows=[bad],
            allowlist=p.AGGTRADE_ALLOWLIST,
        )


def test_projection_requires_monotone_unique_dates() -> None:
    row = _leadership_row("2020-01-01T00:00:00Z")
    with pytest.raises(RuntimeError, match="monotone|duplicate"):
        b.deterministic_project_rows(
            name="leadership",
            header=p.LEADERSHIP_PHYSICAL_HEADER,
            rows=[row, row],
            allowlist=p.LEADERSHIP_ALLOWLIST,
        )


def test_cut_loader_rejects_malformed_flag_and_keeps_invalid_nan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(b, "REPOSITORY_ROOT", tmp_path)
    relative = "cuts/leadership.csv.gz"
    valid_invalid = _projected_dict(
        p.LEADERSHIP_PHYSICAL_HEADER,
        p.LEADERSHIP_ALLOWLIST,
        _leadership_row(
            "2020-01-01T00:00:00Z",
            source_complete=0,
            blank_numeric=True,
        ),
    )
    _write_gzip_cut(
        tmp_path,
        relative,
        p.LEADERSHIP_ALLOWLIST,
        [valid_invalid],
    )
    frame = b.load_cut_frame(
        "leadership", relative, p.LEADERSHIP_ALLOWLIST
    )
    assert frame.loc[0, "source_complete"] == False  # noqa: E712
    assert pd.isna(frame.loc[0, "spot_quote_notional"])

    valid_invalid["source_complete"] = "yes"
    _write_gzip_cut(
        tmp_path,
        "cuts/malformed.csv.gz",
        p.LEADERSHIP_ALLOWLIST,
        [valid_invalid],
    )
    with pytest.raises(RuntimeError, match="non-binary"):
        b.load_cut_frame(
            "leadership",
            "cuts/malformed.csv.gz",
            p.LEADERSHIP_ALLOWLIST,
        )

    forbidden_columns = (*p.LEADERSHIP_ALLOWLIST, "future_return")
    forbidden_row = dict(valid_invalid)
    forbidden_row["source_complete"] = "0"
    forbidden_row["future_return"] = "999"
    _write_gzip_cut(
        tmp_path,
        "cuts/forbidden.csv.gz",
        forbidden_columns,
        [forbidden_row],
    )
    with pytest.raises(RuntimeError, match="physical header"):
        b.load_cut_frame(
            "leadership",
            "cuts/forbidden.csv.gz",
            p.LEADERSHIP_ALLOWLIST,
        )


def test_controls_preserve_clocks_and_apply_exact_transform() -> None:
    dates = pd.date_range("2020-01-01", periods=80, freq="5min", tz="UTC")
    agg = pd.DataFrame(
        {
            "date": dates,
            "first_transact_time_ms": [int(x.timestamp() * 1000) + 1 for x in dates],
            "last_transact_time_ms": [int(x.timestamp() * 1000) + 2 for x in dates],
            "agg_trade_count": 10,
            "quote_notional": range(80),
            "signed_quote_notional": range(100, 180),
            "micro_log_return": range(200, 280),
            "event_notional_hhi": 0.2,
            "normalized_effective_event_count": 5.0,
            "sign_flip_rate": 0.3,
            "max_same_sign_run_share": 0.4,
            "interarrival_mean_ms": 10.0,
            "interarrival_burstiness": 0.1,
        }
    )
    rotated = b.rotate_aggtrade_monthly(agg, rows=37)
    assert rotated["date"].equals(agg["date"])
    assert rotated["first_transact_time_ms"].equals(agg["first_transact_time_ms"])
    assert rotated.loc[37, "quote_notional"] == agg.loc[0, "quote_notional"]
    assert rotated.loc[0, "quote_notional"] == agg.loc[len(agg) - 37, "quote_notional"]

    leadership = pd.DataFrame(
        {
            "date": dates[:2],
            "feature_available_time_utc": dates[:2] + pd.Timedelta(minutes=5),
            "spot_quote_notional": [1.0, 2.0],
            "um_quote_notional": [3.0, 4.0],
            "spot_signed_quote_notional": [5.0, 6.0],
            "um_signed_quote_notional": [7.0, 8.0],
            "spot_to_um_lagged_flow_response_bp": [9.0, 10.0],
            "um_to_spot_lagged_flow_response_bp": [11.0, 12.0],
            "open_basis_bp": [13.0, 14.0],
            "close_basis_bp": [15.0, 16.0],
            "source_complete": True,
            "cross_venue_feature_valid": True,
        }
    )
    swapped = b.swap_cash_perp(leadership)
    assert swapped["date"].equals(leadership["date"])
    assert swapped["spot_quote_notional"].equals(leadership["um_quote_notional"])
    assert swapped["open_basis_bp"].equals(-leadership["open_basis_bp"])


def test_monthly_rotation_never_crosses_month_and_stale_premium_keeps_clock() -> None:
    dates = pd.DatetimeIndex(
        [
            "2020-01-31T23:50:00Z",
            "2020-01-31T23:55:00Z",
            "2020-02-01T00:00:00Z",
            "2020-02-01T00:05:00Z",
        ]
    )
    agg = pd.DataFrame(
        {
            "date": dates,
            "first_transact_time_ms": range(4),
            "last_transact_time_ms": range(10, 14),
            "agg_trade_count": 10,
            "quote_notional": [10.0, 11.0, 20.0, 21.0],
            "signed_quote_notional": [1.0, 2.0, 3.0, 4.0],
            "micro_log_return": [1.0, 2.0, 3.0, 4.0],
            "event_notional_hhi": 0.2,
            "normalized_effective_event_count": 5.0,
            "sign_flip_rate": 0.3,
            "max_same_sign_run_share": 0.4,
            "interarrival_mean_ms": 10.0,
            "interarrival_burstiness": 0.1,
        }
    )
    rotated = b.rotate_aggtrade_monthly(agg, rows=1)
    assert rotated["date"].equals(agg["date"])
    assert rotated["quote_notional"].tolist() == [11.0, 10.0, 21.0, 20.0]

    premium_dates = pd.date_range(
        "2020-01-01", periods=1_442, freq="1min", tz="UTC"
    )
    premium = pd.DataFrame(
        {
            "date": premium_dates,
            "source_close_time": premium_dates
            + pd.Timedelta(seconds=59, milliseconds=999),
            "feature_available_time": premium_dates
            + pd.Timedelta(seconds=61),
            "source_valid": True,
            "premium_open": range(1_442),
            "premium_high": range(1_442),
            "premium_low": range(1_442),
            "premium_close": range(1_442),
        }
    )
    stale = b.stale_premium(premium, minutes=1_440)
    assert stale["date"].equals(premium["date"])
    assert stale["source_close_time"].equals(premium["source_close_time"])
    assert stale["feature_available_time"].equals(
        premium["feature_available_time"]
    )
    assert not stale.loc[:1_439, "source_valid"].any()
    assert stale.loc[1_440, "premium_open"] == premium.loc[0, "premium_open"]


def test_write_once_gzip_is_deterministic_and_rejects_drift(tmp_path: Path) -> None:
    path = tmp_path / "artifact.csv.gz"
    first = b.write_once_gzip_csv(path, ("a", "b"), ((1, 2),))
    second = b.write_once_gzip_csv(path, ("a", "b"), ((1, 2),))
    assert first == second
    with gzip.open(path, "rt") as handle:
        assert handle.read() == "a,b\n1,2\n"
    with pytest.raises(RuntimeError, match="write-once"):
        b.write_once_gzip_csv(path, ("a", "b"), ((2, 1),))


def test_official_projection_streams_deterministic_cutoff_safe_gzip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(b, "REPOSITORY_ROOT", tmp_path)
    source = "source/premium.csv.gz"
    _write_gzip_cut(
        tmp_path,
        source,
        p.PREMIUM_PHYSICAL_HEADER,
        [
            _projected_dict(
                p.PREMIUM_PHYSICAL_HEADER,
                p.PREMIUM_ALLOWLIST,
                _premium_row("2023-12-31T23:59:00Z"),
            ),
            _projected_dict(
                p.PREMIUM_PHYSICAL_HEADER,
                p.PREMIUM_ALLOWLIST,
                _premium_row("2024-01-01T00:00:00Z", poison=True),
            ),
        ],
    )
    output = "cuts/premium.csv.gz"
    first = b.project_source_cut(
        "premium", source, p.PREMIUM_ALLOWLIST, output
    )
    second = b.project_source_cut(
        "premium", source, p.PREMIUM_ALLOWLIST, output
    )
    assert first["cut_sha256"] == second["cut_sha256"]
    assert first["stopped_at"] == "2024-01-01T00:00:00Z"
    assert first["forbidden_columns_projected"] == 0
    assert b.gzip_mtime(output) == 0
    assert "source_contract_failure" not in first
    with gzip.open(tmp_path / output, "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["date"].startswith("2023-12-31")


def test_rank_and_sequence_readiness_use_strict_prior_and_do_not_skip_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_aggregate(
        boundary: pd.Timestamp,
        leadership: pd.DataFrame,
        aggtrade: pd.DataFrame,
        premium: pd.DataFrame,
    ) -> tuple[bool, tuple[str, ...], dict[str, float]]:
        del boundary, leadership, aggtrade, premium
        nonlocal calls
        index = calls
        calls += 1
        if index == 361:
            return False, ("synthetic_invalid",), {}
        value = float(index + 1)
        return True, (), {
            "cash_flow": 1.0,
            "leverage_flow": 1.0,
            "auction_flow": 1.0,
            "auction_return": 1.0,
            "sponsor_score": value,
            "participation_hhi": value,
            "effective_participation": value,
            "flow_flip": value,
            "flow_run": value,
            "arrival_burst": value,
            "arrival_wait": value,
            "basis_change": 1.0,
            "premium_change": 1.0,
            "premium_range": value,
        }

    monkeypatch.setattr(b, "aggregate_primitives", fake_aggregate)
    empty = pd.DataFrame({"date": pd.to_datetime([], utc=True)})
    start = pd.Timestamp("2020-01-01T04:00:00Z")
    states = b.build_states(
        empty,
        empty,
        empty,
        start=start,
        end=start + pd.Timedelta(hours=4 * 366),
    )
    assert states[359].rank_ready is False
    assert states[360].rank_ready is True
    assert states[361].valid is False
    assert states[362].sequence_ready is False
    assert states[363].sequence_ready is False
    assert states[364].sequence_ready is True


def test_sequence_signature_uses_oldest_to_newest_lines_with_terminal_lf() -> None:
    states = [
        _state("2020-01-01T04:00:00Z"),
        _state("2020-01-01T08:00:00Z"),
        _state("2020-01-01T12:00:00Z", sequence_ready=True),
    ]
    rows = b.make_token_rows(states)
    expected = (
        f"{states[0].line}\n{states[1].line}\n{states[2].line}\n"
    ).encode("utf-8")
    assert rows[2]["sequence_signature"] == b.sha256_bytes(expected)
    assert rows[2]["canonical_line"] == states[2].line


def test_append_replay_rebuilds_every_frozen_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = [
        _state("2020-01-01T04:00:00Z"),
        _state("2020-01-01T08:00:00Z"),
        _state("2020-01-01T12:00:00Z", sequence_ready=True),
    ]
    ends: list[pd.Timestamp] = []

    def fake_build_states(
        leadership: pd.DataFrame,
        aggtrade: pd.DataFrame,
        premium: pd.DataFrame,
        *,
        start: pd.Timestamp = b.SOURCE_START + pd.Timedelta(hours=4),
        end: pd.Timestamp = b.SOURCE_END,
    ) -> list[b.PrimitiveState]:
        del start
        ends.append(pd.Timestamp(end))
        assert (
            leadership["feature_available_time_utc"] <= pd.Timestamp(end)
        ).all()
        assert (
            aggtrade["date"] + pd.Timedelta(minutes=5)
            <= pd.Timestamp(end)
        ).all()
        assert (
            premium["feature_available_time"] <= pd.Timestamp(end)
        ).all()
        return [state for state in states if state.boundary < pd.Timestamp(end)]

    monkeypatch.setattr(b, "build_states", fake_build_states)
    dates = pd.DatetimeIndex(
        ["2020-01-01T00:00:00Z", "2025-01-01T00:00:00Z"]
    )
    leadership = pd.DataFrame(
        {
            "date": dates,
            "feature_available_time_utc": dates
            + pd.Timedelta(minutes=5),
        }
    )
    aggtrade = pd.DataFrame({"date": dates})
    premium = pd.DataFrame(
        {
            "date": dates,
            "feature_available_time": dates + pd.Timedelta(seconds=61),
        }
    )
    replay = b.append_replay_check(
        states, leadership, aggtrade, premium
    )
    assert replay["byte_identical"] is True
    assert ends == [
        pd.Timestamp(f"{year}-01-01T00:00:00Z")
        for year in (2021, 2022, 2023, 2024)
    ]


def test_protocol_guard_requires_preregistration_source_and_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_git(*args: str) -> SimpleNamespace:
        calls.append(args)
        if args[:2] == ("rev-parse", "HEAD"):
            return SimpleNamespace(returncode=0, stdout="a" * 40)
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(b, "_git", fake_git)
    monkeypatch.setattr(
        b,
        "sha256_file",
        lambda path: b.CONTRACT_SHA256
        if str(path) == b.CONTRACT_PATH
        else "unused",
    )
    monkeypatch.setattr(p, "assert_boundary_committed", lambda: None)
    assert b.assert_protocol_committed() == "a" * 40
    tracked_call = next(call for call in calls if call[0] == "ls-files")
    assert b.PREREGISTRATION_SOURCE_PATH in tracked_call
    assert b.PREREGISTRATION_TEST_PATH in tracked_call


def test_gate_evaluation_fails_identical_control_and_nonzero_evidence() -> None:
    yearly: dict[str, object] = {}
    for year in ("2020", "2021", "2022", "2023"):
        yearly[year] = {
            "core_valid_share": 1.0,
            "source_invalid_share": 0.0,
            "sequence_ready_core_valid_boundaries": 2100,
            "category_shares": {
                name: {values[0]: 0.5, values[1]: 0.5}
                for name, values in p.TOKEN_SCHEMA
            },
            "flow_buy_share": 0.5,
            "flow_sell_share": 0.5,
            "impact_followthrough_share": 0.5,
            "impact_absorption_share": 0.5,
            "sponsor_cash_share": 0.5,
            "sponsor_leverage_share": 0.5,
            "distinct_signatures": 500,
            "max_signature_share": 0.01,
        }
    for pair in ("jsd_2020_2021", "jsd_2021_2022", "jsd_2022_2023"):
        yearly[pair] = {name: 0.0 for name in p.TOKEN_COLUMNS}
    join = {
        year: {
            "leadership_join_share_5m": 1.0,
            "aggtrade_join_share_5m": 1.0,
            "premium_join_share_1m": 1.0,
        }
        for year in ("2020", "2021", "2022", "2023")
    }
    controls = {
        cid: {"canonical_line_stream_hash_differs": cid != p.CONTROL_IDS[0]}
        for cid in p.CONTROL_IDS
    }
    checks = b.evaluate_support_gates(
        source_contracts_ok=True,
        yearly=yearly,
        join=join,
        replay={"byte_identical": True},
        controls=controls,
        gates=p.build_manifest()["support_gates"],
        outcome_boundary=b.ZERO_OUTCOME_COUNTERS,
        rows=(),
    )
    assert checks[f"gate_12_control_{p.CONTROL_IDS[0]}_differs"] is False
    assert list(checks)[0].startswith("gate_01_")
    assert list(checks)[-1] == "gate_14_outcome_boundary_zero"
    gate_numbers = [int(name.split("_")[1]) for name in checks]
    assert gate_numbers == sorted(gate_numbers)

    mixed_join = {year: dict(values) for year, values in join.items()}
    mixed_join["2021"]["leadership_join_share_5m"] = 0.0
    mixed_yearly = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in yearly.items()
    }
    mixed_yearly["2020"]["core_valid_share"] = 0.0
    mixed_checks = b.evaluate_support_gates(
        source_contracts_ok=True,
        yearly=mixed_yearly,
        join=mixed_join,
        replay={"byte_identical": True},
        controls={
            cid: {"canonical_line_stream_hash_differs": True}
            for cid in p.CONTROL_IDS
        },
        gates=p.build_manifest()["support_gates"],
        outcome_boundary=b.ZERO_OUTCOME_COUNTERS,
        rows=(),
    )
    assert next(name for name, ok in mixed_checks.items() if not ok) == (
        "gate_02_2021_leadership_join_min"
    )

    crossed = dict(b.ZERO_OUTCOME_COUNTERS)
    crossed["future_return_rows_opened"] = 1
    crossed_checks = b.evaluate_support_gates(
        source_contracts_ok=False,
        yearly=yearly,
        join=join,
        replay={"byte_identical": True},
        controls={
            cid: {"canonical_line_stream_hash_differs": True}
            for cid in p.CONTROL_IDS
        },
        gates=p.build_manifest()["support_gates"],
        outcome_boundary=crossed,
        rows=(),
    )
    assert crossed_checks["gate_01_source_contracts_available"] is False
    assert crossed_checks["gate_14_outcome_boundary_zero"] is False


def test_source_contract_readiness_includes_cut_failure_and_forbidden_counter() -> None:
    allowlists = {
        "leadership": p.LEADERSHIP_ALLOWLIST,
        "aggtrade": p.AGGTRADE_ALLOWLIST,
        "premium": p.PREMIUM_ALLOWLIST,
    }
    source_audit = {
        name: {"sha256": f"{name}-parent"}
        for name in p.PRE2024_CUTS
    }
    cuts = {
        name: {
            "parent_sha256": f"{name}-parent",
            "output": output,
            "cut_header": list(allowlists[name]),
            "post_2023_numeric_source_rows_opened": 0,
            "forbidden_columns_projected": 0,
        }
        for name, output in p.PRE2024_CUTS.items()
    }
    manifest = {"cuts": cuts}
    assert b.source_contracts_ready(source_audit, manifest) is True
    cuts["premium"]["post_2023_numeric_source_rows_opened"] = 1
    assert b.source_contracts_ready(source_audit, manifest) is False
    cuts["premium"]["post_2023_numeric_source_rows_opened"] = 0
    cuts["aggtrade"]["source_contract_failure"] = "unexpected_later_row"
    assert b.source_contracts_ready(source_audit, manifest) is False
