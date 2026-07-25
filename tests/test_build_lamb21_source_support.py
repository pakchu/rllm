from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd
import pytest

s = pytest.importorskip("training.build_lamb21_source_support")


BOUNDARIES = pd.date_range(
    "2020-01-01T00:00:00Z",
    "2024-01-01T00:00:00Z",
    freq="8h",
    inclusive="left",
)

EXPECTED_TOKEN_COLUMNS = [
    "boundary_time",
    "core_source_valid",
    "rank_ready",
    "sequence_ready",
    "h41_impulse",
    "rrp_impulse",
    "macro_sponsorship",
    "macro_age",
    "lattice_relation",
    "lattice_concentration",
    "cascade_impact",
    "cascade_intensity",
    "micro_braid",
    "macro_transition",
    "micro_transition",
]

EXPECTED_FORBIDDEN_COUNTERS = [
    "execution_market_rows_opened",
    "funding_rows_opened",
    "future_return_rows_opened",
    "reward_rows_built",
    "model_rows_built",
    "trades_built",
    "pnl_values_computed",
    "cagr_values_computed",
    "mdd_values_computed",
    "post_2023_source_rows_opened",
]


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    fn = getattr(s, name, None)
    assert callable(fn), f"expected API: {name}"
    return fn(*args, **kwargs)


def _h41_frame(levels: list[float] | None = None) -> pd.DataFrame:
    levels = levels or [1.0, 1.5, 1.25, 2.0]
    releases = pd.date_range("2019-12-12", periods=len(levels), freq="7D")
    return pd.DataFrame(
        {
            "release_date": releases.strftime("%Y-%m-%d"),
            "observation_date": (releases - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "available_at_utc": [
                pd.Timestamp(day, tz="America/New_York").tz_convert("UTC").isoformat().replace("+00:00", "Z")
                for day in releases
            ],
            "net_liquidity_usd_millions": [str(value) for value in levels],
        }
    )


def _rrp_frame(complete: list[bool] | None = None) -> pd.DataFrame:
    complete = complete or [True, True, False, True, True]
    ops = pd.date_range("2019-12-27", periods=len(complete), freq="D")
    rows: list[dict[str, str]] = []
    for index, is_complete in enumerate(complete):
        rows.append(
            {
                "operation_date": ops[index].strftime("%Y-%m-%d"),
                "result_available_at_utc": (ops[index] + pd.Timedelta(hours=18)).tz_localize("UTC").isoformat().replace("+00:00", "Z"),
                "total_amount_accepted_usd": f"{1000 + index}.0" if is_complete else "",
                "participating_counterparties": str(index + 2) if is_complete else "",
                "accepted_counterparties": str(index + 1) if is_complete else "",
                "source_complete": "true" if is_complete else "false",
                "quarantine_reason": "" if is_complete else "late preliminary source",
            }
        )
    return pd.DataFrame(rows)


def _micro_frame(
    periods: int = 144,
    *,
    start: str = "2020-01-01T00:00:00Z",
    gap_positions: set[int] | None = None,
    observed_positions: set[int] | None = None,
    source: str = "lattice",
) -> pd.DataFrame:
    gap_positions = gap_positions or set()
    observed_positions = observed_positions or set(range(periods))
    times = pd.date_range(start, periods=periods, freq="5min")
    rows: list[dict[str, object]] = []
    for index, ts in enumerate(times):
        observed = index in observed_positions
        verified_empty = not observed and index not in gap_positions
        base: dict[str, object] = {
            "date": ts.isoformat().replace("+00:00", "Z"),
            "source_observed": "true" if observed else "false",
            "source_complete": "true",
            "source_gap_day": "true" if index in gap_positions else "false",
            "verified_zero_volume_empty": "true" if verified_empty else "false",
            "post_gap_quarantine": "false",
        }
        base_complete = (observed or verified_empty) and index not in gap_positions
        prior_invalid = any(
            position < index and position >= index - 24
            for position in gap_positions
        )
        base["post_gap_quarantine"] = "true" if prior_invalid else "false"
        base["source_complete"] = (
            "true" if base_complete and not prior_invalid else "false"
        )
        if source == "lattice":
            base.update(
                agg_trade_count="2" if observed else "0",
                total_quantity_mbtc="300" if observed else "0",
                coarse_quantity_mbtc="200" if observed else "0",
                coarse_signed_quantity_mbtc="100" if observed else "0",
                fine_quantity_mbtc="50" if observed else "0",
                fine_signed_quantity_mbtc="-25" if observed else "0",
            )
        else:
            start_ms = int(ts.timestamp() * 1000)
            base.update(
                first_transact_time_ms=str(start_ms + 1_000) if observed else "0",
                last_transact_time_ms=str(start_ms + 240_000) if observed else "0",
                agg_trade_count="2" if observed else "0",
                first_price="100.0" if observed else "0",
                last_price="100.1" if observed else "0",
                quote_notional="1000.0" if observed else "0",
                collision_quote_notional="100.0" if observed else "0",
                max_ms_quote_notional="200.0" if observed else "0",
                max_ms_signed_quote_notional="-50.0" if observed else "0",
            )
        rows.append(base)
    return pd.DataFrame(rows)


def _state_frame(n: int = 230) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index, boundary in enumerate(BOUNDARIES[:n]):
        records.append(
            {
                "boundary_time": boundary,
                "core_source_valid": index != 17,
                "_h41_source_position": index // 21,
                "_rrp_source_position": index // 3,
                "h41_delta": float((index % 3) - 1),
                "rrp_amount_delta": float((index % 5) - 2),
                "rrp_breadth_delta": float((index % 3) - 1),
                "h41_age_days": float(index % 6),
                "rrp_age_days": float(index % 4),
                "coarse_flow": float((index % 9) - 4),
                "fine_flow": float(((index * 2) % 11) - 5),
                "coarse_share": float(index % 19) / 20.0,
                "coarse_coherence": float((index * 2) % 23) / 24.0,
                "fine_conviction": float((index * 3) % 29) / 30.0,
                "collision_share": float((index * 5) % 31) / 32.0,
                "cascade_share": float((index * 7) % 37) / 38.0,
                "cascade_coherence": float((index * 11) % 41) / 42.0,
                "cascade_flow": float(((index * 3) % 13) - 6),
                "source_price_response": float(((index * 5) % 17) - 8) / 1000.0,
            }
        )
    return pd.DataFrame(records)


def _source_bundle(boundary_count: int = 260) -> Any:
    end = BOUNDARIES[boundary_count - 1] + pd.Timedelta(hours=8)
    releases = pd.date_range(
        "2019-11-07",
        end.tz_convert(None).normalize(),
        freq="7D",
    )
    h41_levels = [
        10_000.0 + ((index * 7) % 23) - ((index * 3) % 11)
        for index in range(len(releases))
    ]
    h41 = pd.DataFrame(
        {
            "release_date": releases.strftime("%Y-%m-%d"),
            "observation_date": (
                releases - pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d"),
            "available_at_utc": [
                pd.Timestamp(day, tz="America/New_York")
                .tz_convert("UTC")
                .isoformat()
                .replace("+00:00", "Z")
                for day in releases
            ],
            "net_liquidity_usd_millions": [
                str(value) for value in h41_levels
            ],
        }
    )
    operations = pd.date_range(
        "2019-12-01",
        end.tz_convert(None).normalize(),
        freq="D",
    )
    rrp = pd.DataFrame(
        {
            "operation_date": operations.strftime("%Y-%m-%d"),
            "result_available_at_utc": [
                (day + pd.Timedelta(hours=18))
                .tz_localize("UTC")
                .isoformat()
                .replace("+00:00", "Z")
                for day in operations
            ],
            "total_amount_accepted_usd": [
                str(1_000.0 + ((index * 5) % 17))
                for index in range(len(operations))
            ],
            "participating_counterparties": [
                "100" for _ in range(len(operations))
            ],
            "accepted_counterparties": [
                str(50 + ((index * 3) % 13))
                for index in range(len(operations))
            ],
            "source_complete": ["true" for _ in range(len(operations))],
            "quarantine_reason": ["" for _ in range(len(operations))],
        }
    )
    periods = boundary_count * 96
    lattice = _call(
        "validate_micro_source_frame",
        _micro_frame(periods, source="lattice"),
        source="lattice",
        require_full_grid=False,
    )
    cascade = _call(
        "validate_micro_source_frame",
        _micro_frame(periods, source="cascade"),
        source="cascade",
        require_full_grid=False,
    )
    return s.SourceBundle(
        h41=_call("validate_h41_frame", h41),
        rrp=_call("validate_rrp_frame", rrp),
        lattice=lattice,
        cascade=cascade,
    )


def test_contract_constants_are_frozen_and_outcome_blind() -> None:
    assert list(s.TOKEN_COLUMNS) == EXPECTED_TOKEN_COLUMNS
    assert list(s.CONTROL_IDS) == [
        "h41_stale_one_release",
        "rrp_stale_one_operation",
        "lattice_cohort_swap",
        "cascade_delay_37",
        "macro_relation_mask",
    ]
    assert list(s.APPEND_REPLAY_CUTOFFS) == [
        "2020-06-30T23:59:59Z",
        "2020-12-31T23:59:59Z",
        "2021-06-30T23:59:59Z",
        "2021-12-31T23:59:59Z",
        "2022-06-30T23:59:59Z",
        "2022-12-31T23:59:59Z",
        "2023-06-30T23:59:59Z",
        "2023-12-31T23:59:59Z",
    ]
    assert set(EXPECTED_FORBIDDEN_COUNTERS).issubset(s.FORBIDDEN_COUNTERS)
    assert not any("return" in column or "reward" in column for column in s.TOKEN_COLUMNS)


def test_h41_validator_requires_exact_asof_clock_and_prior_release_delta() -> None:
    validated = _call("validate_h41_frame", _h41_frame())

    assert validated["h41_delta"].isna().iloc[0]
    assert validated["h41_delta"].iloc[1] == pytest.approx(0.5)

    bad_observation = _h41_frame()
    bad_observation.loc[1, "observation_date"] = bad_observation.loc[1, "release_date"]
    with pytest.raises(ValueError, match="observation.*strictly prior"):
        _call("validate_h41_frame", bad_observation)

    stale_after_cut = _h41_frame()
    last = len(stale_after_cut) - 1
    stale_after_cut.loc[last, "release_date"] = "2024-01-04"
    stale_after_cut.loc[last, "observation_date"] = "2024-01-03"
    stale_after_cut.loc[last, "available_at_utc"] = "2024-01-04T05:00:00Z"
    with pytest.raises(ValueError, match="2024-or-later"):
        _call("validate_h41_frame", stale_after_cut)


def test_rrp_validator_quarantines_incomplete_rows_and_breaks_delta_segment() -> None:
    validated = _call("validate_rrp_frame", _rrp_frame())

    assert validated.loc[2, "source_complete"] is False or validated.loc[2, "source_complete"] == False
    assert pd.isna(validated.loc[2, "rrp_amount_delta"])
    assert pd.isna(validated.loc[3, "rrp_amount_delta"])
    assert validated.loc[4, "rrp_amount_delta"] == pytest.approx(1.0)

    bad = _rrp_frame()
    bad.loc[0, "source_complete"] = "True"
    with pytest.raises(ValueError, match="exact lowercase boolean"):
        _call("validate_rrp_frame", bad)

    bad_numeric = _rrp_frame()
    bad_numeric.loc[2, "total_amount_accepted_usd"] = "1.0"
    with pytest.raises(ValueError, match="incomplete.*numeric"):
        _call("validate_rrp_frame", bad_numeric)

    selected, reason = _call(
        "_macro_rrp_selection",
        validated,
        pd.Timestamp("2019-12-30T00:00:00Z"),
        stale_offset=0,
    )
    assert selected is None
    assert reason == "rrp_latest_operation_quarantined"


def test_micro_validity_uses_one_bar_shifted_24_bar_post_gap_quarantine() -> None:
    frame = _micro_frame(36, gap_positions={5}, observed_positions=set(range(36)) - {5})
    validated = _call("validate_micro_source_frame", frame, source="lattice", require_full_grid=False)

    assert validated.loc[4, "source_complete"] is True or validated.loc[4, "source_complete"] == True
    assert validated.loc[5, "base_complete"] is False or validated.loc[5, "base_complete"] == False
    assert validated.loc[6:29, "post_gap_quarantine"].all()
    assert not bool(validated.loc[30, "post_gap_quarantine"])


def test_micro_validators_reject_noncanonical_grid_and_lattice_identities() -> None:
    good_lattice = _micro_frame(12, source="lattice")
    assert _call("validate_micro_source_frame", good_lattice, source="lattice", require_full_grid=False)["source_complete"].all()

    skipped = good_lattice.drop(index=3).reset_index(drop=True)
    with pytest.raises(ValueError, match="5m grid"):
        _call("validate_micro_source_frame", skipped, source="lattice", require_full_grid=False)

    impossible = good_lattice.copy()
    impossible.loc[0, "coarse_quantity_mbtc"] = "400"
    with pytest.raises(ValueError, match="coarse.*fine.*total|quantity"):
        _call("validate_micro_source_frame", impossible, source="lattice", require_full_grid=False)


def test_cascade_validator_rejects_forward_ms_and_notional_overflow() -> None:
    good = _micro_frame(12, source="cascade")
    assert _call("validate_micro_source_frame", good, source="cascade", require_full_grid=False)["source_complete"].all()

    bad_clock = good.copy()
    start_ms = int(pd.Timestamp(bad_clock.loc[0, "date"]).timestamp() * 1000)
    bad_clock.loc[0, "last_transact_time_ms"] = str(start_ms + 300_000)
    with pytest.raises(ValueError, match="transaction clock"):
        _call("validate_micro_source_frame", bad_clock, source="cascade", require_full_grid=False)

    bad_notional = good.copy()
    bad_notional.loc[0, "max_ms_quote_notional"] = "1001.0"
    with pytest.raises(ValueError, match="notional"):
        _call("validate_micro_source_frame", bad_notional, source="cascade", require_full_grid=False)


def test_boundary_aggregation_requires_exact_96_prior_rows_and_causal_macro_asof() -> None:
    lattice = _call("validate_micro_source_frame", _micro_frame(192, source="lattice"), source="lattice", require_full_grid=False)
    cascade = _call("validate_micro_source_frame", _micro_frame(192, source="cascade"), source="cascade", require_full_grid=False)
    boundary = pd.Timestamp("2020-01-01T16:00:00Z")

    state = _call("build_boundary_state", boundary, _call("validate_h41_frame", _h41_frame()), _call("validate_rrp_frame", _rrp_frame()), lattice, cascade)

    assert state["boundary_time"] == boundary
    assert state["core_source_valid"] is True
    assert state["micro_rows_lattice"] == 96
    assert state["micro_rows_cascade"] == 96
    assert pd.Timestamp(state["h41_available_at_utc"]) <= boundary
    assert pd.Timestamp(state["rrp_available_at_utc"]) <= boundary

    quarantined = _call(
        "validate_rrp_frame",
        _rrp_frame([True, True, True, True, False]),
    )
    quarantined_state = _call(
        "build_boundary_state",
        boundary,
        _call("validate_h41_frame", _h41_frame()),
        quarantined,
        lattice,
        cascade,
    )
    assert quarantined_state["core_source_valid"] is False
    assert (
        "rrp_latest_operation_quarantined"
        in quarantined_state["invalid_reasons"]
    )

    first = _call(
        "build_boundary_state",
        pd.Timestamp("2020-01-01T00:00:00Z"),
        _call("validate_h41_frame", _h41_frame()),
        _call("validate_rrp_frame", _rrp_frame()),
        lattice,
        cascade,
    )
    assert first["core_source_valid"] is False
    assert "window_not_96" in first["invalid_reasons"]


def test_strict_prior_ranks_exclude_current_invalids_and_emit_safety_line() -> None:
    ranked = _call("attach_strict_prior_ranks", _state_frame(230), lookback=270, minimum=180)
    first_ready = int(ranked.index[ranked["rank_ready"]].min())

    assert first_ready > 180
    assert not ranked.loc[: first_ready - 1, "rank_ready"].any()
    assert 17 not in ranked.loc[: first_ready - 1].index[ranked.loc[: first_ready - 1, "core_source_valid"]]
    tokens = _call("tokenize_states", ranked)

    invalid = tokens.loc[17]
    assert invalid["core_source_valid"] is False or invalid["core_source_valid"] == False
    assert list(invalid[list(s.TOKEN_FIELDS)]) == list(s.SAFETY_TOKENS)
    assert bool(tokens.loc[first_ready, "sequence_ready"])
    assert first_ready >= 20


def test_transition_after_invalid_prior_preserves_current_tokens_but_mixes_transitions() -> None:
    states = _state_frame(225)
    states.loc[203, "core_source_valid"] = False
    ranked = _call("attach_strict_prior_ranks", states, lookback=270, minimum=180)
    tokens = _call("tokenize_states", ranked)

    current = tokens.loc[204]
    assert current["rank_ready"] is True or current["rank_ready"] == True
    assert current["h41_impulse"] not in s.SAFETY_TOKENS
    assert current["macro_transition"] == s.MIXED_MACRO_TRANSITION
    assert current["micro_transition"] == s.MIXED_MICRO_TRANSITION


def test_controls_are_independently_rebuilt_distinct_and_backward_only_for_cascade_delay() -> None:
    bundle = _source_bundle(260)
    primary = _call(
        "build_primary_token_frame",
        bundle,
        boundaries=BOUNDARIES[:260],
    )
    controls = _call(
        "build_control_token_frames",
        bundle,
        primary,
        boundaries=BOUNDARIES[:260],
    )

    assert list(controls) == list(s.CONTROL_IDS)
    delayed = controls["cascade_delay_37"]
    valid_delayed = delayed.loc[delayed["cascade_source_time"].notna()]
    assert (
        pd.to_datetime(valid_delayed["cascade_source_time"], utc=True)
        < pd.to_datetime(valid_delayed["boundary_time"], utc=True)
    ).all()
    streams = {
        control_id: b"\n".join(
            _call("serialize_token_line", row).encode()
            for _, row in frame.iterrows()
        )
        for control_id, frame in controls.items()
    }
    primary_stream = b"\n".join(
        _call("serialize_token_line", row).encode()
        for _, row in primary.iterrows()
    )
    assert all(stream != primary_stream for stream in streams.values())
    assert len(set(streams.values())) == len(streams)
    assert controls["macro_relation_mask"].loc[220, "macro_transition"] == s.MIXED_MACRO_TRANSITION


def test_cascade_delay_is_exactly_37_prior_rows_and_never_wraps() -> None:
    cascade = _call(
        "validate_micro_source_frame",
        _micro_frame(100, source="cascade"),
        source="cascade",
        require_full_grid=False,
    )
    delayed = _call("_delay_cascade_inside_month", cascade)

    assert not delayed.loc[:36, "source_complete"].any()
    assert delayed.loc[:36, "_source_time"].isna().all()
    assert delayed.loc[37, "_source_time"] == cascade.loc[0, "date"]
    assert (
        delayed.loc[37, "max_ms_signed_quote_notional"]
        == cascade.loc[0, "max_ms_signed_quote_notional"]
    )


def test_true_append_replay_physically_hides_future_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _source_bundle(230)
    full = _call(
        "build_primary_token_frame",
        bundle,
        boundaries=BOUNDARIES[:230],
    )
    cutoff = BOUNDARIES[220] + pd.Timedelta(hours=7, minutes=59)
    original = s.build_primary_token_frame
    observed_prefixes: list[Any] = []

    def spy(prefix: Any, **kwargs: Any) -> pd.DataFrame:
        observed_prefixes.append(prefix)
        assert prefix.h41["available_at_utc"].max() <= cutoff
        assert prefix.rrp["result_available_at_utc"].max() <= cutoff
        assert (prefix.lattice["date"] + pd.Timedelta(minutes=5)).max() <= cutoff
        assert (prefix.cascade["date"] + pd.Timedelta(minutes=5)).max() <= cutoff
        assert len(prefix.lattice) < len(bundle.lattice)
        return original(prefix, **kwargs)

    monkeypatch.setattr(s, "build_primary_token_frame", spy)
    audit = _call(
        "true_append_replay_audit",
        full,
        bundle,
        cutoffs=(cutoff,),
    )

    assert len(observed_prefixes) == 1
    assert audit["byte_identical"] is True
    assert audit["compared_full_build_to_itself"] is False
    assert audit["filtered_full_result_only"] is False

    corrupted = full.copy()
    corrupted.loc[220, "micro_braid"] = "MICRO_NEUTRAL"
    mismatch = _call(
        "true_append_replay_audit",
        corrupted,
        bundle,
        cutoffs=(cutoff,),
    )
    assert mismatch["byte_identical"] is False


def test_empty_eligible_years_fail_closed_without_nonfinite_json() -> None:
    tokens = _call(
        "tokenize_states",
        _call(
            "attach_strict_prior_ranks",
            _state_frame(30),
            lookback=270,
            minimum=180,
        ),
    )
    annual = _call("_yearly_token_support", tokens)

    assert all(
        value is None
        for fields in annual["adjacent_year_jsd"].values()
        for value in fields.values()
    )
    encoded = _call("deterministic_report_bytes", {"annual": annual})
    assert json.loads(encoded)["annual"]["2020"][
        "sequence_ready_current_core_valid"
    ] == 0


def test_gate_order_short_circuits_decision_and_uses_sequence_ready_core_denominator() -> None:
    diagnostics = {
        "gate_01": {"name": "protocol_source_validation", "passed": True},
        "gate_02": {"name": "annual_micro_grid_join", "passed": True},
        "gate_03": {"name": "annual_core_valid_share", "passed": True},
        "gate_04": {"name": "sequence_ready_counts", "passed": False},
        "gate_05": RuntimeError("must not evaluate after first failure"),
    }
    decision = _call("evaluate_gates", diagnostics)

    assert decision["first_failed_gate"] == "gate_04"
    assert decision["first_failed_name"] == "sequence_ready_counts"
    assert decision["decision"] == "fail"
    assert decision["authorized_next_stage"] is None
    assert decision["failure_action"] == "retire_lamb21_unchanged_before_rewards"
    assert decision["gate_denominator"] == "sequence_ready_current_core_valid_by_utc_year"


def test_forbidden_counter_guard_aborts_before_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wrote: list[Path] = []
    monkeypatch.setattr(s, "write_outputs", lambda *args, **kwargs: wrote.append(tmp_path / "unexpected"))
    counters = {name: 0 for name in s.COUNTERS}
    counters["source_value_rows_decoded"] = 10
    counters["joint_state_rows_built"] = 1
    counters["future_return_rows_opened"] = 1

    with pytest.raises(RuntimeError, match="forbidden counter"):
        _call("finalize_outputs", pd.DataFrame(), {}, counters)
    assert wrote == []


def test_deterministic_write_once_gzip_and_report_reject_drift(tmp_path: Path) -> None:
    rows = pd.DataFrame([{column: "SAFE" for column in s.TOKEN_COLUMNS}])
    rows.loc[0, "boundary_time"] = "2020-01-01T00:00:00Z"
    payload = {"decision": "pass", "authorized_next_stage": "authorize_stage_0_5_reward_evaluator_freeze"}

    token_bytes = _call("deterministic_token_gzip_bytes", rows)
    assert token_bytes == _call("deterministic_token_gzip_bytes", rows.sample(frac=1, random_state=1))
    with gzip.GzipFile(fileobj=io.BytesIO(token_bytes), mode="rb") as handle:
        decoded = handle.read().decode("utf-8")
        assert handle.mtime == 0
        assert decoded.splitlines()[0].split(",") == list(s.TOKEN_COLUMNS)

    report_bytes = _call("deterministic_report_bytes", dict(reversed(tuple(payload.items()))))
    assert report_bytes == _call("deterministic_report_bytes", payload)
    assert report_bytes.endswith(b"\n")
    assert json.loads(report_bytes) == payload

    out = tmp_path / "token_support.csv.gz"
    assert _call("write_once_bytes", out, token_bytes) == "created"
    assert _call("write_once_bytes", out, token_bytes) == "verified_existing"
    with pytest.raises(RuntimeError, match="drift|write-once"):
        _call("write_once_bytes", out, token_bytes + b"x")


def test_physical_loader_uses_exact_usecols_and_rejects_load_all_then_drop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append({"args": args, "kwargs": kwargs})
        return pd.DataFrame({column: ["1"] for column in kwargs["usecols"]})

    monkeypatch.setattr(s.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(s, "verify_physical_source", lambda *args, **kwargs: None)
    frame = _call("load_exact_projection", Path("synthetic.csv.gz"), ("a", "b"), expected_header=("a", "b", "forbidden"))

    assert list(frame.columns) == ["a", "b"]
    assert calls[0]["kwargs"] == {
        "usecols": ("a", "b"),
        "dtype": "string",
        "keep_default_na": False,
        "na_filter": False,
    }


def test_protocol_guard_runs_before_any_real_source_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    def fake_git(*_args: str) -> subprocess.CompletedProcess[str]:
        events.append("git_guard")
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")

    def forbidden_loader(*_args: Any, **_kwargs: Any) -> None:
        events.append("loader")
        raise AssertionError("source loader must not run after guard failure")

    monkeypatch.setattr(s, "_git_check", fake_git)
    monkeypatch.setattr(s, "load_source_frames", forbidden_loader)

    with pytest.raises(RuntimeError, match="committed|HEAD|protocol"):
        _call("build_real_support_payload")
    assert events == ["git_guard"]


def test_output_paths_and_import_boundary_are_frozen() -> None:
    source = Path(s.__file__).read_text(encoding="utf-8")
    forbidden_import_fragments = [
        "execution",
        "funding",
        "future_return",
        "reward",
        "trainer",
        "checkpoint",
        "simulator",
        "portfolio",
    ]

    assert s.TOKEN_OUTPUT == Path("data/lamb21_source_support/token_support.csv.gz")
    assert s.REPORT_OUTPUT == Path("results/lamb21_source_support_2026-07-25.json")
    assert not any(f"import {fragment}" in source or f"from {fragment}" in source for fragment in forbidden_import_fragments)
