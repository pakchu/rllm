from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_cboe_cross_surface_pressure_grammar_support as s
from training import preregister_cboe_cross_surface_pressure_grammar as p


def _source_panels(
    rows: int = 180,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2018-01-01", periods=rows, freq="D", tz="UTC")
    phase = np.arange(rows, dtype=np.float64)
    vix = 20.0 + 0.01 * phase
    term = pd.DataFrame(
        {
            "observation_date": dates,
            "VIX9D_close": vix * (0.95 + 0.03 * np.sin(phase / 11.0)),
            "VIX_close": vix,
            "VIX3M_close": vix * (1.05 + 0.03 * np.cos(phase / 13.0)),
        }
    )
    tail = pd.DataFrame(
        {
            "observation_date": dates,
            "SKEW_close": 120.0 + 3.0 * np.sin(phase / 7.0),
            "VVIX_close": 90.0 + 4.0 * np.cos(phase / 9.0),
            "VIX_close": vix,
        }
    )
    option = pd.DataFrame(
        {
            "observation_date": dates,
            "total_volume": 100_000.0 + 100.0 * phase,
            "index_call_volume": 20_000.0 + 50.0 * np.sin(phase / 5.0),
            "index_put_volume": 21_000.0 + 55.0 * np.cos(phase / 6.0),
            "index_volume": 41_000.0 + 70.0 * np.sin(phase / 8.0),
            "equity_call_volume": 30_000.0 + 60.0 * np.cos(phase / 7.0),
            "equity_put_volume": 31_000.0 + 65.0 * np.sin(phase / 9.0),
            "vix_call_volume": 8_000.0 + 40.0 * np.sin(phase / 4.0),
            "vix_put_volume": 7_500.0 + 35.0 * np.cos(phase / 3.0),
        }
    )
    return term, tail, option


def _dense_states() -> pd.DataFrame:
    dates = pd.date_range(
        "2020-01-02",
        "2023-12-28",
        freq="B",
        tz="UTC",
    )
    rng = np.random.default_rng(20_260_724)
    payload: dict[str, object] = {"observation_date": dates}
    for column, vocabulary in p.TOKEN_SCHEMA:
        payload[column] = rng.choice(vocabulary, size=len(dates), replace=True)
    return pd.DataFrame(payload)


def test_loader_passes_exact_allowlist_to_read_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
        calls.append({"path": path, **kwargs})
        columns = list(kwargs["usecols"])
        row = {
            column: (
                "2023-01-01"
                if column == "observation_date"
                else "1.0"
            )
            for column in columns
        }
        return pd.DataFrame([row], columns=columns)

    monkeypatch.setattr(s.pd, "read_csv", fake_read_csv)
    s.load_term_source()
    s.load_tail_source()
    s.load_option_source()
    assert [call["usecols"] for call in calls] == [
        list(p.TERM_ALLOWLIST),
        list(p.TAIL_ALLOWLIST),
        list(p.OPTION_ALLOWLIST),
    ]
    assert all(call["dtype"] for call in calls)


def test_source_validation_rejects_schema_date_value_and_future_drift() -> None:
    valid = pd.DataFrame(
        {
            "observation_date": ["2023-01-01", "2023-01-02"],
            "VIX9D_close": [19.0, 20.0],
            "VIX_close": [20.0, 21.0],
            "VIX3M_close": [21.0, 22.0],
        }
    )
    result = s.validate_source_frame(
        valid,
        allowlist=p.TERM_ALLOWLIST,
        source_name="term",
    )
    assert list(result.columns) == list(p.TERM_ALLOWLIST)
    assert str(result["observation_date"].dtype) == "datetime64[ns, UTC]"

    with pytest.raises(RuntimeError, match="loader did not preserve"):
        s.validate_source_frame(
            valid.loc[:, list(reversed(valid.columns))],
            allowlist=p.TERM_ALLOWLIST,
            source_name="term",
        )
    duplicate = pd.concat([valid.iloc[[0]], valid.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicated"):
        s.validate_source_frame(
            duplicate,
            allowlist=p.TERM_ALLOWLIST,
            source_name="term",
        )
    descending = valid.iloc[::-1].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="not increasing"):
        s.validate_source_frame(
            descending,
            allowlist=p.TERM_ALLOWLIST,
            source_name="term",
        )
    nonpositive = valid.copy()
    nonpositive.loc[0, "VIX_close"] = 0.0
    with pytest.raises(RuntimeError, match="primitive invalid"):
        s.validate_source_frame(
            nonpositive,
            allowlist=p.TERM_ALLOWLIST,
            source_name="term",
        )
    future = valid.copy()
    future.loc[1, "observation_date"] = "2024-01-01"
    with pytest.raises(RuntimeError, match="2024-or-later"):
        s.validate_source_frame(
            future,
            allowlist=p.TERM_ALLOWLIST,
            source_name="term",
        )


def test_strict_prior_rank_excludes_current_midranks_and_truncates() -> None:
    values = [1.0, 2.0, 2.0, 3.0]
    ranks = s.strict_prior_midranks(values, lookback=3, minimum=2)
    assert np.isnan(ranks[0])
    assert np.isnan(ranks[1])
    assert ranks[2] == 0.75
    assert ranks[3] == 1.0
    truncated = s.strict_prior_midranks(
        [100.0, 0.0, 0.0, 5.0],
        lookback=2,
        minimum=2,
    )
    assert truncated[-1] == 1.0


def test_option_pressure_uses_immediately_prior_source_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, option = _source_panels(5)
    captured: list[np.ndarray] = []

    def fake_rank(values: object, **_: object) -> np.ndarray:
        array = np.asarray(list(values), dtype=np.float64)
        captured.append(array)
        return np.zeros(len(array), dtype=np.float64)

    monkeypatch.setattr(s, "strict_prior_midranks", fake_rank)
    features = s.build_option_features(option)
    assert len(captured) == 3
    assert all(np.isnan(values[0]) for values in captured)
    assert all(np.isfinite(values[1:]).all() for values in captured)
    assert len(features) == len(option)
    assert features["option_pressure"].eq(0.0).all()


def test_common_states_exact_join_vix_equality_and_predecessor_tokens() -> None:
    term, tail, option = _source_panels()
    missing_date = option.loc[150, "observation_date"]
    option = option.drop(index=150).reset_index(drop=True)
    states, funnel = s.build_common_states(term, tail, option)
    assert missing_date not in set(states["observation_date"])
    assert funnel["exact_common_dates"] == len(option)
    assert funnel["token_ready_common_states"] == (
        funnel["rank_complete_common_states"] - 1
    )
    assert list(states.columns) == [
        "observation_date",
        "term_pressure",
        "tail_pressure",
        "option_pressure",
        *p.TOKEN_COLUMNS,
    ]
    for row in states.loc[:, list(p.TOKEN_COLUMNS)].to_dict("records"):
        p.validate_tokens(row)

    mismatched_tail = tail.copy()
    mismatched_tail.loc[160, "VIX_close"] += 0.01
    with pytest.raises(RuntimeError, match="cross-panel mismatch"):
        s.build_common_states(term, mismatched_tail, option)


def test_future_append_invariance_holds_on_synthetic_prefix() -> None:
    term, tail, option = _source_panels(220)
    assert s.prefix_invariance(term, tail, option, trim_rows=20) is True


def test_calendar_clock_and_global_reservation_are_action_independent() -> None:
    states = _dense_states().iloc[:3].copy()
    states.loc[:, "observation_date"] = pd.to_datetime(
        ["2023-03-10", "2023-03-11", "2023-03-12"],
        utc=True,
    )
    raw = s.raw_candidates(states)
    assert raw.loc[0, "signal_available_time"] == pd.Timestamp(
        "2023-03-11T14:30:00Z"
    )
    assert raw.loc[1, "signal_available_time"] == pd.Timestamp(
        "2023-03-12T13:30:00Z"
    )
    assert raw.loc[1, "entry_time"] == pd.Timestamp("2023-03-12T13:35:00Z")
    assert raw.loc[1, "exit_time"] == pd.Timestamp("2023-03-13T13:35:00Z")
    assert "action" not in raw.columns
    assert "side" not in raw.columns

    reserved = s.reserve_nonoverlap(raw)
    assert reserved["source_date"].tolist() == [
        pd.Timestamp("2023-03-10T00:00:00Z"),
        pd.Timestamp("2023-03-12T00:00:00Z"),
    ]
    # The spring-forward local clock advances by 23 UTC hours, so the middle
    # 24-hour interval overlaps and is suppressed before any policy action.
    assert s._reservation_integrity(reserved) is True


def test_dense_balanced_synthetic_states_pass_frozen_support_gates() -> None:
    raw = s.raw_candidates(_dense_states())
    reserved = s.reserve_nonoverlap(raw)
    (
        statistics,
        partitions,
        token_report,
        source_checks,
        token_checks,
    ) = s.support_checks(reserved, prefix_invariant=True)
    assert statistics["global"]["events"] >= 820
    assert partitions["2023"]["2023_q4"] >= 48
    assert token_report["2023"]["events"] >= 230
    assert all(source_checks.values()), {
        key: value for key, value in source_checks.items() if not value
    }
    assert all(token_checks.values()), {
        key: value for key, value in token_checks.items() if not value
    }
    assert s.first_failure(
        source_checks,
        token_checks,
        artifact_eligible=False,
    ) == ("artifact_eligibility", "synthetic_or_injected_build")


def test_downstream_unseen_token_fails_closed() -> None:
    raw = s.raw_candidates(_dense_states())
    reserved = s.reserve_nonoverlap(raw)
    reserved.loc[
        reserved["source_date"].dt.year.eq(2023),
        "stress_leader",
    ] = "TIE"
    reserved.loc[
        reserved["source_date"].dt.year.lt(2023),
        "stress_leader",
    ] = "TERM"
    _, _, _, _, checks = s.support_checks(
        reserved,
        prefix_invariant=True,
    )
    assert checks["2023:stress_leader:seen_in_train"] is False


def test_clock_bytes_are_canonical_and_outcome_blind() -> None:
    rows = s.reserve_nonoverlap(s.raw_candidates(_dense_states().iloc[:5]))
    first = s.deterministic_clock_bytes(rows)
    second = s.deterministic_clock_bytes(rows)
    assert first == second
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as handle:
        header = handle.readline().decode("utf-8").strip().split(",")
    assert header == list(s.CLOCK_COLUMNS)
    assert not any(
        token in column.lower()
        for column in header
        for token in s.FORBIDDEN_CLOCK_TOKENS
    )


def test_synthetic_report_never_opens_outcomes_or_authorizes_next_stage() -> None:
    report, clock = s.build_support_from_states(_dense_states())
    assert clock
    assert report["artifact_eligible"] is False
    assert report["source_support_passed"] is True
    assert report["token_support_passed"] is True
    assert report["outcomes_opened"] is False
    assert report["market_loaded"] is False
    assert report["funding_loaded"] is False
    assert report["comparators_opened"] is False
    assert report["authorized_next_stage"] is None
    boundary = report["outcome_boundary"]
    assert boundary["BTC_market_rows_decoded"] == 0
    assert boundary["funding_rows_decoded"] == 0
    assert boundary["comparator_rows_decoded"] == 0
    assert boundary["future_return_rows_decoded"] == 0


def test_write_once_accepts_identity_and_rejects_drift(tmp_path: Path) -> None:
    output = tmp_path / "artifact.bin"
    assert s._write_once(output, b"one") == "created"
    assert s._write_once(output, b"one") == "verified_existing"
    with pytest.raises(RuntimeError, match="noncanonical"):
        s._write_once(output, b"two")


def test_contract_preregistration_and_source_bindings_are_hash_bound() -> None:
    payload = s.validate_preregistration()
    audit = s.verify_pre_source_bindings(payload)
    assert s.sha256_file(s.IMPLEMENTATION_CONTRACT) == (
        s.IMPLEMENTATION_CONTRACT_SHA256
    )
    assert s.sha256_file(s.PREREGISTRATION) == s.PREREGISTRATION_SHA256
    assert payload["manifest_hash"] == s.PREREGISTRATION_MANIFEST_HASH
    assert set(audit) == set(s._source_dependencies())
    assert s.CLOCK_COLUMNS[5:] == p.TOKEN_COLUMNS
    assert json.loads(
        s._path(s.PREREGISTRATION).read_text(encoding="utf-8")
    ) == p.build_manifest()
