import hashlib
import json
import math
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_mstr_relative_short_volume_pressure_relay_support as support
from training import preregister_high_volatility_mstr_relative_short_volume_pressure_relay as prereg


def _finra_file(date: str = "20230103") -> bytes:
    return (
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        f"{date}|AAPL|20|0|50|Q,N\n"
        f"{date}|QQQ|50|2|100|B,Q,N\n"
        f"{date}|MSTR|40|1|100|Q,N\n"
    ).encode()


def _pair(dates: pd.DatetimeIndex) -> pd.DataFrame:
    count = len(dates)
    return pd.DataFrame(
        {
            "source_date": dates,
            "feature_available_time": dates + pd.Timedelta(days=1),
            "mstr_short_volume": np.arange(count) + 40,
            "mstr_total_volume": 100,
            "qqq_short_volume": np.arange(count) + 30,
            "qqq_total_volume": 100,
        },
        columns=support.PAIR_COLUMNS,
    )


def _bars(start: pd.Timestamp, periods: int = 1440) -> pd.DataFrame:
    closes = 100 * np.exp(np.linspace(0, 0.02, periods))
    return pd.DataFrame(
        {
            "ts": pd.date_range(start, periods=periods, freq="1min"),
            "close": closes,
        }
    )


def _features(dates: list[str] | None = None) -> pd.DataFrame:
    source_dates = pd.to_datetime(
        dates
        or [
            "2024-06-28T00:00:00Z",
            "2024-06-29T00:00:00Z",
            "2024-06-30T00:00:00Z",
            "2024-07-01T00:00:00Z",
            "2024-07-02T00:00:00Z",
        ]
    )
    count = len(source_dates)
    return pd.DataFrame(
        {
            "source_date": source_dates,
            "feature_available_time": source_dates + pd.Timedelta(days=1),
            "mstr_short_volume": [40.0] * count,
            "mstr_total_volume": [100.0] * count,
            "qqq_short_volume": [30.0] * count,
            "qqq_total_volume": [100.0] * count,
            "btc_source_valid": [True] * count,
            "source_valid": [True] * count,
            "mstr_short_share": [0.4] * count,
            "qqq_short_share": [0.3] * count,
            "relative_pressure": [0.1] * count,
            "pressure_change": [1.0, 0.0, -1.0, 1.0, -1.0][:count],
            "mstr_share_change": [1.0, 0.0, -1.0, 1.0, -1.0][:count],
            "realized_variation": [0.1] * count,
            "absolute_pressure_change_rank": [0.9, 0.0, 0.9, 0.9, 0.9][:count],
            "realized_variation_rank": [0.9] * count,
        },
        columns=support.FEATURE_COLUMNS,
    )


class _Response:
    def __init__(
        self,
        status: int,
        raw: bytes,
        headers: dict[str, str] | None = None,
        url: str = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol20230103.txt",
    ):
        self.status = status
        self.raw = raw
        self.headers = headers or {}
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.raw

    def geturl(self) -> str:
        return self.url


def test_preregistration_controls_policy_and_exact_artifact_are_bound() -> None:
    assert support.PREREG_SHA256 == (
        "4e51be73a940bffa9a4cf9534c2802490322d12e85c8f1feb3a833434db877eb"
    )
    assert support.sha(prereg.DEFAULT_OUTPUT) == support.PREREG_SHA256
    assert json.loads(prereg.DEFAULT_OUTPUT.read_text()) == prereg.build()
    assert support.CONTROLS == tuple(prereg.build()["diagnostic_controls"]["names"])
    assert support.POLICY["history_source_days"] == 252
    assert support.POLICY["minimum_history_source_days"] == 126


def test_query_is_causal_btcusdt_one_minute_ohlc_only() -> None:
    assert support.START == pd.Timestamp("2023-01-01T00:00:00Z")
    assert support.END == pd.Timestamp("2026-08-01T00:00:00Z")
    assert "SELECT ts,close\n" in support.QUERY
    assert "FROM bars_binance\n" in support.QUERY
    assert "symbol='BTCUSDT'" in support.QUERY and "interval='1m'" in support.QUERY
    for forbidden in ("funding", "outcome", "pnl", "gross9", "execution"):
        assert forbidden not in support.QUERY.lower()


def test_parse_target_rows_date_first_strict_schema_and_exact_pair() -> None:
    date = pd.Timestamp("2023-01-03T00:00:00Z")
    rows = support.parse_target_rows(_finra_file(), date)
    assert [row["symbol"] for row in rows] == ["MSTR", "QQQ"]
    assert rows[0]["short_volume"] == 40 and rows[1]["total_volume"] == 100

    symbol_first = _finra_file().replace(
        b"Date|Symbol|ShortVolume", b"Symbol|Date|ShortVolume", 1
    )
    with pytest.raises(ValueError, match="schema drift"):
        support.parse_target_rows(symbol_first, date)
    with pytest.raises(ValueError, match="identity invalid"):
        support.parse_target_rows(_finra_file("20230104"), date)
    malformed_non_target = _finra_file().replace(b"20230103|AAPL|20|0|50|Q,N", b"20230103|AAPL|20|0|50")
    with pytest.raises(ValueError, match="row schema drift"):
        support.parse_target_rows(malformed_non_target, date)
    duplicate_non_target = _finra_file() + b"20230103|AAPL|1|0|2|Q,N\n"
    with pytest.raises(ValueError, match="duplicate symbol/date"):
        support.parse_target_rows(duplicate_non_target, date)


@pytest.mark.parametrize(
    "old, new, message",
    [
        (b"40|1|100", b"40.0|1|100", "unsigned integer"),
        (b"40|1|100", b"101|1|100", "volume invalid"),
        (b"40|1|100", b"40|1|0", "volume invalid"),
    ],
)
def test_parse_target_rows_rejects_invalid_target_volumes(
    old: bytes, new: bytes, message: str
) -> None:
    raw = _finra_file().replace(old, new, 1)
    with pytest.raises(ValueError, match=message):
        support.parse_target_rows(raw, pd.Timestamp("2023-01-03T00:00:00Z"))


def test_parse_target_rows_rejects_missing_or_duplicate_target() -> None:
    date = pd.Timestamp("2023-01-03T00:00:00Z")
    missing = _finra_file().replace(b"20230103|QQQ|50|2|100|B,Q,N\n", b"")
    with pytest.raises(RuntimeError, match="exactly one MSTR and one QQQ"):
        support.parse_target_rows(missing, date)
    duplicate = _finra_file() + b"20230103|MSTR|41|0|100|Q,N\n"
    with pytest.raises(ValueError, match="duplicate symbol/date"):
        support.parse_target_rows(duplicate, date)


def test_download_date_accepts_200_and_hashes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _finra_file()
    monkeypatch.setattr(
        support.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(200, raw, {"ETag": "abc", "Last-Modified": "then"}),
    )
    rows, binding = support.download_date(pd.Timestamp("2023-01-03T00:00:00Z"))
    assert rows is not None and len(rows) == 2
    assert binding == {
        "url": "https://cdn.finra.org/equity/regsho/daily/CNMSshvol20230103.txt",
        "status": 200,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "etag": "abc",
        "last_modified": "then",
    }


def test_download_date_rejects_redirected_final_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        support.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(200, _finra_file(), url="https://example.invalid/file"),
    )
    with pytest.raises(RuntimeError, match="redirect forbidden"):
        support.download_date(pd.Timestamp("2023-01-03T00:00:00Z"))


def test_download_date_accepts_and_hashes_404_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"not found"
    error = urllib.error.HTTPError("url", 404, "missing", {"ETag": "none"}, None)
    error.read = lambda: raw
    monkeypatch.setattr(
        support.urllib.request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(error)
    )
    rows, binding = support.download_date(pd.Timestamp("2023-01-01T00:00:00Z"))
    assert rows is None
    assert binding["status"] == 404
    assert binding["response_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("status", [201, 403, 429, 500])
def test_download_date_fails_closed_for_every_other_status(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    monkeypatch.setattr(
        support.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response(status, b"failure")
    )
    with pytest.raises(RuntimeError, match=f"fail-closed HTTP status {status}"):
        support.download_date(pd.Timestamp("2023-01-03T00:00:00Z"))


def test_download_pair_panel_probes_every_calendar_day_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = pd.Timestamp("2023-01-01T00:00:00Z")
    end = pd.Timestamp("2023-01-05T00:00:00Z")
    monkeypatch.setattr(support, "START", start)
    monkeypatch.setattr(support, "END", end)
    calls: list[pd.Timestamp] = []

    def fake_download(date: pd.Timestamp):
        calls.append(date)
        raw = f"response-{date.day}".encode()
        binding = support._response_binding("official", 404 if date.day in {1, 2} else 200, raw, {})
        if date.day in {1, 2}:
            return None, binding
        rows = support.parse_target_rows(_finra_file(date.strftime("%Y%m%d")), date)
        return rows, binding

    monkeypatch.setattr(support, "download_date", fake_download)
    pair, transport = support.download_pair_panel(workers=3)
    requested = list(pd.date_range(start, end, freq="1d", inclusive="left"))
    assert sorted(calls) == requested  # includes Sunday January 1 and Monday holiday January 2
    assert pair.source_date.tolist() == requested[2:]
    assert [item["date"] for item in transport["responses"]] == [
        "2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"
    ]
    assert transport["http_200_days"] == 2 and transport["http_404_days"] == 2
    assert transport["all_responses_sha256_bound"] is True
    assert transport["normalized_panel_sha256"] == support._normalized_panel_hash(pair)


def test_threaded_download_raises_first_failure_by_date_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(support, "START", pd.Timestamp("2023-01-01T00:00:00Z"))
    monkeypatch.setattr(support, "END", pd.Timestamp("2023-01-05T00:00:00Z"))

    def fake_download(date: pd.Timestamp):
        if date.day in {2, 4}:
            raise RuntimeError(f"failure-{date.day}")
        return None, support._response_binding("official", 404, b"", {})

    monkeypatch.setattr(support, "download_date", fake_download)
    with pytest.raises(RuntimeError, match="failure-2"):
        support.download_pair_panel(workers=4)


def test_load_bars_uses_env_engine_connection_query_and_disposes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _bars(pd.Timestamp("2023-01-01T00:00:00Z"), periods=2)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Engine:
        disposed = False

        def connect(self):
            return Connection()

        def dispose(self):
            self.disposed = True

    engine = Engine()
    observed: dict[str, object] = {}

    def fake_read(query, connection, params):
        observed.update(query=str(query), connection=connection, params=params)
        return expected.copy()

    monkeypatch.setattr(support, "postgres_engine", lambda: engine)
    monkeypatch.setattr(pd, "read_sql_query", fake_read)
    actual = support.load_bars()
    pd.testing.assert_frame_equal(actual, expected)
    assert observed["query"] == support.QUERY
    assert observed["params"] == {
        "start": support.START.to_pydatetime(), "end": support.END.to_pydatetime()
    }
    assert engine.disposed is True


def test_feature_panel_uses_exact_prior_1440_close_to_close_returns() -> None:
    date = pd.DatetimeIndex([pd.Timestamp("2023-01-01T00:00:00Z")])
    pair = _pair(date)
    bars = _bars(date[0])
    panel = support.feature_panel(pair, bars)
    expected = float(np.square(np.diff(np.log(bars.close.to_numpy(float)))).sum())
    assert panel.btc_source_valid.tolist() == [True]
    assert panel.realized_variation.iloc[0] == pytest.approx(expected)
    assert panel.feature_available_time.iloc[0] == pd.Timestamp("2023-01-02T00:00:00Z")

    missing = bars.drop(index=100).reset_index(drop=True)
    invalid = support.feature_panel(pair, missing)
    assert invalid.btc_source_valid.tolist() == [False]
    assert math.isnan(invalid.realized_variation.iloc[0])


def test_pressure_rank_history_is_source_day_based_not_btc_bar_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2023-01-01T00:00:00Z", periods=5, freq="1d")
    pair = _pair(dates)
    bars = pd.concat([_bars(date) for date in dates], ignore_index=True).drop_duplicates("ts")
    bars = bars[~bars.ts.eq(dates[2] + pd.Timedelta(minutes=10))].reset_index(drop=True)
    monkeypatch.setitem(support.POLICY, "history_source_days", 3)
    monkeypatch.setitem(support.POLICY, "minimum_history_source_days", 2)
    panel = support.feature_panel(pair, bars)
    expected = support.strict_prior_midrank(panel.pressure_change.abs(), 3, 2)
    pd.testing.assert_series_equal(panel.absolute_pressure_change_rank, expected, check_names=False)
    assert panel.btc_source_valid.tolist()[2] is False


def test_strict_prior_midrank_excludes_current_skips_nonfinite_and_caps_252() -> None:
    values = pd.Series([*map(float, range(253)), np.nan, 252.0])
    ranked = support.strict_prior_midrank(values, lookback=252, minimum=126)
    assert ranked.iloc[:126].isna().all()
    assert ranked.iloc[126] == 1.0
    assert ranked.iloc[252] == 1.0
    assert np.isnan(ranked.iloc[253])
    assert ranked.iloc[254] == pytest.approx((251 + 0.5) / 252)


def test_source_day_midrank_never_reaches_beyond_fixed_prior_positions() -> None:
    values = pd.Series([1.0] * 126 + [np.nan] * 252 + [2.0])
    ranked = support.strict_prior_source_day_midrank(values, lookback=252, minimum=126)
    assert np.isnan(ranked.iloc[-1])


def test_gates_are_inclusive_side_is_negative_change_and_controls_are_isolated() -> None:
    features = _features()
    features.loc[1, ["pressure_change", "absolute_pressure_change_rank"]] = [1.0, 0.799]
    features.loc[2, "realized_variation_rank"] = 0.649
    features.loc[3, ["absolute_pressure_change_rank", "realized_variation_rank"]] = [0.80, 0.65]
    eligible, side, _ = support.eligible_and_side(features)
    assert eligible.tolist() == [True, False, False, True, True]
    assert side.tolist() == [-1, -1, 1, -1, 1]
    assert support.eligible_and_side(features, "no_pressure_magnitude_gate")[0].tolist() == [True, True, False, True, True]
    assert support.eligible_and_side(features, "no_volatility_gate")[0].tolist() == [True, False, True, True, True]
    assert support.eligible_and_side(features, "direction_flip")[1].tolist() == [1, 1, -1, 1, -1]
    assert support.eligible_and_side(features, "forced_long")[1].tolist() == [1] * 5


def test_onset_reservation_split_and_stale_control_use_frozen_clocks() -> None:
    features = _features()
    primary = support.candidate_clock(features)
    # Row 1 is ineligible, so rows 0 and 2 are onsets. Later continuously eligible rows are not.
    assert primary.source_date.tolist() == [
        pd.Timestamp("2024-06-28T00:00:00Z"), pd.Timestamp("2024-06-30T00:00:00Z")
    ]
    assert (primary.entry_time == primary.decision_time + pd.Timedelta(minutes=5)).all()
    assert (primary.exit_time == primary.entry_time + pd.Timedelta(hours=24)).all()
    assert primary.entry_time.iloc[1] >= primary.exit_time.iloc[0]
    assert primary.side.tolist() == [-1, 1]
    assert set(primary.split) == {"test"}

    stale = support.candidate_clock(features, "one_source_day_stale_features")
    assert (stale.feature_available_time == stale.source_date + pd.Timedelta(days=1)).all()
    assert (stale.decision_time == stale.feature_available_time + pd.Timedelta(days=1)).all()


def test_stage_stats_and_support_gate_inputs() -> None:
    frame = pd.DataFrame(
        {
            "split": ["test"] * 4,
            "side": [1, 1, 1, -1],
            "entry_time": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-02-01", "2024-03-01"], utc=True
            ),
        }
    )
    assert support.stage_stats(frame, "test") == {
        "events": 4,
        "longs": 3,
        "shorts": 1,
        "minority_side_share": 0.25,
        "max_month_share": 0.5,
    }


def test_immutable_writes_are_idempotent_and_reject_drift(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    support.write_immutable_json({"a": 1}, destination)
    support.write_immutable_json({"a": 1}, destination)
    with pytest.raises(RuntimeError, match="immutable HVMRSVP artifact"):
        support.write_immutable_json({"a": 2}, destination)

    csv_destination = tmp_path / "artifact.csv.gz"
    support.write_immutable_csv(pd.DataFrame({"a": [1.0]}), csv_destination)
    first_hash = support.sha(csv_destination)
    support.write_immutable_csv(pd.DataFrame({"a": [1.0]}), csv_destination)
    assert support.sha(csv_destination) == first_hash
    with pytest.raises(RuntimeError, match="immutable HVMRSVP artifact"):
        support.write_immutable_csv(pd.DataFrame({"a": [2.0]}), csv_destination)


def test_run_with_mocked_http_materialization_and_db_writes_bound_terminal_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_dir = tmp_path / "sources"
    control_dir = tmp_path / "controls"
    monkeypatch.setattr(support, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(support, "PAIR_PANEL", source_dir / "pair.csv.gz")
    monkeypatch.setattr(support, "FEATURE_PANEL", source_dir / "features.csv.gz")
    monkeypatch.setattr(support, "SOURCE_MANIFEST", source_dir / "manifest.json")
    monkeypatch.setattr(support, "CLOCK", tmp_path / "clock.csv.gz")
    monkeypatch.setattr(support, "CONTROL_DIR", control_dir)
    monkeypatch.setattr(support, "RESULT", tmp_path / "result.json")

    pair = _pair(pd.date_range("2024-01-01T00:00:00Z", periods=3, freq="1d"))
    transport = {
        "source_days": len(pair),
        "normalized_panel_sha256": support._normalized_panel_hash(pair),
        "responses": [],
    }
    features = _features(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"])
    monkeypatch.setattr(support, "download_pair_panel", lambda workers=8: (pair, transport))
    monkeypatch.setattr(support, "load_bars", lambda: _bars(pd.Timestamp("2024-01-01T00:00:00Z"), 2))
    monkeypatch.setattr(support, "feature_panel", lambda _pair, _bars: features)

    result = support.run(workers=2)
    manifest = json.loads((source_dir / "manifest.json").read_text())
    assert json.loads((tmp_path / "result.json").read_text()) == result
    assert result["policy_id"] == "HVMRSVP-24"
    assert result["ranking"] == {
        "lookback_source_days": 252,
        "minimum_prior_source_days": 126,
        "current_excluded": True,
    }
    assert result["reservation"]["scope"] == "global"
    assert result["reservation"]["interval"] == "half_open"
    assert set(result["controls"]) == set(support.CONTROLS)
    assert all(not item["promotion_authorized"] for item in result["controls"].values())
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert manifest["pair_panel"]["normalized_sha256"] == transport["normalized_panel_sha256"]
    assert manifest["bars_query"] == support.QUERY
    assert result["manifest_hash"] == support.canonical_hash(
        {key: value for key, value in result.items() if key != "manifest_hash"}
    )
