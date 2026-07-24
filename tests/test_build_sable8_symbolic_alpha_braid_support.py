from __future__ import annotations

from collections import OrderedDict
import csv
import gzip
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_sable8_symbolic_alpha_braid_support as s
from training import preregister_sable8_symbolic_alpha_braid as p


def _write_source(
    path: Path,
    header: tuple[str, ...],
    rows: list[list[str]],
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _market_row(timestamp: str, *, close: str = "100") -> list[str]:
    values = {
        "date": timestamp,
        "open": "100",
        "high": "101",
        "low": "99",
        "close": close,
        "volume": "10",
        "quote_asset_volume": "1000",
        "number_of_trades": "50",
        "taker_buy_base": "5",
        "taker_buy_quote": "520",
        "tic": "hidden",
        "day": "hidden",
        "dxy": "100",
        "kimchi_premium": "1.5",
        "usdkrw": "1300",
        "btckrw": "hidden",
        "dxy_available": "1",
        "kimchi_available": "1",
        "usdkrw_available": "1",
        "external_any_available": "hidden",
        "dxy_zscore": "hidden",
        "dxy_momentum": "hidden",
        "kimchi_premium_zscore": "hidden",
        "kimchi_premium_change": "hidden",
        "usdkrw_zscore": "hidden",
        "usdkrw_momentum": "hidden",
        "open_interest": "1000000",
        "open_interest_value": "hidden",
        "cmc_circulating_supply": "hidden",
        "open_interest_available": "1",
    }
    return [values[column] for column in p.MARKET_PHYSICAL_HEADER]


def _spec(path: Path, output: Path) -> dict[str, object]:
    with gzip.open(path, "rb") as handle:
        header_bytes = handle.readline()
    return {
        "source": str(path),
        "source_sha256": s.sha256_file(path),
        "physical_header": p.MARKET_PHYSICAL_HEADER,
        "physical_header_sha256": hashlib.sha256(
            header_bytes
        ).hexdigest(),
        "allowlist": p.MARKET_ALLOWLIST,
        "timestamp_field": "date",
        "output": str(output),
    }


def _market_bundle(
    *,
    periods: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range(
        "2019-11-01T00:00:00Z",
        periods=periods,
        freq="5min",
    )
    index = np.arange(periods, dtype=float)
    returns = 0.00001 + 0.00002 * np.sin(index / 19.0)
    close = 10_000.0 * np.exp(np.cumsum(returns))
    quote = 1_000_000.0 + 200_000.0 * (1.0 + np.sin(index / 31.0))
    taker = quote * (0.5 + 0.1 * np.sin(index / 17.0))
    availability = np.ones(periods, dtype=float)
    market = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "quote_asset_volume": quote,
            "taker_buy_quote": taker,
            "dxy": 100.0 * np.exp(index * 1e-7),
            "kimchi_premium": 1.0 + 0.1 * np.sin(index / 101.0),
            "usdkrw": 1_300.0 * np.exp(index * 2e-8),
            "dxy_available": availability,
            "kimchi_available": availability,
            "usdkrw_available": availability,
            "open_interest": 1_000_000.0 * np.exp(index * 4e-8),
            "open_interest_available": availability,
        },
        columns=p.MARKET_ALLOWLIST,
    )
    start = dates[0].floor("8h")
    end = dates[-1].ceil("8h") + pd.Timedelta(hours=8)
    funding_times = pd.date_range(start, end, freq="8h")
    funding = pd.DataFrame(
        {
            "date": funding_times,
            "funding_rate": np.full(len(funding_times), 0.0001),
            "funding_time": funding_times,
        },
        columns=p.FUNDING_ALLOWLIST,
    )
    premium_times = pd.date_range(start, end, freq="1h")
    premium = pd.DataFrame(
        {
            "date": premium_times - pd.Timedelta(hours=1),
            "close": np.linspace(-0.001, 0.001, len(premium_times)),
            "close_time": premium_times,
        },
        columns=p.PREMIUM_ALLOWLIST,
    )
    return market, funding, premium


def _primitive_rows(count: int) -> pd.DataFrame:
    boundaries = pd.date_range(
        "2020-01-01T00:00:00Z",
        periods=count,
        freq="8h",
    )
    rows: list[dict[str, object]] = []
    for index, boundary in enumerate(boundaries):
        record: dict[str, object] = {
            "boundary": boundary,
            "state_cutoff": boundary + pd.Timedelta(minutes=5),
            "decision_time": boundary + pd.Timedelta(minutes=10),
            "execution_time": boundary + pd.Timedelta(minutes=15),
            "otherwise_eligible": True,
            "oi_fresh": True,
            "kimchi_fresh": True,
            "usdkrw_fresh": True,
            "dxy_fresh": True,
        }
        for offset, primitive in enumerate(p.PRIMITIVES):
            record[primitive] = float(index + offset / 100.0)
        rows.append(record)
    return pd.DataFrame(rows)


def test_preregistration_binding_is_exact() -> None:
    payload = s.validate_preregistration()
    assert payload["manifest_hash"] == s.PREREGISTRATION_MANIFEST_HASH
    assert s.sha256_file(s.PREREGISTRATION) == s.PREREGISTRATION_SHA256


def test_timestamp_parser_accepts_iso_and_binance_milliseconds() -> None:
    expected = pd.Timestamp("2021-01-01T00:00:00Z")
    assert s._parse_timestamp(
        "2021-01-01T00:00:00Z",
        field="date",
    ) == expected
    assert s._parse_timestamp(
        "1609459200000",
        field="funding_time",
    ) == expected


def test_stream_projection_stops_before_future_numeric_poison(
    tmp_path: Path,
) -> None:
    source = tmp_path / "market.csv.gz"
    output = tmp_path / "cut.csv.gz"
    rows = [
        _market_row("2023-12-31T23:55:00Z"),
        _market_row("2024-01-01T00:00:00Z", close="NOT_NUMERIC"),
    ]
    rows[1][p.MARKET_PHYSICAL_HEADER.index("dxy_zscore")] = "POISON"
    rows[1].append("FUTURE_EXTRA_COLUMN")
    _write_source(source, p.MARKET_PHYSICAL_HEADER, rows)

    audit = s.stream_project_source(
        "market",
        _spec(source, output),
        output=output,
    )
    assert audit["rows"] == 1
    assert audit["post_cut_non_timestamp_values_converted"] == 0
    assert audit["unprojected_values_converted"] == 0
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        assert tuple(next(reader)) == p.MARKET_ALLOWLIST
        projected = list(reader)
    assert len(projected) == 1
    assert "POISON" not in str(projected)


def test_stream_projection_skips_pre2020_before_numeric_conversion(
    tmp_path: Path,
) -> None:
    source = tmp_path / "market.csv.gz"
    output = tmp_path / "cut.csv.gz"
    early = _market_row("2019-12-31T23:55:00Z", close="NOT_NUMERIC")
    early.append("EARLY_EXTRA_COLUMN")
    _write_source(
        source,
        p.MARKET_PHYSICAL_HEADER,
        [
            early,
            _market_row("2020-01-01T00:00:00Z"),
            _market_row("2024-01-01T00:00:00Z"),
        ],
    )
    audit = s.stream_project_source(
        "market",
        _spec(source, output),
        output=output,
    )
    assert audit["rows"] == 1
    assert audit["skipped_before_start"] == 1
    assert audit["first_timestamp"] == "2020-01-01T00:00:00+00:00"


def test_stream_projection_rejects_physical_header_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "market.csv.gz"
    output = tmp_path / "cut.csv.gz"
    broken = tuple(reversed(p.MARKET_PHYSICAL_HEADER))
    _write_source(source, broken, [list(reversed(_market_row("2023-01-01")))])
    spec = _spec(source, output)
    spec["physical_header"] = p.MARKET_PHYSICAL_HEADER
    with pytest.raises(RuntimeError, match="physical header/order"):
        s.stream_project_source(
            "market",
            spec,
            output=output,
        )


def test_unavailable_context_may_be_blank_but_cannot_authorize_freshness(
    tmp_path: Path,
) -> None:
    source = tmp_path / "market.csv.gz"
    output = tmp_path / "cut.csv.gz"
    first = _market_row("2023-12-31T23:55:00Z")
    first[p.MARKET_PHYSICAL_HEADER.index("dxy")] = ""
    first[p.MARKET_PHYSICAL_HEADER.index("dxy_available")] = "0"
    _write_source(
        source,
        p.MARKET_PHYSICAL_HEADER,
        [first, _market_row("2024-01-01T00:00:00Z")],
    )
    audit = s.stream_project_source(
        "market",
        _spec(source, output),
        output=output,
    )
    assert audit["rows"] == 1

    fresh = _market_row("2023-12-31T23:55:00Z")
    fresh[p.MARKET_PHYSICAL_HEADER.index("dxy")] = ""
    _write_source(
        source,
        p.MARKET_PHYSICAL_HEADER,
        [fresh, _market_row("2024-01-01T00:00:00Z")],
    )
    changed = _spec(source, tmp_path / "other.csv.gz")
    with pytest.raises(RuntimeError, match="dxy is not numeric"):
        s.stream_project_source(
            "market",
            changed,
            output=tmp_path / "other.csv.gz",
        )


def test_zero_volume_bar_is_preserved_as_primitive_missingness(
    tmp_path: Path,
) -> None:
    source = tmp_path / "market.csv.gz"
    output = tmp_path / "cut.csv.gz"
    zero = _market_row("2023-12-31T23:55:00Z")
    zero[p.MARKET_PHYSICAL_HEADER.index("quote_asset_volume")] = "0"
    zero[p.MARKET_PHYSICAL_HEADER.index("taker_buy_quote")] = "0"
    _write_source(
        source,
        p.MARKET_PHYSICAL_HEADER,
        [zero, _market_row("2024-01-01T00:00:00Z")],
    )
    audit = s.stream_project_source(
        "market",
        _spec(source, output),
        output=output,
    )
    assert audit["rows"] == 1

    invalid = list(zero)
    invalid[p.MARKET_PHYSICAL_HEADER.index("taker_buy_quote")] = "1"
    _write_source(
        source,
        p.MARKET_PHYSICAL_HEADER,
        [invalid, _market_row("2024-01-01T00:00:00Z")],
    )
    with pytest.raises(RuntimeError, match="quote/taker identity"):
        s.stream_project_source(
            "market",
            _spec(source, tmp_path / "invalid.csv.gz"),
            output=tmp_path / "invalid.csv.gz",
        )


def test_primitive_equations_funding_windows_and_availability() -> None:
    market, funding, premium = _market_bundle()
    end = market["date"].iloc[-1].floor("8h") + pd.Timedelta(hours=8)
    frame = s.build_primitive_frame(
        market,
        funding,
        premium,
        end=end,
    )
    row = frame.loc[frame["otherwise_eligible"]].iloc[-1]
    t = int(market.index[market["date"] == row["boundary"]][0])
    close = market["close"].to_numpy(float)
    returns = np.full(len(market), np.nan)
    returns[1:] = np.log(close[1:] / close[:-1])
    rv = np.sum(np.square(returns[t - 287 : t + 1]))
    bv = (np.pi / 2.0) * np.sum(
        np.abs(returns[t - 287 : t + 1])
        * np.abs(returns[t - 288 : t])
    )

    assert row["price_return_1d"] == pytest.approx(
        np.log(close[t] / close[t - 288])
    )
    assert row["jump_share_1d"] == pytest.approx(max(rv - bv, 0.0) / rv)
    assert row["signed_jump_1d"] == pytest.approx(
        np.sum(np.power(returns[t - 287 : t + 1], 3)) / rv**1.5
    )
    assert row["funding_sum_24h"] == pytest.approx(0.0003)
    cutoff = row["state_cutoff"]
    expected_premium = premium.loc[
        (premium["close_time"] > cutoff - pd.Timedelta(hours=8))
        & (premium["close_time"] <= cutoff),
        "close",
    ]
    assert len(expected_premium) == 8
    assert row["premium_mean_8h"] == pytest.approx(expected_premium.mean())
    assert row["oi_fresh"]
    assert row["dxy_fresh"]

    current = int(market.index[market["date"] == row["boundary"]][0])
    market.loc[current, "dxy_available"] = 0.0
    stale = s.build_primitive_frame(
        market,
        funding,
        premium,
        end=end,
    )
    stale_row = stale.loc[stale["boundary"] == row["boundary"]].iloc[0]
    assert not stale_row["dxy_fresh"]
    assert np.isnan(stale_row["dxy_change_1d"])


def test_volume_clock_uses_prior_target_and_inclusive_backward_interval() -> None:
    market, funding, premium = _market_bundle()
    end = market["date"].iloc[-1].floor("8h") + pd.Timedelta(hours=8)
    row = s.build_primitive_frame(
        market,
        funding,
        premium,
        end=end,
    ).loc[lambda frame: frame["otherwise_eligible"]].iloc[-1]
    t = int(market.index[market["date"] == row["boundary"]][0])
    quote = market["quote_asset_volume"].to_numpy(float)
    aggressor = (
        2.0 * market["taker_buy_quote"].to_numpy(float) - quote
    )
    target = 0.25 * quote[t - 288 : t].sum()
    total = 0.0
    j = t
    while j >= 0 and total < target:
        total += quote[j]
        j -= 1
    j += 1
    expected = (
        aggressor[j : t + 1].sum() / quote[j : t + 1].sum()
    ) / (t - j + 1)
    assert row["volume_clock_flow_speed_25"] == pytest.approx(expected)


def test_token_history_is_strict_prior_and_sequence_resets_on_gap() -> None:
    primitives = _primitive_rows(190)
    tokens = s.build_token_table(primitives)
    assert not tokens.loc[:179, "line_ready"].any()
    assert tokens.loc[180, "line_ready"]
    assert not tokens.loc[:184, "sequence_ready"].any()
    assert tokens.loc[185, "sequence_ready"]

    broken = _primitive_rows(195)
    broken.loc[186, p.CORE_PRIMITIVES[0]] = np.nan
    broken_tokens = s.build_token_table(broken)
    assert not broken_tokens.loc[186:191, "sequence_ready"].any()
    assert broken_tokens.loc[192, "sequence_ready"]

    full = _primitive_rows(195)
    first_after_gap_boundary = full.loc[187, "boundary"]
    missing = full.drop(index=186).reset_index(drop=True)
    missing_tokens = s.build_token_table(missing)
    after_gap = missing_tokens.index[
        missing_tokens["boundary"] == first_after_gap_boundary
    ][0]
    assert not missing_tokens.loc[after_gap : after_gap + 4, "sequence_ready"].any()
    assert missing_tokens.loc[after_gap + 5, "sequence_ready"]


def test_token_table_is_categorical_deterministic_and_outcome_free() -> None:
    tokens = s.build_token_table(_primitive_rows(190))
    first = s.token_table_bytes(tokens)
    second = s.token_table_bytes(tokens)
    assert first == second
    assert not set(tokens.columns).intersection(s.FORBIDDEN_SUPPORT_COLUMNS)
    ready = tokens.loc[tokens["sequence_ready"]].iloc[0]
    assert ready["canonical_line"].startswith("PRICE_RETURN_1D=")
    assert len(ready["sequence_signature"]) == 64


def test_prefix_replay_rebuilds_each_physical_cut_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, pd.Timestamp, Path]] = []

    def fake_stream(
        name: str,
        spec: object,
        *,
        start: pd.Timestamp,
        cutoff: pd.Timestamp,
        output: Path,
    ) -> dict[str, object]:
        del spec, start
        calls.append((name, cutoff, output))
        return {
            "cut_path": str(output),
            "cut_sha256": f"{name}-sha",
            "rows": 1,
            "first_timestamp": "2020-01-01T00:00:00+00:00",
            "last_timestamp": "2022-12-31T23:55:00+00:00",
            "stopped_before_timestamp": "2023-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(s, "stream_project_source", fake_stream)
    monkeypatch.setattr(s, "_load_market", lambda path: object())
    monkeypatch.setattr(s, "_load_funding", lambda path: object())
    monkeypatch.setattr(s, "_load_premium", lambda path: object())
    primitives = _primitive_rows(190)
    monkeypatch.setattr(
        s,
        "build_primitive_frame",
        lambda market, funding, premium, *, end: primitives,
    )
    full_tokens = s.build_token_table(primitives)
    audit = s.prefix_replay_audit(full_tokens)
    assert audit["passed"] is True
    assert [name for name, _, _ in calls] == [
        "market",
        "funding",
        "premium",
    ]
    assert all(cutoff == s.PREFIX_END for _, cutoff, _ in calls)


def test_support_gate_retires_insufficient_language_without_repair() -> None:
    primitives = _primitive_rows(190)
    tokens = s.build_token_table(primitives)
    result = s.evaluate_support(
        primitives,
        tokens,
        prefix_audit={"passed": True},
    )
    assert result["decision"] == "RETIRE"
    assert "development_count" in result["failed_gates"]
    assert result["gates"]["prefix_replay"] is True
    assert set(result["metrics"]["freshness"]["oi"]) == {
        "development_2020_2022",
        "report_only_2023",
    }
    assert set(
        result["metrics"]["core_missing_share"][p.CORE_PRIMITIVES[0]]
    ) == {
        "development_2020_2022",
        "report_only_2023",
    }


def test_market_source_with_outcome_column_is_rejected() -> None:
    market, funding, premium = _market_bundle()
    market["reward"] = 1.0
    with pytest.raises(RuntimeError, match="forbidden outcome"):
        s.build_primitive_frame(market, funding, premium)


def test_write_once_rejects_changed_bytes(tmp_path: Path) -> None:
    output = tmp_path / "artifact.bin"
    first = s._write_bytes_once(output, b"first")
    assert first == hashlib.sha256(b"first").hexdigest()
    assert s._write_bytes_once(output, b"first") == first
    with pytest.raises(RuntimeError, match="write-once"):
        s._write_bytes_once(output, b"second")
