from __future__ import annotations

import csv
import gzip
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from training import evaluate_gdelt_narrative_economic_selection as evaluator


def _bars(start: datetime, end: datetime) -> tuple[Any, ...]:
    return tuple(
        evaluator.prereg.MarketBar(
            open_time=timestamp,
            open=100.0,
            high=110.0,
            low=90.0,
            close=100.0,
        )
        for timestamp in evaluator.prereg._complete_bar_grid(start, end)
    )


def _funding(start: datetime, end: datetime) -> tuple[Any, ...]:
    times = evaluator.prereg._complete_funding_grid(start, end)
    return tuple(
        evaluator.prereg.FundingMark(
            timestamp=timestamp,
            mark_price=100.0,
            funding_rate=0.001 if index == 3 else 0.0,
        )
        for index, timestamp in enumerate(times)
    )


def _schedule(start: datetime, side: int) -> Any:
    source_date = start.date() - timedelta(days=2)
    decision = datetime.combine(
        source_date, datetime.min.time(), tzinfo=evaluator.prereg.UTC
    ) + timedelta(hours=48, minutes=15)
    event = evaluator.prereg.ScheduledEvent(
        source_date=source_date,
        decision_time=decision,
        entry_time=decision + timedelta(minutes=10),
        exit_time=decision + timedelta(minutes=10, days=3),
        side=side,
    )
    return evaluator.prereg.ScheduleResult(1, 1, 0, (event,))


def test_frozen_ancestry_and_source_support_are_exact() -> None:
    expected = {
        evaluator.PREREGISTRATION: evaluator.PREREGISTRATION_SHA256,
        evaluator.PREREGISTRATION_SOURCE: evaluator.PREREGISTRATION_SOURCE_SHA256,
        evaluator.PREREGISTRATION_DOCUMENT: (
            evaluator.PREREGISTRATION_DOCUMENT_SHA256
        ),
        evaluator.SOURCE_SUPPORT_EVALUATOR: (
            evaluator.SOURCE_SUPPORT_EVALUATOR_SHA256
        ),
        evaluator.SOURCE_SUPPORT_REPORT: evaluator.SOURCE_SUPPORT_REPORT_SHA256,
    }
    for path, expected_hash in expected.items():
        assert evaluator.sha256_file(path) == expected_hash
    report = evaluator.validate_source_support_report()
    assert report["decision"] == "advance_to_market"
    assert report["family_support"]["passing_variant_count"] == 17
    assert report["outcome_boundary"]["outcomes_opened"] is False


def test_reconstructed_schedules_equal_the_committed_source_report() -> None:
    report = evaluator.validate_source_support_report()
    rows = evaluator.source_support.load_daily_rows()
    schedules = evaluator.build_schedules(rows)
    evaluator.validate_schedules_against_source_support(schedules, report)
    assert tuple(schedules) == evaluator.prereg.FAMILY_VARIANT_IDS


def test_source_unsupported_variant_never_opens_market_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant_id = evaluator.prereg.FAMILY_VARIANT_IDS[0]
    policy = evaluator.prereg.variants()[0]
    empty = evaluator.prereg.ScheduleResult(0, 0, 0, ())
    source_report = {"variant_support": {variant_id: {"passes": False}}}

    def forbidden_simulation(*args, **kwargs):
        raise AssertionError("source-unsupported variant opened a market outcome")

    monkeypatch.setattr(evaluator, "simulate_market_path", forbidden_simulation)
    result, daily = evaluator.evaluate_supported_variant(
        variant_id,
        policy,
        {"train": empty, "selection": empty},
        source_report,
        {"train": (), "selection": ()},
        {"train": (), "selection": ()},
        evaluator.source_support._split_bounds(),
    )
    assert result["outcome_status"] == "not_opened_source_unsupported"
    assert result["train"]["market_outcome_opened"] is False
    assert result["selection"]["base_2bps_per_side"] is None
    assert np.array_equal(daily, np.zeros(365))


@pytest.mark.parametrize("side", (-1, 1))
@pytest.mark.parametrize("side_cost_bps", (2.0, 4.0))
def test_simulator_matches_frozen_market_path_and_daily_identity(
    side: int, side_cost_bps: float
) -> None:
    start = evaluator.prereg.parse_utc("2021-01-03T00:00:00Z")
    end = evaluator.prereg.parse_utc("2021-01-08T00:00:00Z")
    bars = _bars(start, end)
    funding = _funding(start, end)
    schedule = _schedule(start, side)
    expected = evaluator.prereg.evaluate_market_path(
        schedule,
        bars,
        funding,
        split_start=start,
        split_end_exclusive=end,
        side_cost_bps=side_cost_bps,
    )
    observed, daily = evaluator.simulate_market_path(
        schedule,
        bars,
        funding,
        split_start=start,
        split_end_exclusive=end,
        side_cost_bps=side_cost_bps,
    )
    assert observed == pytest.approx(expected)
    assert len(daily) == 5
    assert float(daily.sum()) == pytest.approx(
        math.log(1.0 + float(observed["absolute_return"])), abs=1e-12
    )


def test_daily_endpoint_captures_overnight_gap_and_ignores_exit_bar_extremes() -> None:
    start = evaluator.prereg.parse_utc("2021-01-03T00:00:00Z")
    end = evaluator.prereg.parse_utc("2021-01-08T00:00:00Z")
    bars = list(_bars(start, end))
    day_one_close = 288 - 1
    bars[day_one_close] = evaluator.prereg.MarketBar(
        open_time=bars[day_one_close].open_time,
        open=100.0,
        high=110.0,
        low=100.0,
        close=110.0,
    )
    day_two_open = 288
    bars[day_two_open] = evaluator.prereg.MarketBar(
        open_time=bars[day_two_open].open_time,
        open=90.0,
        high=100.0,
        low=90.0,
        close=100.0,
    )
    funding = tuple(
        evaluator.prereg.FundingMark(timestamp=time, mark_price=100.0, funding_rate=0.0)
        for time in evaluator.prereg._complete_funding_grid(start, end)
    )
    schedule = _schedule(start, 1)
    baseline, daily = evaluator.simulate_market_path(
        schedule,
        bars,
        funding,
        split_start=start,
        split_end_exclusive=end,
        side_cost_bps=2.0,
    )
    assert daily[0] == pytest.approx(math.log(1.0998), abs=1e-12)
    assert daily[1] == pytest.approx(math.log(0.9998 / 1.0998), abs=1e-12)

    exit_index = int(
        (schedule.admitted_events[0].exit_time - start)
        / evaluator.prereg.BAR_INTERVAL
    )
    exit_bar = bars[exit_index]
    bars[exit_index] = evaluator.prereg.MarketBar(
        open_time=exit_bar.open_time,
        open=exit_bar.open,
        high=10_000.0,
        low=0.1,
        close=10_000.0,
    )
    extreme, extreme_daily = evaluator.simulate_market_path(
        schedule,
        bars,
        funding,
        split_start=start,
        split_end_exclusive=end,
        side_cost_bps=2.0,
    )
    assert extreme == pytest.approx(baseline)
    assert extreme_daily == pytest.approx(daily)


def _write_market(path: Path, start: datetime, *, omit_last: bool = False) -> None:
    times = evaluator.prereg._complete_bar_grid(start, start + timedelta(days=1))
    if omit_last:
        times = times[:-1]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evaluator.EXPECTED_MARKET_COLUMNS)
        writer.writeheader()
        for timestamp in times:
            writer.writerow(
                {
                    "date": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": "100",
                    "high": "101",
                    "low": "99",
                    "close": "100",
                    "volume": "1",
                    "quote_asset_volume": "100",
                    "number_of_trades": "1",
                    "taker_buy_base": "0.5",
                    "taker_buy_quote": "50",
                }
            )


def _write_funding(path: Path, start: datetime, *, bad_offset: bool = False) -> None:
    times = evaluator.prereg._complete_funding_grid(start, start + timedelta(days=1))
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evaluator.EXPECTED_FUNDING_COLUMNS)
        writer.writeheader()
        for timestamp in times:
            mark_ms = int(timestamp.timestamp() * 1_000)
            offset = 61_000 if bad_offset else 1_000
            funding_ms = mark_ms + offset
            funding_time = datetime.fromtimestamp(
                funding_ms / 1_000, tz=evaluator.prereg.UTC
            )
            writer.writerow(
                {
                    "funding_time_ms": funding_ms,
                    "funding_time_utc": funding_time.strftime(
                        "%Y-%m-%dT%H:%M:%S.%fZ"
                    ),
                    "symbol": "BTCUSDT",
                    "funding_rate": "0.0001",
                    "settlement_mark_price": "100",
                    "mark_open_time_ms": mark_ms,
                    "mark_open_time_utc": timestamp.strftime(
                        "%Y-%m-%dT%H:%M:%S.%fZ"
                    ),
                    "funding_time_offset_ms": offset,
                    "mark_source": "binance_8h_mark_price_kline_open",
                }
            )


def test_execution_loaders_require_complete_grids_and_audited_funding_mapping(
    tmp_path: Path,
) -> None:
    start = evaluator.prereg.parse_utc("2021-01-01T00:00:00Z")
    end = start + timedelta(days=1)
    market = tmp_path / "market.csv.gz"
    funding = tmp_path / "funding.csv.gz"
    _write_market(market, start)
    _write_funding(funding, start)
    assert len(
        evaluator.load_market_bars(
            market, start=start.isoformat(), end_exclusive=end.isoformat()
        )
    ) == 288
    assert len(
        evaluator.load_funding_marks(
            funding, start=start.isoformat(), end_exclusive=end.isoformat()
        )
    ) == 3
    _write_market(market, start, omit_last=True)
    with pytest.raises(ValueError, match="5m grid"):
        evaluator.load_market_bars(
            market, start=start.isoformat(), end_exclusive=end.isoformat()
        )
    _write_funding(funding, start, bad_offset=True)
    with pytest.raises(ValueError, match="timestamp mapping"):
        evaluator.load_funding_marks(
            funding, start=start.isoformat(), end_exclusive=end.isoformat()
        )


def test_qualifiers_apply_frozen_train_and_selection_thresholds() -> None:
    passing = {
        "absolute_return": 0.01,
        "cagr": 0.05,
        "strict_mdd": 0.04,
        "cagr_to_strict_mdd": 1.25,
        "full_calendar_days": 365.0,
        "trade_count": 24,
    }
    stress = {**passing, "absolute_return": 0.001}
    assert all(evaluator.qualifier_checks(passing, stress, split="train").values())
    assert all(
        evaluator.qualifier_checks(passing, stress, split="selection").values()
    )
    failing = {**passing, "strict_mdd": 0.250001}
    assert evaluator.qualifier_checks(failing, stress, split="train")[
        "maximum_strict_mdd"
    ] is False
    stress["absolute_return"] = 0.0
    assert evaluator.qualifier_checks(passing, stress, split="train")[
        "stress_absolute_return_positive"
    ] is False


def test_romano_wolf_is_deterministic_stepdown_and_fail_closed() -> None:
    family = evaluator.prereg.FAMILY_VARIANT_IDS
    x = np.arange(365, dtype=np.float64)
    returns = {
        variant_id: 0.001 * np.sin(x / (7.0 + index)) + 0.00001 * index
        for index, variant_id in enumerate(family)
    }
    constant_id = family[2]
    returns[family[1]] = returns[family[0]].copy()
    returns[constant_id] = np.full(365, 0.0001)
    eligible = family[:3]
    first = evaluator.romano_wolf_stepdown(
        returns, eligible, draws=250, block_days=7, seed=123, batch_draws=37
    )
    second = evaluator.romano_wolf_stepdown(
        returns, eligible, draws=250, block_days=7, seed=123, batch_draws=37
    )
    assert first == second
    assert first["adjusted_p"][constant_id] == 1.0
    assert first["variance_positive"][constant_id] is False
    for variant_id in family[3:]:
        assert first["adjusted_p"][variant_id] == 1.0
    ordered = first["ordered_tested_variant_ids"]
    values = [first["adjusted_p"][variant_id] for variant_id in ordered]
    assert values == sorted(values)
    assert first["raw_stepdown_p"][family[0]] == first["raw_stepdown_p"][family[1]]
    assert first["equal_observed_t_removed_as_one_group"] is True


def test_champion_selection_uses_ratio_mdd_then_identifier() -> None:
    first, second, third = evaluator.prereg.FAMILY_VARIANT_IDS[:3]
    results = {
        first: {
            "policy_hash": "a" * 64,
            "selection": {
                "base_2bps_per_side": {
                    "cagr_to_strict_mdd": 2.0,
                    "strict_mdd": 0.10,
                }
            },
        },
        second: {
            "policy_hash": "b" * 64,
            "selection": {
                "base_2bps_per_side": {
                    "cagr_to_strict_mdd": 2.0,
                    "strict_mdd": 0.09,
                }
            },
        },
        third: {
            "policy_hash": "c" * 64,
            "selection": {
                "base_2bps_per_side": {
                    "cagr_to_strict_mdd": 1.9,
                    "strict_mdd": 0.01,
                }
            },
        },
    }
    assert evaluator.select_champion(results, (first, second, third)) == (
        second,
        "b" * 64,
    )
    assert evaluator.select_champion(results, ()) == (None, None)


def test_economic_result_is_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"decision": "synthetic", "manifest_hash": "a" * 64}
    monkeypatch.setattr(evaluator, "build_report", lambda: payload)
    output = tmp_path / "economic.json"
    assert evaluator.write_once(output) == payload
    assert json.loads(output.read_text()) == payload
    with pytest.raises(FileExistsError, match="write-once"):
        evaluator.write_once(output)


def _premarket_seal() -> dict[str, object]:
    return {
        "protocol_version": "gdelt_gnrc_premarket_access_seal_v1",
        "source_support_report_path": str(evaluator.SOURCE_SUPPORT_REPORT),
        "source_support_report_sha256": evaluator.SOURCE_SUPPORT_REPORT_SHA256,
        "evaluator_source_path": str(evaluator.EVALUATOR_SOURCE),
        "evaluator_source_sha256": "1" * 64,
        "protocol_document_path": str(evaluator.PROTOCOL_DOCUMENT),
        "protocol_document_sha256": "2" * 64,
        "test_source_path": str(evaluator.TEST_SOURCE),
        "test_source_sha256": "3" * 64,
        "market_data_path": str(evaluator.MARKET_DATA),
        "market_data_sha256": evaluator.MARKET_DATA_SHA256,
        "market_manifest_path": str(evaluator.MARKET_MANIFEST),
        "market_manifest_sha256": evaluator.MARKET_MANIFEST_SHA256,
        "funding_data_path": str(evaluator.FUNDING_DATA),
        "funding_data_sha256": evaluator.FUNDING_DATA_SHA256,
        "funding_manifest_path": str(evaluator.FUNDING_MANIFEST),
        "funding_manifest_sha256": evaluator.FUNDING_MANIFEST_SHA256,
        "market_values_inspected_before_seal": False,
        "funding_values_inspected_before_seal": False,
        "post_2023_outcomes_inspected_before_seal": False,
        "sealed_at": "2026-07-22T00:00:00Z",
    }


def test_premarket_seal_binds_code_tests_protocol_and_outcome_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _premarket_seal()
    expected_hashes = {
        evaluator.SOURCE_SUPPORT_REPORT: evaluator.SOURCE_SUPPORT_REPORT_SHA256,
        evaluator.EVALUATOR_SOURCE: "1" * 64,
        evaluator.PROTOCOL_DOCUMENT: "2" * 64,
        evaluator.TEST_SOURCE: "3" * 64,
        evaluator.MARKET_DATA: evaluator.MARKET_DATA_SHA256,
        evaluator.MARKET_MANIFEST: evaluator.MARKET_MANIFEST_SHA256,
        evaluator.FUNDING_DATA: evaluator.FUNDING_DATA_SHA256,
        evaluator.FUNDING_MANIFEST: evaluator.FUNDING_MANIFEST_SHA256,
    }
    monkeypatch.setattr(evaluator, "_load_json", lambda path: payload)
    monkeypatch.setattr(
        evaluator, "sha256_file", lambda path: expected_hashes[Path(path)]
    )
    assert evaluator.validate_premarket_access_seal() == payload
    payload["market_values_inspected_before_seal"] = True
    with pytest.raises(ValueError, match="premature outcome access"):
        evaluator.validate_premarket_access_seal()


def test_build_report_refuses_missing_premarket_seal_before_market_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        evaluator, "PREMARKET_ACCESS_SEAL", tmp_path / "missing-seal.json"
    )

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("market loader ran before premarket seal validation")

    monkeypatch.setattr(evaluator, "load_market_bars", forbidden_loader)
    monkeypatch.setattr(evaluator, "load_funding_marks", forbidden_loader)
    with pytest.raises(FileNotFoundError):
        evaluator.build_report()


def _market_manifests() -> tuple[dict[str, object], dict[str, object]]:
    market: dict[str, object] = {
        "config": {
            "symbol": "BTCUSDT",
            "interval": "5m",
            "start": "2020-01-01",
            "end": "2024-01-01",
        },
        "protocol": {
            "source": "official Binance USD-M daily kline archives",
            "archive_checksums_verified": True,
            "end_is_exclusive": True,
            "outcomes_opened": False,
        },
        "combined_output": str(evaluator.MARKET_DATA),
        "combined_sha256": evaluator.MARKET_DATA_SHA256,
        "rows": 1_461 * 24 * 12,
        "first_date": "2020-01-01 00:00:00",
        "last_date": "2023-12-31 23:55:00",
        "columns": list(evaluator.EXPECTED_MARKET_COLUMNS),
    }
    funding_core: dict[str, object] = {
        "protocol_version": "btc_um_funding_settlement_marks_2020_2023_v1",
        "outcomes_opened": False,
        "strategy_outcomes_calculated": [],
        "data": {
            "path": str(evaluator.FUNDING_DATA),
            "sha256": evaluator.FUNDING_DATA_SHA256,
            "rows": 1_461 * 3,
            "columns": list(evaluator.EXPECTED_FUNDING_COLUMNS),
        },
        "mapping": {
            "funding_time": "exact returned fundingTime retained",
            "mark": "open of floor(fundingTime, 8h) official mark-price kline",
            "maximum_allowed_timestamp_offset_ms": 60_000,
        },
    }
    funding = {
        **funding_core,
        "manifest_hash": evaluator.canonical_hash(funding_core),
        "created_at": "2026-07-17T00:00:00Z",
    }
    return market, funding


def test_market_manifest_validation_rejects_opened_outcome_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market, funding = _market_manifests()
    expected_hashes = {
        evaluator.MARKET_DATA: evaluator.MARKET_DATA_SHA256,
        evaluator.MARKET_MANIFEST: evaluator.MARKET_MANIFEST_SHA256,
        evaluator.FUNDING_DATA: evaluator.FUNDING_DATA_SHA256,
        evaluator.FUNDING_MANIFEST: evaluator.FUNDING_MANIFEST_SHA256,
    }
    monkeypatch.setattr(
        evaluator, "sha256_file", lambda path: expected_hashes[Path(path)]
    )
    monkeypatch.setattr(
        evaluator,
        "_load_json",
        lambda path: market if Path(path) == evaluator.MARKET_MANIFEST else funding,
    )
    evaluator.validate_market_manifests()
    protocol = market["protocol"]
    assert isinstance(protocol, dict)
    protocol["outcomes_opened"] = True
    with pytest.raises(ValueError, match="market manifest contract"):
        evaluator.validate_market_manifests()
