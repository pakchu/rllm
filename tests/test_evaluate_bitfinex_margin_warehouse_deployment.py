from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from training import evaluate_bitfinex_margin_warehouse_deployment as evaluator


def _market(*, start: str = "2021-03-01T04:05:00Z", periods: int = 145) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="5min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.full(periods, 100.0),
            "high": np.full(periods, 100.0),
            "low": np.full(periods, 100.0),
            "close": np.full(periods, 100.0),
        }
    )


def _clock(market: pd.DataFrame, *, side: int = 1) -> dict[str, Any]:
    entry = evaluator._utc(market.iloc[0]["date"])
    exit_time = evaluator._utc(market.iloc[evaluator.FROZEN_CONFIG.hold_bars]["date"])
    return {
        "candidate": evaluator.POLICY_ID,
        "variant_id": evaluator.FAMILY_IDS[0],
        "control": "primary",
        "split": "synthetic",
        "symbol": "fUSD" if side > 0 else "fBTC",
        "entry_time": entry,
        "exit_time": exit_time,
        "side": side,
    }


def _funding() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "funding_time": pd.to_datetime([], utc=True),
            "symbol": pd.Series([], dtype="object"),
            "funding_rate": pd.Series([], dtype=float),
            "settlement_mark_price": pd.Series([], dtype=float),
        }
    )


def test_primary_clocks_and_economic_controls_are_deterministic() -> None:
    primary, _ = evaluator.load_primary_clocks()
    first = evaluator.derive_schedules(primary)
    second = evaluator.derive_schedules(primary)

    expected = {
        "bfmwd_w12_d3_z10_h12": {"train": 177, "selection": 61},
        "bfmwd_w24_d3_z10_h12": {"train": 147, "selection": 71},
        "bfmwd_w12_d6_z10_h12": {"train": 152, "selection": 62},
        "bfmwd_w24_d6_z10_h12": {"train": 131, "selection": 65},
    }
    assert tuple(first) == evaluator.FAMILY_IDS
    for variant_id, controls in first.items():
        assert tuple(controls) == evaluator.CONTROL_ORDER
        primary_schedule = controls["primary"].reset_index(drop=True)
        assert {
            stage: len(evaluator._window_schedule(primary_schedule, stage))
            for stage in evaluator.STAGE_ORDER
        } == expected[variant_id]
        assert bool(
            controls["direction_flip"]["side"]
            .reset_index(drop=True)
            .eq(-primary_schedule["side"])
            .all()
        )
        assert set(controls["fUSD_only"]["symbol"]) == {"fUSD"}
        assert set(controls["fBTC_only"]["symbol"]) == {"fBTC"}
        delayed = controls["extra_latency_one_bar"].reset_index(drop=True)
        assert bool(
            delayed["entry_time"]
            .eq(primary_schedule["entry_time"] + pd.Timedelta(minutes=5))
            .all()
        )
        assert bool(
            delayed["exit_time"]
            .eq(primary_schedule["exit_time"] + pd.Timedelta(minutes=5))
            .all()
        )
        for stage, (_, end) in evaluator.STAGE_WINDOWS.items():
            stage_delayed = evaluator._window_schedule(delayed, stage)
            assert bool(stage_delayed["exit_time"].lt(end).all())
        for control in evaluator.CONTROL_ORDER:
            assert evaluator._schedule_hash(
                controls[control]
            ) == evaluator._schedule_hash(second[variant_id][control])


def test_random_control_uses_frozen_ascii_sha_contract() -> None:
    variant_id = evaluator.FAMILY_IDS[0]
    timestamp = evaluator._utc("2021-05-03T17:20:00Z")
    material = f"{evaluator.POLICY_ID}|{variant_id}|{timestamp.isoformat()}".encode(
        "ascii"
    )
    expected = 1 if hashlib.sha256(material).digest()[0] & 1 == 0 else -1
    assert evaluator._deterministic_random_side(variant_id, timestamp) == expected


def test_evaluator_is_directly_executable_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(evaluator.EVALUATOR_SOURCE), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--freeze" in completed.stdout
    assert "--prepare-stage-source" in completed.stdout
    assert "--stage" in completed.stdout


def test_freeze_opens_no_market_funding_or_simulation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str] = []
    real_sha = evaluator._sha256

    def tracking_sha(path: str | Path) -> str:
        seen.append(str(path))
        return real_sha(path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("freeze tried to open or simulate an execution outcome")

    for stage in evaluator.STAGE_ORDER:
        monkeypatch.setitem(
            evaluator.STAGE_OUTPUTS, stage, tmp_path / f"unused-{stage}.json"
        )
    monkeypatch.setattr(evaluator, "_sha256", tracking_sha)
    monkeypatch.setattr(evaluator, "prepare_stage_source", forbidden)
    monkeypatch.setattr(evaluator, "load_execution_window", forbidden)
    monkeypatch.setattr(evaluator, "simulate_strict", forbidden)
    output = tmp_path / "freeze.json"

    report = evaluator.freeze_evaluator(output)

    assert report["opened_windows"] == []
    assert report["sealed_windows"] == list(evaluator.STAGE_ORDER)
    assert report["execution_ohlc_rows_parsed_during_freeze"] == 0
    assert report["funding_rows_parsed_during_freeze"] == 0
    assert report["execution_outcome_data_bytes_hashed_during_freeze"] is False
    assert str(evaluator.LEGACY_MARKET) not in seen
    assert str(evaluator.LEGACY_FUNDING) not in seen
    assert evaluator.verify_evaluator_freeze(output) == report

    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    core["evaluation_config"] = {**core["evaluation_config"], "leverage": 1.0}
    output.write_text(json.dumps(evaluator._seal(core)), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration changed"):
        evaluator.verify_evaluator_freeze(output)


def test_freeze_is_write_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for stage in evaluator.STAGE_ORDER:
        monkeypatch.setitem(
            evaluator.STAGE_OUTPUTS, stage, tmp_path / f"unused-{stage}.json"
        )
    output = tmp_path / "freeze.json"
    evaluator.freeze_evaluator(output)
    with pytest.raises(FileExistsError):
        evaluator.freeze_evaluator(output)


def test_freeze_rejects_preexisting_stage_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for stage in evaluator.STAGE_ORDER:
        monkeypatch.setitem(
            evaluator.STAGE_OUTPUTS, stage, tmp_path / f"unused-{stage}.json"
        )
        monkeypatch.setitem(
            evaluator.STAGE_SOURCE_DIRS, stage, tmp_path / f"source-{stage}"
        )
        monkeypatch.setitem(
            evaluator.STAGE_SOURCE_MANIFESTS,
            stage,
            tmp_path / f"source-{stage}.json",
        )
    evaluator.STAGE_SOURCE_MANIFESTS["train"].write_text("opened")
    with pytest.raises(RuntimeError, match="after a stage source exists"):
        evaluator.freeze_evaluator(tmp_path / "freeze.json")


def test_slice_copies_only_declared_window_without_parsing_values(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv.gz"
    output = tmp_path / "slice.csv.gz"
    second_output = tmp_path / "second-slice.csv.gz"
    with gzip.open(source, "wt", encoding="utf-8", newline="") as handle:
        handle.write("date,open,high,low,close\n")
        handle.write("2020-12-31 23:55:00,SECRET,SECRET,SECRET,SECRET\n")
        handle.write("2021-01-01 00:00:00,100,101,99,100\n")
        handle.write("2022-12-31 23:55:00,100,101,99,100\n")
        # The copier must stop after the frozen row count without even parsing
        # the first excluded timestamp or seeing the future value fields.
        handle.write("NOT_A_TIMESTAMP,FUTURE,FUTURE,FUTURE,FUTURE\n")

    result = evaluator._slice_gzip_csv(
        source,
        output,
        timestamp_column="date",
        start=evaluator._utc("2021-01-01T00:00:00Z"),
        end=evaluator._utc("2023-01-01T00:00:00Z"),
        expected_rows=2,
    )

    assert result["rows"] == 2
    assert result["post_stage_numeric_rows_parsed"] == 0
    assert result["first_excluded_row_read"] is False
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        content = handle.read()
    assert "SECRET" not in content
    assert "FUTURE" not in content
    assert "2022-12-31 23:55:00" in content
    second = evaluator._slice_gzip_csv(
        source,
        second_output,
        timestamp_column="date",
        start=evaluator._utc("2021-01-01T00:00:00Z"),
        end=evaluator._utc("2023-01-01T00:00:00Z"),
        expected_rows=2,
    )
    assert result["sha256"] == second["sha256"]


def test_selection_source_checks_train_gate_before_parent_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "verify_evaluator_freeze",
        lambda: {"manifest_hash": "freeze"},
    )

    def failed_prior(*args: object, **kwargs: object) -> None:
        raise ValueError("train did not pass")

    def forbidden_digest(*args: object, **kwargs: object) -> str:
        raise AssertionError("parent digest ran before the train gate")

    monkeypatch.setattr(evaluator, "_verified_prior_reports", failed_prior)
    monkeypatch.setattr(evaluator, "_sha256", forbidden_digest)
    with pytest.raises(ValueError, match="train did not pass"):
        evaluator.prepare_stage_source("selection")


def test_train_source_never_hashes_sealed_parent_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = evaluator._stage_source_spec("train")
    freeze = {
        "manifest_hash": "freeze",
        "execution_source_specs": {"train": spec},
        "legacy_container_contract": {
            "market": {"path": str(evaluator.LEGACY_MARKET), "sha256": "m"},
            "funding": {"path": str(evaluator.LEGACY_FUNDING), "sha256": "f"},
        },
    }
    stage_dir = tmp_path / "train"
    manifest = tmp_path / "train-source.json"
    monkeypatch.setitem(evaluator.STAGE_SOURCE_DIRS, "train", stage_dir)
    monkeypatch.setitem(evaluator.STAGE_SOURCE_MANIFESTS, "train", manifest)
    monkeypatch.setattr(evaluator, "verify_evaluator_freeze", lambda: freeze)
    monkeypatch.setattr(
        evaluator, "_verified_prior_reports", lambda *args, **kwargs: []
    )

    def forbidden_digest(*args: object, **kwargs: object) -> str:
        raise AssertionError("train preparation hashed sealed parent bytes")

    def fake_slice(
        source: str | Path,
        output: str | Path,
        *,
        timestamp_column: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        expected_rows: int,
    ) -> dict[str, Any]:
        Path(output).write_bytes(b"stage-only")
        return {
            "path": str(output),
            "sha256": "slice",
            "rows": expected_rows,
            "expected_rows": expected_rows,
            "post_stage_numeric_rows_parsed": 0,
        }

    market_rows = len(
        pd.date_range(*evaluator.STAGE_WINDOWS["train"], freq="5min", inclusive="left")
    )
    funding_rows = len(
        pd.date_range(*evaluator.STAGE_WINDOWS["train"], freq="8h", inclusive="left")
    )
    monkeypatch.setattr(evaluator, "_sha256", forbidden_digest)
    monkeypatch.setattr(evaluator, "_slice_gzip_csv", fake_slice)
    monkeypatch.setattr(
        evaluator.strict_source,
        "_parse_market_window",
        lambda *args, **kwargs: (pd.DataFrame(index=range(market_rows)), {}),
    )
    monkeypatch.setattr(
        evaluator.strict_source,
        "_parse_funding_window",
        lambda *args, **kwargs: (pd.DataFrame(index=range(funding_rows)), {}),
    )

    report = evaluator.prepare_stage_source("train")

    assert report["full_parent_compressed_bytes_hashed"] is False
    assert report["parent_digest_deferred_until_selection"] is True
    assert manifest.exists()
    assert (stage_dir / "BTCUSDT_5m.csv.gz").exists()
    assert (stage_dir / "BTCUSDT_funding_marks.csv.gz").exists()


def test_stage_source_orphan_is_preserved_and_blocks_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage_dir = tmp_path / "train"
    stage_dir.mkdir()
    orphan = stage_dir / "BTCUSDT_5m.csv.gz"
    orphan.write_bytes(b"audit-evidence")
    monkeypatch.setitem(evaluator.STAGE_SOURCE_DIRS, "train", stage_dir)
    monkeypatch.setitem(
        evaluator.STAGE_SOURCE_MANIFESTS, "train", tmp_path / "absent.json"
    )
    monkeypatch.setattr(
        evaluator,
        "verify_evaluator_freeze",
        lambda: {"manifest_hash": "freeze"},
    )
    monkeypatch.setattr(
        evaluator, "_verified_prior_reports", lambda *args, **kwargs: []
    )
    with pytest.raises(FileExistsError, match="orphaned write-once source"):
        evaluator.prepare_stage_source("train")
    assert orphan.read_bytes() == b"audit-evidence"


def test_stage_source_manifest_cannot_redirect_frozen_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = evaluator._stage_source_spec("train")
    freeze = {
        "manifest_hash": "freeze",
        "execution_source_specs": {"train": spec},
        "legacy_container_contract": {
            "market": {"path": str(evaluator.LEGACY_MARKET), "sha256": "m"},
            "funding": {"path": str(evaluator.LEGACY_FUNDING), "sha256": "f"},
        },
    }
    payload = evaluator._seal(
        {
            "protocol_version": evaluator.SOURCE_PROTOCOL_VERSION,
            "candidate_family": evaluator.POLICY_ID,
            "stage": "train",
            "evaluator_freeze_manifest_hash": "freeze",
            "physical_window": spec["physical_window"],
            "physical_rows_limited_to_window": True,
            "exit_boundary_required": False,
            "strategy_outcomes_calculated": False,
            "official_manifest_hashes_verified": True,
            "full_parent_compressed_bytes_hashed": spec[
                "full_parent_compressed_bytes_hashed"
            ],
            "parent_digest_deferred_until_selection": spec[
                "parent_digest_deferred_until_selection"
            ],
            "post_stage_numeric_rows_parsed": 0,
            "parent_market": freeze["legacy_container_contract"]["market"],
            "parent_funding": freeze["legacy_container_contract"]["funding"],
            "market": {"path": "data/redirected.csv.gz", "sha256": "x", "rows": 1},
            "funding": {
                "path": str(
                    evaluator.STAGE_SOURCE_DIRS["train"]
                    / "BTCUSDT_funding_marks.csv.gz"
                ),
                "sha256": "y",
                "rows": 1,
            },
        }
    )
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setitem(evaluator.STAGE_SOURCE_MANIFESTS, "train", manifest)
    with pytest.raises(ValueError, match="market path changed"):
        evaluator._load_stage_source("train", freeze=freeze)


def test_execution_diagnostics_bind_exact_source_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "source.json"
    manifest.write_bytes(b"frozen-source-manifest")
    monkeypatch.setitem(evaluator.STAGE_SOURCE_MANIFESTS, "train", manifest)
    freeze = {"manifest_hash": "freeze"}
    contract = {
        "manifest_hash": "source-manifest",
        "market": {"path": "market.csv.gz", "sha256": "market-sha"},
        "funding": {"path": "funding.csv.gz", "sha256": "funding-sha"},
        "parent_market": {"sha256": "parent-market"},
        "parent_funding": {"sha256": "parent-funding"},
    }
    monkeypatch.setattr(evaluator, "verify_evaluator_freeze", lambda: freeze)
    monkeypatch.setattr(
        evaluator, "_verified_prior_reports", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        evaluator, "_load_stage_source", lambda *args, **kwargs: contract
    )
    monkeypatch.setattr(
        evaluator.strict_source,
        "_parse_market_window",
        lambda *args, **kwargs: (pd.DataFrame(), {"rows": 1}),
    )
    monkeypatch.setattr(
        evaluator.strict_source,
        "_parse_funding_window",
        lambda *args, **kwargs: (pd.DataFrame(), {"rows": 1}),
    )

    _, _, diagnostics = evaluator.load_execution_window("train")

    assert diagnostics["execution_source_manifest"] == {
        "path": str(manifest),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "manifest_hash": "source-manifest",
    }
    assert diagnostics["stage_source_paths"] == {
        "market": "market.csv.gz",
        "funding": "funding.csv.gz",
    }
    assert diagnostics["parent_contract"]["market"]["sha256"] == "parent-market"


def test_strict_simulation_daily_returns_match_ending_equity() -> None:
    market = _market()
    clock = pd.DataFrame([_clock(market)])
    start = evaluator._utc(evaluator._utc(market.iloc[0]["date"]).floor("D"))
    end = evaluator._utc(start + pd.Timedelta(days=365))
    metrics, daily = evaluator.simulate_strict(
        market,
        _funding(),
        clock,
        start=start,
        end=end,
        cost_rate_per_side=evaluator.FROZEN_CONFIG.base_cost_notional_per_side,
    )

    assert len(daily) == 365
    assert np.count_nonzero(daily) == 1
    assert np.isclose(daily.sum(), np.log(metrics["ending_equity"]), atol=1e-12)
    assert metrics["absolute_return_pct"] < 0.0
    assert metrics["strict_mdd_pct"] > 0.0


def test_strict_path_uses_favorable_then_adverse_intrabar_extremes() -> None:
    market = _market()
    market.loc[0, "high"] = 120.0
    market.loc[0, "low"] = 80.0
    clock = pd.DataFrame([_clock(market)])
    start = evaluator._utc(evaluator._utc(market.iloc[0]["date"]).floor("D"))
    end = evaluator._utc(start + pd.Timedelta(days=365))

    metrics, _ = evaluator.simulate_strict(
        market,
        _funding(),
        clock,
        start=start,
        end=end,
        cost_rate_per_side=evaluator.FROZEN_CONFIG.base_cost_notional_per_side,
    )

    assert metrics["strict_mdd_pct"] > 18.0


def test_strict_funding_drops_boundary_credits_but_keeps_debits() -> None:
    market = _market()
    clock = pd.DataFrame([_clock(market)])
    entry = evaluator._utc(clock.iloc[0]["entry_time"])
    exit_time = evaluator._utc(clock.iloc[0]["exit_time"])
    funding = pd.DataFrame(
        {
            "funding_time": [entry, entry + pd.Timedelta(hours=8), exit_time],
            "symbol": ["BTCUSDT"] * 3,
            "funding_rate": [0.001, -0.001, -0.001],
            "settlement_mark_price": [100.0] * 3,
        }
    )
    start = evaluator._utc(entry.floor("D"))
    end = evaluator._utc(start + pd.Timedelta(days=365))

    metrics, _ = evaluator.simulate_strict(
        market,
        funding,
        clock,
        start=start,
        end=end,
        cost_rate_per_side=evaluator.FROZEN_CONFIG.base_cost_notional_per_side,
    )
    trade = metrics["trade_details"][0]

    assert trade["visited_funding_events"] == 3
    assert trade["funding_events"] == 2
    assert trade["dropped_boundary_funding_credits"] == 1
    assert trade["funding_cash"] == pytest.approx(0.0, abs=1e-15)


def test_strict_simulation_fails_when_frozen_exit_open_is_missing() -> None:
    market = _market()
    clock = pd.DataFrame([_clock(market)])
    truncated = market.iloc[:-1].copy()
    start = evaluator._utc(evaluator._utc(market.iloc[0]["date"]).floor("D"))
    end = evaluator._utc(start + pd.Timedelta(days=365))
    with pytest.raises(ValueError, match="absent from the market grid"):
        evaluator.simulate_strict(
            truncated,
            _funding(),
            clock,
            start=start,
            end=end,
            cost_rate_per_side=evaluator.FROZEN_CONFIG.base_cost_notional_per_side,
        )


def test_romano_wolf_is_deterministic_stepdown_and_fail_closed() -> None:
    x = np.arange(70, dtype=np.float64)
    returns = {
        variant_id: 0.001 * np.sin(x / (7.0 + index)) + 0.00001 * index
        for index, variant_id in enumerate(evaluator.FAMILY_IDS)
    }
    returns[evaluator.FAMILY_IDS[1]] = returns[evaluator.FAMILY_IDS[0]].copy()
    constant_id = evaluator.FAMILY_IDS[2]
    returns[constant_id] = np.full(70, 0.0001)
    tested = evaluator.FAMILY_IDS[:3]
    first = evaluator.romano_wolf_stepdown(
        returns, tested, draws=250, block_days=7, seed=123, batch_draws=37
    )
    second = evaluator.romano_wolf_stepdown(
        returns, tested, draws=250, block_days=7, seed=123, batch_draws=37
    )

    assert first == second
    assert first["adjusted_p"][constant_id] == 1.0
    assert first["variance_positive"][constant_id] is False
    assert first["adjusted_p"][evaluator.FAMILY_IDS[3]] == 1.0
    ordered = first["ordered_tested_variant_ids"]
    values = [first["adjusted_p"][variant_id] for variant_id in ordered]
    assert values == sorted(values)
    assert (
        first["raw_stepdown_p"][evaluator.FAMILY_IDS[0]]
        == first["raw_stepdown_p"][evaluator.FAMILY_IDS[1]]
    )
    assert first["raw_stepdown_p"][evaluator.FAMILY_IDS[0]] == pytest.approx(
        54.0 / 251.0
    )
    unbatched = evaluator.romano_wolf_stepdown(
        returns, tested, draws=250, block_days=7, seed=123, batch_draws=250
    )
    assert {key: value for key, value in unbatched.items() if key != "batch_draws"} == {
        key: value for key, value in first.items() if key != "batch_draws"
    }


def test_gate_contract_requires_both_sides_halves_stress_delay_and_adjustment() -> None:
    headline = {
        "absolute_return_pct": 5.0,
        "cagr_to_strict_mdd": 3.1,
        "strict_mdd_pct": 14.9,
        "mean_gross_underlying_bp": 30.1,
    }
    checks = evaluator._gate_checks(
        base=headline,
        stress={**headline, "cagr_to_strict_mdd": 2.6},
        halves={"h1": headline, "h2": headline},
        controls={
            "fUSD_only": headline,
            "fBTC_only": headline,
            "extra_latency_one_bar": headline,
        },
        adjusted_p=0.10,
    )
    assert all(checks.values())
    failed = evaluator._gate_checks(
        base=headline,
        stress={**headline, "cagr_to_strict_mdd": 2.6},
        halves={"h1": headline, "h2": {**headline, "absolute_return_pct": 0.0}},
        controls={
            "fUSD_only": headline,
            "fBTC_only": {**headline, "absolute_return_pct": 0.0},
            "extra_latency_one_bar": headline,
        },
        adjusted_p=0.100001,
    )
    assert failed["each_contained_calendar_half_positive"] is False
    assert failed["fBTC_short_contribution_positive"] is False
    assert failed["romano_wolf_adjusted_p_at_most_10pct"] is False


def test_selection_advances_only_train_passing_variants() -> None:
    passing = [evaluator.FAMILY_IDS[1], evaluator.FAMILY_IDS[3]]
    active = evaluator._active_variants("selection", [{"passing_variants": passing}])
    assert active == (evaluator.FAMILY_IDS[1], evaluator.FAMILY_IDS[3])


def test_stage_result_pair_rolls_back_when_document_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "stage.json"
    document = tmp_path / "stage.md"
    monkeypatch.setitem(evaluator.STAGE_OUTPUTS, "train", output)
    monkeypatch.setitem(evaluator.STAGE_DOCS, "train", document)
    monkeypatch.setattr(evaluator, "_build_stage_report", lambda stage: {})
    monkeypatch.setattr(evaluator, "render_stage_doc", lambda report: "doc")
    real_write = evaluator._write_once_bytes
    calls = 0

    def fail_second(path: str | Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated document failure")
        real_write(path, payload)

    monkeypatch.setattr(evaluator, "_write_once_bytes", fail_second)
    with pytest.raises(OSError, match="simulated document failure"):
        evaluator.evaluate_stage("train")
    assert not output.exists()
    assert not document.exists()
