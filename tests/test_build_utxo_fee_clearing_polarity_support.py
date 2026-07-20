from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import build_utxo_fee_clearing_polarity_support as support
from training import preregister_utxo_fee_clearing_polarity as prereg


def _hash(height: int) -> str:
    return f"{height:064x}"


def _source_days(days: int, *, start: str = "2020-06-01", blocks_per_day: int = 72) -> pd.DataFrame:
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    rows = []
    height0 = 700_000
    for day in range(days):
        # The first 120 days are bland prior history. Later days cycle through
        # high-fee/high-polarity, high-fee/low-polarity, and neutral days.
        bucket = day % 6
        fee_per_block = 1_000
        extra_outputs = 0
        if day >= 120 and bucket in {0, 3}:
            fee_per_block = 10_000
            extra_outputs = 4_000 if bucket == 0 else -4_000
        elif day >= 120 and bucket in {1, 4}:
            fee_per_block = 100
            extra_outputs = 4_000 if bucket == 1 else -4_000
        for block in range(blocks_per_day):
            height = height0 + day * blocks_per_day + block
            total_inputs = 10_000
            total_outputs = total_inputs + extra_outputs
            rows.append(
                {
                    "height": height,
                    "id": _hash(height),
                    "previousblockhash": _hash(height - 1),
                    "timestamp": start_ts + day * 86_400 + block * 600,
                    "mediantime": start_ts + day * 86_400 + block * 600 - 600,
                    "tx_count": 1_000,
                    "size": 1_000,
                    "weight": 2_000,
                    "total_fees": fee_per_block,
                    "total_inputs": total_inputs,
                    "total_outputs": total_outputs,
                    "utxo_set_change": total_outputs - total_inputs,
                }
            )
    return pd.DataFrame(rows, columns=support.SOURCE_COLUMNS)


def _prereg_for_source(source_csv: Path) -> dict[str, object]:
    source_output = {
        "path": str(source_csv),
        "sha256": support.sha256_file(source_csv),
        "bytes": source_csv.stat().st_size,
        "columns": support.SOURCE_COLUMNS,
    }
    source_manifest = {
        "path": "synthetic_source_manifest.json",
        "sha256": "0" * 64,
        "manifest_hash": "1" * 64,
        "protocol_version": prereg.SOURCE_PROTOCOL_VERSION,
        "source_output": source_output,
    }
    core = {
        "protocol_version": prereg.PROTOCOL_VERSION,
        "policy_id": support.POLICY_ID,
        "config": {"preregistration_output": "synthetic.json", "source_manifest": "synthetic_source_manifest.json"},
        "source_manifest": source_manifest,
        "policy": prereg.policy(),
        "policy_hash": support.canonical_hash(prereg.policy()),
        "outcomes_opened": False,
        "outcome_boundary": prereg.PREREGISTRATION_OUTCOME_BOUNDARY,
        "research_sequence": {"train_first": "2021-2022", "selection_second": "2023 only after exact train pass", "sealed": "2024+"},
        "preregistration_source": {
            "path": str(prereg.PREREGISTRATION_SOURCE),
            "sha256": support.sha256_file(prereg.PREREGISTRATION_SOURCE),
        },
    }
    return {**core, "manifest_hash": support.canonical_hash(core)}


def _write_prereg(tmp_path: Path, frame: pd.DataFrame) -> tuple[Path, Path]:
    source_csv = tmp_path / "source.csv.gz"
    frame.to_csv(source_csv, index=False)
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(_prereg_for_source(source_csv), sort_keys=True), encoding="utf-8")
    return prereg_path, source_csv


def _allow_synthetic_prereg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(support, "validate_frozen_preregistration", support.validate_preregistration)


def test_midrank_uses_ties_and_excludes_current() -> None:
    assert support.strict_prior_midrank(2.0, [1.0, 2.0, 2.0, 3.0]) == pytest.approx(0.5)
    assert support.strict_prior_midrank(4.0, [1.0, 2.0, 3.0]) == 1.0
    daily = pd.DataFrame(
        {
            "source_day": pd.date_range("2021-01-01", periods=4, tz="UTC"),
            "valid_source_day": [True] * 4,
            "fee_burden": [1.0, 2.0, 2.0, 100.0],
            "utxo_polarity": [1.0, 2.0, 2.0, -100.0],
        }
    )
    ranked = support.attach_strict_prior_ranks(daily, lookback=3, minimum_prior=3)
    assert ranked.loc[3, "fee_rank"] == 1.0
    assert ranked.loc[3, "polarity_rank"] == 0.0
    changed_current = daily.copy()
    changed_current.loc[3, "fee_burden"] = -100.0
    changed = support.attach_strict_prior_ranks(changed_current, lookback=3, minimum_prior=3)
    assert changed.loc[3, "fee_rank"] == 0.0
    assert ranked.loc[2, "fee_rank"] != ranked.loc[2, "fee_rank"]  # current excluded until 3 priors exist


def test_source_validation_rechecks_contiguous_chain_and_utxo_identity() -> None:
    frame = _source_days(3)
    support.validate_source_frame(frame)
    gap = frame.drop(index=10).reset_index(drop=True)
    with pytest.raises(RuntimeError, match="contiguous"):
        support.validate_source_frame(gap)
    broken_chain = frame.copy()
    broken_chain.loc[10, "previousblockhash"] = "bad"
    with pytest.raises(RuntimeError, match="hash-chain"):
        support.validate_source_frame(broken_chain)
    broken_utxo = frame.copy()
    broken_utxo.loc[0, "utxo_set_change"] += 1
    with pytest.raises(RuntimeError, match="UTXO identity"):
        support.validate_source_frame(broken_utxo)


def test_daily_features_require_d2_schedule_six_successors_and_minimum_blocks() -> None:
    frame = _source_days(4)
    daily = support.build_daily_features(frame)
    first = daily.iloc[0]
    assert first["block_count"] == 72
    assert first["valid_source_day"] is True or bool(first["valid_source_day"])
    assert first["successor_end_height"] == frame.iloc[77]["height"]
    assert first["available_time"] == pd.Timestamp("2020-06-03T00:00:00Z")
    assert first["entry_time"] == pd.Timestamp("2020-06-03T00:05:00Z")
    assert first["exit_time"] == pd.Timestamp("2020-06-04T00:05:00Z")
    assert not bool(daily.iloc[-1]["valid_source_day"]), "final included day lacks six successors"

    low_block = _source_days(3).drop(index=range(72, 82)).reset_index(drop=True)
    with pytest.raises(RuntimeError, match="contiguous"):
        support.build_daily_features(low_block)

    low_block_day = _source_days(3)
    # Move ten blocks from day 1 into day 2 without breaking contiguous height chain.
    low_block_day.loc[72:81, "timestamp"] += 86_400
    low_daily = support.build_daily_features(low_block_day)
    assert not bool(low_daily.loc[1, "valid_source_day"])


def test_missing_utc_day_fails_closed() -> None:
    frame = _source_days(3)
    frame.loc[72:143, "timestamp"] += 86_400
    with pytest.raises(RuntimeError, match="missing UTC day"):
        support.build_daily_features(frame)


def test_primary_clock_thresholds_and_nonoverlap_allow_entry_equal_prior_exit() -> None:
    rows = []
    for day, entry, fee, pol in [
        ("2021-01-01", "2021-01-03T00:05:00Z", 0.80, 0.80),
        ("2021-01-02", "2021-01-04T00:05:00Z", 0.80, 0.20),
        ("2021-01-03", "2021-01-04T00:04:00Z", 0.80, 0.80),
        ("2021-01-04", "2021-01-05T00:05:00Z", 0.74, 0.80),
    ]:
        rows.append(
            {
                "source_day": pd.Timestamp(day, tz="UTC"),
                "available_time": pd.Timestamp(entry, tz="UTC") - pd.Timedelta(minutes=5),
                "entry_time": pd.Timestamp(entry, tz="UTC"),
                "exit_time": pd.Timestamp(entry, tz="UTC") + pd.Timedelta(days=1),
                "valid_source_day": True,
                "rank_ready": True,
                "source_start_height": 1,
                "source_end_height": 2,
                "successor_end_height": 8,
                "block_count": 72,
                "edges": 1,
                "fees": 1,
                "fee_burden": 0.0,
                "utxo_polarity": 0.0,
                "fee_rank": fee,
                "polarity_rank": pol,
            }
        )
    primary = support.build_primary_clock(pd.DataFrame(rows))
    assert primary["source_day"].dt.strftime("%Y-%m-%d").tolist() == ["2021-01-01", "2021-01-02"]
    assert primary["side"].tolist() == [1, -1]


def test_controls_preserve_counts_and_random_is_deterministic() -> None:
    daily = support.attach_strict_prior_ranks(support.build_daily_features(_source_days(500, start="2020-07-01")))
    primary = support.build_primary_clock(daily)
    controls_a = support.build_control_clocks(daily, primary)
    controls_b = support.build_control_clocks(daily, primary)
    pd.testing.assert_frame_equal(controls_a, controls_b)
    counts = controls_a["clock"].value_counts().to_dict()
    for same_clock in ["direction_flip", "constant_long_same_clock", "constant_short_same_clock", "one_bar_delayed_entry"]:
        assert counts[same_clock] == len(primary)
    random_clock = controls_a[controls_a["clock"] == "year_side_stratified_random_clock"]
    assert len(random_clock) == len(primary)
    assert random_clock.groupby([random_clock["entry_time"].dt.year, "side"]).size().to_dict() == primary.groupby([primary["entry_time"].dt.year, "side"]).size().to_dict()
    delayed = controls_a[controls_a["clock"] == "one_bar_delayed_entry"].reset_index(drop=True)
    assert (delayed["entry_time"].reset_index(drop=True) == primary["entry_time"].reset_index(drop=True) + pd.Timedelta(minutes=5)).all()


def _clock_for_support(pass_gate: bool = True) -> pd.DataFrame:
    entry = list(pd.date_range("2021-01-01T00:05:00Z", periods=72, freq="10D"))
    entry += list(pd.date_range("2023-01-01T00:05:00Z", periods=36, freq="10D"))
    side = [1, -1] * (len(entry) // 2)
    if not pass_gate:
        side = [1] * len(entry)
    return pd.DataFrame({"entry_time": pd.to_datetime(entry, utc=True), "side": side})


def test_support_gate_pass_fail() -> None:
    daily = pd.DataFrame(
        {
            "source_day": [pd.Timestamp("2021-01-01T00:00:00Z")],
            "valid_source_day": [True],
            "block_count": [72],
            "edges": [1],
            "fees": [1],
        }
    )
    passing = support.support_gate_summary(_clock_for_support(True), daily)
    assert passing["passed"] is True
    failing = support.support_gate_summary(_clock_for_support(False), daily)
    assert failing["passed"] is False
    assert failing["checks"]["train_side_share_25_75"] is False

    missing_day = pd.concat(
        [
            daily,
            daily.assign(source_day=pd.Timestamp("2021-01-03T00:00:00Z")),
        ],
        ignore_index=True,
    )
    continuity = support.support_gate_summary(_clock_for_support(True), missing_day)
    assert continuity["passed"] is False
    assert continuity["checks"]["all_usable_days_no_missing_utc_day"] is False


def test_artifact_hash_deterministic_rerun_and_no_outcome_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _source_days(520, start="2020-07-01")
    prereg_path, _source_csv = _write_prereg(tmp_path, frame)
    _allow_synthetic_prereg(monkeypatch)
    cfg = support.Config(
        preregistration=str(prereg_path),
        output=str(tmp_path / "support.json"),
        primary_clock=str(tmp_path / "primary.csv"),
        control_clocks=str(tmp_path / "controls.csv.gz"),
    )
    first = support.build_support_artifacts(cfg)
    second = support.build_support_artifacts(cfg)
    assert first == second
    core = {key: value for key, value in first.items() if key != "manifest_hash"}
    assert first["manifest_hash"] == support.canonical_hash(core)
    assert first["outcome_boundary"] == support.OUTCOME_BOUNDARY
    assert first["support_builder"] == {
        "path": str(support.SUPPORT_BUILDER),
        "sha256": support.sha256_file(support.SUPPORT_BUILDER),
    }
    text = json.dumps(first).lower()
    assert "pnl" not in text
    assert first["market_rows_loaded"] == 0
    assert first["funding_rows_loaded"] == 0
    assert first["return_rows_loaded"] == 0
    assert first["market_values_read"] == 0
    assert first["funding_values_read"] == 0
    assert first["profit_loss_fields"] == 0
    assert Path(cfg.primary_clock).read_text(encoding="utf-8").splitlines()[0].split(",") == support.CLOCK_COLUMNS
    with gzip.open(cfg.control_clocks, "rt", encoding="utf-8") as handle:
        assert handle.readline().strip().split(",") == support.CLOCK_COLUMNS


def test_prereg_and_source_hash_fail_closed(tmp_path: Path) -> None:
    frame = _source_days(3)
    prereg_path, source_csv = _write_prereg(tmp_path, frame)
    artifact = json.loads(prereg_path.read_text(encoding="utf-8"))
    artifact["policy_id"] = "DRIFT"
    prereg_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest hash"):
        support.validate_preregistration(prereg_path)

    prereg_path, source_csv = _write_prereg(tmp_path, frame)
    with open(source_csv, "ab") as handle:
        handle.write(b"drift")
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        support.load_source_frame(support.validate_preregistration(prereg_path))


def test_frozen_preregistration_rejects_self_consistent_substitution(tmp_path: Path) -> None:
    prereg_path, _source_csv = _write_prereg(tmp_path, _source_days(3))
    support.validate_preregistration(prereg_path)
    with pytest.raises(RuntimeError, match="path differs"):
        support.validate_frozen_preregistration(prereg_path)


def test_committed_frozen_preregistration_validates() -> None:
    artifact = support.validate_frozen_preregistration(support.DEFAULT_PREREGISTRATION)
    assert artifact["manifest_hash"] == support.EXPECTED_PREREGISTRATION_MANIFEST_HASH


def test_output_paths_fail_closed_on_aliases_and_wrong_extensions(tmp_path: Path) -> None:
    prereg_path, source_csv = _write_prereg(tmp_path, _source_days(3))
    artifact = support.validate_preregistration(prereg_path)
    valid = support.Config(
        preregistration=str(prereg_path),
        output=str(tmp_path / "support.json"),
        primary_clock=str(tmp_path / "primary.csv"),
        control_clocks=str(tmp_path / "controls.csv.gz"),
    )
    support._validate_config(valid, artifact)

    source_alias = tmp_path / "source_alias.csv"
    source_alias.symlink_to(source_csv)
    alias_source = support.Config(
        preregistration=str(prereg_path),
        output=str(tmp_path / "support.json"),
        primary_clock=str(source_alias),
        control_clocks=str(tmp_path / "controls.csv.gz"),
    )
    with pytest.raises(ValueError, match="frozen input"):
        support._validate_config(alias_source, artifact)

    duplicate = support.Config(
        preregistration=str(prereg_path),
        output=str(tmp_path / "same.json"),
        primary_clock=str(tmp_path / "same.json"),
        control_clocks=str(tmp_path / "controls.csv.gz"),
    )
    with pytest.raises(ValueError, match="primary clock must be a CSV"):
        support._validate_config(duplicate, artifact)

    same_csv = support.Config(
        preregistration=str(prereg_path),
        output=str(tmp_path / "support.json"),
        primary_clock=str(tmp_path / "same.csv"),
        control_clocks=str(tmp_path / "same.csv"),
    )
    with pytest.raises(ValueError, match="control clocks must be"):
        support._validate_config(same_csv, artifact)
