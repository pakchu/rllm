from __future__ import annotations

import json
from pathlib import Path

import pytest

from training import build_sofr_rate_dislocation_support as support
from training import sofr_rate_dislocation_clock as clock


def test_real_support_replays_all_counts_sides_and_causality(tmp_path: Path) -> None:
    result = support.build(tmp_path / "support.json")
    assert result["support_replay_passed"] is True
    assert result["advance_to_stage1_outcomes"] is True
    assert result["outcomes_opened"] is False
    assert result["outcome_sources_opened"] == []
    assert result["full_source_events"] == 158
    assert result["windows"]["train"]["count"] == 48
    assert result["windows"]["train"]["long"] == 31
    assert result["windows"]["train"]["short"] == 17
    assert result["windows"]["2023"]["count"] == 40
    assert result["windows"]["2023"]["long"] == 20
    assert result["windows"]["2023"]["short"] == 20
    assert all(result["checks"].values())
    assert all(value is None for value in result["performance_statistics"].values())


def test_support_hashes_no_market_or_funding_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[str] = []
    real_sha = support._sha256

    def tracking_sha(path: str | Path) -> str:
        seen.append(str(path))
        return real_sha(path)

    monkeypatch.setattr(support, "_sha256", tracking_sha)
    result = support.build(tmp_path / "support.json")
    assert result["outcome_sources_opened"] == []
    assert not any("kline_reference" in path for path in seen)
    assert not any("funding_marks" in path for path in seen)


def test_replay_rejects_any_ledger_row_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = support.load_preregistration()
    real_ledger = clock.read_event_ledger(
        registration["support_verification"]["clock_ledger"]
    )
    mutated = list(real_ledger)
    mutated[0] = clock.Event(
        **{
            **mutated[0].__dict__,
            "delta_bp": mutated[0].delta_bp + 1,
        }
    )
    monkeypatch.setattr(clock, "read_event_ledger", lambda path: mutated)
    with pytest.raises(ValueError, match="differs from frozen ledger"):
        support.replay_clock(registration)


def test_causality_rejects_early_entry() -> None:
    event = clock.read_event_ledger()[0]
    early = clock.Event(
        **{
            **event.__dict__,
            "entry_time": event.sofr_available_at_utc,
        }
    )
    checks = support._causality_checks([early])
    assert checks["entry_exactly_one_bar_after_rate_availability"] is False


def test_written_support_is_byte_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "support.json"
    first = support.build(output)
    first_bytes = output.read_bytes()
    second = support.build(output)
    assert first == second
    assert output.read_bytes() == first_bytes
    assert json.loads(first_bytes) == first
