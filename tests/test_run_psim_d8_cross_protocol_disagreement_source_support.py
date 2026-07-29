from __future__ import annotations

from pathlib import Path

from training import (
    run_psim_d8_cross_protocol_disagreement_source_support as runner,
)


def _event(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_type": "UPDATE",
        "window_revision_count_bucket": "B1",
        "window_age_bucket": "B2",
        "update_gap_bucket": "B3",
        "dependency_delta_state": "UNCHANGED",
        "dependency_edge_delta_count_bucket": "B0",
        "line_change_count_bucket": "B4",
        "changed_section_count_bucket": "B1",
        "changed_sections": ["SPECIFICATION", "SECURITY"],
    }
    payload.update(overrides)
    return payload


def test_unit_disagreement_is_exact_and_bounded() -> None:
    same = {
        "ethereum": _event(),
        "bitcoin": _event(),
        "counterpart_state": "SAME_DAY_CARTESIAN",
        "memorization_excluded": False,
    }
    opposite = {
        "ethereum": _event(),
        "bitcoin": _event(
            event_type="CREATE",
            window_revision_count_bucket="B9",
            window_age_bucket="B9",
            update_gap_bucket="B9",
            dependency_delta_state="ADDED",
            dependency_edge_delta_count_bucket="B9",
            line_change_count_bucket="B9",
            changed_section_count_bucket="B9",
            changed_sections=["TESTS"],
        ),
        "counterpart_state": "TRAILING_90D",
        "memorization_excluded": False,
    }

    assert runner.unit_disagreement(same) == 0.0
    assert runner.unit_disagreement(opposite) == 1.0


def test_ineligible_units_are_ignored() -> None:
    assert runner.unit_disagreement(
        {
            "ethereum": "NO_COUNTERPART",
            "bitcoin": _event(),
            "counterpart_state": "NO_COUNTERPART",
            "memorization_excluded": False,
        }
    ) is None
    assert runner.unit_disagreement(
        {
            "ethereum": _event(),
            "bitcoin": _event(),
            "counterpart_state": "SAME_DAY_CARTESIAN",
            "memorization_excluded": True,
        }
    ) is None


def test_ewma_state_and_signal_direction_follow_preregistration() -> None:
    state = runner.EwmaState()
    for _ in range(30):
        runner.update_ewmas(state, 0.50)

    assert state.nonmissing_cards == 30
    assert runner.signal_for(state, slow_floor=0.35, gap=0.05) == "flat"
    state.fast = 0.70
    state.slow = 0.50
    assert runner.signal_for(state, slow_floor=0.35, gap=0.05) == "short"
    state.fast = 0.30
    assert runner.signal_for(state, slow_floor=0.35, gap=0.05) == "long"


def test_source_support_does_not_open_market_or_funding(
    monkeypatch,
) -> None:
    opened: list[Path] = []
    real_read = Path.read_bytes

    def recording_read(path: Path) -> bytes:
        opened.append(path)
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", recording_read)
    runner.build_source_support()

    assert runner.prereg.MARKET not in opened
    assert runner.prereg.FUNDING not in opened
