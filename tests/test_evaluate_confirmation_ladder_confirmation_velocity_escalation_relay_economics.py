import gzip
from pathlib import Path

from training import evaluate_confirmation_ladder_confirmation_velocity_escalation_relay_economics as economics


def test_load_clock_allow_empty_preserves_zero_row_control(tmp_path: Path):
    path = tmp_path / "empty.csv.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("entry_time,exit_time,side\n")
    clock = economics.load_clock_allow_empty(
        path,
        "train",
        economics.legacy._utc("2023-07-01T00:00:00Z"),
        economics.legacy._utc("2024-01-01T00:00:00Z"),
    )
    assert clock.empty
    assert list(clock.columns) == ["entry_time", "exit_time", "side"]


def test_economic_gate_contract_is_strict():
    assert economics.LEVERAGE == 0.5
    assert economics.BASE_COST == 0.0006
    assert economics.STRESS_COST == 0.001
    assert tuple(economics.STAGES) == ("train", "test", "eval", "final")
