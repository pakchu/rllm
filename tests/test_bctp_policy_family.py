from __future__ import annotations

from collections import OrderedDict
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import bctp_cheap_policies as cheap
from training import bctp_policy_family as family
from training import freeze_block_clearing_target_position_evaluator as freeze
from training import preregister_block_clearing_relational_topology as bcrt
from training import preregister_block_clearing_target_position_mdp as prereg


def _states(year: int, count: int = 3) -> pd.DataFrame:
    rows = []
    for index in range(count):
        row = {
            "sequence_id": f"sequence-{year}-{index}",
            "entry_time": pd.Timestamp(
                f"{year}-01-01T00:0{index * 5}:00Z"
            ),
            "source_signal_id_m2": f"m2-{index}",
            "source_signal_id_m1": f"m1-{index}",
            "source_signal_id_s0": f"s0-{index}",
            "source_signature": f"signature-{index}",
        }
        for snapshot in ("s_minus_2", "s_minus_1", "s_0"):
            for token_index, (token, values) in enumerate(
                bcrt.TOKEN_SCHEMA
            ):
                row[f"{snapshot}__{token}"] = values[
                    (index + token_index) % len(values)
                ]
        rows.append(row)
    return pd.DataFrame(rows, columns=prereg.SOURCE_SEQUENCE_COLUMNS)


def _transition_arrays(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rewards = np.zeros((count, 3, 3), dtype=float)
    rewards[:, :, 2] = 1.0
    reachable = np.ones((count, 3), dtype=bool)
    reachable[0] = False
    reachable[0, cheap.POSITIONS.index("POSITION_FLAT")] = True
    rewards[~reachable] = np.nan
    terminal = np.zeros(count, dtype=bool)
    terminal[-1] = True
    return rewards, terminal, reachable


def _clock() -> pd.DataFrame:
    rows = []
    for index, entry in enumerate(
        (
            pd.Timestamp("2021-01-02T00:00:00Z"),
            pd.Timestamp("2021-01-03T00:00:00Z"),
        )
    ):
        row = {
            "signal_id": f"clock-{index}",
            "bucket_start": (entry - pd.Timedelta(days=2)).isoformat(),
            "signal_available_time": (
                entry - pd.Timedelta(minutes=5)
            ).isoformat(),
            "entry_time": entry,
            "exit_time": entry + pd.Timedelta(hours=6),
        }
        for token, values in bcrt.TOKEN_SCHEMA:
            row[token] = values[0]
        rows.append(row)
    return pd.DataFrame(rows, columns=family.BCRT_CLOCK_COLUMNS)


def test_fit_family_is_exactly_frozen_and_counts_estimators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = _states(2020)
    rewards, terminal, reachable = _transition_arrays(len(states))

    def fake_fit(*args, **kwargs):
        return cheap.constant_policy("TARGET_LONG")

    monkeypatch.setattr(cheap, "fit_fitted_q", fake_fit)
    monkeypatch.setattr(
        cheap,
        "action_code_permutation_policy",
        fake_fit,
    )
    fitted = family.fit_family(
        states,
        rewards,
        terminal,
        reachable,
    )
    assert isinstance(fitted.policies, OrderedDict)
    assert tuple(fitted.policies) == freeze.FAMILY_IDS
    assert fitted.fitted_estimators == 21
    assert fitted.memory_tables_fit == 1
    assert fitted.state_rows == len(states)


def test_transfer_schedules_include_bcrt_holds_and_exact_delays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = _states(2020)
    rewards, terminal, reachable = _transition_arrays(len(states))

    def fake_fit(*args, **kwargs):
        return cheap.constant_policy("TARGET_LONG")

    monkeypatch.setattr(cheap, "fit_fitted_q", fake_fit)
    monkeypatch.setattr(
        cheap,
        "action_code_permutation_policy",
        fake_fit,
    )
    fitted = family.fit_family(
        states,
        rewards,
        terminal,
        reachable,
    )
    target_states = _states(2021)
    base, delayed = family.build_transfer_schedules(
        fitted,
        target_states,
        stage="2021",
        bcrt_clock=_clock(),
    )
    assert tuple(base) == freeze.FAMILY_IDS
    assert tuple(delayed) == family.PROMOTABLE_PRIMARY_IDS
    assert len(base["bcrt_exact_six_hour_always_long"]) == 4
    assert base["bcrt_exact_six_hour_always_long"]["target"].tolist() == [
        "TARGET_LONG",
        "TARGET_FLAT",
        "TARGET_LONG",
        "TARGET_FLAT",
    ]
    primary = base["categorical_linear_fqi"]
    shifted = delayed["categorical_linear_fqi"]
    delta = pd.to_datetime(shifted["entry_time"], utc=True) - pd.to_datetime(
        primary["entry_time"],
        utc=True,
    )
    assert delta.eq(pd.Timedelta(minutes=5)).all()
    assert shifted["sequence_id"].str.endswith(":delay_5m").all()


def test_synthetic_clock_loader_and_integrity_rejection(tmp_path: Path) -> None:
    frame = _clock()
    path = tmp_path / "clock.csv.gz"
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as compressed:
            compressed.write(
                frame.to_csv(index=False, lineterminator="\n").encode()
            )
    loaded = family.load_bcrt_clock(
        path,
        allow_synthetic_clock=True,
    )
    assert len(loaded) == 2
    overlapping = frame.copy()
    overlapping.loc[1, "entry_time"] = pd.Timestamp(
        "2021-01-02T01:00:00Z"
    )
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            mtime=0,
        ) as compressed:
            compressed.write(
                overlapping.to_csv(
                    index=False,
                    lineterminator="\n",
                ).encode()
            )
    with pytest.raises(ValueError, match="integrity"):
        family.load_bcrt_clock(
            path,
            allow_synthetic_clock=True,
        )


def test_adjacent_bcrt_reservations_coalesce_exit_and_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = _states(2020)
    rewards, terminal, reachable = _transition_arrays(len(states))

    def fake_fit(*args, **kwargs):
        return cheap.constant_policy("TARGET_LONG")

    monkeypatch.setattr(cheap, "fit_fitted_q", fake_fit)
    monkeypatch.setattr(
        cheap,
        "action_code_permutation_policy",
        fake_fit,
    )
    fitted = family.fit_family(
        states,
        rewards,
        terminal,
        reachable,
    )
    clock = _clock()
    clock.loc[1, "entry_time"] = clock.loc[0, "exit_time"]
    clock.loc[1, "exit_time"] = (
        clock.loc[1, "entry_time"] + pd.Timedelta(hours=6)
    )
    base, _ = family.build_transfer_schedules(
        fitted,
        _states(2021),
        stage="2021",
        bcrt_clock=clock,
    )
    comparator = base["bcrt_exact_six_hour_always_long"]
    assert comparator["entry_time"].is_unique
    assert comparator["target"].tolist() == [
        "TARGET_LONG",
        "TARGET_LONG",
        "TARGET_FLAT",
    ]
