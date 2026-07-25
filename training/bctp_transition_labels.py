"""Construct frozen full-information BCTP transition rewards."""
from __future__ import annotations

from collections.abc import Mapping
import gzip
import hashlib
import io
import os
from pathlib import Path
import tempfile
from typing import Any, cast

import numpy as np
import pandas as pd

from training import bctp_strict_economics as economics
from training import freeze_block_clearing_target_position_evaluator as freeze
from training import preregister_block_clearing_target_position_mdp as prereg


SOURCE_COLUMNS = tuple(prereg.SOURCE_SEQUENCE_COLUMNS)
TOKEN_COLUMNS = tuple(prereg.SOURCE_TOKEN_COLUMNS)
POSITION_ORDER = tuple(freeze.POSITIONS)
ACTION_ORDER = tuple(freeze.MODEL_ACTIONS)
POSITION_TARGET = {
    "POSITION_SHORT": -0.5,
    "POSITION_FLAT": 0.0,
    "POSITION_LONG": 0.5,
}
LEDGER_COLUMNS = (
    "sequence_id",
    "entry_time",
    "current_position",
    "action_name",
    "action_target",
    "executed_target",
    "reachable",
    "terminal",
    "reward",
    "multiplier",
    "held_path_downside_fraction",
    "changed_notional_fraction",
    "entry_cost",
    "terminal_cost",
    "funding_cash",
    "bars_held",
)


def _utc(value: Any) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise ValueError("BCTP transition timestamp must be timezone aware")
    return parsed.tz_convert("UTC")


def load_source_states(
    path: str | Path = freeze.SEQUENCES,
) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip", dtype=str)
    if tuple(frame.columns) != SOURCE_COLUMNS:
        raise ValueError("BCTP transition source schema changed")
    frame["entry_time"] = pd.to_datetime(
        frame["entry_time"],
        utc=True,
        errors="raise",
    )
    if (
        not frame["sequence_id"].is_unique
        or not frame["entry_time"].is_monotonic_increasing
        or frame["entry_time"].duplicated().any()
    ):
        raise ValueError("BCTP transition source identity or order changed")
    for column in TOKEN_COLUMNS:
        if frame[column].isna().any() or frame[column].str.len().eq(0).any():
            raise ValueError(f"BCTP transition source token missing: {column}")
    return frame


def stage_source_states(
    source: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    start = _utc(start)
    end = _utc(end)
    terminal = end - pd.Timedelta(minutes=5)
    if end <= start:
        raise ValueError("BCTP transition stage window is invalid")
    selected = cast(
        pd.DataFrame,
        source.loc[
            source["entry_time"].ge(start)
            & source["entry_time"].lt(terminal)
        ].copy(),
    )
    if len(selected) < 2:
        raise ValueError("BCTP transition stage has too few source states")
    return selected.reset_index(drop=True)


def _metric(result: Mapping[str, Any], key: str) -> float:
    value = float(result[key])
    if not np.isfinite(value):
        raise ValueError(f"BCTP transition metric is non-finite: {key}")
    return value


def _timestamp_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str:
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    raise ValueError(f"BCTP interval source misses timestamp column {aliases}")


def _indexed_source(
    frame: pd.DataFrame,
    *,
    aliases: tuple[str, ...],
    kind: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        if frame.index.tz is None:
            raise ValueError(f"BCTP {kind} index must be timezone aware")
        times = frame.index.tz_convert("UTC")
    else:
        column = _timestamp_column(frame, aliases)
        raw = frame[column].tolist()
        parsed = [_utc(value) for value in raw]
        times = pd.DatetimeIndex(parsed)
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError(f"BCTP {kind} timestamps changed")
    indexed = frame.copy()
    indexed.index = times
    return indexed


def _interval_frames(
    market: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    terminal: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    market_slice = market.loc[start:end]
    expected_rows = int((end - start) / pd.Timedelta(minutes=5)) + 1
    if len(market_slice) != expected_rows:
        raise ValueError("BCTP interval market boundary coverage changed")
    if funding.empty:
        funding_slice = funding.copy()
    elif terminal:
        funding_slice = funding.loc[
            (funding.index >= start) & (funding.index <= end)
        ]
    else:
        funding_slice = funding.loc[
            (funding.index >= start) & (funding.index < end)
        ]
    return market_slice, funding_slice


def build_reward_tensor(
    states: pd.DataFrame,
    market: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_rate: float = freeze.AccountingConfig().base_cost_rate,
) -> dict[str, Any]:
    """Build all reachable position/action rewards without a behavior policy."""

    start = _utc(start)
    end = _utc(end)
    terminal_flat_time = end - pd.Timedelta(minutes=5)
    if (
        tuple(states.columns) != SOURCE_COLUMNS
        or len(states) < 2
        or not states["entry_time"].is_monotonic_increasing
    ):
        raise ValueError("BCTP transition states changed")
    entries = [
        _utc(value)
        for value in cast(pd.Series, states["entry_time"])
    ]
    if entries[0] < start or entries[-1] >= terminal_flat_time:
        raise ValueError("BCTP transition states escape stage")

    count = len(states)
    rewards = np.full(
        (count, len(POSITION_ORDER), len(ACTION_ORDER)),
        np.nan,
        dtype=np.float64,
    )
    reachable = np.ones(
        (count, len(POSITION_ORDER)),
        dtype=bool,
    )
    reachable[0, :] = False
    reachable[0, POSITION_ORDER.index("POSITION_FLAT")] = True
    terminal = np.zeros(count, dtype=bool)
    diagnostics: list[dict[str, Any]] = []
    indexed_market = _indexed_source(
        market,
        aliases=("timestamp", "open_time", "time", "date"),
        kind="market",
    )
    indexed_funding = _indexed_source(
        funding,
        aliases=(
            "timestamp",
            "funding_time",
            "funding_time_utc",
            "time",
            "date",
        ),
        kind="funding",
    )

    for state_index, entry_time in enumerate(entries):
        next_time = (
            entries[state_index + 1]
            if state_index + 1 < count
            else terminal_flat_time
        )
        is_terminal = next_time >= terminal_flat_time
        interval_end = min(next_time, terminal_flat_time)
        terminal[state_index] = is_terminal
        interval_market, interval_funding = _interval_frames(
            indexed_market,
            indexed_funding,
            start=entry_time,
            end=interval_end,
            terminal=is_terminal,
        )
        position_accounts: dict[str, tuple[float, float]] = {
            "POSITION_FLAT": (1.0, 0.0)
        }
        if state_index > 0:
            prior_market, prior_funding = _interval_frames(
                indexed_market,
                indexed_funding,
                start=entries[state_index - 1],
                end=entry_time,
                terminal=False,
            )
            for prior_position in (
                "POSITION_SHORT",
                "POSITION_LONG",
            ):
                prior = economics.simulate_counterfactual_interval(
                    prior_market,
                    prior_funding,
                    start=entries[state_index - 1],
                    end=entry_time,
                    pre_equity=1.0,
                    old_quantity=0.0,
                    target=POSITION_TARGET[prior_position],
                    cost_rate=cost_rate,
                    terminal_flatten=False,
                )
                position_accounts[prior_position] = (
                    _metric(prior, "ending_equity"),
                    _metric(prior, "ending_quantity"),
                )
        for position_index, position_name in enumerate(POSITION_ORDER):
            if not reachable[state_index, position_index]:
                continue
            old_target = POSITION_TARGET[position_name]
            pre_equity, old_quantity = position_accounts[position_name]
            for action_index, action_target in enumerate(ACTION_ORDER):
                result = economics.simulate_counterfactual_interval(
                    interval_market,
                    interval_funding,
                    start=entry_time,
                    end=interval_end,
                    pre_equity=pre_equity,
                    old_quantity=old_quantity,
                    target=action_target,
                    cost_rate=cost_rate,
                    terminal_flatten=is_terminal,
                )
                multiplier = _metric(result, "multiplier")
                downside = _metric(
                    result,
                    "held_path_downside_fraction",
                )
                executed_target = _metric(result, "new_target")
                reward = freeze.transition_utility(
                    multiplier,
                    downside,
                    old_target,
                    executed_target,
                )
                rewards[
                    state_index,
                    position_index,
                    action_index,
                ] = reward
                diagnostics.append(
                    {
                        "sequence_id": str(
                            states.iloc[state_index]["sequence_id"]
                        ),
                        "entry_time": entry_time.isoformat(),
                        "current_position": position_name,
                        "action_name": freeze.MODEL_ACTION_NAMES[
                            action_index
                        ],
                        "action_target": float(action_target),
                        "executed_target": executed_target,
                        "reachable": True,
                        "terminal": bool(is_terminal),
                        "reward": float(reward),
                        "multiplier": multiplier,
                        "held_path_downside_fraction": downside,
                        "changed_notional_fraction": _metric(
                            result,
                            "changed_notional_fraction",
                        ),
                        "entry_cost": _metric(result, "entry_cost"),
                        "terminal_cost": _metric(
                            result,
                            "terminal_cost",
                        ),
                        "funding_cash": _metric(result, "funding_cash"),
                        "bars_held": int(result["bars_held"]),
                    }
                )

    if not np.isfinite(rewards[reachable]).all():
        raise RuntimeError("BCTP reachable reward tensor is incomplete")
    if np.isfinite(rewards[~reachable]).any():
        raise RuntimeError("BCTP unreachable reward tensor was populated")
    if not terminal[-1] or terminal[:-1].any():
        raise RuntimeError("BCTP transition terminal mask changed")
    return {
        "states": states.copy(),
        "reward_tensor": rewards,
        "reachable_mask": reachable,
        "terminal": terminal,
        "ledger": pd.DataFrame(
            diagnostics,
            columns=pd.Index(LEDGER_COLUMNS),
        ),
        "position_order": POSITION_ORDER,
        "action_order": ACTION_ORDER,
        "stage_start": start,
        "stage_end": end,
        "terminal_flat_time": terminal_flat_time,
    }


def ledger_frame_hash(frame: pd.DataFrame) -> str:
    if tuple(frame.columns) != LEDGER_COLUMNS:
        raise ValueError("BCTP transition ledger schema changed")
    records = []
    for row in frame.itertuples(index=False):
        records.append(
            {
                "sequence_id": str(row.sequence_id),
                "entry_time": _utc(row.entry_time)
                .isoformat()
                .replace("+00:00", "Z"),
                "current_position": str(row.current_position),
                "action_name": str(row.action_name),
                "action_target": float(row.action_target),
                "executed_target": float(row.executed_target),
                "reachable": bool(row.reachable),
                "terminal": bool(row.terminal),
                "reward": float(row.reward),
                "multiplier": float(row.multiplier),
                "held_path_downside_fraction": float(
                    row.held_path_downside_fraction
                ),
                "changed_notional_fraction": float(
                    row.changed_notional_fraction
                ),
                "entry_cost": float(row.entry_cost),
                "terminal_cost": float(row.terminal_cost),
                "funding_cash": float(row.funding_cash),
                "bars_held": int(row.bars_held),
            }
        )
    return freeze.canonical_hash(records)


def deterministic_gzip_csv_bytes(frame: pd.DataFrame) -> bytes:
    if tuple(frame.columns) != LEDGER_COLUMNS:
        raise ValueError("BCTP transition ledger schema changed")
    text = frame.to_csv(index=False, lineterminator="\n")
    buffer = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        fileobj=buffer,
        mode="wb",
        mtime=0,
    ) as handle:
        handle.write(text.encode("utf-8"))
    return buffer.getvalue()


def write_ledger_once(
    path: str | Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    target = Path(path)
    payload = deterministic_gzip_csv_bytes(frame)
    if target.exists():
        if target.read_bytes() != payload:
            raise RuntimeError(f"BCTP transition ledger drift: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            delete=False,
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise RuntimeError(
                    f"BCTP transition ledger drift: {target}"
                )
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "path": str(target),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "frame_hash": ledger_frame_hash(frame),
        "rows": int(len(frame)),
        "columns": list(LEDGER_COLUMNS),
    }


def read_ledger(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        compression="gzip",
        float_precision="round_trip",
    )
    if tuple(frame.columns) != LEDGER_COLUMNS:
        raise ValueError("BCTP transition ledger schema changed")
    frame["entry_time"] = pd.to_datetime(
        frame["entry_time"],
        utc=True,
        errors="raise",
    )
    numeric = (
        "action_target",
        "executed_target",
        "reward",
        "multiplier",
        "held_path_downside_fraction",
        "changed_notional_fraction",
        "entry_cost",
        "terminal_cost",
        "funding_cash",
        "bars_held",
    )
    if not np.isfinite(frame.loc[:, numeric].to_numpy(float)).all():
        raise ValueError("BCTP transition ledger contains non-finite metrics")
    for column in ("reachable", "terminal"):
        if not frame[column].isin((True, False)).all():
            raise ValueError(f"BCTP transition ledger {column} changed")
        frame[column] = frame[column].astype(bool)
    if (
        frame.duplicated(
            ["sequence_id", "current_position", "action_name"]
        ).any()
        or not frame["entry_time"].is_monotonic_increasing
        or not frame["action_name"].isin(freeze.MODEL_ACTION_NAMES).all()
        or not frame["current_position"].isin(POSITION_ORDER).all()
    ):
        raise ValueError("BCTP transition ledger identity/order changed")
    return frame


def arrays_from_ledger(
    states: pd.DataFrame,
    ledger: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Reconstruct frozen model-order arrays from a validated transition ledger."""

    if tuple(states.columns) != SOURCE_COLUMNS:
        raise ValueError("BCTP transition source schema changed")
    if tuple(ledger.columns) != LEDGER_COLUMNS:
        raise ValueError("BCTP transition ledger schema changed")
    count = len(states)
    rewards = np.full(
        (count, len(POSITION_ORDER), len(ACTION_ORDER)),
        np.nan,
        dtype=np.float64,
    )
    reachable = np.zeros(
        (count, len(POSITION_ORDER)),
        dtype=bool,
    )
    terminal = np.zeros(count, dtype=bool)
    sequence_index = {
        str(sequence_id): index
        for index, sequence_id in enumerate(states["sequence_id"])
    }
    if len(sequence_index) != count:
        raise ValueError("BCTP transition source identities changed")
    for row in ledger.itertuples(index=False):
        state_index = sequence_index.get(str(row.sequence_id))
        if state_index is None:
            raise ValueError("BCTP ledger references unknown source state")
        try:
            position_index = POSITION_ORDER.index(str(row.current_position))
            action_index = freeze.MODEL_ACTION_NAMES.index(str(row.action_name))
        except ValueError as exc:
            raise ValueError("BCTP ledger action/position changed") from exc
        if not bool(row.reachable):
            raise ValueError("BCTP ledger contains unreachable row")
        if np.isfinite(rewards[state_index, position_index, action_index]):
            raise ValueError("BCTP ledger duplicates a transition")
        rewards[state_index, position_index, action_index] = float(row.reward)
        reachable[state_index, position_index] = True
        terminal[state_index] = terminal[state_index] or bool(row.terminal)
    if not np.isfinite(rewards[reachable]).all():
        raise ValueError("BCTP ledger transition cube is incomplete")
    expected_reachable = np.ones_like(reachable)
    expected_reachable[0] = False
    expected_reachable[0, POSITION_ORDER.index("POSITION_FLAT")] = True
    if not np.array_equal(reachable, expected_reachable):
        raise ValueError("BCTP ledger reachability changed")
    if not terminal[-1] or terminal[:-1].any():
        raise ValueError("BCTP ledger terminal mask changed")
    return {
        "reward_tensor": rewards,
        "reachable_mask": reachable,
        "terminal": terminal,
    }
