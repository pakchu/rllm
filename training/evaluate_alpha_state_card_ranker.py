"""Leakage-safe deterministic benchmark for alpha state-card event rows.

The benchmark deliberately has a small model: a train-only, pairwise ridge
ranker.  The feature vocabulary, normalisation, ridge penalty and abstention
margin are all learned with expanding chronological splits wholly inside
2023H2.  The fitted artifact is then frozen before test/eval/final are opened.

Input is JSONL with one event per line.  Each event needs a signal timestamp
(``decision_time`` or ``date``) and ``options``.  Options may put causal numeric
inputs under ``causal_state``, ``features`` or ``state`` and identify the rule
with ``policy``/``family`` plus ``side``.  Train labels can be supplied as a
numeric ``utility``/``target_utility``/``outcome.utility`` per option or as the
event's ``target.choice_id``/``target.family``.  For economic evaluation, the
chosen option needs ``entry_time``, ``exit_time`` and ``side``.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TRAIN_START = pd.Timestamp("2023-07-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2024-01-01T00:00:00Z")
STAGES = {
    "train": (TRAIN_START, TRAIN_END),
    "test": (pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2025-01-01T00:00:00Z")),
    "eval": (pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")),
    "final": (pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
}
LEAKAGE_TOKENS = (
    "target", "outcome", "future", "forward_return", "realized_return", "net_return",
    "gross_return", "pnl", "profit", "utility", "label", "completion",
    "selected_metrics", "path_utility",
)
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0)
ABSTAIN_QUANTILES = (0.0, 0.25, 0.5, 0.75)


@dataclass(frozen=True)
class RankerConfig:
    input_jsonl: str
    market_csv: str
    funding_csv: str
    output: str
    base_cost_per_side: float = 0.0006
    stress_cost_per_side: float = 0.0010
    leverage: float = 0.5
    online_reliability: bool = False
    reliability_weight: float = 0.25
    signflip_draws: int = 20_000
    seed: int = 17


DEFAULT_INPUTS = (
    "data/rllm_alpha_event_gate_train_2026-08-19.jsonl,"
    "data/rllm_alpha_event_gate_test_trainpassed_2026-08-19.jsonl,"
    "data/rllm_alpha_event_gate_eval_trainpassed_2026-08-19.jsonl,"
    "data/rllm_alpha_event_gate_final_trainpassed_2026-08-19.jsonl"
)
DEFAULT_MARKET = "data/rllm_state_card_market_2023_2026.csv.gz"
DEFAULT_FUNDING = (
    "data/binance_um_btcusdt_funding_marks_2020_2023.csv.gz,"
    "data/rllm_state_card_funding_2023_2026.csv.gz"
)
DEFAULT_OUTPUT = "results/alpha_state_card_ranker_benchmark_2026-08-19.json"


@dataclass
class LinearRanker:
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    ridge: float
    abstain_margin: float

    def score(self, vectors: np.ndarray) -> np.ndarray:
        if not len(vectors):
            return np.empty(0, dtype=float)
        return ((vectors - self.mean) / self.scale) @ self.coefficients

    def public(self) -> dict[str, Any]:
        return {
            "kind": "pairwise_ridge_linear",
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "ridge": self.ridge,
            "abstain_margin": self.abstain_margin,
        }


def _utc(value: Any, field: str = "timestamp") -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _event_time(row: dict[str, Any]) -> pd.Timestamp:
    fold = row.get("fold") if isinstance(row.get("fold"), dict) else {}
    for key, value in (
        ("decision_time", row.get("decision_time")),
        ("date", row.get("date")),
        ("event_time", row.get("event_time")),
        ("fold.start", fold.get("start")),
    ):
        if value not in (None, ""):
            return _utc(value, key)
    raise ValueError("event row lacks decision_time/date/event_time/fold.start")


def _stage(ts: pd.Timestamp) -> str | None:
    for name, (start, end) in STAGES.items():
        if start <= ts < end:
            return name
    return None


def _input_paths(raw: str | Path) -> list[Path]:
    paths = [Path(value.strip()) for value in str(raw).split(",") if value.strip()]
    if not paths:
        raise ValueError("input JSONL list is empty")
    return sorted(paths, key=lambda path: path.as_posix())


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for source in _input_paths(path)
        for line in source.read_text().splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: (_event_time(row), str(row.get("event_id", ""))))
    return rows


def _prompt_json(row: dict[str, Any], marker: str) -> dict[str, Any]:
    prefix = marker + ": "
    for line in str(row.get("prompt", "")).splitlines():
        if line.startswith(prefix):
            try:
                parsed = json.loads(line[len(prefix) :])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def _event_gate_utility(row: dict[str, Any]) -> float:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    value = metadata.get("net_return", row.get("net_return"))
    number = _finite_number(value)
    if number is None:
        raise ValueError("eligible event-gate row lacks finite metadata.net_return")
    return number


def event_gate_rows_to_cards(
    rows: Iterable[dict[str, Any]], market: pd.DataFrame, funding: pd.DataFrame
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert executable event-gate rows into deterministic listwise cards.

    Pointwise/pairwise SFT exports are presentation datasets and intentionally
    omit executable clocks, so the evaluator consumes their original source
    rows instead.  Market/funding state is recomputed strictly before entry.
    """
    from training.build_alpha_state_card_rank_sft import signal_time_features

    eligible = [dict(row) for row in rows if row.get("research_train_pass") is True]
    groups: dict[tuple[str, pd.Timestamp], list[dict[str, Any]]] = {}
    feature_cache: dict[pd.Timestamp, dict[str, Any]] = {}
    for row in eligible:
        if str(row.get("task", "")) != "alpha_event_gate":
            raise ValueError(
                "SFT pointwise/pairwise rows are not executable; pass their original "
                "rllm_alpha_event_gate stage JSONLs"
            )
        stage = str(row.get("stage", ""))
        entry = _utc(row.get("entry_time"), "entry_time")
        groups.setdefault((stage, entry), []).append(row)
    cards: list[dict[str, Any]] = []
    for (stage, entry), candidates in sorted(
        groups.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        if entry not in feature_cache:
            feature_cache[entry] = signal_time_features(
                market,
                funding.rename(columns={"funding_time": "date"}),
                entry,
            )
        options: list[dict[str, Any]] = []
        for row in sorted(
            candidates,
            key=lambda item: (
                str(item.get("policy_id", "")), str(item.get("slug", "")),
                int(item.get("side", 0)), str(item.get("exit_time", "")),
            ),
        ):
            formula = _prompt_json(row, "frozen_formula")
            signal_event = _prompt_json(row, "signal_time_event")
            policy_parameters = formula.get("policy", {}) if isinstance(formula, dict) else {}
            options.append(
                {
                    "id": str(row.get("policy_id", "")),
                    "policy": str(row.get("policy_id", "")),
                    "family": str(row.get("slug", "")),
                    "side": int(row["side"]),
                    "features": {
                        "market_state": feature_cache[entry],
                        "signal_time_event": signal_event,
                        "policy_parameters": policy_parameters,
                    },
                    "target_utility": _event_gate_utility(row),
                    "trade": {
                        "entry_time": entry.isoformat(),
                        "exit_time": _utc(row.get("exit_time"), "exit_time").isoformat(),
                        "side": int(row["side"]),
                    },
                }
            )
        options.append(
            {
                "id": "WAIT",
                "policy": "WAIT",
                "family": "WAIT",
                "side": 0,
                "features": {"market_state": feature_cache[entry]},
                "target_utility": 0.0,
            }
        )
        cards.append(
            {
                "event_id": f"{stage}|{entry.isoformat()}",
                "stage": stage,
                "decision_time": entry.isoformat(),
                "options": options,
                "source": "converted_original_alpha_event_gate_rows",
            }
        )
    audit = {
        "mode": "original_event_gate_to_listwise_event_card",
        "input_rows": len(list(rows)) if isinstance(rows, list) else None,
        "eligible_rows": len(eligible),
        "event_cards": len(cards),
        "market_state_strictly_pre_entry": True,
        "outcomes_used_only_as_labels": True,
        "sft_rows_consumed": False,
    }
    return cards, audit


def load_event_cards(
    path: str | Path, market: pd.DataFrame, funding: pd.DataFrame
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = [
        json.loads(line)
        for source in _input_paths(path)
        for line in source.read_text().splitlines()
        if line.strip()
    ]
    if not raw:
        raise ValueError("input JSONLs contain no rows")
    if all(isinstance(row.get("options"), list) for row in raw):
        cards = [dict(row) for row in raw]
        audit = {"mode": "native_event_card_jsonl", "input_rows": len(raw), "event_cards": len(cards)}
    else:
        cards, audit = event_gate_rows_to_cards(raw, market, funding)
    cards.sort(key=lambda row: (_event_time(row), str(row.get("event_id", ""))))
    return cards, audit


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _flatten_numeric(value: Any, prefix: str, out: dict[str, float]) -> None:
    if any(token in prefix.lower() for token in LEAKAGE_TOKENS):
        return
    number = _finite_number(value)
    if number is not None:
        out[prefix] = number
    elif isinstance(value, dict):
        for key in sorted(value):
            _flatten_numeric(value[key], f"{prefix}.{key}" if prefix else str(key), out)


def _policy(option: dict[str, Any]) -> str:
    return str(option.get("policy") or option.get("family") or option.get("candidate_id") or "UNKNOWN")


def _side(option: dict[str, Any]) -> str:
    trade = option.get("trade") if isinstance(option.get("trade"), dict) else {}
    value = option.get("side", trade.get("side"))
    if value in (1, 1.0):
        return "LONG"
    if value in (-1, -1.0):
        return "SHORT"
    text = str(value or "NONE").upper()
    return {"BUY": "LONG", "SELL": "SHORT", "+1": "LONG", "-1": "SHORT"}.get(text, text)


def _option_features(row: dict[str, Any], option: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for root in ("causal_state", "features", "state", "latest_evidence"):
        if isinstance(row.get(root), dict):
            _flatten_numeric(row[root], f"event.{root}", values)
        if isinstance(option.get(root), dict):
            _flatten_numeric(option[root], f"option.{root}", values)
    # State-card builders place several causal scalars directly on each option.
    for key in sorted(option):
        if key not in {"causal_state", "features", "state", "latest_evidence"}:
            _flatten_numeric(option[key], f"option.{key}", values)
    values[f"policy={_policy(option)}"] = 1.0
    values[f"side={_side(option)}"] = 1.0
    return values


def _utility(row: dict[str, Any], option: dict[str, Any]) -> float | None:
    for key in ("target_utility", "utility"):
        number = _finite_number(option.get(key))
        if number is not None:
            return number
    outcome = option.get("outcome")
    if isinstance(outcome, dict):
        for key in ("utility", "net_return", "realized_return", "pnl"):
            number = _finite_number(outcome.get(key))
            if number is not None:
                return number
    target = row.get("target") if isinstance(row.get("target"), dict) else {}
    target_id = target.get("choice_id")
    target_family = target.get("family")
    if target_id is not None or target_family is not None:
        chosen = (target_id is not None and str(option.get("id")) == str(target_id)) or (
            target_family is not None and _policy(option) == str(target_family)
        )
        return float(chosen)
    return None


def _feature_vocabulary(rows: Iterable[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        for option in row.get("options") or []:
            names.update(_option_features(row, option))
    return sorted(names)


def _matrix(row: dict[str, Any], names: list[str]) -> np.ndarray:
    vectors = []
    for option in row.get("options") or []:
        values = _option_features(row, option)
        vectors.append([values.get(name, 0.0) for name in names])
    return np.asarray(vectors, dtype=float)


def _pairwise_data(rows: Iterable[dict[str, Any]], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x: list[np.ndarray] = []
    y: list[float] = []
    for row in rows:
        options = list(row.get("options") or [])
        vectors = _matrix(row, names)
        utilities = [_utility(row, option) for option in options]
        for left, right in itertools.combinations(range(len(options)), 2):
            if utilities[left] is None or utilities[right] is None or utilities[left] == utilities[right]:
                continue
            sign = 1.0 if float(utilities[left]) > float(utilities[right]) else -1.0
            # Add the antisymmetric pair.  This makes ordering irrelevant and
            # gives the closed-form linear fit a true pairwise objective.
            delta = vectors[left] - vectors[right]
            x.extend((delta, -delta))
            y.extend((sign, -sign))
    if not x:
        raise ValueError("train rows contain no labelled option pairs")
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _fit(rows: list[dict[str, Any]], names: list[str], ridge: float, margin: float = 0.0) -> LinearRanker:
    x, y = _pairwise_data(rows, names)
    mean = np.zeros(x.shape[1], dtype=float)  # differences must remain centred at zero
    scale = np.std(x, axis=0)
    scale[scale < 1e-12] = 1.0
    z = x / scale
    coefficients = np.linalg.solve(z.T @ z + float(ridge) * np.eye(z.shape[1]), z.T @ y)
    return LinearRanker(names, mean, scale, coefficients, float(ridge), float(margin))


def _winner(model: LinearRanker, row: dict[str, Any], reliability: dict[str, list[int]] | None = None, reliability_weight: float = 0.0) -> tuple[int | None, float, list[float]]:
    options = list(row.get("options") or [])
    if not options:
        return None, 0.0, []
    scores = model.score(_matrix(row, model.feature_names))
    if reliability is not None:
        for index, option in enumerate(options):
            wins, losses = reliability.get(_policy(option), [0, 0])
            scores[index] += float(reliability_weight) * math.log((wins + 1.0) / (losses + 1.0))
    order = sorted(range(len(options)), key=lambda i: (-float(scores[i]), str(options[i].get("id", i))))
    margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else float("inf")
    return (order[0] if margin >= model.abstain_margin else None), margin, scores.tolist()


def _selection_score(model: LinearRanker, rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    total = correct = selected = 0
    selected_utility = 0.0
    for row in rows:
        options = list(row.get("options") or [])
        idx, _, _ = _winner(model, row)
        labelled = [(i, _utility(row, option)) for i, option in enumerate(options)]
        labelled = [(i, value) for i, value in labelled if value is not None]
        if not labelled:
            continue
        total += 1
        if idx is not None:
            selected += 1
            selected_utility += float(_utility(row, options[idx]) or 0.0)
            best = max(float(value) for _, value in labelled)
            correct += int(float(_utility(row, options[idx]) or 0.0) == best)
    coverage = selected / max(1, total)
    accuracy = correct / max(1, selected)
    # Utility is primary, then accuracy; a tiny coverage term resolves ties
    # against needless abstention without ever consulting OOS.
    return selected_utility / max(1, total), accuracy, coverage


def select_and_fit(train_rows: list[dict[str, Any]]) -> tuple[LinearRanker, dict[str, Any]]:
    train_rows = sorted(train_rows, key=lambda row: (_event_time(row), str(row.get("event_id", ""))))
    if any(not (TRAIN_START <= _event_time(row) < TRAIN_END) for row in train_rows):
        raise ValueError("ranker selection received a row outside 2023H2 train")
    if len(train_rows) < 3:
        raise ValueError("at least three chronological train events are required")
    names = _feature_vocabulary(train_rows)
    if not names:
        raise ValueError("no causal numeric/categorical features found")
    # Expanding folds: first third -> second third, first two thirds -> final third.
    n = len(train_rows)
    cuts = sorted({max(1, n // 3), max(2, 2 * n // 3), n})
    folds = [(train_rows[: cuts[i]], train_rows[cuts[i] : cuts[i + 1]]) for i in range(len(cuts) - 1)]
    folds = [(fit_rows, val_rows) for fit_rows, val_rows in folds if fit_rows and val_rows]
    candidates: list[dict[str, Any]] = []
    for ridge in RIDGE_GRID:
        margins: list[float] = []
        fold_models: list[tuple[LinearRanker, list[dict[str, Any]]]] = []
        for fit_rows, val_rows in folds:
            try:
                fold_model = _fit(fit_rows, names, ridge)
            except ValueError:
                continue
            fold_models.append((fold_model, val_rows))
            margins.extend(_winner(fold_model, row)[1] for row in val_rows)
        if not fold_models:
            continue
        finite_margins = np.asarray([m for m in margins if math.isfinite(m)], dtype=float)
        thresholds = sorted({0.0, *(float(np.quantile(finite_margins, q)) for q in ABSTAIN_QUANTILES)}) if len(finite_margins) else [0.0]
        for threshold in thresholds:
            scores = []
            for fold_model, val_rows in fold_models:
                fold_model.abstain_margin = threshold
                scores.append(_selection_score(fold_model, val_rows))
            aggregate = tuple(float(np.mean([score[i] for score in scores])) for i in range(3))
            candidates.append({"ridge": ridge, "abstain_margin": threshold, "score": aggregate})
    if not candidates:
        raise ValueError("chronological train folds contain no fit-able labelled pairs")
    chosen = max(candidates, key=lambda item: (item["score"], -item["ridge"], -item["abstain_margin"]))
    model = _fit(train_rows, names, chosen["ridge"], chosen["abstain_margin"])
    audit = {
        "selection_boundary": [TRAIN_START.isoformat(), TRAIN_END.isoformat()],
        "method": "expanding_chronological_2023H2_only",
        "folds": [
            {"fit": [_event_time(a[0]).isoformat(), _event_time(a[-1]).isoformat()], "validation": [_event_time(b[0]).isoformat(), _event_time(b[-1]).isoformat()]}
            for a, b in folds
        ],
        "candidate_count": len(candidates),
        "chosen": chosen,
        "oos_used_for_selection": False,
    }
    return model, audit


def _read_market(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time_col = "date" if "date" in frame else "timestamp" if "timestamp" in frame else None
    if time_col is None or not {"open", "high", "low", "close"}.issubset(frame.columns):
        raise ValueError("market CSV requires date/timestamp and open/high/low/close")
    frame = frame.rename(columns={time_col: "date"})[["date", "open", "high", "low", "close"]].copy()
    frame["date"] = pd.to_datetime(
        frame["date"], utc=True, errors="raise", format="mixed"
    )
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return frame


def _read_funding(path: str | Path) -> pd.DataFrame:
    frames = []
    for source in _input_paths(path):
        frame = pd.read_csv(source)
        time_col = next(
            (name for name in ("funding_time", "date", "funding_time_utc", "timestamp") if name in frame),
            None,
        )
        mark_col = next(
            (name for name in ("settlement_mark_price", "mark_price") if name in frame),
            None,
        )
        if time_col is None or mark_col is None or "funding_rate" not in frame:
            raise ValueError(
                "funding CSV requires funding_time/date, funding_rate and exact settlement mark price"
            )
        normalized = frame.rename(
            columns={time_col: "funding_time", mark_col: "mark_price"}
        )[["funding_time", "funding_rate", "mark_price"]].copy()
        normalized["funding_time"] = pd.to_datetime(
            normalized["funding_time"], utc=True, errors="raise", format="mixed"
        )
        normalized["funding_rate"] = pd.to_numeric(normalized["funding_rate"], errors="raise")
        normalized["mark_price"] = pd.to_numeric(normalized["mark_price"], errors="raise")
        frames.append(normalized)
    combined = pd.concat(frames, ignore_index=True)
    # Prefer a positive exact settlement mark when the convenience aggregate
    # overlaps the canonical physical train source.
    combined["valid_mark"] = combined["mark_price"].gt(0).astype(int)
    combined = combined.sort_values(
        ["funding_time", "valid_mark"], kind="mergesort"
    ).drop_duplicates("funding_time", keep="last")
    return combined.drop(columns="valid_mark").sort_values("funding_time").reset_index(drop=True)


def _trade_clock(row: dict[str, Any], option: dict[str, Any]) -> dict[str, Any] | None:
    trade = option.get("trade") if isinstance(option.get("trade"), dict) else option
    if not trade.get("entry_time") or not trade.get("exit_time") or _side(trade) not in {"LONG", "SHORT"}:
        return None
    return {
        "decision_time": _event_time(row),
        "entry_time": _utc(trade["entry_time"], "entry_time"),
        "exit_time": _utc(trade["exit_time"], "exit_time"),
        "side": 1 if _side(trade) == "LONG" else -1,
        "policy": _policy(option),
    }


def predict_schedule(model: LinearRanker, rows: list[dict[str, Any]], *, online_reliability: bool = False, reliability_weight: float = 0.25) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reliability: dict[str, list[int]] = {}
    pending: list[tuple[pd.Timestamp, str, float]] = []
    schedule: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    busy_until: pd.Timestamp | None = None
    updates: list[dict[str, Any]] = []
    for row in sorted(rows, key=_event_time):
        now = _event_time(row)
        if online_reliability:
            matured = sorted((item for item in pending if item[0] <= now), key=lambda item: (item[0], item[1]))
            pending = [item for item in pending if item[0] > now]
            for exit_time, policy, utility in matured:
                counts = reliability.setdefault(policy, [0, 0])
                counts[0 if utility > 0.0 else 1] += 1
                updates.append({"available_at": exit_time.isoformat(), "applied_at": now.isoformat(), "policy": policy})
        idx, margin, scores = _winner(model, row, reliability if online_reliability else None, reliability_weight)
        reason = "selected"
        clock = None
        if idx is None:
            reason = "margin_abstention"
        else:
            clock = _trade_clock(row, list(row.get("options") or [])[idx])
            if clock is None:
                reason = "chosen_option_has_no_trade_clock"
            elif clock["entry_time"] < now or clock["exit_time"] <= clock["entry_time"]:
                raise ValueError("trade clock is not causal and positive-duration")
            elif busy_until is not None and clock["entry_time"] < busy_until:
                reason, clock = "overlap_abstention", None
        if clock is not None:
            busy_until = clock["exit_time"]
            schedule.append(clock)
            utility = _utility(row, list(row.get("options") or [])[idx])
            if online_reliability and utility is not None:
                pending.append((clock["exit_time"], clock["policy"], float(utility)))
        decisions.append({
            "decision_time": now.isoformat(), "choice_index": idx,
            "margin": margin if math.isfinite(margin) else None,
            "scores": scores, "reason": reason,
        })
    return schedule, {
        "events": len(rows), "selected_trades": len(schedule), "abstentions": len(rows) - len(schedule),
        "decisions": decisions, "online_reliability": online_reliability,
        "reliability_updates": updates,
        "update_rule": "outcome becomes available only when candidate exit_time <= next decision_time" if online_reliability else "disabled",
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 1e-15 else None


def strict_economics(schedule: list[dict[str, Any]], market: pd.DataFrame, funding: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp, cost_per_side: float, leverage: float) -> dict[str, Any]:
    positions = {ts: index for index, ts in enumerate(market["date"])}
    equity = peak = 1.0
    max_dd = 0.0
    trades: list[dict[str, Any]] = []

    def mark(value: float) -> None:
        nonlocal peak, max_dd
        peak = max(peak, value)
        max_dd = max(max_dd, 1.0 - value / max(peak, 1e-15))

    previous_exit: pd.Timestamp | None = None
    for clock in schedule:
        entry, exit_ = clock["entry_time"], clock["exit_time"]
        if not (start <= entry < exit_ <= end):
            continue
        if previous_exit is not None and entry < previous_exit:
            raise ValueError("selected schedule overlaps")
        previous_exit = exit_
        if entry not in positions or exit_ not in positions:
            raise ValueError("entry/exit absent from exact market grid")
        first, last = positions[entry], positions[exit_]
        side = int(clock["side"])
        entry_price, exit_price = float(market.iloc[first].open), float(market.iloc[last].open)
        pre = equity
        quantity = pre * float(leverage) / entry_price
        entry_fee = quantity * entry_price * float(cost_per_side)
        cash = pre - entry_fee
        mark(cash)
        exact_funding = funding[(funding.funding_time >= entry) & (funding.funding_time < exit_)].copy()
        if len(exact_funding) and (
            not np.isfinite(exact_funding[["funding_rate", "mark_price"]].to_numpy(float)).all()
            or exact_funding["mark_price"].le(0).any()
        ):
            raise ValueError("held funding rows lack positive exact settlement marks")
        funding_index = 0
        funding_cash = 0.0
        for pos in range(first, last):
            bar = market.iloc[pos]
            next_time = market.iloc[pos + 1].date
            while funding_index < len(exact_funding) and exact_funding.iloc[funding_index].funding_time < next_time:
                event = exact_funding.iloc[funding_index]
                flow = -side * quantity * float(event.mark_price) * float(event.funding_rate)
                cash += flow
                funding_cash += flow
                mark(cash + side * quantity * (float(event.mark_price) - entry_price) - quantity * float(event.mark_price) * cost_per_side)
                funding_index += 1
            favorable = float(bar.high if side > 0 else bar.low)
            adverse = float(bar.low if side > 0 else bar.high)
            mark(cash + side * quantity * (favorable - entry_price))
            mark(cash + side * quantity * (adverse - entry_price) - quantity * adverse * cost_per_side)
        if funding_index != len(exact_funding):
            raise ValueError("exact funding event not mapped to held market interval")
        gross_pnl = side * quantity * (exit_price - entry_price)
        exit_fee = quantity * exit_price * float(cost_per_side)
        equity = cash + gross_pnl - exit_fee
        mark(equity)
        trades.append({
            **{key: (value.isoformat() if isinstance(value, pd.Timestamp) else value) for key, value in clock.items()},
            "pre_entry_equity": pre, "post_exit_equity": equity, "entry_price": entry_price,
            "exit_price": exit_price, "entry_fee": entry_fee, "exit_fee": exit_fee,
            "funding_cash": funding_cash, "funding_events": len(exact_funding),
            "gross_underlying_bp": side * (exit_price / entry_price - 1.0) * 10_000.0,
            "net_return": equity / pre - 1.0,
        })
    years = max((end - start).total_seconds() / (365.25 * 86400.0), 1.0 / 365.25)
    cagr = equity ** (1.0 / years) - 1.0 if equity > 0.0 else -1.0
    return {
        "period": [start.isoformat(), end.isoformat()], "trades": len(trades),
        "absolute_return_pct": (equity - 1.0) * 100.0, "cagr_pct": cagr * 100.0,
        "strict_mdd_pct": max_dd * 100.0, "cagr_to_strict_mdd": _safe_ratio(cagr, max_dd),
        "ending_equity": equity,
        "mean_gross_underlying_bp": float(np.mean([row["gross_underlying_bp"] for row in trades])) if trades else 0.0,
        "funding_cash": sum(row["funding_cash"] for row in trades), "trade_rows": trades,
    }


def cluster_signflip(trades: list[dict[str, Any]], *, draws: int = 20_000, seed: int = 17) -> dict[str, Any]:
    clusters: dict[str, float] = {}
    for trade in trades:
        week = _utc(trade["entry_time"]).strftime("%G-W%V")
        clusters[week] = clusters.get(week, 0.0) + float(trade["net_return"])
    values = np.asarray([clusters[key] for key in sorted(clusters)], dtype=float)
    if not len(values):
        return {"method": "weekly_cluster_signflip_one_sided", "clusters": 0, "pvalue": 1.0, "draws": 0}
    observed = float(values.sum())
    if len(values) <= 16:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))), dtype=float)
        samples = signs @ values
        pvalue = float(np.mean(samples >= observed - 1e-15))
        used = len(samples)
        method = "exact_weekly_cluster_signflip_one_sided"
    else:
        rng = np.random.default_rng(int(seed))
        batch = rng.choice(np.asarray((-1.0, 1.0)), size=(int(draws), len(values)))
        exceed = int(np.sum(batch @ values >= observed - 1e-15))
        pvalue = (exceed + 1.0) / (int(draws) + 1.0)
        used = int(draws)
        method = "deterministic_mc_weekly_cluster_signflip_one_sided"
    return {"method": method, "clusters": len(values), "pvalue": pvalue, "draws": used, "seed": int(seed)}


def _without_trades(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "trade_rows"}


def _ranking_metrics(model: LinearRanker, rows: list[dict[str, Any]]) -> dict[str, Any]:
    labelled = correct = selected = 0
    selected_utility = 0.0
    for row in rows:
        options = list(row.get("options") or [])
        available = [(i, _utility(row, option)) for i, option in enumerate(options)]
        available = [(i, value) for i, value in available if value is not None]
        if not available:
            continue
        labelled += 1
        index, _, _ = _winner(model, row)
        if index is None:
            continue
        selected += 1
        utility = float(_utility(row, options[index]) or 0.0)
        selected_utility += utility
        correct += int(utility == max(float(value) for _, value in available))
    return {
        "labelled_events": labelled,
        "coverage": selected / max(1, labelled),
        "top1_accuracy_when_selected": correct / max(1, selected),
        "mean_selected_utility_per_labelled_event": selected_utility / max(1, labelled),
        "diagnostic_only": True,
    }


def _observed_end(market: pd.DataFrame, start: pd.Timestamp) -> pd.Timestamp:
    observed = market.loc[market.date >= start, "date"]
    if not len(observed):
        raise ValueError(f"market has no observations at or after {start.isoformat()}")
    interval = market.date.diff().dropna().median()
    if pd.isna(interval) or interval <= pd.Timedelta(0):
        interval = pd.Timedelta(minutes=5)
    return observed.max() + interval


def run(cfg: RankerConfig) -> dict[str, Any]:
    market, funding = _read_market(cfg.market_csv), _read_funding(cfg.funding_csv)
    rows, conversion = load_event_cards(cfg.input_jsonl, market, funding)
    train_rows = [row for row in rows if _stage(_event_time(row)) == "train"]
    oos_rows = [row for row in rows if _stage(_event_time(row)) in {"test", "eval", "final"}]
    model, selection = select_and_fit(train_rows)
    # The hash is computed before OOS prediction and records the exact frozen artifact.
    frozen = model.public()
    frozen_hash = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    all_schedule, prediction_audit = predict_schedule(
        model, rows, online_reliability=cfg.online_reliability,
        reliability_weight=cfg.reliability_weight,
    )
    reports: dict[str, Any] = {}
    for name, (start, configured_end) in STAGES.items():
        stage_rows = [row for row in rows if _stage(_event_time(row)) == name]
        # pd.Timestamp.max is unsuitable for annualisation; final ends at the
        # latest available market timestamp (half-open by one observed interval).
        end = configured_end
        if name == "final":
            in_final = market.loc[market.date >= start, "date"]
            end = _observed_end(market, start) if len(in_final) else start + pd.Timedelta(minutes=5)
        stage_schedule = [clock for clock in all_schedule if start <= clock["entry_time"] and clock["exit_time"] <= end]
        base = strict_economics(stage_schedule, market, funding, start=start, end=end, cost_per_side=cfg.base_cost_per_side, leverage=cfg.leverage)
        stress = strict_economics(stage_schedule, market, funding, start=start, end=end, cost_per_side=cfg.stress_cost_per_side, leverage=cfg.leverage)
        reports[name] = {
            "events": len(stage_rows), "selected_trades": len(stage_schedule),
            "ranking": _ranking_metrics(model, stage_rows),
            "base": _without_trades(base), "stress": _without_trades(stress),
            "signflip": cluster_signflip(base["trade_rows"], draws=cfg.signflip_draws, seed=cfg.seed),
        }
    if oos_rows:
        oos_start = STAGES["test"][0]
        oos_end = _observed_end(market, oos_start)
    else:
        oos_start, oos_end = STAGES["test"][0], STAGES["test"][0] + pd.Timedelta(minutes=5)
    oos_schedule = [clock for clock in all_schedule if oos_start <= clock["entry_time"] and clock["exit_time"] <= oos_end]
    oos_base = strict_economics(oos_schedule, market, funding, start=oos_start, end=oos_end, cost_per_side=cfg.base_cost_per_side, leverage=cfg.leverage)
    oos_stress = strict_economics(oos_schedule, market, funding, start=oos_start, end=oos_end, cost_per_side=cfg.stress_cost_per_side, leverage=cfg.leverage)
    report = {
        "protocol_version": "alpha_state_card_ranker_v1", "config": asdict(cfg),
        "input_conversion": conversion,
        "leakage_guard": {
            "fit_rows": "2023H2 only", "feature_vocabulary_from_train_only": True,
            "normalization_from_train_only": True, "hyperparameters_from_train_chronological_splits_only": True,
            "oos_used_for_selection": False, "ranker_frozen_before_oos": True,
        },
        "selection": selection, "frozen_ranker": frozen, "frozen_ranker_sha256": frozen_hash,
        "prediction_audit": prediction_audit, "stages": reports,
        "combined_oos": {
            "window": [oos_start.isoformat(), oos_end.isoformat()], "events": len(oos_rows),
            "selected_trades": len(oos_schedule), "base": _without_trades(oos_base),
            "stress": _without_trades(oos_stress),
            "ranking": _ranking_metrics(model, oos_rows),
            "signflip": cluster_signflip(oos_base["trade_rows"], draws=cfg.signflip_draws, seed=cfg.seed),
        },
        "accounting": {
            "quantity": "fixed side*leverage*pre_entry_equity/entry_open through exit",
            "funding": "exact cash=-side*quantity*settlement_mark*rate for entry<=funding_time<exit",
            "strict_mdd": "global HWM; entry cost; every held OHLC favorable then adverse; funding mark and virtual exit cost; exit cost",
            "costs": {"base_per_notional_side": cfg.base_cost_per_side, "stress_per_notional_side": cfg.stress_cost_per_side},
        },
    }
    destination = Path(cfg.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonls", "--input-jsonl", dest="input_jsonl", default=DEFAULT_INPUTS,
        help="comma-separated native event-card JSONLs or original alpha event-gate stage JSONLs",
    )
    parser.add_argument("--market-csv", default=DEFAULT_MARKET)
    parser.add_argument(
        "--funding-csv", default=DEFAULT_FUNDING,
        help="comma-separated exact funding sources; later valid duplicate marks win",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--base-cost-per-side", type=float, default=RankerConfig.base_cost_per_side)
    parser.add_argument("--stress-cost-per-side", type=float, default=RankerConfig.stress_cost_per_side)
    parser.add_argument("--leverage", type=float, default=RankerConfig.leverage)
    parser.add_argument("--online-reliability", action="store_true")
    parser.add_argument("--reliability-weight", type=float, default=RankerConfig.reliability_weight)
    parser.add_argument("--signflip-draws", type=int, default=RankerConfig.signflip_draws)
    parser.add_argument("--seed", type=int, default=RankerConfig.seed)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = run(RankerConfig(**vars(args)))
    print(json.dumps({"output": args.output, "frozen_ranker_sha256": report["frozen_ranker_sha256"], "combined_oos": report["combined_oos"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
