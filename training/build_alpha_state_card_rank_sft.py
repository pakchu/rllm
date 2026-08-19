"""Build causal state-card ordinal and pairwise data from alpha event-gate rows.

Input event rows may contain realized outcomes in metadata.  Those outcomes are
used only for targets and label metadata.  Every prompt is rebuilt from frozen
candidate descriptors and market/funding observations available before entry.
Ordinal cut points are fitted once from the train split and then frozen for all
other splits.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STAGE_ORDER = {"train": 0, "test": 1, "eval": 2, "final": 3}
RETURN_BARS = {"1h": 12, "4h": 48, "24h": 288, "72h": 864}
VOL_BARS = {"6h": 72, "24h": 288, "72h": 864}
BAR = pd.Timedelta(minutes=5)


@dataclass(frozen=True)
class StateCardRankConfig:
    input_jsonls: str
    market_csv: str
    funding_csv: str
    pointwise_output: str
    pairwise_output: str
    summary_output: str = ""
    min_utility_gap: float = 0.001
    formula_max_chars: int = 2400


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )
    target.write_text(text)


def _load_frame(path: str | Path, *, funding: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time_col = next(
        (name for name in ("date", "ts", "timestamp", "funding_time") if name in frame),
        None,
    )
    if time_col is None:
        raise ValueError(f"missing timestamp column in {path}")
    frame = frame.rename(columns={time_col: "date"})
    frame["date"] = pd.to_datetime(
        frame["date"], utc=True, errors="raise", format="mixed"
    )
    if funding and "funding_rate" not in frame:
        raise ValueError("funding CSV must contain funding_rate")
    required = (
        {"date", "funding_rate"}
        if funding
        else {"date", "open", "high", "low", "close"}
    )
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    value_cols = ["funding_rate"] if funding else ["open", "high", "low", "close"]
    for col in value_cols:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError(f"duplicate timestamps in {path}")
    return frame


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _extract_formula(row: dict[str, Any]) -> Any:
    if "formula" in row:
        return row["formula"]
    prompt = str(row.get("prompt", ""))
    marker = "frozen_formula: "
    for line in prompt.splitlines():
        if line.startswith(marker):
            raw = line[len(marker) :]
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
    return {}


def _compact_formula(formula: Any, limit: int) -> str:
    text = _canonical(formula)
    if len(text) <= limit:
        return text
    # Truncation is presentation-only; preserve a stable digest so two long
    # formulas with a common prefix cannot become indistinguishable.
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{text[: max(0, limit - 30)]}...<sha256:{digest}>"


def _net_return(row: dict[str, Any]) -> float:
    metadata = row.get("metadata")
    raw = metadata.get("net_return") if isinstance(metadata, dict) else None
    if raw is None:
        raw = row.get("net_return")
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError("non-finite net_return")
    return value


def _entry(row: dict[str, Any]) -> pd.Timestamp:
    value = pd.Timestamp(row["entry_time"])
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _hold_minutes(row: dict[str, Any]) -> int:
    if "hold_minutes" in row:
        return int(row["hold_minutes"])
    entry = _entry(row)
    exit_time = pd.Timestamp(row["exit_time"])
    exit_time = (
        exit_time.tz_localize("UTC")
        if exit_time.tzinfo is None
        else exit_time.tz_convert("UTC")
    )
    minutes = int((exit_time - entry) / pd.Timedelta(minutes=1))
    if minutes <= 0:
        raise ValueError("hold must be positive")
    return minutes


def fit_train_quantiles(
    rows: Iterable[dict[str, Any]],
) -> tuple[float, float, float, float]:
    """Fit Q0..Q4 boundaries using eligible train outcomes only."""
    values = [
        _net_return(row)
        for row in rows
        if row.get("research_train_pass") is True
        and str(row.get("stage", "")) == "train"
    ]
    if not values:
        raise ValueError("no research_train_pass train outcomes for ordinal thresholds")
    return tuple(
        float(v)
        for v in np.quantile(np.asarray(values, dtype=float), [0.2, 0.4, 0.6, 0.8])
    )  # type: ignore[return-value]


def ordinal_target(
    net_return: float, thresholds: tuple[float, float, float, float]
) -> str:
    # right-side insertion deterministically assigns exact cut-point ties to
    # the upper bucket.
    return f"Q{int(np.searchsorted(np.asarray(thresholds), net_return, side='right'))}"


def signal_time_features(
    market: pd.DataFrame, funding: pd.DataFrame, entry_time: str | pd.Timestamp
) -> dict[str, float | None]:
    """Derive features from completed bars and settled funding strictly pre-entry."""
    entry = pd.Timestamp(entry_time)
    entry = (
        entry.tz_localize("UTC") if entry.tzinfo is None else entry.tz_convert("UTC")
    )
    market_dates = pd.to_datetime(market["date"], utc=True, errors="raise")
    bars = market.loc[market_dates.lt(entry)].copy()
    bars["date"] = market_dates.loc[market_dates.lt(entry)]
    bars = bars.sort_values("date", kind="mergesort")
    needed = max(RETURN_BARS.values()) + 1
    if len(bars) < needed:
        raise ValueError("insufficient pre-entry 5m history")
    history = bars.iloc[-needed:]
    if history["date"].iloc[-1] + BAR > entry:
        raise ValueError("latest selected bar is not completed before entry")
    if not history["date"].diff().iloc[1:].eq(BAR).all():
        raise ValueError("pre-entry 5m history is not contiguous")
    values = history[["open", "high", "low", "close"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("invalid market history")

    closes = history["close"].to_numpy(float)
    features: dict[str, float | None] = {}
    for name, count in RETURN_BARS.items():
        features[f"return_{name}"] = float(closes[-1] / closes[-count - 1] - 1.0)
    log_returns = np.diff(np.log(closes))
    for name, count in VOL_BARS.items():
        features[f"realized_vol_{name}"] = float(
            np.sqrt(np.square(log_returns[-count:]).sum())
        )

    day = history.iloc[-RETURN_BARS["24h"] :]
    day_low, day_high, latest = (
        float(day["low"].min()),
        float(day["high"].max()),
        float(closes[-1]),
    )
    features["range_position_24h"] = (
        float((latest - day_low) / (day_high - day_low)) if day_high > day_low else 0.5
    )
    long_window = history.iloc[-RETURN_BARS["72h"] :]
    features["drawdown_72h"] = float(latest / float(long_window["high"].max()) - 1.0)
    features["runup_72h"] = float(latest / float(long_window["low"].min()) - 1.0)

    funding_dates = pd.to_datetime(funding["date"], utc=True, errors="raise")
    settled = funding.loc[funding_dates.lt(entry)].copy()
    settled["date"] = funding_dates.loc[funding_dates.lt(entry)]
    settled = settled.sort_values("date", kind="mergesort")
    if settled.empty:
        features["funding_latest"] = None
    else:
        latest_funding = float(settled["funding_rate"].iloc[-1])
        if not np.isfinite(latest_funding):
            raise ValueError("invalid latest funding")
        features["funding_latest"] = latest_funding
    for name, hours in (("24h", 24), ("72h", 72)):
        window = settled.loc[
            settled["date"].ge(entry - pd.Timedelta(hours=hours)), "funding_rate"
        ]
        if not np.isfinite(window.to_numpy(float)).all():
            raise ValueError("invalid trailing funding")
        features[f"funding_sum_{name}"] = float(window.sum())
    return {
        key: (round(value, 10) if isinstance(value, float) else value)
        for key, value in features.items()
    }


def _candidate(
    row: dict[str, Any], features: dict[str, float | None], formula_max_chars: int
) -> dict[str, Any]:
    formula = _extract_formula(row)
    policy = formula.get("policy", {}) if isinstance(formula, dict) else {}
    return {
        "policy_id": str(row.get("policy_id", "")),
        "slug": str(row.get("slug", "")),
        "side": int(row["side"]),
        "hold_minutes": _hold_minutes(row),
        "frozen_policy": policy,
        "frozen_formula": _compact_formula(formula, formula_max_chars),
        "market_state": features,
    }


def _point_prompt(entry: pd.Timestamp, candidate: dict[str, Any]) -> str:
    return "\n".join(
        (
            "You are a BTC alpha state-card quality ranker.",
            "Use only the signal-time state and immutable candidate. Return exactly one ordinal token Q0, Q1, Q2, Q3, or Q4.",
            f"entry_time: {entry.isoformat()}",
            f"candidate: {_canonical(candidate)}",
        )
    )


def _pair_prompt(entry: pd.Timestamp, a: dict[str, Any], b: dict[str, Any]) -> str:
    return "\n".join(
        (
            "You are a BTC alpha same-entry preference ranker.",
            "Use only signal-time state and immutable candidate fields. Return exactly A or B.",
            f"entry_time: {entry.isoformat()}",
            f"A: {_canonical(a)}",
            f"B: {_canonical(b)}",
        )
    )


def build_dataset(
    event_rows: Iterable[dict[str, Any]],
    market: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    min_utility_gap: float = 0.001,
    formula_max_chars: int = 2400,
) -> dict[str, Any]:
    """Return deterministic pointwise/pairwise rows and train-fitted thresholds."""
    if min_utility_gap < 0:
        raise ValueError("min_utility_gap must be non-negative")
    events = [dict(row) for row in event_rows if row.get("research_train_pass") is True]
    events.sort(
        key=lambda row: (
            STAGE_ORDER.get(str(row.get("stage")), 99),
            _entry(row),
            str(row.get("policy_id", "")),
            str(row.get("slug", "")),
            int(row.get("side", 0)),
            _hold_minutes(row),
        )
    )
    thresholds = fit_train_quantiles(events)
    pointwise: list[dict[str, Any]] = []
    enriched: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    skipped: Counter[str] = Counter()
    feature_cache: dict[pd.Timestamp, dict[str, float | None]] = {}
    for row in events:
        try:
            entry = _entry(row)
            if entry not in feature_cache:
                feature_cache[entry] = signal_time_features(market, funding, entry)
            features = feature_cache[entry]
            candidate = _candidate(row, features, formula_max_chars)
            utility = _net_return(row)
        except (KeyError, TypeError, ValueError):
            skipped["invalid_or_incomplete_row"] += 1
            continue
        target = ordinal_target(utility, thresholds)
        record = {
            "task": "alpha_state_card_ordinal",
            "stage": str(row.get("stage", "")),
            "entry_time": entry.isoformat(),
            "policy_id": candidate["policy_id"],
            "prompt": _point_prompt(entry, candidate),
            "target": target,
            "metadata": {
                "net_return": utility,
                "ordinal_thresholds_train_only": list(thresholds),
                "leakage_guard": "prompt contains only observations strictly before entry and immutable candidate fields",
            },
        }
        pointwise.append(record)
        enriched.append((row, candidate, utility))

    groups: dict[
        tuple[str, pd.Timestamp], list[tuple[dict[str, Any], dict[str, Any], float]]
    ] = defaultdict(list)
    for item in enriched:
        groups[(str(item[0].get("stage", "")), _entry(item[0]))].append(item)
    pairwise: list[dict[str, Any]] = []
    for (stage, entry), candidates in sorted(
        groups.items(), key=lambda item: (STAGE_ORDER.get(item[0][0], 99), item[0][1])
    ):
        candidates.sort(
            key=lambda item: (
                item[1]["policy_id"],
                item[1]["slug"],
                item[1]["side"],
                item[1]["hold_minutes"],
            )
        )
        choices = list(candidates)
        if choices:
            wait = {
                "policy_id": "WAIT",
                "slug": "WAIT",
                "side": 0,
                "hold_minutes": 0,
                "frozen_policy": {"action": "WAIT"},
                "frozen_formula": "WAIT",
                "market_state": choices[0][1]["market_state"],
            }
            choices.append(({}, wait, 0.0))
        for (_, cand_a, utility_a), (_, cand_b, utility_b) in itertools.combinations(
            choices, 2
        ):
            orientation = hashlib.sha256(
                f"{stage}|{entry.isoformat()}|{cand_a['policy_id']}|{cand_b['policy_id']}".encode()
            ).digest()[0]
            if orientation & 1:
                cand_a, cand_b = cand_b, cand_a
                utility_a, utility_b = utility_b, utility_a
            gap = float(utility_a - utility_b)
            if abs(gap) <= min_utility_gap:
                skipped["utility_gap_not_above_threshold"] += 1
                # Utility-gap filtering is a train-label decision.  Applying
                # it to later stages would leak the future by changing which
                # comparisons exist at inference time.
                if stage == "train":
                    continue
            target = "A" if gap > 0 else "B"
            pairwise.append(
                {
                    "task": "alpha_state_card_pairwise_preference",
                    "stage": stage,
                    "entry_time": entry.isoformat(),
                    "prompt": _pair_prompt(entry, cand_a, cand_b),
                    "target": target,
                    "chosen": cand_a["policy_id"]
                    if target == "A"
                    else cand_b["policy_id"],
                    "rejected": cand_b["policy_id"]
                    if target == "A"
                    else cand_a["policy_id"],
                    "metadata": {
                        "candidate_a": cand_a["policy_id"],
                        "candidate_b": cand_b["policy_id"],
                        "candidate_a_card": cand_a,
                        "candidate_b_card": cand_b,
                        "utility_a": utility_a,
                        "utility_b": utility_b,
                        "utility_gap_a_minus_b": gap,
                        "leakage_guard": "future utility is label metadata only and never appears in the prompt",
                    },
                }
            )
    return {
        "pointwise": pointwise,
        "pairwise": pairwise,
        "thresholds": list(thresholds),
        "skipped": dict(sorted(skipped.items())),
    }


def build(cfg: StateCardRankConfig) -> dict[str, Any]:
    paths = [
        Path(value.strip()) for value in cfg.input_jsonls.split(",") if value.strip()
    ]
    if not paths:
        raise ValueError("input_jsonls is empty")
    event_rows = [
        row
        for path in sorted(paths, key=lambda p: p.as_posix())
        for row in _read_jsonl(path)
    ]
    market = _load_frame(cfg.market_csv, funding=False)
    funding = _load_frame(cfg.funding_csv, funding=True)
    dataset = build_dataset(
        event_rows,
        market,
        funding,
        min_utility_gap=cfg.min_utility_gap,
        formula_max_chars=cfg.formula_max_chars,
    )
    _write_jsonl(cfg.pointwise_output, dataset["pointwise"])
    _write_jsonl(cfg.pairwise_output, dataset["pairwise"])
    report = {
        "config": asdict(cfg),
        "eligible_input_rows": sum(
            row.get("research_train_pass") is True for row in event_rows
        ),
        "rows": {
            "pointwise": len(dataset["pointwise"]),
            "pairwise": len(dataset["pairwise"]),
        },
        "pointwise_targets": dict(
            sorted(Counter(row["target"] for row in dataset["pointwise"]).items())
        ),
        "pairwise_targets": dict(
            sorted(Counter(row["target"] for row in dataset["pairwise"]).items())
        ),
        "ordinal_thresholds_train_only": dataset["thresholds"],
        "skipped": dataset["skipped"],
        "sha256": {
            "pointwise": hashlib.sha256(
                Path(cfg.pointwise_output).read_bytes()
            ).hexdigest(),
            "pairwise": hashlib.sha256(
                Path(cfg.pairwise_output).read_bytes()
            ).hexdigest(),
        },
        "leakage_guard": {
            "market_bars_strictly_before_entry": True,
            "funding_settlements_strictly_before_entry": True,
            "future_return_in_prompts": False,
            "ordinal_thresholds_fit_on_train_only": True,
        },
    }
    if cfg.summary_output:
        target = Path(cfg.summary_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build causal alpha state-card ordinal and pairwise SFT data"
    )
    parser.add_argument(
        "--input-jsonls",
        required=True,
        help="comma-separated rllm_alpha_event_gate stage JSONLs",
    )
    parser.add_argument(
        "--market-csv", required=True, help="5m OHLC CSV (gzip accepted)"
    )
    parser.add_argument(
        "--funding-csv", required=True, help="settled funding CSV (gzip accepted)"
    )
    parser.add_argument("--pointwise-output", required=True)
    parser.add_argument("--pairwise-output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument(
        "--min-utility-gap", type=float, default=StateCardRankConfig.min_utility_gap
    )
    parser.add_argument(
        "--formula-max-chars", type=int, default=StateCardRankConfig.formula_max_chars
    )
    return parser.parse_args()


def main() -> None:
    print(
        json.dumps(
            build(StateCardRankConfig(**vars(parse_args()))),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
