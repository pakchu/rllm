"""Train-only overlap-allowed Gross9 portfolio optimizer infrastructure.

G9-OVERLAP-PORT-1 changes the Gross9 overlap treatment from an exclusion gate
into a portfolio risk disclosure.  Inter-sleeve positions may overlap; each
input sleeve clock must remain internally non-overlapping.  The optimizer is a
train-only selector: it searches July-November 2023, replays exact fixed-quantity
aggregate-net ledger accounting for proxy finalists, and leaves the December
2023 holdout plus all OOS windows unopened for later sequential validation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from training import evaluate_gross9_qtr_distill_economics as fixed_ledger
from training import evaluate_gross9_async_active_veto_train_economics as train_sources
from training import build_gross9_overlap_portfolio_universe as universe_builder

POLICY_ID = "G9-OVERLAP-PORT-1"
PROTOCOL_VERSION = "gross9_overlap_allowed_portfolio_optimizer_v1"
AS_OF_DATE = "2026-09-03"
DEFAULT_OUTPUT = Path("results/gross9_overlap_portfolio_train_selection_2026-09-03.json")
DEFAULT_CONFIG_OUTPUT = Path("configs/shadow/gross9_overlap_portfolio_2026-09-03.json")
DEFAULT_PREREGISTRATION = Path("results/gross9_overlap_portfolio_preregistration_2026-09-03.json")
DEFAULT_UNIVERSE = Path("results/gross9_overlap_portfolio_universe_2026-09-03.json")
TRAIN_PROXY_WINDOW = ("2023-07-01T00:00:00Z", "2023-12-01T00:00:00Z")
DECEMBER_HOLDOUT_WINDOW = ("2023-12-01T00:00:00Z", "2024-01-01T00:00:00Z")
WEIGHT_GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.250001, 0.05))
GROSS_GRID = tuple(round(x, 2) for x in np.arange(0.25, 1.000001, 0.05))
MAX_SLEEVES = 8
BEAM_WIDTH = 8
PROXY_CANDIDATE_CAP = 12_000
EXACT_FINALIST_COUNT = 64
BASE_COST_BP = 6
STRESS_COST_BP = 10


@dataclass(frozen=True)
class SleeveClock:
    sleeve_id: str
    clock_path: Path | None
    clock_sha256: str | None
    clock: pd.DataFrame
    source: Mapping[str, Any]


@dataclass(frozen=True)
class PortfolioSpec:
    weights: tuple[tuple[str, float], ...]
    proxy_score: float
    proxy_metrics: Mapping[str, Any]

    @property
    def sleeve_ids(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.weights)

    @property
    def gross(self) -> float:
        return float(sum(abs(weight) for _, weight in self.weights))


@dataclass(frozen=True)
class OptimizerConfig:
    weight_grid: tuple[float, ...] = WEIGHT_GRID
    gross_grid: tuple[float, ...] = GROSS_GRID
    min_gross: float = 0.25
    max_gross: float = 1.0
    max_sleeves: int = MAX_SLEEVES
    beam_width: int = BEAM_WIDTH
    proxy_candidate_cap: int = PROXY_CANDIDATE_CAP
    exact_finalist_count: int = EXACT_FINALIST_COUNT
    max_month_share: float = 0.45
    min_trade_count: int = 30
    min_active_weeks: int = 12
    max_strict_mdd_pct: float = 12.0
    min_cagr_to_strict_mdd: float = 3.0
    min_stress_cagr_to_strict_mdd: float = 2.5
    max_mean_gross_exposure: float = 0.85
    max_turnover_weight_per_day: float = 100.0 / 365.25
    max_sleeve_turnover_share: float = 0.40


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _iso_z(timestamp: Any) -> str:
    return _utc(timestamp).isoformat().replace("+00:00", "Z")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz" or path.name.endswith(".csv.gz"):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            return pd.read_csv(handle)
    return pd.read_csv(path)


def load_universe_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{POLICY_ID} universe manifest must be a JSON object")
    if "manifest_hash" in payload:
        core = {key: value for key, value in payload.items() if key != "manifest_hash"}
        if payload["manifest_hash"] != canonical_hash(core):
            raise RuntimeError(f"{POLICY_ID} universe manifest hash drift: {manifest_path}")
    sleeves = payload.get("sleeves")
    if not isinstance(sleeves, list) or not sleeves:
        raise RuntimeError(f"{POLICY_ID} universe manifest requires a non-empty sleeves list")
    if (
        payload.get("policy_id") != POLICY_ID
        or payload.get("protocol_version") != universe_builder.PROTOCOL_VERSION
        or payload.get("precanonical_schedule_count") != 71
        or payload.get("canonical_sleeve_count") != 71
        or len(sleeves) != 71
    ):
        raise RuntimeError(f"{POLICY_ID} frozen universe identity/count drift")
    historical = sum(row.get("provenance", {}).get("kind") == "historical_gross9_near6h_only_reject" for row in sleeves)
    active = sum(row.get("provenance", {}).get("kind") == "active_veto_duplicate_only_canonical" for row in sleeves)
    if (historical, active) != (64, 7):
        raise RuntimeError(f"{POLICY_ID} frozen universe composition drift")
    inventory = payload.get("historical_novelty_inventory", {})
    inventory_path = Path(str(inventory.get("path", "")))
    if (
        not inventory_path.is_file()
        or sha256_file(inventory_path) != inventory.get("sha256")
        or json.loads(inventory_path.read_text(encoding="utf-8")).get("manifest_hash") != inventory.get("manifest_hash")
    ):
        raise RuntimeError(f"{POLICY_ID} historical inventory receipt drift")
    return payload


def load_validated_preregistration(path: Path = DEFAULT_PREREGISTRATION) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{POLICY_ID} missing committed preregistration: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{POLICY_ID} preregistration must be a JSON object")
    core = {key: item for key, item in value.items() if key != "manifest_hash"}
    if value.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(f"{POLICY_ID} preregistration manifest drift")
    module = importlib.import_module("training.preregister_gross9_overlap_portfolio")
    module.validate(value)
    if value != module.build():
        raise RuntimeError(f"{POLICY_ID} preregistration artifact differs from code")
    optimizer = value.get("implementation", {}).get("optimizer", {})
    if optimizer.get("sha256") != sha256_file(__file__):
        raise RuntimeError(f"{POLICY_ID} preregistration optimizer binding drift")
    return value


def load_bound_selection_sources(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    market = train_sources.load_market_hash_bound(start, end)
    funding = train_sources.load_train_funding_hash_bound(start, end)
    source = {
        "market": {
            "path": str(train_sources.econ.v1.MARKET),
            "sha256": train_sources.econ.v1.MARKET_SHA,
            "rows_opened": len(market),
            "start": _iso_z(market["date"].iloc[0]),
            "end": _iso_z(market["date"].iloc[-1]),
        },
        "funding": {
            "path": str(train_sources.econ.TRAIN_FUNDING),
            "sha256": train_sources.econ.TRAIN_FUNDING_SHA,
            "rows_opened": len(funding),
            "start": _iso_z(funding["date"].iloc[0]),
            "end": _iso_z(funding["date"].iloc[-1]),
        },
    }
    return market, funding, source


def _clock_record_for_stage(record: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    stage_clocks = record.get("stage_clocks")
    if isinstance(stage_clocks, Mapping) and stage in stage_clocks:
        value = stage_clocks[stage]
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{POLICY_ID} stage clock record must be an object")
        return value
    clock = record.get("clock")
    if isinstance(clock, Mapping):
        return clock
    return record


def load_sleeve_clock(
    record: Mapping[str, Any],
    stage: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    allow_holdout: bool = False,
    root: Path | None = None,
) -> SleeveClock:
    """Load one train-stage clock without opening December/OOS windows."""
    start = _utc(start); end = _utc(end)
    holdout_start = _utc(DECEMBER_HOLDOUT_WINDOW[0])
    if not allow_holdout and end > holdout_start:
        raise RuntimeError(f"{POLICY_ID} refuses to open December holdout or later windows during selection")
    if stage not in {"train", "train_proxy"}:
        raise RuntimeError(f"{POLICY_ID} optimizer may only load train clocks; requested {stage}")
    sleeve_id = str(record.get("sleeve_id") or record.get("id") or record.get("name") or "")
    if not sleeve_id:
        raise RuntimeError(f"{POLICY_ID} sleeve record missing id")
    clock_record = _clock_record_for_stage(record, stage)
    path_value = clock_record.get("path")
    if not path_value:
        raise RuntimeError(f"{POLICY_ID} sleeve {sleeve_id} missing clock path")
    clock_path = Path(str(path_value))
    if not clock_path.is_absolute() and not clock_path.is_file() and root is not None:
        clock_path = root / clock_path
    expected_hash = clock_record.get("sha256")
    if expected_hash is not None and sha256_file(clock_path) != str(expected_hash):
        raise RuntimeError(f"{POLICY_ID} sleeve {sleeve_id} clock hash drift")
    frame = _read_table(clock_path)
    return SleeveClock(
        sleeve_id=sleeve_id,
        clock_path=clock_path,
        clock_sha256=str(expected_hash) if expected_hash is not None else None,
        clock=normalize_sleeve_clock(frame, sleeve_id=sleeve_id, start=start, end=end),
        source=clock_record,
    )


def normalize_sleeve_clock(frame: pd.DataFrame, *, sleeve_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    required = {"entry_time", "exit_time", "side"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{POLICY_ID} clock schema for {sleeve_id} missing {sorted(required - set(frame.columns))}")
    out = frame.copy()
    if "split" in out.columns:
        out = out[out["split"].astype(str).isin(["train", "train_proxy"])]
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="raise")
    out["exit_time"] = pd.to_datetime(out["exit_time"], utc=True, errors="raise")
    out["side"] = pd.to_numeric(out["side"], errors="raise").astype(int)
    out = out[(out.entry_time >= start) & (out.exit_time <= end)].copy()
    if not out["side"].isin([-1, 1]).all():
        raise RuntimeError(f"{POLICY_ID} clock side must be +/-1 for {sleeve_id}")
    if not (out.entry_time < out.exit_time).all():
        raise RuntimeError(f"{POLICY_ID} non-positive interval in {sleeve_id}")
    out = out.sort_values(["entry_time", "exit_time"]).reset_index(drop=True)
    if len(out) > 1 and (out.entry_time.iloc[1:].to_numpy() < out.exit_time.iloc[:-1].to_numpy()).any():
        raise RuntimeError(f"{POLICY_ID} intra-sleeve overlap is forbidden: {sleeve_id}")
    out["sleeve"] = sleeve_id
    return out


def clock_signature(clock: pd.DataFrame) -> str:
    rows = [
        [
            _iso_z(row.entry_time),
            _iso_z(row.exit_time),
            int(row.side),
        ]
        for row in clock[["entry_time", "exit_time", "side"]].itertuples(index=False)
    ]
    return canonical_hash(rows)


def deduplicate_sleeve_clocks(sleeves: Sequence[SleeveClock]) -> list[SleeveClock]:
    """Keep the first deterministic sleeve for each exact canonical schedule."""
    seen: set[str] = set()
    unique: list[SleeveClock] = []
    for sleeve in sorted(sleeves, key=lambda item: item.sleeve_id):
        signature = clock_signature(sleeve.clock)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(sleeve)
    return unique


def _row_effects(
    clock: pd.DataFrame,
    market_opens: Mapping[pd.Timestamp, float] | None = None,
    round_trip_cost: float = 2 * BASE_COST_BP / 10_000.0,
) -> pd.Series:
    if market_opens is not None:
        values: list[float] = []
        for row in clock.itertuples(index=False):
            entry = float(market_opens[_utc(row.entry_time)])
            exit_price = float(market_opens[_utc(row.exit_time)])
            gross = int(row.side) * (exit_price / entry - 1.0)
            values.append(math.log(max(1e-12, 1.0 + gross - round_trip_cost)))
        return pd.Series(values, index=clock.index, dtype=float)
    if "proxy_log_effect" in clock.columns:
        return pd.to_numeric(clock["proxy_log_effect"], errors="raise")
    if "log_effect" in clock.columns:
        return pd.to_numeric(clock["log_effect"], errors="raise")
    if "net_return" in clock.columns:
        returns = pd.to_numeric(clock["net_return"], errors="raise")
        return np.log1p(returns)
    if "gross_return" in clock.columns:
        returns = pd.to_numeric(clock["gross_return"], errors="raise")
        return np.log1p(returns)
    # Outcome-free fallback exists only for unit-test/config construction.
    return pd.Series(np.zeros(len(clock)), index=clock.index, dtype=float)


def sleeve_proxy_series(sleeve: SleeveClock, market_opens: Mapping[pd.Timestamp, float] | None = None) -> pd.Series:
    effects = _row_effects(sleeve.clock, market_opens)
    if len(effects) == 0:
        return pd.Series(dtype=float)
    times = pd.DatetimeIndex(sleeve.clock["entry_time"])
    return pd.Series(effects.to_numpy(float), index=times).groupby(level=0).sum().sort_index()


def aggregate_proxy_series(weights: Mapping[str, float], sleeve_effects: Mapping[str, pd.Series]) -> pd.Series:
    pieces = []
    for sleeve_id, weight in weights.items():
        series = sleeve_effects[sleeve_id]
        if not series.empty:
            pieces.append(series * float(weight))
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces, axis=1).fillna(0.0).sum(axis=1).sort_index()


def proxy_metrics(effects: pd.Series) -> dict[str, Any]:
    if effects.empty:
        return {
            "log_return": 0.0,
            "absolute_return_pct": 0.0,
            "strict_mdd_pct": 0.0,
            "cagr_to_strict_mdd_proxy": 0.0,
            "weekly_positive_share": 0.0,
            "trade_effect_count": 0,
        }
    equity = np.exp(effects.cumsum().to_numpy(float))
    peak = np.maximum.accumulate(np.r_[1.0, equity])
    draw = 1.0 - np.r_[1.0, equity] / peak
    mdd_pct = float(np.max(draw) * 100.0)
    log_return = float(effects.sum())
    ret_pct = float((math.exp(log_return) - 1.0) * 100.0)
    weeks = effects.groupby([effects.index.isocalendar().year, effects.index.isocalendar().week]).sum()
    positive_share = float((weeks > 0).mean()) if len(weeks) else 0.0
    # Jul-Nov is a fixed 5-month development window; proxy CAGR is only for ranking.
    years = 153.0 / 365.25
    cagr_pct = float((math.exp(log_return / years) - 1.0) * 100.0) if years > 0 else 0.0
    return {
        "log_return": log_return,
        "absolute_return_pct": ret_pct,
        "strict_mdd_pct": mdd_pct,
        "cagr_to_strict_mdd_proxy": cagr_pct / mdd_pct if mdd_pct > 1e-12 else (999.0 if cagr_pct > 0 else 0.0),
        "weekly_positive_share": positive_share,
        "trade_effect_count": int(len(effects)),
    }


def deterministic_score(metrics: Mapping[str, Any], *, gross: float, sleeve_count: int) -> float:
    return float(metrics["cagr_to_strict_mdd_proxy"]) + 0.01 * float(metrics["absolute_return_pct"]) + 0.05 * float(metrics["weekly_positive_share"]) - 0.001 * sleeve_count - 0.0001 * gross


def beam_search_portfolios(sleeve_effects: Mapping[str, pd.Series], cfg: OptimizerConfig = OptimizerConfig()) -> list[PortfolioSpec]:
    ids = tuple(sorted(sleeve_effects))
    if not ids:
        return []
    beams: dict[int, list[PortfolioSpec]] = {0: [PortfolioSpec(weights=(), proxy_score=0.0, proxy_metrics={})]}
    generated = 0
    all_candidates: list[PortfolioSpec] = []
    for size in range(1, cfg.max_sleeves + 1):
        candidates: list[PortfolioSpec] = []
        for prior in beams.get(size - 1, []):
            used = {name for name, _ in prior.weights}
            last_idx = ids.index(prior.weights[-1][0]) if prior.weights else -1
            for sleeve_id in ids[last_idx + 1 :]:
                if sleeve_id in used:
                    continue
                for weight in cfg.weight_grid:
                    weights = tuple(sorted((*prior.weights, (sleeve_id, float(weight)))))
                    gross = sum(w for _, w in weights)
                    if gross > cfg.max_gross + 1e-12:
                        continue
                    if gross < min(cfg.gross_grid) - 1e-12 and size == cfg.max_sleeves:
                        continue
                    metrics = proxy_metrics(aggregate_proxy_series(dict(weights), sleeve_effects))
                    score = deterministic_score(metrics, gross=gross, sleeve_count=len(weights))
                    candidates.append(PortfolioSpec(weights=weights, proxy_score=score, proxy_metrics=metrics))
                    generated += 1
                    if generated >= cfg.proxy_candidate_cap:
                        break
                if generated >= cfg.proxy_candidate_cap:
                    break
            if generated >= cfg.proxy_candidate_cap:
                break
        candidates.sort(key=lambda p: (-p.proxy_score, p.gross, p.sleeve_ids, tuple(w for _, w in p.weights)))
        beams[size] = candidates[: cfg.beam_width]
        all_candidates.extend(c for c in candidates if c.gross + 1e-12 >= cfg.min_gross)
        if generated >= cfg.proxy_candidate_cap:
            break
    # Re-rank all accepted beam products and keep deterministic unique specs.
    unique: dict[tuple[tuple[str, float], ...], PortfolioSpec] = {candidate.weights: candidate for candidate in all_candidates}
    ranked = sorted(unique.values(), key=lambda p: (-p.proxy_score, p.gross, p.sleeve_ids, tuple(w for _, w in p.weights)))
    return ranked


def build_portfolio_clock(spec: PortfolioSpec, sleeve_clocks: Mapping[str, SleeveClock]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for sleeve_id, weight in spec.weights:
        clock = sleeve_clocks[sleeve_id].clock[["sleeve", "entry_time", "exit_time", "side"]].copy()
        clock["weight"] = float(weight)
        rows.append(clock)
    if not rows:
        return pd.DataFrame(columns=["sleeve", "weight", "entry_time", "exit_time", "side"])
    combined = pd.concat(rows, ignore_index=True)
    # qtr evaluator enforces intra-sleeve non-overlap but allows inter-sleeve overlap.
    return fixed_ledger.normalize_portfolio_clock(combined[["sleeve", "weight", "entry_time", "exit_time", "side"]], require_four_sleeves=False)


def exposure_and_turnover(clock: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    start = _utc(start); end = _utc(end)
    if clock.empty:
        return {"max_gross_exposure": 0.0, "mean_gross_exposure": 0.0, "max_abs_net_exposure": 0.0, "turnover_weight": 0.0, "turnover_weight_per_day": 0.0, "max_month_share": 0.0, "max_sleeve_turnover_share": 0.0, "active_iso_weeks": 0}
    sweep: dict[pd.Timestamp, list[tuple[int, float, float]]] = {}
    turnover = 0.0
    for row in clock.itertuples(index=False):
        signed = float(row.weight) * int(row.side)
        abs_weight = abs(float(row.weight))
        sweep.setdefault(_utc(row.entry_time), []).append((1, signed, abs_weight))
        sweep.setdefault(_utc(row.exit_time), []).append((-1, signed, abs_weight))
        turnover += 2.0 * abs_weight

    net = 0.0
    gross = 0.0
    max_gross = 0.0
    max_abs_net = 0.0
    gross_seconds = 0.0
    last = start
    for timestamp in sorted(sweep):
        timestamp = min(max(timestamp, start), end)
        seconds = max(0.0, (timestamp - last).total_seconds())
        gross_seconds += gross * seconds
        last = timestamp
        # Apply all same-timestamp exits and entries atomically.  This avoids
        # treating simultaneous opposite sleeves as transient net exposure while
        # still counting gross risk without side netting.
        for kind, signed, abs_weight in sorted(sweep[timestamp], key=lambda item: item[0]):
            if kind == 1:
                net += signed
                gross += abs_weight
            else:
                net -= signed
                gross -= abs_weight
        if abs(gross) < 1e-12:
            gross = 0.0
        if abs(net) < 1e-12:
            net = 0.0
        max_gross = max(max_gross, gross)
        max_abs_net = max(max_abs_net, abs(net))
    gross_seconds += gross * max(0.0, (end - last).total_seconds())
    months = clock["entry_time"].dt.strftime("%Y-%m").value_counts()
    sleeve_turnover = clock.groupby("sleeve")["weight"].apply(lambda values: float(2.0 * values.abs().sum()))
    iso = clock["entry_time"].dt.isocalendar()
    days = max((end - start).total_seconds() / 86400.0, 1e-12)
    return {
        "max_gross_exposure": float(max_gross),
        "mean_gross_exposure": float(gross_seconds / max((end - start).total_seconds(), 1e-12)),
        "max_abs_net_exposure": float(max_abs_net),
        "turnover_weight": float(turnover),
        "turnover_weight_per_day": float(turnover / days),
        "max_month_share": float(months.max() / len(clock)) if len(months) else 0.0,
        "max_sleeve_turnover_share": float(sleeve_turnover.max() / turnover) if turnover else 0.0,
        "active_iso_weeks": int(len(set(zip(iso["year"].astype(int), iso["week"].astype(int), strict=True)))),
    }


def exact_score(primary: Mapping[str, Any], risk: Mapping[str, Any], spec: PortfolioSpec) -> float:
    base = primary["base"]
    stress = primary["stress"]
    return (
        float(base["cagr_to_strict_mdd"]) * 10.0
        + float(base["absolute_return_pct"])
        + 0.2 * float(stress["cagr_to_strict_mdd"])
        - 0.2 * float(base["strict_mdd_pct"])
        - 0.05 * float(risk["turnover_weight_per_day"])
        - 0.01 * len(spec.weights)
    )


def exact_gates(
    primary: Mapping[str, Any],
    risk: Mapping[str, Any],
    monthly: Sequence[Mapping[str, Any]],
    cfg: OptimizerConfig,
) -> dict[str, bool]:
    base = primary["base"]
    stress = primary["stress"]
    return {
        "source_trade_count_min": int(base["intervals"]) >= cfg.min_trade_count,
        "month_concentration_max": float(risk["max_month_share"]) <= cfg.max_month_share,
        "absolute_return_positive": float(base["absolute_return_pct"]) > 0.0,
        "cagr_to_strict_mdd_min": float(base["cagr_to_strict_mdd"]) >= cfg.min_cagr_to_strict_mdd,
        "strict_mdd_max": float(base["strict_mdd_pct"]) <= cfg.max_strict_mdd_pct,
        "stress_absolute_return_positive": float(stress["absolute_return_pct"]) > 0.0,
        "stress_cagr_to_strict_mdd_min": float(stress["cagr_to_strict_mdd"]) >= cfg.min_stress_cagr_to_strict_mdd,
        "mean_gross_edge_min_20bp": float(base.get("mean_exposure_weighted_gross_edge_bp", 0.0)) >= 20.0,
        "both_chronological_halves_positive": all(float(row["absolute_return_pct"]) > 0.0 for row in primary["calendar_halves"].values()),
        "active_iso_weeks_min": int(risk["active_iso_weeks"]) >= cfg.min_active_weeks,
        "base_positive_months_min_4_of_5": sum(float(row["base_return_pct"]) > 0.0 for row in monthly) >= 4,
        "stress_positive_months_min_3_of_5": sum(float(row["stress_return_pct"]) > 0.0 for row in monthly) >= 3,
        "worst_stress_month_min_minus_2_5": min((float(row["stress_return_pct"]) for row in monthly), default=-100.0) >= -2.5,
        "mean_gross_exposure_cap": float(risk["mean_gross_exposure"]) <= cfg.max_mean_gross_exposure,
        "max_gross_exposure_cap": float(risk["max_gross_exposure"]) <= cfg.max_gross + 1e-9,
        "turnover_cap": float(risk["turnover_weight_per_day"]) <= cfg.max_turnover_weight_per_day,
        "sleeve_turnover_share_cap": float(risk["max_sleeve_turnover_share"]) <= cfg.max_sleeve_turnover_share,
    }


def evaluate_monthly_stability(
    clock: pd.DataFrame,
    market: pd.DataFrame,
    funding: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for month_start in pd.date_range(start, end, freq="MS", inclusive="left"):
        month_end = min(month_start + pd.offsets.MonthBegin(1), end)
        subset = clock.loc[clock["entry_time"].ge(month_start) & clock["exit_time"].le(month_end)]
        base = fixed_ledger.simulate_portfolio(subset, market, funding, month_start, month_end, BASE_COST_BP / 10_000.0)
        stress = fixed_ledger.simulate_portfolio(subset, market, funding, month_start, month_end, STRESS_COST_BP / 10_000.0)
        rows.append({
            "month": month_start.strftime("%Y-%m"),
            "base_return_pct": float(base["absolute_return_pct"]),
            "stress_return_pct": float(stress["absolute_return_pct"]),
            "intervals": int(base["intervals"]),
        })
    return rows


def evaluate_exact_finalists(
    proxy_ranked: Sequence[PortfolioSpec],
    sleeve_clocks: Mapping[str, SleeveClock],
    market: pd.DataFrame,
    funding: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: OptimizerConfig = OptimizerConfig(),
) -> list[dict[str, Any]]:
    if len(proxy_ranked) < cfg.exact_finalist_count:
        raise RuntimeError(
            f"{POLICY_ID} requires at least {cfg.exact_finalist_count} proxy finalists"
        )
    finalists = list(proxy_ranked[: cfg.exact_finalist_count])
    evaluated: list[dict[str, Any]] = []
    for proxy_rank, spec in enumerate(finalists, start=1):
        clock = build_portfolio_clock(spec, sleeve_clocks)
        primary = fixed_ledger.evaluate_primary(clock, market, funding, _utc(start), _utc(end))
        risk = exposure_and_turnover(clock, _utc(start), _utc(end))
        monthly = evaluate_monthly_stability(clock, market, funding, _utc(start), _utc(end))
        gates = exact_gates(primary, risk, monthly, cfg)
        score = exact_score(primary, risk, spec)
        evaluated.append(
            {
                "proxy_rank": proxy_rank,
                "sleeve_weights": dict(spec.weights),
                "gross": spec.gross,
                "proxy_score": spec.proxy_score,
                "proxy_metrics": dict(spec.proxy_metrics),
                "primary": primary,
                "risk": risk,
                "monthly_stability": monthly,
                "gates": gates,
                "passed": all(gates.values()),
                "exact_score": score,
            }
        )
    evaluated.sort(key=lambda row: (not bool(row["passed"]), -float(row["exact_score"]), int(row["proxy_rank"]), json.dumps(row["sleeve_weights"], sort_keys=True)))
    if len(evaluated) != cfg.exact_finalist_count:
        raise RuntimeError(f"{POLICY_ID} exact finalist count drift")
    return evaluated


def select_authoritative_rank1(evaluated: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    passing = [row for row in evaluated if row.get("passed") is True]
    if not passing:
        raise RuntimeError(f"{POLICY_ID} no exact finalist passed gates; terminal train reject")
    winner = dict(passing[0])
    winner["authoritative_rank"] = 1
    winner["selection_status"] = "frozen_train_rank1_before_december_holdout"
    return winner


def max_t_signflip_pvalue(candidate_weekly_effects: Mapping[str, Sequence[float]], draws: int = 10_000, seed: int = 20260903) -> dict[str, Any]:
    """Shared-sign weekly max-T API for evaluated proxy/finalist effects."""
    ids = tuple(sorted(candidate_weekly_effects))
    if not ids:
        return {"method": "shared_weekly_signflip_max_t", "candidate_count": 0, "adjusted_pvalues": {}, "draws": draws, "seed": seed}
    width = max(len(candidate_weekly_effects[cid]) for cid in ids)
    matrix = np.zeros((len(ids), width), dtype=float)
    for i, cid in enumerate(ids):
        vals = np.asarray(candidate_weekly_effects[cid], dtype=float)
        matrix[i, : vals.size] = vals
    observed = matrix.sum(axis=1)
    rng = np.random.default_rng(seed)
    null_max = np.empty(draws, dtype=float)
    for draw in range(draws):
        signs = rng.choice(np.array([-1.0, 1.0]), size=width)
        null_max[draw] = np.max(matrix @ signs)
    adjusted = {cid: float((1 + int((null_max >= observed[i]).sum())) / (draws + 1)) for i, cid in enumerate(ids)}
    return {"method": "shared_weekly_signflip_max_t", "candidate_count": len(ids), "draws": draws, "seed": seed, "adjusted_pvalues": adjusted}


def build_overlap_allowed_config(selection: Mapping[str, Any] | None = None, cfg: OptimizerConfig = OptimizerConfig()) -> dict[str, Any]:
    sleeve_weights = dict(selection.get("sleeve_weights", {})) if selection else {}
    core: dict[str, Any] = {
        "name": "gross9_overlap_allowed_portfolio_2026_09_03",
        "policy_id": POLICY_ID,
        "status": "train_selected_shadow_only_not_live" if selection else "optimizer_protocol_shadow_only_not_live",
        "as_of": AS_OF_DATE,
        "shadow_only": True,
        "live_capital_authorized": False,
        "order_submission_enabled": False,
        "overlap_policy": {
            "inter_sleeve_overlap_allowed": True,
            "intra_sleeve_overlap_allowed": False,
            "gross_risk_nets_opposite_sides": False,
            "near_6h_gross9_overlap": "diagnostic_not_exclusion_gate_for_this_replacement_portfolio_objective",
        },
        "execution": {
            "position_model": "fixed quantity per sleeve until exit",
            "same_timestamp_transition": "exit then simultaneous entry, aggregate quantity delta netted",
            "base_cost_each_notional_side_bp": BASE_COST_BP,
            "stress_cost_each_notional_side_bp": STRESS_COST_BP,
            "funding": "exact settlement on post-transition aggregate net quantity",
            "renormalization": False,
            "volatility_targeting": False,
        },
        "optimizer": {
            "proxy_window": list(TRAIN_PROXY_WINDOW),
            "december_holdout_unopened": True,
            "holdout_window": list(DECEMBER_HOLDOUT_WINDOW),
            "weight_grid": list(cfg.weight_grid),
            "gross_grid": list(cfg.gross_grid),
            "min_gross": cfg.min_gross,
            "max_gross": cfg.max_gross,
            "max_sleeves": cfg.max_sleeves,
            "beam_width": cfg.beam_width,
            "proxy_candidate_cap": cfg.proxy_candidate_cap,
            "exact_finalist_count": cfg.exact_finalist_count,
            "selection_rule": "fast vectorized proxy beam on Jul-Nov, exact aggregate-net ledger replay of top64, then deterministic raw rank1 among gate-passing finalists",
        },
        "risk_caps": {
            "gross_exposure_cap": cfg.max_gross,
            "mean_gross_exposure_cap": cfg.max_mean_gross_exposure,
            "max_turnover_weight_per_day": cfg.max_turnover_weight_per_day,
            "max_month_share": cfg.max_month_share,
            "gross_risk_does_not_net": True,
        },
        "sleeve_weights": sleeve_weights,
        "selected_gross_exposure": float(sum(abs(float(v)) for v in sleeve_weights.values())),
        "evidence_boundary": {
            "train_proxy_jul_nov_opened_by_selection": bool(selection),
            "december_holdout_opened_by_selection": False,
            "oos_2024_2025_2026_opened_by_selection": False,
            "repair_after_holdout_or_oos_failure_authorized": False,
        },
    }
    core["protocol_hash"] = canonical_hash(core)
    return core


def optimize_from_manifest(
    manifest_path: str | Path,
    market: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    cfg: OptimizerConfig = OptimizerConfig(),
    output: str | Path | None = None,
    config_output: str | Path | None = None,
    preregistration_receipt: Mapping[str, Any] | None = None,
    source_receipts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (output is not None or config_output is not None) and (preregistration_receipt is None or source_receipts is None):
        raise RuntimeError(f"{POLICY_ID} persistent selection requires frozen preregistration and source receipts")
    manifest_path = Path(manifest_path)
    manifest = load_universe_manifest(manifest_path)
    root = Path.cwd()
    start = _utc(TRAIN_PROXY_WINDOW[0]); end = _utc(TRAIN_PROXY_WINDOW[1])
    market = market.copy()
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise")
    for column in ("open", "high", "low", "close"):
        market[column] = pd.to_numeric(market[column], errors="raise")
    market = market.loc[market["date"].ge(start) & market["date"].le(end)].reset_index(drop=True)
    funding = funding.copy()
    funding["date"] = pd.to_datetime(funding["date"], utc=True, errors="raise")
    for column in ("funding_rate", "mark_price"):
        funding[column] = pd.to_numeric(funding[column], errors="raise")
    funding = funding.loc[funding["date"].ge(start) & funding["date"].lt(end)].reset_index(drop=True)
    fixed_ledger.validate_market(market, start, end)
    fixed_ledger.validate_funding(funding, start, end)
    sleeves = [load_sleeve_clock(row, "train", start, end, root=root) for row in manifest["sleeves"]]
    unique_sleeves = deduplicate_sleeve_clocks(sleeves)
    sleeve_map = {s.sleeve_id: s for s in unique_sleeves}
    market_opens = {
        pd.Timestamp(timestamp): float(price)
        for timestamp, price in zip(market["date"], market["open"], strict=True)
    }
    effects = {s.sleeve_id: sleeve_proxy_series(s, market_opens) for s in unique_sleeves}
    proxy_ranked = beam_search_portfolios(effects, cfg)
    evaluated = evaluate_exact_finalists(proxy_ranked, sleeve_map, market, funding, start, end, cfg)
    winner = select_authoritative_rank1(evaluated)
    config = build_overlap_allowed_config(winner, cfg)
    core = {
        "protocol_version": PROTOCOL_VERSION,
        "policy_id": POLICY_ID,
        "as_of_date": AS_OF_DATE,
        "universe_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path), "manifest_hash": manifest.get("manifest_hash")},
        "preregistration": dict(preregistration_receipt or {}),
        "selection_sources": dict(source_receipts or {}),
        "selection_window": [_iso_z(start), _iso_z(end)],
        "december_holdout_unopened": True,
        "oos_unopened": True,
        "sleeve_count_loaded": len(sleeves),
        "unique_schedule_sleeve_count": len(unique_sleeves),
        "duplicate_schedule_sleeves_removed": len(sleeves) - len(unique_sleeves),
        "proxy_portfolios_evaluated": len(proxy_ranked),
        "exact_finalists_evaluated": len(evaluated),
        "authoritative_rank1": winner,
        "shadow_config": config,
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    if config_output is not None:
        Path(config_output).parent.mkdir(parents=True, exist_ok=True)
        Path(config_output).write_text(json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return result


def run_frozen(
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
    universe_path: Path = DEFAULT_UNIVERSE,
    output: Path = DEFAULT_OUTPUT,
    config_output: Path = DEFAULT_CONFIG_OUTPUT,
) -> dict[str, Any]:
    preregistration = load_validated_preregistration(preregistration_path)
    expected_universe = preregistration["immutable_universe"]
    if (
        str(universe_path) != expected_universe["path"]
        or sha256_file(universe_path) != expected_universe["sha256"]
    ):
        raise RuntimeError(f"{POLICY_ID} preregistered universe receipt drift")
    start = _utc(TRAIN_PROXY_WINDOW[0])
    end = _utc(TRAIN_PROXY_WINDOW[1])
    market, funding, source = load_bound_selection_sources(start, end)
    receipt = {
        "path": str(preregistration_path),
        "sha256": sha256_file(preregistration_path),
        "manifest_hash": preregistration["manifest_hash"],
    }
    return optimize_from_manifest(
        universe_path,
        market,
        funding,
        output=output,
        config_output=config_output,
        preregistration_receipt=receipt,
        source_receipts=source,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG_OUTPUT)
    args = parser.parse_args(argv)
    result = run_frozen(args.preregistration, args.universe, args.output, args.config_output)
    print(json.dumps({"policy_id": POLICY_ID, "output": str(args.output), "config_output": str(args.config_output), "rank1": result["authoritative_rank1"]["sleeve_weights"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
