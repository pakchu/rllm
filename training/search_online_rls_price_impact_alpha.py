"""Preflight a causal online-RLS price-impact alpha without opening OOS.

The model recursively estimates completed-bar BTC return from contemporaneous
taker quote imbalance.  Signals use the *pre-update* impact slope and the
current completed-bar residual standardized only by residual history through
the previous bar.  Exactly eight policies are fixed before inspecting 2024+.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.search_funding_premium_external_state_gate_alpha import (
    _frame_hash,
    _manifest_core_hash,
    _validate_manifest,
)
from training.search_positioning_hgb_path_alpha import _feature_hash, _read_before
from training.search_premium_intrabar_shape_alpha import WINDOWS, _fmt, _stats, _window_mask
from training.search_spot_perp_absorption_alpha import SELECTION_END, _make_event


HALF_LIVES = (576, 2016)
RESIDUAL_Z_THRESHOLD = 1.5
SLOPE_QUANTILES = (0.20, 0.80)
INITIAL_COVARIANCE = 1_000.0
INITIAL_RESIDUAL_VARIANCE = 1e-6
MIN_RESIDUAL_VARIANCE = 1e-12


@dataclass(frozen=True)
class OnlineRlsImpactConfig:
    input_csv: str
    output: str
    manifest_output: str
    docs_output: str
    leverage: float = 0.5
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0001
    refresh_manifest: bool = False


def _load_market(cfg: OnlineRlsImpactConfig, *, cutoff: str = SELECTION_END) -> tuple[pd.DataFrame, dict[str, str]]:
    raw = _read_before(cfg.input_csv, "date", cutoff)
    prefix_hashes = {"market": _frame_hash(raw)}
    market = raw.copy()
    market["date"] = pd.to_datetime(market["date"], utc=True, errors="raise", format="mixed").dt.tz_convert(None)
    market = market.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    required = {"open", "high", "low", "close", "quote_asset_volume", "taker_buy_quote"}
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(f"online-RLS market data lacks columns: {missing}")
    dates = pd.to_datetime(market["date"])
    if len(dates) and dates.max() >= pd.Timestamp(cutoff):
        raise RuntimeError("online-RLS selection frame contains a row at or after the OOS cutoff")
    intervals = dates.diff().dropna()
    if len(intervals) and not intervals.eq(pd.Timedelta("5min")).all():
        raise ValueError("online-RLS search requires a complete 5-minute grid")
    return market, prefix_hashes


def _rls_path(
    returns: np.ndarray,
    imbalance: np.ndarray,
    *,
    half_life: int,
) -> dict[str, np.ndarray]:
    """Return causal pre-update RLS state and prior-scaled residual path."""

    size = len(returns)
    slope = np.full(size, np.nan, dtype=np.float64)
    prediction = np.full(size, np.nan, dtype=np.float64)
    residual = np.full(size, np.nan, dtype=np.float64)
    residual_z = np.full(size, np.nan, dtype=np.float64)
    observations = np.zeros(size, dtype=np.int64)

    forgetting = float(2.0 ** (-1.0 / float(half_life)))
    burn_in = max(6048, 3 * int(half_life))
    theta = np.zeros(2, dtype=np.float64)
    covariance = np.eye(2, dtype=np.float64) * INITIAL_COVARIANCE
    residual_mean = 0.0
    residual_variance = INITIAL_RESIDUAL_VARIANCE
    seen = 0

    for pos, (target, flow) in enumerate(zip(returns, imbalance)):
        if not np.isfinite(target) or not np.isfinite(flow):
            continue
        x = np.array([1.0, float(flow)], dtype=np.float64)

        # These emitted values are the model state through t-1.  In particular,
        # neither the current target nor current residual has updated theta or
        # the residual denominator yet.
        fitted = float(x @ theta)
        error = float(target - fitted)
        slope[pos] = float(theta[1])
        prediction[pos] = fitted
        residual[pos] = error
        observations[pos] = seen
        if seen >= burn_in:
            residual_z[pos] = (error - residual_mean) / np.sqrt(max(residual_variance, MIN_RESIDUAL_VARIANCE))

        px = covariance @ x
        gain = px / (forgetting + float(x @ px))
        theta = theta + gain * error
        covariance = (covariance - np.outer(gain, x) @ covariance) / forgetting

        previous_mean = residual_mean
        residual_mean = forgetting * residual_mean + (1.0 - forgetting) * error
        residual_variance = (
            forgetting * residual_variance
            + (1.0 - forgetting) * (error - previous_mean) * (error - residual_mean)
        )
        residual_variance = max(float(residual_variance), MIN_RESIDUAL_VARIANCE)
        seen += 1

    return {
        "slope": slope,
        "prediction": prediction,
        "residual": residual,
        "residual_z": residual_z,
        "observations": observations,
    }


def build_features(market: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(market["close"], errors="coerce").to_numpy(float)
    quote = pd.to_numeric(market["quote_asset_volume"], errors="coerce").to_numpy(float)
    taker_buy = pd.to_numeric(market["taker_buy_quote"], errors="coerce").to_numpy(float)
    returns = np.full(len(market), np.nan, dtype=np.float64)
    valid_close = np.isfinite(close) & (close > 0.0)
    valid_pair = valid_close[1:] & valid_close[:-1]
    returns[1:][valid_pair] = np.log(close[1:][valid_pair]) - np.log(close[:-1][valid_pair])
    imbalance = np.full(len(market), np.nan, dtype=np.float64)
    valid_quote = np.isfinite(quote) & np.isfinite(taker_buy) & (quote > 0.0)
    imbalance[valid_quote] = np.clip((2.0 * taker_buy[valid_quote] - quote[valid_quote]) / quote[valid_quote], -1.0, 1.0)

    columns: dict[str, np.ndarray] = {
        "rls_log_return": returns,
        "rls_taker_quote_imbalance": imbalance,
    }
    for half_life in HALF_LIVES:
        state = _rls_path(returns, imbalance, half_life=half_life)
        for name, values in state.items():
            columns[f"rls_{name}_{half_life}"] = values
    return pd.DataFrame(columns, index=market.index).replace([np.inf, -np.inf], np.nan).astype(np.float32)


def _fit_quantile(values: pd.Series, fit_mask: np.ndarray, quantile: float) -> float:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    reference = array[fit_mask & np.isfinite(array)]
    if len(reference) < 20_000:
        raise ValueError(f"insufficient fit observations for online-RLS quantile: {len(reference)}")
    return float(np.quantile(reference, quantile))


def _policy_specs(features: pd.DataFrame, fit_mask: np.ndarray) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    schedule = {
        576: {"continuation": (24, 6), "fade": (72, 12)},
        2016: {"continuation": (72, 12), "fade": (144, 12)},
    }
    for half_life in HALF_LIVES:
        slope = features[f"rls_slope_{half_life}"]
        lower = _fit_quantile(slope, fit_mask, SLOPE_QUANTILES[0])
        upper = _fit_quantile(slope, fit_mask, SLOPE_QUANTILES[1])
        for slope_state in ("high", "low"):
            for direction in ("continuation", "fade"):
                hold, stride = schedule[half_life][direction]
                specs.append(
                    {
                        "half_life": half_life,
                        "slope_state": slope_state,
                        "direction": direction,
                        "hold": hold,
                        "stride": stride,
                        "slope_lower": lower,
                        "slope_upper": upper,
                        "residual_z_threshold": RESIDUAL_Z_THRESHOLD,
                    }
                )
    return specs


def _signals(features: pd.DataFrame, spec: dict[str, Any], *, flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    half_life = int(spec["half_life"])
    slope = pd.to_numeric(features[f"rls_slope_{half_life}"], errors="coerce").to_numpy(float)
    residual_z = pd.to_numeric(features[f"rls_residual_z_{half_life}"], errors="coerce").to_numpy(float)
    finite = np.isfinite(slope) & np.isfinite(residual_z)
    if str(spec["slope_state"]) == "high":
        state = slope >= float(spec["slope_upper"])
    elif str(spec["slope_state"]) == "low":
        state = slope <= float(spec["slope_lower"])
    else:
        raise KeyError(spec["slope_state"])
    sampled = np.arange(len(features), dtype=np.int64) % int(spec["stride"]) == 0
    active = finite & state & (np.abs(residual_z) >= float(spec["residual_z_threshold"])) & sampled
    side = np.zeros(len(features), dtype=np.int8)
    side[finite] = np.sign(residual_z[finite]).astype(np.int8)
    if str(spec["direction"]) == "fade":
        side = -side
    elif str(spec["direction"]) != "continuation":
        raise KeyError(spec["direction"])
    if flip:
        side = -side
    side[~active] = 0
    return active, side


def _build_events(
    market: pd.DataFrame,
    features: pd.DataFrame,
    spec: dict[str, Any],
    cfg: OnlineRlsImpactConfig,
    *,
    flip: bool = False,
) -> list[dict[str, Any]]:
    active, sides = _signals(features, spec, flip=flip)
    residual_z = pd.to_numeric(features[f"rls_residual_z_{int(spec['half_life'])}"], errors="coerce").to_numpy(float)
    events: list[dict[str, Any]] = []
    next_allowed = 0
    for pos in np.flatnonzero(active):
        if int(pos) < next_allowed:
            continue
        event = _make_event(
            market,
            residual_z,
            int(pos),
            int(sides[pos]),
            int(np.sign(residual_z[pos])),
            max_hold=int(spec["hold"]),
            dynamic_exit=False,
            exit_abs_z=0.0,
            cost_rate=float(cfg.fee_rate + cfg.slippage_rate),
            leverage=float(cfg.leverage),
            name="online_rls_impact_flip" if flip else "online_rls_impact",
        )
        if event is None:
            continue
        events.append(event)
        next_allowed = int(event["exit_pos"])
    return events


def _selection_score(stats: dict[str, Any]) -> float:
    fit, full = stats["fit"], stats["select_2023"]
    h1, h2 = stats["select_2023_h1"], stats["select_2023_h2"]
    if (
        fit["return_pct"] <= 0.0
        or fit["trades"] < 30
        or full["return_pct"] <= 0.0
        or full["ratio"] < 1.0
        or full["trades"] < 12
        or min(h1["return_pct"], h2["return_pct"]) <= 0.0
        or min(h1["trades"], h2["trades"]) < 4
    ):
        return -1e12
    return float(min(full["ratio"], h1["ratio"], h2["ratio"]) + 0.1 * fit["ratio"] + 0.01 * min(full["trades"], 100))


def _path_hash(events: list[dict[str, Any]], dates: pd.Series, window: str) -> str:
    mask = _window_mask(dates, window)
    positions = np.flatnonzero(mask)
    first, last = int(positions[0]), int(positions[-1]) + 1
    rows = [(event["side"], int(event["signal_pos"]), int(event["exit_pos"])) for event in events if first <= int(event["signal_pos"]) and int(event["exit_pos"]) < last]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _select_manifest(cfg: OnlineRlsImpactConfig) -> dict[str, Any]:
    market, prefix_hashes = _load_market(cfg)
    dates = pd.to_datetime(market["date"])
    features = build_features(market)
    specs = _policy_specs(features, _window_mask(dates, "fit"))
    diagnostics: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    seen: set[str] = set()
    selection_windows = ("fit", "select_2023", "select_2023_h1", "select_2023_h2")
    for spec in specs:
        events = _build_events(market, features, spec, cfg)
        flipped = _build_events(market, features, spec, cfg, flip=True)
        stats = _stats(events, dates, selection_windows)
        score = _selection_score(stats)
        diagnostics.append(
            {
                "spec": spec,
                "selection_score": score,
                "stats": stats,
                "direction_flipped": _stats(flipped, dates, selection_windows),
            }
        )
        if score <= -1e11:
            continue
        event_hash = _path_hash(events, dates, "select_2023")
        if event_hash in seen:
            continue
        seen.add(event_hash)
        eligible.append({"spec": spec, "selection_score": score, "selection_stats": stats, "selection_path_hash": event_hash})
    eligible.sort(key=lambda row: (-row["selection_score"], json.dumps(row["spec"], sort_keys=True)))
    core = {
        "protocol": {
            "hypothesis": "a recursively estimated taker-flow price-impact slope separates informative residual continuation from crowded residual fade",
            "feature": "completed 5m log return regressed online on same completed bar taker quote imbalance",
            "causal_order": "signal uses theta through t-1 and current residual scaled by residual moments through t-1; RLS and moments update only after signal state emission",
            "half_lives": list(HALF_LIVES),
            "burn_in": "max(6048, 3*half_life) valid observations",
            "thresholds": "fit-only slope q20/q80; fixed |residual z| >= 1.5",
            "search_cap": "exactly eight fixed high/low-slope x continuation/fade policies; no grid",
            "selection": {name: WINDOWS[name] for name in selection_windows},
            "post_cutoff_rows_excluded_before_feature_construction": True,
            "source_access_semantics": "the chunked CSV reader may buffer the cutoff-crossing chunk, but no row at or after 2024 enters the returned market frame, features, policy selection, or metrics",
            "later_metrics_included": False,
            "entry_exit": "fixed stride on completed signal bar; next 5m open; fixed hold; global non-overlap",
            "cost": "6bp/side, 0.5x",
            "mdd": "strict entry cost plus intrabar adverse OHLC and realized high-water",
            "status_ceiling": "preflight research",
        },
        "source_prefix_hashes": prefix_hashes,
        "feature_hash": _feature_hash(features, dates),
        "search_space": {"raw_specs": len(specs), "eligible_unique_paths": len(eligible)},
        "preflight_diagnostics": diagnostics,
        "selected": eligible,
    }
    manifest = {"as_of": datetime.now(timezone.utc).isoformat(), "sha256": _manifest_core_hash(core), **core}
    path = Path(cfg.manifest_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    return manifest


def _write_doc(cfg: OnlineRlsImpactConfig, report: dict[str, Any]) -> None:
    lines = [
        "# Online RLS price-impact alpha preflight (2026-07-13)",
        "",
        "Metric format: `absolute return / CAGR / strict MDD / CAGR-MDD / trades`.",
        "",
        "| policy | fit | 2023 | 2023H1 | 2023H2 | flipped 2023 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["preflight_diagnostics"]:
        spec, stats = row["spec"], row["stats"]
        policy = f"H{spec['half_life']} {spec['slope_state']} {spec['direction']}"
        lines.append(
            f"| `{policy}` | {_fmt(stats['fit'])} | {_fmt(stats['select_2023'])} | {_fmt(stats['select_2023_h1'])} | {_fmt(stats['select_2023_h2'])} | {_fmt(row['direction_flipped']['select_2023'])} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"- Eligible fixed policies: **{report['search_space']['eligible_unique_paths']} / {report['search_space']['raw_specs']}**; OOS metrics calculated: **no**.",
        "- Fast/high/fade was the only policy with a meaningful positive fit result, but it had only four 2023 trades and lost slightly; no 2023H2 trade existed.",
        "- Low-impact regimes generated enough trades but lost materially in 2023 in continuation and fade mappings.",
        "- Direction flips did not create fit plus two-half stability. Reject these exact static mappings without spending 2024-2026 evidence.",
        "- The continuous pre-update slope and prior-scaled residual remain beta context only, not an executable alpha.",
        "",
        "## Leakage controls",
        "",
        "- The selection frame is verified to contain no row at or after 2024 before feature construction. The CSV reader may buffer a cutoff-crossing chunk; buffered later rows are filtered before entering the market frame.",
        "- The current target never updates the slope used at that bar; the current residual never updates its own z-score denominator.",
        "- All trades enter at the next 5m open and strict MDD includes intratrade adverse OHLC.",
        "",
        "## Reproduction",
        "",
        "```bash",
        f"PYTHONPATH=. .venv/bin/python -m training.search_online_rls_price_impact_alpha --input-csv {cfg.input_csv} --manifest-output {cfg.manifest_output} --output {cfg.output} --docs-output {cfg.docs_output}",
        "```",
    ]
    path = Path(cfg.docs_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run(cfg: OnlineRlsImpactConfig) -> dict[str, Any]:
    manifest_path = Path(cfg.manifest_output)
    if manifest_path.exists() and not cfg.refresh_manifest:
        manifest = json.loads(manifest_path.read_text())
        _validate_manifest(manifest)
    else:
        manifest = _select_manifest(cfg)
    # This work unit is intentionally preflight-only.  OOS replay is a separate
    # frozen phase and is not reachable unless a pre-2024 policy is admitted.
    report = {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "manifest": cfg.manifest_output,
        "manifest_sha256": manifest["sha256"],
        "protocol": manifest["protocol"],
        "source_prefix_hashes": manifest["source_prefix_hashes"],
        "search_space": manifest["search_space"],
        "preflight_only": True,
        "oos_opened": False,
        "oos_metrics_opened": False,
        "post_cutoff_rows_used": False,
        "preflight_diagnostics": manifest["preflight_diagnostics"],
        "selected": manifest["selected"],
        "alpha_pool_qualifiers": [],
        "live_grade": [],
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n")
    _write_doc(cfg, report)
    return report


def parse_args() -> OnlineRlsImpactConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--docs-output", required=True)
    parser.add_argument("--refresh-manifest", action="store_true")
    return OnlineRlsImpactConfig(**vars(parser.parse_args()))


def main() -> None:
    report = run(parse_args())
    print(json.dumps({"manifest": report["manifest"], "eligible": len(report["selected"]), "oos_opened": report["oos_opened"]}, indent=2))


if __name__ == "__main__":
    main()
