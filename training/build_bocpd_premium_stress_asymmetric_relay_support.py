"""Build outcome-blind source support for the frozen BCPPSR-ASYM relay."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import backtest_all_alpha_month as month
from training import preregister_bocpd_premium_stress_asymmetric_relay as prereg
from training.build_binance_aggtrade_microstructure import _write_gzip_csv
from training.search_bocpd_state_gated_alpha import (
    _map_output,
    _state_from_mapped,
    bocpd_student_t,
)
from training.search_gaussian_hmm_regime_alpha import hourly_features


SOURCE_PANEL = Path(
    "data/bocpd_premium_stress_asymmetric_relay_sources_2023_2026/"
    "source_panel.csv.gz"
)
CLOCK = Path("data/bocpd_premium_stress_asymmetric_relay_clock_2023_2026.csv.gz")
RESULT = Path("results/bocpd_premium_stress_asymmetric_relay_support_2026-08-09.json")
HISTORICAL_CACHE = Path(
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
)
SPLITS = {
    name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
    for name, bounds in prereg.build()["stages"].items()
}
MINIMUM = {"train": 8, "test": 12, "eval": 12, "final": 8}
MINORITY_SIDE_SHARE_MIN = 0.2
MAX_MONTH_SHARE = 0.45
CONTROLS = (
    "long_state_only",
    "short_state_only",
    "no_bocpd_gate",
    "premium_panic_only",
    "kimchi_unwind_only",
    "direction_flip",
)
FEATURES = (
    "funding_rate",
    "trend_96",
    "premium_index_change",
    "htf_1d_return_4",
    "htf_3d_range_pos",
    "premium_index_zscore",
    "htf_3d_return_1",
    "kimchi_premium_change",
)
CLOCK_COLUMNS = (
    "candidate",
    "control",
    "split",
    "decision_time",
    "feature_available_time",
    "entry_time",
    "exit_time",
    "side",
    "state",
)
ECONOMIC_OUTCOMES_AUTHORIZED = False


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_frozen_contract(registration: dict[str, Any]) -> None:
    if registration.get("manifest_hash") != prereg.build()["manifest_hash"]:
        raise RuntimeError("BCPPSR preregistration differs from frozen source code")
    gates = registration["source_support_gates"]
    expected = {
        "minimum_events": MINIMUM,
        "minority_side_share_min": MINORITY_SIDE_SHARE_MIN,
        "max_month_share": MAX_MONTH_SHARE,
    }
    if gates != expected:
        raise RuntimeError("BCPPSR frozen source-support gates drifted")
    if registration["stopping_rule"] != "terminal first failure; no repair":
        raise RuntimeError("BCPPSR frozen terminal stopping rule drifted")


def gate_mask(frame: pd.DataFrame, gates: list[dict[str, Any]]) -> np.ndarray:
    active = np.ones(len(frame), dtype=bool)
    for gate in gates:
        values = pd.to_numeric(
            frame[str(gate["feature"])], errors="coerce"
        ).to_numpy(float)
        finite = np.isfinite(values)
        threshold = float(gate["threshold"])
        if gate["op"] in (">=", "ge"):
            active &= finite & (values >= threshold)
        elif gate["op"] in ("<=", "le"):
            active &= finite & (values <= threshold)
        else:
            raise ValueError(f"unsupported gate operator: {gate['op']}")
    return active


def bocpd_hourly_output(
    hourly_feature: pd.DataFrame, contract: dict[str, Any]
) -> pd.DataFrame:
    """Filter completed hourly inputs with only frozen BOCPD parameters."""
    columns = tuple(str(value) for value in contract["hourly_inputs"])
    good = hourly_feature.loc[:, columns].notna().all(axis=1).to_numpy()
    raw = hourly_feature.loc[good, columns].to_numpy(float)
    if not len(raw):
        raise RuntimeError("BCPPSR has no finite completed hourly BOCPD inputs")
    mean = np.asarray(contract["standardization_mean"], dtype=float)
    std = np.asarray(contract["standardization_std"], dtype=float)
    standardized = ((raw - mean) / std).clip(-12.0, 12.0)
    posterior = bocpd_student_t(
        standardized,
        hazard_lambda=float(contract["hazard_lambda_hours"]),
        max_run_length=int(contract["max_run_length"]),
        prior_kappa=float(contract["prior_kappa"]),
        prior_alpha=float(contract["prior_alpha"]),
        prior_beta=float(contract["prior_beta"]),
        short_run_horizon=int(contract["short_run_horizon_hours"]),
    )
    return pd.DataFrame(
        {
            "date": hourly_feature.index[good],
            "primary": posterior["posterior_mean"][:, 0],
            "short_mass": posterior["short_mass"],
            "run_drop": posterior["run_drop"],
            "secondary": posterior["posterior_mean"][:, columns.index("flow24")],
            "surprise": posterior["surprise"],
        }
    )


def build_bocpd_state(
    market: pd.DataFrame,
    registration: dict[str, Any],
    historical_cache: Path = HISTORICAL_CACHE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extend the bound cache and map completed hourly state backward-as-of."""
    required = (
        "date",
        "open",
        "high",
        "low",
        "close",
        "quote_asset_volume",
        "taker_buy_quote",
    )
    historical = pd.read_csv(historical_cache, compression="infer", usecols=required)
    current = market.loc[:, required].copy()
    combined = pd.concat([historical, current], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], utc=True).dt.tz_convert(None)
    combined = (
        combined[combined["date"] < pd.Timestamp("2026-08-01")]
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if combined["date"].duplicated().any() or not combined["date"].is_monotonic_increasing:
        raise RuntimeError("BCPPSR causal market context time drift")
    _hourly_market, hourly_feature = hourly_features(combined)
    hourly = bocpd_hourly_output(hourly_feature, registration["bocpd_contract"])
    market_dates = pd.to_datetime(market["date"], utc=True).dt.tz_convert(None)
    mapped = _map_output(market_dates, hourly)
    state = _state_from_mapped(
        mapped, registration["bocpd_contract"]["state_thresholds"]
    )
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(market["date"], utc=True),
            "bocpd_primary": mapped["primary"].to_numpy(float),
            "bocpd_short_mass": mapped["short_mass"].to_numpy(float),
            "bocpd_secondary": mapped["secondary"].to_numpy(float),
            "bocpd_state": state,
        }
    )
    return output, {
        "historical_cache": str(historical_cache),
        "historical_cache_sha256": sha256(historical_cache),
        "causal_context_first": str(combined["date"].iloc[0]),
        "causal_context_last": str(combined["date"].iloc[-1]),
        "completed_hourly_first": str(hourly["date"].iloc[0]),
        "completed_hourly_last": str(hourly["date"].iloc[-1]),
        "filter_restarted_at_support_boundary": False,
    }


def state_signals(
    panel: pd.DataFrame, control: str = "primary"
) -> tuple[np.ndarray, np.ndarray]:
    registration = prereg.build()
    states = registration["frozen_states"]
    long_components = states["long_bocpd_funding_premium"]["components"]
    long_signal = np.zeros(len(panel), dtype=bool)
    for gates in long_components.values():
        long_signal |= gate_mask(panel, list(gates))
    if control != "no_bocpd_gate":
        allowed = np.asarray(registration["bocpd_contract"]["allowed_states"], dtype=int)
        state = pd.to_numeric(panel["bocpd_state"], errors="coerce").to_numpy(float)
        long_signal &= np.isin(state, allowed)

    short_components = states["short_premium_kimchi_stress"]["components"]
    if control == "premium_panic_only":
        short_components = {"premium_panic": short_components["premium_panic"]}
    elif control == "kimchi_unwind_only":
        short_components = {"kimchi_unwind": short_components["kimchi_unwind"]}
    short_signal = np.zeros(len(panel), dtype=bool)
    for gates in short_components.values():
        short_signal |= gate_mask(panel, list(gates))

    # Frozen hourly decision grid: the 23:55 bar is completed at 00:00, etc.
    dates = pd.to_datetime(panel["date"], utc=True)
    decision_grid = dates.dt.minute.eq(55).to_numpy()
    long_signal &= decision_grid
    short_signal &= decision_grid
    if control == "long_state_only":
        short_signal[:] = False
    elif control == "short_state_only":
        long_signal[:] = False
    return long_signal, short_signal


def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    long_signal, short_signal = state_signals(panel, control)
    long_signal = np.asarray(long_signal, dtype=bool)
    short_signal = np.asarray(short_signal, dtype=bool)
    dates = pd.to_datetime(panel["date"], utc=True)
    rows: list[dict[str, Any]] = []
    next_allowed: pd.Timestamp | None = None
    for position in np.flatnonzero(long_signal | short_signal):
        if long_signal[position] and short_signal[position]:
            continue
        decision = dates.iloc[position] + pd.Timedelta(minutes=5)
        entry = decision
        side = 1 if long_signal[position] else -1
        exit_time = entry + pd.Timedelta(hours=48 if side == 1 else 24)
        if next_allowed is not None and entry < next_allowed:
            continue
        split = next(
            (
                name
                for name, (start, end) in SPLITS.items()
                if entry >= start and exit_time <= end
            ),
            None,
        )
        if split is None:
            continue
        next_allowed = exit_time
        rows.append(
            {
                "candidate": "BCPPSR-ASYM",
                "control": control,
                "split": split,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": entry,
                "exit_time": exit_time,
                "side": -side if control == "direction_flip" else side,
                "state": (
                    "long_bocpd_funding_premium"
                    if side == 1
                    else "short_premium_kimchi_stress"
                ),
            }
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def query_source_panel(
    registration: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = month.Config(
        env_path=Path(registration["database_contract"]["env_file"]),
        start="2023-06-20T00:00:00Z",
        end="2026-08-01T00:00:00Z",
        asof="2026-08-01T00:02:00Z",
        lookback_minutes=1_650_000,
    )
    market, features, _funding, engine = asyncio.run(month._query_frames(cfg))
    if engine is not None:
        engine.dispose()
    missing = sorted(set(FEATURES) - set(features.columns))
    if missing:
        raise RuntimeError(f"BCPPSR source columns missing: {missing}")
    dates = pd.to_datetime(market["date"], utc=True)
    keep = (dates >= pd.Timestamp("2023-06-20T00:00:00Z")) & (
        dates < pd.Timestamp("2026-08-01T00:00:00Z")
    )
    market = market.loc[keep].reset_index(drop=True)
    features = features.loc[keep].reset_index(drop=True)
    dates = pd.to_datetime(market["date"], utc=True)
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise RuntimeError("BCPPSR source panel time drift")
    if not dates.diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError("BCPPSR source panel is not a complete 5-minute grid")
    bocpd, context = build_bocpd_state(market, registration)
    panel = pd.DataFrame(
        {
            "date": dates,
            **{
                name: pd.to_numeric(features[name], errors="coerce")
                for name in FEATURES
            },
            **{name: bocpd[name] for name in bocpd.columns if name != "date"},
        }
    )
    return panel, {
        "mode": "bound_cache_plus_read_only_postgres_completed_sources",
        "read_only": True,
        "rows": len(panel),
        "first": str(panel["date"].iloc[0]),
        "last": str(panel["date"].iloc[-1]),
        "end_exclusive": "2026-08-01 00:00:00+00:00",
        "signal_dependent_tables": registration["database_contract"][
            "live_extension_tables"
        ],
        "external_backward_asof_feature": "kimchi_premium_change",
        **context,
    }


def stats(clock: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = clock[clock["split"].eq(split)]
    if rows.empty:
        return {
            "events": 0,
            "longs": 0,
            "shorts": 0,
            "minority_side_share": 0.0,
            "max_month_share": 0.0,
        }
    longs = int(rows["side"].eq(1).sum())
    shorts = int(rows["side"].eq(-1).sum())
    months = pd.to_datetime(rows["entry_time"], utc=True).dt.strftime("%Y-%m")
    return {
        "events": int(len(rows)),
        "longs": longs,
        "shorts": shorts,
        "minority_side_share": min(longs, shorts) / len(rows),
        "max_month_share": int(months.value_counts().max()) / len(rows),
    }


def support_verdict(clock: pd.DataFrame) -> tuple[dict[str, Any], dict[str, bool], bool]:
    support = {name: stats(clock, name) for name in SPLITS}
    checks: dict[str, bool] = {}
    for name, item in support.items():
        checks[f"{name}_minimum_events"] = item["events"] >= MINIMUM[name]
        checks[f"{name}_side_balance"] = (
            item["minority_side_share"] >= MINORITY_SIDE_SHARE_MIN
        )
        checks[f"{name}_month_concentration"] = (
            item["max_month_share"] <= MAX_MONTH_SHARE
        )
    return support, checks, all(checks.values())


def run() -> dict[str, Any]:
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    prereg.validate(registration)
    validate_frozen_contract(registration)
    panel, source = query_source_panel(registration)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    support, checks, passed = support_verdict(primary)

    SOURCE_PANEL.parent.mkdir(parents=True, exist_ok=True)
    CLOCK.parent.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_csv(panel, SOURCE_PANEL)
    _write_gzip_csv(primary, CLOCK)
    core = {
        "protocol_version": "bcppsr_asym_source_support_v1",
        "policy_id": "BCPPSR-ASYM",
        "preregistration": {
            "path": str(prereg.DEFAULT_OUTPUT),
            "sha256": sha256(prereg.DEFAULT_OUTPUT),
            "manifest_hash": registration["manifest_hash"],
        },
        "source": source,
        "source_panel": {
            "path": str(SOURCE_PANEL),
            "sha256": sha256(SOURCE_PANEL),
            "rows": len(panel),
        },
        "clock": {"path": str(CLOCK), "sha256": sha256(CLOCK), "rows": len(primary)},
        "completed_preentry_sources_opened": True,
        "btc_postentry_return_or_pnl_opened": False,
        "gross9_rows_opened": False,
        "economics_opened": False,
        "controls": {
            name: {"rows": len(frame), "promotion_authorized": False}
            for name, frame in controls.items()
        },
        "support_gates": {
            "minimum_events": MINIMUM,
            "minority_side_share_min": MINORITY_SIDE_SHARE_MIN,
            "max_month_share": MAX_MONTH_SHARE,
        },
        "support": support,
        "support_checks": checks,
        "support_passed": passed,
        "terminal_without_repair": not passed,
        "repair_authorized": False,
        "advance_to_gross9_novelty": passed,
        "advance_to_economic_outcomes": ECONOMIC_OUTCOMES_AUTHORIZED,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    RESULT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    report = run()
    print(json.dumps({"passed": report["support_passed"], "support": report["support"]}, indent=2))
