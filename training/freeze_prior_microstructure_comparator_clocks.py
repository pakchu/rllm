"""Freeze prior microstructure comparator clocks without loading BAFR artifacts.

This is an independent prior-family reproduction step.  It may use each prior
strategy's completed-bar signal inputs, but it cannot read the BAFR source,
clock, support result, or any BAFR post-entry outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from training.preregister_metaorder_fragmentation_impact_curvature import (
    Candidate as MFICCandidate,
    Config as MFICSourceConfig,
    compute_mfic,
    load_causal_frame,
    nonoverlapping_schedule,
)
from training.preregister_notional_event_topology_fracture import (
    Candidate as NETFCandidate,
    Config as NETFConfig,
    compute_netf,
    nonoverlapping_netf_schedule,
)
from training.search_orderflow_campaign_terminal_absorption_alpha import (
    base_events,
    campaign_signals,
    terminal_absorption_scores,
    terminal_absorption_signals,
)
from training.search_wasserstein_flow_response_strain_alpha import (
    build_response_inputs,
    build_transport_state,
    fit_score_threshold,
    policy_masks,
)


SELECTION_END = pd.Timestamp("2024-01-01")
FIVE_MINUTES = pd.Timedelta(minutes=5)

MFIC_CANDIDATES = (
    MFICCandidate("mfic_fast", 12, 3, 3, 6),
    MFICCandidate("mfic_slow", 24, 6, 6, 12),
)
NETF_CANDIDATES = (
    NETFCandidate("netf_fast", 6, 48),
    NETFCandidate("netf_slow", 12, 96),
)

WFRS_LOOKBACK = 288
WFRS_SCORE_QUANTILE = 0.90
WFRS_HOLD_BARS = 144
WFRS_FLOW_TAIL = 0.2063984418359628
WFRS_SCORE_THRESHOLD = 0.3377891045604261
WFRS_FIT_START = pd.Timestamp("2020-10-15")

TERMINAL_PROFILE = (12, 24, 6)
TERMINAL_CAMPAIGN_LOOKBACK = 144
TERMINAL_CAMPAIGN_MIN_EVENTS = 2
TERMINAL_CAMPAIGN_MAX_OPPOSITE = 1
TERMINAL_MAX_WAIT_BARS = 72
TERMINAL_HOLD_BARS = 72
TERMINAL_ABSORPTION_THRESHOLD = 1.9190149879917735
TERMINAL_FIT_START = pd.Timestamp("2020-06-01")

FROZEN_ARTIFACT_SHA256 = {
    "cbfr": "79b4838ae634efcff705e028a0ddff8b75d28d79180e3ac89f54b9cab7e5005f",
    "mfic": "03bc5b2f67f974efa04715511920701c0db875b8bb4251f2e4c734a591aa80c8",
    "netf": "4062da454fbd83e10e04dc9b3b01d0884277023d68e209219bb1bcc76d38588e",
    "wfrs": "20fd98879850bad447a59331c208fc94383401e2017073c5f730809151515ff0",
    "terminal_absorption": (
        "7e5d877fdfc82eb4690fc5120fc09f142dd20bfe398a559cee12eb98bf4e9500"
    ),
}
FROZEN_IMPLEMENTATION_SHA256 = {
    "training/preregister_metaorder_fragmentation_impact_curvature.py": (
        "51e99dbdc5ba13e6b4ac15e3915ec5b30e36dff89c1e5b31a5f3f7f272f01a59"
    ),
    "training/preregister_notional_event_topology_fracture.py": (
        "2229c93495d246e949dc768860c4df942189a86b6f700183b09ae54c2873c578"
    ),
    "training/search_wasserstein_flow_response_strain_alpha.py": (
        "5d0f2b511811150c57bc9a598614416543848c2a59d2c3e45ddb2ec0004ac553"
    ),
    "training/search_orderflow_campaign_terminal_absorption_alpha.py": (
        "d7291191531cb607ae3c2704aa37028e889fc4e68cfbc88bf53ef309536b4110"
    ),
    "training/search_orderflow_trophic_campaign_alpha.py": (
        "80e5d6f54dc370abba6862fdb9fa91763f91a8389131c49bc2aed1eaf392d461"
    ),
    "training/search_orderflow_trophic_succession_alpha.py": (
        "b3dac0dc7c85f8af794c8bb11b96b99a90c87a20be3722559ead7eb8ac933a6b"
    ),
}
EXPECTED_COMPARATOR_CLOCKS = {
    "cbfr72": (144, "087560a60cb610bfb3b6963022cc5039c6d8b932950aa8fb83fa289089813dc0"),
    "mfic_fast": (1566, "18361c909f7f93451c193713b5de29a5d49262176a88c086fca6bee26de20256"),
    "mfic_slow": (1635, "825740a533e8dc6564a0bfa4dceb9d20df476ab4637d8d78542fb592d4c27cf1"),
    "mfic_union": (3019, "487a5eef086468aa670e79b047bed6417eeca420098b21d6ddabdab497a4f3a7"),
    "netf_fast": (319, "27c8b4af37eb154bba4b965a316ae374cca5a11628fea29143aaa01443788e4e"),
    "netf_slow": (267, "689e6cfc82733726c4c5334db1622289293ad9e53e64411c1fbb7412f87b2b8d"),
    "netf_union": (586, "4755c8b6d8c7972db0f445deed7f238a6b606e84bb0cc62c39b78cacc79995f9"),
    "wfrs_l288_q90_h144": (
        278,
        "5115e72b722cf64005ba2903aaf990ad64ac4835732c6aec0d40231a3a4e99a0",
    ),
    "terminal_absorption_wait72_h72": (
        100,
        "f901c9d6209f93186cf9665b7d27f2950e88f93f8e490747d3bf7cc4f04724b3",
    ),
}


@dataclass(frozen=True)
class FreezeConfig:
    cbfr_clock: str = (
        "results/cross_collateral_book_validated_flow_rejection_event_clock_2026-07-18.json"
    )
    mfic_support: str = (
        "results/metaorder_fragmentation_impact_curvature_support_2026-07-14.json"
    )
    netf_support: str = (
        "results/notional_event_topology_fracture_support_2026-07-14.json"
    )
    wfrs_result: str = (
        "results/wasserstein_flow_response_strain_alpha_scan_2026-07-14.json"
    )
    terminal_result: str = (
        "results/orderflow_campaign_terminal_absorption_alpha_scan_2026-07-14.json"
    )
    mfic_features: str = (
        "data/binance_um_aggtrade_microstructure_btc_2020_2023/"
        "BTCUSDT_aggtrade_5m_2020-01-01_2023-12-31.csv.gz"
    )
    mfic_feature_manifest: str = (
        "data/binance_um_aggtrade_microstructure_btc_2020_2023/build_manifest.json"
    )
    mfic_market: str = (
        "data/binance_um_kline_reference_btc_2020_2023/"
        "BTCUSDT_5m_2020-01-01_2023-12-31.csv.gz"
    )
    mfic_market_manifest: str = (
        "data/binance_um_kline_reference_btc_2020_2023/build_manifest.json"
    )
    wfrs_market: str = "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
    terminal_market: str = (
        "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01_oi.csv.gz"
    )
    output: str = "results/prior_microstructure_comparator_clock_bundle_2026-07-20.json"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_sha256(path: str | Path, expected: str, *, label: str) -> str:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{label} hash differs from its frozen value")
    return observed


def normalize_clock(clock: pd.DataFrame) -> pd.DataFrame:
    required = {"signal_date", "side"}
    if not required.issubset(clock.columns):
        raise ValueError(f"clock is missing columns: {sorted(required - set(clock.columns))}")
    output = clock.loc[:, ["signal_date", "side"]].copy()
    output["signal_date"] = pd.to_datetime(
        output["signal_date"], utc=True, errors="raise"
    ).dt.tz_convert(None)
    output["side"] = pd.to_numeric(output["side"], errors="raise").astype(np.int8)
    if not output["side"].isin((-1, 1)).all():
        raise ValueError("clock side must be exactly -1 or +1")
    if output["signal_date"].duplicated().any():
        raise ValueError("clock contains duplicate signal timestamps")
    if not output["signal_date"].is_monotonic_increasing:
        raise ValueError("clock timestamps must be strictly increasing")
    return output.reset_index(drop=True)


def clock_hash(clock: pd.DataFrame) -> str:
    events = clock_events(clock)
    payload = json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def clock_events(clock: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = normalize_clock(clock)
    return [
        {
            "signal_date": row.signal_date.strftime("%Y-%m-%d %H:%M:%S"),
            "side": int(row.side),
        }
        for row in normalized.itertuples(index=False)
    ]


def union_clock(clocks: Iterable[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    combined = pd.concat([normalize_clock(clock) for clock in clocks], ignore_index=True)
    side_count = combined.groupby("signal_date")["side"].nunique()
    conflicts = int(side_count.gt(1).sum())
    combined = (
        combined.sort_values(["signal_date", "side"])
        .drop_duplicates("signal_date", keep="first")
        .reset_index(drop=True)
    )
    return normalize_clock(combined), conflicts


def prior_no_stop_schedule(
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    hold_bars: int,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Reproduce the prior families' `_simulate_no_stop` reservation only."""
    if hold_bars < 1:
        raise ValueError("hold bars must be positive")
    timestamps = pd.to_datetime(dates, utc=True, errors="raise").dt.tz_convert(None)
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("signal timestamps must be unique and monotonic")
    long_mask = np.asarray(long_active, dtype=bool)
    short_mask = np.asarray(short_active, dtype=bool)
    if len(timestamps) != len(long_mask) or long_mask.shape != short_mask.shape:
        raise ValueError("signal arrays must align with timestamps")
    if np.any(long_mask & short_mask):
        raise ValueError("signal cannot be simultaneously long and short")
    period = (
        timestamps.ge(pd.Timestamp(start)) & timestamps.lt(pd.Timestamp(end))
    ).to_numpy(bool)
    candidates = np.flatnonzero(period & (long_mask | short_mask))
    rows: list[dict[str, Any]] = []
    next_position = 0
    for signal_position in candidates:
        if signal_position < next_position:
            continue
        entry_position = int(signal_position + 1)
        exit_position = int(entry_position + hold_bars)
        if exit_position >= len(timestamps) or not period[exit_position]:
            continue
        rows.append(
            {
                "signal_date": timestamps.iloc[signal_position],
                "side": 1 if long_mask[signal_position] else -1,
            }
        )
        next_position = exit_position + 1
    return normalize_clock(pd.DataFrame(rows, columns=["signal_date", "side"]))


def split_no_stop_schedule(
    dates: pd.Series,
    long_active: np.ndarray,
    short_active: np.ndarray,
    *,
    hold_bars: int,
    fit_start: pd.Timestamp,
) -> pd.DataFrame:
    pieces = [
        prior_no_stop_schedule(
            dates,
            long_active,
            short_active,
            hold_bars=hold_bars,
            start=start,
            end=end,
        )
        for start, end in (
            (fit_start, pd.Timestamp("2023-01-01")),
            (pd.Timestamp("2023-01-01"), SELECTION_END),
        )
    ]
    return normalize_clock(pd.concat(pieces, ignore_index=True))


def _load_signal_market(
    path: str | Path,
    *,
    columns: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    requested = tuple(dict.fromkeys(("date", *columns)))
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=list(requested),
        chunksize=100_000,
    ):
        dates = pd.to_datetime(
            chunk["date"], format="mixed", utc=True, errors="raise"
        ).dt.tz_convert(None)
        before = dates.lt(SELECTION_END)
        if before.any():
            selected = chunk.loc[before].copy()
            selected["date"] = dates.loc[before].to_numpy()
            pieces.append(selected)
        if (~before).any():
            break
    if not pieces:
        raise ValueError("signal market contains no pre-2024 rows")
    market = (
        pd.concat(pieces, ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    dates = market["date"]
    if dates.max() >= SELECTION_END:
        raise RuntimeError("future rows entered the prior signal frame")
    deltas = dates.diff().dropna()
    if len(deltas) and not deltas.eq(FIVE_MINUTES).all():
        raise ValueError("prior signal market must form a complete five-minute grid")
    return market, {
        "path": str(path),
        "sha256": _sha256(path),
        "columns_loaded": list(requested),
        "future_outcome_transform_applied": False,
        "first_date": str(dates.min()),
        "last_date": str(dates.max()),
        "rows": int(len(market)),
    }


def _load_cbfr(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    expected_identity = {
        "protocol": "CBFR-72 canonical outcome-blind event-clock freeze",
        "selection_end_exclusive": "2024-01-01 00:00:00",
        "event_count": 144,
        "event_clock_sha256": (
            "d2cdcad8f57867722c220e32029d0ccbf1f1aa511e5ae590cf43411a588af4bd"
        ),
        "entry_or_later_ohlc_loaded": False,
        "post_entry_outcomes_opened": False,
    }
    for key, expected in expected_identity.items():
        if payload.get(key) != expected:
            raise ValueError(f"CBFR identity mismatch: {key}")
    clock = normalize_clock(pd.DataFrame(payload.get("events", [])))
    if len(clock) != expected_identity["event_count"]:
        raise ValueError("CBFR event list count differs from frozen identity")
    return clock, {
        "path": str(path),
        "sha256": _sha256(path),
        "rows": int(len(clock)),
        "clock_sha256": clock_hash(clock),
    }


def _descriptor(
    clock: pd.DataFrame,
    *,
    family: str,
    coverage_start: pd.Timestamp,
    candidate: dict[str, Any] | None = None,
    union_side_conflicts: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "family": family,
        "coverage_start_inclusive": str(coverage_start),
        "coverage_end_exclusive": str(SELECTION_END),
        "clock_rows": int(len(clock)),
        "clock_sha256": clock_hash(clock),
        "events": clock_events(clock),
    }
    if candidate is not None:
        result["candidate"] = candidate
    if union_side_conflicts is not None:
        result["union_side_conflicts"] = union_side_conflicts
    return result


def build_bundle(cfg: FreezeConfig) -> dict[str, Any]:
    artifact_paths = {
        "cbfr": cfg.cbfr_clock,
        "mfic": cfg.mfic_support,
        "netf": cfg.netf_support,
        "wfrs": cfg.wfrs_result,
        "terminal_absorption": cfg.terminal_result,
    }
    frozen_artifacts = {
        name: {
            "path": path,
            "sha256": _require_sha256(
                path, FROZEN_ARTIFACT_SHA256[name], label=name
            ),
        }
        for name, path in artifact_paths.items()
    }
    implementation_hashes = {
        path: _require_sha256(path, expected, label=path)
        for path, expected in FROZEN_IMPLEMENTATION_SHA256.items()
    }

    comparators: dict[str, dict[str, Any]] = {}
    sources: dict[str, Any] = {
        "frozen_artifacts": frozen_artifacts,
        "implementation_hashes": implementation_hashes,
    }

    cbfr, sources["cbfr"] = _load_cbfr(cfg.cbfr_clock)
    comparators["cbfr72"] = _descriptor(
        cbfr,
        family="CBFR",
        coverage_start=pd.Timestamp("2023-01-01"),
    )

    mfic_cfg = MFICSourceConfig(
        features=cfg.mfic_features,
        feature_manifest=cfg.mfic_feature_manifest,
        market=cfg.mfic_market,
        market_manifest=cfg.mfic_market_manifest,
    )
    mfic_frame, mfic_source = load_causal_frame(mfic_cfg)
    mfic_clocks: list[pd.DataFrame] = []
    for candidate in MFIC_CANDIDATES:
        signal = compute_mfic(mfic_frame, candidate, mfic_cfg)
        clock = normalize_clock(nonoverlapping_schedule(signal, mfic_frame))
        mfic_clocks.append(clock)
        comparators[candidate.name] = _descriptor(
            clock,
            family="MFIC",
            coverage_start=mfic_frame["date"].min(),
            candidate=asdict(candidate),
        )
    mfic_union, conflicts = union_clock(mfic_clocks)
    comparators["mfic_union"] = _descriptor(
        mfic_union,
        family="MFIC",
        coverage_start=mfic_frame["date"].min(),
        union_side_conflicts=conflicts,
    )
    sources["mfic"] = {
        **mfic_source,
        "features_path": cfg.mfic_features,
        "feature_manifest_path": cfg.mfic_feature_manifest,
        "market_path": cfg.mfic_market,
        "market_manifest_path": cfg.mfic_market_manifest,
    }

    netf_cfg = NETFConfig()
    netf_clocks: list[pd.DataFrame] = []
    for candidate in NETF_CANDIDATES:
        signal = compute_netf(mfic_frame, candidate, netf_cfg)
        clock = normalize_clock(nonoverlapping_netf_schedule(signal, mfic_frame))
        netf_clocks.append(clock)
        comparators[candidate.name] = _descriptor(
            clock,
            family="NETF",
            coverage_start=mfic_frame["date"].min(),
            candidate=asdict(candidate),
        )
    netf_union, conflicts = union_clock(netf_clocks)
    comparators["netf_union"] = _descriptor(
        netf_union,
        family="NETF",
        coverage_start=mfic_frame["date"].min(),
        union_side_conflicts=conflicts,
    )
    sources["netf"] = {"config": asdict(netf_cfg), "shared_source": "mfic"}

    wfrs_market, sources["wfrs"] = _load_signal_market(
        cfg.wfrs_market,
        columns=("open", "close", "quote_asset_volume", "taker_buy_quote"),
    )
    wfrs_dates = wfrs_market["date"]
    inputs, flow_tail = build_response_inputs(wfrs_market, wfrs_dates)
    if not np.isclose(flow_tail, WFRS_FLOW_TAIL, rtol=0.0, atol=1e-12):
        raise ValueError("WFRS flow-tail threshold differs from frozen value")
    state = build_transport_state(inputs, lookback=WFRS_LOOKBACK, flow_tail=flow_tail)
    score_threshold = fit_score_threshold(
        state["score"].to_numpy(float), wfrs_dates, WFRS_SCORE_QUANTILE
    )
    if not np.isclose(score_threshold, WFRS_SCORE_THRESHOLD, rtol=0.0, atol=1e-12):
        raise ValueError("WFRS score threshold differs from frozen value")
    wfrs_long, wfrs_short = policy_masks(
        state["score"].to_numpy(float),
        state["decision"].to_numpy(bool),
        score_threshold,
    )
    wfrs_clock = split_no_stop_schedule(
        wfrs_dates,
        wfrs_long,
        wfrs_short,
        hold_bars=WFRS_HOLD_BARS,
        fit_start=WFRS_FIT_START,
    )
    comparators["wfrs_l288_q90_h144"] = _descriptor(
        wfrs_clock,
        family="WFRS",
        coverage_start=WFRS_FIT_START,
        candidate={
            "lookback": WFRS_LOOKBACK,
            "score_quantile": WFRS_SCORE_QUANTILE,
            "hold_bars": WFRS_HOLD_BARS,
            "flow_tail": flow_tail,
            "score_threshold": score_threshold,
            "split_reservation_reset": "2023-01-01",
        },
    )

    terminal_market, sources["terminal_absorption"] = _load_signal_market(
        cfg.terminal_market,
        columns=(
            "close",
            "high",
            "low",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_quote",
        ),
    )
    terminal_dates = terminal_market["date"]
    event_long, event_short, features, thresholds = base_events(
        terminal_market, terminal_dates, TERMINAL_PROFILE
    )
    if not np.isclose(
        thresholds["absorption_role"],
        TERMINAL_ABSORPTION_THRESHOLD,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("terminal-absorption threshold differs from frozen value")
    campaign_long, campaign_short, _ = campaign_signals(
        event_long,
        event_short,
        lookback_bars=TERMINAL_CAMPAIGN_LOOKBACK,
        min_same_events=TERMINAL_CAMPAIGN_MIN_EVENTS,
        max_opposite_events=TERMINAL_CAMPAIGN_MAX_OPPOSITE,
    )
    long_score, short_score = terminal_absorption_scores(features)
    terminal_long, terminal_short, _ = terminal_absorption_signals(
        campaign_long,
        campaign_short,
        long_score,
        short_score,
        threshold=thresholds["absorption_role"],
        max_wait_bars=TERMINAL_MAX_WAIT_BARS,
    )
    terminal_clock = split_no_stop_schedule(
        terminal_dates,
        terminal_long,
        terminal_short,
        hold_bars=TERMINAL_HOLD_BARS,
        fit_start=TERMINAL_FIT_START,
    )
    comparators["terminal_absorption_wait72_h72"] = _descriptor(
        terminal_clock,
        family="terminal_absorption",
        coverage_start=TERMINAL_FIT_START,
        candidate={
            "profile": list(TERMINAL_PROFILE),
            "campaign_lookback": TERMINAL_CAMPAIGN_LOOKBACK,
            "campaign_min_events": TERMINAL_CAMPAIGN_MIN_EVENTS,
            "campaign_max_opposite": TERMINAL_CAMPAIGN_MAX_OPPOSITE,
            "max_wait_bars": TERMINAL_MAX_WAIT_BARS,
            "hold_bars": TERMINAL_HOLD_BARS,
            "absorption_threshold": thresholds["absorption_role"],
            "split_reservation_reset": "2023-01-01",
        },
    )

    if tuple(comparators) != tuple(EXPECTED_COMPARATOR_CLOCKS):
        raise ValueError("comparator clock set or order differs from freeze")
    for name, (expected_rows, expected_hash) in EXPECTED_COMPARATOR_CLOCKS.items():
        descriptor = comparators[name]
        if descriptor["clock_rows"] != expected_rows:
            raise ValueError(f"{name} clock row count differs from freeze")
        if descriptor["clock_sha256"] != expected_hash:
            raise ValueError(f"{name} canonical clock hash differs from freeze")

    return {
        "as_of": "2026-07-20",
        "stage": "prior_comparator_clock_freeze",
        "protocol": {
            "bafr_source_loaded": False,
            "bafr_clock_loaded": False,
            "bafr_support_loaded": False,
            "bafr_outcomes_opened": False,
            "post_entry_outcomes_computed": False,
            "output_fields": ["signal_date", "side"],
            "source_cutoff_exclusive": str(SELECTION_END),
            "prior_market_usage": "only causal completed-bar inputs required to reproduce already-frozen prior-family signals",
            "scheduling": "MFIC/NETF use their support schedules; WFRS/terminal reproduce each prior no-stop fit/select reservation separately",
        },
        "config": asdict(cfg),
        "sources": sources,
        "comparators": comparators,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for field in FreezeConfig.__dataclass_fields__.values():
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            default=getattr(FreezeConfig, field.name),
        )
    cfg = FreezeConfig(**vars(parser.parse_args()))
    result = build_bundle(cfg)
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "bafr_source_loaded": result["protocol"]["bafr_source_loaded"],
                "post_entry_outcomes_computed": result["protocol"][
                    "post_entry_outcomes_computed"
                ],
                "comparators": {
                    name: {
                        "clock_rows": value["clock_rows"],
                        "clock_sha256": value["clock_sha256"],
                    }
                    for name, value in result["comparators"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
