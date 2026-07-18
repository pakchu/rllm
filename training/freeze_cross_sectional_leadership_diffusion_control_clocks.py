"""Freeze outcome-blind mechanism-control clocks for CLD-72."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from training import preregister_cross_sectional_leadership_diffusion as cld


SUPPORT = Path("results/cross_sectional_leadership_diffusion_support_2026-07-18.json")
PRIMARY_CLOCK = Path(
    "results/cross_sectional_leadership_diffusion_event_clock_2026-07-18.json"
)
OUTPUT = Path(
    "results/cross_sectional_leadership_diffusion_control_clocks_2026-07-18.json"
)
SUPPORT_SHA256 = "e2e23be7504473edc0d5df44b5a25d2fa2ec6f82770206cf35bdf9ca66e020dc"
PRIMARY_CLOCK_SHA256 = "089ae3f854459a76bade4e3fd6682d1b1a9a6d600dc990a367840c179c0e623d"
PRIMARY_EVENT_CLOCK_HASH = "dcbed47f339ff8f602008ed4cdad482f2b9fcc73dc522ac3411014ca1420396e"
PREREGISTRATION_SOURCE_SHA256 = (
    "c85201b86da38a28f79885b60b4cfa0c132f6bc892e12f34b8d33118dde871c5"
)
SELECTED = {
    "move_quantile": 0.6,
    "prior_hhi_quantile": 0.6,
    "maximum_hhi_ratio": 0.9,
    "minimum_participation": 5 / 6,
    "minimum_flow_alignment": 4 / 6,
    "turnover_quantile": 0.5,
    "leader_decline_quantile": 0.6,
}


def _load_and_verify_support() -> tuple[dict[str, Any], dict[str, Any]]:
    if cld.sha256(cld.PREREGISTRATION_SOURCE) != PREREGISTRATION_SOURCE_SHA256:
        raise RuntimeError("CLD preregistration source changed after support freeze")
    if cld.sha256(SUPPORT) != SUPPORT_SHA256:
        raise RuntimeError("CLD support artifact changed")
    if cld.sha256(PRIMARY_CLOCK) != PRIMARY_CLOCK_SHA256:
        raise RuntimeError("CLD primary clock artifact changed")
    support = json.loads(SUPPORT.read_text())
    support_body = {key: value for key, value in support.items() if key != "manifest_hash"}
    if cld.canonical_hash(support_body) != support.get("manifest_hash"):
        raise RuntimeError("CLD support manifest hash changed")
    if support.get("all_support_gates_pass") is not True:
        raise RuntimeError("CLD support did not pass")
    if support.get("protocol", {}).get("evidence_boundary", {}).get(
        "post_entry_outcomes_opened"
    ) is not False:
        raise RuntimeError("CLD support opened an outcome")
    selected = support["support_selection"]["selected_cell"]
    if {key: selected[key] for key in SELECTED} != SELECTED:
        raise RuntimeError("CLD selected support cell changed")
    primary = json.loads(PRIMARY_CLOCK.read_text())
    if primary.get("post_entry_outcomes_opened") is not False:
        raise RuntimeError("CLD primary clock opened an outcome")
    if primary.get("event_clock_sha256") != PRIMARY_EVENT_CLOCK_HASH:
        raise RuntimeError("CLD primary event hash changed")
    if cld.canonical_hash(primary["events"]) != PRIMARY_EVENT_CLOCK_HASH:
        raise RuntimeError("CLD primary event records changed")
    return support, primary


def _base_transition_mask(panel: pd.DataFrame) -> pd.Series:
    return (
        panel["clean"]
        & panel["median_return_abs"].ge(panel["move_q60"])
        & panel["prior_residual_hhi"].ge(panel["prior_hhi_q60"])
        & panel["hhi_ratio"].le(0.9)
        & panel["participation"].ge(5 / 6)
        & panel["rank_turnover"].ge(panel["turnover_q50"])
        & panel["leader_decline"].ge(panel["leader_decline_q60"])
        & panel["leader_changed"]
    )


def _signal_from_mask(
    panel: pd.DataFrame,
    mask: pd.Series,
    *,
    branch: str,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    sides = pd.Series(0, index=panel.index, dtype=np.int8)
    chosen = panel["direction"] if side is None else side
    sides.loc[mask] = chosen.loc[mask].astype(np.int8)
    return pd.DataFrame(
        {
            "signal_date": panel["signal_date"],
            "feature_boundary": panel["feature_boundary"],
            "entry_date": panel["entry_date"],
            "side": sides,
            "branch": np.where(sides.ne(0), branch, "none"),
        }
    )


def _canonical_events(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    return cld._event_clock_payload(schedule)["events"]


def _replay_primary(panel: pd.DataFrame, frozen: dict[str, Any], cfg: cld.Config) -> pd.DataFrame:
    signal = cld.build_signal(panel, **SELECTED)
    schedule = cld.quarterly_schedule(signal, cfg)
    events = _canonical_events(schedule)
    if cld.canonical_hash(events) != PRIMARY_EVENT_CLOCK_HASH:
        raise RuntimeError("CLD primary signal no longer replays its frozen clock")
    if events != frozen["events"]:
        raise RuntimeError("CLD primary event records no longer replay exactly")
    return schedule


def build_control_schedules(
    panel: pd.DataFrame,
    primary: pd.DataFrame,
    cfg: cld.Config,
) -> dict[str, pd.DataFrame]:
    move = panel["median_return_abs"].ge(panel["move_q60"])
    breadth = panel["participation"].ge(5 / 6)
    flow = panel["flow_alignment"].ge(4 / 6)
    lag = panel["btc_lag"].gt(0.0)
    transition = _base_transition_mask(panel)
    static_signal = _signal_from_mask(
        panel,
        panel["clean"] & move & breadth & flow & lag,
        branch="static_alt_breadth",
    )
    no_flow_signal = _signal_from_mask(
        panel,
        transition & lag,
        branch="leadership_transition_without_flow",
    )
    no_lag_signal = _signal_from_mask(
        panel,
        transition & flow,
        branch="leadership_transition_without_btc_lag",
    )

    btc_close = cld._load_btc_hourly_close()
    btc_momentum = np.sign(
        np.log(btc_close / btc_close.shift(cfg.return_lookback_hours))
    )
    btc_momentum.index = panel.index
    primary_boundaries = set(pd.to_datetime(primary["feature_boundary"]))
    primary_mask = pd.to_datetime(panel["feature_boundary"]).isin(primary_boundaries)
    btc_signal = _signal_from_mask(
        panel,
        pd.Series(primary_mask, index=panel.index) & btc_momentum.ne(0.0),
        branch="btc_momentum_at_primary_opportunities",
        side=btc_momentum,
    )
    controls = {
        "static_alt_breadth": cld.quarterly_schedule(static_signal, cfg),
        "transition_without_flow": cld.quarterly_schedule(no_flow_signal, cfg),
        "transition_without_btc_lag": cld.quarterly_schedule(no_lag_signal, cfg),
        "btc_momentum_at_primary_opportunities": cld.quarterly_schedule(
            btc_signal, cfg
        ),
    }
    if len(controls["btc_momentum_at_primary_opportunities"]) != len(primary):
        raise RuntimeError("CLD BTC momentum control lost a primary opportunity")
    primary_entries = pd.to_datetime(primary["entry_date"]).tolist()
    if (
        pd.to_datetime(controls["btc_momentum_at_primary_opportunities"]["entry_date"]).tolist()
        != primary_entries
    ):
        raise RuntimeError("CLD BTC momentum control changed primary entry times")
    return controls


def _payload_for_schedule(schedule: pd.DataFrame) -> dict[str, Any]:
    events = _canonical_events(schedule)
    return {
        "event_count": len(events),
        "event_clock_sha256": cld.canonical_hash(events),
        "events": events,
    }


def run(output: str | Path = OUTPUT) -> dict[str, Any]:
    path = Path(output)
    if path != OUTPUT:
        raise ValueError("CLD control-clock output path is immutable")
    if path.exists():
        raise RuntimeError("CLD control-clock artifact already exists")
    _, frozen_primary = _load_and_verify_support()
    cfg = cld.Config()
    panel, source = cld.build_feature_panel(cfg)
    primary = _replay_primary(panel, frozen_primary, cfg)
    controls = build_control_schedules(panel, primary, cfg)
    core = {
        "protocol": "CLD-72 outcome-blind mechanism-control clocks v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "post_entry_outcomes_opened": False,
        "entry_or_later_ohlc_loaded": False,
        "funding_loaded": False,
        "controls_are_diagnostics_not_repair_candidates": True,
        "primary_event_clock_sha256": PRIMARY_EVENT_CLOCK_HASH,
        "source_contract": source,
        "controls": {
            name: _payload_for_schedule(schedule)
            for name, schedule in controls.items()
        },
    }
    core["manifest_hash"] = cld.canonical_hash(core)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(core, indent=2, sort_keys=True) + "\n")
    return core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    result = run(args.output)
    print(
        json.dumps(
            {
                "post_entry_outcomes_opened": False,
                "controls": {
                    name: {
                        "events": item["event_count"],
                        "hash": item["event_clock_sha256"],
                    }
                    for name, item in result["controls"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
