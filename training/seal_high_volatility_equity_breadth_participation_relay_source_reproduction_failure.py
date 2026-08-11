"""Seal terminal source-reproduction failure for HVEBPR-24."""
from __future__ import annotations

import hashlib, json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import build_high_volatility_equity_breadth_participation_relay_support as support

OUTPUT = Path("results/high_volatility_equity_breadth_participation_relay_source_reproduction_failure_2026-08-12.json")
SECOND = {
    "rsp_raw": "d79e3feb9f0c0768953332c26eed0b7a9b168008df86402b040e6f3edec0e7ff",
    "spy_raw": "c95b371767bdf4d2742d311e2020227bb0972422e0b21d79210ab5f3e09a4a03",
    "equity_panel": "5461620cc9a4021be3dacc52b3eb42a1c5ad2332ec7f1bd24bd3803b12293d59",
}
PATHS = {
    "rsp_raw": support.RAW_PATHS["RSP"],
    "spy_raw": support.RAW_PATHS["SPY"],
    "equity_panel": support.EQUITY_PANEL,
    "features": support.FEATURE_PANEL,
    "clock": support.CLOCK,
    "support": support.RESULT,
}


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def build() -> dict[str, Any]:
    report=json.loads(support.RESULT.read_text())
    if report["support_passed"] is not True or report["decision"]!="pass_to_novelty":
        raise RuntimeError("HVEBPR source-incidence report drift")
    third={name:sha(path) for name,path in PATHS.items()}
    changed=[name for name in SECOND if SECOND[name]!=third[name]]
    if changed != ["rsp_raw","spy_raw","equity_panel"]:
        raise RuntimeError(f"HVEBPR reproduction evidence drift: {changed}")
    core={
        "protocol_version":"hvebpr_24_source_reproduction_failure_v1",
        "policy_id":"HVEBPR-24",
        "source_support_statistics_passed":True,
        "source_support":report["support"],
        "reproduction_attempts":3,
        "second_attempt_sha256":SECOND,
        "third_attempt_sha256":third,
        "mismatch":{
            "raw_sources_changed":["RSP Yahoo adjusted-close vector","SPY Yahoo adjusted-close vector"],
            "derived_outputs_changed":["common adjusted-session panel","preentry features","candidate and control clocks","source manifest","support artifact"],
            "event_statistics_unchanged":True,
            "fixed_request_window_unchanged":True,
        },
        "scientific_gate":"byte-identical completed-source reproduction",
        "failure_reason":"Yahoo returned different historical adjusted-close vectors across immediate identical requests",
        "decision":"terminal_source_reproduction_reject_no_repair",
        "advance_to_gross9_novelty":False,
        "advance_to_economic_outcomes":False,
        "repair_authorized":False,
        "prohibited_repairs":["substitute raw close","select one response","change provider","change symbols","change spread","change threshold","change clock"],
        "postentry_return_pnl_execution_price_opened":False,
        "gross9_rows_opened":False,
    }
    return {**core,"manifest_hash":canonical_hash(core)}


if __name__=="__main__":
    x=build(); OUTPUT.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n"); print(OUTPUT)
