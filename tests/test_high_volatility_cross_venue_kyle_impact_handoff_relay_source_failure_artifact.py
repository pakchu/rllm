import hashlib
import json
from pathlib import Path

import pandas as pd


RESULT = Path("results/high_volatility_cross_venue_kyle_impact_handoff_relay_support_2026-08-10.json")
PANEL = Path("data/high_volatility_cross_venue_kyle_impact_handoff_relay_sources_2023_2026/eight_hour_impact_handoff_panel.csv.gz")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hvckihr_source_failure_is_terminal_and_outcomes_stay_sealed():
    report = json.loads(RESULT.read_text())
    assert report["policy_id"] == "HVCKIHR-8"
    assert report["support_passed"] is False
    assert report["decision"] == "terminal_source_support_reject"
    assert report["advance_to_gross9_novelty"] is False
    assert report["advance_to_economic_outcomes"] is False
    assert report["postentry_return_pnl_execution_price_opened"] is False
    assert report["gross9_rows_opened"] is False
    assert [report["support"][split]["events"] for split in ("train", "test", "eval", "final")] == [0, 0, 0, 0]


def test_hvckihr_source_panel_proves_july_only_spot_coverage():
    report = json.loads(RESULT.read_text())
    manifest_path = Path(report["source_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["output"]["sha256"] == sha(PANEL)
    assert manifest["output"]["valid_rows"] == 81
    panel = pd.read_csv(PANEL, compression="gzip")
    valid = panel[panel.source_valid.astype(str).str.lower().eq("true")]
    decision = pd.to_datetime(valid.decision_time, utc=True)
    assert decision.min() == pd.Timestamp("2026-07-05T00:00:00Z")
    assert decision.max() == pd.Timestamp("2026-07-31T16:00:00Z")
    assert pd.to_numeric(valid.handoff_rank, errors="coerce").notna().sum() == 0
    assert pd.to_numeric(valid.variation_rank, errors="coerce").notna().sum() == 0


def test_hvckihr_later_stage_artifacts_do_not_exist():
    assert not Path("results/high_volatility_cross_venue_kyle_impact_handoff_relay_gross9_novelty_2026-08-10.json").exists()
    assert not Path("results/high_volatility_cross_venue_kyle_impact_handoff_relay_train_economics_2026-08-10.json").exists()
