from __future__ import annotations

import pandas as pd
import pytest
import json

from training import build_gross9_overlap_portfolio_universe as u


def _clock(rows):
    return pd.DataFrame(rows, columns=["candidate","control","split","decision_time","feature_available_time","entry_time","exit_time","side"])


def test_failed_novelty_inventory_is_exact_64_and_near_only():
    rows = u.failed_novelty_records()
    assert len(rows) == 64
    assert len({row["policy_id"] for row in rows}) == 64
    assert "G9QTR-DISTILL-8" not in {row["policy_id"] for row in rows}
    assert all(row["failed_near_6h"] for row in rows)


def test_frozen_novelty_inventory_rejects_receipt_drift(tmp_path):
    inventory = json.loads(u.HISTORICAL_INVENTORY.read_text(encoding="utf-8"))
    inventory["records"][0]["sha256"] = "0" * 64
    core = {key: value for key, value in inventory.items() if key != "manifest_hash"}
    inventory["manifest_hash"] = u.canonical_hash(core)
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(RuntimeError, match="historical novelty SHA drift"):
        u.failed_novelty_records(path)


def test_historical_clocks_are_hash_bound_full_split_and_nonoverlapping():
    rows = u.historical_sleeves()
    assert len(rows) == 64
    assert all(set(row["split_counts"]) == set(u.SPLITS) for row in rows)
    assert all(sum(row["split_counts"].values()) == row["clock"]["rows"] for row in rows)


def test_active_duplicate_only_collapses_13_ids_to_7_schedules(tmp_path):
    rows = u.active_duplicate_only_sleeves(tmp_path)
    assert len(rows) == 7
    assert sum(1 + len(row["provenance"]["aliases"]) for row in rows) == 13
    assert all(set(row["split_counts"]) == set(u.SPLITS) for row in rows)


def test_active_veto_strict_lower_and_same_time_latest():
    def row(component, time, side):
        t=pd.Timestamp(time);return [component,"primary","train",t-pd.Timedelta("5m"),t-pd.Timedelta("5m"),t,t+pd.Timedelta("8h"),side]
    base=_clock([row("A","2023-07-01T06:00:00Z",1),row("A","2023-07-01T14:00:00Z",1)])
    veto=_clock([row("B","2023-07-01T00:00:00Z",-1),row("B","2023-07-01T06:00:00Z",1),row("B","2023-07-01T14:00:00Z",-1)])
    out=u.build_active_full_clock("A__ASYNC_ACTIVE_OPPOSITE_VETO_6H__B","A","B",base,veto)
    assert out.entry_time.tolist()==[pd.Timestamp("2023-07-01T06:00:00Z")]


def test_build_preserves_outcome_blind_boundary(tmp_path):
    report=u.build(tmp_path/"universe.json",tmp_path/"active")
    assert report["precanonical_schedule_count"]==71
    assert report["canonical_sleeve_count"]<=71
    assert report["evidence_boundary"]["market_rows_opened"]==0
    assert report["evidence_boundary"]["oos_outcomes_opened"] is False
