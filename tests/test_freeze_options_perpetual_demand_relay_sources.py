from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from training import freeze_options_perpetual_demand_relay_sources as freeze


def test_real_sources_validate_without_outcomes() -> None:
    cfg = freeze.Config()
    bvol, manifest = freeze.load_bvol(cfg)
    dvol, summary = freeze.load_dvol(cfg)
    assert len(bvol) == 26_568
    assert int(bvol["feature_valid"].sum()) == 23_771
    assert manifest["protocol"]["outcomes_opened"] is False
    assert len(dvol) == 26_569
    assert summary["availability"] == (
        "candle values join on close_time, never date/open time"
    )


def test_dvol_fails_if_hash_bound_source_is_replaced(tmp_path: Path) -> None:
    cfg = freeze.Config()
    replacement = tmp_path / "dvol.csv.gz"
    pd.DataFrame(
        {
            "date": ["2023-06-20"],
            "close_time": ["2023-06-20 01:00"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
        }
    ).to_csv(replacement, index=False, compression="gzip")
    changed = freeze.Config(
        preregistration=cfg.preregistration,
        bvol=cfg.bvol,
        bvol_manifest=cfg.bvol_manifest,
        dvol=str(replacement),
        dvol_summary=cfg.dvol_summary,
        output=cfg.output,
    )
    with pytest.raises(ValueError, match="DVOL bytes changed"):
        freeze.load_dvol(changed)
