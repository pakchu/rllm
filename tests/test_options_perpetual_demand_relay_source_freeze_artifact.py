from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from training import freeze_options_perpetual_demand_relay_sources as freeze


ARTIFACT = Path(
    "results/options_perpetual_demand_relay_source_freeze_2026-07-19.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_opdr_source_freeze_is_hash_bound_and_outcome_blind() -> None:
    assert _sha256(ARTIFACT) == (
        "5801b8b819f4951a141700a0249c9cd421ab88922931dc1336ec15de8d1c7883"
    )
    report = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert report["manifest_hash"] == (
        "43aa11881204627e779ae5e1e562f9e9ab50485a89f4b0eaa6337b934a07741c"
    )
    assert report["outcomes_opened"] is False
    assert report["outcome_sources_opened"] == []
    assert report["btc_execution_rows_loaded"] == 0
    assert report["funding_rows_loaded"] == 0
    assert report["bvol"]["rows"] == 26_568
    assert report["bvol"]["valid_rows"] == 23_771
    assert report["bvol"]["archive_status"] == {
        "archive_missing": 26,
        "verified": 1081,
    }
    assert report["dvol"]["rows"] == 26_569
    assert report["premium"]["values_opened_in_this_stage"] is False
    implementation = Path(cast(str, freeze.__file__))
    assert report["implementation_sha256"] == _sha256(implementation)
    assert report["ready_for_outcome_blind_support"] is True
