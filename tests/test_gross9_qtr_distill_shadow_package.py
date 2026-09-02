from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/shadow/gross9_qtr_distill_2026-09-02.json"
READINESS = ROOT / "results/gross9_qtr_distill_shadow_readiness_2026-09-02.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: dict) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def test_shadow_package_is_hash_bound_and_fail_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config_core = {key: value for key, value in config.items() if key != "protocol_hash"}
    assert config["protocol_hash"] == _canonical_hash(config_core)
    assert config["shadow_only"] is True
    assert config["live_capital_authorized"] is False
    assert config["order_submission_enabled"] is False
    assert config["additive_to_gross9_authorized"] is False
    assert math.isclose(sum(config["sleeve_weights"].values()), 0.5)
    assert config["structural_novelty"]["passed"] is False
    assert config["oos"]["economic_outcomes_opened"] is False

    for record in config["artifacts"].values():
        assert _sha(ROOT / record["path"]) == record["sha256"]

    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    readiness_core = {
        key: value for key, value in readiness.items() if key != "manifest_hash"
    }
    assert readiness["manifest_hash"] == _canonical_hash(readiness_core)
    assert readiness["config"]["sha256"] == _sha(CONFIG)
    assert readiness["decision"] == "ready_for_forward_shadow_monitoring_only"
    assert readiness["readiness_checks"]["gross9_additive_novelty_passed"] is False
    assert readiness["readiness_checks"]["live_capital_authorized"] is False
