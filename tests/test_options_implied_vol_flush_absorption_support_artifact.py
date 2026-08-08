import hashlib
import json
from pathlib import Path

from training import build_options_implied_vol_flush_absorption_support as support


RESULT = Path("results/options_implied_vol_flush_absorption_reversal_support_2026-08-08.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_oifar_support_is_frozen_terminal_before_novelty_and_economics():
    assert digest(RESULT) == "9976e185171dea79b801d4937f32a273abc83032dfa8950c4fa57ad2f14fbb03"
    report = json.loads(RESULT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core) == "fb252b0ab9779d29c0128a334960dc6bd27425c15e0a74ee996a2e18c128a259"
    assert report["clock"]["sha256"] == digest(support.CLOCK)
    assert all(item["sha256"] == digest(Path(item["path"])) for item in report["controls"].values())
    assert report["support_passed"] is False
    assert report["decision"] == "terminal_source_support_reject"
    assert report["advance_to_gross9_novelty"] is False
    assert report["advance_to_economic_outcomes"] is False
    assert report["support_checks"]["final_side_balance"] is False
    assert all(
        report["support_checks"][f"{split}_side_balance"]
        for split in ("train", "test", "eval")
    )
