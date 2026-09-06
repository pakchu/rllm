import hashlib
import json
from pathlib import Path

from training import build_options_oi_chase_exhaustion_support as support
from training import materialize_options_oi_chase_exhaustion_sources as sources


SOURCE_MANIFEST = Path("data/options_oi_chase_exhaustion_sources_2023_2026/manifest.json")
RESULT = Path("results/options_oi_chase_exhaustion_reversal_support_2026-08-08.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_snapshot_is_feature_only_and_frozen():
    assert digest(SOURCE_MANIFEST) == "3e350d16da72da7b60d9e91fbfb1ff4c2e13e5cb954b52b19ceaddf8c4f0e66d"
    report = json.loads(SOURCE_MANIFEST.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == sources.chash(core) == "bcca0c0ad88870d9b05a021ded3860f8e4124c4bc9fed153da7a1b431caa3c92"
    assert report["post_entry_return_pnl_or_execution_price_opened"] is False
    assert report["candidate_incidence_opened"] is False


def test_support_artifact_is_terminal_before_novelty_or_economics():
    assert digest(RESULT) == "64e595ccf31519864a726bfca1c4ca03b24b5b25901799ae0630c4cc08100987"
    report = json.loads(RESULT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.chash(core) == "0f7fc1c8797e0f116dd6435288b4196b767520eb51f52e3f10b7e2e5426a588b"
    assert report["clock"]["sha256"] == digest(support.CLOCK)
    assert all(item["sha256"] == digest(Path(item["path"])) for item in report["controls"].values())
    assert report["support_passed"] is False
    assert report["decision"] == "terminal_source_support_reject"
    assert report["advance_to_gross9_novelty"] is False
    assert report["advance_to_economic_outcomes"] is False
    assert all(not report["support_checks"][f"{split}_side_balance"] for split in ("train", "test", "eval"))
    assert report["support_checks"]["final_side_balance"] is True
