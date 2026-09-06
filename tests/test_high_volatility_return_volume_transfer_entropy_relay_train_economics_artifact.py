import hashlib
import json
from pathlib import Path

from training import evaluate_high_volatility_return_volume_transfer_entropy_relay_economics as economics


RESULT = Path(
    "results/high_volatility_return_volume_transfer_entropy_relay_train_economics_2026-08-13.json"
)


def test_terminal_train_rejection_is_sealed_and_later_stages_absent() -> None:
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "c17ce9f18a54368ae6a824374146cc02040fd415c083b0eb8d7bc61d5d514927"
    )
    report = json.loads(RESULT.read_text())
    manifest_hash = report.pop("manifest_hash")
    assert economics.canonical_hash(report) == manifest_hash
    assert report["policy_id"] == "HVRVTE-8"
    assert report["stage"] == "train"
    assert not report["passed"]
    assert report["decision"] == "terminal_reject_no_repair"
    assert not report["later_stage_outcomes_opened"]
    base = report["primary"]["base"]
    assert base["trades"] == 31
    assert base["absolute_return_pct"] == -3.090348181916569
    assert base["mean_gross_underlying_bp"] == -8.640638931431035
    assert base["strict_mdd_pct"] == 6.116195076851727
    assert report["primary"]["stress"]["absolute_return_pct"] == -4.287356053768033
    assert report["primary"]["cluster_signflip"]["pvalue"] == 0.8548214517854822
    assert report["primary"]["calendar_halves"]["first"]["absolute_return_pct"] > 0
    assert report["primary"]["calendar_halves"]["second"]["absolute_return_pct"] < 0
    assert sum(report["checks"].values()) == 1
    for stage in ("test", "eval", "final"):
        assert not Path(
            f"results/high_volatility_return_volume_transfer_entropy_relay_{stage}_economics_2026-08-13.json"
        ).exists()
