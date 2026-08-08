import hashlib
import json
from pathlib import Path

from training import build_options_loaded_quiet_price_breakout_support as support


RESULT = Path("results/options_loaded_quiet_price_breakout_support_2026-08-08.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_olqpb_support_is_frozen_terminal_before_novelty_and_economics():
    assert digest(RESULT) == "b1769e753fcf240f6379a420ebd68191e9164ea0b096617d440edd2e967895ec"
    report = json.loads(RESULT.read_text())
    core = {key: value for key, value in report.items() if key != "manifest_hash"}
    assert report["manifest_hash"] == support.canonical_hash(core) == "5c13dbcbc6621653477247253c989fe85d87408dda1bd6c1074e32ca30f35e06"
    assert report["clock"]["sha256"] == digest(support.CLOCK)
    assert report["support"]["eval"]["events"] == 10
    assert report["support_checks"]["eval_minimum_events"] is False
    assert report["support_checks"]["final_month_concentration"] is False
    assert report["support_passed"] is False
    assert report["advance_to_gross9_novelty"] is False
    assert report["advance_to_economic_outcomes"] is False
