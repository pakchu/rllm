import hashlib
import json
from pathlib import Path

from training import (
    build_high_volatility_internal_bar_strength_extreme_reversal_relay_support as support,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ibs_source_rejection_is_terminal_and_sealed() -> None:
    result = json.loads(support.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == support.canonical_hash(core)
    assert result["support_passed"] is False
    assert result["decision"] == "terminal_source_support_reject"
    assert result["advance_to_gross9_novelty"] is False
    assert result["advance_to_economic_outcomes"] is False
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
    assert [
        result["support"][stage]["events"]
        for stage in ("train", "test", "eval", "final")
    ] == [11, 40, 41, 23]
    assert result["support_checks"]["train_month_concentration"] is False
    assert sha(support.RESULT) == (
        "a100bd66824e2d6a7bb78d3f9cff1198ea69a56f09332e4e5fad40d84dfc5697"
    )
    assert result["clock"]["sha256"] == sha(Path(result["clock"]["path"]))
