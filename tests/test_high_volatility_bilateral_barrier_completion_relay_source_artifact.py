import hashlib
import json

from training import build_high_volatility_bilateral_barrier_completion_relay_support as source


def test_hvbbc_source_pass_is_frozen_and_reproduced() -> None:
    assert hashlib.sha256(source.RESULT.read_bytes()).hexdigest() == "7ad9add351215bd1b18b16a0d7fa1babd12b1f9395aa15944faaaf2d827a5d19"
    result = json.loads(source.RESULT.read_text())
    core = {key: value for key, value in result.items() if key != "manifest_hash"}
    assert result["manifest_hash"] == source.chash(core) == "f5dc469040bc2f2bc71c136853fe869dc6c1efb42cdf7b3728061adc744dd79d"
    assert result["support_passed"] is True and all(result["support_checks"].values())
    assert result["clock"]["sha256"] == "3f1873fe8c9c5d181702cac2b9b0836731ab9db35bab0c0eb99496ccf8ac0ccb"
    assert result["advance_to_gross9_novelty"] is True
    assert result["postentry_return_pnl_execution_price_opened"] is False
    assert result["gross9_rows_opened"] is False
