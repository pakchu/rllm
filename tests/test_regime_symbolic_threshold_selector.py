from pathlib import Path

from training.regime_symbolic_threshold_selector import (
    SymbolicThresholdSelectorCfg,
    _candidate_score,
    _parse_csv,
    _split_history,
)
from training.symbolic_action_ridge import write_jsonl


def test_parse_csv_typed_values():
    assert _parse_csv("a,b", str) == ["a", "b"]
    assert _parse_csv("0.1,-0.2", float) == [0.1, -0.2]


def test_candidate_score_rejects_bad_validation():
    cfg = SymbolicThresholdSelectorCfg(history_jsonl="h", final_eval_jsonl="e", market_csv="m", output="o")
    bt = {
        "sim": {"trade_entries": 151, "cagr_pct": -26.6, "strict_mdd_pct": 32.1, "cagr_to_strict_mdd": -0.83},
        "trade_stats": {"p_value_mean_ret_approx": 0.02},
    }
    score, reasons = _candidate_score(bt, cfg)
    assert score < -1000
    assert "val_cagr_below_min" in reasons
    assert "val_mdd_above_max" in reasons


def test_split_history_uses_validation_half_open_bounds(tmp_path: Path):
    src = tmp_path / "rows.jsonl"
    write_jsonl(
        src,
        [
            {"date": "2024-12-31 00:00:00"},
            {"date": "2025-01-01 00:00:00"},
            {"date": "2026-01-01 00:00:00"},
        ],
    )
    fit, val, counts = _split_history(str(src), "2025-01-01", "2026-01-01", tmp_path)
    assert counts == {"history": 3, "fit": 1, "validation": 1}
    assert Path(fit).read_text().count("date") == 1
    assert Path(val).read_text().count("date") == 1
