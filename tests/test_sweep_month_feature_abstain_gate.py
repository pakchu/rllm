import json
from pathlib import Path

from training.sweep_month_feature_abstain_gate import apply_month_rule


def test_apply_month_rule_blocks_only_matching_month_trade_rows(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    preds = tmp_path / "preds.jsonl"
    preds.write_text("\n".join([
        json.dumps({"date": "2025-01-02", "signal_pos": 1, "prediction": {"gate": "TRADE", "side": "LONG", "hold_bars": 144}}),
        json.dumps({"date": "2025-02-02", "signal_pos": 2, "prediction": {"gate": "TRADE", "side": "SHORT", "hold_bars": 144}}),
    ]) + "\n")
    out = tmp_path / "out.jsonl"
    meta = apply_month_rule(str(preds), {"2025-01": {"x": 2.0}, "2025-02": {"x": 0.0}}, feature="x", op=">=", threshold=1.0, output=str(out))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert meta["blocked_trade_rows"] == 1
    assert rows[0]["prediction"]["gate"] == "NO_TRADE"
    assert rows[1]["prediction"]["gate"] == "TRADE"
