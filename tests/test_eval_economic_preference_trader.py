import json
from pathlib import Path

from training.eval_economic_preference_trader import evaluate_economic_preference_trader


def test_target_echo_exports_predictions(tmp_path: Path):
    data = tmp_path / "pref.jsonl"
    row = {
        "date": "2025-01-01 00:00:00",
        "signal_pos": 1,
        "prompt": "past only",
        "chosen": json.dumps({"gate": "TRADE", "side": "LONG", "hold_bars": 36}, sort_keys=True),
        "rejected": json.dumps({"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}, sort_keys=True),
        "chosen_action": {"utility": 0.01},
        "rejected_action": {"utility": 0.0},
        "utility_gap": 0.01,
    }
    data.write_text(json.dumps(row) + "\n")
    report_path = tmp_path / "report.json"
    preds_path = tmp_path / "preds.jsonl"

    report = evaluate_economic_preference_trader(
        eval_jsonl=str(data),
        output=str(report_path),
        predictions_output=str(preds_path),
        prediction_mode="target_echo",
    )

    assert report["metrics_vs_chosen"]["exact_action_accuracy"] == 1.0
    pred = json.loads(preds_path.read_text().strip())
    assert pred["prediction"] == {"gate": "TRADE", "side": "LONG", "hold_bars": 36}
    assert report_path.exists()
