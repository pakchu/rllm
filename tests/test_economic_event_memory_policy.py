import json

from training.economic_event_memory_policy import build_action_memory, regime_key


def _row(regime, trend, utility, action):
    prompt = json.dumps(
        {
            "regime": regime,
            "symbolic_features": {"trend_alignment": trend},
        }
    )
    return {"prompt": prompt, "utility": utility, "action": json.dumps(action)}


def test_regime_key_reads_summary_and_symbolic_fields():
    row = _row("bull", "up", 0.01, {"gate": "TRADE", "side": "LONG", "hold_bars": 72})
    assert regime_key(row, ("regime", "trend_alignment", "missing")) == ("bull", "up", "NA")


def test_build_action_memory_uses_train_positive_buckets_and_global_fallback():
    rows = [
        _row("bull", "up", 0.01, {"gate": "TRADE", "side": "LONG", "hold_bars": 72}),
        _row("bull", "up", 0.02, {"gate": "TRADE", "side": "SHORT", "hold_bars": 72}),
        _row("bull", "up", 0.03, {"gate": "TRADE", "side": "SHORT", "hold_bars": 72}),
        _row("bear", "down", 0.02, {"gate": "TRADE", "side": "LONG", "hold_bars": 288}),
        _row("bear", "down", -0.10, {"gate": "TRADE", "side": "SHORT", "hold_bars": 432}),
    ]

    memory, fallback = build_action_memory(rows, utility_threshold=0.003, fields=("regime", "trend_alignment"), min_bucket=2)

    assert memory[("bull", "up")] == {"gate": "TRADE", "side": "SHORT", "hold_bars": 72}
    assert ("bear", "down") not in memory
    assert fallback == {"gate": "TRADE", "side": "SHORT", "hold_bars": 72}
