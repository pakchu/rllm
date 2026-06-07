import json
from pathlib import Path

from training.economic_preference_sft_data import build_economic_preference_sft_jsonl, preference_rows_to_sft


def _row(date: str, pos: int, chosen: dict, util: float):
    return {
        "date": date,
        "signal_pos": pos,
        "prompt": "past",
        "chosen": json.dumps(chosen, sort_keys=True, separators=(",", ":")),
        "chosen_action": {"utility": util},
    }


def _ranked_row(date: str, pos: int, chosen: dict, util: float, rank_util: float):
    row = _row(date, pos, chosen, util)
    row["chosen_action"]["rank_utility"] = rank_util
    return row


def test_preference_rows_to_sft_dedupes_to_best_utility_per_signal():
    rows = [
        _row("2025-01-01 00:00:00", 1, {"gate": "TRADE", "side": "LONG", "hold_bars": 36}, 0.01),
        _row("2025-01-01 00:00:00", 1, {"gate": "TRADE", "side": "LONG", "hold_bars": 72}, 0.02),
        _row("2025-01-01 00:05:00", 2, {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}, 0.0),
    ]
    out = preference_rows_to_sft(rows)
    assert len(out) == 2
    assert json.loads(out[0]["target"])["hold_bars"] == 72
    assert out[0]["leakage_guard"]["one_row_per_signal_timestamp"] is True


def test_preference_rows_to_sft_uses_rank_utility_when_present():
    rows = [
        _ranked_row("2025-01-01 00:00:00", 1, {"gate": "TRADE", "side": "LONG", "hold_bars": 432}, 0.03, 0.001),
        _ranked_row("2025-01-01 00:00:00", 1, {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}, 0.0, 0.01),
    ]
    out = preference_rows_to_sft(rows)
    assert len(out) == 1
    assert json.loads(out[0]["target"]) == {"gate": "NO_TRADE", "side": "NONE", "hold_bars": 0}


def test_build_economic_preference_sft_jsonl_writes_summary(tmp_path: Path):
    pref = tmp_path / "pref.jsonl"
    rows = [
        _row("2025-01-01 00:00:00", 1, {"gate": "TRADE", "side": "SHORT", "hold_bars": 144}, 0.03),
        _row("2025-01-01 00:00:00", 1, {"gate": "TRADE", "side": "SHORT", "hold_bars": 72}, 0.02),
    ]
    pref.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "sft.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = build_economic_preference_sft_jsonl(preferences=str(pref), output=str(out), summary_output=str(summary_path))
    assert summary["summary"]["sft_rows"] == 1
    assert summary["summary"]["deduped_preference_pairs"] == 1
    assert out.exists()
    assert summary_path.exists()
