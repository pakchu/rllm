import json
from pathlib import Path

from training.build_score_direction_binary_sft import ScoreDirectionBinarySFTConfig, run


def test_binary_sft_skips_abstain_and_buckets_features(tmp_path):
    src = tmp_path / "src.jsonl"
    out = tmp_path / "out.jsonl"
    rows = [
        {"fold": {"start": "2024-01-01"}, "features": {"rex_8640_range_width_pct_last": 0.4, "bb_z_last": 1.2}, "scoreboard_summary": {"families": ["a", "b"], "scores": [2, -1]}, "target": json.dumps({"direction_regime": "HIGH_SCORE_WINS"})},
        {"fold": {"start": "2024-02-01"}, "features": {"rex_8640_range_width_pct_last": 0.05, "bb_z_last": -1.2}, "scoreboard_summary": {"families": ["a", "b"], "scores": [2, -1]}, "target": json.dumps({"direction_regime": "LOW_SCORE_WINS"})},
        {"fold": {"start": "2024-03-01"}, "features": {}, "scoreboard_summary": {}, "target": json.dumps({"direction_regime": "ABSTAIN"})},
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    report = run(ScoreDirectionBinarySFTConfig(input_jsonl=str(src), output_jsonl=str(out), split_name="train"))
    built = [json.loads(line) for line in out.read_text().splitlines()]
    assert report["targets"] == {"HIGH": 1, "LOW": 1}
    assert len(built) == 2
    assert "very_wide" in built[0]["prompt"]
    assert json.loads(built[1]["target"])["trust_score_rank"] == "LOW"
