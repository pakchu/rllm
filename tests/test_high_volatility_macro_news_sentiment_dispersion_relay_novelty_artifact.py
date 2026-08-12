import hashlib
import json
from pathlib import Path


RESULT = Path(
    "results/high_volatility_macro_news_sentiment_dispersion_relay_"
    "gross9_novelty_2026-08-13.json"
)


def test_novelty_artifact_rejects_before_economic_outcomes():
    payload = json.loads(RESULT.read_text())
    assert payload["gross9_novelty_status"] == "failed"
    assert payload["advance_to_economic_outcomes"] is False
    kimchi = payload["gross9_sleeves"]["fresh_kimchi_fx"]
    assert kimchi["metrics"]["one_to_one_6h_max_matched_share"] > 0.35
    assert kimchi["checks"]["one_to_one_6h_max_matched_share"] is False
    boundary = payload["evidence_boundary"]
    assert boundary["outcomes_opened"] is False
    assert boundary["btc_execution_rows_opened"] == 0
    assert boundary["funding_rows_opened"] == 0
    assert hashlib.sha256(RESULT.read_bytes()).hexdigest() == (
        "59b25fd7c089126f4580ce8c8a678652d3b4a9b3c72e46e626d410ef6abe77da"
    )
