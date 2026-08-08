import hashlib
import json
from pathlib import Path

SOURCE_DIR = Path("data/binance_stablecoin_quote_flow_btc_2023_2026_aug")
MANIFEST = SOURCE_DIR / "build_manifest.json"
PANEL = SOURCE_DIR / "BTC_stablecoin_quote_flow_1h_2023-07-01_2026-07-31T23.csv.gz"

def test_augmented_stablecoin_quote_flow_source_is_outcome_blind_and_hash_bound():
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == "b9c64c3ce651934d9761a6d0731e814b2a92f5237b3040e11f794d7eb024a898"
    assert hashlib.sha256(PANEL.read_bytes()).hexdigest() == "44374b9a2298ae4b64f0c1e7208665b1c08c8221045308694311123deae1c805"
    payload = json.loads(MANIFEST.read_text())
    assert payload["combined_sha256"] == "44374b9a2298ae4b64f0c1e7208665b1c08c8221045308694311123deae1c805"
    assert payload["protocol"]["outcomes_opened"] is False
    assert payload["protocol"]["price_fields_retained"] is False
    assert payload["protocol"]["archive_checksums_verified"] is True
    assert payload["rows"] == payload["complete_rows"] == payload["expected_rows"] == 80320
    assert payload["last_date"] == "2026-07-31T23:00:00"
