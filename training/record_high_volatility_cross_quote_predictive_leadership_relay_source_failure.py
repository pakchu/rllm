"""Record the frozen HVQPLR-6 source-contract failure without opening outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training import preregister_high_volatility_cross_quote_predictive_leadership_relay as prereg


ENV_FILE = Path("/home/pakchu/rllm/.env")
PREREG_SHA = "110ee4376d5d4421f08bb980c72be52cccf92a224c76455f1391c6bf48970880"
OUTPUT = Path("results/high_volatility_cross_quote_predictive_leadership_relay_source_failure_2026-08-11.json")
QUERY = """SELECT symbol,interval,min(ts) AS first_ts,max(ts) AS last_ts,count(*) AS rows
FROM bars_binance_spot
WHERE symbol IN ('BTCUSDT','BTCUSDC','BTCFDUSD') AND interval='1m'
GROUP BY symbol,interval ORDER BY symbol"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA: raise RuntimeError("HVQPLR preregistration drift")
    from sqlalchemy import text
    from preprocessing.live_db_features import sqlalchemy_engine_from_env
    engine = sqlalchemy_engine_from_env(ENV_FILE)
    try:
        with engine.connect() as connection:
            rows = [dict(row._mapping) for row in connection.execute(text(QUERY))]
    finally:
        engine.dispose()
    available = {row["symbol"] for row in rows}
    required = {"BTCUSDT", "BTCUSDC", "BTCFDUSD"}
    serialized = [{key: value.isoformat() if hasattr(value, "isoformat") else value for key, value in row.items()} for row in rows]
    core = {
        "protocol_version": "hvqplr_6_source_failure_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA},
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(), "table": "bars_binance_spot",
        "required_symbols": sorted(required), "observed_metadata": serialized, "missing_symbols": sorted(required - available),
        "source_contract_passed": required.issubset(available), "source_incidence_opened": True,
        "candidate_clock_rows_built": 0, "postentry_return_pnl_execution_price_opened": False,
        "gross9_rows_opened": False, "advance_to_gross9_novelty": False, "advance_to_economic_outcomes": False,
        "decision": "terminal_source_contract_reject", "repair_prohibited": "no table, cache, symbol, book, or source substitution",
    }
    if core["source_contract_passed"]: raise RuntimeError("HVQPLR source unexpectedly became complete")
    result = {**core, "manifest_hash": prereg.canonical_hash(core)}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__": print(json.dumps(run(), indent=2))
