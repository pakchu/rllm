import hashlib
import json

from training import record_high_volatility_cross_maturity_depth_migration_source_failure as record


def test_failure_is_terminal_and_outcome_blind() -> None:
    payload = record.build()
    core = {key: value for key, value in payload.items() if key != "manifest_hash"}
    expected = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert payload["manifest_hash"] == expected
    assert payload["failure"]["http_status"] == 404
    assert payload["failure"]["contract"] == "BTCUSD_230929"
    assert payload["output_boundary"]["candidate_incidence_opened"] is False
    assert payload["output_boundary"]["gross9_rows_opened"] is False
    assert payload["output_boundary"]["return_pnl_cagr_mdd_opened"] is False
    assert payload["support_passed"] is False
    assert payload["advance_to_gross9_novelty"] is False
    assert payload["advance_to_economic_outcomes"] is False
    assert payload["decision"] == "terminal_source_contract_reject_no_repair"
