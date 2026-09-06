import json
def test_cvbrpcr_support_artifact_passes_without_outcomes():
 r=json.load(open("results/crypto_volatility_beta_residual_price_confirmation_relay_support_2026-08-08.json"));assert r["decision"]=="pass_to_novelty";assert r["support_passed"] is True;assert r["advance_to_gross9_novelty"] is True;assert r["advance_to_economic_outcomes"] is False;assert r["postentry_return_pnl_execution_price_opened"] is False;assert r["gross9_rows_opened"] is False
