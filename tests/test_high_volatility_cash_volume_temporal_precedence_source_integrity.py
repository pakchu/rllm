from training import evaluate_high_volatility_cash_volume_temporal_precedence_source_integrity as s
def test_contract_is_outcome_blind_and_bound():
 assert s.PREREG_SHA=='573eaefe87cf1c69501264243f887032281955c8e4cbe65f22f8950d196836af';source=open(s.__file__).read();assert 'postentry_outcomes_opened' in source and 'funding_rates_binance' not in source
