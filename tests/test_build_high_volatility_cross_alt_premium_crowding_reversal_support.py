from training import build_high_volatility_cross_alt_premium_crowding_reversal_support as support
def test_source_coverage_requires_every_frozen_alt():
 rows=[{"symbol":symbol,"interval":"1m","rows":1} for symbol in support.prereg.build()["features"]["alts"]];assert support.evaluate(rows)["source_support_passed"] is True
 rows.pop();result=support.evaluate(rows);assert result["source_support_passed"] is False;assert result["missing_symbols"]==["XRPUSDT"]
