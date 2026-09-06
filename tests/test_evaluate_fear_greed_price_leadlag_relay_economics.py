from pathlib import Path
from training import evaluate_fear_greed_price_leadlag_relay_economics as economics

def test_fgplr_economics_is_sequential_and_frozen():
 s=Path(economics.__file__).read_text();assert economics.POLICY_ID=="FGPLR-24";assert economics.LEVERAGE==.5;assert economics.BASE_COST==.0006;assert economics.STRESS_COST==.001;assert economics.PREDECESSOR=={"test":"train","eval":"test","final":"eval"};assert "terminal_reject_no_repair" in s;assert '"later_stage_outcomes_opened": False' in s

def test_fgplr_economics_binds_authorization():
 assert economics.sha256(economics.PREREG)==economics.PREREG_SHA;assert economics.sha256(economics.SUPPORT)==economics.SUPPORT_SHA;assert economics.sha256(economics.NOVELTY)==economics.NOVELTY_SHA;assert economics.sha256(economics.CLOCK)==economics.CLOCK_SHA;assert economics.CONTROLS==("no_volatility_gate","no_sentiment_change_tail","no_direction_disagreement","sentiment_direction","one_day_stale_sentiment_change","direction_flip")
