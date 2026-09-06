from pathlib import Path
from training import evaluate_cross_asset_volatility_breadth_relay_economics as economics

def test_cavbr_economics_is_sequential_and_frozen():
 s=Path(economics.__file__).read_text();assert economics.POLICY_ID=="CAVBR-12";assert economics.LEVERAGE==.5;assert economics.BASE_COST==.0006;assert economics.STRESS_COST==.001;assert economics.PREDECESSOR=={"test":"train","eval":"test","final":"eval"};assert "terminal_reject_no_repair" in s;assert '"later_stage_outcomes_opened": False' in s

def test_cavbr_economics_binds_authorization():
 assert economics.sha256(economics.PREREG)==economics.PREREG_SHA;assert economics.sha256(economics.SUPPORT)==economics.SUPPORT_SHA;assert economics.sha256(economics.NOVELTY)==economics.NOVELTY_SHA;assert economics.sha256(economics.CLOCK)==economics.CLOCK_SHA;assert economics.CONTROLS==("no_volatility_gate","vix_only","equity_pair_only","one_session_stale_breadth","direction_flip")
