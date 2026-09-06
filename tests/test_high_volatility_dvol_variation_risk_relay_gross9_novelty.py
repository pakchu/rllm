import pandas as pd
from training import evaluate_high_volatility_dvol_variation_risk_relay_gross9_novelty as novelty

def test_pair_enforces_all_registered_limits(monkeypatch):
    monkeypatch.setattr(novelty.metric,"evaluate_pair",lambda a,b:{"metrics":{"exact_entry_jaccard":.1,"one_to_one_6h_max_matched_share":.35,"occupied_5m_bar_jaccard":.25,"absolute_signed_exposure_pearson":.35}})
    result=novelty.pair(pd.DataFrame(),pd.DataFrame());assert result["passed"];assert all(result["checks"].values())
