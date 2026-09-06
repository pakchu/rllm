import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_approved_choice_is_exact_and_does_not_enable_orders():
    r=json.loads((ROOT/'configs/approved/g9_macro1_dollar_short05_2026-09-07.json').read_text())
    source=json.loads((ROOT/'configs/shadow/ten_alpha_portfolio_research_2026-09-07.json').read_text())
    assert r['portfolio_selection_approved']
    assert r['weights_notional']==source['descriptive_anchored_candidates']['g9_macro1_d0.5_r0.0']
    assert r['weights_notional']['macro_flow']==1
    assert r['weights_notional']['dollar_rally_short']==.5
    assert r['weights_notional']['failed_rebound_short']==0
    assert not r['enabled'] and not r['allow_live_orders'] and not r['runtime_ready']
    assert r['deployment_actions_performed']==[]
    for path,digest in r['evidence'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
