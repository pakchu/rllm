from training.monthly_selector_anti_persistence_overlay import AntiPersistenceCfg, _blocked_months


def test_blocked_months_flags_validation_spikes():
    report = {"months": [{"month": "2026-04", "selected": {"backtest": {"sim": {"cagr_to_strict_mdd": 20, "cagr_pct": 150}, "trade_stats": {"t_stat_like": 2.5}}}}}]}
    cfg = AntiPersistenceCfg(selector_report="r", predictions_jsonl="p", market_csv="m", output="o", max_val_ratio=10, max_val_cagr_pct=100, max_val_t=2)
    blocked = _blocked_months(report, cfg)
    assert "2026-04" in blocked
    assert "val_ratio_too_high" in blocked["2026-04"]["reasons"]
