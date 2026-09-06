import pytest

from training import calibrate_pposm_rlvr_margin_gate as calibration


def test_identity_alignment_is_exact_and_ordered():
    rows = [{"identity": "b", "margin": 2.0}, {"identity": "a", "margin": 1.0}]
    assert calibration.identity_align_margins(rows, ["a", "b"]) == [1.0, 2.0]
    with pytest.raises(ValueError, match="do not equal"):
        calibration.identity_align_margins(rows, ["a", "c"])


def test_threshold_selection_uses_only_supplied_train_metrics():
    trades = [object()] * 10
    margins = list(range(10))

    def metrics(selected):
        count = len(selected)
        return {
            "eligible": count >= 6,
            "combined_pre2024": {
                "cagr_to_strict_mdd": float(count),
                "absolute_return_pct": float(count),
            },
            "periods": {},
        }

    threshold, reports = calibration.select_threshold(
        margins, trades, strategy_cfg=object(), metric_function=metrics
    )
    assert threshold == float("-inf")
    assert reports[0]["selected_trade_count"] == 10
