from training.economic_pressure_filter_sweep import apply_filter


def test_apply_filter_skips_when_teacher_disagrees():
    rows = [{"prediction": {"direction_pressure": "LONG_FAVORED"}, "teacher": {"teacher_pressure": "SHORT_FAVORED", "teacher_confidence": 0.9}}]
    out = apply_filter(rows, min_conf=0.0, require_agree=True, allowed={"LONG_FAVORED", "SHORT_FAVORED"})
    assert out[0]["prediction"] == {"direction_pressure": "NO_TRADE_FAVORED"}


def test_apply_filter_keeps_allowed_confident_prediction():
    rows = [{"prediction": {"direction_pressure": "SHORT_FAVORED"}, "teacher": {"teacher_pressure": "SHORT_FAVORED", "teacher_confidence": 0.9}}]
    out = apply_filter(rows, min_conf=0.5, require_agree=True, allowed={"SHORT_FAVORED"})
    assert out[0]["prediction"] == {"direction_pressure": "SHORT_FAVORED"}
