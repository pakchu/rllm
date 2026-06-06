from training.economic_teacher_pressure_sft_data import confidence_bucket, teacher_prompt


def test_confidence_bucket_thresholds():
    assert confidence_bucket(0.56) == "HIGH"
    assert confidence_bucket(0.43) == "MEDIUM"
    assert confidence_bucket(0.2) == "LOW"


def test_teacher_prompt_adds_hint_without_removing_schema():
    prompt = teacher_prompt({"state": {"regime": "RANGE"}}, {"teacher_pressure": "LONG_FAVORED", "teacher_confidence": 0.5})
    assert "direction_pressure" in prompt
    assert "teacher_pressure" in prompt
    assert "MEDIUM" in prompt
