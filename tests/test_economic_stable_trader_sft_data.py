from training.economic_stable_trader_sft_data import make_target, reward_bucket, risk_from_score, split_name_for_fold


def test_reward_bucket_labels_no_trade():
    assert reward_bucket(None) == "NO_TRADE"


def test_risk_from_score_thresholds():
    assert risk_from_score({"score": 0.003}) == "LOW"
    assert risk_from_score({"score": 0.001}) == "MEDIUM"
    assert risk_from_score({"score": 0.0}) == "HIGH"


def test_make_target_normalizes_none():
    assert make_target("NONE", {"score": None})["action"] == "NO_TRADE"


def test_split_name_for_fold():
    assert split_name_for_fold("2024_h1") == "train"
    assert split_name_for_fold("2025_h1_val") == "val"
    assert split_name_for_fold("2025_h2_oos") == "eval"
