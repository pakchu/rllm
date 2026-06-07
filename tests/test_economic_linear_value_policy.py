import numpy as np

from training.economic_linear_value_policy import LinearValueModel, build_vocab, feature_tokens, policy_rows, vectorize


def test_feature_tokens_include_side_interactions():
    row = {"prompt": 'Compact features: {"state":{"regime":"RANGE"},"symbolic":{"Macro Dollar State":"MACRO_NEUTRAL"}}\n\n', "teacher": {"teacher_pressure": "LONG_FAVORED", "teacher_confidence": 0.9}}
    toks = feature_tokens(row, "LONG")
    assert "side=LONG" in toks
    assert any(t.startswith("side:LONG|regime=") for t in toks)


def test_vectorize_sets_bias():
    row = {"prompt": 'Compact features: {}\n\n'}
    vocab = build_vocab([row], min_count=1)
    x = vectorize(row, "LONG", vocab)
    assert x[vocab["bias=1"]] == 1.0


def test_policy_rows_abstains_below_threshold():
    row = {"prompt": 'Compact features: {}\n\n', "date": "2025-01-01 00:00:00", "signal_pos": 0}
    vocab = build_vocab([row], min_count=1)
    model = LinearValueModel(vocab=vocab, beta=np.zeros(len(vocab)), xtx_inv=np.eye(len(vocab)), residual_std=0.0, alpha=1.0)
    rows, counts = policy_rows([row], model, threshold=0.1, risk_penalty=0.0, min_gap=0.0)
    assert counts["NONE"] == 1
    assert rows[0]["prediction"]["direction_pressure"] == "NO_TRADE_FAVORED"
