import unittest
from unittest.mock import Mock

import torch

from training.fast_score_action_value_candidates import _batched_label_scores, _label_score_from_prompt_cache, _prediction_rows


class TinyModel:
    device = torch.device("cpu")
    def __call__(self, input_ids, use_cache=False, past_key_values=None):
        vocab = 5
        logits = torch.zeros((1, input_ids.shape[1], vocab), dtype=torch.float32)
        if past_key_values is None:
            logits[0, -1, 1] = 3.0
            pkv = ((torch.zeros(1),),)
        else:
            logits[0, 0, 2] = 3.0
            pkv = None
        return Mock(logits=logits, past_key_values=pkv)


class TinyBatchModel:
    device = torch.device("cpu")
    def __call__(self, input_ids, attention_mask=None, **kwargs):
        vocab = 5
        logits = torch.zeros((input_ids.shape[0], input_ids.shape[1], vocab), dtype=torch.float32)
        masks = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        for b in range(input_ids.shape[0]):
            seq_len = int(masks[b].sum().item())
            pad = input_ids.shape[1] - seq_len
            logits[b, pad + seq_len - 3, 1] = 3.0
            logits[b, pad + seq_len - 2, 2] = 3.0
        return Mock(logits=logits)


class TinyTokenizer:
    eos_token_id = None
    padding_side = "right"
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0, 0]}
    def pad(self, data, return_tensors=None):
        sequences = data["input_ids"]
        width = max(len(s) for s in sequences)
        padded = []
        masks = []
        for seq in sequences:
            pad = width - len(seq)
            if self.padding_side == "left":
                padded.append([0] * pad + seq)
                masks.append([0] * pad + [1] * len(seq))
            else:
                padded.append(seq + [0] * pad)
                masks.append([1] * len(seq) + [0] * pad)
        return {"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(masks)}


class TestFastScoreActionValueCandidates(unittest.TestCase):
    def test_scores_label_with_prompt_cache(self):
        good = _label_score_from_prompt_cache(TinyModel(), torch.tensor([[0, 0]]), [1, 2], "sum")
        bad = _label_score_from_prompt_cache(TinyModel(), torch.tensor([[0, 0]]), [2, 1], "sum")
        self.assertGreater(good, bad)

    def test_batched_scores_labels_for_multiple_prompts(self):
        scores = _batched_label_scores(TinyBatchModel(), TinyTokenizer(), ["a", "b"], {"GOOD": [1, 2], "BAD": [2, 1]}, "sum")
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0]["GOOD"]["sum"], scores[0]["BAD"]["sum"])
        self.assertGreater(scores[1]["GOOD"]["sum"], scores[1]["BAD"]["sum"])

    def test_threshold_outputs_no_trade(self):
        rows = [{"date": "d", "signal_pos": 1, "margin_mean_take_minus_skip": -0.1, "action": {"side": "LONG", "hold_bars": 72}, "action_audit": {}}]
        out = _prediction_rows(rows, margin_field="margin_mean_take_minus_skip", threshold=0.0)
        self.assertEqual(out[0]["prediction"]["gate"], "NO_TRADE")


if __name__ == "__main__":
    unittest.main()
