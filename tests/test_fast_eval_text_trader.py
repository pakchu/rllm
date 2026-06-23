import unittest
from unittest.mock import Mock

import torch

from training.fast_eval_text_trader import _candidate_logprob_from_prompt_cache


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


class TestFastEvalTextTrader(unittest.TestCase):
    def test_scores_candidate_with_prompt_cache(self):
        score_good = _candidate_logprob_from_prompt_cache(TinyModel(), torch.tensor([[0, 0]]), [1, 2], "sum")
        score_bad = _candidate_logprob_from_prompt_cache(TinyModel(), torch.tensor([[0, 0]]), [2, 1], "sum")
        self.assertGreater(score_good, score_bad)


if __name__ == "__main__":
    unittest.main()
