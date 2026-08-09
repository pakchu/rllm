import numpy as np
import pandas as pd

from training import build_doj_digital_asset_enforcement_lifecycle_relay_support as support


def test_taxonomy_classifies_unambiguous_lifecycle_only():
    assert support.classify("Bitcoin exchange operator charged", "") == -1
    assert support.classify("Cryptocurrency fraudster sentenced", "") == 1
    assert support.classify("Bitcoin defendant charged and sentenced", "") == 0
    assert support.classify("Ordinary fraudster sentenced", "") == 0


def test_title_only_control_does_not_read_body():
    assert support.classify("Operator charged", "bitcoin exchange", title_only=True) == 0
    assert support.classify("Bitcoin operator charged", "", title_only=True) == -1


def test_rank_excludes_current_day():
    ranks = support.strict_prior_midrank(pd.Series(np.arange(127, dtype=float)))
    assert np.isnan(ranks.iloc[125])
    assert ranks.iloc[126] == 1.0


def test_transport_parallelism_is_bounded():
    assert support.FETCH_WORKERS == 4
    assert support.PAGE_SIZE == 50
