import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from training import build_high_volatility_cross_maturity_depth_migration_relay_support as support


def test_contract_pair_rolls_at_exact_expiry() -> None:
    before = support.contract_pair(pd.Timestamp("2023-03-31T07:59:59Z"))
    after = support.contract_pair(pd.Timestamp("2023-03-31T08:00:00Z"))
    assert before == ("BTCUSD_230331", "BTCUSD_230630")
    assert after == ("BTCUSD_230630", "BTCUSD_230929")


def test_parse_archive_reduces_complete_snapshots() -> None:
    timestamps = [pd.Timestamp("2023-01-01T00:00:03Z"), pd.Timestamp("2023-01-01T00:00:31Z")]
    rows = []
    for timestamp in timestamps:
        for level in support.LEVELS:
            rows.append({"timestamp": timestamp, "percentage": level, "depth": 100 + level, "notional": 1000 + level})
    csv = pd.DataFrame(rows).to_csv(index=False).encode()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("BTCUSD_230331-bookDepth-2023-01-01.csv", csv)
    job = support.Job(pd.Timestamp("2023-01-01T00:00:00Z"), "BTCUSD_230331")
    result = support.parse_archive(payload.getvalue(), job)
    assert len(result) == 1
    assert result.iloc[0].snapshots == 2
    assert result.iloc[0].pressure == pytest.approx(np.log(99 / 101))
    assert result.iloc[0].mass == 200


def test_parse_archive_rejects_only_malformed_snapshot() -> None:
    rows = []
    for timestamp in (pd.Timestamp("2023-01-01T00:00:03Z"), pd.Timestamp("2023-01-01T00:00:31Z")):
        for level in support.LEVELS:
            rows.append({"timestamp": timestamp, "percentage": level, "depth": 100 + level, "notional": 1000 + level})
    rows[-1]["depth"] = -1
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("x.csv", pd.DataFrame(rows).to_csv(index=False).encode())
    job = support.Job(pd.Timestamp("2023-01-01T00:00:00Z"), "BTCUSD_230331")
    result = support.parse_archive(payload.getvalue(), job)
    assert result.iloc[0].snapshots == 1


def test_checksum_parser_binds_filename() -> None:
    digest = hashlib.sha256(b"x").hexdigest()
    assert support.expected_checksum(f"{digest}  x.zip\n".encode(), "x.zip") == digest


def test_strict_prior_rank_excludes_current() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    ranked = support.strict_prior_midrank(values, lookback=3, minimum=2)
    assert np.isnan(ranked.iloc[0]) and np.isnan(ranked.iloc[1])
    assert ranked.iloc[2] == 1.0 and ranked.iloc[3] == 1.0
