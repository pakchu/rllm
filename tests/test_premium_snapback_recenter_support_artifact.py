from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd


RESULT = Path("results/premium_snapback_recenter_support_2026-07-19.json")
PRIMARY = Path("data/premium_snapback_recenter_clocks_2020_2026.csv.gz")
EXPECTED_RESULT_SHA256 = (
    "f33708368b089dd588051971b8d17b4174aaac304ead7a30b07ebb3ee3520b4f"
)
EXPECTED_PRIMARY_SHA256 = (
    "cb209ed35f9baa08cc2fb3dd5bd60b8e747b1408c09507b774ca275e0b2b2db6"
)
EXPECTED_MANIFEST_HASH = (
    "cd22c414d395dea4a45a63daf93888e4703560d0e5625e2d3ea64c172acc3fc8"
)
EXPECTED_CONTROL_HASHES = {
    "direction_flip": "060d56d60451b2a6614aeac96678c419fc19c5c266d0c9f3e8aed847df4d07a1",
    "simple_level": "0c324be7985ce5fc12c44a942a6e85d60b0597aad93da30f330faf2bbf3160ad",
    "no_recenter": "268248c0fc3aefe1cb04b19ab1a0cefcf506358ab4db80257f4a6973f9228dd3",
    "extra_latency": "b17b3fa4be09993db2af85e4d6e2105330af2aeca3ca786f92dcd43922afefa7",
    "future_premium_placebo": "9c618f0da3ca91abd0d2628a4dc7645c890091f4af60d862164088613b3fe57f",
    "random": "c2035b7cedbc183446cf00f5c6267e15b59a1ba0398d96dd875c7d4cd9d45822",
}
EXPECTED_COMPARATOR_HASHES = {
    "psi_2016": "4e413c1eb6d656f541734ba17b2a010aceb50b508005acadd8e2cb8bbbb7e03a",
    "psi_8640": "58fde45f300949b5c55e6e3025be6a9a4fe95451d3476e8ab0c03a83e3d81410",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_clock(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        compression="gzip",
        parse_dates=[
            "path_start_time",
            "decision_time",
            "feature_available_time",
            "entry_time",
            "planned_exit_time",
        ],
    )


def test_frozen_support_result_is_outcome_blind_and_rejects_train_open() -> None:
    assert _sha256(RESULT) == EXPECTED_RESULT_SHA256
    result = json.loads(RESULT.read_text())
    assert result["manifest_hash"] == EXPECTED_MANIFEST_HASH
    protocol = result["protocol"]
    assert protocol["source_only"] is True
    assert protocol["outcomes_opened"] is False
    assert protocol["btc_execution_prices_opened"] is False
    assert protocol["funding_opened"] is False
    assert protocol["candidate_count"] == 1
    assert protocol["threshold_grid"] is False
    assert result["support_passes"] is False
    assert result["novelty_passes"] is True
    assert result["may_open_train"] is False

    train = result["support"]["train"]
    assert (train["total"], train["long"], train["short"]) == (305, 167, 138)
    assert train["subperiod_counts"] == {
        "2020_partial": 15,
        "2021": 92,
        "2022": 198,
    }
    assert result["support"]["test"]["total"] == 231
    assert result["support"]["eval"]["total"] == 611
    assert all(
        row["exact_jaccard"] <= 0.10
        and row["within_30m_primary_share"] <= 0.20
        for row in result["novelty"].values()
    )


def test_primary_and_control_clocks_match_frozen_contract() -> None:
    result: dict[str, Any] = json.loads(RESULT.read_text())
    assert _sha256(PRIMARY) == EXPECTED_PRIMARY_SHA256
    primary = _read_clock(PRIMARY)
    assert len(primary) == 1_147
    assert bool(primary["candidate"].eq("PSR-30/6").all())
    assert set(primary["direction"]) == {-1, 1}
    assert bool(
        cast(pd.Series, primary["entry_time"] - primary["decision_time"])
        .eq(pd.Timedelta(minutes=10))
        .all()
    )
    assert bool(
        cast(pd.Series, primary["planned_exit_time"] - primary["entry_time"])
        .eq(pd.Timedelta(minutes=30))
        .all()
    )
    assert bool((primary["feature_available_time"] < primary["entry_time"]).all())
    for _split, group in primary.groupby("split"):
        assert bool(
            (
                group["entry_time"].iloc[1:].reset_index(drop=True)
                >= group["planned_exit_time"].iloc[:-1].reset_index(drop=True)
            ).all()
        )

    for name, expected_hash in EXPECTED_CONTROL_HASHES.items():
        item = result["controls"][name]
        path = Path(item["path"])
        assert item["sha256"] == expected_hash == _sha256(path)
        assert len(_read_clock(path)) == item["rows"]

    for name, expected_hash in EXPECTED_COMPARATOR_HASHES.items():
        item = result["novelty_comparators"][name]
        path = Path(item["path"])
        assert item["outcome_evaluation_allowed"] is False
        assert item["sha256"] == expected_hash == _sha256(path)
        psi = _read_clock(path)
        assert len(psi) == item["rows"]
        assert bool(psi["entry_time"].equals(psi["decision_time"]))
        assert bool((psi["feature_available_time"] > psi["entry_time"]).all())
        assert bool(
            cast(pd.Series, psi["planned_exit_time"] - psi["entry_time"])
            .eq(pd.Timedelta(hours=8))
            .all()
        )
