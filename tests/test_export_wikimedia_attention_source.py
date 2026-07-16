from __future__ import annotations

import gzip
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from training import export_wikimedia_attention_source as source


def _article_payload(article: str, views: tuple[int, int]) -> dict:
    return {
        "items": [
            {
                "project": "en.wikipedia",
                "article": article,
                "granularity": "daily",
                "timestamp": timestamp,
                "access": "all-access",
                "agent": "user",
                "views": value,
            }
            for timestamp, value in zip(("2020010100", "2020010200"), views)
        ]
    }


def _aggregate_payload() -> dict:
    return {
        "items": [
            {
                "project": "en.wikipedia",
                "access": "all-access",
                "agent": "user",
                "granularity": "daily",
                "timestamp": timestamp,
                "views": value,
            }
            for timestamp, value in (("2020010100", 1_000_000), ("2020010200", 2_000_000))
        ]
    }


def test_urls_pin_user_traffic_and_selection_dates() -> None:
    assert source.article_url("Bitcoin", "2020-01-01", "2020-01-02").endswith(
        "/en.wikipedia.org/all-access/user/Bitcoin/daily/20200101/20200102"
    )
    assert source.aggregate_url("2020-01-01", "2020-01-02").endswith(
        "/en.wikipedia.org/all-access/user/daily/2020010100/2020010200"
    )


def test_build_frame_normalizes_by_same_day_project_total() -> None:
    series = {
        article: source.parse_article_payload(_article_payload(article, (100, 200)), article)
        for article in source.ARTICLES
    }
    aggregate = source.parse_aggregate_payload(_aggregate_payload())
    frame, quality = source.build_daily_frame(
        series, aggregate, start="2020-01-01", end="2020-01-02"
    )
    assert quality["complete_days"] == 2
    assert frame["bitcoin_per_million"].tolist() == [100.0, 100.0]
    assert frame["source_complete"].tolist() == [1, 1]


def test_missing_article_day_is_explicit_and_fail_closed() -> None:
    series = {
        article: source.parse_article_payload(_article_payload(article, (100, 200)), article)
        for article in source.ARTICLES
    }
    series["Bitcoin"] = series["Bitcoin"].iloc[:1]
    frame, quality = source.build_daily_frame(
        series,
        source.parse_aggregate_payload(_aggregate_payload()),
        start="2020-01-01",
        end="2020-01-02",
    )
    assert quality["missing_by_column"]["bitcoin_views"] == 1
    assert frame["source_complete"].tolist() == [1, 0]
    assert pd.isna(frame.loc[1, "bitcoin_per_million"])


def test_gzip_output_is_deterministic(tmp_path: Path) -> None:
    frame = pd.DataFrame({"date": ["2020-01-01"], "views": [10]})
    one = tmp_path / "one.csv.gz"
    two = tmp_path / "two.csv.gz"
    source.deterministic_gzip_csv(frame, one)
    source.deterministic_gzip_csv(frame, two)
    assert one.read_bytes() == two.read_bytes()
    with gzip.open(one, "rt") as handle:
        assert handle.read() == "date,views\n2020-01-01,10\n"


def test_selection_exporter_refuses_2023_holdout() -> None:
    with pytest.raises(RuntimeError, match="sealed 2023 holdout"):
        source.validate_config(replace(source.Config(), end="2023-01-01"))
