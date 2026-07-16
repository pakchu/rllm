"""Download and freeze selection-only Wikimedia attention inputs.

The default request ends on 2022-12-31.  This exporter deliberately refuses
to cross the preregistered 2023 holdout boundary.  A later holdout exporter
must be invoked only after a selection policy manifest has been committed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.preregister_wikimedia_attention_divergence_alpha import (
    ACCESS,
    AGENT,
    ARTICLES,
    DEFAULT_OUTPUT as DEFAULT_PREREGISTRATION,
    PROJECT,
    SELECTION_END,
    canonical_hash,
    validate_manifest,
)


DEFAULT_OUTPUT = "data/wikimedia_crypto_attention_daily_2020_2022.csv.gz"
DEFAULT_MANIFEST = (
    "results/wikimedia_crypto_attention_source_manifest_2020_2022_2026-07-16.json"
)
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2022-12-31"
USER_AGENT = "rllm-wikimedia-alpha-research/1.0 (https://github.com/pakchu/rllm)"


@dataclass(frozen=True)
class Config:
    output: str = DEFAULT_OUTPUT
    manifest_output: str = DEFAULT_MANIFEST
    preregistration: str = DEFAULT_PREREGISTRATION
    start: str = DEFAULT_START
    end: str = DEFAULT_END
    timeout_seconds: float = 60.0
    retries: int = 4
    retry_delay_seconds: float = 2.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_token(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def article_url(article: str, start: str, end: str) -> str:
    encoded = urllib.parse.quote(article.replace(" ", "_"), safe="")
    return (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{PROJECT}/{ACCESS}/{AGENT}/{encoded}/daily/"
        f"{_date_token(start)}/{_date_token(end)}"
    )


def aggregate_url(start: str, end: str) -> str:
    return (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/"
        f"{PROJECT}/{ACCESS}/{AGENT}/daily/"
        f"{_date_token(start)}00/{_date_token(end)}00"
    )


def fetch_json(url: str, cfg: Config) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(cfg.retries):
        try:
            with urllib.request.urlopen(request, timeout=cfg.timeout_seconds) as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("Wikimedia response must be a JSON object")
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < cfg.retries:
                time.sleep(cfg.retry_delay_seconds * (attempt + 1))
    raise RuntimeError(f"Wikimedia request failed after {cfg.retries} attempts: {url}") from last_error


def _parse_timestamp(value: Any) -> pd.Timestamp:
    token = str(value)
    if len(token) != 10 or not token.isdigit() or not token.endswith("00"):
        raise ValueError(f"unexpected Wikimedia daily timestamp: {value!r}")
    return pd.Timestamp(datetime.strptime(token, "%Y%m%d%H"))


def parse_article_payload(payload: dict[str, Any], article: str) -> pd.Series:
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Wikimedia article payload has no items list: {article}")
    values: dict[pd.Timestamp, int] = {}
    expected_project = PROJECT.removesuffix(".org")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Wikimedia article item must be an object")
        if str(item.get("project")) != expected_project:
            raise ValueError(f"article project mismatch: {item.get('project')}")
        if str(item.get("access")) != ACCESS or str(item.get("agent")) != AGENT:
            raise ValueError("article access/agent mismatch")
        if str(item.get("granularity")) != "daily":
            raise ValueError("article granularity mismatch")
        timestamp = _parse_timestamp(item.get("timestamp"))
        views = int(item.get("views"))
        if views < 0 or timestamp in values:
            raise ValueError("article views must be nonnegative and dates unique")
        values[timestamp] = views
    return pd.Series(values, name=article, dtype="int64").sort_index()


def parse_aggregate_payload(payload: dict[str, Any]) -> pd.Series:
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Wikimedia aggregate payload has no items list")
    values: dict[pd.Timestamp, int] = {}
    expected_project = PROJECT.removesuffix(".org")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Wikimedia aggregate item must be an object")
        if str(item.get("project")) != expected_project:
            raise ValueError(f"aggregate project mismatch: {item.get('project')}")
        if str(item.get("access")) != ACCESS or str(item.get("agent")) != AGENT:
            raise ValueError("aggregate access/agent mismatch")
        if str(item.get("granularity")) != "daily":
            raise ValueError("aggregate granularity mismatch")
        timestamp = _parse_timestamp(item.get("timestamp"))
        views = int(item.get("views"))
        if views <= 0 or timestamp in values:
            raise ValueError("aggregate views must be positive and dates unique")
        values[timestamp] = views
    return pd.Series(values, name="project_user_views", dtype="int64").sort_index()


def build_daily_frame(
    article_series: dict[str, pd.Series],
    aggregate: pd.Series,
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected = pd.date_range(start, end, freq="D")
    frame = pd.DataFrame(index=expected)
    for article in ARTICLES:
        if article not in article_series:
            raise ValueError(f"missing fixed article series: {article}")
        frame[f"{article.lower()}_views"] = article_series[article].reindex(expected)
    frame["project_user_views"] = aggregate.reindex(expected)
    missing = {
        column: int(frame[column].isna().sum())
        for column in frame.columns
    }
    finite = frame.notna().all(axis=1)
    denominator = frame["project_user_views"].where(frame["project_user_views"] > 0)
    for article in ARTICLES:
        raw = frame[f"{article.lower()}_views"]
        frame[f"{article.lower()}_per_million"] = raw / denominator * 1_000_000.0
    normalized_columns = [f"{article.lower()}_per_million" for article in ARTICLES]
    normalized = frame[normalized_columns].to_numpy(float)
    if not np.isfinite(normalized[finite.to_numpy(bool)]).all():
        raise ValueError("finite Wikimedia rows produced invalid normalization")
    frame.insert(0, "date", expected.strftime("%Y-%m-%d"))
    frame["source_complete"] = finite.astype(np.int8).to_numpy()
    return frame.reset_index(drop=True), {
        "expected_days": int(len(expected)),
        "complete_days": int(finite.sum()),
        "missing_by_column": missing,
        "first_date": str(expected.min().date()),
        "last_date": str(expected.max().date()),
    }


def deterministic_gzip_csv(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = frame.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")
    with output.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as zipped:
            zipped.write(raw)


def validate_config(cfg: Config) -> None:
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    if start < pd.Timestamp("2015-07-01") or end < start:
        raise ValueError("invalid Wikimedia pageview date range")
    if end >= pd.Timestamp(SELECTION_END):
        raise RuntimeError("selection exporter refuses to open the sealed 2023 holdout")
    if cfg.retries < 1:
        raise ValueError("retries must be positive")


def run(cfg: Config) -> dict[str, Any]:
    validate_config(cfg)
    prereg_path = Path(cfg.preregistration)
    prereg = json.loads(prereg_path.read_text())
    validate_manifest(prereg)
    requests: list[dict[str, Any]] = []
    article_series: dict[str, pd.Series] = {}
    for article in ARTICLES:
        url = article_url(article, cfg.start, cfg.end)
        payload = fetch_json(url, cfg)
        article_series[article] = parse_article_payload(payload, article)
        requests.append(
            {
                "kind": "per_article",
                "article": article,
                "url": url,
                "payload_sha256": canonical_hash(payload),
                "items": int(len(payload["items"])),
            }
        )
    url = aggregate_url(cfg.start, cfg.end)
    aggregate_payload = fetch_json(url, cfg)
    aggregate = parse_aggregate_payload(aggregate_payload)
    requests.append(
        {
            "kind": "aggregate",
            "url": url,
            "payload_sha256": canonical_hash(aggregate_payload),
            "items": int(len(aggregate_payload["items"])),
        }
    )
    frame, quality = build_daily_frame(
        article_series, aggregate, start=cfg.start, end=cfg.end
    )
    deterministic_gzip_csv(frame, cfg.output)
    manifest_core: dict[str, Any] = {
        "protocol_version": "wikimedia_attention_source_freeze_v1",
        "phase": "selection_only_2020_2022",
        "outcomes_opened": False,
        "start": cfg.start,
        "end": cfg.end,
        "selection_end_exclusive": SELECTION_END,
        "preregistration_path": str(prereg_path),
        "preregistration_file_sha256": sha256_file(prereg_path),
        "preregistration_manifest_hash": prereg["manifest_hash"],
        "user_agent": USER_AGENT,
        "requests": requests,
        "quality": quality,
        "output": str(Path(cfg.output)),
        "output_bytes": Path(cfg.output).stat().st_size,
        "output_sha256": sha256_file(cfg.output),
        "historical_snapshot_is_point_in_time": False,
        "known_traffic_loss_is_normalized_by_project_total": True,
        "future_data_requested": False,
    }
    manifest = {
        **manifest_core,
        "manifest_hash": canonical_hash(manifest_core),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = Path(cfg.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("output", "manifest_output", "preregistration", "start", "end"):
        parser.add_argument(f"--{field.replace('_', '-')}", default=getattr(Config(), field))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(Config(**vars(args)))
    print(
        json.dumps(
            {
                "output": manifest["output"],
                "manifest_hash": manifest["manifest_hash"],
                "quality": manifest["quality"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
