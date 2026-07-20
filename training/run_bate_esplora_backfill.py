"""Run the frozen BATE Esplora backfill over one persistent HTTPS connection.

This module changes transport only.  The frozen downloader still owns URL
scope, page validation, SQLite transactions, final audits, canonical output,
and the source manifest.
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import time
from typing import Any, Callable
import urllib.parse

from training import download_bitcoin_block_summaries as source


FROZEN_BASE_URL = "https://mempool.space/api"
USER_AGENT = "rllm-private-research/1.0"


ConnectionFactory = Callable[[], http.client.HTTPSConnection]
Sleep = Callable[[float], None]


class PersistentEsploraFetch:
    def __init__(
        self,
        cfg: source.Config,
        *,
        connection_factory: ConnectionFactory | None = None,
        sleep: Sleep = time.sleep,
    ) -> None:
        if cfg.base_url != FROZEN_BASE_URL:
            raise ValueError("persistent BATE transport requires the frozen host")
        self.cfg = cfg
        self.sleep = sleep
        self._connection_factory = connection_factory or (
            lambda: http.client.HTTPSConnection(
                "mempool.space", timeout=cfg.timeout_sec
            )
        )
        self._connection: http.client.HTTPSConnection | None = None

    def _reset(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None

    def close(self) -> None:
        self._reset()

    def _path(self, url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        expected = urllib.parse.urlparse(FROZEN_BASE_URL)
        prefix = f"{expected.path.rstrip('/')}/blocks/"
        suffix = parsed.path.removeprefix(prefix)
        if (
            parsed.scheme != expected.scheme
            or parsed.netloc != expected.netloc
            or parsed.params
            or parsed.query
            or parsed.fragment
            or not parsed.path.startswith(prefix)
            or not suffix.isdigit()
            or str(int(suffix)) != suffix
        ):
            raise ValueError("persistent BATE request escaped the frozen blocks path")
        return parsed.path

    def __call__(self, url: str) -> Any:
        path = self._path(url)
        for attempt in range(self.cfg.maximum_retries + 1):
            if self._connection is None:
                self._connection = self._connection_factory()
            try:
                self._connection.request(
                    "GET",
                    path,
                    headers={
                        "User-Agent": USER_AGENT,
                        "Accept": "application/json",
                        "Connection": "keep-alive",
                    },
                )
                response = self._connection.getresponse()
                raw = response.read()
                status = response.status
                retry_after = response.getheader("Retry-After")
                will_close = response.will_close
            except (
                http.client.HTTPException,
                OSError,
                TimeoutError,
            ):
                self._reset()
                if attempt >= self.cfg.maximum_retries:
                    raise
                self.sleep(self._retry_delay(attempt, None))
                continue
            if will_close:
                self._reset()
            if status == 200:
                return source._decode_payload(raw)
            retryable = status == 429 or 500 <= status <= 599
            self._reset()
            if not retryable or attempt >= self.cfg.maximum_retries:
                raise RuntimeError(
                    f"persistent Esplora HTTP status {status} for {path}"
                )
            self.sleep(self._retry_delay(attempt, retry_after))
        raise AssertionError("unreachable persistent retry loop")

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        try:
            header_delay = float(retry_after) if retry_after else math.nan
        except ValueError:
            header_delay = math.nan
        delay = header_delay if math.isfinite(header_delay) else 2.0**attempt
        return max(self.cfg.request_pause_sec, min(60.0, max(0.0, delay)))


def frozen_config(
    *,
    request_pause_sec: float = 0.10,
    maximum_retries: int = 20,
    timeout_sec: float = 30.0,
) -> source.Config:
    cfg = source.Config(
        base_url=FROZEN_BASE_URL,
        request_pause_sec=request_pause_sec,
        maximum_retries=maximum_retries,
        timeout_sec=timeout_sec,
    )
    source._validate_config(cfg)
    return cfg


def run(cfg: source.Config) -> dict[str, Any]:
    if cfg != frozen_config(
        request_pause_sec=cfg.request_pause_sec,
        maximum_retries=cfg.maximum_retries,
        timeout_sec=cfg.timeout_sec,
    ):
        raise ValueError("persistent BATE transport config changed frozen scope")
    client = PersistentEsploraFetch(cfg)
    try:
        return source.run(cfg, fetch=client, sleep=time.sleep)
    finally:
        client.close()


def parse_args() -> source.Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request-pause-sec", type=float, default=0.10
    )
    parser.add_argument("--maximum-retries", type=int, default=20)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    args = parser.parse_args()
    return frozen_config(**vars(args))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
