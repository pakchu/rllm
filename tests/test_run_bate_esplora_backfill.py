from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from typing import Any

import pytest

from training import run_bate_esplora_backfill as transport


class FakeResponse:
    def __init__(
        self,
        status: int,
        payload: Any,
        *,
        retry_after: str | None = None,
        will_close: bool = False,
    ) -> None:
        self.status = status
        self.payload = payload
        self.retry_after = retry_after
        self.will_close = will_close

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def getheader(self, name: str) -> str | None:
        return self.retry_after if name.lower() == "retry-after" else None


class FakeConnection:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.closed = False

    def request(self, method: str, path: str, *, headers: dict[str, str]) -> None:
        self.requests.append((method, path, headers))

    def getresponse(self) -> FakeResponse:
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def _client(
    responses: list[FakeResponse],
    *,
    maximum_retries: int = 2,
    connections: list[FakeConnection] | None = None,
    sleeps: list[float] | None = None,
) -> tuple[transport.PersistentEsploraFetch, list[FakeConnection], list[float]]:
    cfg = transport.frozen_config(
        request_pause_sec=0.1,
        maximum_retries=maximum_retries,
    )
    made = connections if connections is not None else []
    delays = sleeps if sleeps is not None else []

    def factory() -> FakeConnection:
        connection = FakeConnection(responses)
        made.append(connection)
        return connection

    client = transport.PersistentEsploraFetch(
        cfg,
        connection_factory=factory,  # type: ignore[arg-type]
        sleep=delays.append,
    )
    return client, made, delays


def test_persistent_client_reuses_connection_and_exact_path() -> None:
    payloads = [
        [{"height": 823_785, "difficulty": 1.25}],
        [{"height": 823_775, "difficulty": 2.5}],
    ]
    responses = [FakeResponse(200, payload) for payload in payloads]
    client, connections, sleeps = _client(responses)
    first = client("https://mempool.space/api/blocks/823785")
    second = client("https://mempool.space/api/blocks/823775")
    assert len(connections) == 1
    assert [request[1] for request in connections[0].requests] == [
        "/api/blocks/823785",
        "/api/blocks/823775",
    ]
    assert first[0]["difficulty"] == Decimal("1.25")
    assert second[0]["difficulty"] == Decimal("2.5")
    assert sleeps == []
    client.close()
    assert connections[0].closed is True


def test_retryable_status_resets_connection_and_honors_retry_after() -> None:
    responses = [
        FakeResponse(429, {"error": "slow"}, retry_after="3"),
        FakeResponse(200, []),
    ]
    client, connections, sleeps = _client(responses)
    assert client("https://mempool.space/api/blocks/823785") == []
    assert len(connections) == 2
    assert connections[0].closed is True
    assert sleeps == [3.0]


@pytest.mark.parametrize(
    "url",
    [
        "https://blockstream.info/api/blocks/823785",
        "http://mempool.space/api/blocks/823785",
        "https://mempool.space/api/v1/blocks/823785",
        "https://mempool.space/api/blocks/823785?future=1",
        "https://mempool.space/api/blocks/not-a-height",
        "https://mempool.space/api/blocks/0823785",
    ],
)
def test_request_cannot_escape_frozen_host_or_path(url: str) -> None:
    client, connections, _ = _client([FakeResponse(200, [])])
    with pytest.raises(ValueError, match="escaped the frozen blocks path"):
        client(url)
    assert connections == []


def test_nonretryable_status_fails_closed() -> None:
    client, connections, sleeps = _client([FakeResponse(404, {})])
    with pytest.raises(RuntimeError, match="HTTP status 404"):
        client("https://mempool.space/api/blocks/823785")
    assert connections[0].closed is True
    assert sleeps == []


def test_frozen_config_exposes_only_transport_tuning() -> None:
    cfg = transport.frozen_config(
        request_pause_sec=0.0,
        maximum_retries=3,
        timeout_sec=12.0,
    )
    assert cfg.base_url == transport.FROZEN_BASE_URL
    assert cfg.start_height == transport.source.FROZEN_START_HEIGHT
    assert cfg.end_height == transport.source.FROZEN_END_HEIGHT
    assert cfg.end_timestamp_exclusive == transport.source.FIRST_2024_TIMESTAMP
    with pytest.raises(ValueError, match="frozen host"):
        transport.PersistentEsploraFetch(
            replace(cfg, base_url="https://blockstream.info/api")
        )


def test_run_rejects_any_scope_change_before_transport(monkeypatch) -> None:
    cfg = replace(transport.frozen_config(), end_height=823_784)
    monkeypatch.setattr(
        transport,
        "PersistentEsploraFetch",
        lambda _: pytest.fail("scope drift must fail before creating transport"),
    )
    with pytest.raises(ValueError, match="changed frozen scope"):
        transport.run(cfg)
