from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from training import download_deribit_btc_option_deliveries as delivery


def _ms(value: str) -> int:
    return int(pd.Timestamp(value, tz="UTC").timestamp() * 1000)


def _row(
    instrument: str,
    timestamp: str,
    *,
    position: float,
    index_price: float = 100.0,
    mark_price: float = 0.0,
) -> dict[str, object]:
    return {
        "position": position,
        "timestamp": _ms(timestamp),
        "type": "delivery",
        "instrument_name": instrument,
        "index_price": index_price,
        "mark_price": mark_price,
        "profit_loss": 0.0,
        "session_profit_loss": 1.0,
    }


def _expiry_rows(date_code: str, date: str) -> list[dict[str, object]]:
    timestamp = f"{date} 08:00:00.100"
    return [
        _row(f"BTC-{date_code}", timestamp, position=1000.0),
        _row(f"BTC-{date_code}-90-C", timestamp, position=10.0),
        _row(f"BTC-{date_code}-110-C", timestamp, position=5.0),
        _row(f"BTC-{date_code}-90-P", timestamp, position=7.0),
        _row(f"BTC-{date_code}-110-P", timestamp, position=20.0),
        _row(f"BTC-{date_code}-100-C", timestamp, position=3.0),
    ]


def _cfg(tmp_path: Path, **changes: object) -> delivery.Config:
    cfg = delivery.Config(
        output_csv=str(tmp_path / "source.csv.gz"),
        manifest_output=str(tmp_path / "manifest.json"),
        start="2019-01-01",
        end_exclusive="2021-01-01",
        page_size=6,
        request_pause_sec=0.0,
    )
    return replace(cfg, **changes)


def _payload(
    rows: list[dict[str, object]], continuation: str | None
) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "result": {"settlements": rows, "continuation": continuation},
    }


def test_download_aggregates_terminal_delta_release_and_excludes_future(
    tmp_path: Path,
) -> None:
    recent = _expiry_rows("3JAN20", "2020-01-03")
    older = _expiry_rows("27DEC19", "2019-12-27")
    before = _expiry_rows("28DEC18", "2018-12-28")
    pages = {
        None: _payload(recent, "next"),
        "next": _payload(older, "old"),
        "old": _payload(before, None),
    }
    calls: list[dict[str, object]] = []

    def fetch(params: dict[str, object]) -> dict[str, object]:
        calls.append(params.copy())
        return pages[params.get("continuation")]

    cfg = _cfg(tmp_path)
    manifest = delivery.run(cfg, fetch=fetch, sleep=lambda _: None)
    assert [call.get("continuation") for call in calls] == [None, "next", "old"]
    assert all(call["search_start_timestamp"] == _ms("2021-01-01") - 1 for call in calls)
    assert manifest["source_audit"]["option_rows_selected"] == 10
    assert manifest["source_audit"]["futures_rows_excluded"] == 2
    assert manifest["source_audit"]["expiry_events"] == 2
    assert manifest["source_audit"]["crossed_start_boundary"] is True
    assert manifest["outcome_boundary"]["binance_market_rows_loaded"] == 0
    assert manifest["outcome_boundary"]["raw_deribit_rows_persisted"] is False
    assert manifest["causal_availability"][
        "source_observation_latency_seconds"
    ] == 3900

    frame = pd.read_csv(cfg.output_csv)
    assert frame["release_side"].tolist() == [1, 1]
    assert frame["itm_call_position"].tolist() == [10.0, 10.0]
    assert frame["itm_put_position"].tolist() == [20.0, 20.0]
    assert frame["net_release_position"].tolist() == [10.0, 10.0]
    assert frame["atm_position"].tolist() == [3.0, 3.0]
    assert frame["option_count"].tolist() == [5, 5]
    assert frame["source_observation_earliest"].str.endswith("09:05:00.000000").all()
    with gzip.open(cfg.output_csv, "rt", encoding="utf-8") as handle:
        public = handle.read() + json.dumps(manifest)
    assert "BTC-3JAN20-90-C" not in public


def test_aggregate_maps_put_release_long_and_call_release_short() -> None:
    cfg = delivery.Config()
    rows = _expiry_rows("3JAN20", "2020-01-03")
    aggregate, _ = delivery.aggregate_deliveries(rows, cfg)
    assert aggregate.loc[0, "release_side"] == 1
    changed = [dict(row) for row in rows]
    for row in changed:
        if row["instrument_name"] == "BTC-3JAN20-90-C":
            row["position"] = 30.0
    aggregate, _ = delivery.aggregate_deliveries(changed, cfg)
    assert aggregate.loc[0, "net_release_position"] == -10.0
    assert aggregate.loc[0, "release_side"] == -1


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda row: row.update(position=0.0), "position must be positive"),
        (lambda row: row.update(type="settlement"), "another settlement type"),
        (
            lambda row: row.update(instrument_name="BTC-BAD-90-C"),
            "unsupported Deribit",
        ),
        (
            lambda row: row.update(timestamp=_ms("2020-01-03 08:00:06")),
            "outside the frozen 08:00 window",
        ),
    ],
)
def test_parser_rejects_source_semantic_defects(mutator, message: str) -> None:
    rows = _expiry_rows("3JAN20", "2020-01-03")
    mutator(rows[1])
    with pytest.raises(ValueError, match=message):
        delivery.aggregate_deliveries(rows, delivery.Config())


@pytest.mark.parametrize(
    ("month", "number"),
    [
        ("JAN", 1),
        ("FEB", 2),
        ("MAR", 3),
        ("APR", 4),
        ("MAY", 5),
        ("JUN", 6),
        ("JUL", 7),
        ("AUG", 8),
        ("SEP", 9),
        ("OCT", 10),
        ("NOV", 11),
        ("DEC", 12),
    ],
)
def test_expiry_parser_is_locale_independent(month: str, number: int) -> None:
    date = f"2020-{number:02d}-03"
    rows = _expiry_rows(f"3{month}20", date)
    aggregate, _ = delivery.aggregate_deliveries(rows, delivery.Config())
    assert aggregate.loc[0, "expiry_time"] == pd.Timestamp(
        f"{date} 08:00:00", tz="UTC"
    )


def test_aggregate_rejects_duplicate_instrument_and_index_disagreement() -> None:
    rows = _expiry_rows("3JAN20", "2020-01-03")
    with pytest.raises(RuntimeError, match="duplicate instruments"):
        delivery.aggregate_deliveries(rows + [dict(rows[1])], delivery.Config())
    changed = [dict(row) for row in rows]
    changed[2]["index_price"] = 101.0
    with pytest.raises(RuntimeError, match="delivery index"):
        delivery.aggregate_deliveries(changed, delivery.Config())


def test_download_rejects_forward_pagination_and_continuation_loop(
    tmp_path: Path,
) -> None:
    recent = _expiry_rows("3JAN20", "2020-01-03")
    newer = _expiry_rows("10JAN20", "2020-01-10")
    pages = iter([_payload(recent, "same"), _payload(newer, None)])
    with pytest.raises(RuntimeError, match="moved forward"):
        delivery.download(_cfg(tmp_path), fetch=lambda _: next(pages), sleep=lambda _: None)

    pages = iter([_payload(recent, "same"), _payload(recent, "same")])
    with pytest.raises(RuntimeError, match="continuation loop"):
        delivery.download(
            _cfg(tmp_path / "loop"),
            fetch=lambda _: next(pages),
            sleep=lambda _: None,
        )


def test_download_rejects_newest_first_and_end_boundary(tmp_path: Path) -> None:
    rows = _expiry_rows("3JAN20", "2020-01-03")
    reversed_rows = list(reversed(rows))
    # Equal timestamps are a valid tie, so add one older row ahead of newer rows.
    reversed_rows[0] = _row(
        "BTC-27DEC19-90-C", "2019-12-27 08:00:00.100", position=1.0
    )
    with pytest.raises(RuntimeError, match="not newest-first"):
        delivery.download(
            _cfg(tmp_path),
            fetch=lambda _: _payload(reversed_rows, None),
            sleep=lambda _: None,
        )


def test_download_rejects_history_that_does_not_cross_start(
    tmp_path: Path,
) -> None:
    rows = _expiry_rows("3JAN20", "2020-01-03")
    with pytest.raises(RuntimeError, match="before crossing"):
        delivery.download(
            _cfg(tmp_path),
            fetch=lambda _: _payload(rows, None),
            sleep=lambda _: None,
        )

    pages = iter([_payload(rows, "next"), _payload([], None)])
    with pytest.raises(RuntimeError, match="before crossing"):
        delivery.download(
            _cfg(tmp_path / "empty"),
            fetch=lambda _: next(pages),
            sleep=lambda _: None,
        )
    post_end = _expiry_rows("1JAN21", "2021-01-01")
    with pytest.raises(RuntimeError, match="end boundary"):
        delivery.download(
            _cfg(tmp_path / "end"),
            fetch=lambda _: _payload(post_end, None),
            sleep=lambda _: None,
        )


def test_configuration_and_payload_contracts_fail_closed(tmp_path: Path) -> None:
    for page_size in (0, 1001):
        with pytest.raises(ValueError, match="page_size"):
            delivery.download(
                _cfg(tmp_path / str(page_size), page_size=page_size),
                fetch=lambda _: _payload([], None),
            )
    with pytest.raises(ValueError, match="currency/type"):
        delivery.download(
            _cfg(tmp_path / "currency", currency="ETH"),
            fetch=lambda _: _payload([], None),
        )
    with pytest.raises(RuntimeError, match="API error"):
        delivery._parse_payload({"error": {"message": "bad"}}, page_size=1)
    with pytest.raises(RuntimeError, match="not a list"):
        delivery._parse_payload(
            {"result": {"settlements": {}}}, page_size=1
        )


def test_deterministic_aggregate_bytes(tmp_path: Path) -> None:
    frame, _ = delivery.aggregate_deliveries(
        _expiry_rows("3JAN20", "2020-01-03"), delivery.Config()
    )
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    delivery._write_deterministic_csv(first, frame)
    delivery._write_deterministic_csv(second, frame)
    assert first.read_bytes() == second.read_bytes()


def test_manifest_bytes_and_hash_are_deterministic(tmp_path: Path) -> None:
    recent = _expiry_rows("3JAN20", "2020-01-03")
    before = _expiry_rows("28DEC18", "2018-12-28")
    pages = {
        None: _payload(recent, "old"),
        "old": _payload(before, None),
    }

    def fetch(params: dict[str, object]) -> dict[str, object]:
        return pages[params.get("continuation")]

    cfg = _cfg(tmp_path)
    first = delivery.run(cfg, fetch=fetch, sleep=lambda _: None)
    first_bytes = Path(cfg.manifest_output).read_bytes()
    second = delivery.run(cfg, fetch=fetch, sleep=lambda _: None)
    assert Path(cfg.manifest_output).read_bytes() == first_bytes
    assert first["manifest_hash"] == second["manifest_hash"]
