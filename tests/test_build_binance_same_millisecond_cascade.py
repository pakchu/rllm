from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from training import build_binance_aggtrade_microstructure as base
from training import build_binance_same_millisecond_cascade as builder


def _archive(
    day: date,
    *,
    price: float = 100.0,
    internal_gap: bool = False,
    underlying_gap: bool = False,
    underlying_overlap: bool = False,
    start_agg_id: int = 1,
    start_trade_id: int = 10,
    timestamp_day: date | None = None,
) -> bytes:
    timestamp_ms = int(pd.Timestamp(timestamp_day or day, tz="UTC").timestamp() * 1_000)
    offsets = [0, 2, 3] if internal_gap else [0, 1, 2]
    identifiers = [start_agg_id + offset for offset in offsets]
    if underlying_overlap:
        trade_offsets = [0, 1, 1]
    else:
        trade_offsets = [0, 2, 3] if underlying_gap else offsets
    trade_ids = [start_trade_id + offset for offset in trade_offsets]
    rows = [
        [identifiers[0], price, 1.0, trade_ids[0], trade_ids[0], timestamp_ms, "false"],
        [identifiers[1], price + 1.0, 1.0, trade_ids[1], trade_ids[1], timestamp_ms + 1, "false"],
        [identifiers[2], price + 2.0, 1.0, trade_ids[2], trade_ids[2], timestamp_ms + 1, "false"],
    ]
    text = io.StringIO()
    pd.DataFrame(rows, columns=base.RAW_COLUMNS).to_csv(text, index=False)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-aggTrades-test.csv", text.getvalue())
    return output.getvalue()


def _full_day_archive(
    day: date,
    *,
    start_agg_id: int,
    start_trade_id: int,
) -> bytes:
    timestamp_ms = int(pd.Timestamp(day, tz="UTC").timestamp() * 1_000)
    rows = [
        [
            start_agg_id + position,
            100.0,
            1.0,
            start_trade_id + position,
            start_trade_id + position,
            timestamp_ms + position * 300_000,
            "false",
        ]
        for position in range(288)
    ]
    text = io.StringIO()
    pd.DataFrame(rows, columns=base.RAW_COLUMNS).to_csv(text, index=False)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-aggTrades-test.csv", text.getvalue())
    return output.getvalue()


class _FakeFetcher:
    def __init__(self, payloads: dict[date, bytes]) -> None:
        self.payloads = payloads

    def __call__(self, url: str, *, retries: int, timeout: int) -> bytes:
        del retries, timeout
        for day, payload in self.payloads.items():
            if day.isoformat() not in url:
                continue
            if url.endswith(".CHECKSUM"):
                digest = hashlib.sha256(payload).hexdigest()
                return f"{digest}  archive.zip\n".encode()
            return payload
        raise AssertionError(f"unexpected URL: {url}")


def _contract(
    payloads: dict[date, bytes],
    *,
    gap_days: frozenset[str] = frozenset(),
    underlying_overlap_counts: dict[str, int] | None = None,
    zero_bins: frozenset[pd.Timestamp] = frozenset(),
) -> builder.SourceContract:
    facts: dict[str, dict[str, int]] = {}
    for day, payload in payloads.items():
        raw = base.read_archive(payload)
        observed = builder.aggregate_same_millisecond_five_minute(raw)
        facts[day.isoformat()] = {
            "agg_trade_rows": int(len(raw)),
            "five_minute_rows": int(len(observed)),
            "first_agg_trade_id": int(raw["agg_trade_id"].iloc[0]),
            "last_agg_trade_id": int(raw["agg_trade_id"].iloc[-1]),
            "first_underlying_trade_id": int(raw["first_trade_id"].iloc[0]),
            "last_underlying_trade_id": int(raw["last_trade_id"].iloc[-1]),
        }
    return builder.SourceContract(
        archive_sha256_by_date={
            day.isoformat(): hashlib.sha256(payload).hexdigest()
            for day, payload in payloads.items()
        },
        archive_facts_by_date=facts,
        aggregate_gap_days=gap_days,
        underlying_overlap_counts=underlying_overlap_counts or {},
        verified_zero_volume_bins=zero_bins,
    )


def test_build_is_outcome_blind_deterministic_and_rechecks_checksum(tmp_path: Path) -> None:
    day = date(2021, 1, 1)
    fetcher = _FakeFetcher({day: _archive(day)})
    cfg = builder.BuildConfig(
        start="2021-01-01",
        end="2021-01-02",
        output_dir=str(tmp_path),
        workers=1,
    )
    source_contract = _contract(fetcher.payloads)
    first = builder.build(cfg, fetcher=fetcher, source_contract=source_contract)
    second = builder.build(cfg, fetcher=fetcher, source_contract=source_contract)
    assert first["combined_sha256"] == second["combined_sha256"]
    assert first["protocol"]["outcomes_opened"] is False
    assert first["protocol"]["raw_archives_persisted"] is False
    assert first["columns"] == list(builder.BAR_COLUMNS)
    output = pd.read_csv(first["combined_output"], compression="gzip")
    assert len(output) == 288
    assert output["source_observed"].sum() == 1
    assert output["source_complete"].sum() == 1

    fetcher.payloads[day] = _archive(day, price=101.0)
    with pytest.raises(ValueError, match="changed after source audit"):
        builder.build(cfg, fetcher=fetcher, source_contract=source_contract)
    manifest = json.loads((tmp_path / "build_manifest.json").read_text())
    assert manifest["protocol"]["selected_group"].startswith("maximum quote notional")


def test_known_gap_day_is_fully_quarantined(tmp_path: Path) -> None:
    day = date(2021, 2, 9)
    payload = _archive(day, internal_gap=True)
    fetcher = _FakeFetcher({day: payload})
    contract = _contract({day: payload}, gap_days=frozenset({day.isoformat()}))
    cfg = builder.BuildConfig(
        start=day.isoformat(),
        end="2021-02-10",
        output_dir=str(tmp_path),
        workers=1,
    )
    result = builder.build(cfg, fetcher=fetcher, source_contract=contract)
    output = pd.read_csv(result["combined_output"], compression="gzip")
    assert output["source_gap_day"].all()
    assert not output["source_complete"].any()


def test_underlying_id_holes_do_not_invent_aggregate_source_gap(tmp_path: Path) -> None:
    day = date(2020, 3, 1)
    payload = _archive(day, underlying_gap=True)
    fetcher = _FakeFetcher({day: payload})
    cfg = builder.BuildConfig(
        start=day.isoformat(),
        end="2020-03-02",
        output_dir=str(tmp_path),
        workers=1,
    )

    result = builder.build(
        cfg,
        fetcher=fetcher,
        source_contract=_contract({day: payload}),
    )

    output = pd.read_csv(result["combined_output"], compression="gzip")
    assert not output["source_gap_day"].any()
    assert output["source_complete"].sum() == 1


def test_frozen_underlying_overlap_day_is_fully_quarantined(tmp_path: Path) -> None:
    day = date(2020, 1, 15)
    payload = _archive(day, underlying_overlap=True)
    fetcher = _FakeFetcher({day: payload})
    cfg = builder.BuildConfig(
        start=day.isoformat(),
        end="2020-01-16",
        output_dir=str(tmp_path),
        workers=1,
    )
    result = builder.build(
        cfg,
        fetcher=fetcher,
        source_contract=_contract(
            {day: payload},
            underlying_overlap_counts={day.isoformat(): 1},
        ),
    )
    output = pd.read_csv(result["combined_output"], compression="gzip")
    assert output["source_gap_day"].all()
    assert not output["source_complete"].any()


def test_unregistered_underlying_overlap_fails_closed(tmp_path: Path) -> None:
    day = date(2020, 1, 15)
    payload = _archive(day, underlying_overlap=True)
    with pytest.raises(ValueError, match="underlying-trade ID overlap contract changed"):
        builder.build(
            builder.BuildConfig(
                start=day.isoformat(),
                end="2020-01-16",
                output_dir=str(tmp_path),
                workers=1,
            ),
            fetcher=_FakeFetcher({day: payload}),
            source_contract=_contract({day: payload}),
        )


def test_resume_metadata_cannot_redirect_verified_output(tmp_path: Path) -> None:
    day = date(2021, 1, 1)
    payload = _archive(day)
    fetcher = _FakeFetcher({day: payload})
    contract = _contract({day: payload})
    cfg = builder.BuildConfig(
        start=day.isoformat(),
        end="2021-01-02",
        output_dir=str(tmp_path),
        workers=1,
    )
    first = builder.build(cfg, fetcher=fetcher, source_contract=contract)
    monthly_metadata = next((tmp_path / "monthly").glob("*.json"))
    metadata = json.loads(monthly_metadata.read_text())
    alternate = tmp_path / "redirect.csv.gz"
    redirected = pd.read_csv(first["combined_output"], compression="gzip")
    redirected.loc[0, "last_price"] = 999.0
    redirected.to_csv(alternate, index=False, compression="gzip")
    metadata["output"] = str(alternate)
    monthly_metadata.write_text(json.dumps(metadata))

    rebuilt = builder.build(cfg, fetcher=fetcher, source_contract=contract)
    output = pd.read_csv(rebuilt["combined_output"], compression="gzip")
    assert output.loc[0, "last_price"] != 999.0
    assert json.loads(monthly_metadata.read_text())["output"] != str(alternate)


def test_resume_metadata_cannot_tamper_frozen_archive_facts(tmp_path: Path) -> None:
    day = date(2021, 1, 1)
    payload = _archive(day)
    fetcher = _FakeFetcher({day: payload})
    contract = _contract({day: payload})
    cfg = builder.BuildConfig(
        start=day.isoformat(),
        end="2021-01-02",
        output_dir=str(tmp_path),
        workers=1,
    )
    builder.build(cfg, fetcher=fetcher, source_contract=contract)
    monthly_metadata = next((tmp_path / "monthly").glob("*.json"))
    metadata = json.loads(monthly_metadata.read_text())
    metadata["archives"][0]["first_agg_trade_id"] = 999_999
    monthly_metadata.write_text(json.dumps(metadata))

    builder.build(cfg, fetcher=fetcher, source_contract=contract)
    repaired = json.loads(monthly_metadata.read_text())
    assert repaired["archives"][0]["first_agg_trade_id"] == 1


def test_resume_rebuilds_rowwise_source_flag_tampering(tmp_path: Path) -> None:
    day = date(2021, 1, 1)
    payload = _archive(day)
    fetcher = _FakeFetcher({day: payload})
    contract = _contract({day: payload})
    cfg = builder.BuildConfig(
        start=day.isoformat(),
        end="2021-01-02",
        output_dir=str(tmp_path),
        workers=1,
    )
    builder.build(cfg, fetcher=fetcher, source_contract=contract)
    monthly_output = next((tmp_path / "monthly").glob("*.csv.gz"))
    monthly_metadata = next((tmp_path / "monthly").glob("*.json"))
    frame = pd.read_csv(monthly_output, compression="gzip")
    assert frame.loc[1, "source_observed"] == 0
    frame.loc[1, "source_complete"] = True
    base._write_gzip_csv(frame, monthly_output)
    metadata = json.loads(monthly_metadata.read_text())
    metadata["output_sha256"] = hashlib.sha256(monthly_output.read_bytes()).hexdigest()
    monthly_metadata.write_text(json.dumps(metadata))

    result = builder.build(cfg, fetcher=fetcher, source_contract=contract)
    repaired = pd.read_csv(result["combined_output"], compression="gzip")
    assert repaired.loc[1, "source_observed"] == 0
    assert repaired.loc[1, "source_complete"] == 0


def test_cross_day_id_regression_fails_closed(tmp_path: Path) -> None:
    first = date(2021, 1, 1)
    second = date(2021, 1, 2)
    payloads = {first: _archive(first), second: _archive(second)}
    cfg = builder.BuildConfig(
        start=first.isoformat(),
        end="2021-01-03",
        output_dir=str(tmp_path),
        workers=1,
    )
    with pytest.raises(ValueError, match="cross-day aggregate-trade ID discontinuity"):
        builder.build(
            cfg,
            fetcher=_FakeFetcher(payloads),
            source_contract=_contract(payloads),
        )


def test_wrong_day_archive_timestamp_fails_before_reindex(tmp_path: Path) -> None:
    day = date(2021, 1, 1)
    payload = _archive(day, timestamp_day=date(2021, 1, 2))
    cfg = builder.BuildConfig(
        start=day.isoformat(),
        end="2021-01-02",
        output_dir=str(tmp_path),
        workers=1,
    )
    with pytest.raises(ValueError, match="timestamps outside"):
        builder.build(
            cfg,
            fetcher=_FakeFetcher({day: payload}),
            source_contract=_contract({day: payload}),
        )


def test_following_twenty_four_bars_are_quarantined_across_day_boundary(
    tmp_path: Path,
) -> None:
    gap_day = date(2021, 2, 9)
    next_day = date(2021, 2, 10)
    payloads = {
        gap_day: _archive(gap_day, internal_gap=True),
        next_day: _full_day_archive(next_day, start_agg_id=5, start_trade_id=14),
    }
    cfg = builder.BuildConfig(
        start=gap_day.isoformat(),
        end="2021-02-11",
        output_dir=str(tmp_path),
        workers=1,
    )
    result = builder.build(
        cfg,
        fetcher=_FakeFetcher(payloads),
        source_contract=_contract(
            payloads, gap_days=frozenset({gap_day.isoformat()})
        ),
    )
    output = pd.read_csv(result["combined_output"], compression="gzip")
    following = output.loc[output["date"].str.startswith(next_day.isoformat())]
    assert following.iloc[:24]["post_gap_quarantine"].all()
    assert not following.iloc[:24]["source_complete"].any()
    assert following.iloc[24]["post_gap_quarantine"] == 0


def test_repository_source_contract_is_hash_bound_and_complete() -> None:
    contract = builder.load_source_contract()
    assert len(contract.archive_sha256_by_date) == 1_461
    assert contract.source_gap_days == frozenset(
        {
            "2020-01-15",
            "2020-04-15",
            "2021-02-09",
            "2021-02-24",
            "2021-05-19",
            "2022-09-06",
        }
    )
    assert contract.underlying_overlap_counts == {"2020-01-15": 1}
    assert len(contract.verified_zero_volume_bins) == 26
