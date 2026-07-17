"""Build FQPR-3 support clocks without opening any strategy outcome.

The builder is intentionally limited to the frozen preregistration and the
official fiat-quote source panel.  It never reads execution OHLC, funding,
returns, CAGR, or drawdown.  Q is selected only from 2021-2022 support; the
support-seen but outcome-sealed 2023 window may only pass or reject that Q.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_fiat_quote_participation_rotation as prereg


PREREGISTRATION = prereg.DEFAULT_OUTPUT
PREREGISTRATION_SHA256 = (
    "c9e7a3fde7ca421b2efb341da80a1f173c7e15d6b6d920978af5966e63ead21c"
)
PREREGISTRATION_COMMIT = "b7f17846497fc9cd7e367a4c84d3ff3296996fb9"
DEFAULT_OUTPUT = "results/fiat_quote_participation_rotation_support_2026-07-17.json"
DEFAULT_CLOCKS = "results/fiat_quote_participation_rotation_clocks_2026-07-17.csv"
REQUIRED_COLUMNS = (
    "date",
    "symbol",
    "open_time_ms",
    "close_time_ms",
    "base_volume_btc",
    "trade_count",
    "taker_buy_base_btc",
    "taker_sell_base_btc",
    "taker_buy_fraction",
    "source_complete",
)
TRAIN = (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01"))
TRAIN_2021 = (pd.Timestamp("2021-06-30"), pd.Timestamp("2022-01-01"))
TRAIN_2022 = (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01"))
YEAR_2023 = (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01"))
H1_2023 = (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-07-01"))
H2_2023 = (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-01-01"))
BOOK_LABELS = {"BTCEUR": "EUR", "BTCTRY": "TRY", "BTCBRL": "BRL"}


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_preregistration(path: str | Path = PREREGISTRATION) -> dict[str, Any]:
    if _sha256(path) != PREREGISTRATION_SHA256:
        raise ValueError("FQPR-3 preregistration file changed")
    payload = _load_json(path)
    prereg.validate_manifest(payload, verify_sources=False)
    if payload["policy"]["policy_id"] != "FQPR-3":
        raise ValueError("unexpected FQPR-3 policy identity")
    return payload


def load_source(preregistration: dict[str, Any]) -> pd.DataFrame:
    """Load only the frozen flow panel and its source manifest."""
    source = preregistration["source_contract"]
    if _sha256(source["flow"]) != source["flow_sha256"]:
        raise ValueError("FQPR-3 flow panel changed")
    if _sha256(source["flow_manifest"]) != source["flow_manifest_sha256"]:
        raise ValueError("FQPR-3 flow manifest changed")
    manifest = _load_json(source["flow_manifest"])
    if manifest.get("combined_sha256") != source["flow_sha256"]:
        raise ValueError("FQPR-3 source manifest does not bind the frozen panel")
    if manifest.get("protocol", {}).get("outcomes_opened") is not False:
        raise ValueError("FQPR-3 source manifest opened an outcome")

    frame = pd.read_csv(source["flow"], compression="gzip", parse_dates=["date"])
    if tuple(frame.columns) != REQUIRED_COLUMNS:
        raise ValueError(f"unexpected FQPR-3 source columns: {frame.columns.tolist()}")
    if frame[["date", "symbol"]].duplicated().any():
        raise ValueError("FQPR-3 source has duplicate date-symbol rows")
    if not frame["source_complete"].astype(bool).all():
        raise ValueError("FQPR-3 source has incomplete rows")

    policy = preregistration["policy"]
    symbols = sorted([policy["reference_symbol"], *policy["fiat_quote_symbols"]])
    dates = pd.date_range(
        source["available_start"],
        source["available_end_exclusive"],
        freq="1D",
        inclusive="left",
    )
    observed = pd.MultiIndex.from_frame(
        frame.sort_values(["date", "symbol"])[["date", "symbol"]]
    )
    expected = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    if not observed.equals(expected):
        raise ValueError("FQPR-3 source does not match the exact date-symbol grid")
    numeric = frame.loc[:, REQUIRED_COLUMNS[2:-1]]
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("FQPR-3 source contains non-finite observables")
    if (frame[["base_volume_btc", "trade_count"]] <= 0.0).any().any():
        raise ValueError("FQPR-3 source contains non-positive participation")
    if (
        frame[["taker_buy_base_btc", "taker_sell_base_btc"]] <= 0.0
    ).any().any():
        raise ValueError("FQPR-3 taker odds require positive two-sided volume")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def prior_midrank(series: pd.Series, window: int) -> pd.Series:
    """Rank each current value against exactly its strictly prior window."""
    values = series.to_numpy(float)
    output = np.full(len(values), np.nan, dtype=float)
    for index in range(window, len(values)):
        current = values[index]
        prior = values[index - window : index]
        if np.isfinite(current) and np.isfinite(prior).all():
            output[index] = (
                float(np.count_nonzero(prior < current))
                + 0.5 * float(np.count_nonzero(prior == current))
            ) / float(window)
    return pd.Series(output, index=series.index, dtype=float)


def build_features(
    source: pd.DataFrame, preregistration: dict[str, Any]
) -> pd.DataFrame:
    policy = preregistration["policy"]
    books = list(policy["fiat_quote_symbols"])
    reference = policy["reference_symbol"]
    window = int(policy["baseline_days"])
    if int(policy["baseline_min_periods"]) != window:
        raise ValueError("FQPR-3 forbids an expanding rank fallback")

    wide = {
        column: source.pivot(index="date", columns="symbol", values=column).sort_index()
        for column in (
            "base_volume_btc",
            "trade_count",
            "taker_buy_base_btc",
            "taker_sell_base_btc",
        )
    }
    features = pd.DataFrame(index=wide["base_volume_btc"].index)
    reference_odds = np.log(
        wide["taker_buy_base_btc"][reference]
        / wide["taker_sell_base_btc"][reference]
    )
    features["reference_buy_odds"] = reference_odds

    raw_volume_rank: dict[str, pd.Series] = {}
    raw_ticket_rank: dict[str, pd.Series] = {}
    for symbol in [reference, *books]:
        raw_volume_rank[symbol] = prior_midrank(wide["base_volume_btc"][symbol], window)
        raw_ticket_rank[symbol] = prior_midrank(wide["trade_count"][symbol], window)

    for symbol in books:
        label = BOOK_LABELS[symbol].lower()
        volume_share = np.log(
            wide["base_volume_btc"][symbol] / wide["base_volume_btc"][reference]
        )
        ticket_share = np.log(
            wide["trade_count"][symbol] / wide["trade_count"][reference]
        )
        volume_share_rank = prior_midrank(volume_share, window)
        ticket_share_rank = prior_midrank(ticket_share, window)
        buy_odds = np.log(
            wide["taker_buy_base_btc"][symbol]
            / wide["taker_sell_base_btc"][symbol]
        )
        features[f"volume_share_rank_{label}"] = volume_share_rank
        features[f"ticket_share_rank_{label}"] = ticket_share_rank
        features[f"participation_score_{label}"] = (
            volume_share_rank + ticket_share_rank
        ) / 2.0
        features[f"relative_taker_pressure_{label}"] = buy_odds - reference_odds
        features[f"absolute_participation_{label}"] = (
            raw_volume_rank[symbol] + raw_ticket_rank[symbol]
        ) / 2.0

    pressure_columns = [
        f"relative_taker_pressure_{BOOK_LABELS[symbol].lower()}" for symbol in books
    ]
    features["median_relative_taker_pressure"] = features[pressure_columns].median(
        axis=1, skipna=False
    )
    features["reference_raw_participation"] = (
        raw_volume_rank[reference] + raw_ticket_rank[reference]
    ) / 2.0
    features.index.name = "date"
    return features


def build_flags(features: pd.DataFrame, q: float) -> dict[str, pd.Series]:
    participation = features[
        ["participation_score_eur", "participation_score_try", "participation_score_brl"]
    ]
    volume_rank = features[
        ["volume_share_rank_eur", "volume_share_rank_try", "volume_share_rank_brl"]
    ]
    pressure = features[
        [
            "relative_taker_pressure_eur",
            "relative_taker_pressure_try",
            "relative_taker_pressure_brl",
        ]
    ]
    absolute = features[
        [
            "absolute_participation_eur",
            "absolute_participation_try",
            "absolute_participation_brl",
        ]
    ]
    median_pressure = features["median_relative_taker_pressure"]
    participation_ready = participation.notna().all(axis=1)
    volume_ready = volume_rank.notna().all(axis=1)
    pressure_ready = pressure.notna().all(axis=1)
    absolute_ready = absolute.notna().all(axis=1)
    reference_ready = features[
        ["reference_raw_participation", "reference_buy_odds"]
    ].notna().all(axis=1)
    primary = (
        participation_ready
        & pressure_ready
        & participation.ge(q).sum(axis=1).ge(2)
        & median_pressure.gt(0.0)
    )
    return {
        "primary": primary,
        "no_ticket": (
            volume_ready
            & pressure_ready
            & volume_rank.ge(q).sum(axis=1).ge(2)
            & median_pressure.gt(0.0)
        ),
        "no_taker": participation_ready & participation.ge(q).sum(axis=1).ge(2),
        "volume_only": volume_ready & volume_rank.ge(q).sum(axis=1).ge(2),
        "flow_only": pressure_ready & median_pressure.gt(0.0),
        "single_book_eur": (
            participation_ready
            & pressure_ready
            & participation["participation_score_eur"].ge(q)
            & pressure["relative_taker_pressure_eur"].gt(0.0)
        ),
        "single_book_try": (
            participation_ready
            & pressure_ready
            & participation["participation_score_try"].ge(q)
            & pressure["relative_taker_pressure_try"].gt(0.0)
        ),
        "single_book_brl": (
            participation_ready
            & pressure_ready
            & participation["participation_score_brl"].ge(q)
            & pressure["relative_taker_pressure_brl"].gt(0.0)
        ),
        "usdt_only": (
            reference_ready
            & features["reference_raw_participation"].ge(q)
            & features["reference_buy_odds"].gt(0.0)
        ),
        "reference_suppression": (
            reference_ready
            & pressure_ready
            & features["reference_raw_participation"].le(1.0 - q)
            & median_pressure.gt(0.0)
        ),
        "absolute_book_participation": (
            absolute_ready
            & pressure_ready
            & absolute.ge(q).sum(axis=1).ge(2)
            & median_pressure.gt(0.0)
        ),
    }


def false_to_true_days(flag: pd.Series) -> pd.DatetimeIndex:
    clean = flag.fillna(False).astype(bool)
    onset = clean & ~clean.shift(1, fill_value=False)
    return pd.DatetimeIndex(clean.index[onset])


def reserve_signal_days(
    signal_days: pd.DatetimeIndex,
    *,
    clock_name: str,
    q: float,
    hold_bars: int,
    execution_delay_bars: int,
    signal_delay_days: int = 0,
) -> pd.DataFrame:
    hold = pd.Timedelta(minutes=5 * hold_bars)
    execution_delay = pd.Timedelta(minutes=5 * execution_delay_bars)
    rows: list[dict[str, Any]] = []
    next_entry = pd.Timestamp.min
    for source_day in signal_days.sort_values():
        signal_day = source_day + pd.Timedelta(days=signal_delay_days)
        decision_time = signal_day + pd.Timedelta(days=1)
        entry_time = decision_time + execution_delay
        exit_time = entry_time + hold
        if entry_time < next_entry:
            continue
        rows.append(
            {
                "clock_name": clock_name,
                "q": q,
                "source_signal_day": source_day,
                "signal_day": signal_day,
                "decision_time": decision_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "side": "LONG",
            }
        )
        next_entry = exit_time
    return pd.DataFrame(
        rows,
        columns=(
            "clock_name",
            "q",
            "source_signal_day",
            "signal_day",
            "decision_time",
            "entry_time",
            "exit_time",
            "side",
        ),
    )


def build_clocks(
    features: pd.DataFrame, q: float, preregistration: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    policy = preregistration["policy"]
    flags = build_flags(features, q)
    clocks = {
        name: reserve_signal_days(
            false_to_true_days(flag),
            clock_name=name,
            q=q,
            hold_bars=int(policy["hold_bars"]),
            execution_delay_bars=int(policy["execution_delay_bars"]),
        )
        for name, flag in flags.items()
    }
    primary_signal_days = false_to_true_days(flags["primary"])
    clocks["one_day_signal_delay"] = reserve_signal_days(
        primary_signal_days,
        clock_name="one_day_signal_delay",
        q=q,
        hold_bars=int(policy["hold_bars"]),
        execution_delay_bars=int(policy["execution_delay_bars"]),
        signal_delay_days=1,
    )
    direction = clocks["primary"].copy()
    direction["clock_name"] = "direction_flip"
    direction["side"] = "SHORT"
    clocks["direction_flip"] = direction
    random_side = clocks["primary"].copy()
    random_side["clock_name"] = "random_side"
    generator = np.random.default_rng(20_260_717)
    if len(random_side):
        random_side["side"] = np.where(
            generator.integers(0, 2, size=len(random_side)).astype(bool),
            "LONG",
            "SHORT",
        )
    clocks["random_side"] = random_side
    return clocks


def split_clock(
    clock: pd.DataFrame, window: tuple[pd.Timestamp, pd.Timestamp]
) -> pd.DataFrame:
    if clock.empty:
        return clock.copy()
    start, end = window
    mask = (
        clock["signal_day"].ge(start)
        & clock["entry_time"].ge(start)
        & clock["exit_time"].le(end)
    )
    return clock.loc[mask].copy()


def _jaccard(left: pd.Series, right: pd.Series) -> float:
    left_set = set(pd.to_datetime(left))
    right_set = set(pd.to_datetime(right))
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 0.0


def primary_summary(
    primary: pd.DataFrame,
    features: pd.DataFrame,
    q: float,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> dict[str, Any]:
    clock = split_clock(primary, window)
    if clock.empty:
        return {
            "entries": 0,
            "maximum_single_month_share": None,
            "book_involvement_share": {label: None for label in ("EUR", "TRY", "BRL")},
            "maximum_participating_book_set_share": None,
            "participating_book_set_counts": {},
        }
    months = clock["entry_time"].dt.to_period("M").astype(str)
    month_share = float(months.value_counts(normalize=True).max())
    involvement = {label: 0 for label in ("EUR", "TRY", "BRL")}
    sets: list[str] = []
    for signal_day in clock["source_signal_day"]:
        books = [
            label.upper()
            for label in ("eur", "try", "brl")
            if features.loc[signal_day, f"participation_score_{label}"] >= q
        ]
        for label in books:
            involvement[label] += 1
        sets.append("+".join(books))
    counts = pd.Series(sets, dtype="string").value_counts().sort_index()
    return {
        "entries": int(len(clock)),
        "maximum_single_month_share": month_share,
        "book_involvement_share": {
            label: float(involvement[label] / len(clock)) for label in involvement
        },
        "maximum_participating_book_set_share": float(counts.max() / len(clock)),
        "participating_book_set_counts": {
            str(label): int(count) for label, count in counts.items()
        },
    }


def control_jaccards(
    clocks: dict[str, pd.DataFrame],
    window: tuple[pd.Timestamp, pd.Timestamp],
    control_names: list[str],
) -> dict[str, float]:
    primary = split_clock(clocks["primary"], window)["entry_time"]
    return {
        name: _jaccard(primary, split_clock(clocks[name], window)["entry_time"])
        for name in control_names
    }


def train_evaluation(
    q: float,
    clocks: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    support = preregistration["support_calibration"]
    control_names = list(support["maximum_signal_jaccard"])
    summary = primary_summary(clocks["primary"], features, q, TRAIN)
    year_2021 = len(split_clock(clocks["primary"], TRAIN_2021))
    year_2022 = len(split_clock(clocks["primary"], TRAIN_2022))
    jaccards = control_jaccards(clocks, TRAIN, control_names)
    gates: dict[str, bool] = {
        "minimum_nonoverlap_train": summary["entries"]
        >= support["minimum_nonoverlap_train"],
        "minimum_2021_after_warmup": year_2021
        >= support["minimum_2021_after_warmup"],
        "minimum_2022": year_2022 >= support["minimum_2022"],
        "maximum_single_month_share": summary["maximum_single_month_share"]
        is not None
        and summary["maximum_single_month_share"]
        <= support["maximum_single_month_share"],
        "minimum_each_book_involvement_share": all(
            value is not None
            and value >= support["minimum_each_book_involvement_share"]
            for value in summary["book_involvement_share"].values()
        ),
        "maximum_single_pair_share": summary["maximum_participating_book_set_share"]
        is not None
        and summary["maximum_participating_book_set_share"]
        <= support["maximum_single_pair_share"],
        "maximum_signal_jaccard": all(
            jaccards[name] <= support["maximum_signal_jaccard"][name]
            for name in control_names
        ),
    }
    return {
        "q": q,
        "primary": summary,
        "subperiod_entries": {"2021_after_warmup": year_2021, "2022": year_2022},
        "control_jaccards": jaccards,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def validation_evaluation(
    q: float,
    clocks: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    support = preregistration["support_calibration"]
    control_names = list(support["maximum_signal_jaccard"])
    summary = primary_summary(clocks["primary"], features, q, YEAR_2023)
    h1 = len(split_clock(clocks["primary"], H1_2023))
    h2 = len(split_clock(clocks["primary"], H2_2023))
    jaccards = control_jaccards(clocks, YEAR_2023, control_names)
    gates: dict[str, bool] = {
        "minimum_2023": summary["entries"] >= support["minimum_2023"],
        "minimum_each_2023_half": min(h1, h2)
        >= support["minimum_each_2023_half"],
        "maximum_single_month_share": summary["maximum_single_month_share"]
        is not None
        and summary["maximum_single_month_share"]
        <= support["maximum_single_month_share"],
        "minimum_each_book_involvement_share": all(
            value is not None
            and value >= support["minimum_each_book_involvement_share"]
            for value in summary["book_involvement_share"].values()
        ),
        "maximum_single_pair_share": summary["maximum_participating_book_set_share"]
        is not None
        and summary["maximum_participating_book_set_share"]
        <= support["maximum_single_pair_share"],
        "maximum_signal_jaccard": all(
            jaccards[name] <= support["maximum_signal_jaccard"][name]
            for name in control_names
        ),
    }
    return {
        "q": q,
        "primary": summary,
        "half_entries": {"2023_h1": h1, "2023_h2": h2},
        "control_jaccards": jaccards,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def _attach_features(
    clock: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    output = clock.copy()
    feature_columns = [
        "participation_score_eur",
        "participation_score_try",
        "participation_score_brl",
        "relative_taker_pressure_eur",
        "relative_taker_pressure_try",
        "relative_taker_pressure_brl",
        "median_relative_taker_pressure",
    ]
    if output.empty:
        for column in feature_columns:
            output[column] = pd.Series(dtype="float64")
        output["participating_books"] = pd.Series(dtype="string")
        output["participation_breadth"] = pd.Series(dtype="int64")
        return output
    source_days = pd.DatetimeIndex(output["source_signal_day"])
    selected = features.loc[source_days, feature_columns].reset_index(drop=True)
    for column in feature_columns:
        output[column] = selected[column].to_numpy()
    score_columns = [
        "participation_score_eur",
        "participation_score_try",
        "participation_score_brl",
    ]
    output["participating_books"] = [
        "+".join(
            label.upper()
            for label in ("eur", "try", "brl")
            if row[f"participation_score_{label}"] >= row["q"]
        )
        for _, row in output.iterrows()
    ]
    output["participation_breadth"] = output[score_columns].ge(
        output["q"], axis=0
    ).sum(axis=1)
    return output


def _write_clocks(
    clocks: dict[str, pd.DataFrame], features: pd.DataFrame, path: str | Path
) -> str:
    frames = [_attach_features(clock, features) for clock in clocks.values()]
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["clock_name", "entry_time", "source_signal_day"], kind="mergesort"
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    return _sha256(path)


def build(
    *,
    preregistration_path: str = PREREGISTRATION,
    output_path: str = DEFAULT_OUTPUT,
    clocks_path: str = DEFAULT_CLOCKS,
) -> dict[str, Any]:
    registration = load_preregistration(preregistration_path)
    source = load_source(registration)
    features = build_features(source, registration)
    quantiles = [float(value) for value in registration["policy"]["participation_quantiles"]]
    train_grid: list[dict[str, Any]] = []
    clocks_by_q: dict[float, dict[str, pd.DataFrame]] = {}
    for q in sorted(quantiles, reverse=True):
        clocks = build_clocks(features, q, registration)
        clocks_by_q[q] = clocks
        train_grid.append(train_evaluation(q, clocks, features, registration))
    selected = next((item for item in train_grid if item["passed"]), None)
    selected_q = float(selected["q"]) if selected is not None else None
    if selected_q is None:
        selected_clocks = {
            "primary": reserve_signal_days(
                pd.DatetimeIndex([]),
                clock_name="primary",
                q=float("nan"),
                hold_bars=int(registration["policy"]["hold_bars"]),
                execution_delay_bars=int(registration["policy"]["execution_delay_bars"]),
            )
        }
        validation = None
    else:
        selected_clocks = clocks_by_q[selected_q]
        validation = validation_evaluation(
            selected_q, selected_clocks, features, registration
        )
    source_contract = registration["source_contract"]
    full_source_window = (
        pd.Timestamp(source_contract["available_start"]),
        pd.Timestamp(source_contract["available_end_exclusive"]),
    )
    contained_clocks = {
        name: split_clock(clock, full_source_window)
        for name, clock in selected_clocks.items()
    }
    clocks_hash = _write_clocks(contained_clocks, features, clocks_path)
    result = {
        "protocol_version": "fiat_quote_participation_rotation_support_v1",
        "policy_id": "FQPR-3",
        "as_of_date": "2026-07-17",
        "outcomes_opened": False,
        "outcome_sources_opened": [],
        "preregistration": {
            "path": preregistration_path,
            "sha256": _sha256(preregistration_path),
            "manifest_hash": registration["manifest_hash"],
            "commit": PREREGISTRATION_COMMIT,
        },
        "source": {
            "flow": source_contract["flow"],
            "flow_sha256": source_contract["flow_sha256"],
            "flow_manifest": source_contract["flow_manifest"],
            "flow_manifest_sha256": source_contract["flow_manifest_sha256"],
            "rows": int(len(source)),
            "first_date": source["date"].min().isoformat(),
            "last_date": source["date"].max().isoformat(),
        },
        "selection_rule": registration["support_calibration"]["selection_rule"],
        "train_grid_descending_q": train_grid,
        "selected_q": selected_q,
        "selected_train": selected,
        "selected_2023": validation,
        "support_passed": bool(
            selected is not None
            and validation is not None
            and selected["passed"]
            and validation["passed"]
        ),
        "advance_to_stage1_outcomes": bool(
            selected is not None
            and validation is not None
            and selected["passed"]
            and validation["passed"]
        ),
        "clocks": {
            "path": clocks_path,
            "sha256": clocks_hash,
            "rows": int(sum(len(clock) for clock in contained_clocks.values())),
            "clock_counts": {
                name: int(len(clock)) for name, clock in sorted(contained_clocks.items())
            },
        },
        "sealed": ["all entry-to-exit returns", "2024", "2025", "2026_ytd"],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default=PREREGISTRATION)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--clocks", default=DEFAULT_CLOCKS)
    args = parser.parse_args()
    result = build(
        preregistration_path=args.preregistration,
        output_path=args.output,
        clocks_path=args.clocks,
    )
    print(
        json.dumps(
            {
                "selected_q": result["selected_q"],
                "support_passed": result["support_passed"],
                "advance_to_stage1_outcomes": result["advance_to_stage1_outcomes"],
                "clock_rows": result["clocks"]["rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
