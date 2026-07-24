"""Build TPCT-120 source support without opening comparators or outcomes."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from fractions import Fraction
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence, TextIO

from training import preregister_tri_party_composition_topology as prereg


UTC = timezone.utc
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = "training/build_tri_party_composition_topology_support.py"
TEST_PATH = "tests/test_build_tri_party_composition_topology_support.py"
PREREGISTRATION = (
    "results/tri_party_composition_topology_preregistration_2026-07-24.json"
)
PREREGISTRATION_SHA256 = (
    "6cce962531826cdd67c79c2e80759693ba4e42d9395fd04bc57afe1e91f446e5"
)
PREREGISTRATION_MANIFEST_HASH = (
    "c62df08de8ba667a52380fb11fb050e9aff0cc67c2f31aabf13f6262bf795ce7"
)
PREREGISTRATION_CONTRACT_HASH = (
    "666e9b94466e1445d609e604628e2a26f0e644628a83c2d9968c30d07e87d44c"
)
PROTOCOL_VERSION = "tri_party_composition_topology_source_support_v1"
DEFAULT_CLOCK = Path(
    "results/tri_party_composition_topology_source_clock_2026-07-24.csv.gz"
)
DEFAULT_REPORT = Path(
    "results/tri_party_composition_topology_source_support_2026-07-24.json"
)
EXPECTED_SOURCE_ROWS = 77_369
SOURCE_WINDOW_START = date(2019, 1, 2)
SOURCE_WINDOW_END = date(2023, 12, 29)
FEED_FLOOR = prereg.TRAIN_START
TRAIN_END = prereg.SELECTION_START
SELECTION_END = prereg.EVAL_START
PAIR_TOKENS = prereg.TOKEN_COLUMNS[:6]
LEADER_TOKENS = ("high_leader", "low_leader")
CLOCK_COLUMNS = (
    "candidate",
    "split",
    "observation_date",
    "available_at_utc",
    "signal_available",
    "entry",
    "exit",
    *prereg.TOKEN_COLUMNS,
    "token_signature_sha256",
)
FORBIDDEN_CLOCK_COLUMNS = {
    "value",
    "raw_value",
    "rank",
    "action",
    "side",
    "market",
    "funding",
    "return",
    "pnl",
    "cagr",
    "mdd",
    "reward",
    "label",
    "outcome",
}
EXPECTED_SUBSET_METADATA = {
    "OO": ("Tenor", "Overnight/Open"),
    "B27": ("Tenor", "Term, 2 - 7 Days"),
    "B830": ("Tenor", "Term, 8 - 30 Days"),
    "G30": ("Tenor", "Term, >30 Days"),
    "T": ("Collateral", "U.S. Treasury Securities"),
    "AG": ("Collateral", "Federal Agency and GSE Securities"),
    "CORD": ("Collateral", "Corporate Debt"),
    "O": ("Collateral", "Other Collateral"),
}


@dataclass(frozen=True)
class SelectedRow:
    mnemonic: str
    observation_date: date
    available_at: datetime
    value: Fraction | None
    valid_value: bool


@dataclass(frozen=True)
class SourceVector:
    observation_date: date
    available_at: datetime
    valid: bool
    invalid_reasons: tuple[str, ...]
    primitives: OrderedDict[str, Fraction] | None


@dataclass(frozen=True)
class RankDecision:
    observation_date: date
    available_at: datetime
    ranks: OrderedDict[str, Fraction]


@dataclass(frozen=True)
class Opportunity:
    observation_date: date
    available_at: datetime
    signal_available: datetime
    entry: datetime
    exit: datetime
    tokens: OrderedDict[str, str]
    reserved: bool
    split: str | None

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(self.tokens[name] for name in prereg.TOKEN_COLUMNS)


@dataclass(frozen=True)
class SourceAudit:
    physical_rows_read: int
    eligible_selected_rows_seen: int
    eligible_selected_values_converted: int
    eligible_source_dates: int
    complete_vectors: int
    invalid_source_dates: int
    sealed_values_converted: int = 0
    sealed_candidate_statistics: int = 0


@dataclass(frozen=True)
class BuildAudit:
    availability_batches: int
    rank_complete_decisions: int
    predecessor_only_decisions: int
    token_states: int
    reservation_suppressed: int
    split_rejected_after_reservation: int


def repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise RuntimeError("TPCT support path must be repository-relative")
    if not candidate.parts or any(part == ".." for part in candidate.parts):
        raise RuntimeError("TPCT support path escaped repository")
    return REPOSITORY_ROOT / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("TPCT timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("TPCT source timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise RuntimeError("TPCT source timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def expected_availability(day: date) -> datetime:
    delayed = datetime.combine(day + timedelta(days=8), time(), UTC)
    return max(delayed, FEED_FLOOR)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def assert_protocol_committed() -> str:
    required = (SCRIPT_PATH, TEST_PATH, PREREGISTRATION)
    tracked = _git("ls-files", "--error-unmatch", "--", *required)
    if tracked.returncode != 0:
        raise RuntimeError("TPCT source-support protocol is not committed")
    if _git("diff", "--quiet", "HEAD", "--", SCRIPT_PATH, TEST_PATH).returncode:
        raise RuntimeError("TPCT source-support protocol differs from HEAD")
    if _git("diff", "--cached", "--quiet").returncode:
        raise RuntimeError("TPCT source-support index is not clean")
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("TPCT source-support worktree is not HEAD-clean")
    head = _git("rev-parse", "HEAD")
    if head.returncode != 0:
        raise RuntimeError("TPCT source-support HEAD is unavailable")
    return head.stdout.strip()


def load_registration() -> dict[str, Any]:
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise RuntimeError("TPCT preregistration file hash mismatch")
    payload = json.loads(
        repository_path(PREREGISTRATION).read_text(encoding="utf-8")
    )
    if payload.get("candidate") != prereg.POLICY_ID:
        raise RuntimeError("TPCT preregistration candidate changed")
    if payload.get("manifest_hash") != PREREGISTRATION_MANIFEST_HASH:
        raise RuntimeError("TPCT preregistration manifest hash changed")
    if payload.get("contract_hash") != PREREGISTRATION_CONTRACT_HASH:
        raise RuntimeError("TPCT preregistration contract hash changed")
    prereg.validate_manifest(payload, revalidate_files=False)
    decision = payload.get("decision")
    if not isinstance(decision, dict) or not decision.get(
        "source_support_authorized"
    ):
        raise RuntimeError("TPCT source support is not authorized")
    if any(
        decision.get(key)
        for key in (
            "market_outcomes_authorized",
            "model_training_authorized",
            "sealed_eval_authorized",
        )
    ):
        raise RuntimeError("TPCT preregistration opened a later stage")
    return payload


def selected_semantics() -> dict[str, tuple[str, str, str, str]]:
    rows = prereg._read_metadata()
    selected = {
        str(row.get("mnemonic")): row
        for row in rows
        if row.get("mnemonic") in prereg.SOURCE_ALLOWLIST
    }
    if set(selected) != set(prereg.SOURCE_ALLOWLIST):
        raise RuntimeError("TPCT selected metadata set changed")
    ordered = [selected[mnemonic] for mnemonic in prereg.SOURCE_ALLOWLIST]
    if prereg.canonical_hash(ordered) != prereg.SELECTED_METADATA_HASH:
        raise RuntimeError("TPCT selected metadata hash changed")
    for mnemonic, row in selected.items():
        subset = str(row["subset"])
        subsetting, display_name = EXPECTED_SUBSET_METADATA[subset]
        description = row.get("metadata", {}).get("description", {})
        if (
            description.get("subsetting") != subsetting
            or f": {display_name} (Preliminary)"
            not in str(description.get("name"))
        ):
            raise RuntimeError(
                f"TPCT subset non-overlap metadata changed: {mnemonic}"
            )
    return {
        mnemonic: (
            str(row["segment"]),
            str(row["measure"]),
            str(row["subset"]),
            str(row["series_name"]),
        )
        for mnemonic, row in selected.items()
    }


def _selected_row_from_fields(
    fields: Sequence[str],
    *,
    day: date,
    available_at: datetime,
    semantics: Mapping[str, tuple[str, str, str, str]],
) -> SelectedRow:
    mnemonic = fields[0]
    expected = semantics[mnemonic]
    observed = (fields[5], fields[6], fields[7], fields[8])
    if observed != expected:
        raise RuntimeError(f"TPCT source semantics changed: {mnemonic}")
    valid_value = fields[4] == "0" and bool(fields[3])
    value: Fraction | None = None
    if valid_value:
        try:
            value = prereg.parse_exact_decimal(fields[3])
        except ValueError:
            valid_value = False
    return SelectedRow(
        mnemonic=mnemonic,
        observation_date=day,
        available_at=available_at,
        value=value,
        valid_value=valid_value,
    )


def parse_source_stream(
    handle: TextIO,
    *,
    expected_rows: int | None,
    semantics: Mapping[str, tuple[str, str, str, str]],
) -> tuple[list[SourceVector], SourceAudit]:
    reader = csv.reader(handle)
    try:
        header = tuple(next(reader))
    except StopIteration as exc:
        raise RuntimeError("TPCT source is empty") from exc
    if header != prereg.SOURCE_COLUMNS:
        raise RuntimeError("TPCT source columns changed")

    by_date: dict[date, dict[str, SelectedRow]] = defaultdict(dict)
    physical_rows = 0
    selected_rows_seen = 0
    values_converted = 0
    for fields in reader:
        physical_rows += 1
        if len(fields) != len(prereg.SOURCE_COLUMNS):
            raise RuntimeError("TPCT source row width changed")
        try:
            day = date.fromisoformat(fields[1])
        except ValueError as exc:
            raise RuntimeError("TPCT observation date is invalid") from exc
        if not SOURCE_WINDOW_START <= day <= SOURCE_WINDOW_END:
            raise RuntimeError("TPCT source date escaped frozen window")
        available_at = parse_timestamp(fields[2])
        if available_at != expected_availability(day):
            raise RuntimeError("TPCT source availability changed")

        # This branch is deliberately before allowlist comparison, semantic
        # inspection, value parsing, or candidate-specific counting.
        if not prereg.source_value_may_decode(available_at):
            continue

        mnemonic = fields[0]
        if mnemonic not in prereg.SOURCE_ALLOWLIST:
            continue
        selected_rows_seen += 1
        if mnemonic in by_date[day]:
            raise RuntimeError("TPCT selected source row is duplicated")
        row = _selected_row_from_fields(
            fields,
            day=day,
            available_at=available_at,
            semantics=semantics,
        )
        if row.value is not None:
            values_converted += 1
        by_date[day][mnemonic] = row

    if expected_rows is not None and physical_rows != expected_rows:
        raise RuntimeError("TPCT physical source row count changed")

    vectors: list[SourceVector] = []
    complete_vectors = 0
    invalid_dates = 0
    required = set(prereg.SOURCE_ALLOWLIST)
    for day in sorted(by_date):
        rows = by_date[day]
        reasons: list[str] = []
        if set(rows) != required:
            reasons.append("incomplete_mnemonic_set")
        if any(not row.valid_value for row in rows.values()):
            reasons.append("invalid_or_disclosure_edited_value")
        values: OrderedDict[str, Fraction] | None = None
        if not reasons:
            values = OrderedDict(
                (
                    prereg.MNEMONIC_TO_VALUE_KEY[mnemonic],
                    rows[mnemonic].value,
                )
                for mnemonic in prereg.SOURCE_ALLOWLIST
            )
            if any(value is None for value in values.values()):
                reasons.append("missing_value")
            else:
                for key, value in values.items():
                    if key.startswith("TV_") and value <= 0:
                        reasons.append("nonpositive_transaction_volume")
                        break
        primitives: OrderedDict[str, Fraction] | None = None
        if not reasons and values is not None:
            primitives = prereg.build_primitives(values)
            complete_vectors += 1
        else:
            invalid_dates += 1
        vectors.append(
            SourceVector(
                observation_date=day,
                available_at=expected_availability(day),
                valid=not reasons,
                invalid_reasons=tuple(sorted(set(reasons))),
                primitives=primitives,
            )
        )

    audit = SourceAudit(
        physical_rows_read=physical_rows,
        eligible_selected_rows_seen=selected_rows_seen,
        eligible_selected_values_converted=values_converted,
        eligible_source_dates=len(vectors),
        complete_vectors=complete_vectors,
        invalid_source_dates=invalid_dates,
    )
    return vectors, audit


def load_source() -> tuple[list[SourceVector], SourceAudit]:
    assert_protocol_committed()
    if sha256_file(prereg.SOURCE) != prereg.SOURCE_SHA256:
        raise RuntimeError("TPCT source observation hash mismatch")
    if sha256_file(prereg.METADATA) != prereg.METADATA_SHA256:
        raise RuntimeError("TPCT source metadata hash mismatch")
    if (
        sha256_file(prereg.SOURCE_MANIFEST)
        != prereg.SOURCE_MANIFEST_SHA256
    ):
        raise RuntimeError("TPCT source manifest file hash mismatch")
    source_manifest = json.loads(
        repository_path(prereg.SOURCE_MANIFEST).read_text(encoding="utf-8")
    )
    recorded = source_manifest.get("manifest_hash")
    if recorded != prereg.SOURCE_MANIFEST_HASH:
        raise RuntimeError("TPCT source manifest hash changed")
    core = dict(source_manifest)
    core.pop("manifest_hash", None)
    if canonical_hash(core) != recorded:
        raise RuntimeError("TPCT source manifest canonical hash changed")
    checks = source_manifest.get("source_checks")
    if not isinstance(checks, dict) or not all(checks.values()):
        raise RuntimeError("TPCT source manifest contains failed checks")
    with gzip.open(
        repository_path(prereg.SOURCE),
        "rt",
        encoding="utf-8",
        newline="",
    ) as handle:
        return parse_source_stream(
            handle,
            expected_rows=EXPECTED_SOURCE_ROWS,
            semantics=selected_semantics(),
        )


def split_name(entry: datetime, exit_time: datetime) -> str | None:
    if prereg.split_contains(entry, exit_time, FEED_FLOOR, TRAIN_END):
        return "train"
    if prereg.split_contains(entry, exit_time, TRAIN_END, SELECTION_END):
        return "selection"
    return None


def build_opportunities(
    vectors: Sequence[SourceVector],
) -> tuple[list[Opportunity], BuildAudit]:
    batches: dict[datetime, list[SourceVector]] = defaultdict(list)
    for vector in vectors:
        if vector.available_at.tzinfo is None:
            raise ValueError("TPCT vector availability must be timezone-aware")
        if prereg.source_value_may_decode(vector.available_at) is not True:
            raise RuntimeError("TPCT sealed vector entered source support")
        batches[vector.available_at].append(vector)

    history: dict[str, list[Fraction]] = {
        primitive: [] for primitive in prereg.PRIMITIVES
    }
    previous_decision: RankDecision | None = None
    last_reserved_exit: datetime | None = None
    opportunities: list[Opportunity] = []
    rank_complete_decisions = 0
    predecessor_only = 0
    token_states = 0
    reservation_suppressed = 0
    split_rejected = 0

    for available_at in sorted(batches):
        batch = sorted(
            batches[available_at],
            key=lambda vector: vector.observation_date,
        )
        invalid = [vector for vector in batch if not vector.valid]
        complete = [
            vector
            for vector in batch
            if vector.valid and vector.primitives is not None
        ]
        ranks: dict[date, OrderedDict[str, Fraction]] = {}
        if all(
            len(history[primitive]) >= prereg.Policy().rank_lookback
            for primitive in prereg.PRIMITIVES
        ):
            for vector in complete:
                assert vector.primitives is not None
                ranks[vector.observation_date] = OrderedDict(
                    (
                        primitive,
                        prereg.strict_prior_midrank(
                            vector.primitives[primitive],
                            history[primitive][
                                -prereg.Policy().rank_lookback :
                            ],
                        ),
                    )
                    for primitive in prereg.PRIMITIVES
                )

        current_decision: RankDecision | None = None
        if not invalid and complete:
            selected = complete[-1]
            selected_ranks = ranks.get(selected.observation_date)
            if selected_ranks is not None:
                rank_complete_decisions += 1
                current_decision = RankDecision(
                    observation_date=selected.observation_date,
                    available_at=available_at,
                    ranks=selected_ranks,
                )
                if previous_decision is None:
                    predecessor_only += 1
                else:
                    tokens = prereg.build_tokens(
                        current_decision.ranks,
                        previous_decision.ranks,
                    )
                    times = prereg.opportunity_times(available_at)
                    reserved = (
                        last_reserved_exit is None
                        or times["entry"] >= last_reserved_exit
                    )
                    split: str | None = None
                    if reserved:
                        last_reserved_exit = times["exit"]
                        split = split_name(times["entry"], times["exit"])
                        if split is None:
                            split_rejected += 1
                    else:
                        reservation_suppressed += 1
                    opportunities.append(
                        Opportunity(
                            observation_date=current_decision.observation_date,
                            available_at=available_at,
                            signal_available=times["signal_available"],
                            entry=times["entry"],
                            exit=times["exit"],
                            tokens=tokens,
                            reserved=reserved,
                            split=split,
                        )
                    )
                    token_states += 1

        for vector in complete:
            assert vector.primitives is not None
            for primitive in prereg.PRIMITIVES:
                history[primitive].append(vector.primitives[primitive])

        if invalid:
            previous_decision = None
        elif current_decision is not None:
            previous_decision = current_decision

    audit = BuildAudit(
        availability_batches=len(batches),
        rank_complete_decisions=rank_complete_decisions,
        predecessor_only_decisions=predecessor_only,
        token_states=token_states,
        reservation_suppressed=reservation_suppressed,
        split_rejected_after_reservation=split_rejected,
    )
    return opportunities, audit


def fraction_payload(value: Fraction | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _contained_opportunities(
    opportunities: Sequence[Opportunity],
) -> list[Opportunity]:
    return sorted(
        (
            row
            for row in opportunities
            if row.reserved and row.split in {"train", "selection"}
        ),
        key=lambda row: (row.entry, row.observation_date),
    )


def _year_rows(
    rows: Sequence[Opportunity],
    year: int,
) -> list[Opportunity]:
    return [row for row in rows if row.entry.year == year]


def _half(row: Opportunity) -> int:
    return 1 if row.entry.month <= 6 else 2


def _quarter(row: Opportunity) -> int:
    return (row.entry.month - 1) // 3 + 1


def _max_gap(rows: Sequence[Opportunity]) -> Fraction | None:
    ordered = sorted(rows, key=lambda row: row.entry)
    if len(ordered) < 2:
        return None
    return max(
        (
            Fraction(
                int((right.entry - left.entry).total_seconds()),
                86_400,
            )
            for left, right in zip(ordered, ordered[1:])
        ),
        default=None,
    )


def summarize_rows(
    rows: Sequence[Opportunity],
    *,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row.entry)
    month_counts = Counter(row.entry.strftime("%Y-%m") for row in ordered)
    maximum_month_share = (
        Fraction(max(month_counts.values()), len(ordered))
        if ordered
        else None
    )
    first_delay = (
        Fraction(int((ordered[0].entry - start).total_seconds()), 86_400)
        if ordered
        else None
    )
    end_lead = (
        Fraction(int((end - ordered[-1].exit).total_seconds()), 86_400)
        if ordered
        else None
    )
    return {
        "opportunities": len(ordered),
        "first_entry": canonical_utc(ordered[0].entry) if ordered else None,
        "last_entry": canonical_utc(ordered[-1].entry) if ordered else None,
        "last_exit": canonical_utc(ordered[-1].exit) if ordered else None,
        "active_months": sorted(month_counts),
        "month_counts": dict(sorted(month_counts.items())),
        "max_month_share": fraction_payload(maximum_month_share),
        "max_entry_gap_days": fraction_payload(_max_gap(ordered)),
        "first_entry_delay_days": fraction_payload(first_delay),
        "last_exit_lead_days": fraction_payload(end_lead),
    }


def token_summary(rows: Sequence[Opportunity]) -> dict[str, Any]:
    distributions: dict[str, dict[str, int]] = {}
    for token, vocabulary in prereg.TOKEN_SCHEMA:
        counts = Counter(row.tokens[token] for row in rows)
        distributions[token] = {
            value: counts.get(value, 0) for value in vocabulary
        }
    signatures = Counter(row.signature for row in rows)
    return {
        "rows": len(rows),
        "distributions": distributions,
        "distinct_signatures": len(signatures),
        "max_signature_share": fraction_payload(
            Fraction(max(signatures.values()), len(rows)) if rows else None
        ),
    }


def _share(count: int, total: int) -> Fraction:
    return Fraction(count, total) if total else Fraction(1)


def token_support_checks(
    train: Sequence[Opportunity],
    selection: Sequence[Opportunity],
) -> tuple[dict[str, bool], dict[str, Any]]:
    checks: dict[str, bool] = {}
    summaries = {
        "train": token_summary(train),
        "selection": token_summary(selection),
    }
    for split_name_value, rows in (
        ("train", train),
        ("selection", selection),
    ):
        total = len(rows)
        distributions = summaries[split_name_value]["distributions"]
        for token in PAIR_TOKENS:
            counts = distributions[token]
            checks[f"{split_name_value}.{token}.each_count_min_3"] = all(
                count >= 3 for count in counts.values()
            )
            checks[f"{split_name_value}.{token}.each_share_min_0_03"] = all(
                _share(count, total) >= Fraction(3, 100)
                for count in counts.values()
            )
            checks[f"{split_name_value}.{token}.max_share_0_85"] = (
                bool(total)
                and max(_share(count, total) for count in counts.values())
                <= Fraction(85, 100)
            )

        leader_distinct_min = 5 if split_name_value == "train" else 4
        for token in LEADER_TOKENS:
            counts = distributions[token]
            non_tie = {
                value: count
                for value, count in counts.items()
                if value != "TIE" and count
            }
            checks[
                f"{split_name_value}.{token}.distinct_nontie_min_"
                f"{leader_distinct_min}"
            ] = len(non_tie) >= leader_distinct_min
            checks[
                f"{split_name_value}.{token}.max_nontie_share_0_50"
            ] = bool(total) and bool(non_tie) and max(
                Fraction(count, total)
                for count in non_tie.values()
            ) <= Fraction(1, 2)
            checks[f"{split_name_value}.{token}.tie_share_0_20"] = (
                bool(total)
                and Fraction(counts.get("TIE", 0), total)
                <= Fraction(1, 5)
            )

        for token, minimum_count, maximum_share in (
            ("rank_breadth", 3, Fraction(90, 100)),
            ("extreme_occupancy", 2, Fraction(92, 100)),
            ("order_transition", 2, Fraction(92, 100)),
        ):
            counts = distributions[token]
            checks[
                f"{split_name_value}.{token}.each_count_min_{minimum_count}"
            ] = all(count >= minimum_count for count in counts.values())
            checks[
                f"{split_name_value}.{token}.max_share_"
                f"{maximum_share.numerator}_{maximum_share.denominator}"
            ] = bool(total) and max(
                _share(count, total) for count in counts.values()
            ) <= maximum_share

        leader_transition_counts = distributions["leader_transition"]
        distinct_min = 4 if split_name_value == "train" else 3
        checks[
            f"{split_name_value}.leader_transition.distinct_min_"
            f"{distinct_min}"
        ] = (
            sum(count > 0 for count in leader_transition_counts.values())
            >= distinct_min
        )
        checks[
            f"{split_name_value}.leader_transition.max_share_0_85"
        ] = bool(total) and max(
            _share(count, total)
            for count in leader_transition_counts.values()
        ) <= Fraction(85, 100)
        max_signature = summaries[split_name_value]["max_signature_share"]
        checks[f"{split_name_value}.max_signature_share_0_15"] = (
            max_signature is not None
            and Fraction(
                max_signature["numerator"],
                max_signature["denominator"],
            )
            <= Fraction(15, 100)
        )

    train_values = {
        token: {
            row.tokens[token]
            for row in train
        }
        for token in prereg.TOKEN_COLUMNS
    }
    selection_values = {
        token: {
            row.tokens[token]
            for row in selection
        }
        for token in prereg.TOKEN_COLUMNS
    }
    checks["selection_token_values_seen_in_train"] = all(
        selection_values[token].issubset(train_values[token])
        for token in prereg.TOKEN_COLUMNS
    )
    return checks, summaries


def source_support(
    opportunities: Sequence[Opportunity],
    *,
    source_audit: SourceAudit,
) -> dict[str, Any]:
    emitted = _contained_opportunities(opportunities)
    train = [row for row in emitted if row.split == "train"]
    selection = [row for row in emitted if row.split == "selection"]
    checks: dict[str, bool] = {
        "train_total_min_75": len(train) >= 75,
        "train_2020_min_15": len(_year_rows(train, 2020)) >= 15,
        "train_2021_min_50": len(_year_rows(train, 2021)) >= 50,
        "selection_2022_min_55": len(selection) >= 55,
        "2020_active_months_min_3": len(
            {row.entry.month for row in _year_rows(train, 2020)}
        )
        >= 3,
        "2021_active_months_min_11": len(
            {row.entry.month for row in _year_rows(train, 2021)}
        )
        >= 11,
        "2022_active_months_min_11": len(
            {row.entry.month for row in selection}
        )
        >= 11,
        "sealed_values_converted_zero": (
            source_audit.sealed_values_converted == 0
        ),
        "sealed_candidate_statistics_zero": (
            source_audit.sealed_candidate_statistics == 0
        ),
    }
    for year, rows in (
        (2021, _year_rows(train, 2021)),
        (2022, selection),
    ):
        for half in (1, 2):
            checks[f"{year}_half_{half}_min_23"] = (
                sum(_half(row) == half for row in rows) >= 23
            )
        for quarter in (1, 2, 3, 4):
            checks[f"{year}_quarter_{quarter}_min_10"] = (
                sum(_quarter(row) == quarter for row in rows) >= 10
            )

    train_summary = summarize_rows(
        train,
        start=FEED_FLOOR,
        end=TRAIN_END,
    )
    selection_summary = summarize_rows(
        selection,
        start=TRAIN_END,
        end=SELECTION_END,
    )

    def payload_fraction(
        summary: Mapping[str, Any],
        key: str,
    ) -> Fraction | None:
        raw = summary.get(key)
        if not isinstance(raw, dict):
            return None
        return Fraction(raw["numerator"], raw["denominator"])

    for name, summary in (
        ("train", train_summary),
        ("selection", selection_summary),
    ):
        month_share = payload_fraction(summary, "max_month_share")
        max_gap = payload_fraction(summary, "max_entry_gap_days")
        end_lead = payload_fraction(summary, "last_exit_lead_days")
        checks[f"{name}_max_month_share_0_20"] = (
            month_share is not None and month_share <= Fraction(1, 5)
        )
        checks[f"{name}_max_gap_days_10"] = (
            max_gap is not None and max_gap <= 10
        )
        checks[f"{name}_last_exit_lead_days_15"] = (
            end_lead is not None
            and 0 <= end_lead <= 15
        )

    train_start_delay = payload_fraction(
        train_summary,
        "first_entry_delay_days",
    )
    selection_start_delay = payload_fraction(
        selection_summary,
        "first_entry_delay_days",
    )
    checks["train_start_delay_days_21"] = (
        train_start_delay is not None
        and 0 <= train_start_delay <= 21
    )
    checks["selection_start_delay_days_15"] = (
        selection_start_delay is not None
        and 0 <= selection_start_delay <= 15
    )
    blackout = (
        Fraction(
            int((selection[0].entry - train[-1].entry).total_seconds()),
            86_400,
        )
        if train and selection
        else None
    )
    checks["cross_boundary_blackout_days_20"] = (
        blackout is not None and 0 <= blackout <= 20
    )

    token_checks, token_summaries = token_support_checks(train, selection)
    checks.update(token_checks)
    return {
        "passed": all(checks.values()),
        "checks": dict(sorted(checks.items())),
        "train": train_summary,
        "selection": selection_summary,
        "cross_boundary_blackout_days": fraction_payload(blackout),
        "tokens": token_summaries,
    }


def _clock_row(row: Opportunity) -> dict[str, str]:
    if not row.reserved or row.split not in {"train", "selection"}:
        raise ValueError("TPCT clock row is not emitted")
    payload = {
        "candidate": prereg.POLICY_ID,
        "split": str(row.split),
        "observation_date": row.observation_date.isoformat(),
        "available_at_utc": canonical_utc(row.available_at),
        "signal_available": canonical_utc(row.signal_available),
        "entry": canonical_utc(row.entry),
        "exit": canonical_utc(row.exit),
        **{name: row.tokens[name] for name in prereg.TOKEN_COLUMNS},
        "token_signature_sha256": canonical_hash(
            {
                name: row.tokens[name]
                for name in prereg.TOKEN_COLUMNS
            }
        ),
    }
    if tuple(payload) != CLOCK_COLUMNS:
        raise RuntimeError("TPCT clock schema changed")
    if FORBIDDEN_CLOCK_COLUMNS.intersection(payload):
        raise RuntimeError("TPCT clock opened a forbidden column")
    return payload


def gzip_clock(rows: Sequence[Opportunity]) -> bytes:
    raw = io.StringIO(newline="")
    writer = csv.DictWriter(
        raw,
        fieldnames=CLOCK_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in _contained_opportunities(rows):
        writer.writerow(_clock_row(row))
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as zipped:
        zipped.write(raw.getvalue().encode("utf-8"))
    return output.getvalue()


def atomic_write(path: Path, payload: bytes) -> str:
    target = repository_path(path)
    if target.exists():
        if target.read_bytes() != payload:
            raise RuntimeError(f"existing TPCT artifact differs: {path}")
        return "verified"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        if target.exists():
            if target.read_bytes() != payload:
                raise RuntimeError(
                    f"concurrent TPCT artifact differs: {path}"
                )
        else:
            os.replace(temporary, target)
            if os.name == "posix":
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                directory_fd = os.open(target.parent, flags)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def build_report(
    *,
    protocol_commit: str,
    source_audit: SourceAudit,
    build_audit: BuildAudit,
    opportunities: Sequence[Opportunity],
    support: Mapping[str, Any],
    clock_sha256: str,
) -> dict[str, Any]:
    emitted = _contained_opportunities(opportunities)
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "candidate": prereg.POLICY_ID,
        "protocol_commit": protocol_commit,
        "bindings": {
            "preregistration": {
                "path": PREREGISTRATION,
                "sha256": PREREGISTRATION_SHA256,
                "manifest_hash": PREREGISTRATION_MANIFEST_HASH,
                "contract_hash": PREREGISTRATION_CONTRACT_HASH,
            },
            "source": {
                "path": prereg.SOURCE,
                "sha256": prereg.SOURCE_SHA256,
            },
            "metadata": {
                "path": prereg.METADATA,
                "sha256": prereg.METADATA_SHA256,
                "selected_hash": prereg.SELECTED_METADATA_HASH,
            },
        },
        "source_audit": {
            "physical_rows_read": source_audit.physical_rows_read,
            "eligible_selected_rows_seen": (
                source_audit.eligible_selected_rows_seen
            ),
            "eligible_selected_values_converted": (
                source_audit.eligible_selected_values_converted
            ),
            "eligible_source_dates": source_audit.eligible_source_dates,
            "complete_vectors": source_audit.complete_vectors,
            "invalid_source_dates": source_audit.invalid_source_dates,
            "sealed_values_converted": source_audit.sealed_values_converted,
            "sealed_candidate_statistics": (
                source_audit.sealed_candidate_statistics
            ),
            "sealed_row_count_reported": False,
        },
        "build_audit": {
            "availability_batches": build_audit.availability_batches,
            "rank_complete_decisions": (
                build_audit.rank_complete_decisions
            ),
            "predecessor_only_decisions": (
                build_audit.predecessor_only_decisions
            ),
            "token_states": build_audit.token_states,
            "reservation_suppressed": build_audit.reservation_suppressed,
            "split_rejected_after_reservation": (
                build_audit.split_rejected_after_reservation
            ),
            "emitted_opportunities": len(emitted),
        },
        "source_support": dict(support),
        "clock_artifact": {
            "path": str(DEFAULT_CLOCK),
            "sha256": clock_sha256,
            "rows": len(emitted),
            "columns": list(CLOCK_COLUMNS),
            "contains_raw_values": False,
            "contains_numeric_ranks": False,
            "contains_action_or_side": False,
            "contains_market_funding_or_outcomes": False,
        },
        "decision": (
            "pass_source_support_and_authorize_preoutcome_clock_novelty"
            if support["passed"]
            else "retire_TPCT_120_unchanged_before_comparators_or_outcomes"
        ),
        "outcome_boundary": {
            "comparator_rows_read": 0,
            "market_rows_read": 0,
            "funding_rows_read": 0,
            "return_or_pnl_rows_read": 0,
            "model_labels_created": 0,
            "model_training_runs": 0,
            "sealed_values_converted": 0,
            "network_calls": 0,
            "subprocess_scope": "fixed git commit/cleanliness checks only",
        },
        "next_action": (
            "commit report and run frozen pre-outcome comparator clock novelty"
            if support["passed"]
            else "retire TPCT-120 unchanged and select a new mechanism"
        ),
    }
    report["manifest_hash"] = canonical_hash(report)
    return report


def run(
    *,
    clock_path: Path = DEFAULT_CLOCK,
    report_path: Path = DEFAULT_REPORT,
) -> tuple[dict[str, Any], dict[str, str]]:
    protocol_commit = assert_protocol_committed()
    load_registration()
    vectors, source_audit = load_source()
    opportunities, build_audit = build_opportunities(vectors)
    support = source_support(
        opportunities,
        source_audit=source_audit,
    )
    clock_bytes = gzip_clock(opportunities)
    clock_sha256 = hashlib.sha256(clock_bytes).hexdigest()
    report = build_report(
        protocol_commit=protocol_commit,
        source_audit=source_audit,
        build_audit=build_audit,
        opportunities=opportunities,
        support=support,
        clock_sha256=clock_sha256,
    )
    report["clock_artifact"]["path"] = str(clock_path)
    core = {
        key: value
        for key, value in report.items()
        if key != "manifest_hash"
    }
    report["manifest_hash"] = canonical_hash(core)
    statuses = {
        "clock": atomic_write(clock_path, clock_bytes),
        "report": atomic_write(report_path, canonical_json_bytes(report)),
    }
    return report, statuses


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock", default=str(DEFAULT_CLOCK))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report, statuses = run(
        clock_path=Path(args.clock),
        report_path=Path(args.report),
    )
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "decision": report["decision"],
                "manifest_hash": report["manifest_hash"],
                "support_passed": report["source_support"]["passed"],
                "train_opportunities": report["source_support"]["train"][
                    "opportunities"
                ],
                "selection_opportunities": report["source_support"][
                    "selection"
                ]["opportunities"],
                "outcome_boundary": report["outcome_boundary"],
                "statuses": statuses,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
