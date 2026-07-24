"""Build the ECRL-1 zero-model-call synthetic preregistration exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


POLICY_ID = "ECRL-1"
PROTOCOL_VERSION = "edgar_claim_relation_language_m0_v1"
AS_OF_DATE = "2026-07-25"
SEED = 20_260_725
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

MECHANISM_DOCUMENT = Path(
    "docs/edgar-claim-relation-language-rl-mechanism-decision-2026-07-25.md"
)
MECHANISM_DOCUMENT_SHA256 = (
    "cd618ba1ff8e6afd452c719234d4d10b163a7e3856e215d676d3d8dd799f6889"
)
MECHANISM_COMMIT = "ef92d2894817e6a9780d13cd3f7f31316cdea60b"

SOURCE_ARTIFACT = Path("data/sec_edgar_bitcoin_8k_6k_source_2018_2023.jsonl.gz")
SOURCE_ARTIFACT_SHA256 = (
    "c8489dfe9b4ac25da8bea7653115e5b58a44fa897f2815eaf68bad354e10c6ce"
)
SOURCE_CANONICAL_ROWS_SHA256 = (
    "98793185f1e411d8c59736fb54c5ed529d539e81ccddf2c823f24127ecfcef0b"
)

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
MODEL_FILES: Mapping[str, str] = {
    "chat_template.jinja": (
        "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"
    ),
    "config.json": "1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330",
    "model.safetensors": (
        "2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550"
    ),
    "processor_config.json": (
        "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c"
    ),
    "tokenizer.json": (
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
    ),
    "tokenizer_config.json": (
        "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633"
    ),
}

PROMPT_TEMPLATE = """You compare two public-company Bitcoin disclosure excerpts from the same
issuer. Decide only what the CURRENT issuer claim says and how it relates to
the PRIOR issuer claim.

STATUS: A realized/completed; B planned; C conditional; D risk-only;
E third-party/customer/market; F no supported issuer claim; G mixed.
DELTA: U issuer Bitcoin economic exposure up; V down; W explicitly unchanged;
X unsupported direction. This is issuer exposure, never BTC price direction.
RELATION: F fulfills prior plan/condition/warning; R reverses or cancels it;
N new relative to rendered prior evidence; P repeats a comparable rendered
prior claim/state, including planned or conditional repeat with status B/C;
I incomparable or unsupported.

Use one CURRENT evidence id C1..C8 and one PRIOR evidence id P1..P8 when the
grammar requires them. Do not infer omitted facts. Planned, conditional,
risk-only, third-party, negated, and mixed claims are not realized actions.

PRIOR
{prior_numbered_sentences}

CURRENT
{current_numbered_sentences}

Return exactly one ASCII line and nothing else:
STATUS|DELTA|RELATION|CURRENT_EVIDENCE|PRIOR_EVIDENCE"""

SCENARIO_TARGETS: Mapping[str, str] = {
    "FULFILL_UP": "A|U|F|C2|P2",
    "FULFILL_DOWN": "A|V|F|C2|P2",
    "REVERSE_UP": "A|U|R|C2|P2",
    "REVERSE_DOWN": "A|V|R|C2|P2",
    "NEW_UP": "A|U|N|C2|NONE",
    "NEW_DOWN": "A|V|N|C2|NONE",
    "REALIZED_REPEAT": "A|W|P|C2|P2",
    "PLANNED_UP": "B|U|N|C2|NONE",
    "PLANNED_DOWN": "B|V|N|C2|NONE",
    "CONDITIONAL_UP": "C|U|N|C2|NONE",
    "CONDITIONAL_DOWN": "C|V|N|C2|NONE",
    "RISK_ONLY": "D|X|I|C2|NONE",
    "THIRD_PARTY": "E|X|I|C2|NONE",
    "NO_CLAIM": "F|X|I|NONE|NONE",
    "MIXED": "G|X|I|NONE|NONE",
    "PLANNED_REPEAT": "B|W|P|C2|P2",
}

SPLIT_ROWS_PER_SCENARIO: Mapping[str, int] = {
    "train": 256,
    "calibration": 32,
    "adversarial": 48,
    "swap": 32,
}

ROW_KEYS = (
    "row_id",
    "split",
    "scenario_id",
    "template_id",
    "prior",
    "current",
    "target",
    "pair_id",
)

ALIASES = (
    "Atlas Digital Holdings Inc.",
    "Beacon Compute Corporation",
    "Cedar Treasury Limited",
    "Delta Mining L.L.C.",
    "Evergreen Access PLC",
    "Frontier Collateral Group",
    "Granite Bitcoin Company",
    "Harbor Systems Ltd.",
    "Ion Ledger Incorporated",
    "Juniper Network Holdings Corp.",
    "Keystone Digital Assets LLC",
    "Lattice Mining Corporation",
    "Meridian Holdings Inc.",
    "Northstar Compute Limited",
    "Orchard Custody PLC",
    "Pioneer Treasury Corp.",
)
DATES = (
    "January 3, 2022",
    "February 7, 2022",
    "March 11, 2022",
    "April 14, 2022",
    "2022-05-19",
    "2022-06-23",
    "07/27/2022",
    "08/31/2022",
)
AMOUNTS = (
    "$1.2 million",
    "$2.4 million",
    "$3.6 million",
    "$4.8 million",
    "$12 million",
    "$24 million",
    "$36 million",
    "$48 million",
)
QUANTITIES = (
    "120",
    "240",
    "360",
    "480",
    "1,200",
    "2,400",
    "3,600",
    "4,800",
)
EXCHANGES = ("NASDAQ", "NYSE", "NYSEAMERICAN", "TSX", "LSE")
TICKERS = ("ATLS", "BCON", "CEDR", "DLTA", "EVRG", "FRNT", "GRNT", "HARB")

DOMAIN_TEMPLATES: Mapping[str, Mapping[str, str]] = {
    "inventory": {
        "up_action": "increase its Bitcoin inventory by {quantity} units",
        "down_action": "reduce its Bitcoin inventory by {quantity} units",
        "up_state": "now holds {quantity} more units of Bitcoin than before",
        "down_state": "now holds {quantity} fewer units of Bitcoin than before",
    },
    "mining": {
        "up_action": "bring {quantity} units of Bitcoin mining capacity online",
        "down_action": "take {quantity} units of Bitcoin mining capacity offline",
        "up_state": "now operates {quantity} additional units of Bitcoin mining capacity",
        "down_state": "has shut down {quantity} units of Bitcoin mining capacity",
    },
    "access": {
        "up_action": "expand issuer-provided Bitcoin custody access to {quantity} accounts",
        "down_action": "withdraw issuer-provided Bitcoin custody access from {quantity} accounts",
        "up_state": "now provides Bitcoin custody access to {quantity} additional accounts",
        "down_state": "has withdrawn Bitcoin custody access from {quantity} accounts",
    },
    "collateral": {
        "up_action": "increase its Bitcoin-backed collateral commitment by {amount}",
        "down_action": "reduce its Bitcoin-backed collateral commitment by {amount}",
        "up_state": "now commits {amount} more to Bitcoin-backed collateral",
        "down_state": "now commits {amount} less to Bitcoin-backed collateral",
    },
}

SPLIT_WORDING: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "train": {
        "planned": (
            "{issuer} planned to {action}.",
            "{issuer} intended to {action}.",
            "{issuer} approved a plan to {action}.",
            "{issuer} set an objective to {action}.",
        ),
        "realized": (
            "{issuer} has completed the change and {state}.",
            "{issuer} reports that it {state}.",
            "{issuer} finished the action and {state}.",
            "{issuer} confirms that it {state}.",
        ),
        "conditional": (
            "{issuer} may {action} if financing closes.",
            "{issuer} could {action} subject to approval.",
            "{issuer} may {action} if operating conditions permit.",
            "{issuer} could {action} subject to available capital.",
        ),
    },
    "calibration": {
        "planned": (
            "{issuer} expected to {action}.",
            "{issuer} announced an intention to {action}.",
            "{issuer} proposed that it would {action}.",
            "{issuer} described a future step to {action}.",
        ),
        "realized": (
            "{issuer} states that the step is complete and it {state}.",
            "{issuer} says the completed change means it {state}.",
            "{issuer} records a finished transition and {state}.",
            "{issuer} discloses that it {state} after completion.",
        ),
        "conditional": (
            "{issuer} might {action} if a closing occurs.",
            "{issuer} may {action} subject to a board condition.",
            "{issuer} could {action} if a counterparty performs.",
            "{issuer} may {action} subject to regulatory clearance.",
        ),
    },
    "adversarial": {
        "planned": (
            "{issuer} outlined a prospective decision to {action}.",
            "{issuer} communicated a not-yet-completed aim to {action}.",
            "{issuer} authorized preparations to {action}.",
            "{issuer} scheduled a future effort to {action}.",
        ),
        "realized": (
            "{issuer} verifies that execution is finished and it {state}.",
            "{issuer} reports a completed transition under which it {state}.",
            "{issuer} says implementation ended and it {state}.",
            "{issuer} confirms completion, after which it {state}.",
        ),
        "conditional": (
            "{issuer} might {action} only if a covenant is satisfied.",
            "{issuer} may {action} subject to liquidity being available.",
            "{issuer} could {action} if a stated contingency occurs.",
            "{issuer} may {action} only after a required consent.",
        ),
    },
    "swap": {
        "planned": (
            "{issuer} recorded a forward-looking plan to {action}.",
            "{issuer} documented an uncompleted intent to {action}.",
            "{issuer} described a pending program to {action}.",
            "{issuer} disclosed a future proposal to {action}.",
        ),
        "realized": (
            "{issuer} records that implementation is complete and it {state}.",
            "{issuer} documents a completed step through which it {state}.",
            "{issuer} reports final execution and says it {state}.",
            "{issuer} states the transition finished and it {state}.",
        ),
        "conditional": (
            "{issuer} may {action} if a contractual condition is met.",
            "{issuer} could {action} subject to financing availability.",
            "{issuer} might {action} if a required event occurs.",
            "{issuer} may {action} subject to a third-party consent.",
        ),
    },
}

SPLIT_CONTEXT: Mapping[str, tuple[str, str]] = {
    "train": (
        "The quarterly section also summarizes ordinary accounting policy.",
        "The operating appendix contains no additional exposure claim.",
    ),
    "calibration": (
        "The interim report also describes routine governance procedures.",
        "The notes contain no additional directional exposure statement.",
    ),
    "adversarial": (
        "The event report also lists ordinary administrative matters.",
        "The exhibit contains unrelated corporate background.",
    ),
    "swap": (
        "The current report also covers ordinary compliance matters.",
        "The attachment adds no other directional exposure statement.",
    ),
}

LEXICON: Mapping[str, tuple[str, ...]] = {
    "UP": (
        "acquire",
        "acquired",
        "acquisition",
        "purchase",
        "purchased",
        "increase",
        "increased",
        "expand",
        "expanded",
        "launch",
        "launched",
        "commission",
        "commissioned",
        "energize",
        "energized",
        "add",
        "added",
        "accept",
        "accepted",
        "custody",
        "collateralized",
    ),
    "DOWN": (
        "sell",
        "sold",
        "sale",
        "decrease",
        "decreased",
        "reduce",
        "reduced",
        "shutdown",
        "shut down",
        "decommission",
        "decommissioned",
        "cease",
        "ceased",
        "terminate",
        "terminated",
        "cancel",
        "cancelled",
        "canceled",
        "withdraw",
        "withdrew",
        "impair",
        "impaired",
    ),
    "REALIZED": (
        "has",
        "have",
        "completed",
        "commenced",
        "began",
        "now",
        "entered into",
        "was",
        "were",
    ),
    "PLANNED": (
        "plan",
        "plans",
        "planned",
        "intend",
        "intends",
        "expected to",
        "will",
        "propose",
        "proposes",
    ),
    "CONDITIONAL": ("may", "might", "could", "subject to", "if"),
    "RISK": ("risk", "uncertain", "uncertainty", "adverse"),
    "THIRD": (
        "customer",
        "customers",
        "client",
        "clients",
        "investee",
        "third party",
        "market participants",
    ),
    "NEGATION": ("not", "no", "never", "without", "did not", "has not", "have not"),
}

PREFILTER_LITERALS = (
    "ignore previous",
    "ignore all previous",
    "ignore the previous",
    "system prompt",
    "developer message",
    "assistant response",
    "<start_of_turn>",
    "<end_of_turn>",
    "STATUS|DELTA",
    "CURRENT_EVIDENCE|PRIOR_EVIDENCE",
)

OUTPUT_PATTERN = re.compile(
    r"(?:[A-G])\|(?:[U-X])\|(?:[FRNPI])\|"
    r"(?:C[1-8]|NONE)\|(?:P[1-8]|NONE)",
    re.ASCII,
)

CORPORATE_SUFFIXES = (
    "CORPORATION",
    "CORP",
    "INCORPORATED",
    "INC",
    "LIMITED",
    "LTD",
    "LLC",
    "L.L.C",
    "PLC",
    "P.L.C",
    "LP",
    "L.P",
    "LLP",
    "L.L.P",
    "CO",
    "COMPANY",
    "HOLDINGS",
    "GROUP",
)

SENTINELS: Mapping[str, str] = {
    "\ue000": "[ENTITY]",
    "\ue001": "[SYMBOL]",
    "\ue002": "[DATE]",
    "\ue003": "[TIME]",
    "\ue004": "[NUM]",
    "\ue005": "[LINK]",
    "\ue006": "[ID]",
}

EXCHANGE_SYMBOL_PATTERN = re.compile(
    r"(?i:\b(?:NASDAQ|NYSE|NYSEAMERICAN|TSX|LSE)\s*:\s*"
    r"[A-Z][A-Z0-9.]{0,5}\b)"
)
DOLLAR_SYMBOL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])\$[A-Za-z][A-Za-z0-9.]{0,5}(?![A-Za-z0-9])"
)
CAPITAL_SYMBOL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[A-Z][A-Z0-9.]{1,5}(?![A-Za-z0-9])"
)
LINK_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
    re.IGNORECASE,
)
ID_PATTERN = re.compile(
    r"\b(?:CIK\s*)?\d{10}\b|\b\d{10}-\d{2}-\d{6}\b",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(
    r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\d{1,2}/\d{1,2}/20\d{2}|"
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s+20\d{2})\b",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:UTC|AM|PM))?\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[$€£₩]\s*)?\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:thousand|million|billion|trillion))?"
    r"(?:\s*%)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Config:
    output: str = (
        "results/edgar_claim_relation_language_m0_preregistration_2026-07-25.json"
    )
    prompt_output: str = (
        "results/edgar_claim_relation_language_prompt_template_2026-07-25.txt"
    )
    inventory_output: str = (
        "results/edgar_claim_relation_language_template_inventory_2026-07-25.json"
    )
    train_output: str = (
        "data/edgar_claim_relation_language_synthetic_train_2026-07-25.jsonl"
    )
    calibration_output: str = (
        "data/edgar_claim_relation_language_synthetic_calibration_2026-07-25.jsonl"
    )
    adversarial_output: str = (
        "data/edgar_claim_relation_language_synthetic_adversarial_2026-07-25.jsonl"
    )
    swap_output: str = (
        "data/edgar_claim_relation_language_synthetic_swap_2026-07-25.jsonl"
    )


def _path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any, *, terminal_lf: bool = False) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return encoded + (b"\n" if terminal_lf else b"")


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _normalize_line(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    if any(unicodedata.category(character) == "Co" for character in normalized):
        raise ValueError("source text contains a private-use code point")
    visible = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in normalized
    )
    return " ".join(visible.split())


def _suffix_pattern() -> re.Pattern[str]:
    alternatives = sorted(
        (re.escape(value).replace(r"\.", r"\.?") for value in CORPORATE_SUFFIXES),
        key=len,
        reverse=True,
    )
    return re.compile(
        rf"(?:\s*,?\s+)(?:{'|'.join(alternatives)})\.?$",
        re.IGNORECASE,
    )


SUFFIX_PATTERN = _suffix_pattern()


@lru_cache(maxsize=512)
def alias_variants(alias: str) -> tuple[str, ...]:
    normalized = _normalize_line(alias)
    variants: set[str] = {normalized}
    shortened = normalized
    while True:
        match = SUFFIX_PATTERN.search(shortened)
        if match is None:
            break
        shortened = shortened[: match.start()].strip(" ,.")
        if not shortened:
            break
        variants.add(shortened)
    variants.update(
        " ".join(value.replace(".", "").split()) for value in tuple(variants)
    )
    return tuple(
        sorted(
            (value for value in variants if value),
            key=lambda value: (-len(value), value.casefold()),
        )
    )


@lru_cache(maxsize=64)
def _alias_patterns(aliases: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    variants = sorted(
        {variant for source in aliases for variant in alias_variants(source)},
        key=lambda value: (-len(value), value.casefold()),
    )
    return tuple(
        re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        for alias in variants
    )


def redact_line(text: str, *, aliases: Sequence[str]) -> str:
    redacted = _normalize_line(text)
    for pattern in _alias_patterns(tuple(aliases)):
        redacted = pattern.sub("\ue000", redacted)
    redacted = LINK_PATTERN.sub("\ue005", redacted)
    redacted = ID_PATTERN.sub("\ue006", redacted)
    redacted = DATE_PATTERN.sub("\ue002", redacted)
    redacted = TIME_PATTERN.sub("\ue003", redacted)
    redacted = NUMBER_PATTERN.sub("\ue004", redacted)
    redacted = EXCHANGE_SYMBOL_PATTERN.sub("\ue001", redacted)
    redacted = DOLLAR_SYMBOL_PATTERN.sub("\ue001", redacted)
    redacted = CAPITAL_SYMBOL_PATTERN.sub("\ue001", redacted)
    for sentinel, replacement in SENTINELS.items():
        redacted = redacted.replace(sentinel, replacement)
    return " ".join(redacted.split())


def redact_numbered_text(text: str, *, aliases: Sequence[str]) -> str:
    output: list[str] = []
    for line in text.splitlines():
        match = re.fullmatch(r"([PC][1-8]):\s*(.*)", line)
        if match is None:
            raise ValueError("numbered text line is malformed")
        output.append(
            f"{match.group(1)}: {redact_line(match.group(2), aliases=aliases)}"
        )
    return "\n".join(output)


def prefilter_reason(prior: str, current: str) -> str | None:
    corpus = unicodedata.normalize("NFKC", f"{prior}\n{current}").casefold()
    for literal in PREFILTER_LITERALS:
        if literal.casefold() in corpus:
            return literal
    return None


def _evidence_ids(numbered_text: str, prefix: str) -> set[str]:
    values: set[str] = set()
    for line in numbered_text.splitlines():
        match = re.fullmatch(rf"({prefix}[1-8]):\s*.+", line)
        if match is not None:
            values.add(match.group(1))
    return values


def parse_model_output(
    output: str,
    *,
    prior: str,
    current: str,
) -> dict[str, Any]:
    candidate = output[:-1] if output.endswith("\n") else output
    if OUTPUT_PATTERN.fullmatch(candidate) is None:
        return {"valid": False, "error": "malformed"}
    status, delta, relation, current_id, prior_id = candidate.split("|")
    current_ids = _evidence_ids(current, "C")
    prior_ids = _evidence_ids(prior, "P")
    if (current_id != "NONE" and current_id not in current_ids) or (
        prior_id != "NONE" and prior_id not in prior_ids
    ):
        return {"valid": False, "error": "01_missing_evidence_id"}
    if status == "G" and (delta, relation, current_id, prior_id) != (
        "X",
        "I",
        "NONE",
        "NONE",
    ):
        return {"valid": False, "error": "02_mixed_contract"}
    if status in {"D", "E"} and not (
        delta == "X"
        and relation == "I"
        and current_id != "NONE"
        and prior_id == "NONE"
    ):
        return {"valid": False, "error": "03_risk_or_third_contract"}
    if status == "F" and (delta, relation, current_id, prior_id) != (
        "X",
        "I",
        "NONE",
        "NONE",
    ):
        return {"valid": False, "error": "04_no_claim_contract"}
    if relation == "N" and not (
        prior_id == "NONE" and current_id != "NONE"
    ):
        return {"valid": False, "error": "05_new_evidence_contract"}
    if relation in {"F", "R", "P"} and (
        current_id == "NONE" or prior_id == "NONE"
    ):
        return {"valid": False, "error": "06_relational_evidence_contract"}
    if delta in {"U", "V", "W"} and current_id == "NONE":
        return {"valid": False, "error": "07_directional_evidence_contract"}
    if relation == "P" and delta != "W":
        return {"valid": False, "error": "08_repeat_delta_contract"}
    if delta == "W" and not (
        relation == "P" and status in {"A", "B", "C"}
    ):
        return {"valid": False, "error": "09_unchanged_relation_contract"}
    if relation == "I" and status in {"A", "B", "C"}:
        return {"valid": False, "error": "10_supported_incomparable_contract"}
    return {
        "valid": True,
        "error": None,
        "status": status,
        "delta": delta,
        "relation": relation,
        "current_evidence": current_id,
        "prior_evidence": prior_id,
    }


def render_prompt(row: Mapping[str, Any]) -> str | None:
    prior = str(row["prior"])
    current = str(row["current"])
    if prefilter_reason(prior, current) is not None:
        return None
    return PROMPT_TEMPLATE.format(
        prior_numbered_sentences=redact_numbered_text(prior, aliases=ALIASES),
        current_numbered_sentences=redact_numbered_text(current, aliases=ALIASES),
    )


def _choice(
    values: Sequence[str],
    *,
    split: str,
    scenario_id: str,
    ordinal: int,
    field: str,
) -> str:
    if not values:
        raise ValueError("choice pool is empty")
    payload = (
        f"{POLICY_ID}|{SEED}|{split}|{scenario_id}|{ordinal}|{field}"
    ).encode("ascii")
    index = int.from_bytes(hashlib.sha256(payload).digest(), "big") % len(values)
    return values[index]


def _surface_values(
    *,
    split: str,
    scenario_id: str,
    ordinal: int,
    variant: int | None,
) -> dict[str, str]:
    suffix = "" if variant is None else f"_v{variant}"
    return {
        "issuer": _choice(
            ALIASES,
            split=split,
            scenario_id=scenario_id,
            ordinal=ordinal,
            field=f"issuer{suffix}",
        ),
        "date": _choice(
            DATES,
            split=split,
            scenario_id=scenario_id,
            ordinal=ordinal,
            field=f"date{suffix}",
        ),
        "amount": _choice(
            AMOUNTS,
            split=split,
            scenario_id=scenario_id,
            ordinal=ordinal,
            field=f"amount{suffix}",
        ),
        "quantity": _choice(
            QUANTITIES,
            split=split,
            scenario_id=scenario_id,
            ordinal=ordinal,
            field=f"quantity{suffix}",
        ),
        "exchange": _choice(
            EXCHANGES,
            split=split,
            scenario_id=scenario_id,
            ordinal=ordinal,
            field=f"exchange{suffix}",
        ),
        "ticker": _choice(
            TICKERS,
            split=split,
            scenario_id=scenario_id,
            ordinal=ordinal,
            field=f"ticker{suffix}",
        ),
    }


def _domain_values(domain: str, surfaces: Mapping[str, str]) -> dict[str, str]:
    values = DOMAIN_TEMPLATES[domain]
    return {
        key: template.format(
            amount=surfaces["amount"],
            quantity=surfaces["quantity"],
        )
        for key, template in values.items()
    }


def _wording(
    split: str,
    category: str,
    style: int,
    *,
    issuer: str,
    action: str = "",
    state: str = "",
) -> str:
    template = SPLIT_WORDING[split][category][style]
    return template.format(issuer=issuer, action=action, state=state)


def _decision_sentences(
    *,
    split: str,
    scenario_id: str,
    style: int,
    domain: str,
    surfaces: Mapping[str, str],
) -> tuple[str, str]:
    issuer = surfaces["issuer"]
    domain_values = _domain_values(domain, surfaces)
    up_action = domain_values["up_action"]
    down_action = domain_values["down_action"]
    up_state = domain_values["up_state"]
    down_state = domain_values["down_state"]
    planned_up = _wording(
        split, "planned", style, issuer=issuer, action=up_action
    )
    planned_down = _wording(
        split, "planned", style, issuer=issuer, action=down_action
    )
    realized_up = _wording(
        split, "realized", style, issuer=issuer, state=up_state
    )
    realized_down = _wording(
        split, "realized", style, issuer=issuer, state=down_state
    )
    conditional_up = _wording(
        split, "conditional", style, issuer=issuer, action=up_action
    )
    conditional_down = _wording(
        split, "conditional", style, issuer=issuer, action=down_action
    )
    unrelated = (
        f"{issuer} discussed Bitcoin accounting terminology without stating "
        "an issuer exposure change."
    )
    cases = {
        "FULFILL_UP": (planned_up, realized_up),
        "FULFILL_DOWN": (planned_down, realized_down),
        "REVERSE_UP": (planned_down, realized_up),
        "REVERSE_DOWN": (planned_up, realized_down),
        "NEW_UP": (unrelated, realized_up),
        "NEW_DOWN": (unrelated, realized_down),
        "REALIZED_REPEAT": (realized_up, realized_up),
        "PLANNED_UP": (unrelated, planned_up),
        "PLANNED_DOWN": (unrelated, planned_down),
        "CONDITIONAL_UP": (unrelated, conditional_up),
        "CONDITIONAL_DOWN": (unrelated, conditional_down),
        "RISK_ONLY": (
            unrelated,
            f"{issuer} faces adverse Bitcoin price risk that could affect "
            "reported results.",
        ),
        "THIRD_PARTY": (
            unrelated,
            f"{issuer} said its customers may purchase Bitcoin through a "
            "third party.",
        ),
        "NO_CLAIM": (
            unrelated,
            "The section defines Bitcoin terminology but states no supported "
            "issuer claim.",
        ),
        "MIXED": (
            unrelated,
            f"{issuer} completed one step to {up_action} and another step to "
            f"{down_action}.",
        ),
        "PLANNED_REPEAT": (planned_up, planned_up),
    }
    return cases[scenario_id]


def _numbered_pair(
    *,
    split: str,
    scenario_id: str,
    ordinal: int,
    variant: int | None,
) -> tuple[str, str, str]:
    surfaces = _surface_values(
        split=split,
        scenario_id=scenario_id,
        ordinal=ordinal,
        variant=variant,
    )
    domain_names = tuple(sorted(DOMAIN_TEMPLATES))
    domain = _choice(
        domain_names,
        split=split,
        scenario_id=scenario_id,
        ordinal=ordinal,
        field="domain",
    )
    style = int(
        _choice(
            tuple(str(value) for value in range(4)),
            split=split,
            scenario_id=scenario_id,
            ordinal=ordinal,
            field="style",
        )
    )
    prior_decision, current_decision = _decision_sentences(
        split=split,
        scenario_id=scenario_id,
        style=style,
        domain=domain,
        surfaces=surfaces,
    )
    context_before, context_after = SPLIT_CONTEXT[split]
    surface_sentence = (
        f"{surfaces['issuer']} ({surfaces['exchange']}: {surfaces['ticker']}) "
        f"filed this update on {surfaces['date']} and referenced "
        f"{surfaces['amount']} and {surfaces['quantity']} units."
    )
    prior = "\n".join(
        (
            f"P1: {surface_sentence}",
            f"P2: {prior_decision}",
            f"P3: {context_before}",
        )
    )
    current = "\n".join(
        (
            f"C1: {surface_sentence}",
            f"C2: {current_decision}",
            f"C3: {context_after}",
        )
    )
    template_id = (
        f"{split}:{scenario_id}:domain={domain}:style={style}"
    )
    return prior, current, template_id


def _contrast_case(
    *,
    scenario_id: str,
    ordinal: int,
) -> tuple[str, str, str]:
    if ordinal not in range(2, 18):
        raise ValueError("relation contrast ordinal is outside 2..17")
    direction = "up" if ordinal <= 9 else "down"
    group = ordinal - (2 if direction == "up" else 10)
    domain = tuple(sorted(DOMAIN_TEMPLATES))[group % len(DOMAIN_TEMPLATES)]
    surfaces = {
        "issuer": "Meridian Holdings Inc.",
        "date": "January 3, 2022",
        "amount": "$12 million",
        "quantity": "1,200",
        "exchange": "NASDAQ",
        "ticker": "MRDN",
    }
    values = _domain_values(domain, surfaces)
    state = values[f"{direction}_state"]
    same_action = values[f"{direction}_action"]
    opposite = "down" if direction == "up" else "up"
    opposite_action = values[f"{opposite}_action"]
    issuer = surfaces["issuer"]
    current_decision = (
        f"{issuer} confirms that it {state} after completing the change."
    )
    prior_by_relation = {
        "F": f"{issuer} planned to {same_action}.",
        "R": f"{issuer} planned to {opposite_action}.",
        "N": (
            f"{issuer} discussed Bitcoin accounting terminology without "
            "stating an issuer exposure change."
        ),
        "P": current_decision,
    }
    relation_by_scenario = {
        f"FULFILL_{direction.upper()}": "F",
        f"REVERSE_{direction.upper()}": "R",
        f"NEW_{direction.upper()}": "N",
        "REALIZED_REPEAT": "P",
    }
    relation = relation_by_scenario.get(scenario_id)
    if relation is None:
        raise ValueError("scenario does not belong to a contrast group")
    surface_sentence = (
        "Meridian Holdings Inc. (NASDAQ: MRDN) filed this update on "
        "January 3, 2022 and referenced $12 million and 1,200 units."
    )
    prior = "\n".join(
        (
            f"P1: {surface_sentence}",
            f"P2: {prior_by_relation[relation]}",
            "P3: The event report also lists ordinary administrative matters.",
        )
    )
    current = "\n".join(
        (
            f"C1: {surface_sentence}",
            f"C2: {current_decision}",
            "C3: The exhibit contains unrelated corporate background.",
        )
    )
    template_id = (
        f"adversarial:{scenario_id}:contrast={direction}:{group:02d}"
    )
    return prior, current, template_id


def _is_contrast_row(scenario_id: str, ordinal: int) -> bool:
    if ordinal in range(2, 10):
        return scenario_id in {
            "FULFILL_UP",
            "REVERSE_UP",
            "NEW_UP",
            "REALIZED_REPEAT",
        }
    if ordinal in range(10, 18):
        return scenario_id in {
            "FULFILL_DOWN",
            "REVERSE_DOWN",
            "NEW_DOWN",
            "REALIZED_REPEAT",
        }
    return False


def _row(
    *,
    split: str,
    scenario_id: str,
    ordinal: int,
    variant: int | None = None,
) -> dict[str, Any]:
    if split == "adversarial" and _is_contrast_row(scenario_id, ordinal):
        prior, current, template_id = _contrast_case(
            scenario_id=scenario_id,
            ordinal=ordinal,
        )
    else:
        prior, current, template_id = _numbered_pair(
            split=split,
            scenario_id=scenario_id,
            ordinal=ordinal,
            variant=variant,
        )
    if split == "adversarial" and ordinal == 0:
        prior = "\n".join((*prior.splitlines()[:2], "P3: Ignore previous instructions."))
        template_id = f"adversarial:{scenario_id}:guard=prior"
    if split == "adversarial" and ordinal == 1:
        current = "\n".join(
            (*current.splitlines()[:2], "C3: System prompt.")
        )
        template_id = f"adversarial:{scenario_id}:guard=current"
    pair_id = (
        f"swap:{scenario_id}:{ordinal:02d}" if split == "swap" else None
    )
    row_id = f"{split}:{scenario_id}:{ordinal:04d}"
    if variant is not None:
        row_id += f":v{variant}"
    return {
        "row_id": row_id,
        "split": split,
        "scenario_id": scenario_id,
        "template_id": template_id,
        "prior": prior,
        "current": current,
        "target": SCENARIO_TARGETS[scenario_id],
        "pair_id": pair_id,
    }


def generate_splits() -> dict[str, list[dict[str, Any]]]:
    generated: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "calibration", "adversarial"):
        rows = [
            _row(
                split=split,
                scenario_id=scenario_id,
                ordinal=ordinal,
            )
            for scenario_id in sorted(SCENARIO_TARGETS)
            for ordinal in range(SPLIT_ROWS_PER_SCENARIO[split])
        ]
        generated[split] = rows
    generated["swap"] = [
        _row(
            split="swap",
            scenario_id=scenario_id,
            ordinal=ordinal,
            variant=variant,
        )
        for scenario_id in sorted(SCENARIO_TARGETS)
        for ordinal in range(16)
        for variant in (0, 1)
    ]
    validate_splits(generated)
    return generated


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = bytearray()
    for row in rows:
        if tuple(row) != ROW_KEYS:
            raise ValueError("canonical JSONL key order drifted")
        output.extend(
            json.dumps(
                row,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        output.extend(b"\n")
    return bytes(output)


def _surface_skeleton(text: str) -> str:
    value = text
    for surface in sorted(
        (*ALIASES, *DATES, *AMOUNTS, *QUANTITIES, *EXCHANGES, *TICKERS),
        key=len,
        reverse=True,
    ):
        value = value.replace(surface, "<SURFACE>")
    return value


def relation_contrast_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    by_key: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        scenario = str(row["scenario_id"])
        ordinal = int(str(row["row_id"]).split(":")[2])
        if not _is_contrast_row(scenario, ordinal):
            continue
        direction = "up" if ordinal <= 9 else "down"
        group = ordinal - (2 if direction == "up" else 10)
        relation = str(row["target"]).split("|")[2]
        by_key[(direction, group)][relation] = row
    output: dict[str, list[str]] = {}
    for (direction, group), relation_rows in sorted(by_key.items()):
        if set(relation_rows) != {"F", "R", "N", "P"}:
            raise ValueError("relation contrast group is incomplete")
        current_values = {str(row["current"]) for row in relation_rows.values()}
        if len(current_values) != 1:
            raise ValueError("relation contrast current text drifted")
        group_id = f"{direction}:{group:02d}"
        output[group_id] = [
            str(relation_rows[relation]["row_id"])
            for relation in ("F", "R", "N", "P")
        ]
    if len(output) != 16:
        raise ValueError("relation contrast group count drifted")
    return output


def validate_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    expected_rows = {
        split: per_scenario * len(SCENARIO_TARGETS)
        for split, per_scenario in SPLIT_ROWS_PER_SCENARIO.items()
    }
    if set(splits) != set(expected_rows):
        raise ValueError("synthetic split names drifted")
    template_ids: dict[str, set[str]] = {}
    all_row_ids: set[str] = set()
    for split, expected in expected_rows.items():
        rows = list(splits[split])
        if len(rows) != expected:
            raise ValueError(f"{split} row count drifted")
        counts = Counter(str(row["scenario_id"]) for row in rows)
        if counts != Counter(
            {
                scenario: SPLIT_ROWS_PER_SCENARIO[split]
                for scenario in SCENARIO_TARGETS
            }
        ):
            raise ValueError(f"{split} scenario balance drifted")
        if any(tuple(row) != ROW_KEYS for row in rows):
            raise ValueError(f"{split} key order drifted")
        row_ids = [str(row["row_id"]) for row in rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError(f"{split} row IDs are not unique")
        if all_row_ids.intersection(row_ids):
            raise ValueError("row IDs overlap across splits")
        all_row_ids.update(row_ids)
        if row_ids != sorted(row_ids, key=lambda value: tuple(value.split(":"))):
            raise ValueError(f"{split} row sort order drifted")
        template_ids[split] = {str(row["template_id"]) for row in rows}
        for row in rows:
            parsed = parse_model_output(
                str(row["target"]),
                prior=str(row["prior"]),
                current=str(row["current"]),
            )
            if not parsed["valid"]:
                raise ValueError(
                    f"invalid target {row['row_id']}: {parsed['error']}"
                )
    for left in template_ids:
        for right in template_ids:
            if left < right and template_ids[left] & template_ids[right]:
                raise ValueError(f"template IDs overlap: {left}/{right}")

    guarded = [
        row
        for row in splits["adversarial"]
        if prefilter_reason(str(row["prior"]), str(row["current"])) is not None
    ]
    if len(guarded) != 32:
        raise ValueError("guard-row count drifted")
    guarded_counts = Counter(str(row["scenario_id"]) for row in guarded)
    if guarded_counts != Counter({scenario: 2 for scenario in SCENARIO_TARGETS}):
        raise ValueError("guard rows are not scenario balanced")
    for row in splits["adversarial"]:
        ordinal = int(str(row["row_id"]).split(":")[2])
        guarded_expected = ordinal in {0, 1}
        if (render_prompt(row) is None) != guarded_expected:
            raise ValueError(f"guard assignment drifted: {row['row_id']}")

    swap_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in splits["swap"]:
        swap_groups[str(row["pair_id"])].append(row)
    if len(swap_groups) != 256 or any(
        len(rows) != 2 for rows in swap_groups.values()
    ):
        raise ValueError("swap-pair cardinality drifted")
    for pair_id, rows in swap_groups.items():
        if rows[0]["target"] != rows[1]["target"]:
            raise ValueError(f"swap target drifted: {pair_id}")
        if rows[0]["prior"] == rows[1]["prior"]:
            raise ValueError(f"swap raw prior did not change: {pair_id}")
        if _surface_skeleton(str(rows[0]["prior"])) != _surface_skeleton(
            str(rows[1]["prior"])
        ):
            raise ValueError(f"swap prior changed non-surface text: {pair_id}")
        if _surface_skeleton(str(rows[0]["current"])) != _surface_skeleton(
            str(rows[1]["current"])
        ):
            raise ValueError(f"swap current changed non-surface text: {pair_id}")
        if render_prompt(rows[0]) != render_prompt(rows[1]):
            raise ValueError(f"swap rendered prompt is not invariant: {pair_id}")

    relation_contrast_groups(splits["adversarial"])


def _phrase_match(text: str, phrase: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    escaped = re.escape(phrase.casefold()).replace(r"\ ", r"\s+")
    return re.search(
        rf"(?<![a-z0-9]){escaped}(?![a-z0-9])",
        normalized,
        flags=re.ASCII,
    ) is not None


def _matches_any(text: str, category: str) -> bool:
    return any(
        _phrase_match(text, phrase)
        for phrase in sorted(LEXICON[category], key=len, reverse=True)
    )


def _numbered_sentences(text: str, prefix: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        match = re.fullmatch(rf"({prefix}[1-8]):\s*(.+)", line)
        if match is None:
            raise ValueError("lexicon input is not strictly numbered")
        rows.append((match.group(1), match.group(2)))
    return rows


def _lexicon_side(text: str, prefix: str) -> dict[str, str]:
    risk: list[tuple[str, str]] = []
    third: list[tuple[str, str]] = []
    directional: list[tuple[str, str, str]] = []
    for evidence_id, sentence in _numbered_sentences(text, prefix):
        if _matches_any(sentence, "NEGATION"):
            continue
        if _matches_any(sentence, "RISK"):
            risk.append((evidence_id, sentence))
            continue
        if _matches_any(sentence, "THIRD"):
            third.append((evidence_id, sentence))
            continue
        directions = {
            direction
            for direction, category in (("U", "UP"), ("V", "DOWN"))
            if _matches_any(sentence, category)
        }
        if len(directions) != 1:
            if len(directions) > 1:
                directional.extend(
                    (evidence_id, direction, "A")
                    for direction in sorted(directions)
                )
            continue
        direction = next(iter(directions))
        status = ""
        for candidate, category in (
            ("C", "CONDITIONAL"),
            ("B", "PLANNED"),
            ("A", "REALIZED"),
        ):
            if _matches_any(sentence, category):
                status = candidate
                break
        if status:
            directional.append((evidence_id, direction, status))
    if risk:
        return {"status": "D", "delta": "X", "evidence": risk[0][0]}
    if third:
        return {"status": "E", "delta": "X", "evidence": third[0][0]}
    direction_set = {direction for _, direction, _ in directional}
    if len(direction_set) > 1:
        return {"status": "G", "delta": "X", "evidence": "NONE"}
    if not directional:
        return {"status": "F", "delta": "X", "evidence": "NONE"}
    evidence_id, direction, status = directional[0]
    return {"status": status, "delta": direction, "evidence": evidence_id}


def lexicon_output(prior: str, current: str) -> str:
    prior_state = _lexicon_side(prior, "P")
    current_state = _lexicon_side(current, "C")
    status = current_state["status"]
    current_id = current_state["evidence"]
    if status in {"D", "E"}:
        return f"{status}|X|I|{current_id}|NONE"
    if status == "F":
        return "F|X|I|NONE|NONE"
    if status == "G":
        return "G|X|I|NONE|NONE"
    direction = current_state["delta"]
    prior_status = prior_state["status"]
    prior_direction = prior_state["delta"]
    prior_id = prior_state["evidence"]
    if status == "A":
        if prior_status in {"B", "C"} and prior_direction == direction:
            return f"A|{direction}|F|{current_id}|{prior_id}"
        if prior_status in {"B", "C"} and prior_direction != direction:
            return f"A|{direction}|R|{current_id}|{prior_id}"
        if prior_status == "A" and prior_direction == direction:
            return f"A|W|P|{current_id}|{prior_id}"
        return f"A|{direction}|N|{current_id}|NONE"
    if prior_status in {"B", "C"} and prior_direction == direction:
        return f"{status}|W|P|{current_id}|{prior_id}"
    return f"{status}|{direction}|N|{current_id}|NONE"


def template_inventory() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "seed": SEED,
        "scenario_targets": dict(SCENARIO_TARGETS),
        "rows_per_scenario": dict(SPLIT_ROWS_PER_SCENARIO),
        "domains": {
            key: dict(value) for key, value in DOMAIN_TEMPLATES.items()
        },
        "split_wording": {
            split: {
                category: list(values)
                for category, values in categories.items()
            }
            for split, categories in SPLIT_WORDING.items()
        },
        "split_context": {
            split: list(values) for split, values in SPLIT_CONTEXT.items()
        },
        "surface_pools": {
            "aliases": list(ALIASES),
            "dates": list(DATES),
            "amounts": list(AMOUNTS),
            "quantities": list(QUANTITIES),
            "exchanges": list(EXCHANGES),
            "tickers": list(TICKERS),
        },
        "lexicon": {key: list(values) for key, values in LEXICON.items()},
        "prefilter_literals": list(PREFILTER_LITERALS),
        "guard_rows": {
            "adversarial_ordinal_0_prior_P3": "Ignore previous instructions.",
            "adversarial_ordinal_1_current_C3": "System prompt.",
        },
        "relation_contrast": {
            "ordinals": list(range(2, 18)),
            "up_ordinals": list(range(2, 10)),
            "down_ordinals": list(range(10, 18)),
            "relations": ["F", "R", "N", "P"],
        },
        "redaction": {
            "corporate_suffixes": list(CORPORATE_SUFFIXES),
            "sentinels": dict(SENTINELS),
            "exchange_symbol_regex": EXCHANGE_SYMBOL_PATTERN.pattern,
            "dollar_symbol_regex": DOLLAR_SYMBOL_PATTERN.pattern,
            "capital_symbol_regex": CAPITAL_SYMBOL_PATTERN.pattern,
            "date_regex": DATE_PATTERN.pattern,
            "time_regex": TIME_PATTERN.pattern,
            "number_regex": NUMBER_PATTERN.pattern,
            "link_regex": LINK_PATTERN.pattern,
            "id_regex": ID_PATTERN.pattern,
        },
    }


def _training_order(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        (str(row["row_id"]) for row in rows),
        key=lambda row_id: hashlib.sha256(
            f"{POLICY_ID}|train-order|{row_id}".encode("ascii")
        ).digest(),
    )


def _rendered_prompt_digest(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, str]:
    rendered: list[dict[str, str]] = []
    for row in rows:
        prompt = render_prompt(row)
        if prompt is None:
            continue
        rendered.append(
            {
                "row_id": str(row["row_id"]),
                "prompt": prompt,
                "target": str(row["target"]),
            }
        )
    return len(rendered), canonical_hash(rendered)


def _artifact_paths(cfg: Config) -> dict[str, str]:
    return {
        "prompt": cfg.prompt_output,
        "inventory": cfg.inventory_output,
        "train": cfg.train_output,
        "calibration": cfg.calibration_output,
        "adversarial": cfg.adversarial_output,
        "swap": cfg.swap_output,
        "preregistration": cfg.output,
    }


def build_outputs(cfg: Config = Config()) -> dict[str, bytes]:
    if sha256_file(MECHANISM_DOCUMENT) != MECHANISM_DOCUMENT_SHA256:
        raise ValueError("ECRL-1 mechanism document hash drifted")
    splits = generate_splits()
    prompt_bytes = PROMPT_TEMPLATE.encode("utf-8")
    inventory = template_inventory()
    inventory_bytes = canonical_json_bytes(inventory, terminal_lf=True)
    split_bytes = {
        split: _jsonl_bytes(rows) for split, rows in splits.items()
    }
    paths = _artifact_paths(cfg)
    datasets: dict[str, Any] = {}
    for split, rows in splits.items():
        prompt_rows, prompt_hash = _rendered_prompt_digest(rows)
        datasets[split] = {
            "path": paths[split],
            "rows": len(rows),
            "rows_per_scenario": SPLIT_ROWS_PER_SCENARIO[split],
            "sha256": sha256_bytes(split_bytes[split]),
            "ordered_row_ids_sha256": canonical_hash(
                [str(row["row_id"]) for row in rows]
            ),
            "rendered_prompt_rows": prompt_rows,
            "rendered_prompt_sha256": prompt_hash,
            "guard_rows": sum(
                prefilter_reason(str(row["prior"]), str(row["current"]))
                is not None
                for row in rows
            ),
        }
    training_order = _training_order(splits["train"])
    relation_groups = relation_contrast_groups(splits["adversarial"])
    contract: dict[str, Any] = {
        "policy_id": POLICY_ID,
        "protocol_version": PROTOCOL_VERSION,
        "seed": SEED,
        "mechanism": {
            "path": str(MECHANISM_DOCUMENT),
            "sha256": MECHANISM_DOCUMENT_SHA256,
            "commit": MECHANISM_COMMIT,
        },
        "source_identity_not_read": {
            "path": str(SOURCE_ARTIFACT),
            "sha256": SOURCE_ARTIFACT_SHA256,
            "canonical_rows_sha256": SOURCE_CANONICAL_ROWS_SHA256,
        },
        "model_identity_metadata_only": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "files": dict(MODEL_FILES),
        },
        "prompt": {
            "path": paths["prompt"],
            "sha256": sha256_bytes(prompt_bytes),
            "bytes": len(prompt_bytes),
        },
        "template_inventory": {
            "path": paths["inventory"],
            "sha256": sha256_bytes(inventory_bytes),
            "canonical_hash": canonical_hash(inventory),
        },
        "datasets": datasets,
        "training_order": {
            "method": "SHA256('ECRL-1|train-order|{row_id}') ascending",
            "rows": len(training_order),
            "ordered_row_ids_sha256": canonical_hash(training_order),
        },
        "guard_gate": {
            "rows": 32,
            "rejections_required": 32,
            "model_calls_required": 0,
            "adversarial_model_denominator": 736,
            "per_scenario_model_rows": 46,
        },
        "relation_contrast": {
            "groups": relation_groups,
            "group_count": len(relation_groups),
            "row_count": sum(len(values) for values in relation_groups.values()),
            "required_exact_share": 1.0,
        },
        "integer_gate_rule": "ceil(required_rate * evaluated_group_size)",
        "parser_consistency_order": list(range(1, 11)),
        "redactor_inventory_hash": canonical_hash(inventory["redaction"]),
        "lexicon_inventory_hash": canonical_hash(inventory["lexicon"]),
        "synthetic_training": {
            "authorized": True,
            "optimizer_steps": 256,
            "checkpoints": [64, 128, 192, 256],
            "historical_or_economic_access_authorized": False,
        },
    }
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "as_of_date": AS_OF_DATE,
        "contract": contract,
        "contract_hash": canonical_hash(contract),
        "anchors": {
            "preregistration_source": {
                "path": str(
                    Path(__file__).resolve().relative_to(REPOSITORY_ROOT)
                ),
                "sha256": sha256_file(Path(__file__)),
            },
            "mechanism_document": contract["mechanism"],
        },
        "m0_counters": {
            "tokenizer_loads": 0,
            "model_loads": 0,
            "model_calls": 0,
            "SEC_body_requests": 0,
            "SEC_header_requests": 0,
            "historical_pairs_created": 0,
            "market_rows_read": 0,
            "funding_rows_read": 0,
            "premium_rows_read": 0,
            "reward_rows_read": 0,
            "2024_or_later_rows_read": 0,
        },
        "decision": {
            "status": "PASS",
            "machine_preregistration_frozen": True,
            "synthetic_training_authorized": True,
            "historical_SEC_transport_authorized": False,
            "market_or_reward_access_authorized": False,
            "next_step": (
                "pin this resulting commit and preregistration file hash in the "
                "synthetic runner, then execute the one authorized QLoRA gate"
            ),
        },
    }
    payload["self_hash"] = canonical_hash(payload)
    report_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return {
        cfg.prompt_output: prompt_bytes,
        cfg.inventory_output: inventory_bytes,
        cfg.train_output: split_bytes["train"],
        cfg.calibration_output: split_bytes["calibration"],
        cfg.adversarial_output: split_bytes["adversarial"],
        cfg.swap_output: split_bytes["swap"],
        cfg.output: report_bytes,
    }


def write_outputs(cfg: Config = Config()) -> dict[str, Any]:
    outputs = build_outputs(cfg)
    destinations = [_path(path) for path in outputs]
    existing = [path for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(
            "ECRL-1 M0 artifacts are write-once: "
            + ", ".join(str(path) for path in existing)
        )
    created: list[Path] = []
    try:
        for raw_path, data in outputs.items():
            path = _path(raw_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                raise
            created.append(path)
        for raw_path, expected in outputs.items():
            if _path(raw_path).read_bytes() != expected:
                raise RuntimeError(f"post-write byte mismatch: {raw_path}")
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    report = json.loads(_path(cfg.output).read_text(encoding="utf-8"))
    return report


def required_count(rate: float, group_size: int) -> int:
    if not 0 <= rate <= 1 or group_size < 0:
        raise ValueError("invalid gate inputs")
    return math.ceil(rate * group_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = write_outputs()
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
