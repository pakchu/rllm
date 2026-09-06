"""Freeze BFWC-288 without decoding CSV rows or deriving incidence/outcomes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import errno
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Iterable, Mapping


POLICY_ID = "BFWC-288"
PROTOCOL_VERSION = "blockspace_fee_witness_concordance_preregistration_v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(
    "training/preregister_blockspace_fee_witness_concordance.py"
)
DOCUMENT_PATH = Path(
    "docs/blockspace-fee-witness-concordance-"
    "preregistration-2026-07-30.md"
)
DEFAULT_OUTPUT = Path(
    "results/blockspace_fee_witness_concordance_"
    "preregistration_2026-07-30.json"
)

BFRT_SOURCE_MANIFEST = Path(
    "results/mempool_block_feerates_source_manifest_2026-07-20.json"
)
WCTR_SOURCE_MANIFEST = Path(
    "results/mempool_witness_composition_source_manifest_2026-07-20.json"
)
BFRT_SOURCE_MANIFEST_HASH = (
    "fe616bcf294e8b3b2abc6dec124e922f77df4bca47a86249fc270f2af6b46f21"
)
WCTR_SOURCE_MANIFEST_HASH = (
    "55914b3ec31fe8fb66d8a8dc31acb3784a10b256625073a5aeff1d317660ea8d"
)

BFRT_NORMALIZED = Path(
    "data/mempool_block_feerates_3y_2026-07-20.csv.gz"
)
WCTR_NORMALIZED = Path(
    "data/mempool_witness_composition_4y_2026-07-20.csv.gz"
)
BFRT_PRIMARY_CLOCK = Path(
    "results/block_feerate_breadth_transport_primary_clock_2026-07-20.csv.gz"
)
WCTR_PRIMARY_CLOCK = Path(
    "results/witness_composition_transport_primary_clock_2026-07-20.csv.gz"
)
MARKET_DATA = Path(
    "data/cache_market_ext_5m_wavefull_2020-01-01_2026-06-01.csv.gz"
)
FUNDING_DATA = Path(
    "data/binance_um_aux_btc_2020_2026/"
    "BTCUSDT_funding_2020-01-01_2026-06-01.csv.gz"
)
PREMIUM_DATA = Path(
    "data/binance_um_aux_btc_2020_2026/"
    "BTCUSDT_premium_1h_2020-01-01_2026-06-01.csv.gz"
)

GROSS9_WEIGHTS = {
    "cand_rex_veto_7": 1.6,
    "fresh_kimchi_fx": 2.0,
    "frozen_annual_rank7": 3.0,
    "markov_transition_long": 2.0,
    "rex_taker_low_range_position": 0.4,
}

FROZEN_DEPENDENCIES: dict[str, str] = {
    str(DOCUMENT_PATH): (
        "2ac4189d89153db7846659e117036ac0744796acf60dce43bf033454cb34b04c"
    ),
    str(BFRT_SOURCE_MANIFEST): (
        "1ad4ee8bc9e81d3f7e7169426de21f0398bc6ab1d739e6f27e6d9ff02f331555"
    ),
    "data/mempool_block_feerates_3y_2026-07-20.raw.json.gz": (
        "4309dfbbdb08b89cd9cc92a341bd6186146b1e67adc2c3f926c8154ddabc4898"
    ),
    str(BFRT_NORMALIZED): (
        "007d13ba756fd29faae1ae87caa11554438b54bb5028f24b2f0c21ddf3a0e55d"
    ),
    "docs/block-feerate-breadth-transport-mechanism-decision-2026-07-20.md": (
        "f6dd7d52b03d1370483a1157e24efad45b6886230bafed0931b0ac88cbde82cb"
    ),
    "docs/block-feerate-breadth-transport-source-freeze-2026-07-20.md": (
        "0336a5d567b894964ad3aaca541548a7962e16b6a79a34dcda4bb5478e2f3092"
    ),
    "docs/block-feerate-breadth-transport-preregistration-2026-07-20.md": (
        "04b77d6358aead4ecae494f75c71e1c0f40066fc466df01e4a7b9ede4c7fab2c"
    ),
    "training/download_mempool_block_feerate_history.py": (
        "ebd30dd109a92c4dc5a2a6a444a5d5760fa4360c7fd848b02923f0670e4a2910"
    ),
    "training/preregister_block_feerate_breadth_transport.py": (
        "2cc282d40e6791b0e4a5b5c1fb6f5081335eceb9a144db17faf157a8965deb5f"
    ),
    "training/build_block_feerate_breadth_transport_support.py": (
        "de1ba0ac1424579b5869e8cd09986fca4383a95b3cfdffc0ef3694ecbd19ef1d"
    ),
    "results/block_feerate_breadth_transport_preregistration_2026-07-20.json": (
        "73b06b94db2f844d993dcd76d6ad9e60f9b6d332d4c4d6ba8d41a3de970dae42"
    ),
    "results/block_feerate_breadth_transport_support_2026-07-20.json": (
        "b980a0d76bd9a3084410d40e9fbf920acf1e2065bce174aa48823f41052f3bd8"
    ),
    str(BFRT_PRIMARY_CLOCK): (
        "33428d29c2ace9b23672b2dc9dc3e9ba0e3020fa1a6e3845d55fa5d75230d64a"
    ),
    "results/block_feerate_breadth_transport_control_clocks_2026-07-20.csv.gz": (
        "d85197948f9418f4ab50e88825638fe33629f2db808fd1e572f8ed4d685c5a92"
    ),
    str(WCTR_SOURCE_MANIFEST): (
        "2506429ebcbf9b2ada6c745bcc58bd9ec3b0fdbe245f726a408ff99bd3111342"
    ),
    "data/mempool_witness_composition_4y_2026-07-20.raw.json.gz": (
        "ddd3615294d501ed3b24c5d43e2fc16319bd87f1add3c14dde2362c4b789c4c1"
    ),
    str(WCTR_NORMALIZED): (
        "ee761e813085dfdee675ca9d420516f814c4c2824f3f5cef604acc3871d46c61"
    ),
    "docs/witness-composition-transport-mechanism-decision-2026-07-20.md": (
        "101e84303efb3146dae587048c179b6688c93b50bb4c99edec8ba8daab72bc98"
    ),
    "docs/witness-composition-transport-source-freeze-2026-07-20.md": (
        "6697872b690dc38c8bfe4700d75fb3e7fce8fbab507794ff4200aa0b1ff410aa"
    ),
    "docs/witness-composition-transport-preregistration-2026-07-20.md": (
        "23b94a5c7dcf8af16e31ff6c1e62483b74a0ffeb7457eb6884f2412bf9cb4a96"
    ),
    "training/download_mempool_witness_composition_history.py": (
        "d5cd3f2cab5e501d5484539f1ea3c5aac5a96916dd65aa1060bb561fa639d721"
    ),
    "training/preregister_witness_composition_transport.py": (
        "58a3725d8e64b47181a8e9310c370c583084fa27ea9c633f6e6c41dae266bf20"
    ),
    "training/build_witness_composition_transport_support.py": (
        "77a585848311c2323f8b45b172305f257a5734c7a68db0e78256bc0e3159e6c9"
    ),
    "results/witness_composition_transport_preregistration_2026-07-20.json": (
        "f1d8d5641f1773d00dc2a99a4ca7e11b68cbc5b0cebc1456514fcbbcd9c9d3d1"
    ),
    "results/witness_composition_transport_support_2026-07-20.json": (
        "35e3c4623be670d690b914f96e6a40d8e314e3d14554d3024d1f201fcf8ffb30"
    ),
    str(WCTR_PRIMARY_CLOCK): (
        "7a6b56a3024d0d087322fad7b3229276c539b93374691cd2812af0630dc752b1"
    ),
    "results/witness_composition_transport_control_clocks_2026-07-20.csv.gz": (
        "ca5ef092d30ed9135429d8d2b830546681e289f0798ab50ef60c85ed5fd9a1f7"
    ),
    "docs/block-clearing-relational-topology-boundary-2026-07-24.md": (
        "ab3d71d5b7f52254a3d25f4eeada35acab16ec7a2460528e2727b56ae8039560"
    ),
    "docs/block-clearing-relational-topology-mechanism-decision-2026-07-24.md": (
        "95d3ebffea956fbaee261d8da280d8ced8f12164823a9d86755771d0fb98b991"
    ),
    "training/preregister_block_clearing_relational_topology.py": (
        "e04fc7d16f550bf2c0cdde9a3359f079b2f233ede3dba315b41285f50e326e2b"
    ),
    "training/build_block_clearing_relational_topology_support.py": (
        "8a351be18a2f9b44a2ae8bdb48e5555e84393b704ff91b4c36801266e49f6a5e"
    ),
    "results/block_clearing_relational_topology_preregistration_2026-07-24.json": (
        "322f91b41fce1aee06250a010d5a569557b83cc3f493ee3c47f5d6974aafe6a8"
    ),
    "results/block_clearing_relational_topology_support_2026-07-24.json": (
        "9ccccf7a3176fcf86baddacb65c11bbde78ea73ed7ab18d3594b0e6327567055"
    ),
    "configs/shadow/portfolio_rank7_capacity_candidate_2026-07-28.json": (
        "006f82e1f0affad9f96a08a6c600542feec4a0e1198ed99b8630627de4913450"
    ),
    "results/gross9_pre2025_authoritative_anchor_2026-07-28.json": (
        "329878d90b6cd9c731eb4871ac041256f95f03c14dd261ada681d3a370709875"
    ),
    "training/audit_gross9_fixed_candidate_state_substitution.py": (
        "b727d3fb0c45801e839265e0595be457a7fb204b8f64b92247cb82d50e49f4c2"
    ),
    "training/portfolio_opt_added_alpha_update.py": (
        "0bbe02d60ed7393e704a72c65d92d2a952eafee0d5a3d32310a782b80e2ae901"
    ),
    "training/portfolio_opt_rank7_capacity_update.py": (
        "41df5d0bb712da09f70e56d39efb2e0344410665727aead24ff9f28d754b57e8"
    ),
    str(MARKET_DATA): (
        "a77cd0ae5b88b3c95e509d8d2610773d34af3afdc9170c63d88564bc3d0b990c"
    ),
    str(FUNDING_DATA): (
        "4d381be086e275bacaf31df431dc31307a71a26b3947b7082efffc10bb129dd7"
    ),
    str(PREMIUM_DATA): (
        "b45fcc5a3cf75c8e594effe61a698c4652f841b1d304107e9669524e0fc9d0d7"
    ),
}

EXACT_CSV_HEADERS: dict[str, str] = {
    str(BFRT_NORMALIZED): (
        "bucket_start_utc,bucket_end_utc,available_at_utc,avg_height,"
        "avg_timestamp,fee_p0,fee_p10,fee_p25,fee_p50,fee_p75,fee_p90,"
        "fee_p100\n"
    ),
    str(WCTR_NORMALIZED): (
        "bucket_start_utc,bucket_end_utc,available_at_utc,avg_height,"
        "avg_timestamp,avg_size,avg_weight\n"
    ),
    str(BFRT_PRIMARY_CLOCK): (
        "policy_id,clock,window,bucket_start_utc,source_available_at_utc,"
        "entry_time_utc,exit_time_utc,side,location,signed_coherence,"
        "coherence,tail_divergence,magnitude_rank,tail_divergence_rank\n"
    ),
    "results/block_feerate_breadth_transport_control_clocks_2026-07-20.csv.gz": (
        "policy_id,clock,window,bucket_start_utc,source_available_at_utc,"
        "entry_time_utc,exit_time_utc,side,location,signed_coherence,"
        "coherence,tail_divergence,magnitude_rank,tail_divergence_rank\n"
    ),
    str(WCTR_PRIMARY_CLOCK): (
        "policy_id,clock,window,bucket_start_utc,source_available_at_utc,"
        "entry_time_utc,exit_time_utc,side,witness_share,fullness,"
        "transport_7d,impulse_24h,log_size_7d,log_size_24h,log_weight_7d,"
        "log_weight_24h,magnitude_rank,fullness_rank,impulse_magnitude_rank,"
        "log_size_magnitude_rank,log_weight_magnitude_rank\n"
    ),
    "results/witness_composition_transport_control_clocks_2026-07-20.csv.gz": (
        "policy_id,clock,window,bucket_start_utc,source_available_at_utc,"
        "entry_time_utc,exit_time_utc,side,witness_share,fullness,"
        "transport_7d,impulse_24h,log_size_7d,log_size_24h,log_weight_7d,"
        "log_weight_24h,magnitude_rank,fullness_rank,impulse_magnitude_rank,"
        "log_size_magnitude_rank,log_weight_magnitude_rank\n"
    ),
    str(MARKET_DATA): (
        "date,open,high,low,close,volume,quote_asset_volume,number_of_trades,"
        "taker_buy_base,taker_buy_quote,tic,day,dxy,kimchi_premium,usdkrw,"
        "btckrw,dxy_available,kimchi_available,usdkrw_available,"
        "external_any_available,dxy_zscore,dxy_momentum,"
        "kimchi_premium_zscore,kimchi_premium_change,usdkrw_zscore,"
        "usdkrw_momentum\n"
    ),
    str(FUNDING_DATA): "date,symbol,funding_rate,funding_time,mark_price\n",
    str(PREMIUM_DATA): "date,symbol,open,high,low,close,close_time\n",
}


@dataclass(frozen=True)
class Policy:
    policy_id: str = POLICY_ID
    rank_history_max: int = 180
    rank_history_min: int = 120
    rank_threshold: float = 0.75
    source_bucket_seconds: int = 43_200
    hold_bars_5m: int = 288
    hold_seconds: int = 86_400
    leverage: float = 0.5
    base_cost_bp_per_notional_side: float = 6.0
    stress_cost_bp_per_notional_side: float = 10.0
    signflip_draws: int = 20_000


def _repository_path(path: str | Path) -> Path:
    candidate = Path(path)
    raw = str(path)
    if (
        candidate.is_absolute()
        or raw.startswith("~")
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("BFWC-288 path must be repository-relative")
    lexical = REPOSITORY_ROOT / candidate
    try:
        lexical.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise RuntimeError("BFWC-288 path escaped repository") from error
    return lexical


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _repository_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"BFWC-288 duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_unique_json(path: str | Path) -> dict[str, Any]:
    with _repository_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_unique_object)
    if not isinstance(payload, dict):
        raise RuntimeError("BFWC-288 source manifest must be an object")
    return payload


def csv_header_bytes(path: str | Path) -> bytes:
    source = _repository_path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rb") as handle:
        header = handle.readline(65_537)
    if (
        not header
        or len(header) > 65_536
        or not header.endswith(b"\n")
        or b"\n" in header[:-1]
    ):
        raise RuntimeError("BFWC-288 CSV header is not one bounded line")
    try:
        header.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("BFWC-288 CSV header is not UTF-8") from error
    return header


def sha256_csv_header(path: str | Path) -> str:
    return hashlib.sha256(csv_header_bytes(path)).hexdigest()


def exact_midrank(current: float, prior: Iterable[float]) -> float:
    """Reference formula for tests/evaluator; current is never added to prior."""

    values = list(prior)
    if not values:
        raise ValueError("BFWC-288 midrank requires prior values")
    if not math.isfinite(current) or any(
        not math.isfinite(value) for value in values
    ):
        raise ValueError("BFWC-288 midrank requires finite values")
    lower = sum(value < current for value in values)
    equal = sum(value == current for value in values)
    return (lower + 0.5 * equal) / len(values)


def ceil_5m_plus_one_bar(epoch_seconds: int) -> int:
    if isinstance(epoch_seconds, bool) or not isinstance(epoch_seconds, int):
        raise TypeError("BFWC-288 epoch seconds must be an integer")
    return ((epoch_seconds + 299) // 300) * 300 + 300


def deterministic_random_side(signal_id: str) -> str:
    token = f"{POLICY_ID}|{signal_id}|RANDOM_SIDE".encode("utf-8")
    return "LONG" if hashlib.sha256(token).digest()[0] < 128 else "SHORT"


def _validate_source_manifest(path: Path, expected_hash: str) -> None:
    payload = _load_unique_json(path)
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError(
            f"BFWC-288 source manifest internal hash changed: {path}"
        )
    if payload["manifest_hash"] != expected_hash:
        raise RuntimeError(
            f"BFWC-288 source manifest identity changed: {path}"
        )


def validate_frozen_dependencies() -> None:
    for path, expected in FROZEN_DEPENDENCIES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"BFWC-288 frozen dependency changed: {path}: "
                f"{actual} != {expected}"
            )
    for path, expected_text in EXACT_CSV_HEADERS.items():
        actual = csv_header_bytes(path)
        expected = expected_text.encode("utf-8")
        if actual != expected:
            raise RuntimeError(f"BFWC-288 exact CSV header changed: {path}")
    _validate_source_manifest(
        BFRT_SOURCE_MANIFEST,
        BFRT_SOURCE_MANIFEST_HASH,
    )
    _validate_source_manifest(
        WCTR_SOURCE_MANIFEST,
        WCTR_SOURCE_MANIFEST_HASH,
    )


def _artifact(path: str | Path) -> dict[str, Any]:
    text = str(path)
    result: dict[str, Any] = {
        "path": text,
        "sha256": FROZEN_DEPENDENCIES[text],
    }
    if text in EXACT_CSV_HEADERS:
        result["header"] = EXACT_CSV_HEADERS[text].removesuffix("\n").split(",")
        result["header_sha256"] = hashlib.sha256(
            EXACT_CSV_HEADERS[text].encode("utf-8")
        ).hexdigest()
    return result


def _source_contracts() -> dict[str, Any]:
    return {
        "BFRT": {
            "source_manifest": {
                **_artifact(BFRT_SOURCE_MANIFEST),
                "internal_manifest_hash": BFRT_SOURCE_MANIFEST_HASH,
            },
            "raw": _artifact(
                "data/mempool_block_feerates_3y_2026-07-20.raw.json.gz"
            ),
            "normalized": _artifact(BFRT_NORMALIZED),
            "documents": [
                _artifact(
                    "docs/block-feerate-breadth-transport-"
                    "mechanism-decision-2026-07-20.md"
                ),
                _artifact(
                    "docs/block-feerate-breadth-transport-"
                    "source-freeze-2026-07-20.md"
                ),
                _artifact(
                    "docs/block-feerate-breadth-transport-"
                    "preregistration-2026-07-20.md"
                ),
            ],
            "builders": [
                _artifact("training/download_mempool_block_feerate_history.py"),
                _artifact(
                    "training/preregister_block_feerate_breadth_transport.py"
                ),
                _artifact(
                    "training/build_block_feerate_breadth_transport_support.py"
                ),
            ],
            "prior_support": [
                _artifact(
                    "results/block_feerate_breadth_transport_"
                    "preregistration_2026-07-20.json"
                ),
                _artifact(
                    "results/block_feerate_breadth_transport_"
                    "support_2026-07-20.json"
                ),
                _artifact(BFRT_PRIMARY_CLOCK),
                _artifact(
                    "results/block_feerate_breadth_transport_"
                    "control_clocks_2026-07-20.csv.gz"
                ),
            ],
        },
        "WCTR": {
            "source_manifest": {
                **_artifact(WCTR_SOURCE_MANIFEST),
                "internal_manifest_hash": WCTR_SOURCE_MANIFEST_HASH,
            },
            "raw": _artifact(
                "data/mempool_witness_composition_4y_2026-07-20.raw.json.gz"
            ),
            "normalized": _artifact(WCTR_NORMALIZED),
            "documents": [
                _artifact(
                    "docs/witness-composition-transport-"
                    "mechanism-decision-2026-07-20.md"
                ),
                _artifact(
                    "docs/witness-composition-transport-"
                    "source-freeze-2026-07-20.md"
                ),
                _artifact(
                    "docs/witness-composition-transport-"
                    "preregistration-2026-07-20.md"
                ),
            ],
            "builders": [
                _artifact(
                    "training/download_mempool_witness_composition_history.py"
                ),
                _artifact(
                    "training/preregister_witness_composition_transport.py"
                ),
                _artifact(
                    "training/build_witness_composition_transport_support.py"
                ),
            ],
            "prior_support": [
                _artifact(
                    "results/witness_composition_transport_"
                    "preregistration_2026-07-20.json"
                ),
                _artifact(
                    "results/witness_composition_transport_"
                    "support_2026-07-20.json"
                ),
                _artifact(WCTR_PRIMARY_CLOCK),
                _artifact(
                    "results/witness_composition_transport_"
                    "control_clocks_2026-07-20.csv.gz"
                ),
            ],
        },
    }


def _core_manifest() -> dict[str, Any]:
    policy = Policy()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "policy": asdict(policy),
        "singleton": True,
        "status": "outcome_blind_write_once_preincidence",
        "frozen_preregistration": {
            "document": _artifact(DOCUMENT_PATH),
            "executable": {
                "path": str(SCRIPT_PATH),
                "self_hash_excluded_to_avoid_circular_identity": True,
            },
            "serialization": {
                "encoding": "UTF-8",
                "sort_keys": True,
                "indent": 2,
                "ensure_ascii": True,
                "allow_nan": False,
                "trailing_lf_count": 1,
                "manifest_hash": (
                    "SHA256 compact sorted-key JSON of all top-level fields "
                    "except manifest_hash"
                ),
            },
        },
        "source_contracts": _source_contracts(),
        "evidence_disclosure": {
            "source_values_previously_seen": True,
            "BFRT_individual_incidence_previously_seen": True,
            "WCTR_individual_incidence_previously_seen": True,
            "BCRT_broad_relational_family_previously_seen": True,
            "pristine_source_claim": False,
            "candidate_exact_primitives_opened": False,
            "candidate_exact_incidence_opened": False,
            "candidate_overlap_opened": False,
            "candidate_outcomes_opened": False,
            "csv_data_rows_decoded_by_preregistration": 0,
            "csv_headers_inspected": len(EXACT_CSV_HEADERS),
            "source_manifests_parsed": 2,
            "network_calls": 0,
            "subprocess_calls": 0,
        },
        "join_and_feature": {
            "join": {
                "type": "exact inner one-to-one",
                "keys": ["bucket_start_utc", "bucket_end_utc"],
                "required_consecutive_rows": ["t", "t-1", "t-2"],
                "bucket_seconds": policy.source_bucket_seconds,
                "common_domain": (
                    "later first key through earlier last key, intersect full "
                    "horizon"
                ),
                "outer_join_gaps_required": 0,
                "nearest_tolerance_imputation_or_fill": False,
            },
            "arithmetic": "IEEE-754 binary64; no pre-comparison rounding",
            "x": "x[p,t]=log1p(fee_p[t]), p={10,25,75,90}",
            "delta2": "Delta2 z[t]=z[t]-z[t-2]",
            "R": (
                "0.5*((Delta2 x[10]+Delta2 x[25])-"
                "(Delta2 x[75]+Delta2 x[90]))"
            ),
            "witness_share": (
                "(4*avg_size-avg_weight)/(3*avg_size)"
            ),
            "W": "Delta2 witness_share",
            "fullness": "avg_weight/4_000_000",
            "U": "Delta2 fullness",
            "base_valid": (
                "exact join plus consecutive t,t-1,t-2 and finite defined "
                "R,W,U with primitive domain checks"
            ),
            "primitive_domains": {
                "fee_percentiles": "finite and >=0",
                "avg_size": "finite and >0",
                "avg_weight": "finite and in [0,4000000]",
                "witness_share": "finite and in [0,1]",
            },
        },
        "rank_and_signal": {
            "history": (
                "latest 180 strictly prior base-valid joint rows; current "
                "excluded"
            ),
            "minimum_prior_rows": policy.rank_history_min,
            "ties": "exact binary64 equality; no tie breaker",
            "L": "count(prior abs(R) < current abs(R))",
            "E": "count(prior abs(R) == current abs(R))",
            "midrank": "(L+0.5*E)/n",
            "eligibility": [
                "midrank(abs(R))>=0.75",
                "R!=0 exactly",
                "W!=0 exactly",
                "U!=0 exactly",
                "sign(R)=sign(W)=sign(U)",
            ],
            "side": {"positive_R": "LONG", "negative_R": "SHORT"},
            "economic_interpretation": (
                "R>0 means lower fee percentiles rose relative to upper fee "
                "percentiles over two buckets (flattening/broadening); W>0 "
                "means witness share rose; U>0 means fullness rose; positive "
                "concordance is LONG and the exact negative mirror is SHORT; "
                "polarity is fixed and is not a causal claim"
            ),
            "canonical_signal_id": (
                "sha256_utf8(BFWC-288|bucket_start_utc|bucket_end_utc|"
                "joint_available_at_utc|side)"
            ),
            "grids": {
                "threshold": False,
                "sign": False,
                "horizon": False,
                "latency": False,
                "leverage": False,
            },
        },
        "execution": {
            "joint_availability": (
                "max(BFRT.available_at_utc,WCTR.available_at_utc)"
            ),
            "ceil_5m": (
                "smallest Unix-epoch multiple of 300s >= joint availability"
            ),
            "entry": "ceil_5m(joint_availability)+300 seconds",
            "aligned_availability_still_waits_seconds": 300,
            "entry_price": "BTCUSDT perpetual 5m open at entry",
            "exit_price": "BTCUSDT perpetual 5m open at exit",
            "hold_bars_5m": policy.hold_bars_5m,
            "hold_seconds": policy.hold_seconds,
            "exit": "entry+86400 seconds",
            "leverage": policy.leverage,
            "quantity": "0.5*pre_entry_equity/entry_open; fixed through exit",
            "base_cost_bp_per_notional_side": (
                policy.base_cost_bp_per_notional_side
            ),
            "stress_cost_bp_per_notional_side": (
                policy.stress_cost_bp_per_notional_side
            ),
            "funding_interval": "entry_time <= funding_time < exit_time",
            "split_containment": (
                "entry and [entry,exit) in one split; exit may equal "
                "exclusive split end"
            ),
            "candidate_order": [
                "entry_time",
                "bucket_start_utc",
                "canonical_signal_id",
            ],
            "reservation": {
                "scope": "global chronological",
                "interval": "[entry_time,exit_time)",
                "accept": "entry_time >= prior accepted exit_time",
                "queue_shift_replacement_or_release": False,
                "source_state_independent_of_suppression": True,
            },
        },
        "calendar": {
            "full": [
                "2023-06-01T00:00:00Z",
                "2026-06-01T00:00:00Z",
            ],
            "full_wall_clock_years": 3,
            "idle_warmup_included": True,
            "selection": [
                "2023-06-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
            ],
            "future_2025": [
                "2025-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ],
            "future_2026": [
                "2026-01-01T00:00:00Z",
                "2026-06-01T00:00:00Z",
            ],
            "timezone": "UTC",
            "attribution": "entry time after split containment",
        },
        "support_gates": {
            "opens_before_novelty_or_outcomes": True,
            "selection": {
                "total_min": 45,
                "2023_NovDec_min": 6,
                "2024_H1_min": 12,
                "2024_H2_min": 12,
                "each_side_min": 14,
                "maximum_month_share": 0.20,
            },
            "future_2025": {
                "total_min": 30,
                "each_half_min": 10,
                "each_side_min": 10,
                "maximum_month_share": 0.25,
            },
            "future_2026": {
                "total_min": 15,
                "Q1_min": 6,
                "AprMay_min": 4,
                "each_side_min": 5,
                "maximum_month_share": 0.30,
            },
            "exact_join_gaps": 0,
            "future_append_invariance": {
                "required": True,
                "prefix_fields_byte_identical": [
                    "base_valid",
                    "rank_L",
                    "rank_E",
                    "rank_n",
                    "raw_candidates",
                    "accepted_signal_ids",
                    "sides",
                    "entry_exit_clocks",
                ],
            },
            "undefined_empty_or_zero_denominator": "fail",
            "failure_action": "retire exact BFWC-288 unchanged",
        },
        "controls": {
            "component_own_nonoverlap": [
                "fee_rotation_only",
                "witness_fullness_only",
                "drop_witness",
                "drop_fullness",
                "one_bucket_stale_witness_fullness",
            ],
            "definitions": {
                "fee_rotation_only": (
                    "primary abs(R) rank>=0.75; R exact nonzero; side=sign(R); "
                    "drop W,U"
                ),
                "witness_fullness_only": (
                    "Q=0.5*(abs(W)+abs(U)); same 180/120 strict-prior "
                    "midrank(Q)>=0.75; W,U exact nonzero equal sign; "
                    "side=common sign; no R"
                ),
                "drop_witness": (
                    "primary abs(R) rank>=0.75; R,U exact nonzero equal sign; "
                    "side=sign(R); drop W"
                ),
                "drop_fullness": (
                    "primary abs(R) rank>=0.75; R,W exact nonzero equal sign; "
                    "side=sign(R); drop U"
                ),
                "one_bucket_stale_witness_fullness": (
                    "current R and rank with fully formed W,U from t-1; "
                    "current availability; exact nonzero equal signs"
                ),
                "exact_direction_flip": "primary clock; side multiplied by -1",
                "deterministic_random_side": (
                    "SHA256(BFWC-288|signal_id|RANDOM_SIDE); first byte <128 "
                    "LONG else SHORT"
                ),
                "constant_long": "primary clock and accepted set; fixed LONG",
                "constant_short": "primary clock and accepted set; fixed SHORT",
                "one_bar_delayed_entry": (
                    "primary accepted set; entry and exit +300s; reject only "
                    "if shifted interval is not split-contained; no rerun"
                ),
            },
            "same_parent_set": [
                "exact_direction_flip",
                "deterministic_random_side",
                "constant_long",
                "constant_short",
                "one_bar_delayed_entry",
            ],
            "component_only": [
                "fee_rotation_only",
                "witness_fullness_only",
            ],
            "joint_specificity_rejection": (
                "either component-only control passing the complete selection "
                "and every-future economic gate rejects joint-specificity"
            ),
            "controls_cannot_replace_primary": True,
        },
        "novelty": {
            "opens_only_after_support": True,
            "candidate_outcomes_opened": False,
            "metrics": {
                "exact_entry_jaccard": (
                    "distinct exact UTC entry intersection/union"
                ),
                "candidate_6h_containment": (
                    "fraction candidate entries with any comparator entry "
                    "within +/-6 elapsed hours; matches not consumed"
                ),
                "occupied_bar_jaccard": (
                    "intersection/union of complete occupied 5m bars"
                ),
                "absolute_signed_exposure_pearson": (
                    "abs Pearson on full 5m grid; LONG=1 SHORT=-1 idle=0"
                ),
                "interval": "[entry,exit)",
                "invalid": (
                    "duplicate/overlap/off-grid/empty/undefined/nonfinite fails"
                ),
            },
            "prior_primary_clocks": [
                {
                    "id": "BFRT-288",
                    "artifact": _artifact(BFRT_PRIMARY_CLOCK),
                    "thresholds": {
                        "exact_entry_jaccard_max": 0.20,
                        "candidate_6h_containment_max": 0.50,
                        "absolute_signed_exposure_pearson_max": 0.40,
                    },
                },
                {
                    "id": "WCTR-288",
                    "artifact": _artifact(WCTR_PRIMARY_CLOCK),
                    "thresholds": {
                        "exact_entry_jaccard_max": 0.20,
                        "candidate_6h_containment_max": 0.50,
                        "absolute_signed_exposure_pearson_max": 0.40,
                    },
                },
            ],
            "gross9": {
                "rebuild_after_prior_clock_novelty": True,
                "compare_each_sleeve_separately": True,
                "weights": GROSS9_WEIGHTS,
                "gross": 9.0,
                "config": _artifact(
                    "configs/shadow/"
                    "portfolio_rank7_capacity_candidate_2026-07-28.json"
                ),
                "anchor": _artifact(
                    "results/gross9_pre2025_authoritative_"
                    "anchor_2026-07-28.json"
                ),
                "builders": [
                    _artifact(
                        "training/audit_gross9_fixed_"
                        "candidate_state_substitution.py"
                    ),
                    _artifact(
                        "training/portfolio_opt_added_alpha_update.py"
                    ),
                    _artifact(
                        "training/portfolio_opt_rank7_capacity_update.py"
                    ),
                ],
                "sealed_inputs": {
                    "market": _artifact(MARKET_DATA),
                    "funding": _artifact(FUNDING_DATA),
                    "premium": _artifact(PREMIUM_DATA),
                },
                "thresholds_each_sleeve": {
                    "exact_entry_jaccard_max": 0.10,
                    "candidate_6h_containment_max": 0.35,
                    "occupied_bar_jaccard_max": 0.25,
                    "absolute_signed_exposure_pearson_max": 0.35,
                },
                "all_five_sleeves_must_pass": True,
            },
            "failure_action": "retire exact BFWC-288 before outcomes",
        },
        "economic_contract": {
            "opens_only_after_support_novelty_and_separate_evaluator_commit": True,
            "instrument": "Binance USD-M BTCUSDT perpetual",
            "initial_equity": 1.0,
            "side_sign": {"LONG": 1, "SHORT": -1},
            "cost_cash": "abs(quantity)*execution_price*bp/10000",
            "funding_cash": (
                "-side_sign*quantity*funding_rate*settlement_mark_price"
            ),
            "full_calendar_cagr": (
                "(ending_equity/starting_equity)^(1/exact wall-clock years)-1; "
                "stitched horizon exponent is 1/3"
            ),
            "strict_mdd": (
                "global pre-entry HWM; entry fee; funding credits before "
                "favorable extreme and debits before adverse extreme; adverse "
                "intrabar before favorable; adverse virtual exit fee; exit fee"
            ),
            "mean_gross_underlying_bp": (
                "mean(side*(exit_open/entry_open-1)*10000)"
            ),
            "weekly_cluster_signflip": {
                "cluster": "UTC ISO entry week base-cost net trade PnL",
                "draws": policy.signflip_draws,
                "token": (
                    "BFWC-288|<PERIOD>|<DRAW_00000>|"
                    "<ISO_YEAR_4>-W<ISO_WEEK_2>"
                ),
                "flip": "most significant bit of SHA256 byte zero is one",
                "p": "(1+count(flipped_total>=observed_total))/20001",
            },
            "nonpositive_equity_or_nonfinite": "fail",
            "positive_cagr_zero_mdd_ratio": "positive infinity; passes",
        },
        "economic_gates": {
            "order": ["selection", "future_2025", "future_2026"],
            "complete_gate_each_period": {
                "standalone_base_and_stress_absolute_return": ">0",
                "base_and_stress_full_calendar_cagr_to_strict_mdd": ">=3.0",
                "base_and_stress_strict_mdd": "<=0.15",
                "mean_gross_underlying_bp": ">=20",
                "weekly_cluster_one_sided_signflip_p": "<=0.10",
                "long_subaccount_base_and_stress_absolute_return": ">0",
                "short_subaccount_base_and_stress_absolute_return": ">0",
                "one_bar_delayed_base_and_stress_absolute_return": ">0",
            },
            "component_only_complete_gate": (
                "same complete gate in selection and both future periods"
            ),
            "future_is_veto_only": True,
        },
        "same_gross_marginal": {
            "opens_only_after_selection_primary_complete_gate": True,
            "candidate_weights": [0.25, 0.50, 0.75, 1.00],
            "candidate_path_unit_leverage": 0.5,
            "baseline_scale": "(9-w)/9 applied pro rata to every Gross9 sleeve",
            "configured_total_gross": 9.0,
            "comparison": "unscaled authoritative Gross9 baseline",
            "selection_requirements": {
                "base_and_stress_cagr_mdd_absolute_ratio_improvement_min_each": (
                    0.05
                ),
                "base_and_stress_strict_mdd_nonworse": True,
                "base_and_stress_absolute_return_positive": True,
                "base_and_stress_baseline_return_retention_min": 0.97,
            },
            "ranking_selection_only": [
                "larger minimum base/stress ratio improvement",
                "lower maximum base/stress strict MDD",
                "larger minimum base/stress return retention",
                "lower candidate weight",
            ],
            "top_n": 1,
            "future_can_rerank_repair_or_select_rank2": False,
            "each_future_period_must_pass": True,
            "stitched_exact_3y_report_required": True,
            "stitched_report_can_repair": False,
        },
        "strict_sequence": [
            "preregistration_document_commit",
            "write_once_preregistration_commit",
            "separately_committed_evaluator_bound_to_file_and_manifest_hash",
            "dependency_and_exact_header_validation",
            "source_support_and_future_append_invariance",
            "BFRT_WCTR_primary_clock_novelty",
            "Gross9_each_sleeve_novelty",
            "selection_outcomes_and_top1_freeze",
            "future_2025_veto",
            "future_2026_veto",
            "stitched_3y_report",
        ],
        "sequence_rules": {
            "stop_at_first_failure": True,
            "no_repair_grid_fallback_or_polarity_flip": True,
            "future_cannot_rerank_or_repair": True,
            "control_cannot_replace_primary": True,
            "exact_rule_retired_on_any_failure": True,
        },
        "source_rows_opened": False,
        "source_incidence_opened": False,
        "candidate_overlap_opened": False,
        "economic_rows_opened": False,
        "outcomes_opened": False,
    }


def build_manifest() -> dict[str, Any]:
    core = _core_manifest()
    return {**core, "manifest_hash": canonical_hash(core)}


def validate_manifest(payload: Mapping[str, Any]) -> None:
    expected = build_manifest()
    if dict(payload) != expected:
        raise RuntimeError("BFWC-288 preregistration differs from code")
    core = {
        key: value for key, value in payload.items() if key != "manifest_hash"
    }
    if payload.get("manifest_hash") != canonical_hash(core):
        raise RuntimeError("BFWC-288 internal manifest hash mismatch")
    for key in (
        "source_rows_opened",
        "source_incidence_opened",
        "candidate_overlap_opened",
        "economic_rows_opened",
        "outcomes_opened",
    ):
        if payload.get(key) is not False:
            raise RuntimeError(f"BFWC-288 evidence boundary opened: {key}")
    disclosure = payload["evidence_disclosure"]
    if disclosure["csv_data_rows_decoded_by_preregistration"] != 0:
        raise RuntimeError("BFWC-288 preregistration decoded CSV rows")
    if disclosure["network_calls"] != 0 or disclosure["subprocess_calls"] != 0:
        raise RuntimeError("BFWC-288 preregistration used external effects")
    if sum(GROSS9_WEIGHTS.values()) != 9.0:
        raise RuntimeError("BFWC-288 Gross9 weights do not sum to nine")


def canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    validate_manifest(payload)
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _output_relative(path: str | Path) -> Path:
    candidate = Path(path)
    raw = str(path)
    if (
        candidate.is_absolute()
        or raw.startswith("~")
        or ".." in candidate.parts
        or candidate.name in {"", ".", ".."}
    ):
        raise RuntimeError("BFWC-288 output must be repository-relative")
    return candidate


def _assert_secure_io_capabilities() -> None:
    for name in ("O_NOFOLLOW", "O_DIRECTORY"):
        value = getattr(os, name, None)
        if not isinstance(value, int) or value == 0:
            raise RuntimeError(f"BFWC-288 requires nonzero os.{name}")
    for function in (os.open, os.link, os.unlink):
        if function not in os.supports_dir_fd:
            raise RuntimeError(
                f"BFWC-288 requires dir_fd support for {function.__name__}"
            )
    if os.link not in os.supports_follow_symlinks:
        raise RuntimeError(
            "BFWC-288 requires follow_symlinks support for os.link"
        )


def _open_parent(candidate: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current = os.open(REPOSITORY_ROOT, flags)
    try:
        for part in candidate.parent.parts:
            next_descriptor = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = next_descriptor
        return current
    except OSError as error:
        os.close(current)
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RuntimeError("BFWC-288 output parent is unsafe") from error
        raise


def _read_regular(directory: int, name: str) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise RuntimeError("BFWC-288 output path is unsafe") from error
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("BFWC-288 output is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def write_once(
    output: str | Path = DEFAULT_OUTPUT,
    payload: Mapping[str, Any] | None = None,
) -> str:
    candidate = _output_relative(output)
    _assert_secure_io_capabilities()
    validate_frozen_dependencies()
    expected = build_manifest() if payload is None else dict(payload)
    canonical = canonical_manifest_bytes(expected)
    parent = _open_parent(candidate)
    temporary_name = (
        f".{candidate.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    temporary_created = False
    try:
        try:
            existing = _read_regular(parent, candidate.name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != canonical:
                raise RuntimeError(
                    "BFWC-288 existing preregistration is noncanonical"
                )
            return "verified_existing"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        temporary_created = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary_name,
                candidate.name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_regular(parent, candidate.name) != canonical:
                raise RuntimeError("BFWC-288 preregistration race drift")
            return "verified_existing"
        os.fsync(parent)
        return "created"
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent)
            except FileNotFoundError:
                pass
            os.fsync(parent)
        os.close(parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_manifest()
    status = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "output": str(args.output),
                "manifest_hash": payload["manifest_hash"],
                "source_rows_opened": payload["source_rows_opened"],
                "outcomes_opened": payload["outcomes_opened"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
