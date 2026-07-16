"""Freeze the Wikimedia attention-divergence alpha protocol before outcomes.

This module intentionally contains no market or page-view loader.  It only
materializes the immutable hypothesis family and staged disclosure contract:
2020-2022 may be used for selection, 2023 is a sealed pre-2024 holdout, and
2024+ remains unopened until the selected policy survives 2023.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = (
    "results/wikimedia_attention_divergence_preregistration_2026-07-16.json"
)
DEFAULT_DOCS = "docs/wikimedia-attention-divergence-preregistration-2026-07-16.md"

ARTICLES = ("Bitcoin", "Ethereum", "Cryptocurrency", "Blockchain")
PROJECT = "en.wikipedia.org"
ACCESS = "all-access"
AGENT = "user"
SELECTION_END = "2023-01-01"
PRE2024_HOLDOUT = ("2023-01-01", "2024-01-01")
FUTURE_SEAL_START = "2024-01-01"


@dataclass(frozen=True, order=True)
class Policy:
    family: str
    attention_threshold: float
    price_horizon_days: int
    price_threshold: float
    hold_days: int


def policy_grid() -> list[Policy]:
    """Return the complete, bounded family in deterministic order."""
    policies: list[Policy] = []
    for attention in (2.0, 3.0):
        for price in (0.04, 0.08):
            for hold in (1, 3):
                policies.append(
                    Policy("broad_attention_reversal", attention, 1, price, hold)
                )
    for hold in (1, 3):
        policies.append(
            Policy("bitcoin_share_reversal", 2.0, 3, 0.08, hold)
        )
    for price in (0.04, 0.08):
        for hold in (1, 3):
            policies.append(
                Policy("silent_impulse_continuation", 0.0, 1, price, hold)
            )
    return sorted(policies)


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_manifest() -> dict[str, Any]:
    core: dict[str, Any] = {
        "protocol_version": "wikimedia_attention_divergence_v1",
        "outcomes_opened": False,
        "hypothesis": {
            "broad_attention_reversal": (
                "unusually broad human attention accompanying a large completed "
                "BTC move marks demand/supply exhaustion; fade the move"
            ),
            "bitcoin_share_reversal": (
                "BTC-specific attention concentration after a large three-day move "
                "marks late narrative crowding; fade the move"
            ),
            "silent_impulse_continuation": (
                "a large BTC move without elevated public attention is more likely "
                "information-led than retail-crowded; follow the move"
            ),
        },
        "source_contract": {
            "api": "Wikimedia Analytics API REST v1 pageviews per-article",
            "endpoint_template": (
                "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                "{project}/{access}/{agent}/{article}/daily/{start}/{end}"
            ),
            "aggregate_endpoint_template": (
                "https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/"
                "{project}/{access}/{agent}/daily/{start}/{end}"
            ),
            "official_reference": (
                "https://doc.wikimedia.org/generated-data-platform/aqs/"
                "analytics-api/reference/page-views.html"
            ),
            "availability_reference": (
                "https://doc.wikimedia.org/generated-data-platform/aqs/"
                "analytics-api/documentation/troubleshooting.html"
            ),
            "project": PROJECT,
            "access": ACCESS,
            "agent": AGENT,
            "articles": list(ARTICLES),
            "redirects_aggregated": False,
            "missing_day_policy": "fail_closed_no_imputation",
            "historical_snapshot_is_point_in_time": False,
            "promotion_requires_retrieval_timestamped_forward_shadow": True,
            "known_source_issue": (
                "official pageview_hourly history reports approximately 2.80% to "
                "4.34% traffic loss between 2021-06-04 and 2022-01-26"
            ),
            "known_source_issue_reference": (
                "https://wikitech.wikimedia.org/wiki/"
                "Data_Platform/Data_Lake/Traffic/Pageview_hourly"
            ),
        },
        "availability_contract": {
            "observation": "UTC calendar day D pageviews and BTC 23:55 5m close",
            "historical_assumed_available_at": "D+2 12:05 UTC",
            "minimum_delay_after_observation_end_hours": 36.0,
            "signal_anchor": "D+2 12:05 UTC",
            "execution": "next 5m open at D+2 12:10 UTC",
            "reason": (
                "official docs say daily data normally loads within hours but can "
                "take 24 hours or more; the conservative delay is still not a "
                "historical PIT proof"
            ),
        },
        "feature_contract": {
            "normalization": (
                "each article's daily user views divided by same-day aggregate "
                "en.wikipedia user views, expressed per million project views"
            ),
            "article_transform": "log1p(normalized daily views per million)",
            "broad_attention": (
                "sum normalized views-per-million across the four fixed articles"
            ),
            "broad_attention_z": (
                "robust z of current log1p broad attention versus strictly prior "
                "90 days, minimum 45; median and 1.4826*MAD"
            ),
            "bitcoin_share": "Bitcoin views / broad attention views",
            "bitcoin_share_z": (
                "robust z of current share versus strictly prior 90 days, minimum 45"
            ),
            "price_returns": (
                "log return of completed UTC 23:55 BTC closes over fixed 1d or 3d"
            ),
            "finite_only": True,
        },
        "policies": [asdict(policy) for policy in policy_grid()],
        "family_rules": {
            "broad_attention_reversal": (
                "broad_attention_z >= threshold and abs(price_return) >= threshold; "
                "side = -sign(price_return)"
            ),
            "bitcoin_share_reversal": (
                "bitcoin_share_z >= threshold, broad_attention_z >= 1.0, and "
                "abs(3d price_return) >= threshold; side = -sign(price_return)"
            ),
            "silent_impulse_continuation": (
                "broad_attention_z <= threshold and abs(price_return) >= threshold; "
                "side = sign(price_return)"
            ),
        },
        "selection_protocol": {
            "fit": ["2020-01-01", "2022-01-01"],
            "selection": ["2022-01-01", SELECTION_END],
            "sealed_pre2024_holdout": list(PRE2024_HOLDOUT),
            "future_seal_start": FUTURE_SEAL_START,
            "selection_data_must_be_physically_truncated_before": SELECTION_END,
            "rank": [
                "descending minimum(fit_2020, fit_2021, selection_2022) CAGR/strict_MDD",
                "descending combined_2020_2022 CAGR/strict_MDD",
                "ascending policy tuple",
            ],
            "selection_gates": {
                "combined_absolute_return_positive": True,
                "combined_cagr_to_strict_mdd_min": 2.0,
                "selection_2022_absolute_return_positive": True,
                "selection_2022_cagr_to_strict_mdd_min": 2.0,
                "every_calendar_year_absolute_return_positive": True,
                "combined_trades_min": 18,
                "each_calendar_year_trades_min": 4,
                "strict_mdd_pct_max": 15.0,
                "double_cost_combined_positive": True,
                "double_cost_combined_ratio_min": 1.5,
                "inverted_side_combined_absolute_return_max": 0.0,
            },
            "holdout_2023_gates": {
                "absolute_return_positive": True,
                "cagr_to_strict_mdd_min": 3.0,
                "strict_mdd_pct_max": 15.0,
                "trades_min": 6,
                "h1_absolute_return_nonnegative": True,
                "h2_absolute_return_nonnegative": True,
                "combined_2020_2023_cagr_to_strict_mdd_min": 3.0,
                "familywise_weekly_block_bootstrap_p_max": 0.10,
            },
            "multiple_testing_hypotheses": len(policy_grid()),
            "familywise_adjustment": "Bonferroni over all preregistered policies",
        },
        "execution_contract": {
            "market": "BTCUSDT perpetual 5m OHLC",
            "entry_delay_bars": 1,
            "nonoverlap": True,
            "leverage": 0.5,
            "fee_rate_per_side": 0.0005,
            "slippage_rate_per_side": 0.0001,
            "realized_funding": True,
            "strict_mdd": (
                "global high-water plus intratrade favorable-before-adverse OHLC "
                "path, fees, slippage, and funding debit"
            ),
            "cagr_clock": "full split calendar including idle time",
        },
        "controls": {
            "double_cost": True,
            "inverted_side": True,
            "one_day_later_entry": True,
            "random_same_side_same_count": 5000,
            "random_seed": 20260716,
        },
        "orthogonality_gate_after_standalone_survival": {
            "exact_entry_jaccard_max": 0.02,
            "candidate_entries_near_existing_6h_fraction_max": 0.25,
            "position_jaccard_max": 0.15,
            "absolute_daily_pnl_pearson_max": 0.30,
            "minimum_nonzero_daily_pnl_days": 10,
            "undefined_metric": "fail_closed",
            "marginal_portfolio_improvement_required": True,
        },
    }
    return {
        **core,
        "manifest_hash": canonical_hash(core),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    core = {k: v for k, v in manifest.items() if k not in {"manifest_hash", "created_at"}}
    if canonical_hash(core) != manifest.get("manifest_hash"):
        raise RuntimeError("Wikimedia preregistration manifest hash mismatch")
    if manifest.get("outcomes_opened") is not False:
        raise RuntimeError("preregistration cannot claim outcomes are open")
    if manifest.get("policies") != [asdict(policy) for policy in policy_grid()]:
        raise RuntimeError("policy family differs from preregistration code")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_manifest()
    validate_manifest(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "manifest_hash": payload["manifest_hash"], "policies": len(payload["policies"])}, indent=2))


if __name__ == "__main__":
    main()
