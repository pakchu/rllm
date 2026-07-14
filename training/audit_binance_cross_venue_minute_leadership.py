"""Audit cross-venue minute-order descriptors before any outcome is opened."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.build_binance_cross_venue_minute_leadership import (
    AUDIT_COLUMNS,
    FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    SCHEMA_VERSION,
    SEALED_END_EXCLUSIVE,
)


@dataclass(frozen=True)
class AuditConfig:
    features: str = (
        "data/binance_cross_venue_minute_leadership_btc_2020_2023/"
        "BTCUSDT_cross_venue_minute_leadership_5m_2020-01_2023-12.csv.gz"
    )
    manifest: str = (
        "data/binance_cross_venue_minute_leadership_btc_2020_2023/"
        "build_manifest.json"
    )
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    output: str = "results/binance_cross_venue_minute_leadership_audit_2026-07-14.json"
    minimum_source_complete_fraction: float = 0.995
    minimum_feature_valid_fraction: float = 0.995


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _allclose(left: pd.Series, right: pd.Series, *, atol: float = 1e-9) -> bool:
    return bool(
        np.allclose(
            left.to_numpy(float),
            right.to_numpy(float),
            rtol=1e-9,
            atol=atol,
        )
    )


def _expected_months(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    final_instant = end - pd.Timedelta("1ns")
    return [str(period) for period in pd.period_range(start, final_instant, freq="M")]


def _manifest_checks(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    features_path: Path,
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, bool], dict[str, Any]]:
    months = manifest.get("months", [])
    expected_months = _expected_months(start, end)
    monthly_hashes_valid = True
    for month in months:
        output_path = Path(str(month.get("output", "")))
        monthly_hashes_valid &= (
            output_path.is_file()
            and _sha256(output_path) == month.get("output_sha256")
        )
    archive_hashes_valid = all(
        _is_sha256(month.get("spot_archive_sha256"))
        and _is_sha256(month.get("um_archive_sha256"))
        for month in months
    )
    protocol = manifest.get("protocol", {})
    config = manifest.get("config", {})
    checks = {
        "manifest_is_sibling_of_features": manifest_path.parent == features_path.parent,
        "manifest_combined_path_exact": manifest.get("combined_output") == str(features_path),
        "manifest_combined_hash_valid": _sha256(features_path)
        == manifest.get("combined_sha256"),
        "manifest_columns_exact": manifest.get("columns") == list(OUTPUT_COLUMNS),
        "manifest_row_counts_exact": (
            manifest.get("rows") == len(frame)
            and manifest.get("source_complete_rows")
            == int(frame["source_complete"].astype(bool).sum())
            and manifest.get("feature_valid_rows")
            == int(frame["cross_venue_feature_valid"].astype(bool).sum())
            and manifest.get("quarantined_rows")
            == int((~frame["cross_venue_feature_valid"].astype(bool)).sum())
        ),
        "manifest_config_interval_exact": (
            pd.Timestamp(config.get("start")) == start
            and pd.Timestamp(config.get("end")) == end
        ),
        "manifest_months_exact": [month.get("month") for month in months]
        == expected_months,
        "manifest_schema_versions_exact": all(
            month.get("schema_version") == SCHEMA_VERSION for month in months
        ),
        "manifest_monthly_hashes_valid": bool(monthly_hashes_valid),
        "manifest_archive_hashes_well_formed": archive_hashes_valid,
        "manifest_outcomes_unopened": protocol.get("outcomes_opened") is False,
        "manifest_raw_archives_not_persisted": (
            protocol.get("raw_archives_persisted") is False
            and not any(manifest_path.parent.rglob("*.zip"))
        ),
        "manifest_seal_exact": protocol.get("sealed_end_exclusive")
        == SEALED_END_EXCLUSIVE.isoformat(),
    }
    diagnostics = {
        "months": len(months),
        "expected_months": expected_months,
        "combined_sha256": manifest.get("combined_sha256"),
        "monthly_outputs_bytes": int(
            sum(
                Path(str(month["output"])).stat().st_size
                for month in months
                if Path(str(month.get("output", ""))).is_file()
            )
        ),
    }
    return checks, diagnostics


def _feature_checks(
    frame: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: AuditConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    expected = pd.date_range(start, end, inclusive="left", freq="5min")
    source_complete = frame["source_complete"].astype(bool)
    feature_valid = frame["cross_venue_feature_valid"].astype(bool)
    calculated_source_complete = (
        frame["spot_rows"].eq(5)
        & frame["um_rows"].eq(5)
        & frame["spot_missing_minutes"].eq(0)
        & frame["um_missing_minutes"].eq(0)
        & frame["spot_invalid_source_minutes"].eq(0)
        & frame["um_invalid_source_minutes"].eq(0)
    )
    valid_features = frame.loc[feature_valid, FEATURE_COLUMNS]
    invalid_features = frame.loc[~feature_valid, FEATURE_COLUMNS]
    centroid_columns = [
        column
        for column in FEATURE_COLUMNS
        if column.endswith("_time_centroid") and not column.startswith("um_minus_")
    ]
    bounded_columns = (
        "flow_transfer_asymmetry",
        "return_leadership_asymmetry",
        "simultaneous_flow_sign_agreement",
        "simultaneous_return_sign_agreement",
        "spot_to_um_lagged_directional_alignment",
        "um_to_spot_lagged_directional_alignment",
        "lagged_directional_alignment_diff",
        "reverse_flow_transfer_asymmetry",
        "reverse_return_leadership_asymmetry",
        "reverse_spot_to_um_lagged_directional_alignment",
        "reverse_um_to_spot_lagged_directional_alignment",
        "reverse_lagged_directional_alignment_diff",
    )
    forbidden_tokens = (
        "future",
        "forward",
        "label",
        "target",
        "reward",
        "action",
        "profit",
        "pnl",
    )
    checks = {
        "columns_exact": tuple(frame.columns) == OUTPUT_COLUMNS,
        "grid_exact": frame["date"].equals(pd.Series(expected, name="date")),
        "timestamps_unique_monotonic": (
            not frame["date"].duplicated().any() and frame["date"].is_monotonic_increasing
        ),
        "feature_availability_exact": frame["feature_available_time_utc"].equals(
            frame["date"] + pd.Timedelta("5min")
        ),
        "earliest_trade_exact": frame["trade_earliest_time_utc"].equals(
            frame["feature_available_time_utc"]
        ),
        "sealed_interval_respected": bool(
            frame["date"].min() >= start
            and frame["date"].max() < end
            and end <= pd.Timestamp(SEALED_END_EXCLUSIVE)
        ),
        "source_complete_identity": source_complete.equals(calculated_source_complete),
        "feature_valid_requires_complete_source": bool((~feature_valid | source_complete).all()),
        "valid_rows_have_four_lagged_pairs": bool(
            frame.loc[feature_valid, "lagged_pair_count"].eq(4).all()
        ),
        "valid_rows_have_four_reverse_lagged_pairs": bool(
            frame.loc[feature_valid, "reverse_lagged_pair_count"].eq(4).all()
        ),
        "valid_rows_have_ok_reason": bool(
            frame.loc[feature_valid, "feature_invalid_reason"].eq("ok").all()
        ),
        "invalid_rows_have_reason": bool(
            frame.loc[~feature_valid, "feature_invalid_reason"].ne("ok").all()
        ),
        "valid_features_finite": bool(
            np.isfinite(valid_features.to_numpy(float)).all()
        ),
        "invalid_features_quarantined": bool(invalid_features.isna().all().all()),
        "normalized_features_bounded": bool(
            valid_features.loc[:, bounded_columns]
            .apply(lambda values: values.between(-1.0 - 1e-10, 1.0 + 1e-10).all())
            .all()
        ),
        "timing_centroids_bounded": bool(
            valid_features.loc[:, centroid_columns]
            .apply(lambda values: values.between(0.0, 1.0).all())
            .all()
        ),
        "lagged_flow_diff_identity": _allclose(
            valid_features["lagged_flow_response_diff_bp"],
            valid_features["spot_to_um_lagged_flow_response_bp"]
            - valid_features["um_to_spot_lagged_flow_response_bp"],
        ),
        "directional_diff_identity": _allclose(
            valid_features["lagged_directional_alignment_diff"],
            0.5
            * (
                valid_features["spot_to_um_lagged_directional_alignment"]
                - valid_features["um_to_spot_lagged_directional_alignment"]
            ),
        ),
        "reverse_lagged_flow_diff_identity": _allclose(
            valid_features["reverse_lagged_flow_response_diff_bp"],
            valid_features["reverse_spot_to_um_lagged_flow_response_bp"]
            - valid_features["reverse_um_to_spot_lagged_flow_response_bp"],
        ),
        "reverse_directional_diff_identity": _allclose(
            valid_features["reverse_lagged_directional_alignment_diff"],
            0.5
            * (
                valid_features[
                    "reverse_spot_to_um_lagged_directional_alignment"
                ]
                - valid_features[
                    "reverse_um_to_spot_lagged_directional_alignment"
                ]
            ),
        ),
        "basis_change_identity": _allclose(
            valid_features["basis_change_bp"],
            valid_features["close_basis_bp"] - valid_features["open_basis_bp"],
            atol=1e-8,
        ),
        "activity_timing_diff_identity": _allclose(
            valid_features["um_minus_spot_activity_time_centroid"],
            valid_features["um_activity_time_centroid"]
            - valid_features["spot_activity_time_centroid"],
        ),
        "flow_timing_diff_identity": _allclose(
            valid_features["um_minus_spot_flow_time_centroid"],
            valid_features["um_flow_time_centroid"]
            - valid_features["spot_flow_time_centroid"],
        ),
        "return_timing_diff_identity": _allclose(
            valid_features["um_minus_spot_return_time_centroid"],
            valid_features["um_return_time_centroid"]
            - valid_features["spot_return_time_centroid"],
        ),
        "source_complete_fraction_sufficient": float(source_complete.mean())
        >= cfg.minimum_source_complete_fraction,
        "feature_valid_fraction_sufficient": float(feature_valid.mean())
        >= cfg.minimum_feature_valid_fraction,
        "no_outcome_columns": not any(
            token in column.lower().split("_")
            for column in frame.columns
            for token in forbidden_tokens
        ),
        "audit_columns_present": all(column in frame.columns for column in AUDIT_COLUMNS),
    }
    by_year: dict[str, dict[str, int]] = {}
    for year in sorted(frame["date"].dt.year.unique()):
        mask = frame["date"].dt.year.eq(year)
        by_year[str(year)] = {
            "rows": int(mask.sum()),
            "source_complete_rows": int((mask & source_complete).sum()),
            "feature_valid_rows": int((mask & feature_valid).sum()),
        }
    diagnostics = {
        "rows": int(len(frame)),
        "source_complete_rows": int(source_complete.sum()),
        "source_complete_fraction": float(source_complete.mean()),
        "feature_valid_rows": int(feature_valid.sum()),
        "feature_valid_fraction": float(feature_valid.mean()),
        "quarantined_rows": int((~feature_valid).sum()),
        "first_date": str(frame["date"].min()),
        "last_date": str(frame["date"].max()),
        "invalid_reason_counts": {
            str(reason): int(count)
            for reason, count in frame.loc[
                ~feature_valid, "feature_invalid_reason"
            ].value_counts().items()
        },
        "source_gap_totals": {
            column: int(frame[column].sum())
            for column in (
                "spot_missing_minutes",
                "um_missing_minutes",
                "spot_invalid_source_minutes",
                "um_invalid_source_minutes",
            )
        },
        "by_year": by_year,
        "feature_bounds": {
            column: {
                "min": float(valid_features[column].min()),
                "max": float(valid_features[column].max()),
            }
            for column in bounded_columns
        },
    }
    return checks, diagnostics


def run_audit(cfg: AuditConfig) -> dict[str, Any]:
    features_path = Path(cfg.features)
    manifest_path = Path(cfg.manifest)
    if not features_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("cross-venue features or build manifest is missing")
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    if start >= end:
        raise ValueError("audit start must precede exclusive end")
    manifest = json.loads(manifest_path.read_text())
    frame = pd.read_csv(
        features_path,
        compression="gzip",
        parse_dates=["date", "feature_available_time_utc", "trade_earliest_time_utc"],
    )
    manifest_checks, manifest_diagnostics = _manifest_checks(
        manifest,
        manifest_path=manifest_path,
        features_path=features_path,
        frame=frame,
        start=start,
        end=end,
    )
    feature_checks, feature_diagnostics = _feature_checks(
        frame, start=start, end=end, cfg=cfg
    )
    checks = {
        **{f"manifest.{name}": value for name, value in manifest_checks.items()},
        **{f"feature.{name}": value for name, value in feature_checks.items()},
    }
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "config": asdict(cfg),
        "passed": not failed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
        "diagnostics": {
            **feature_diagnostics,
            "manifest": manifest_diagnostics,
        },
        "manifest_diagnostics": manifest_diagnostics,
        "feature_diagnostics": feature_diagnostics,
    }
    output_path = Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=AuditConfig.features)
    parser.add_argument("--manifest", default=AuditConfig.manifest)
    parser.add_argument("--start", default=AuditConfig.start)
    parser.add_argument("--end", default=AuditConfig.end)
    parser.add_argument("--output", default=AuditConfig.output)
    parser.add_argument(
        "--minimum-source-complete-fraction",
        type=float,
        default=AuditConfig.minimum_source_complete_fraction,
    )
    parser.add_argument(
        "--minimum-feature-valid-fraction",
        type=float,
        default=AuditConfig.minimum_feature_valid_fraction,
    )
    result = run_audit(AuditConfig(**vars(parser.parse_args())))
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "failed_checks": result["failed_checks"],
                **{
                    key: result["feature_diagnostics"][key]
                    for key in (
                        "rows",
                        "source_complete_fraction",
                        "feature_valid_fraction",
                        "quarantined_rows",
                    )
                },
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
