"""Audit symbolic feature/token distribution drift across JSONL splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.symbolic_action_ridge import row_tokens


def load_token_counts(path: str | Path) -> tuple[Counter[str], int]:
    counts: Counter[str] = Counter()
    n = 0
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            counts.update(set(row_tokens(row)))
            n += 1
    return counts, n


def freq(counts: Counter[str], n: int, token: str) -> float:
    return counts[token] / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="append", nargs=2, metavar=("LABEL", "JSONL"), required=True)
    parser.add_argument("--min-frequency", type=float, default=0.01)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--filter-prefix", action="append", default=[], help="Only include tokens with one of these prefixes. Can be repeated.")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    labels = [label for label, _ in args.split]
    loaded = {label: load_token_counts(path) for label, path in args.split}
    all_tokens = set().union(*(set(counts) for counts, _ in loaded.values()))
    if args.filter_prefix:
        prefixes = tuple(args.filter_prefix)
        all_tokens = {token for token in all_tokens if token.startswith(prefixes)}

    comparisons = []
    for left, right in zip(labels, labels[1:]):
        left_counts, left_n = loaded[left]
        right_counts, right_n = loaded[right]
        rows = []
        for token in all_tokens:
            lf = freq(left_counts, left_n, token)
            rf = freq(right_counts, right_n, token)
            if max(lf, rf) < float(args.min_frequency):
                continue
            rows.append({"token": token, "left_frequency": lf, "right_frequency": rf, "delta": rf - lf, "abs_delta": abs(rf - lf)})
        comparisons.append({"left": left, "right": right, "top_drift": sorted(rows, key=lambda r: r["abs_delta"], reverse=True)[: int(args.top_n)]})

    payload = {"splits": {label: {"rows": loaded[label][1], "path": path} for label, path in args.split}, "comparisons": comparisons}
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
