#!/usr/bin/env python3
"""Prune generated runtime artifacts before the WSL disk grows too much.

The script is intentionally conservative by default: it removes only generated
checkpoints/results/logs/data artifacts and keeps the newest N checkpoint
folders plus all explicitly named paths.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


GENERATED_DIRS = ("checkpoints", "results", "logs", "data")


def _bytes_to_gib(n: int) -> float:
    return float(n) / (1024.0 ** 3)


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                total += child.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _children_by_age(path: Path) -> list[Path]:
    if not path.exists():
        return []
    children = [p for p in path.iterdir() if p.name != ".gitkeep"]
    return sorted(children, key=lambda p: p.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--max-gib", type=float, default=280.0)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Path to preserve. Can be repeated. Relative paths are resolved under --root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    keep_paths = set()
    for item in args.keep:
        p = Path(item)
        keep_paths.add((p if p.is_absolute() else root / p).resolve())

    generated_roots = [root / name for name in GENERATED_DIRS]
    repo_size = _path_size(root)
    generated_size = sum(_path_size(p) for p in generated_roots)
    print(
        f"root={root} repo_size={_bytes_to_gib(repo_size):.2f}GiB "
        f"generated_size={_bytes_to_gib(generated_size):.2f}GiB "
        f"limit={float(args.max_gib):.2f}GiB"
    )

    # Always keep newest checkpoint folders. They are usually the active branch
    # outputs and are expensive to regenerate.
    ckpt_dir = root / "checkpoints"
    ckpts = _children_by_age(ckpt_dir)
    newest_ckpts = set(p.resolve() for p in ckpts[-max(0, int(args.keep_checkpoints)) :])
    keep_paths.update(newest_ckpts)

    if generated_size <= int(float(args.max_gib) * 1024**3):
        print("under-limit: no pruning required")
        for p in sorted(keep_paths):
            if p.exists():
                print(f"keep {p.relative_to(root)}")
        return 0

    candidates: list[Path] = []
    for base in generated_roots:
        for child in _children_by_age(base):
            resolved = child.resolve()
            if resolved in keep_paths:
                continue
            candidates.append(child)

    removed = 0
    for child in candidates:
        if generated_size <= int(float(args.max_gib) * 1024**3):
            break
        size = _path_size(child)
        print(f"prune {child.relative_to(root)} size={_bytes_to_gib(size):.2f}GiB")
        if not args.dry_run:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
        generated_size -= size
        removed += size

    print(
        f"done removed={_bytes_to_gib(removed):.2f}GiB "
        f"generated_now={_bytes_to_gib(max(0, generated_size)):.2f}GiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
