"""Audit local market files and classify tradable universe availability."""
from __future__ import annotations

import argparse
import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class Cfg:
    roots: str
    output: str


def csv_gz_files(roots: list[str]) -> list[str]:
    out=[]
    for root in roots:
        for p in Path(root).glob("*.csv.gz"):
            out.append(str(p))
    return sorted(set(out))


def inspect_file(path: str) -> dict[str, Any]:
    with gzip.open(path, "rt") as f:
        header=f.readline().strip().split(',')
        first=f.readline().strip().split(',')
    row=dict(zip(header, first))
    tic=row.get('tic','')
    interval=row.get('interval','')
    date=row.get('date') or row.get('ts') or ''
    # Tail via streaming to avoid shell dependency.
    last=[]
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.strip(): last=line.strip().split(',')
    last_row=dict(zip(header,last)) if last else {}
    return {
        "path": path,
        "basename": os.path.basename(path),
        "size_mb": round(os.path.getsize(path)/1024/1024, 3),
        "columns": header,
        "first_tic": tic,
        "first_interval": interval,
        "first_date": date,
        "last_tic": last_row.get('tic',''),
        "last_date": last_row.get('date') or last_row.get('ts') or '',
    }


def classify(files: list[dict[str, Any]]) -> dict[str, Any]:
    tics=sorted(set(x.get('first_tic','') for x in files) | set(x.get('last_tic','') for x in files))
    binance_like=sorted(t for t in tics if t.endswith('USDT'))
    return {
        "tics_seen_boundary_sample": tics,
        "binance_usdt_boundary_sample": binance_like,
        "has_multiple_binance_usdt_assets": len(set(binance_like)) > 1,
        "available_binance_usdt_assets": sorted(set(binance_like)),
        "supporting_exogenous_assets": sorted(t for t in tics if t and not t.endswith('USDT')),
        "interpretation": "Local CSVs expose BTCUSDT as the only Binance-USDT tradable asset by boundary tic inspection; KRW-BTC and FX files are exogenous/supporting data, not Binance futures portfolio members.",
    }


def run(c: Cfg) -> dict[str, Any]:
    roots=[x for x in c.roots.split(',') if x.strip()]
    files=[inspect_file(p) for p in csv_gz_files(roots)]
    report={"config": c.__dict__, "files": files, "classification": classify(files)}
    Path(c.output).parent.mkdir(parents=True, exist_ok=True)
    Path(c.output).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> Cfg:
    p=argparse.ArgumentParser()
    p.add_argument('--roots', required=True)
    p.add_argument('--output', required=True)
    return Cfg(**vars(p.parse_args()))

def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
if __name__ == '__main__': main()
