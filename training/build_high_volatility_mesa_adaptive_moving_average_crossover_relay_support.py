"""Build outcome-blind source support for frozen HVMAMA-C05-005-24."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training import preregister_high_volatility_mesa_adaptive_moving_average_crossover_relay as prereg

ENV_FILE = "/home/pakchu/rllm/.env"
START = pd.Timestamp("2020-01-01T00:00:00Z")
END = pd.Timestamp("2026-08-01T00:00:00Z")
PREREG_SHA = "fd4e72a1d35ce7b22d5fff12d5252fb9d081f53082a6dd651b72023e6cd9e073"
REG = prereg.build()
P = REG["policy"]
SPLITS = {name: tuple(map(pd.Timestamp, window)) for name, window in REG["stages"].items()}
GATES = REG["source_support_gates"]
CONTROLS = tuple(REG["diagnostic_controls"]["names"])
QUERY = """SELECT ts,open,high,low,close FROM bars_binance WHERE symbol='BTCUSDT' AND interval='1m' AND ts>=:start AND ts<:end ORDER BY ts"""
ROOT = Path("data/high_volatility_mesa_adaptive_moving_average_crossover_relay_sources_2023_2026")
PANEL = ROOT / "four_hour_states.csv.gz"
MANIFEST = ROOT / "manifest.json"
CLOCK = Path("data/high_volatility_mesa_adaptive_moving_average_crossover_relay_clocks_2023_2026.csv.gz")
SPLIT_DIR = Path("data/high_volatility_mesa_adaptive_moving_average_crossover_relay_split_clocks_2023_2026")
CONTROL_DIR = Path("data/high_volatility_mesa_adaptive_moving_average_crossover_relay_controls_2023_2026")
RESULT = Path("results/high_volatility_mesa_adaptive_moving_average_crossover_relay_support_2026-08-11.json")
BUILDER = Path(__file__).relative_to(Path.cwd())
PANEL_COLUMNS = (
    "source_start", "feature_available_time", "source_valid", "bar_open", "bar_high",
    "bar_low", "bar_close", "median_price", "smooth", "detrender", "in_phase1",
    "quadrature1", "period", "phase", "alpha", "mama", "fama", "entry_side",
    "raw_median_side", "variation", "variation_rank", "eligible",
)
CLOCK_COLUMNS = (
    "candidate", "control", "split", "source_start", "feature_available_time",
    "entry_time", "exit_time", "side", "median_price", "smooth", "detrender",
    "in_phase1", "quadrature1", "period", "phase", "alpha", "mama", "fama",
    "entry_side", "raw_median_side", "variation", "variation_rank", "eligible",
)

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def prior_rank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    output = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        prior = np.asarray(history[-P["variation_history_decisions"] :], float)
        if math.isfinite(value) and len(prior) >= P["minimum_variation_history_decisions"]:
            output[index] = (np.sum(prior < value) + 0.5 * np.sum(prior == value)) / len(prior)
        if math.isfinite(value):
            history.append(float(value))
    return pd.Series(output, index=series.index)


def postgres_engine():
    from sqlalchemy import create_engine
    from preprocessing.live_db_features import load_env_file, postgres_url_from_env

    load_env_file(ENV_FILE)
    return create_engine(postgres_url_from_env(ENV_FILE), connect_args={"connect_timeout": 10})


def load_source() -> pd.DataFrame:
    from sqlalchemy import text

    database = postgres_engine()
    try:
        with database.connect() as connection:
            return pd.read_sql_query(
                text(QUERY), connection,
                params={"start": START.to_pydatetime(), "end": END.to_pydatetime()},
            )
    finally:
        database.dispose()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.tolist() != ["ts", "open", "high", "low", "close"]:
        raise RuntimeError("HVMAMA source schema drift")
    source = frame.copy()
    source["ts"] = pd.to_datetime(source.ts, utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    if source.ts.isna().any() or source.ts.duplicated().any():
        raise RuntimeError("HVMAMA invalid source key")
    prices = source[["open", "high", "low", "close"]]
    source["row_valid"] = (
        np.isfinite(prices).all(axis=1)
        & prices.gt(0).all(axis=1)
        & source.high.ge(prices[["open", "close"]].max(axis=1))
        & source.low.le(prices[["open", "close"]].min(axis=1))
        & source.high.ge(source.low)
    )
    source["minute_sq_return"] = np.square(np.log(source.close / source.open)).where(source.row_valid)
    return source.set_index("ts").sort_index()


class LeanMama:
    """Online port of QuantConnect LEAN MesaAdaptiveMovingAverage."""
    def __init__(self, fast_limit: float, slow_limit: float):
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        self.samples = 0
        self.price: list[float] = []
        self.smooth_history: list[float] = []
        self.detrend_history: list[float] = []
        self.in_phase_history: list[float] = []
        self.quadrature_history: list[float] = []
        self.prev_period = self.prev_in_phase2 = self.prev_quadrature2 = 0.0
        self.prev_real = self.prev_imaginary = self.prev_smooth_period = 0.0
        self.prev_phase = self.prev_mama = self.fama = 0.0

    @staticmethod
    def at(history: list[float], index: int) -> float:
        return history[index] if index < len(history) else 0.0

    @staticmethod
    def add(history: list[float], value: float, size: int) -> None:
        history.insert(0, float(value))
        del history[size:]

    def update(self, high: float, low: float) -> dict[str, float]:
        self.samples += 1
        median = (high + low) / 2.0
        self.add(self.price, median, 13)
        empty = {name: math.nan for name in ("smooth", "detrender", "in_phase1", "quadrature1", "period", "phase", "alpha", "mama", "fama")}
        if len(self.price) < 13:
            return {"median_price": median, **empty}

        adjusted_period = 0.075 * self.prev_period + 0.54
        smooth = (4*self.price[0] + 3*self.price[1] + 2*self.price[2] + self.price[3]) / 10.0
        detrender = (0.0962*smooth + 0.5769*self.at(self.smooth_history,1) - 0.5769*self.at(self.smooth_history,3) - 0.0962*self.at(self.smooth_history,5)) * adjusted_period
        quadrature1 = (0.0962*detrender + 0.5769*self.at(self.detrend_history,1) - 0.5769*self.at(self.detrend_history,3) - 0.0962*self.at(self.detrend_history,5)) * adjusted_period
        in_phase1 = self.at(self.detrend_history,2)
        adjusted_in_phase = (0.0962*in_phase1 + 0.5769*self.at(self.in_phase_history,1) - 0.5769*self.at(self.in_phase_history,3) - 0.0962*self.at(self.in_phase_history,5)) * adjusted_period
        adjusted_quadrature = (0.0962*quadrature1 + 0.5769*self.at(self.quadrature_history,1) - 0.5769*self.at(self.quadrature_history,3) - 0.0962*self.at(self.quadrature_history,5)) * adjusted_period
        in_phase2 = 0.2*(in_phase1-adjusted_quadrature) + 0.8*self.prev_in_phase2
        quadrature2 = 0.2*(quadrature1+adjusted_in_phase) + 0.8*self.prev_quadrature2
        real = 0.2*(in_phase2*self.prev_in_phase2 + quadrature2*self.prev_quadrature2) + 0.8*self.prev_real
        imaginary = 0.2*(in_phase2*self.prev_quadrature2 - quadrature2*self.prev_in_phase2) + 0.8*self.prev_imaginary
        period = 0.0
        if imaginary != 0 and real != 0:
            angle = math.atan(imaginary / real) * 180.0 / math.pi
            period = 360.0 / angle if angle > 0 else 0.0
        if period > 1.5*self.prev_period: period = 1.5*self.prev_period
        if period < 0.67*self.prev_period: period = 0.67*self.prev_period
        period = min(50.0, max(6.0, period))
        period = 0.2*period + 0.8*self.prev_period
        smooth_period = 0.33*period + 0.67*self.prev_smooth_period
        phase = math.atan(quadrature1/in_phase1)*180.0/math.pi if in_phase1 != 0 else 0.0
        delta_phase = max(1.0, self.prev_phase-phase)
        alpha = max(self.slow_limit, self.fast_limit/delta_phase)
        mama = alpha*self.price[0] + (1-alpha)*self.prev_mama
        fama = 0.5*alpha*mama + (1-0.5*alpha)*self.fama
        self.prev_in_phase2, self.prev_quadrature2 = in_phase2, quadrature2
        self.prev_real, self.prev_imaginary = real, imaginary
        self.prev_period, self.prev_smooth_period, self.prev_phase = period, smooth_period, phase
        self.prev_mama, self.fama = mama, fama
        self.add(self.smooth_history,smooth,6);self.add(self.detrend_history,detrender,6)
        self.add(self.in_phase_history,in_phase1,6);self.add(self.quadrature_history,quadrature1,6)
        return {"median_price":median,"smooth":smooth,"detrender":detrender,"in_phase1":in_phase1,
                "quadrature1":quadrature1,"period":period,"phase":phase,"alpha":alpha,
                "mama":mama if self.samples>=P["warmup_bars"] else math.nan,
                "fama":fama if self.samples>=P["warmup_bars"] else math.nan}


def mama_crossover(high: pd.Series, low: pd.Series, valid: pd.Series) -> pd.DataFrame:
    names=("median_price","smooth","detrender","in_phase1","quadrature1","period","phase","alpha","mama","fama")
    values={name:np.full(len(high),np.nan) for name in names}
    entry_side=np.zeros(len(high),dtype=int);raw_side=np.zeros(len(high),dtype=int)
    state=LeanMama(P["fast_limit"],P["slow_limit"]);prior_relation=prior_raw=0;prior_median=math.nan
    for index,is_valid in enumerate(valid.to_numpy(bool)):
        if not is_valid:
            state=LeanMama(P["fast_limit"],P["slow_limit"]);prior_relation=prior_raw=0;prior_median=math.nan;continue
        row=state.update(float(high.iloc[index]),float(low.iloc[index]))
        for name in names: values[name][index]=row[name]
        if math.isfinite(row["mama"]) and math.isfinite(row["fama"]):
            relation=1 if row["mama"]>row["fama"] else -1 if row["mama"]<row["fama"] else 0
            entry_side[index]=relation if relation and relation!=prior_relation else 0;prior_relation=relation
        median=row["median_price"]
        if math.isfinite(prior_median):
            relation=1 if median>prior_median else -1 if median<prior_median else 0
            raw_side[index]=relation if relation and relation!=prior_raw else 0;prior_raw=relation
        prior_median=median
    return pd.DataFrame({**values,"entry_side":entry_side,"raw_median_side":raw_side},index=high.index)

def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    source=prepare(raw).reindex(pd.date_range(START,END,freq="1min",inclusive="left"));groups=source.groupby(source.index.floor("4h"),sort=True)
    bars=pd.DataFrame({"rows":groups.row_valid.sum(),"bar_open":groups.open.first(),"bar_high":groups.high.max(),"bar_low":groups.low.min(),"bar_close":groups.close.last(),"variation_component":groups.minute_sq_return.sum(min_count=240)})
    fields=["bar_open","bar_high","bar_low","bar_close"];bars["valid_bar"]=bars.rows.eq(240)&np.isfinite(bars[fields]).all(axis=1)&bars[fields].gt(0).all(axis=1)
    bars=bars.join(mama_crossover(bars.bar_high,bars.bar_low,bars.valid_bar));bars["variation"]=np.sqrt(bars.variation_component.rolling(P["variation_hours"]//4,min_periods=P["variation_hours"]//4).sum());bars["source_valid"]=bars.valid_bar&np.isfinite(bars[["median_price","smooth","detrender","in_phase1","quadrature1","period","phase","alpha","mama","fama","variation"]]).all(axis=1)&bars.variation.gt(0)
    panel=bars.reset_index(names="source_start");panel["feature_available_time"]=panel.source_start+pd.Timedelta("4h");panel["variation_rank"]=prior_rank(panel.variation.where(panel.source_valid));panel["eligible"]=panel.source_valid&panel.entry_side.ne(0)&panel.variation_rank.ge(P["variation_rank_min"])
    return panel.loc[:,PANEL_COLUMNS]

def active(panel: pd.DataFrame, control: str = "primary"):
    if control not in ("primary",*CONTROLS):raise ValueError(control)
    used=panel.copy();valid=used.source_valid.eq(True);variation=used.variation_rank.ge(P["variation_rank_min"]);side=pd.to_numeric(used.entry_side,errors="coerce").fillna(0).astype(int);state=valid&side.ne(0)&variation
    if control=="no_variation_gate":state=valid&side.ne(0)
    elif control=="one_bar_stale_cross":state=state.shift(1,fill_value=False);side=side.shift(1,fill_value=0)
    elif control=="direction_flip":side=-side
    elif control=="raw_median_change":side=pd.to_numeric(used.raw_median_side,errors="coerce").fillna(0).astype(int);state=valid&side.ne(0)&variation
    return state&side.ne(0),side,used

def build_clock(panel: pd.DataFrame, control: str = "primary") -> pd.DataFrame:
    activity, side, used = active(panel, control)
    rows = []
    reserved_until: pd.Timestamp | None = None
    for index in panel.index[activity]:
        decision = pd.Timestamp(panel.at[index, "feature_available_time"])
        entry = decision + pd.Timedelta(minutes=P["entry_delay_minutes"])
        exit_time = entry + pd.Timedelta(hours=P["hold_hours"])
        if reserved_until is not None and entry < reserved_until:
            continue
        split = next((name for name, (start, end) in SPLITS.items() if entry >= start and exit_time <= end), None)
        if split is None:
            continue
        reserved_until = exit_time
        rows.append(
            {"candidate": prereg.POLICY_ID, "control": control, "split": split,
             "source_start": pd.Timestamp(used.at[index, "source_start"]),
             "feature_available_time": decision, "entry_time": entry, "exit_time": exit_time,
             "side": int(side.at[index]),
             **{column: bool(used.at[index, column]) if column == "eligible" else float(used.at[index, column])
                for column in CLOCK_COLUMNS[8:]}}
        )
    return pd.DataFrame(rows, columns=CLOCK_COLUMNS)


def stats(clock: pd.DataFrame, split: str) -> dict[str, float | int]:
    selected = clock[clock.split.eq(split)]
    if selected.empty:
        return {"events": 0, "longs": 0, "shorts": 0, "minority_side_share": 0.0, "max_month_share": 0.0}
    longs = int(selected.side.eq(1).sum())
    shorts = int(selected.side.eq(-1).sum())
    months = pd.to_datetime(selected.entry_time, utc=True).dt.strftime("%Y-%m").value_counts()
    return {"events": len(selected), "longs": longs, "shorts": shorts,
            "minority_side_share": min(longs, shorts) / len(selected),
            "max_month_share": int(months.max()) / len(selected)}


def csv_gz(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    raw = frame.to_csv(index=False, float_format="%.12g", lineterminator="\n").encode()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=6, mtime=0) as output:
        output.write(raw)
    return buffer.getvalue()


def immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"refusing overwrite {path}")
    path.write_bytes(content)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def run() -> dict[str, Any]:
    if sha(prereg.DEFAULT_OUTPUT) != PREREG_SHA:
        raise RuntimeError("HVMAMA prereg drift")
    raw = load_source()
    panel = build_panel(raw)
    primary = build_clock(panel)
    controls = {name: build_clock(panel, name) for name in CONTROLS}
    splits = {name: primary[primary.split.eq(name)].copy() for name in SPLITS}
    immutable(PANEL, csv_gz(panel)); immutable(CLOCK, csv_gz(primary))
    for name, frame in controls.items(): immutable(CONTROL_DIR / f"{name}.csv.gz", csv_gz(frame))
    for name, frame in splits.items(): immutable(SPLIT_DIR / f"{name}.csv.gz", csv_gz(frame))
    source_core = {
        "protocol_version": "hvmama_c05_005_24_source_v1", "query": QUERY,
        "query_sha256": hashlib.sha256(QUERY.encode()).hexdigest(),
        "window": [START.isoformat(), END.isoformat()], "physical_rows": len(raw),
        "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
        "panel": {"path": str(PANEL), "sha256": sha(PANEL), "rows": len(panel),
                  "valid_rows": int(panel.source_valid.sum())},
        "outcomes_opened": False, "gross9_rows_opened": False, "no_imputation": True,
    }
    manifest = {**source_core, "manifest_hash": canonical_hash(source_core)}
    immutable(MANIFEST, json_bytes(manifest))
    support = {name: stats(primary, name) for name in SPLITS}
    checks = {key: passed for name, values in support.items() for key, passed in (
        (f"{name}_minimum_events", values["events"] >= GATES["minimum_events"][name]),
        (f"{name}_side_balance", values["minority_side_share"] >= GATES["minority_side_share_min"]),
        (f"{name}_month_concentration", values["max_month_share"] <= GATES["max_month_share"]),
    )}
    passed = all(checks.values())
    registration = json.loads(prereg.DEFAULT_OUTPUT.read_text())
    core = {
        "protocol_version": "hvmama_c05_005_24_source_support_v1", "policy_id": prereg.POLICY_ID,
        "preregistration": {"path": str(prereg.DEFAULT_OUTPUT), "sha256": PREREG_SHA,
                            "manifest_hash": registration["manifest_hash"]},
        "source_manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST),
                            "manifest_hash": manifest["manifest_hash"]},
        "completed_preentry_sources_opened": True, "candidate_incidence_opened": True,
        "postentry_return_pnl_execution_price_opened": False, "funding_values_opened": False,
        "gross9_rows_opened": False,
        "clock": {"path": str(CLOCK), "sha256": sha(CLOCK), "rows": len(primary)},
        "split_artifacts": {name: {"path": str(SPLIT_DIR / f"{name}.csv.gz"),
                                   "sha256": sha(SPLIT_DIR / f"{name}.csv.gz"), "rows": len(frame)}
                            for name, frame in splits.items()},
        "controls": {name: {"path": str(CONTROL_DIR / f"{name}.csv.gz"),
                            "sha256": sha(CONTROL_DIR / f"{name}.csv.gz"), "rows": len(frame),
                            "promotion_authorized": False} for name, frame in controls.items()},
        "support": support, "support_checks": checks, "support_passed": passed,
        "advance_to_gross9_novelty": passed, "advance_to_economic_outcomes": False,
        "decision": "pass_to_novelty" if passed else "terminal_source_support_reject",
    }
    result = {**core, "manifest_hash": canonical_hash(core)}
    immutable(RESULT, json_bytes(result))
    return result


if __name__ == "__main__":
    print(json.dumps({"passed": run()["support_passed"], "result": str(RESULT)}))
