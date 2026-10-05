"""SDS portfolio monitor.

Tracks IDX model portfolios against a benchmark with prices from Yahoo Finance.

    pip install -r requirements.txt
    streamlit run app.py

Portfolio files live in ./portfolios, one CSV per portfolio (the file name is the
portfolio name) with the columns Date, Symbol, Sector, Portfolio %. To record a
rebalance, append rows with the new date: returns are chain-linked across dates.
"""
from __future__ import annotations

import html
import inspect
import io
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except ImportError:  # reported in the UI instead of crashing
    yf = None

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
APP_TITLE = "SDS portfolio monitor"
APP_DIR = Path(__file__).resolve().parent
PORTFOLIO_DIR = APP_DIR / "portfolios"

WIB = timezone(timedelta(hours=7), "WIB")
EXCHANGE_SUFFIX = ".JK"          # IDX tickers on Yahoo Finance
DEFAULT_BENCHMARK = "^JKSE"      # IHSG (Jakarta Composite)
BENCHMARK_NAMES = {"^JKSE": "IHSG", "^JKLQ45": "LQ45", "^JKII": "JII"}
DEFAULT_RISK_FREE = 5.75         # BI Rate, % a year (held at 5.75% on 23 Sep 2026)
DEFAULT_CAPITAL = 100_000_000    # rupiah, for P&L and lot sizes
HISTORY_YEARS = 3                # price history downloaded for the backtest
PRICE_TTL = 15 * 60              # seconds before prices are downloaded again
TRADING_DAYS = 252
MIN_STAT_DAYS = 20               # below this, live statistics are mostly noise
SUSPECT_MOVE = 0.35              # daily moves beyond IDX price limits are likely data errors
LOT_SIZE = 100
CASH = "CASH"
BACKTEST_WINDOWS = {"6M": 126, "1Y": 252, "2Y": 504, "3Y": 756}
HORIZONS = ["1D", "1W", "MTD", "1M", "3M", "YTD", "Since start"]

# One color per portfolio. Each keeps at least 3:1 contrast on the light and the dark theme.
PORTFOLIO_COLORS = ["#2F6FDE", "#C9780F", "#9A5CDB", "#14897F", "#C2416B", "#6E8F1F", "#8C6A43", "#3F86A6"]
BENCH_COLOR = "#8B929A"
UP_COLOR = "#1F8A4C"
DOWN_COLOR = "#D1453B"
NEUTRAL_COLOR = "#9AA0A6"
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
MINUS = "\u2212"

CSS = """
.block-container {padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1480px;}
h1 {letter-spacing: -0.025em;}
h2, h3 {letter-spacing: -0.01em;}
.stTabs [data-baseweb="tab-list"] {gap: 1.8rem;}
.stTabs [data-baseweb="tab"] {padding-left: 0; padding-right: 0;}
.stTabs [data-baseweb="tab"] p {font-size: 0.98rem; font-weight: 650;}

/* Standings board: the benchmark sits inside the ranking as a par line */
.pm-board {font-variant-numeric: tabular-nums; margin-top: .2rem;}
.pm-row {display: grid; align-items: center; column-gap: 1rem; padding: .78rem .15rem;
  grid-template-columns: 1.7rem minmax(11.5rem, 1fr) minmax(6rem, 2.4fr) 7rem 6rem 7rem;
  border-bottom: 1px solid rgba(128, 128, 128, .2);}
.pm-row.pm-head {padding: 0 .15rem .5rem; font-size: .78rem; font-weight: 500; opacity: .66;
  border-bottom-color: rgba(128, 128, 128, .38);}
.pm-rank {font-size: 1.05rem; font-weight: 700; opacity: .45;}
.pm-name {display: flex; align-items: center; gap: .6rem; font-weight: 650; min-width: 0; overflow-wrap: anywhere;}
.pm-board {max-width: 1180px;}
.pm-name i {flex: none; width: .3rem; height: 1.35rem; border-radius: 1px;}
.pm-track {position: relative; height: .6rem; background: rgba(128, 128, 128, .13); border-radius: 1px;}
.pm-bar {position: absolute; top: 0; bottom: 0; border-radius: 1px;}
.pm-zero {position: absolute; top: -.3rem; bottom: -.3rem; width: 1px; background: rgba(128, 128, 128, .75);}
.pm-par {position: absolute; top: -.5rem; bottom: -.5rem; border-left: 1.5px dashed #8B929A;}
.pm-si {font-size: 1.42rem; font-weight: 750; text-align: right; letter-spacing: -.015em;}
.pm-num {font-size: .92rem; font-weight: 600; text-align: right;}
.pm-r {text-align: right;}
.pm-up {color: #1F8A4C;}
.pm-down {color: #D1453B;}
.pm-benchrow {border-top: 1.5px dashed rgba(139, 146, 154, .95); border-bottom: 1.5px dashed rgba(139, 146, 154, .95);
  margin-top: -1px;}
.pm-benchrow .pm-name, .pm-benchrow .pm-num {font-weight: 500; opacity: .8;}
.pm-benchrow .pm-si {font-size: 1.08rem; font-weight: 600; opacity: .8;}
.pm-benchrow .pm-name i {background: repeating-linear-gradient(180deg, #8B929A 0 3px, transparent 3px 6px);}
@media (max-width: 720px) {
  .pm-row {grid-template-columns: 1.4rem minmax(0, 1fr) 5.2rem 5.4rem; column-gap: .55rem;}
  .pm-row .pm-hide-sm {display: none;}
  .pm-si {font-size: 1.12rem;}
}
.pm-notes {margin: .25rem 0 0; padding-left: 1.05rem; line-height: 1.58;}
.pm-notes li {margin-bottom: .72rem; max-width: 64ch;}
.pm-notes li::marker {color: rgba(128, 128, 128, .8);}
.pm-notes b {font-weight: 700;}
"""


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _bad(x) -> bool:
    try:
        return x is None or not math.isfinite(float(x))
    except (TypeError, ValueError):
        return True


def fmt_pct(x, digits: int = 2, sign: bool = True) -> str:
    if _bad(x):
        return "—"
    if round(float(x), digits + 2) == 0:
        return f"{0:.{digits}%}"
    text = f"{x:+.{digits}%}" if sign else f"{x:.{digits}%}"
    return text.replace("-", MINUS)


def fmt_pp(x, digits: int = 2) -> str:
    if _bad(x):
        return "—"
    if round(float(x) * 100, digits) == 0:
        return f"{0:.{digits}f} pp"
    return f"{x * 100:+.{digits}f} pp".replace("-", MINUS)


def fmt_num(x, digits: int = 2) -> str:
    if _bad(x):
        return "—"
    return f"{x:,.{digits}f}".replace("-", MINUS)


def fmt_idr(x) -> str:
    if _bad(x):
        return "—"
    return f"{MINUS if x < 0 else ''}Rp {abs(x):,.0f}"


def fmt_date(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} {d:%b %Y}"


def esc(text) -> str:
    return html.escape(str(text))


def sign_class(x) -> str:
    if _bad(x) or x == 0:
        return ""
    return "pm-up" if x > 0 else "pm-down"


def sign_css(x) -> str:
    if _bad(x) or x == 0:
        return ""
    return f"color: {UP_COLOR}" if x > 0 else f"color: {DOWN_COLOR}"


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# ---------------------------------------------------------------------------
# Portfolio files
# ---------------------------------------------------------------------------
COLUMN_KEYS = {
    "date": {"date", "asof", "asofdate", "rebalancedate", "tanggal"},
    "symbol": {"symbol", "ticker", "code", "kode", "stock", "saham", "kodesaham"},
    "sector": {"sector", "sektor", "industry", "gicssector"},
    "weight": {"portfolio", "weight", "bobot", "allocation", "pct", "percent", "portfolioweight", "target"},
}
DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y", "%m/%d/%y")


@dataclass
class Book:
    """One portfolio: weight snapshots by date, plus anything worth flagging."""

    name: str
    holdings: pd.DataFrame  # date, symbol, sector, weight; each date sums to 1
    raw_sums: dict
    notes: list = field(default_factory=list)

    def snapshots(self) -> dict:
        return {d: g.set_index("symbol")["weight"] for d, g in self.holdings.groupby("date", sort=True)}

    def latest(self) -> pd.Series:
        last = self.holdings["date"].max()
        return self.holdings.loc[self.holdings["date"] == last].set_index("symbol")["weight"]


def _col_key(name) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def _parse_weights(s: pd.Series) -> pd.Series:
    txt = s.astype(str).str.strip().str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
    had_pct = bool(txt.str.contains("%", regex=False).any())
    txt = txt.str.replace("%", "", regex=False)
    if txt.str.contains(",", regex=False).any() and not txt.str.contains(".", regex=False).any():
        txt = txt.str.replace(",", ".", regex=False)  # decimal comma: 7,20
    else:
        txt = txt.str.replace(",", "", regex=False)  # thousands separator
    w = pd.to_numeric(txt, errors="coerce")
    if had_pct or w.sum(skipna=True) > 1.5:
        w = w / 100.0
    return w


def _parse_dates(s: pd.Series) -> pd.Series:
    """Read dates in whichever format fits every row. When both month/day and day/month
    fit (10/1/2026), prefer the reading with no future dates and no weekends, since
    holdings are dated on trading days. Remaining ties go to month/day/year."""
    txt = s.astype(str).str.strip()
    today = pd.Timestamp(datetime.now(WIB).date())
    fits = []
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(txt, format=fmt, errors="coerce")
        if parsed.notna().all():
            fits.append(parsed.dt.normalize())
    if not fits:
        return pd.to_datetime(txt, errors="coerce", format="mixed").dt.normalize()
    return min(fits, key=lambda d: (int((d > today).sum()), int((d.dt.dayofweek >= 5).sum())))


def _read_table(data: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("the file is not readable as text")
    first = text.splitlines()[0] if text.strip() else ""
    sep = ";" if first.count(";") > first.count(",") else ("\t" if "\t" in first and "," not in first else ",")
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str).dropna(how="all")


def parse_book(name: str, data: bytes) -> Book:
    raw = _read_table(data)
    cols = {}
    for c in raw.columns:
        key = _col_key(c)
        for target, keys in COLUMN_KEYS.items():
            if target not in cols and key in keys:
                cols[target] = c
    missing = [t for t in ("symbol", "weight") if t not in cols]
    if missing:
        raise ValueError(f"missing the {' and '.join(missing)} column (expected Date, Symbol, Sector, Portfolio %)")

    notes = []
    df = pd.DataFrame({"symbol": raw[cols["symbol"]].astype(str).str.strip().str.upper()})
    df["symbol"] = df["symbol"].str.replace(r"\.JK$", "", regex=True)
    df["weight"] = _parse_weights(raw[cols["weight"]])
    if "sector" in cols:
        df["sector"] = raw[cols["sector"]].fillna("Unclassified").astype(str).str.strip()
    else:
        df["sector"] = "Unclassified"
    if "date" in cols:
        df["date"] = _parse_dates(raw[cols["date"]])
    else:
        df["date"] = pd.Timestamp(datetime.now(WIB).date())
        notes.append("No Date column, so the holdings are treated as starting today.")

    bad = df["date"].isna() | df["weight"].isna() | df["symbol"].isin(["", "NAN", "NONE"])
    if bad.any():
        notes.append(f"Skipped {plural(int(bad.sum()), 'row')} with a missing date, symbol or weight.")
    df = df.loc[~bad & (df["weight"] > 0)]
    if df.empty:
        raise ValueError("no rows with a symbol and a positive weight")

    df = df.groupby(["date", "symbol"], as_index=False).agg(sector=("sector", "first"), weight=("weight", "sum"))
    parts, raw_sums = [], {}
    for d, g in df.groupby("date", sort=True):
        total = float(g["weight"].sum())
        raw_sums[d] = total
        if total < 0.995:
            cash = pd.DataFrame({"date": [d], "symbol": [CASH], "sector": ["Cash"], "weight": [1 - total]})
            g = pd.concat([g, cash], ignore_index=True)
            notes.append(f"{fmt_date(d)}: weights add up to {total:.2%}; the other {1 - total:.2%} is treated as cash.")
        else:
            if abs(total - 1) > 0.005:
                notes.append(f"{fmt_date(d)}: weights add up to {total:.2%}, so they were scaled to 100%.")
            g = g.assign(weight=g["weight"] / total)
        parts.append(g)
    return Book(name, pd.concat(parts, ignore_index=True), raw_sums, notes)


def load_books(uploads) -> tuple[dict, list]:
    sources = {}
    if PORTFOLIO_DIR.is_dir():
        for path in sorted(PORTFOLIO_DIR.glob("*.csv")):
            sources[path.stem] = path.read_bytes()
    for f in uploads or []:
        sources[Path(f.name).stem] = f.getvalue()  # an upload with the same name replaces the saved file
    books, errors = {}, []
    for name, data in sources.items():
        try:
            books[name] = parse_book(name, data)
        except Exception as exc:  # one bad file shouldn't take the dashboard down
            errors.append(f"{name}: {exc}")
    return books, errors


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def yahoo_symbol(code: str) -> str | None:
    code = code.strip().upper()
    if code == CASH:
        return None
    if code.startswith("^") or "." in code or "=" in code:
        return code
    return code + EXCHANGE_SUFFIX


def _price_field(raw: pd.DataFrame, name: str, symbols: tuple) -> pd.DataFrame | None:
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        for level in range(cols.nlevels):
            if name in set(cols.get_level_values(level)):
                return raw.xs(name, axis=1, level=level)
        return None
    if name in cols and len(symbols) == 1:
        return raw[[name]].set_axis([symbols[0]], axis=1)
    return None


def _tidy(df: pd.DataFrame, symbols: tuple) -> pd.DataFrame:
    df = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Jakarta").tz_localize(None)
    df.index = idx.normalize()
    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.where(df > 0)  # zero or negative prices are data errors
    return df.reindex(columns=list(symbols))


@st.cache_data(ttl=PRICE_TTL, show_spinner=False)
def download_prices(symbols: tuple, start: str) -> dict:
    if yf is None:
        raise RuntimeError("The yfinance package isn't installed. Run `pip install -r requirements.txt`.")
    params = inspect.signature(yf.download).parameters
    kwargs = dict(start=start, auto_adjust=False, actions=False, progress=False, threads=True, group_by="column")
    if "multi_level_index" in params:
        kwargs["multi_level_index"] = True
    if "timeout" in params:
        kwargs["timeout"] = 30
    raw, error = None, "no data returned"
    for attempt in range(3):
        try:
            raw = yf.download(list(symbols), **kwargs)
        except Exception as exc:
            raw, error = None, f"{type(exc).__name__}: {exc}"
        if raw is not None and not raw.empty:
            break
        time.sleep(2 * (attempt + 1))
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no prices ({error}).")
    close = _price_field(raw, "Close", symbols)
    if close is None:
        raise RuntimeError("Yahoo Finance returned no closing prices.")
    adj = _price_field(raw, "Adj Close", symbols)
    if adj is None:
        adj = close
    return {"adj": _tidy(adj, symbols), "close": _tidy(close, symbols), "fetched_at": datetime.now(WIB)}


@dataclass
class Prices:
    adj: pd.DataFrame          # dividend-adjusted closes (total return), columns are portfolio codes
    close: pd.DataFrame        # actual closes, for display
    bench: pd.Series | None    # benchmark level
    first_date: pd.Series      # first day each code has a price
    missing: list              # codes Yahoo returned nothing for
    yahoo: dict                # code -> Yahoo symbol
    fetched_at: datetime


def build_prices(raw: dict, codes: list, bench_symbol: str) -> Prices:
    ymap = {c: yahoo_symbol(c) for c in codes if c != CASH}
    adj_raw, close_raw = raw["adj"], raw["close"]
    stock_cols = [s for s in ymap.values() if s in adj_raw.columns]
    counts = adj_raw[stock_cols].notna().sum(axis=1) if stock_cols else pd.Series(0, index=adj_raw.index)
    # Trading calendar: days on which at least half the stocks traded (drops stray partial rows)
    days = counts.index[counts >= 0.5 * counts.max()] if counts.max() > 0 else adj_raw.index
    to_code = {v: k for k, v in ymap.items()}
    adj = adj_raw.reindex(index=days, columns=list(ymap.values())).rename(columns=to_code)
    close = close_raw.reindex(index=days, columns=list(ymap.values())).rename(columns=to_code)
    first_date = adj.apply(lambda s: s.first_valid_index())
    missing = [c for c in adj.columns if adj[c].notna().sum() == 0]
    adj, close = adj.ffill(), close.ffill()
    if CASH in codes:
        adj[CASH] = 1.0
        close[CASH] = 1.0
    bench = None
    if bench_symbol in adj_raw.columns:
        b = adj_raw[bench_symbol].reindex(days).ffill()
        if b.notna().sum() > 1:
            bench = b
    return Prices(adj, close, bench, first_date, missing, ymap, raw["fetched_at"])


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
@dataclass
class Live:
    nav: pd.Series   # value of 1 rupiah invested at the first anchor close
    periods: list    # [{"date", "anchor", "weights"}], one per holding period
    excluded: list   # codes left out because they had no price at entry


def run_live(snapshots: dict, adj: pd.DataFrame, start: pd.Timestamp, include_start_day: bool) -> Live | None:
    """Buy-and-hold between snapshot dates, chain-linked across rebalances.

    Holdings dated d are bought at the close of d (or of the day before when
    include_start_day is on) and earn returns from the next session.
    """
    days = adj.index
    dates = sorted(snapshots)
    in_force = [d for d in dates if d <= start]
    first = in_force[-1] if in_force else dates[0]
    plan = [(start, snapshots[first])] + [(d, snapshots[d]) for d in dates if d > start]

    def anchor(d):
        pos = days.searchsorted(d, side="left" if include_start_day else "right") - 1
        return int(pos) if pos >= 0 else None

    anchors = [anchor(d) for d, _ in plan]
    segments, periods, excluded = [], [], []
    level = 1.0
    for i, ((d, w), a) in enumerate(zip(plan, anchors)):
        if a is None:
            continue
        later = [x for x in anchors[i + 1:] if x is not None]
        end = later[0] if later else len(days) - 1
        if end < a or (end == a and later):
            continue  # superseded by a snapshot that starts on the same close
        px = adj.reindex(columns=w.index).iloc[a:end + 1]
        base = px.iloc[0]
        ok = base.notna() & (base > 0)
        excluded += list(w.index[~ok])
        if not ok.any():
            continue
        w_ok = w[ok] / w[ok].sum()
        seg = px.loc[:, w_ok.index].div(base[w_ok.index]).mul(w_ok).sum(axis=1) * level
        if segments:
            seg = seg.iloc[1:]  # the anchor close is already the last point of the previous period
        if len(seg):
            segments.append(seg)
            level = float(seg.iloc[-1])
        periods.append({"date": d, "anchor": days[a], "weights": w_ok})
    if not segments:
        return None
    return Live(pd.concat(segments), periods, sorted(set(excluded)))


def positions(live: Live, prices: Prices, sectors: dict, capital: float) -> pd.DataFrame:
    """Per-stock view of the current holding period."""
    p = live.periods[-1]
    w, a = p["weights"], p["anchor"]
    adj, close = prices.adj, prices.close
    days = adj.index
    t, ia, it = days[-1], days.get_loc(a), len(days) - 1
    cols = list(w.index)
    rel = adj.loc[t, cols] / adj.loc[a, cols]
    ret = rel - 1
    cur_w = w * rel / (w * rel).sum()
    if it > ia:
        prev = days[it - 1]
        rel_prev = adj.loc[prev, cols] / adj.loc[a, cols]
        w_prev = w * rel_prev / (w * rel_prev).sum()
        r1 = adj.loc[t, cols] / adj.loc[prev, cols] - 1
        c1 = w_prev * r1
    else:
        r1 = c1 = pd.Series(np.nan, index=cols)
    cap_a = capital * float(live.nav.get(a, 1.0))
    entry, last = close.loc[a, cols], close.loc[t, cols]
    lots = np.floor(cap_a * w / (entry * LOT_SIZE))
    lots[[c == CASH for c in cols]] = np.nan
    return pd.DataFrame({
        "Symbol": cols,
        "Sector": [sectors.get(c, "") for c in cols],
        "Target weight": w.values,
        "Current weight": cur_w.values,
        "Entry price": entry.values,
        "Last price": last.values,
        "Return": ret.values,
        "Contribution": (w * ret).values,
        "Today": r1.values,
        "Today's contribution": c1.values,
        "P&L (Rp)": (cap_a * w * ret).values,
        "Lots": lots.values,
    }).sort_values("Contribution", ascending=False, ignore_index=True)


def period_returns(nav: pd.Series) -> dict:
    out = {h: np.nan for h in HORIZONS}
    if nav is None or len(nav) < 2:
        return out
    t, last = nav.index[-1], float(nav.iloc[-1])

    def since(ref):
        sub = nav.loc[:ref]
        return last / float(sub.iloc[-1]) - 1 if len(sub) else np.nan

    out["1D"] = last / float(nav.iloc[-2]) - 1
    out["1W"] = since(t - pd.Timedelta(days=7))
    out["MTD"] = since(t.replace(day=1) - pd.Timedelta(days=1))
    out["1M"] = since(t - pd.DateOffset(months=1))
    out["3M"] = since(t - pd.DateOffset(months=3))
    out["YTD"] = since(pd.Timestamp(t.year - 1, 12, 31))
    out["Since start"] = last / float(nav.iloc[0]) - 1
    return out


def simulate(w: pd.Series, rets: pd.DataFrame) -> pd.Series:
    """Daily returns of fixed weights. Names without a price that day are left
    out and the rest scaled up, so recent listings don't count as cash."""
    r = rets.reindex(columns=w.index)
    wt = r.notna().astype(float).mul(w, axis=1)
    denom = wt.sum(axis=1)
    return ((r.fillna(0.0) * wt).sum(axis=1) / denom.where(denom > 0)).dropna()


def risk_stats(r: pd.Series, b: pd.Series | None, rf: float) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 2:
        return {}
    rf_d = (1 + rf) ** (1 / TRADING_DAYS) - 1
    nav = (1 + r).cumprod()
    total = float(nav.iloc[-1] - 1)
    cagr = (1 + total) ** (TRADING_DAYS / n) - 1 if total > -1 else np.nan
    sd = float(r.std(ddof=1))
    ex = r - rf_d
    downside = math.sqrt(float((np.minimum(ex, 0) ** 2).mean())) * math.sqrt(TRADING_DAYS)
    path = np.r_[1.0, nav.to_numpy()]
    mdd = float((path / np.maximum.accumulate(path) - 1).min())
    q05 = float(np.quantile(r, 0.05))
    s = {
        "n": n, "total": total, "cagr": cagr, "vol": sd * math.sqrt(TRADING_DAYS),
        "sharpe": ex.mean() / sd * math.sqrt(TRADING_DAYS) if sd > 0 else np.nan,
        "sortino": ex.mean() * TRADING_DAYS / downside if downside > 0 else np.nan,
        "mdd": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "var95": -q05, "cvar95": -float(r[r <= q05].mean()),
        "hit": float((r > 0).mean()), "best": float(r.max()), "worst": float(r.min()),
    }
    if b is not None:
        j = pd.concat({"r": r, "b": b}, axis=1).dropna()
        if len(j) > 5 and j["b"].var() > 0:
            beta = j["r"].cov(j["b"]) / j["b"].var()
            active = j["r"] - j["b"]
            te = float(active.std(ddof=1)) * math.sqrt(TRADING_DAYS)
            up, dn = j["b"] > 0, j["b"] < 0
            s.update(
                beta=beta, corr=j["r"].corr(j["b"]), te=te,
                ir=active.mean() * TRADING_DAYS / te if te > 0 else np.nan,
                alpha=((j["r"] - rf_d).mean() - beta * (j["b"] - rf_d).mean()) * TRADING_DAYS,
                upcap=j["r"][up].mean() / j["b"][up].mean() if up.any() else np.nan,
                dncap=j["r"][dn].mean() / j["b"][dn].mean() if dn.any() else np.nan,
                batting=float((active > 0).mean()),
            )
    return s


def risk_shares(w: pd.Series, rets: pd.DataFrame) -> pd.Series | None:
    """Each holding's share of portfolio variance (sums to 1)."""
    r = rets.reindex(columns=w.index)
    has_data = r.notna().sum() >= 40
    w = w[has_data]
    if w.empty:
        return None
    w = w / w.sum()
    cov = r[w.index].cov(min_periods=40).fillna(0.0) * TRADING_DAYS
    marginal = cov.to_numpy() @ w.to_numpy()
    total = float(w.to_numpy() @ marginal)
    if total <= 0:
        return None
    return pd.Series(w.to_numpy() * marginal / total, index=w.index)


def gap_breakdown(wa: pd.Series, wb: pd.Series, stock_ret: pd.Series, sectors: dict) -> pd.DataFrame:
    """Return gap between two buy-and-hold portfolios = sum of (weight difference x stock return)."""
    idx = wa.index.union(wb.index)
    dw = wa.reindex(idx, fill_value=0.0) - wb.reindex(idx, fill_value=0.0)
    r = stock_ret.reindex(idx)
    out = pd.DataFrame({
        "Symbol": list(idx),
        "Sector": [sectors.get(s, "") for s in idx],
        "Weight difference": dw.to_numpy(),
        "Return": r.to_numpy(),
        "Effect": (dw * r.fillna(0.0)).to_numpy(),
    })
    return out.loc[out["Effect"].abs().sort_values(ascending=False).index].reset_index(drop=True)


def overlap_matrix(weights: dict) -> pd.DataFrame:
    names = list(weights)
    m = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            idx = weights[a].index.union(weights[b].index)
            m.loc[a, b] = float(np.minimum(weights[a].reindex(idx, fill_value=0), weights[b].reindex(idx, fill_value=0)).sum())
    return m


def concentration(w: pd.Series, sectors: dict) -> dict:
    stocks = w.drop(CASH, errors="ignore").sort_values(ascending=False)
    stocks = stocks / stocks.sum()
    by_sector = stocks.groupby(lambda s: sectors.get(s, "")).sum().sort_values(ascending=False)
    return {
        "Stocks": len(stocks),
        "Largest position": stocks.iloc[0],
        "Top 5": stocks.iloc[:5].sum(),
        "Top 10": stocks.iloc[:10].sum(),
        "Effective stocks": 1 / float((stocks ** 2).sum()),
        "Largest sector": f"{by_sector.index[0]} ({by_sector.iloc[0]:.0%})",
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _layout(fig: go.Figure, height: int, legend: bool = True, hovermode: str = "x unified") -> go.Figure:
    fig.update_layout(
        height=height, hovermode=hovermode, showlegend=legend,
        margin=dict(l=4, r=14, t=40 if legend else 14, b=4),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
    )
    return fig


def _date_axis(fig: go.Figure, index) -> None:
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], showgrid=False)
    if len(idx) <= 12:
        fig.update_xaxes(tickvals=list(idx), tickformat="%-d %b")
    elif (idx[-1] - idx[0]).days <= 150:
        fig.update_xaxes(tickformat="%-d %b")
    else:
        fig.update_xaxes(tickformat="%b %Y")


def _decimals(values: np.ndarray) -> int:
    """Decimals needed (in percent units) so tick labels at ~5 ticks don't repeat."""
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return 1
    step = max(float(vals.max() - vals.min()), 1e-6) / 5 * 100
    return int(min(2, max(0, -math.floor(math.log10(step)))))


def fig_lines(series: dict, colors: dict, bench: pd.Series | None = None, bench_label: str = "",
              height: int = 380, signed: bool = True, pp: bool = False, zero: bool = True) -> go.Figure:
    """Lines for return-like series (fractions). pp=True shows percentage points."""
    fig = go.Figure()
    union = pd.DatetimeIndex([])
    items = list(series.items()) + ([(bench_label, bench)] if bench is not None else [])
    allv = np.concatenate([s.to_numpy(dtype=float) for _, s in items]) if items else np.array([])
    d = _decimals(allv)
    sign = "+" if signed else ""
    scale, unit = (100.0, " pp") if pp else (1.0, "")
    tick = f"{sign}.{d}f" if pp else f"{sign}.{d}%"
    hover = f"{sign}.{min(d + 1, 2)}f" if pp else f"{sign}.2%"
    for name, s in items:
        is_bench = bench is not None and s is bench
        union = union.union(s.index)
        fig.add_trace(go.Scatter(
            x=s.index, y=s.to_numpy(dtype=float) * scale, name=name, mode="lines+markers" if len(s) <= 30 else "lines",
            line=dict(color=BENCH_COLOR if is_bench else colors.get(name, NEUTRAL_COLOR),
                      width=1.9 if is_bench else 2.4, dash="dash" if is_bench else "solid"),
            marker=dict(size=5 if is_bench else 6),
            hovertemplate=f"{esc(name)}: %{{y:{hover}}}{unit}<extra></extra>",
        ))
    if zero:
        fig.add_hline(y=0, line_width=1, line_color="rgba(128,128,128,.55)")
    fig.update_yaxes(tickformat=tick, ticksuffix=unit)
    _date_axis(fig, union)
    return _layout(fig, height)


def fig_signed_bars(labels, values, height: int | None = None, hover_label: str = "") -> go.Figure:
    df = pd.DataFrame({"label": list(labels), "v": list(values)}).dropna().sort_values("v")
    x = df["v"].to_numpy() * 100
    lo, hi = min(0.0, x.min(initial=0.0)), max(0.0, x.max(initial=0.0))
    pad = (hi - lo) * 0.22 + 0.02
    fig = go.Figure(go.Bar(
        x=x, y=df["label"], orientation="h",
        marker_color=[UP_COLOR if v >= 0 else DOWN_COLOR for v in x],
        text=[f"{v:+.2f}".replace("-", MINUS) if abs(v) >= 0.005 else "" for v in x],
        textposition="outside", cliponaxis=False,
        hovertemplate=f"%{{y}}: %{{x:+.2f}} pp{hover_label}<extra></extra>",
    ))
    fig.update_xaxes(ticksuffix=" pp", range=[lo - (pad if lo < 0 else pad * 0.15), hi + (pad if hi > 0 else pad * 0.15)],
                     zeroline=True, zerolinewidth=1, zerolinecolor="rgba(128,128,128,.7)", showgrid=True)
    fig.update_yaxes(type="category", automargin=True, showgrid=False)
    return _layout(fig, height or max(200, 24 * len(df) + 50), legend=False, hovermode="closest")


def fig_treemap(pos: pd.DataFrame) -> go.Figure:
    df = pos.loc[pos["Symbol"] != CASH]
    sec = df.groupby("Sector").agg(c=("Contribution", "sum"), tw=("Target weight", "sum"))
    sec_ret = sec["c"] / sec["tw"]
    m = max(0.015, float(df["Return"].abs().quantile(0.9)))
    labels = list(sec.index) + list(df["Symbol"])
    color = list(sec_ret) + list(df["Return"])
    weight = list(df.groupby("Sector")["Current weight"].sum().reindex(sec.index)) + list(df["Current weight"])
    custom = np.c_[[fmt_pct(v) for v in color], [fmt_pct(v, 1, False) for v in weight]]
    fig = go.Figure(go.Treemap(
        ids=[f"s:{s}" for s in sec.index] + [f"t:{s}" for s in df["Symbol"]],
        labels=labels, parents=[""] * len(sec) + [f"s:{s}" for s in df["Sector"]],
        values=[0] * len(sec) + list(df["Current weight"]), branchvalues="remainder",
        marker=dict(colors=color, colorscale=[[0, DOWN_COLOR], [0.5, NEUTRAL_COLOR], [1, UP_COLOR]],
                    cmin=-m, cmax=m, line=dict(width=1, color="rgba(128,128,128,.35)")),
        customdata=custom, texttemplate="<b>%{label}</b><br>%{customdata[0]}",
        hovertemplate="<b>%{label}</b><br>Return %{customdata[0]}<br>Weight %{customdata[1]}<extra></extra>",
        tiling=dict(pad=2), pathbar=dict(visible=False), sort=True,
    ))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=6, b=0))
    return fig


def fig_grouped_bars(mat: pd.DataFrame, colors: dict, suffix: str = "%", hover_digits: int = 1) -> go.Figure:
    """Horizontal grouped bars: rows of mat on the y axis, one trace per column."""
    order = mat.mean(axis=1).sort_values().index
    fig = go.Figure()
    for name in reversed(mat.columns):  # reversed so the first column draws on top in each group
        fig.add_trace(go.Bar(
            y=list(order), x=mat.loc[order, name].to_numpy() * 100, name=name, orientation="h",
            marker_color=colors.get(name, NEUTRAL_COLOR),
            hovertemplate=f"{esc(name)}: %{{x:.{hover_digits}f}}{suffix}<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06, legend_traceorder="reversed")
    fig.update_xaxes(ticksuffix=suffix, showgrid=True)
    fig.update_yaxes(automargin=True, showgrid=False)
    return _layout(fig, 70 + len(order) * (len(mat.columns) * 8 + 16), hovermode="y unified")


def fig_risk_return(stats: dict, colors: dict, bench_stats: dict | None, bench_label: str) -> go.Figure:
    fig = go.Figure()
    xs, ys = [], []
    for name, s in stats.items():
        if not s:
            continue
        xs.append(s["vol"])
        ys.append(s["cagr"])
        fig.add_trace(go.Scatter(
            x=[s["vol"]], y=[s["cagr"]], name=name, mode="markers",
            marker=dict(size=15, color=colors.get(name, NEUTRAL_COLOR), line=dict(width=1.5, color="rgba(255,255,255,.85)")),
            hovertemplate=(f"<b>{esc(name)}</b><br>Return %{{y:+.1%}} a year<br>Volatility %{{x:.1%}}"
                           f"<br>Sharpe {s['sharpe']:.2f}<extra></extra>"),
        ))
    if bench_stats:
        xs.append(bench_stats["vol"])
        ys.append(bench_stats["cagr"])
        fig.add_trace(go.Scatter(
            x=[bench_stats["vol"]], y=[bench_stats["cagr"]], name=bench_label, mode="markers",
            marker=dict(size=13, symbol="diamond", color=BENCH_COLOR),
            hovertemplate=f"<b>{esc(bench_label)}</b><br>Return %{{y:+.1%}} a year<br>Volatility %{{x:.1%}}<extra></extra>",
        ))
    if xs:
        xpad = (max(xs) - min(xs)) * 0.25 + 0.01
        ypad = (max(ys) - min(ys)) * 0.25 + 0.01
        fig.update_xaxes(range=[min(xs) - xpad, max(xs) + xpad])
        fig.update_yaxes(range=[min(ys) - ypad, max(ys) + ypad])
    fig.update_xaxes(title_text="Volatility, a year", tickformat=".0%")
    fig.update_yaxes(title_text="Return, a year", tickformat="+.0%")
    return _layout(fig, 400, hovermode="closest")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def show_table(data, rows: int, cap: int | None = None, **kwargs) -> None:
    """Fit short tables to their content; long ones scroll inside `cap` pixels."""
    height = "content" if cap is None or rows <= 12 else cap
    st.dataframe(data, width="stretch", height=height, placeholder="—", **kwargs)


def returns_styler(df: pd.DataFrame, cols: list, digits: int = 2):
    return (df.style.format({c: (lambda v, d=digits: fmt_pct(v, d)) for c in cols}, na_rep="—")
            .map(sign_css, subset=cols))


def rank_tint(rank) -> str:
    """Green tint for first and second place: 'better', independent of the value's sign."""
    if _bad(rank):
        return ""
    alpha = {1: 0.24, 2: 0.11}.get(int(rank), 0)
    return f"background-color: rgba(31, 138, 76, {alpha})" if alpha else ""


def row_formats(styler, kinds: dict):
    fmts = {"pct": lambda v: fmt_pct(v, 2), "pct1": lambda v: fmt_pct(v, 1), "pct0": lambda v: fmt_pct(v, 1, False),
            "cap": lambda v: fmt_pct(v, 0, False), "num": lambda v: fmt_num(v, 2), "rank": lambda v: fmt_num(v, 0),
            "avg": lambda v: fmt_num(v, 1)}
    for row, kind in kinds.items():
        styler = styler.format(fmts[kind], subset=pd.IndexSlice[[row], :], na_rep="—")
    return styler


# ---------------------------------------------------------------------------
# Summary pieces (HTML)
# ---------------------------------------------------------------------------
def standings_html(rows: list, bench: dict | None, colors: dict, bench_label: str, day_label: str) -> str:
    values = [r["si"] for r in rows] + ([bench["si"]] if bench else [])
    lo, hi = min(0.0, min(values)), max(0.0, max(values))
    span = (hi - lo) or 1.0

    def pos(v):
        return (v - lo) / span * 100

    zero = pos(0.0)
    par = pos(bench["si"]) if bench else None

    def track(v, color):
        left, right = sorted((zero, pos(v)))
        par_line = f'<div class="pm-par" style="left:{par:.2f}%"></div>' if par is not None else ""
        return (f'<div class="pm-track pm-hide-sm"><div class="pm-zero" style="left:{zero:.2f}%"></div>'
                f'<div class="pm-bar" style="left:{left:.2f}%;width:{max(right - left, 0.5):.2f}%;background:{color}"></div>'
                f"{par_line}</div>")

    rel_head = f'<span class="pm-r">vs {esc(bench_label)}</span>' if bench else '<span></span>'
    out = ['<div class="pm-board">',
           f'<div class="pm-row pm-head"><span></span><span>Portfolio</span><span class="pm-hide-sm"></span>'
           f'<span class="pm-r">Since start</span><span class="pm-r pm-hide-sm">{esc(day_label)}</span>{rel_head}</div>']
    bench_row = ""
    if bench:
        bench_row = (f'<div class="pm-row pm-benchrow"><span></span><span class="pm-name"><i></i>{esc(bench_label)}</span>'
                     f'{track(bench["si"], BENCH_COLOR)}<span class="pm-si">{fmt_pct(bench["si"])}</span>'
                     f'<span class="pm-num pm-hide-sm">{fmt_pct(bench["d1"])}</span><span class="pm-num">—</span></div>')
    placed = False
    for rank, r in enumerate(rows, start=1):
        if bench and not placed and bench["si"] > r["si"]:
            out.append(bench_row)
            placed = True
        rel = (f'<span class="pm-num {sign_class(r["rel"])}">{fmt_pp(r["rel"])}</span>' if bench else "<span></span>")
        out.append(
            f'<div class="pm-row"><span class="pm-rank">{rank}</span>'
            f'<span class="pm-name"><i style="background:{colors.get(r["name"], NEUTRAL_COLOR)}"></i>{esc(r["name"])}</span>'
            f'{track(r["si"], colors.get(r["name"], NEUTRAL_COLOR))}'
            f'<span class="pm-si">{fmt_pct(r["si"])}</span>'
            f'<span class="pm-num pm-hide-sm {sign_class(r["d1"])}">{fmt_pct(r["d1"])}</span>{rel}</div>'
        )
    if bench and not placed:
        out.append(bench_row)
    out.append("</div>")
    return "".join(out)


def notes_html(notes: list) -> str:
    return '<ul class="pm-notes">' + "".join(f"<li>{n}</li>" for n in notes) + "</ul>"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
st.html(f"<style>{CSS}</style>")

now = datetime.now(WIB)
today = pd.Timestamp(now.date())

with st.sidebar:
    st.subheader("Portfolios")
    pick_box = st.container()
    uploads = st.file_uploader(
        "Add or replace files", type=["csv"], accept_multiple_files=True,
        help="Same layout as the files in the portfolios folder: Date, Symbol, Sector, Portfolio %. "
             "A file with the same name as a saved one replaces it for this session.",
    )

books, load_errors = load_books(uploads)
if not books:
    st.title(APP_TITLE)
    st.info("No portfolio files yet. Put CSV files with Date, Symbol, Sector and Portfolio % columns in the "
            "`portfolios` folder next to app.py, or add them in the sidebar.")
    for err in load_errors:
        st.warning(f"Couldn't read {err}")
    st.stop()

names_all = list(books)
colors = {n: PORTFOLIO_COLORS[i % len(PORTFOLIO_COLORS)] for i, n in enumerate(names_all)}
with pick_box:
    selected = st.multiselect("Show", names_all, default=names_all)

# Latest of the portfolios' first dates: from then on every portfolio has real holdings.
first_snapshot = max(b.holdings["date"].min() for b in books.values())
with st.sidebar:
    st.subheader("Tracking")
    start_date = st.date_input(
        "Track from", value=min(first_snapshot, today).date(),
        min_value=(today - pd.DateOffset(years=HISTORY_YEARS)).date(), max_value=today.date(), format="YYYY-MM-DD",
        help="Defaults to the date in the portfolio files. Dates that could be month/day or day/month are read "
             "the way that lands on a trading day, so 10/1/2026 is 1 October 2026. "
             "Returns are measured from this day's close. Before a portfolio's first file date, its first "
             "holdings are applied backwards.",
    )
    include_start = st.toggle(
        "Include the start day's move", value=False,
        help="Off: returns run from the close on the start date, as if bought at that close. "
             "On: returns run from the previous close, so the start day's own move counts.",
    )
    bench_symbol = st.text_input("Benchmark", value=DEFAULT_BENCHMARK,
                                 help="Yahoo Finance symbol. ^JKSE is IHSG, ^JKLQ45 is LQ45.").strip().upper() or DEFAULT_BENCHMARK
    capital = st.number_input("Capital (Rp)", min_value=0, value=DEFAULT_CAPITAL, step=10_000_000,
                              help="Used for P&L in rupiah and for lot sizes (100 shares a lot).")
    st.subheader("Backtest")
    window = st.segmented_control("Window", list(BACKTEST_WINDOWS), default="1Y", required=True,
                                  help="How much price history the backtest of current weights uses.") or "1Y"
    rf = st.number_input("Risk-free rate (% a year)", min_value=0.0, max_value=25.0, value=DEFAULT_RISK_FREE, step=0.25,
                         help="BI Rate by default. Used for the Sharpe and Sortino ratios.") / 100
    st.caption("Returns include dividends (Yahoo's adjusted closes); prices shown are actual closes. "
               "Prices refresh every 15 minutes and can lag the market during trading hours.")

bench_label = BENCHMARK_NAMES.get(bench_symbol, bench_symbol)
start_ts = pd.Timestamp(start_date)

head_left, head_right = st.columns([6, 1.1], vertical_alignment="bottom")
with head_left:
    st.title(APP_TITLE)
    subline = st.empty()
with head_right:
    if st.button("Refresh prices", width="stretch", help="Prices are kept for 15 minutes. This fetches them again now."):
        download_prices.clear()
        st.rerun()

for err in load_errors:
    st.warning(f"Couldn't read {err}")
if not selected:
    st.info("Pick at least one portfolio in the sidebar.")
    st.stop()

sectors = {}
for b in books.values():
    for sym, sec in zip(b.holdings["symbol"], b.holdings["sector"]):
        sectors.setdefault(sym, sec)
latest_w = {n: books[n].latest() for n in selected}
codes = sorted({s for b in books.values() for s in b.holdings["symbol"]})
symbols = tuple(sorted({yahoo_symbol(c) for c in codes if c != CASH} | {bench_symbol}))
dl_start = (min(start_ts, today - pd.DateOffset(years=HISTORY_YEARS)) - pd.Timedelta(days=20)).strftime("%Y-%m-%d")

prices, price_error = None, None
try:
    with st.spinner("Loading prices from Yahoo Finance…"):
        prices = build_prices(download_prices(symbols, dl_start), codes, bench_symbol)
except Exception as exc:
    price_error = str(exc)


def price_problem() -> None:
    st.error(f"Prices couldn't be loaded. {price_error} Check the internet connection, update yfinance "
             "with `pip install -U yfinance`, then press Refresh prices. The Holdings tab works without prices.")


# --- live record -----------------------------------------------------------------------------
live, bench_nav, anchor0, n_days = {}, None, None, 0
if prices is not None:
    for n in selected:
        res = run_live(books[n].snapshots(), prices.adj, start_ts, include_start)
        if res is not None:
            live[n] = res
    if live:
        anchor0 = min(r.periods[0]["anchor"] for r in live.values())
        n_days = max(len(r.nav) for r in live.values()) - 1
        if prices.bench is not None and not _bad(prices.bench.get(anchor0)):
            bench_nav = prices.bench.loc[anchor0:] / prices.bench.loc[anchor0]
live_ok = n_days >= 1
si = {n: float(r.nav.iloc[-1] / r.nav.iloc[0] - 1) for n, r in live.items()} if live_ok else {}
d1 = {n: float(r.nav.iloc[-1] / r.nav.iloc[-2] - 1) for n, r in live.items() if len(r.nav) >= 2}
bench_si = float(bench_nav.iloc[-1] - 1) if bench_nav is not None and len(bench_nav) >= 2 else None
bench_d1 = float(bench_nav.iloc[-1] / bench_nav.iloc[-2] - 1) if bench_si is not None else None
pos_by = {n: positions(live[n], prices, sectors, capital) for n in live} if live_ok else {}

# --- backtest of current weights -------------------------------------------------------------
bt, bt_stats, bench_r, bench_stats, rets_w, short_history = {}, {}, None, None, None, []
if prices is not None:
    rets = prices.adj / prices.adj.shift(1) - 1
    rets_w = rets.iloc[-BACKTEST_WINDOWS[window]:]
    if prices.bench is not None:
        bench_r = (prices.bench / prices.bench.shift(1) - 1).reindex(rets_w.index)
        bench_stats = risk_stats(bench_r, None, rf)
    for n in selected:
        bt[n] = simulate(latest_w[n], rets_w)
        bt_stats[n] = risk_stats(bt[n], bench_r, rf)
    held = sorted({s for n in selected for s in latest_w[n].index if s != CASH})
    for s in held:
        first = prices.first_date.get(s)
        if s not in prices.missing and first is not None and not pd.isna(first) and first > rets_w.index[0]:
            short_history.append(f"{s} (from {fmt_date(first)})")

# --- header line -----------------------------------------------------------------------------
if prices is not None:
    last_day = prices.adj.index[-1]
    intraday = last_day == today and now.hour * 60 + now.minute < 16 * 60 + 15
    price_note = f"{fmt_date(last_day)} {'intraday' if intraday else 'close'}, fetched {prices.fetched_at:%H:%M} WIB"
else:
    last_day, intraday, price_note = None, False, "not available"
subline.caption(
    f"{plural(len(selected), 'IDX model portfolio')} against {bench_label}, tracked from {fmt_date(start_ts)}. "
    f"Prices from Yahoo Finance: {price_note}."
)
if prices is not None:
    gone = [c for c in prices.missing if any(c in latest_w[n].index for n in selected)]
    if gone:
        st.warning(f"Yahoo Finance returned no prices for {', '.join(gone)}. Their weight is spread across the other "
                   "holdings until prices come back. See the Data tab.")
    if prices.bench is None:
        st.warning(f"No prices for the benchmark {bench_symbol}. Check the symbol in the sidebar; "
                   "comparisons with the benchmark are hidden.")

day_label = "Today" if last_day is not None and last_day == today else (
    f"{last_day.day} {last_day:%b}" if last_day is not None else "Last day")
tabs = st.tabs(["Summary", "Performance", "Attribution", "Holdings", "Risk & backtest", "Data"])

# ============================================================================================
# Summary
# ============================================================================================
with tabs[0]:
    if prices is None:
        price_problem()
    elif not live_ok:
        st.info(f"Live tracking starts after the first close following {fmt_date(start_ts)}. Until then the "
                "scorecard and the Risk & backtest tab show how today's weights would have behaved.")
    else:
        order = sorted(si, key=si.get, reverse=True)
        rows = [{"name": n, "si": si[n], "d1": d1.get(n, np.nan),
                 "rel": si[n] - bench_si if bench_si is not None else np.nan} for n in order]
        bench_row = {"si": bench_si, "d1": bench_d1} if bench_si is not None else None

        # Analyst notes: who leads, why, what moved today, how risky, how much to trust it
        notes = []
        lead, last = order[0], order[-1]
        lead_text = f"<b>{esc(lead)}</b> leads with {fmt_pct(si[lead])}"
        if bench_si is not None:
            lead_text += f", {fmt_pp(si[lead] - bench_si)} against {esc(bench_label)}"
        if len(order) > 1:
            notes.append(lead_text + f". <b>{esc(last)}</b> is last at {fmt_pct(si[last])}, "
                         f"{abs(si[lead] - si[last]) * 100:.2f} pp behind.")
            pa, pb = live[lead].periods[-1], live[last].periods[-1]
            common = max(pa["anchor"], pb["anchor"])
            gap = gap_breakdown(pa["weights"], pb["weights"], prices.adj.iloc[-1] / prices.adj.loc[common] - 1, sectors)
            helpers = gap.loc[gap["Effect"] > 0]
            if not helpers.empty and si[lead] > si[last]:
                g = helpers.iloc[0]
                rebalanced = len(live[lead].periods) > 1 or len(live[last].periods) > 1
                opener = "Since the last rebalance, the biggest single reason" if rebalanced else "The biggest single reason"
                move = f"up {abs(g['Return']):.1%}" if g["Return"] >= 0 else f"down {abs(g['Return']):.1%}"
                notes.append(
                    f"{opener} for the gap is <b>{esc(g['Symbol'])}</b>: {esc(lead)} holds "
                    f"{abs(g['Weight difference']) * 100:.1f} pp {'more' if g['Weight difference'] > 0 else 'less'} of it "
                    f"and it is {move}, worth {g['Effect'] * 100:.2f} pp of the {(si[lead] - si[last]) * 100:.2f} pp gap."
                )
        else:
            notes.append(lead_text + ".")
        if d1 and len(d1) > 1:
            best_d, worst_d = max(d1, key=d1.get), min(d1, key=d1.get)
            when = "Today" if day_label == "Today" else f"On {fmt_date(last_day)}"
            text = f"{when}, {esc(best_d)} is the strongest at {fmt_pct(d1[best_d])} and {esc(worst_d)} the weakest at {fmt_pct(d1[worst_d])}"
            if bench_d1 is not None:
                text += f"; {esc(bench_label)} is {fmt_pct(bench_d1)}"
            notes.append(text + ".")
        valid = {n: s for n, s in bt_stats.items() if s}
        if valid:
            best_sh = max(valid, key=lambda n: valid[n].get("sharpe", -np.inf))
            calm = max(valid, key=lambda n: valid[n].get("mdd", -np.inf))
            if best_sh == calm:
                notes.append(f"In the {window} backtest of current weights, {esc(best_sh)} has both the best Sharpe ratio "
                             f"({valid[best_sh]['sharpe']:.2f}) and the smallest drawdown ({fmt_pct(valid[calm]['mdd'], 1)}).")
            else:
                notes.append(f"In the {window} backtest of current weights, {esc(best_sh)} has the best Sharpe ratio "
                             f"({valid[best_sh]['sharpe']:.2f}) and {esc(calm)} the smallest drawdown "
                             f"({fmt_pct(valid[calm]['mdd'], 1)}).")
        if n_days < MIN_STAT_DAYS:
            notes.append(f"With {plural(n_days, 'trading day')} of live history the ranking is mostly noise. "
                         f"Lean on the backtest for risk until there are at least {MIN_STAT_DAYS} days.")

        st.subheader("Who is ahead")
        par_note = f" The dashed line is {bench_label}." if bench_si is not None else ""
        st.caption(f"Total return since the close on {fmt_date(anchor0)}, {plural(n_days, 'trading day')}.{par_note}")
        st.html(standings_html(rows, bench_row, colors, bench_label, day_label))
        backfilled = [n for n in order if books[n].holdings["date"].min() > start_ts]
        if backfilled:
            st.caption(f"{', '.join(backfilled)} {'has' if len(backfilled) == 1 else 'have'} no holdings file before "
                       f"the start date, so the first file's weights are applied backwards to {fmt_date(start_ts)}.")

        left, right = st.columns([5, 7], gap="large")
        with left:
            st.subheader("What stands out")
            st.html(notes_html(notes))
        with right:
            st.subheader("Cumulative return")
            navs = {n: live[n].nav - 1 for n in order}
            st.plotly_chart(fig_lines(navs, colors, bench_nav - 1 if bench_nav is not None else None, bench_label,
                                      height=360), width="stretch", config=PLOT_CONFIG)

    if prices is not None and any(bt_stats.values()):
        left, right = st.columns([8, 4], gap="large")
        with left:
            st.subheader("Scorecard")
            metric_defs = [
                ("Since start", None, True, "pct"),
                (f"{window} return", "total", True, "pct1"),
                ("Sharpe ratio", "sharpe", True, "num"),
                ("Max drawdown", "mdd", True, "pct1"),
                ("Info ratio", "ir", True, "num"),
                ("Down capture", "dncap", False, "cap"),
            ]
            vals = pd.DataFrame(index=selected, dtype=float)
            for label, key, _, _ in metric_defs:
                vals[label] = [si.get(n, np.nan) if key is None else bt_stats[n].get(key, np.nan) for n in selected]
            used = [m for m in metric_defs if vals[m[0]].notna().any()]
            ranks = pd.DataFrame({label: vals[label].rank(ascending=not higher, method="min")
                                  for label, _, higher, _ in used})
            avg = ranks.mean(axis=1)
            overall = avg.rank(method="min")
            cols_sc = sorted(selected, key=lambda n: (overall[n], avg[n]))
            board = pd.DataFrame([overall[cols_sc].to_numpy()] + [vals.loc[cols_sc, m[0]].to_numpy() for m in used]
                                 + [avg[cols_sc].to_numpy()],
                                 index=["Overall rank"] + [m[0] for m in used] + ["Average rank"], columns=cols_sc)
            board.index.name = "Measure"
            css_rows = {label: [rank_tint(ranks.loc[c, label]) for c in cols_sc] for label, *_ in used}
            css_rows["Overall rank"] = [rank_tint(overall[c]) + "; font-weight: 750" for c in cols_sc]
            css = pd.DataFrame([css_rows.get(r, [""] * len(cols_sc)) for r in board.index],
                               index=board.index, columns=cols_sc, dtype=object)
            styler = row_formats(board.style, {"Overall rank": "rank", **{m[0]: m[3] for m in used}, "Average rank": "avg"})
            styler = styler.apply(lambda _: css, axis=None)
            st.caption(f"Columns run from best to worst overall. Greener cells rank higher on that measure. Since "
                       f"start is live; the rest come from the {window} backtest of current weights.")
            show_table(styler, len(board))
        with right:
            st.subheader("Risk against return")
            st.caption(f"Each portfolio's yearly return and volatility in the {window} backtest.")
            st.plotly_chart(fig_risk_return(bt_stats, colors, bench_stats, bench_label), width="stretch",
                            config=PLOT_CONFIG)

# ============================================================================================
# Performance
# ============================================================================================
with tabs[1]:
    if prices is None:
        price_problem()
    elif not live_ok:
        st.info(f"Live tracking starts after the first close following {fmt_date(start_ts)}.")
    else:
        order = sorted(si, key=si.get, reverse=True)
        st.subheader("Returns by period")
        table = pd.DataFrame({n: period_returns(live[n].nav) for n in order}).T
        if bench_nav is not None:
            table.loc[bench_label] = pd.Series(period_returns(bench_nav))
        table = table.astype(float)
        table.index.name = "Portfolio"
        st.caption("Live record only. A dash means the portfolios haven't been tracked that long yet.")
        show_table(returns_styler(table, HORIZONS), len(table))

        left, right = st.columns(2, gap="large")
        with left:
            if bench_nav is not None:
                st.subheader(f"Lead over {bench_label}")
                st.caption(f"Cumulative return minus {bench_label}'s, in percentage points.")
                excess = {n: live[n].nav - bench_nav.reindex(live[n].nav.index) for n in order}
                st.plotly_chart(fig_lines(excess, colors, pp=True), width="stretch", config=PLOT_CONFIG)
            else:
                st.subheader("Lead over the benchmark")
                st.info("Needs benchmark prices.")
        with right:
            st.subheader("Drawdown from peak")
            st.caption("How far each portfolio sits below its own best level since the start.")
            dds = {n: live[n].nav / live[n].nav.cummax() - 1 for n in order}
            bench_dd = bench_nav / bench_nav.cummax() - 1 if bench_nav is not None else None
            st.plotly_chart(fig_lines(dds, colors, bench_dd, bench_label, signed=False), width="stretch",
                            config=PLOT_CONFIG)

        st.subheader("Daily returns")
        daily = pd.DataFrame({n: live[n].nav.pct_change() for n in order})
        if bench_nav is not None:
            daily[bench_label] = bench_nav.pct_change()
        daily = daily.iloc[1:].sort_index(ascending=False)
        daily.index = [fmt_date(d) for d in daily.index]
        daily.index.name = "Date"
        show_table(returns_styler(daily, list(daily.columns)), len(daily), cap=440)

# ============================================================================================
# Attribution
# ============================================================================================
with tabs[2]:
    if prices is None:
        price_problem()
    elif not live_ok:
        st.info(f"Attribution starts after the first close following {fmt_date(start_ts)}.")
    else:
        order = sorted(si, key=si.get, reverse=True)
        pick = st.segmented_control("Portfolio", order, default=order[0], required=True, key="attr_pick") or order[0]
        if pick not in live:
            pick = order[0]
        res, pos = live[pick], pos_by[pick]
        period = res.periods[-1]
        since_text = ("since the start" if len(res.periods) == 1
                      else f"since the last rebalance on {fmt_date(period['date'])}")
        st.caption(f"Contribution is each stock's target weight times its return {since_text} "
                   f"(close of {fmt_date(period['anchor'])}). The contributions add up to {pick}'s "
                   f"{fmt_pct(float(pos['Contribution'].sum()))}.")
        if res.excluded:
            st.caption(f"Left out for lack of a price at entry: {', '.join(res.excluded)}.")

        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("What drove the return")
            stock_pos = pos.loc[pos["Symbol"] != CASH]
            st.plotly_chart(fig_signed_bars(stock_pos["Symbol"], stock_pos["Contribution"]),
                            width="stretch", config=PLOT_CONFIG)
        with right:
            st.subheader("By sector")
            by_sector = pos.groupby("Sector")["Contribution"].sum()
            st.plotly_chart(fig_signed_bars(by_sector.index, by_sector.to_numpy()), width="stretch", config=PLOT_CONFIG)
            st.subheader("Position map")
            st.caption("Box size is the current weight; color is the return since entry.")
            st.plotly_chart(fig_treemap(pos), width="stretch", config=PLOT_CONFIG)

        st.subheader("Positions")
        cols_order = ["Symbol", "Sector", "Target weight", "Return", "Contribution", "Today", "Today's contribution",
                      "P&L (Rp)", "Current weight", "Entry price", "Last price", "Lots"]
        show = pos[cols_order]
        pct_cols, pp_cols = ["Return", "Today"], ["Contribution", "Today's contribution"]
        styler = (show.style
                  .format({"Target weight": lambda v: fmt_pct(v, 2, False), "Current weight": lambda v: fmt_pct(v, 2, False),
                           "Entry price": lambda v: fmt_num(v, 0), "Last price": lambda v: fmt_num(v, 0),
                           "P&L (Rp)": fmt_idr, "Lots": lambda v: fmt_num(v, 0),
                           **{c: (lambda v: fmt_pct(v)) for c in pct_cols},
                           **{c: (lambda v: fmt_pp(v)) for c in pp_cols}}, na_rep="—")
                  .map(sign_css, subset=pct_cols + pp_cols + ["P&L (Rp)"]))
        st.caption(f"Entry is the close on {fmt_date(period['anchor'])}. P&L and lots assume {fmt_idr(capital)} "
                   "invested at the start. Returns include dividends.")
        show_table(styler, len(show), hide_index=True,
                   column_config={"Symbol": st.column_config.TextColumn(pinned=True),
                                  "Sector": st.column_config.TextColumn(width=160)})

        if len(order) > 1:
            st.subheader("Head to head")
            c1, c2, _ = st.columns([1, 1, 2])
            a = c1.selectbox("Portfolio", order, index=0, key="h2h_a")
            others = [n for n in order if n != a]
            b = c2.selectbox("Against", others, index=len(others) - 1, key="h2h_b")
            pa, pb = live[a].periods[-1], live[b].periods[-1]
            common = max(pa["anchor"], pb["anchor"])
            gap = gap_breakdown(pa["weights"], pb["weights"], prices.adj.iloc[-1] / prices.adj.loc[common] - 1, sectors)
            total_gap = float(gap["Effect"].sum())
            st.caption(f"{a} is {abs(total_gap) * 100:.2f} pp {'ahead of' if total_gap >= 0 else 'behind'} {b} since the "
                       f"close on {fmt_date(common)}. Each bar is the difference in weight times the stock's return: "
                       f"bars to the right helped {a}, bars to the left helped {b}.")
            left, right = st.columns(2, gap="large")
            with left:
                top = gap.loc[gap["Effect"].abs() > 1e-6].head(12)
                st.plotly_chart(fig_signed_bars(top["Symbol"], top["Effect"]), width="stretch", config=PLOT_CONFIG)
            with right:
                sec_gap = gap.groupby("Sector")["Effect"].sum()
                sec_gap = sec_gap.loc[sec_gap.abs() > 1e-6]
                st.plotly_chart(fig_signed_bars(sec_gap.index, sec_gap.to_numpy()), width="stretch", config=PLOT_CONFIG)

# ============================================================================================
# Holdings
# ============================================================================================
with tabs[3]:
    st.subheader("Concentration")
    conc = pd.DataFrame({n: concentration(latest_w[n], sectors) for n in selected}).T
    conc.index.name = "Portfolio"
    st.caption("Effective stocks is 1 / sum of squared weights: how many equal positions would carry the same "
               "concentration.")
    show_table(conc.style.format({"Stocks": "{:.0f}", "Largest position": lambda v: fmt_pct(v, 1, False),
                                  "Top 5": lambda v: fmt_pct(v, 1, False), "Top 10": lambda v: fmt_pct(v, 1, False),
                                  "Effective stocks": "{:.1f}"}), len(conc))

    st.subheader("Sector weights")
    sec_mat = pd.DataFrame({n: latest_w[n].groupby(lambda s: sectors.get(s, "")).sum() for n in selected}).fillna(0.0)
    st.plotly_chart(fig_grouped_bars(sec_mat, colors), width="stretch", config=PLOT_CONFIG)

    st.subheader("Stock weights")
    wmat = pd.DataFrame({n: latest_w[n] for n in selected}).fillna(0.0)
    wmat = wmat.loc[wmat.mean(axis=1).sort_values(ascending=False).index] * 100
    wmat.insert(0, "Sector", [sectors.get(s, "") for s in wmat.index])
    wmat.index.name = "Symbol"
    top_w = float(wmat[selected].to_numpy().max())
    cfg = {n: st.column_config.ProgressColumn(n, format="%.2f%%", min_value=0.0, max_value=top_w, color=colors[n])
           for n in selected}
    cfg["Sector"] = st.column_config.TextColumn(width=170)
    st.caption("Sorted by average weight across the portfolios shown. A 0.00% bar means the portfolio doesn't hold it.")
    show_table(wmat, len(wmat), column_config=cfg)

    st.subheader("Overlap")
    ov = overlap_matrix({n: latest_w[n] for n in selected})
    ov.index.name = "Portfolio"
    msg = "Share of weight two portfolios hold in common. 100% would mean identical holdings."
    if len(selected) > 1:
        pairs = [(ov.loc[a, b], a, b) for i, a in enumerate(selected) for b in selected[i + 1:]]
        lo_pair, hi_pair = min(pairs), max(pairs)
        msg += (f" Most alike: {hi_pair[1]} and {hi_pair[2]} ({hi_pair[0]:.0%}). "
                f"Most different: {lo_pair[1]} and {lo_pair[2]} ({lo_pair[0]:.0%}).")
    st.caption(msg)
    show_table(ov.style.format(lambda v: fmt_pct(v, 0, False)).map(
        lambda v: f"background-color: rgba(47, 111, 222, {max(0.0, (v - 0.5) / 0.5) * 0.3:.3f})"), len(ov))

# ============================================================================================
# Risk & backtest
# ============================================================================================
with tabs[4]:
    if prices is None:
        price_problem()
    else:
        span_text = f"{fmt_date(rets_w.index[0])} to {fmt_date(rets_w.index[-1])}"
        st.caption(f"Applies each portfolio's current target weights to daily returns from {span_text}, rebalanced "
                   "daily. It shows how today's mix would have behaved, not what was actually earned.")
        if short_history:
            st.caption(f"Shorter price history: {', '.join(short_history)}. Until they listed, their weight is spread "
                       "across the other holdings.")
        order_bt = sorted(bt, key=lambda n: bt_stats[n].get("total", -np.inf), reverse=True)
        growth = {n: (1 + bt[n]).cumprod() - 1 for n in order_bt}
        bench_clean = bench_r.dropna() if bench_r is not None else None
        bench_growth = (1 + bench_clean).cumprod() - 1 if bench_clean is not None else None
        st.subheader("How the current mix would have grown")
        st.plotly_chart(fig_lines(growth, colors, bench_growth, bench_label, height=420), width="stretch",
                        config=PLOT_CONFIG)

        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Drawdowns")
            dd = {n: (1 + bt[n]).cumprod() / (1 + bt[n]).cumprod().cummax() - 1 for n in order_bt}
            bench_dd = ((1 + bench_clean).cumprod() / (1 + bench_clean).cumprod().cummax() - 1
                        if bench_clean is not None else None)
            st.plotly_chart(fig_lines(dd, colors, bench_dd, bench_label, height=330, signed=False), width="stretch",
                            config=PLOT_CONFIG)
        with right:
            st.subheader("Rolling 3-month volatility")
            roll = {n: (bt[n].rolling(63).std() * math.sqrt(TRADING_DAYS)).dropna() for n in order_bt}
            roll_b = (bench_r.rolling(63).std() * math.sqrt(TRADING_DAYS)).dropna() if bench_r is not None else None
            if all(len(v) for v in roll.values()):
                st.plotly_chart(fig_lines(roll, colors, roll_b, bench_label, height=330, signed=False, zero=False),
                                width="stretch", config=PLOT_CONFIG)
            else:
                st.info("Needs a window longer than three months.")

        st.subheader("Risk measures")
        measure_defs = [
            ("Return over window", "total", "pct", True), ("Return, a year", "cagr", "pct", True),
            ("Volatility, a year", "vol", "pct0", False), ("Sharpe ratio", "sharpe", "num", True),
            ("Sortino ratio", "sortino", "num", True), ("Max drawdown", "mdd", "pct", True),
            ("Calmar ratio", "calmar", "num", True), ("Beta", "beta", "num", None),
            ("Correlation with benchmark", "corr", "num", None), ("Tracking error", "te", "pct0", None),
            ("Information ratio", "ir", "num", True), ("Alpha, a year", "alpha", "pct", True),
            ("Upside capture", "upcap", "pct0", True), ("Downside capture", "dncap", "pct0", False),
            ("Days beating benchmark", "batting", "pct0", True), ("1-day VaR (95%)", "var95", "pct0", False),
            ("1-day CVaR (95%)", "cvar95", "pct0", False), ("Best day", "best", "pct", None),
            ("Worst day", "worst", "pct", True), ("Up days", "hit", "pct0", None),
        ]
        if BACKTEST_WINDOWS[window] == TRADING_DAYS:
            measure_defs = [m for m in measure_defs if m[1] != "cagr"]  # identical to the window return
        cols_rm = order_bt + ([bench_label] if bench_stats else [])
        all_stats = {**bt_stats, **({bench_label: bench_stats} if bench_stats else {})}
        rm = pd.DataFrame({c: [all_stats[c].get(k, np.nan) for _, k, _, _ in measure_defs] for c in cols_rm},
                          index=[m[0] for m in measure_defs]).astype(float)
        rm = rm.loc[rm.notna().any(axis=1)]
        rm.index.name = "Measure"
        better = {m[0]: m[3] for m in measure_defs}
        css = pd.DataFrame("", index=rm.index, columns=rm.columns, dtype=object)
        for row in rm.index:
            pref = better.get(row)
            vals_row = rm.loc[row, order_bt]
            if pref is not None and vals_row.notna().sum() >= 2:
                best = vals_row.idxmax() if pref else vals_row.idxmin()
                css.loc[row, best] = rank_tint(1) + "; font-weight: 750"
        rm_style = row_formats(rm.style, {m[0]: m[2] for m in measure_defs if m[0] in rm.index})
        rm_style = rm_style.apply(lambda _: css, axis=None)
        st.caption("The best portfolio on each measure is tinted green. Capture ratios compare average returns on the "
                   "benchmark's up days and down days; below 100% on down days means it fell less.")
        show_table(rm_style, len(rm))

        st.subheader("Correlation")
        corr_df = pd.DataFrame({n: bt[n] for n in order_bt})
        if bench_r is not None:
            corr_df[bench_label] = bench_r
        corr = corr_df.dropna().corr()
        corr.index.name = "Daily returns"
        corr_msg = ""
        if len(order_bt) > 1:
            off = [corr.loc[a, b] for i, a in enumerate(order_bt) for b in order_bt[i + 1:]]
            corr_msg = (f"The portfolios' daily returns are {min(off):.2f} to {max(off):.2f} correlated with each other, "
                        "so their differences come from tilts within largely the same stocks.")
        st.caption(corr_msg or "Correlation of daily returns over the backtest window.")
        show_table(corr.style.format("{:.2f}").map(
            lambda v: f"background-color: rgba(47, 111, 222, {max(0.0, min(1.0, (v - 0.6) / 0.4)) * 0.28:.3f})"),
            len(corr))

        st.subheader("Where the risk comes from")
        left, right = st.columns([7, 5], gap="large")
        with right:
            rpick = st.selectbox("Portfolio", order_bt, key="risk_pick")
            shares = risk_shares(latest_w[rpick], rets_w)
        if shares is None:
            right.info("Not enough price history to split the risk.")
        else:
            w_now = latest_w[rpick].reindex(shares.index)
            w_now = w_now / w_now.sum()
            by_sec = pd.DataFrame({
                "Weight": w_now.groupby(lambda s: sectors.get(s, "")).sum(),
                "Share of risk": shares.groupby(lambda s: sectors.get(s, "")).sum(),
            })
            with left:
                st.plotly_chart(fig_grouped_bars(by_sec, {"Weight": BENCH_COLOR, "Share of risk": colors[rpick]}),
                                width="stretch", config=PLOT_CONFIG)
            with right:
                over = (by_sec["Share of risk"] - by_sec["Weight"]).idxmax()
                st.caption(f"Share of risk splits the portfolio's variance across holdings, using the covariance of "
                           f"daily returns. {over} is {by_sec.loc[over, 'Weight']:.0%} of the weight but "
                           f"{by_sec.loc[over, 'Share of risk']:.0%} of the risk in {rpick}.")
                top_r = pd.DataFrame({"Weight": w_now, "Share of risk": shares}).sort_values("Share of risk",
                                                                                           ascending=False).head(8)
                top_r.index.name = "Biggest risks"
                show_table(top_r.style.format(lambda v: fmt_pct(v, 1, False)), len(top_r))

        st.subheader("Monthly returns")
        monthly = pd.DataFrame({n: (1 + bt[n]).resample("ME").prod() - 1 for n in order_bt})
        if bench_clean is not None:
            monthly[bench_label] = (1 + bench_clean).resample("ME").prod() - 1
        monthly = monthly.sort_index(ascending=False)
        labels = [f"{d:%b %Y}" for d in monthly.index]
        if len(labels) and monthly.index[0] > rets_w.index[-1]:
            labels[0] += " (to date)"
        monthly.index = labels
        monthly.index.name = "Month"
        show_table(returns_styler(monthly, list(monthly.columns), digits=1), len(monthly), cap=480)

# ============================================================================================
# Data
# ============================================================================================
with tabs[5]:
    if prices is not None:
        st.subheader("Price coverage")
        held_by = {c: ", ".join(n for n in selected if c in latest_w[n].index) for c in codes}
        cov_rows = []
        for c in codes:
            if c == CASH:
                continue
            first = prices.first_date.get(c)
            has = c not in prices.missing
            status = "No prices" if not has else (
                "Short history" if first is not None and first > rets_w.index[0] else "OK")
            cov_rows.append({"Symbol": c, "Yahoo symbol": prices.yahoo.get(c, ""), "Held by": held_by.get(c, ""),
                             "First price": fmt_date(first) if has else "—",
                             "Last close": prices.close[c].iloc[-1] if has else np.nan, "Status": status})
        cov = pd.DataFrame(cov_rows)
        show_table(cov.style.format({"Last close": lambda v: fmt_num(v, 0)}, na_rep="—").map(
            lambda v: f"color: {DOWN_COLOR}; font-weight: 600" if v == "No prices" else "", subset=["Status"]),
            len(cov), cap=480, hide_index=True)

        moves = (prices.adj / prices.adj.shift(1) - 1).drop(columns=[CASH], errors="ignore").stack()
        moves = moves.loc[moves.abs() > SUSPECT_MOVE]
        if len(moves):
            st.subheader("Moves to double-check")
            st.caption(f"Daily moves above {SUSPECT_MOVE:.0%} are beyond IDX price limits and usually point to a "
                       "data error such as an unadjusted split. They are included in the numbers as reported.")
            mv = moves.rename("Move").reset_index()
            mv.columns = ["Date", "Symbol", "Move"]
            mv["Date"] = mv["Date"].map(fmt_date)
            show_table(returns_styler(mv, ["Move"], 1), len(mv), cap=360, hide_index=True)

    st.subheader("Portfolio files")
    file_rows = []
    for n, b in books.items():
        snaps = sorted(b.raw_sums)
        file_rows.append({"Portfolio": n, "Dates": ", ".join(fmt_date(d) for d in snaps),
                          "Stocks (latest)": int((b.latest().index != CASH).sum()),
                          "Weights as written": ", ".join(f"{b.raw_sums[d]:.2%}" for d in snaps),
                          "Notes": " ".join(b.notes) or "—"})
    show_table(pd.DataFrame(file_rows), len(file_rows), hide_index=True)
    st.caption("To record a rebalance, add rows with the new date to the portfolio's file. Returns chain-link "
               "from one date to the next.")

    st.subheader("Downloads")
    d1c, d2c, d3c = st.columns(3)
    if live_ok:
        nav_df = pd.DataFrame({n: live[n].nav for n in live})
        if bench_nav is not None:
            nav_df[bench_label] = bench_nav
        d1c.download_button("Live values (CSV)", nav_df.to_csv(index_label="Date").encode("utf-8"),
                            file_name="live_values.csv", mime="text/csv", width="stretch")
        all_pos = pd.concat([p.assign(Portfolio=n) for n, p in pos_by.items()], ignore_index=True)
        d3c.download_button("Positions (CSV)", all_pos.to_csv(index=False).encode("utf-8"),
                            file_name="positions.csv", mime="text/csv", width="stretch")
    if bt:
        bt_df = pd.DataFrame(bt)
        if bench_r is not None:
            bt_df[bench_label] = bench_r
        d2c.download_button(f"Backtest daily returns, {window} (CSV)", bt_df.to_csv(index_label="Date").encode("utf-8"),
                            file_name=f"backtest_{window}.csv", mime="text/csv", width="stretch")
