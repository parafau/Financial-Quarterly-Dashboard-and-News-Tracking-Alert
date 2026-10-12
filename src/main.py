"""
Stock Trend Dashboard (Streamlit)

Run:
    pip install streamlit yfinance pandas plotly
    streamlit run main.py

Tabs
  1. Company dashboard : quarterly trend charts for each parameter, QoQ / YoY
                         change, rule-based flags and a plain-English summary.
  2. Portfolio         : same analysis for every stock the user holds.

Reference: parameters follow Screener's quarterly table
(Sales, Expenses, Operating Profit, OPM %, Interest, PBT, Tax %, Net Profit, EPS)
plus the ratio/market items from the project's ticker_parameters list.
"""
import glob
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Trend Dashboard", layout="wide")

CRORE = 1e7  # yfinance returns raw INR; Indian reports use Rs crore

# Parameters shown on the dashboard (column names of the metrics table)
BAR_PARAMS = ["Sales", "Expenses", "Operating Profit", "Profit before tax",
              "Interest", "Net Profit"]
LINE_PARAMS = ["OPM %", "Tax %", "EPS", "ROE % (annualised)",
               "ROCE % (annualised)", "Price (qtr-end)",
               "P/E (annualised qtr EPS)", "Market Cap", "Book Value / share",
               "Dividend Yield %", "52W High", "52W Low"]
ALL_PARAMS = BAR_PARAMS + LINE_PARAMS
DEFAULT_PARAMS = ["Sales", "Net Profit", "OPM %", "EPS",
                  "ROE % (annualised)", "ROCE % (annualised)"]
ICON = {"Improving": "▲", "Declining": "▼", "Flat": "▬", "n/a": "–"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def normalize_symbol(sym: str) -> str:
    """'tcs' -> 'TCS.NS' (works for any NSE symbol, not just a fixed list)."""
    sym = sym.strip().upper()
    if not sym:
        return sym
    return sym if sym.endswith((".NS", ".BO")) else sym + ".NS"


def load_symbols() -> list:
    """Read symbols from the latest data/nifty100_*.csv (download_nifty100.py)."""
    files = sorted(glob.glob("data/nifty100_*.csv"))
    if not files:
        return []
    df = pd.read_csv(files[-1])
    col = next((c for c in df.columns if c.strip().lower() == "symbol"), None)
    if col is None:
        return []
    return sorted(df[col].dropna().astype(str).str.upper().unique())


def get_row(df, names, cols=None) -> pd.Series:
    """First matching row of a yfinance statement (labels vary by company)."""
    if df is not None and not df.empty:
        for n in names:
            if n in df.index:
                s = pd.to_numeric(df.loc[n], errors="coerce")
                return s if cols is None else s.reindex(cols)
    return pd.Series(np.nan, index=cols if cols is not None else [])


# --------------------------------------------------------------------------
# Data -> metrics table (one row per quarter)
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching quarterly data...")
def build_metrics(symbol: str) -> pd.DataFrame:
    t = yf.Ticker(symbol)
    fin = t.quarterly_financials
    if fin is None or fin.empty:
        return pd.DataFrame()
    bs = t.quarterly_balance_sheet
    cols = fin.columns

    # ---- income statement items
    rev = get_row(fin, ["Total Revenue", "Operating Revenue"])
    opinc = get_row(fin, ["Operating Income"])
    expenses = get_row(fin, ["Total Expenses"]).fillna(rev - opinc)
    ebit = get_row(fin, ["EBIT"]).fillna(opinc)
    interest = get_row(fin, ["Interest Expense"])
    pbt = get_row(fin, ["Pretax Income"])
    tax = get_row(fin, ["Tax Provision"])
    ni = get_row(fin, ["Net Income", "Net Income Common Stockholders"])
    eps = get_row(fin, ["Basic EPS", "Diluted EPS"])

    # ---- balance sheet items (aligned to income-statement quarter dates)
    equity = get_row(bs, ["Stockholders Equity", "Common Stock Equity",
                          "Total Equity Gross Minority Interest"], cols)
    assets = get_row(bs, ["Total Assets"], cols)
    cur_liab = get_row(bs, ["Current Liabilities"], cols)
    shares = get_row(bs, ["Ordinary Shares Number", "Share Issued"], cols)

    # ---- price based items, measured at each quarter-end date
    hist = t.history(period="3y", auto_adjust=False)
    close = hist["Close"] if "Close" in hist else pd.Series(dtype=float)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    divs = t.dividends
    divs.index = pd.to_datetime(divs.index).tz_localize(None)

    qdates = pd.to_datetime(cols)
    price, hi, lo, dps = [], [], [], []
    for d in qdates:
        price.append(close.asof(d) if len(close) else np.nan)
        window = close[(close.index <= d) & (close.index > d - pd.Timedelta(days=365))]
        hi.append(window.max() if len(window) else np.nan)
        lo.append(window.min() if len(window) else np.nan)
        dps.append(divs[(divs.index <= d) &
                        (divs.index > d - pd.Timedelta(days=365))].sum())
    price = pd.Series(price, index=cols)
    hi = pd.Series(hi, index=cols)
    lo = pd.Series(lo, index=cols)
    dps = pd.Series(dps, index=cols)

    m = pd.DataFrame({
        "Sales": rev / CRORE,
        "Expenses": expenses / CRORE,
        "Operating Profit": opinc / CRORE,
        "OPM %": opinc / rev * 100,
        "Interest": interest / CRORE,
        "Profit before tax": pbt / CRORE,
        "Tax %": tax / pbt * 100,
        "Net Profit": ni / CRORE,
        "EPS": eps,
        "ROE % (annualised)": ni * 4 / equity * 100,
        "ROCE % (annualised)": ebit * 4 / (assets - cur_liab) * 100,
        "Price (qtr-end)": price,
        "P/E (annualised qtr EPS)": price / (eps * 4).where(eps > 0),
        "Market Cap": price * shares / CRORE,
        "Book Value / share": equity / shares,
        "Dividend Yield %": dps / price * 100,
        "52W High": hi,
        "52W Low": lo,
    })
    m.index = qdates
    m = m.sort_index()
    m.index.name = "Quarter end"
    return m


@st.cache_data(ttl=600)
def live_snapshot(symbol: str) -> dict:
    try:
        fi = yf.Ticker(symbol).fast_info
        return {"Price": fi["lastPrice"], "Market cap (Rs Cr)": fi["marketCap"] / CRORE,
                "52W high": fi["yearHigh"], "52W low": fi["yearLow"]}
    except Exception:
        return {}


# --------------------------------------------------------------------------
# Rule-based analysis (baseline to compare the ML model against)
# --------------------------------------------------------------------------
def trend_status(name: str, s: pd.Series) -> str:
    """Current quarter vs two quarters back: Improving / Declining / Flat."""
    s = s.dropna()
    if len(s) < 3:
        return "n/a"
    a, b = s.iloc[-3], s.iloc[-1]
    if "%" in name:                      # ratios: change in percentage points
        chg, tol = b - a, 1.0
    else:                                # absolute values: % change
        if a == 0:
            return "n/a"
        chg, tol = (b - a) / abs(a) * 100, 3.0
    return "Improving" if chg > tol else "Declining" if chg < -tol else "Flat"


def rule_flags(m: pd.DataFrame) -> list:
    flags = []

    def diffs(col):
        return m[col].dropna().diff().dropna()

    d = diffs("OPM %")
    if len(d) >= 2 and (d.iloc[-2:] < 0).all() and d.iloc[-2:].sum() <= -2:
        flags.append("Operating margin fell for 2 straight quarters "
                     f"({d.iloc[-2:].sum():.1f} points in total).")
    d = diffs("Net Profit")
    if len(d) >= 2 and (d.iloc[-2:] < 0).all():
        flags.append("Net profit declined for 2 straight quarters.")
    for col, label in [("Sales", "Sales"), ("EPS", "EPS")]:
        s = m[col].dropna()
        if len(s) >= 5 and s.iloc[-5] > 0 and s.iloc[-1] < s.iloc[-5]:
            flags.append(f"{label} is lower than the same quarter last year.")
    s = m["Interest"].dropna()
    if len(s) >= 2 and s.iloc[-2] > 0 and s.iloc[-1] > 1.25 * s.iloc[-2]:
        flags.append("Interest cost jumped by more than 25% vs last quarter.")
    return flags


def plain_summary(m: pd.DataFrame, flags: list) -> str:
    words = {"Improving": "improving", "Declining": "declining",
             "Flat": "flat", "n/a": "not available"}
    parts = [f"{label} is {words[trend_status(col, m[col])]}"
             for col, label in [("Sales", "sales"), ("Net Profit", "net profit"),
                                ("OPM %", "operating margin"), ("EPS", "EPS")]]
    text = "Over the last 3 quarters, " + ", ".join(parts) + "."
    text += (f" {len(flags)} warning flag(s) raised." if flags
             else " No warning flags raised by the rules.")
    return text


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📈 Stock Trend Dashboard")
st.caption("Quarterly trends from company reports (via Yahoo Finance). "
           "Amounts in Rs crore unless stated. Educational project, not investment advice.")

tab_company, tab_portfolio = st.tabs(["📊 Company dashboard", "💼 Portfolio"])

# ---------------- Tab 1: single company ----------------
with tab_company:
    symbols = load_symbols()
    c1, c2 = st.columns([2, 1])
    if symbols:
        choice = c1.selectbox("Nifty 100 symbol", symbols,
                              index=symbols.index("TCS") if "TCS" in symbols else 0)
        custom = c2.text_input("...or type any NSE symbol", "")
        raw = custom or choice
    else:
        raw = c1.text_input("NSE symbol", "TCS")
    symbol = normalize_symbol(raw)

    m = build_metrics(symbol) if symbol else pd.DataFrame()
    if m.empty:
        st.error(f"No quarterly data found for {symbol}. Check the symbol.")
    else:
        snap = live_snapshot(symbol)
        if snap:
            for col, (k, v) in zip(st.columns(len(snap)), snap.items()):
                col.metric(k, f"{v:,.2f}")

        flags = rule_flags(m)
        st.info(plain_summary(m, flags))
        for f in flags:
            st.warning(f)

        chosen = st.multiselect("Parameters to chart", ALL_PARAMS, DEFAULT_PARAMS)
        labels = m.index.strftime("%b %Y")
        grid = st.columns(2)
        for i, p in enumerate(chosen):
            data = pd.DataFrame({"Quarter": labels, p: m[p].values})
            if p in BAR_PARAMS:
                fig = px.bar(data, x="Quarter", y=p, text_auto=".3s")
            else:
                fig = px.line(data, x="Quarter", y=p, markers=True)
            status = trend_status(p, m[p])
            fig.update_layout(title=f"{p}  {ICON[status]} {status}",
                              margin=dict(l=10, r=10, t=50, b=10), height=300)
            fig.update_xaxes(type="category")
            grid[i % 2].plotly_chart(fig, width="stretch")

        with st.expander("Data table, QoQ and YoY change"):
            tbl = m.copy()
            tbl.index = labels
            st.dataframe(tbl.T.round(2), width="stretch")
            st.markdown("**QoQ change (%)**")
            q = m.pct_change(1, fill_method=None) * 100
            q.index = labels
            st.dataframe(q.T.round(1), width="stretch")
            if len(m) >= 5:
                st.markdown("**YoY change (%)**")
                y = m.pct_change(4, fill_method=None) * 100
                y.index = labels
                st.dataframe(y.T.dropna(how="all").round(1), width="stretch")
            else:
                st.caption("YoY needs 5 quarters of data; Yahoo returned fewer.")

# ---------------- Tab 2: portfolio ----------------
with tab_portfolio:
    holdings = st.text_area("Your holdings (NSE symbols, comma or newline separated)",
                            "TCS, INFY, RELIANCE, HDFCBANK")
    if st.button("Analyze portfolio"):
        syms = [s for s in re.split(r"[,\s]+", holdings) if s]
        rows, flag_map = [], {}
        bar = st.progress(0.0)
        for i, s in enumerate(syms, 1):
            sym = normalize_symbol(s)
            try:
                pm = build_metrics(sym)
            except Exception:
                pm = pd.DataFrame()
            if pm.empty:
                rows.append({"Stock": sym, "Sales": "n/a", "Net Profit": "n/a",
                             "OPM %": "n/a", "EPS": "n/a", "Flags": "no data"})
            else:
                fl = rule_flags(pm)
                flag_map[sym] = fl
                row = {"Stock": sym}
                for col in ["Sales", "Net Profit", "OPM %", "EPS"]:
                    st_ = trend_status(col, pm[col])
                    row[col] = f"{ICON[st_]} {st_}"
                row["Flags"] = len(fl)
                rows.append(row)
            bar.progress(i / len(syms))
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        for sym, fl in flag_map.items():
            if fl:
                with st.expander(f"⚠️ {sym}: {len(fl)} flag(s)"):
                    for f in fl:
                        st.write("•", f)
