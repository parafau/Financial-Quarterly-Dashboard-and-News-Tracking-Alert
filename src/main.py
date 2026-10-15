import html

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

CRORE = 1e7  # yfinance returns raw INR; Indian reports use Rs crore

PARAMS = json.loads(
    Path(__file__).with_name("parameters.json").read_text(encoding="utf-8"))
ALL_PARAMS = list(PARAMS)
BAR_PARAMS = [p for p, v in PARAMS.items() if v["chart"] == "bar"]
RS_CR = {p for p, v in PARAMS.items() if v["unit"] == "rs_cr"}
INFO = PARAMS



# Helpers

def normalize_symbol(sym: str) -> str:
    """'tcs' -> 'TCS.NS' (works for any NSE symbol, not just a fixed list)."""
    sym = sym.strip().upper()
    if not sym:
        return sym
    return sym if sym.endswith((".NS", ".BO")) else sym + ".NS"


def get_row(df, names, cols=None) -> pd.Series:
    """First matching row of a yfinance statement (labels vary by company)."""
    if df is not None and not df.empty:
        if cols is None:
            cols = df.columns
        for n in names:
            if n in df.index:
                s = pd.to_numeric(df.loc[n], errors="coerce")
                return s.reindex(cols)
    return pd.Series(np.nan, index=cols if cols is not None else [])



# Data -> metrics table (one row per quarter)

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



# Wording helpers

def fmt(name: str, v: float) -> str:
    if "%" in name:
        return f"{v:.1f}%"
    if name in RS_CR:
        return f"Rs {v:,.0f} Cr"
    return f"{v:,.2f}"


def trend(name: str, s: pd.Series):
    """'up' / 'down' / 'flat' from first to last quarter, or None."""
    s = s.dropna()
    if len(s) < 2:
        return None
    a, b = s.iloc[0], s.iloc[-1]
    if "%" in name:
        change, tol = b - a, 1.0
    else:
        change, tol = ((b - a) / abs(a) * 100 if a else 0.0), 3.0
    return "up" if change > tol else "down" if change < -tol else "flat"


def data_line(name: str, s: pd.Series) -> str:
    t = trend(name, s)
    s = s.dropna()
    word = {"up": "Going up", "down": "Going down", "flat": "Roughly steady"}[t]
    return f"{word}: {fmt(name, s.iloc[0])} to {fmt(name, s.iloc[-1])} over {len(s)} quarters."


def popup_html(i: int, name: str, s: pd.Series, co: str) -> str:
    """Hover popup: section 1 (open) = meaning, section 2 = effect on this company."""
    info, t, co_e = INFO[name], trend(name, s), html.escape(co)
    if t is None:
        effect = f"There isn't enough data in {co_e}'s reports yet to say how this affects it."
    else:
        effect = (f"<b>{html.escape(data_line(name, s))}</b><br>"
                  + info[t].format(co=co_e))
    return f"""
<div class="tip" tabindex="0">
  <span class="lbl">{html.escape(name)} &#9432;</span>
  <div class="pop">
    <details name="g{i}" open>
      <summary>What is this?</summary><p>{info["meaning"]}</p>
    </details>
    <details name="g{i}">
      <summary>Effect on {co_e}</summary><p>{effect}</p>
    </details>
  </div>
</div>"""


POPUP_CSS = """
<style>
.tip{position:relative;display:inline-block;padding-bottom:8px;outline:none}
.tip .lbl{font-weight:600;border-bottom:1px dotted #888;cursor:help}
.tip .pop{display:none;position:absolute;left:0;top:100%;z-index:9999;
  width:min(320px,80vw);background:#fff;color:#222;border:1px solid #ddd;
  border-radius:8px;box-shadow:0 6px 20px rgba(0,0,0,.18);padding:2px 12px;
  font-size:.88rem;line-height:1.45}
.tip:hover .pop,.tip:focus-within .pop{display:block}
.tip details{padding:8px 0;border-bottom:1px solid #eee}
.tip details:last-child{border-bottom:none}
.tip summary{cursor:pointer;font-weight:600}
.tip details p{margin:6px 0 2px 0}
</style>
"""



# Page

st.set_page_config(page_title="Quarterly Trends", layout="wide")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>",
            unsafe_allow_html=True)
st.html(POPUP_CSS)
st.title("Quarterly Trends")

c1, c2 = st.columns([1, 2])
raw = c1.text_input("Symbol", placeholder="e.g. TCS")
params = c2.multiselect("Parameters", ALL_PARAMS,
                        default=["Sales", "Net Profit", "OPM %"])

if raw and params:
    symbol = normalize_symbol(raw)
    company = symbol.rsplit(".", 1)[0]
    try:
        m = build_metrics(symbol)
    except Exception:
        m = None
        st.error("Could not fetch data right now. Please try again in a minute.")

    if m is not None:
        if m.empty:
            st.error(f"No quarterly data found for {symbol}. Check the symbol.")
        else:
            labels = m.index.strftime("%b %Y")
            st.caption(f"{symbol}  |  {labels[0]} to {labels[-1]}")
            for i, p in enumerate(params):
                st.divider()
                left, right = st.columns([1, 3])
                with left:
                    st.html(popup_html(i, p, m[p], company))
                if m[p].notna().any():
                    data = pd.DataFrame({"Quarter": labels, p: m[p].values})
                    if p in BAR_PARAMS:
                        fig = px.bar(data, x="Quarter", y=p, text_auto=".3s")
                    else:
                        fig = px.line(data, x="Quarter", y=p, markers=True)
                    fig.update_layout(template="simple_white", height=230,
                                      margin=dict(l=0, r=0, t=10, b=0),
                                      xaxis_title=None, yaxis_title=None)
                    fig.update_xaxes(type="category")
                    right.plotly_chart(fig, width="stretch")
                else:
                    right.caption("Data not available for this company.")