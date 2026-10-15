import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

CRORE = 1e7  # yfinance returns raw INR; Indian reports use Rs crore

BAR_PARAMS = ["Sales", "Expenses", "Operating Profit", "Profit before tax",
              "Interest", "Net Profit"]
LINE_PARAMS = ["OPM %", "Tax %", "EPS", "ROE % (annualised)",
               "ROCE % (annualised)", "Price (qtr-end)",
               "P/E (annualised qtr EPS)", "Market Cap", "Book Value / share",
               "Dividend Yield %", "52W High", "52W Low"]
ALL_PARAMS = BAR_PARAMS + LINE_PARAMS
RS_CR = set(BAR_PARAMS) | {"Market Cap"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def normalize_symbol(sym: str) -> str:
    """'tcs' -> 'TCS.NS' (works for any NSE symbol, not just a fixed list)."""
    sym = sym.strip().upper()
    if not sym:
        return sym
    return sym if sym.endswith((".NS", ".BO")) else sym + ".NS"


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


# --------------------------------------------------------------------------
# Layman wording
# --------------------------------------------------------------------------
def fmt(name: str, v: float) -> str:
    if "%" in name:
        return f"{v:.1f}%"
    if name in RS_CR:
        return f"Rs {v:,.0f} Cr"
    return f"{v:,.2f}"


def layman(name: str, s: pd.Series) -> str:
    s = s.dropna()
    if len(s) < 2:
        return "Not enough data to show a trend."
    a, b = s.iloc[0], s.iloc[-1]
    if "%" in name:
        change, tol = b - a, 1.0
    else:
        change, tol = ((b - a) / abs(a) * 100 if a else 0.0), 3.0
    word = ("Going up" if change > tol
            else "Going down" if change < -tol else "Roughly steady")
    return f"{word}: {fmt(name, a)} to {fmt(name, b)} over {len(s)} quarters."


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.set_page_config(page_title="Quarterly Trends", layout="wide")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>",
            unsafe_allow_html=True)
st.title("Quarterly Trends")

c1, c2 = st.columns([1, 2])
raw = c1.text_input("Symbol", placeholder="e.g. TCS")
params = c2.multiselect("Parameters", ALL_PARAMS,
                        default=["Sales", "Net Profit", "OPM %"])

if raw and params:
    symbol = normalize_symbol(raw)
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
            for p in params:
                st.divider()
                left, right = st.columns([1, 3])
                left.markdown(f"**{p}**")
                left.caption(layman(p, m[p]))
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