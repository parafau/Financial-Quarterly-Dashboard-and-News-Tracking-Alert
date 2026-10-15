"""
Quarterly Trends (Streamlit) - Flow 1

Type a company symbol, pick parameters; each parameter is listed on the left
with its quarterly-report trend on the right. Hovering (or tapping) a
parameter name opens a popup: what it means in general, and what it means
for the company shown. Hovering (or tapping) a
parameter name opens a pop-up that explains it in plain English.

Run:
    pip install -r requirements.txt
    streamlit run main.py
"""
import html

import html

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
        if cols is None:
            cols = df.columns
        for n in names:
            if n in df.index:
                s = pd.to_numeric(df.loc[n], errors="coerce")
                return s.reindex(cols)
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
# Plain-English content for the pop-up
#   meaning        : what the parameter is, in general
#   up/down/flat   : what a rising / falling / steady trend usually says about
#                    the company ({c} = company symbol)
#   now (optional) : one line about the latest value
# --------------------------------------------------------------------------
def _info(meaning, up, down, flat, now=None):
    return dict(meaning=meaning, up=up, down=down, flat=flat, now=now)


PARAM_INFO = {
    "Sales": _info(
        "The total money the company earned from selling its products or "
        "services in the quarter, before any costs are taken out. Also called revenue.",
        "{c} is selling more than before. This usually means more customers are buying.",
        "{c} is selling less than before. This can mean weaker demand or tougher competition.",
        "{c} is selling about the same each quarter: stable, but not growing."),
    "Expenses": _info(
        "All the money the company spent to run the business in the quarter: "
        "salaries, raw materials, rent, electricity and so on.",
        "{c} is spending more. That is fine if sales are growing too, but a worry if they are not.",
        "{c} is spending less, which can leave more profit, unless it is cutting things that hurt future growth.",
        "{c}'s spending is steady."),
    "Operating Profit": _info(
        "What is left from sales after paying the day-to-day costs of running "
        "the business, before interest and tax. It shows how well the core business earns.",
        "{c} is earning more from its core business, which is a healthy sign.",
        "{c} is keeping less from its core business. Costs may be growing faster than sales.",
        "{c}'s core earnings are steady."),
    "Profit before tax": _info(
        "The profit left after all costs, including interest on loans, but before tax is paid.",
        "{c} is earning more before tax, which is good news for its final profit.",
        "{c} is earning less before tax, which will pull down its final profit.",
        "{c}'s profit before tax is steady."),
    "Interest": _info(
        "The money the company pays on its loans. Lower is usually better, "
        "because less of the earnings go to lenders.",
        "{c} is paying more interest, usually because it borrowed more or rates rose. This eats into profit.",
        "{c} is paying less interest, often by repaying debt. This leaves more profit for shareholders.",
        "{c}'s interest burden is steady."),
    "Net Profit": _info(
        "The final profit after every cost, interest and tax. This is what really "
        "belongs to the company's owners (the shareholders).",
        "{c} is earning more for its owners. This is usually what investors most want to see.",
        "{c} is earning less for its owners. Check whether falling sales or rising costs caused it.",
        "{c}'s final profit is steady."),
    "OPM %": _info(
        "Operating Profit Margin: out of every Rs 100 of sales, how many rupees stay "
        "as operating profit. Higher means the company is better at turning sales into profit.",
        "{c}'s margin is widening. It is becoming more efficient or earning better prices.",
        "{c}'s margin is shrinking. Costs may be rising faster than sales, or prices are under pressure.",
        "{c}'s margin is steady.",
        lambda v: f"Right now it keeps about Rs {v:.0f} of every Rs 100 it sells as operating profit."),
    "Tax %": _info(
        "The share of the company's pre-tax profit that goes to the government as tax.",
        "{c} is paying a bigger slice of its profit as tax, which leaves less final profit.",
        "{c} is paying a smaller slice as tax, which leaves more final profit (this may not last).",
        "{c}'s tax rate is steady.",
        lambda v: f"It pays about Rs {v:.0f} tax on every Rs 100 of pre-tax profit."),
    "EPS": _info(
        "Earnings Per Share: the company's net profit divided by its number of shares. "
        "It tells you how much profit each single share earned.",
        "Each {c} share is earning more, which usually supports a higher share price over time.",
        "Each {c} share is earning less.",
        "Each {c} share is earning about the same."),
    "ROE % (annualised)": _info(
        "Return on Equity: how much profit the company makes in a year for every "
        "Rs 100 that shareholders have put in. Higher means the owners' money is used well. "
        "(The quarter's profit is multiplied by 4 to estimate a full year.)",
        "{c} is getting more profit out of its shareholders' money.",
        "{c} is getting less profit out of its shareholders' money.",
        "{c}'s return on shareholders' money is steady.",
        lambda v: f"It currently earns roughly Rs {v:.0f} a year for every Rs 100 of shareholders' money."),
    "ROCE % (annualised)": _info(
        "Return on Capital Employed: yearly operating profit for every Rs 100 of all the "
        "money (shareholders' plus borrowed) used in the business. It shows how well all "
        "that money is put to work. (The quarter's profit is multiplied by 4.)",
        "{c} is putting its money to better use.",
        "{c} is getting less out of the money it uses.",
        "{c}'s use of money is steady.",
        lambda v: f"It currently earns roughly Rs {v:.0f} a year for every Rs 100 used in the business."),
    "Price (qtr-end)": _info(
        "The share price on the last trading day of that quarter.",
        "{c}'s share price has risen, so buyers have been willing to pay more for it.",
        "{c}'s share price has fallen. This may reflect worries, or it may be a bargain.",
        "{c}'s share price has barely moved."),
    "P/E (annualised qtr EPS)": _info(
        "Price-to-Earnings: how many rupees investors pay for each Re 1 the company earns in "
        "a year. A high P/E means investors expect strong growth (or the shares are expensive); "
        "a low one can mean cheap shares or low expectations. (Uses the quarter's EPS x 4.)",
        "{c}'s shares are getting more expensive compared with its profits.",
        "{c}'s shares are getting cheaper compared with its profits. Either profits grew faster "
        "than the price, or investors are less hopeful.",
        "{c}'s price compared with its profits is steady.",
        lambda v: f"Investors are paying about Rs {v:.0f} for every Rs 1 of its yearly earnings."),
    "Market Cap": _info(
        "The total value of the whole company on the stock market: share price x number of "
        "shares. It is the company's size as the market sees it.",
        "The market values {c} higher than before.",
        "The market values {c} lower than before.",
        "The market's valuation of {c} is steady."),
    "Book Value / share": _info(
        "What each share would be worth on paper if the company sold everything it owns and "
        "paid all its debts: the company's net worth per share.",
        "{c}'s net worth per share is growing. It is building up value.",
        "{c}'s net worth per share is shrinking.",
        "{c}'s net worth per share is steady."),
    "Dividend Yield %": _info(
        "The cash a company paid shareholders over the past year, as a percentage of the share "
        "price. It is like interest on your investment, but it is not guaranteed.",
        "{c}'s yield is rising, because it is paying more or because its share price has fallen.",
        "{c}'s yield is falling, because it is paying less or because its share price has risen.",
        "{c}'s yield is steady.",
        lambda v: (f"Over the past year it paid about Rs {v:.1f} in dividends for every Rs 100 of share price."
                   if v > 0 else "It has not paid any dividend in the past year.")),
    "52W High": _info(
        "The highest price the share touched in the 12 months before that quarter ended. "
        "It marks the top of its recent range.",
        "{c}'s recent peaks are getting higher.",
        "{c}'s recent peaks are getting lower.",
        "{c}'s recent peaks are about the same."),
    "52W Low": _info(
        "The lowest price the share touched in the 12 months before that quarter ended. "
        "It marks the bottom of its recent range.",
        "Even {c}'s low points are rising, which suggests a stronger floor.",
        "{c}'s low points are dropping.",
        "{c}'s low points are about the same."),
}


# --------------------------------------------------------------------------
# Plain-language explanations (rule-based, no AI needed)
#   meaning : what the parameter is, for any company
#   up/down/flat : what the trend means for the company on screen ({co})
# --------------------------------------------------------------------------
INFO = {
    "Sales": dict(
        meaning="The total money the company brought in from selling its products or services in the quarter, before any costs are taken out.",
        up="{co} is selling more than before, so more money is coming in. Healthy, as long as costs don't grow faster.",
        down="{co} is bringing in less money than before. This can mean fewer customers or lower prices, and it puts pressure on profit.",
        flat="{co}'s sales are holding steady: not growing, not shrinking."),
    "Expenses": dict(
        meaning="The money spent to run the business in the quarter: salaries, raw materials, rent, and so on.",
        up="{co}'s costs are rising. That's fine if sales are rising too, but a worry if sales are flat or falling.",
        down="{co} is spending less to run the business, which helps profit, unless it is cutting things that bring in future sales.",
        flat="{co}'s running costs are steady."),
    "Operating Profit": dict(
        meaning="What is left from sales after paying the everyday costs of running the business, before interest and tax. It shows how well the core business earns.",
        up="{co}'s core business is earning more from its day-to-day work.",
        down="{co}'s core business is earning less. Costs may be growing faster than sales.",
        flat="{co}'s core earnings are steady."),
    "OPM %": dict(
        meaning="Operating profit margin: out of every Rs 100 of sales, how many rupees remain as operating profit. Higher means the company is better at turning sales into profit.",
        up="{co} is keeping more of every Rs 100 it sells, a sign of better efficiency or stronger pricing.",
        down="{co} is keeping less of every Rs 100 it sells. Costs are eating into its earnings.",
        flat="{co} is keeping about the same share of each sale as before."),
    "Interest": dict(
        meaning="The cost of the money the company has borrowed. Companies with a lot of loans pay a lot of interest.",
        up="{co} is paying more to its lenders, which reduces profit. It may be borrowing more or at higher rates.",
        down="{co} is paying less interest. It may have repaid some loans, which leaves more profit for shareholders.",
        flat="{co}'s borrowing cost is steady."),
    "Profit before tax": dict(
        meaning="What the company earned after all costs and interest, but before paying tax.",
        up="{co} is earning more before the tax bill, a good sign for the final profit.",
        down="{co} is earning less before tax, which will shrink the final profit too.",
        flat="{co}'s pre-tax earnings are steady."),
    "Tax %": dict(
        meaning="The share of the company's pre-tax profit that goes to the government as tax.",
        up="{co} is paying a bigger slice of its profit as tax, so less is left for shareholders. One-off items can cause this.",
        down="{co} is paying a smaller slice as tax, which leaves more profit for shareholders. This can be a one-off.",
        flat="{co}'s tax share is steady."),
    "Net Profit": dict(
        meaning="The final profit after every cost, interest and tax. This is the money truly left for the owners (shareholders).",
        up="{co} is keeping more money at the end, which can go to dividends, repaying loans or growing the business.",
        down="{co} is keeping less money at the end, leaving less for dividends, repaying loans or growth.",
        flat="{co}'s final profit is steady."),
    "EPS": dict(
        meaning="Earnings per share: the company's net profit divided by its number of shares. It's the profit that belongs to one share.",
        up="Each {co} share is earning more profit than before, which usually supports the share's value over time.",
        down="Each {co} share is earning less profit than before, which can weigh on the share's value.",
        flat="Each {co} share is earning about the same as before."),
    "ROE % (annualised)": dict(
        meaning="Return on equity: how much profit the company makes for every Rs 100 the owners have invested in it. Higher means the owners' money is working well. Shown as a yearly figure (quarter x 4).",
        up="{co} is making more profit from its owners' money than before.",
        down="{co} is making less profit from its owners' money than before.",
        flat="{co} is using its owners' money about as well as before."),
    "ROCE % (annualised)": dict(
        meaning="Return on capital employed: profit made for every Rs 100 of all the money (owners' plus borrowed) working in the business. Higher means the business uses its money well. Shown as a yearly figure (quarter x 4).",
        up="{co} is earning more from all the money it has put to work.",
        down="{co} is earning less from all the money it has put to work.",
        flat="{co} is using its money about as well as before."),
    "Price (qtr-end)": dict(
        meaning="The price of one share on the last trading day of the quarter.",
        up="{co}'s shares are worth more than before: investors are paying more for them. This reflects market mood, not just results.",
        down="{co}'s shares are worth less than before: investors are paying less for them. This reflects market mood, not just results.",
        flat="{co}'s share price is about where it was."),
    "P/E (annualised qtr EPS)": dict(
        meaning="Price-to-earnings: how many rupees investors pay for every Rs 1 the company earns in a year. A high number means the share is pricey compared to its profit (or investors expect growth); a low number can mean cheap, or that investors are worried.",
        up="Investors are paying more for each rupee of {co}'s profit. The share got pricier compared to what it earns.",
        down="Investors are paying less for each rupee of {co}'s profit. The share got cheaper compared to what it earns.",
        flat="{co}'s price compared to its profit is about the same."),
    "Market Cap": dict(
        meaning="Market capitalisation: the total value of all the company's shares at today's price, or what the market thinks the whole company is worth. Bigger usually means a larger, more established company.",
        up="The market values {co} higher than before.",
        down="The market values {co} lower than before.",
        flat="The market's value of {co} is about the same."),
    "Book Value / share": dict(
        meaning="What each share would be worth on paper if the company sold everything it owns and paid off all its debts.",
        up="{co}'s net assets per share are growing, so the company is building up value over time.",
        down="{co}'s net assets per share are shrinking, which can come from losses, big payouts or more debt.",
        flat="{co}'s net assets per share are steady."),
    "Dividend Yield %": dict(
        meaning="The yearly cash dividend a company pays, shown as a percentage of the share price. It's the income you get just for holding the share.",
        up="Holding {co} now pays more cash each year compared to the share price (the price may also have dropped).",
        down="Holding {co} now pays less cash each year compared to the share price (the price may also have risen).",
        flat="{co}'s cash payout compared to its price is steady."),
    "52W High": dict(
        meaning="The highest price the share reached in the last 52 weeks (one year).",
        up="{co}'s best price of the year keeps getting higher, so the share has been trending upward.",
        down="{co}'s best price of the year is getting lower, so the older high prices are dropping out of the window.",
        flat="{co}'s best price of the year hasn't changed."),
    "52W Low": dict(
        meaning="The lowest price the share fell to in the last 52 weeks (one year).",
        up="Even {co}'s lowest price of the year is rising, a sign of a steady climb.",
        down="{co} keeps hitting lower prices, so the share has been falling.",
        flat="{co}'s lowest price of the year hasn't changed."),
}


# --------------------------------------------------------------------------
# Wording helpers
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
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