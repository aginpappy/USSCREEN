"""
Fundamentals data via SEC EDGAR (data.sec.gov) - FREE, OFFICIAL, NO API KEY.

This is the actual primary source - the raw XBRL figures companies file with
the SEC in their 10-K/10-Q reports. Every other provider (FMP, yfinance,
Finnhub, etc.) ultimately gets its numbers from here. It will not get
paywalled or rate-limited away since public company filings are required by
law to be public.

Used for screener rules 11-15:
 11. ROE  (most recent annual report)     >= 10%
 12. ROCE (most recent annual report)     >= 10%
 13. Net Sales        (most recent quarter) > 0
 14. Net Profit       (most recent quarter) > 0
 15. Operating Margin (most recent quarter) >= 10%

REQUIRED ONE-TIME SETUP
---------------------------------------------------------------
The SEC requires every automated requester to identify itself with a real
User-Agent (name + contact email) - generic/default User-Agents get
blocked with a 403. Edit SEC_USER_AGENT below with your own info before
running:

    SEC_USER_AGENT = "Your Name your-email@example.com"

This is NOT an API key, no signup needed - just a courtesy identification
string SEC's fair-access policy requires. See:
https://www.sec.gov/os/webmaster-faq#developers

HOW IT WORKS
---------------------------------------------------------------
1. Ticker -> CIK lookup, from SEC's own ticker list (one shared file,
   fetched once and cached in memory for the whole run).
2. Per symbol: one call to the "companyfacts" endpoint, which returns every
   XBRL fact the company has ever filed (revenue, net income, assets,
   liabilities, equity, etc.), tagged with the filing form (10-K/10-Q),
   period start/end dates, and filed date.
3. We pull out the specific tags we need and pick the most recent
   appropriate value for each (latest 10-K for annual figures, latest
   single-quarter 10-Q for quarterly figures).

NOTES
---------------------------------------------------------------
- ROE/ROCE/Operating Margin are computed manually from raw figures (the SEC
  doesn't publish pre-calculated ratios) - same idea as before, just from a
  more authoritative source:
      ROE  = Net Income (annual) / Stockholders Equity (latest)
      ROCE = Operating Income (annual) / (Total Assets - Current Liabilities)
- XBRL tag names vary slightly by company (e.g. "Revenues" vs
  "RevenueFromContractWithCustomerExcludingAssessedTax"), so each field
  tries several known tag names and uses whichever is present.
- Some companies (especially financials/banks) don't break out
  "LiabilitiesCurrent" on a classified balance sheet - ROCE will be
  unavailable for those, and that symbol is skipped rather than guessed at.
- SEC asks for no more than ~10 requests/second; this module paces itself
  well under that by default (see FUNDAMENTALS_PAUSE_SECONDS in
  us_screener.py).
"""

import time

import requests

SEC_USER_AGENT = "Anchu Abraham anchu.abraham@example.com"  # <-- EDIT THIS

BASE_URL = "https://data.sec.gov"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_CIK_MAP = None  # lazy-loaded, cached for the life of the process

# A single-quarter duration is ~70-100 days; this filters out cumulative
# (e.g. "9 months ended") values that share the same tag in XBRL filings.
QUARTER_MIN_DAYS = 70
QUARTER_MAX_DAYS = 100


def _headers():
    return {"User-Agent": SEC_USER_AGENT}


def _load_cik_map():
    global _CIK_MAP
    if _CIK_MAP is not None:
        return _CIK_MAP
    resp = requests.get(TICKERS_URL, headers=_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # SEC returns a dict keyed by index strings ("0", "1", ...), not a list
    rows = data.values() if isinstance(data, dict) else data
    mapping = {}
    for row in rows:
        ticker = row.get("ticker", "").upper()
        cik = row.get("cik_str")
        if ticker and cik is not None:
            mapping[ticker] = str(cik).zfill(10)
    _CIK_MAP = mapping
    return mapping


def get_cik(symbol):
    """Look up a ticker's 10-digit CIK. Tries a couple of common dash/dot
    variants (e.g. BRK-B) since SEC and other providers don't always agree
    on share-class formatting."""
    mapping = _load_cik_map()
    sym = symbol.upper()
    for candidate in (sym, sym.replace("-", ""), sym.replace("-", ".")):
        if candidate in mapping:
            return mapping[candidate]
    return None


def _get_company_facts(cik, retries=3, pause=1.0):
    url = f"{BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_headers(), timeout=20)
            if resp.status_code in (403, 429):
                time.sleep(pause * (attempt + 1))
                continue
            if resp.status_code == 404:
                return None  # company has no XBRL facts on file
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(pause)
    if last_err:
        raise last_err
    return None


def _usd_units(facts, tag_names):
    """Return the list of USD unit entries for the first matching tag name."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tag_names:
        node = us_gaap.get(tag)
        if node and "USD" in node.get("units", {}):
            return node["units"]["USD"]
    return None


def _latest_annual_duration(facts, tag_names):
    """Most recent 10-K value for a duration-type (income statement) tag."""
    units = _usd_units(facts, tag_names)
    if not units:
        return None
    annual = [u for u in units if u.get("form") in ("10-K", "10-K/A") and u.get("val") is not None]
    if not annual:
        return None
    annual.sort(key=lambda u: (u.get("end", ""), u.get("filed", "")), reverse=True)
    return float(annual[0]["val"])


def _latest_quarter_duration(facts, tag_names):
    """Most recent single-quarter (not cumulative YTD) 10-Q value for a
    duration-type (income statement) tag."""
    units = _usd_units(facts, tag_names)
    if not units:
        return None
    quarterly = []
    for u in units:
        if u.get("form") not in ("10-Q", "10-Q/A") or u.get("val") is None:
            continue
        try:
            from datetime import date
            start = date.fromisoformat(u["start"])
            end = date.fromisoformat(u["end"])
        except (KeyError, ValueError):
            continue
        days = (end - start).days
        if QUARTER_MIN_DAYS <= days <= QUARTER_MAX_DAYS:
            quarterly.append(u)
    if not quarterly:
        return None
    quarterly.sort(key=lambda u: (u.get("end", ""), u.get("filed", "")), reverse=True)
    return float(quarterly[0]["val"])


def _latest_instant(facts, tag_names):
    """Most recent point-in-time (balance sheet) value for a tag, from
    either a 10-K or 10-Q (whichever is most recent)."""
    units = _usd_units(facts, tag_names)
    if not units:
        return None
    instants = [
        u for u in units
        if u.get("form") in ("10-K", "10-K/A", "10-Q", "10-Q/A") and u.get("val") is not None and u.get("end")
    ]
    if not instants:
        return None
    instants.sort(key=lambda u: (u.get("end", ""), u.get("filed", "")), reverse=True)
    return float(instants[0]["val"])


def get_fundamentals(symbol):
    """Returns a dict with roe, roce, revenue, net_income, operating_margin
    (roe/roce/operating_margin as decimals, e.g. 0.15 == 15%), or None if
    essential data isn't available for this symbol."""
    cik = get_cik(symbol)
    if cik is None:
        return None

    facts = _get_company_facts(cik)
    if facts is None:
        return None

    revenue_tags = [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ]
    net_income_tags = ["NetIncomeLoss", "ProfitLoss"]
    operating_income_tags = ["OperatingIncomeLoss"]
    assets_tags = ["Assets"]
    current_liab_tags = ["LiabilitiesCurrent"]
    equity_tags = [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ]

    # quarterly figures (rules 13-15)
    revenue_q = _latest_quarter_duration(facts, revenue_tags)
    net_income_q = _latest_quarter_duration(facts, net_income_tags)
    operating_income_q = _latest_quarter_duration(facts, operating_income_tags)

    margin = None
    if revenue_q not in (None, 0) and operating_income_q is not None:
        margin = operating_income_q / revenue_q

    # annual figures (rules 11-12)
    net_income_annual = _latest_annual_duration(facts, net_income_tags)
    operating_income_annual = _latest_annual_duration(facts, operating_income_tags)
    equity = _latest_instant(facts, equity_tags)
    total_assets = _latest_instant(facts, assets_tags)
    current_liab = _latest_instant(facts, current_liab_tags)

    roe = None
    if net_income_annual is not None and equity:
        roe = net_income_annual / equity

    roce = None
    if operating_income_annual is not None and total_assets is not None and current_liab is not None:
        capital_employed = total_assets - current_liab
        if capital_employed:
            roce = operating_income_annual / capital_employed

    if None in (roe, roce, revenue_q, net_income_q, margin):
        return None

    return {
        "roe": roe,
        "roce": roce,
        "revenue": revenue_q,
        "net_income": net_income_q,
        "operating_margin": margin,
    }
