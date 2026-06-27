"""
US Stock Breakout Screener (with FMP fundamentals)
===================================================
Replaces the earlier trend/momentum screener. Adapted from a Chartink-style
breakout scan (originally written for the Indian cash-equity market) to US
equities: yfinance for price/volume/technicals, Financial Modeling Prep
(FMP) for fundamentals.

WHAT IT CHECKS (translated 1:1 from the original filter list, minus the
Market Cap rule - dropped since symbols.txt is already a vetted universe)
---------------------------------------------------------------
 1. Last completed week's Close  >  20-week SMA of weekly Close
 2. Today's High   <=  max High of the PRIOR 6 days (tight range, no new high)
 3. Today's Low    >=  min Low  of the PRIOR 6 days (tight range, no new low)
 4. Today's Close  ==  max Close of the last 10 days (fresh 10-day closing high)
 5. Close >= 1.05 * yesterday's Close   (breakout move, at least +5%)
 6. Close <= 1.20 * yesterday's Close   (not a blow-off, at most +20%)
 7. (High - max(Close,Open)) <= 0.50 * |Close - Open|  (closed strong, small wick)
 8. MACD Line(12,26,9)  >  MACD Signal(12,26,9)        (bullish momentum)
 9. In the last 15 days, MACD Line was <= the PRIOR day's Signal at least
    once (confirms #8 is a recent crossover, not a stale trend)
10. Volume >= 1.30 * yesterday's Volume   (volume surge on the breakout)
11. ROE   (most recent annual report)  >= 10%
12. ROCE  (most recent annual report)  >= 10%
13. Net Sales        (most recent quarter) > 0
14. Net Profit       (most recent quarter) > 0
15. Operating Margin (most recent quarter) >= 10%
16. ATR(14) / Close * 100  <=  8%   (caps day-to-day volatility)

SUGGESTED RISK MANAGEMENT (informational only - not a pass/fail filter)
---------------------------------------------------------------
For any symbol that passes, the output also includes two suggested exit
levels (you decide what to actually do with them - this is just math, not
trading advice):
 - Hard stop-loss: 10% below the entry-day close
 - SMA(10) trailing exit: the classic breakout/swing-trade rule of exiting
   if a later day's close falls below the 10-day SMA. The output also flags
   if today's close is ALREADY below its own 10-day SMA (a sign today isn't
   a clean entry by this rule, even if it passed all 16 filters above).

NOTES
---------------------------------------------------------------
- Rules 1-10 and 16 use yfinance daily price/volume data.
- Rules 11-15 use SEC EDGAR (data.sec.gov) - free, official, no API key.
  This is the actual raw filing data companies submit to the SEC, the same
  source every paid fundamentals provider ultimately pulls from. ROE/ROCE/
  Operating Margin are computed manually from the raw figures since SEC
  doesn't publish pre-calculated ratios (see fundamentals.py for the exact
  formulas). IMPORTANT: edit SEC_USER_AGENT in fundamentals.py with your
  own name/email before running - SEC requires this and blocks requests
  that don't identify a real requester.
- Rule 1 deliberately skips the most recent (possibly still-forming) week
  and uses the week before it, so it isn't distorted by a partial week.
- Like the previous version of this script: run AFTER the 4pm ET close for
  clean daily volume figures (rule 10), and ideally after Friday's close for
  clean weekly figures (rule 1).

USAGE
---------------------------------------------------------------
    pip install yfinance pandas numpy requests
    (edit SEC_USER_AGENT in fundamentals.py first - see above)
    python us_screener.py

Reads tickers from symbols.txt (one comma-separated line, same folder).
Writes results to screener_results.csv and prints a summary table.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from indicators import atr, macd, resample_weekly, sma
from fundamentals import get_fundamentals
from telegram_notify import send_telegram_message

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
SYMBOLS_FILE = "symbols.txt"
HISTORY_PERIOD = "2y"            # plenty for weekly SMA20, MACD(26), ATR(14)
BATCH_SIZE = 40                   # tickers per yfinance batch download
PAUSE_BETWEEN_BATCHES = 1.5       # seconds, be polite to Yahoo's servers
FUNDAMENTALS_PAUSE_SECONDS = 0.2  # seconds between SEC EDGAR lookups (SEC asks <=10 req/sec)
OUTPUT_CSV = "screener_results.csv"
OUTPUT_JSON = "docs/results.json"  # consumed by the GitHub Pages dashboard

CONSOLIDATION_WINDOW = 6          # rules 2-3: tight-range lookback (days, excl. today)
BREAKOUT_WINDOW = 10              # rule 4: new closing-high lookback (days, incl. today)
MIN_BREAKOUT_PCT = 1.05           # rule 5
MAX_BREAKOUT_PCT = 1.20           # rule 6
MAX_UPPER_WICK_RATIO = 0.50       # rule 7
MACD_CROSS_LOOKBACK = 15          # rule 9
VOLUME_SURGE_MULT = 1.30          # rule 10
MIN_ROE = 0.10                    # rule 11 (10%)
MIN_ROCE = 0.10                   # rule 12 (10%)
MIN_OPERATING_MARGIN = 0.10       # rule 15 (10%)
MAX_ATR_PCT = 8.0                 # rule 16 (ATR/Close * 100 <= 8)
STOP_LOSS_PCT = 0.10               # suggested hard stop-loss, 10% below entry close
SMA_EXIT_WINDOW = 10               # suggested trailing exit: close below this SMA


def load_symbols(path):
    with open(path) as f:
        raw = f.read().strip()
    syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return syms


def evaluate_price_rules(df):
    """Rules 1-10 and 16 - pure price/volume/technicals via yfinance data."""
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(df) < 60:
        return None  # not enough bars for MACD/weekly SMA to stabilize

    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    weekly = resample_weekly(df)
    if len(weekly) < 21:
        return None  # not enough weekly bars for the 20-week SMA
    weekly_sma20 = sma(weekly["Close"], 20)

    # rules 2-3: tight range over the PRIOR N days, excluding today
    prior_high_max = high.shift(1).rolling(CONSOLIDATION_WINDOW).max()
    prior_low_min = low.shift(1).rolling(CONSOLIDATION_WINDOW).min()
    # rule 4: fresh closing high including today
    close_max_n = close.rolling(BREAKOUT_WINDOW).max()

    macd_line, macd_signal, macd_hist = macd(close, 12, 26, 9)
    atr14 = atr(df, 14)
    sma10 = sma(close, SMA_EXIT_WINDOW)

    try:
        c0 = close.iloc[-1]
        o0 = open_.iloc[-1]
        h0 = high.iloc[-1]
        v0 = volume.iloc[-1]
        c_prev = close.iloc[-2]
        v_prev = volume.iloc[-2]

        # rule 1: last completed week's close vs 20-week SMA (skip the
        # still-forming current week by using the row before it)
        r1 = bool(weekly["Close"].iloc[-2] > weekly_sma20.iloc[-2])

        r2 = bool(h0 <= prior_high_max.iloc[-1])
        r3 = bool(low.iloc[-1] >= prior_low_min.iloc[-1])
        r4 = bool(c0 == close_max_n.iloc[-1])
        r5 = bool(c0 >= MIN_BREAKOUT_PCT * c_prev)
        r6 = bool(c0 <= MAX_BREAKOUT_PCT * c_prev)

        body = abs(c0 - o0)
        upper_wick = h0 - max(c0, o0)
        r7 = bool(upper_wick <= MAX_UPPER_WICK_RATIO * body)

        r8 = bool(macd_line.iloc[-1] > macd_signal.iloc[-1])

        recent_line = macd_line.iloc[-MACD_CROSS_LOOKBACK:]
        prev_signal = macd_signal.shift(1).iloc[-MACD_CROSS_LOOKBACK:]
        r9 = bool((recent_line <= prev_signal).sum() >= 1)

        r10 = bool(v0 >= VOLUME_SURGE_MULT * v_prev)

        atr_pct = float(atr14.iloc[-1] / c0 * 100)
        r16 = bool(atr_pct <= MAX_ATR_PCT)

        rules = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r16]
        if any(pd.isna(x) for x in rules):
            return None

        # --- suggested risk management (informational, not a screen filter) ---
        # Hard stop: exit if price falls 10% below today's close.
        stop_loss_10pct = round(float(c0) * (1 - STOP_LOSS_PCT), 2)
        # Trailing exit: exit if a future day's close falls below the
        # 10-day SMA (classic breakout/swing-trade trailing-stop rule).
        # "below_now" just flags whether that's ALREADY true as of today -
        # if so, today wasn't a clean entry by this rule's own logic.
        sma10_value = round(float(sma10.iloc[-1]), 2)
        sma10_already_broken = bool(c0 < sma10.iloc[-1])

    except (IndexError, KeyError, ZeroDivisionError):
        return None

    return {
        "close": round(float(c0), 2),
        "rule_1_weekly_close_gt_weekly_sma20": r1,
        "rule_2_tight_range_high": r2,
        "rule_3_tight_range_low": r3,
        "rule_4_new_10d_closing_high": r4,
        "rule_5_breakout_ge_5pct": r5,
        "rule_6_breakout_le_20pct": r6,
        "rule_7_strong_close_small_wick": r7,
        "rule_8_macd_line_gt_signal": r8,
        "rule_9_recent_macd_cross": r9,
        "rule_10_volume_surge_1.3x": r10,
        "rule_16_atr_pct_le_8": r16,
        "atr_pct_of_close": round(atr_pct, 2),
        "stop_loss_10pct": stop_loss_10pct,
        "sma10_exit_level": sma10_value,
        "sma10_already_broken": sma10_already_broken,
    }


def evaluate_fundamental_rules(symbol):
    """Rules 11-15 - via yfinance (Yahoo Finance) fundamentals."""
    try:
        fund = get_fundamentals(symbol)
    except Exception as e:
        print(f"  Fundamentals lookup failed for {symbol}: {e}")
        return None

    if fund is None:
        return None

    roe = fund["roe"]
    roce = fund["roce"]
    revenue = fund["revenue"]
    net_income = fund["net_income"]
    margin = fund["operating_margin"]

    r11 = bool(roe >= MIN_ROE)
    r12 = bool(roce >= MIN_ROCE)
    r13 = bool(revenue > 0)
    r14 = bool(net_income > 0)
    r15 = bool(margin >= MIN_OPERATING_MARGIN)

    return {
        "rule_11_roe_ge_10pct": r11,
        "rule_12_roce_ge_10pct": r12,
        "rule_13_net_sales_gt_0": r13,
        "rule_14_net_profit_gt_0": r14,
        "rule_15_operating_margin_ge_10pct": r15,
        "roe_pct": round(float(roe) * 100, 1),
        "roce_pct": round(float(roce) * 100, 1),
        "operating_margin_pct": round(float(margin) * 100, 1),
    }


def evaluate_symbol(symbol, df):
    """Apply all 16 screener rules to a single symbol."""
    price_res = evaluate_price_rules(df)
    if price_res is None:
        return None

    fund_res = evaluate_fundamental_rules(symbol)
    time.sleep(FUNDAMENTALS_PAUSE_SECONDS)
    if fund_res is None:
        return None

    all_rules = {**price_res, **fund_res}
    rule_keys = [k for k in all_rules if k.startswith("rule_")]
    passed = all(all_rules[k] for k in rule_keys)

    result = {"symbol": symbol, "passed": bool(passed)}
    result.update(price_res)
    result.update(fund_res)
    return result


def fetch_batch(tickers):
    import yfinance as yf
    data = yf.download(
        tickers=tickers,
        period=HISTORY_PERIOD,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    return data


def run_screener(symbols):
    results = []
    n = len(symbols)
    for start in range(0, n, BATCH_SIZE):
        batch = symbols[start:start + BATCH_SIZE]
        print(f"Fetching {start + 1}-{min(start + BATCH_SIZE, n)} of {n}: {batch}")
        try:
            data = fetch_batch(batch)
        except Exception as e:
            print(f"  Batch download failed ({e}); skipping batch.")
            continue

        for sym in batch:
            try:
                if len(batch) == 1:
                    df = data
                else:
                    df = data[sym].copy()
                df = df.dropna(how="all")
                if df.empty:
                    continue
                res = evaluate_symbol(sym, df)
                if res:
                    results.append(res)
            except Exception as e:
                print(f"  Skipping {sym}: {e}")
                continue

        time.sleep(PAUSE_BETWEEN_BATCHES)

    return pd.DataFrame(results)


RULE_COLUMNS = [
    "rule_1_weekly_close_gt_weekly_sma20",
    "rule_2_tight_range_high",
    "rule_3_tight_range_low",
    "rule_4_new_10d_closing_high",
    "rule_5_breakout_ge_5pct",
    "rule_6_breakout_le_20pct",
    "rule_7_strong_close_small_wick",
    "rule_8_macd_line_gt_signal",
    "rule_9_recent_macd_cross",
    "rule_10_volume_surge_1.3x",
    "rule_11_roe_ge_10pct",
    "rule_12_roce_ge_10pct",
    "rule_13_net_sales_gt_0",
    "rule_14_net_profit_gt_0",
    "rule_15_operating_margin_ge_10pct",
    "rule_16_atr_pct_le_8",
]


def build_telegram_message(df_all, passed):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "*US Breakout Screener*",
        now,
        f"Evaluated: {len(df_all)} / {len(load_symbols(SYMBOLS_FILE))} symbols",
        f"Passed all 16 filters: *{len(passed)}*",
    ]
    if len(passed) > 0:
        lines.append("")
        lines.append("Passed:")
        for _, row in passed.iterrows():
            warn = " \u26a0 already below SMA10" if row.get("sma10_already_broken") else ""
            lines.append(
                f"  {row['symbol']}  close {row['close']}  "
                f"| SL -10%: {row['stop_loss_10pct']}  | SMA10 exit: {row['sma10_exit_level']}{warn}"
            )
    else:
        # show the closest near-misses so a 0-pass day is still useful info
        almost = df_all.copy()
        almost["rules_passed_count"] = almost[RULE_COLUMNS].sum(axis=1)
        almost = almost.sort_values("rules_passed_count", ascending=False).head(5)
        lines.append("")
        lines.append("Closest near-misses (rules passed / 16):")
        for _, row in almost.iterrows():
            lines.append(f"  {row['symbol']}: {int(row['rules_passed_count'])}/16")
    return "\n".join(lines)


def write_dashboard_json(df_all, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = df_all.copy()
    df["rules_passed_count"] = df[RULE_COLUMNS].sum(axis=1)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_symbols": len(load_symbols(SYMBOLS_FILE)),
        "total_evaluated": len(df),
        "total_passed": int(df["passed"].sum()),
        "rule_columns": RULE_COLUMNS,
        "rows": df.sort_values(
            ["passed", "rules_passed_count"], ascending=[False, False]
        ).to_dict(orient="records"),
    }
    with open(path, "w") as f:
        json.dump(payload, f, default=str)


def main():
    symbols = load_symbols(SYMBOLS_FILE)
    print(f"Loaded {len(symbols)} symbols from {SYMBOLS_FILE}")

    df_all = run_screener(symbols)
    if df_all.empty:
        print("No results returned (check network access / yfinance+SEC EDGAR setup).")
        send_telegram_message("*US Breakout Screener*\nNo results returned this run - check the logs.")
        sys.exit(0)

    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull rule breakdown for all evaluated symbols saved to {OUTPUT_CSV}")

    write_dashboard_json(df_all, OUTPUT_JSON)
    print(f"Dashboard data written to {OUTPUT_JSON}")

    passed = df_all[df_all["passed"]].sort_values("close", ascending=False)
    print(f"\n=== {len(passed)} / {len(df_all)} symbols PASSED all 16 filters ===")
    if not passed.empty:
        cols = ["symbol", "close", "roe_pct", "roce_pct", "operating_margin_pct", "atr_pct_of_close"]
        print(passed[cols].to_string(index=False))

    message = build_telegram_message(df_all, passed)
    send_telegram_message(message)


if __name__ == "__main__":
    main()
