"""
Technical indicator library used by the US trend/momentum screener.
All functions take/return pandas Series or DataFrames indexed by date (ascending).
"""

import numpy as np
import pandas as pd


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def sma(series, window):
    return series.rolling(window).mean()


def wilder_smooth(series, period):
    """Wilder's smoothing == EMA with alpha = 1/period."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


def true_range(df):
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df, period=14):
    return wilder_smooth(true_range(df), period)


def adx_dmi(df, period=14):
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    atr_ = atr(df, period)
    plus_di = 100 * wilder_smooth(plus_dm, period) / atr_
    minus_di = 100 * wilder_smooth(minus_dm, period) / atr_

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_ = wilder_smooth(dx, period)
    return adx_, plus_di, minus_di


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(series, period=20, num_std=2):
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def rolling_vwap(df, period=20):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    pv = typical_price * df["Volume"]
    return pv.rolling(period).sum() / df["Volume"].rolling(period).sum()


def obv(df):
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def ichimoku(df, tenkan_p=9, kijun_p=26, senkou_b_p=52, displacement=26):
    tenkan = (df["High"].rolling(tenkan_p).max() + df["Low"].rolling(tenkan_p).min()) / 2
    kijun = (df["High"].rolling(kijun_p).max() + df["Low"].rolling(kijun_p).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(displacement)
    senkou_b = (
        (df["High"].rolling(senkou_b_p).max() + df["Low"].rolling(senkou_b_p).min()) / 2
    ).shift(displacement)
    return tenkan, kijun, senkou_a, senkou_b


def supertrend(df, period=10, multiplier=3):
    hl2 = (df["High"] + df["Low"]) / 2
    atr_ = atr(df, period)
    basic_upper = hl2 + multiplier * atr_
    basic_lower = hl2 - multiplier * atr_

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    n = len(df)
    close = df["Close"].values
    bu = basic_upper.values
    bl = basic_lower.values
    fu = final_upper.values.copy()
    fl = final_lower.values.copy()

    for i in range(1, n):
        fu[i] = min(bu[i], fu[i - 1]) if close[i - 1] <= fu[i - 1] else bu[i]
        fl[i] = max(bl[i], fl[i - 1]) if close[i - 1] >= fl[i - 1] else bl[i]

    direction = np.ones(n, dtype=int)
    st = np.zeros(n)
    direction[0] = 1
    st[0] = fl[0]

    for i in range(1, n):
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < fl[i] else 1
        else:
            direction[i] = 1 if close[i] > fu[i] else -1
        st[i] = fl[i] if direction[i] == 1 else fu[i]

    return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)


def parabolic_sar(df, af_start=0.02, af_step=0.02, af_max=0.2):
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    n = len(df)

    sar = np.zeros(n)
    trend = np.zeros(n, dtype=int)  # 1 = up, -1 = down

    trend[0] = 1
    sar[0] = low[0]
    ep = high[0]
    af = af_start

    for i in range(1, n):
        prev_sar = sar[i - 1]

        if trend[i - 1] == 1:
            candidate = prev_sar + af * (ep - prev_sar)
            lim = low[i - 2] if i >= 2 else low[i - 1]
            candidate = min(candidate, low[i - 1], lim)

            if low[i] < candidate:
                trend[i] = -1
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                trend[i] = 1
                sar[i] = candidate
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            candidate = prev_sar + af * (ep - prev_sar)
            lim = high[i - 2] if i >= 2 else high[i - 1]
            candidate = max(candidate, high[i - 1], lim)

            if high[i] > candidate:
                trend[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                trend[i] = -1
                sar[i] = candidate
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    return pd.Series(sar, index=df.index), pd.Series(trend, index=df.index)


def macd(series, fast=12, slow=26, signal=9):
    """Standard MACD. Chartink labels this 'Macd(26,12,9)' (slow,fast,signal)
    but the calculation is the usual fast/slow/signal EMA setup."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def resample_weekly(df):
    weekly = df.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return weekly.dropna()
