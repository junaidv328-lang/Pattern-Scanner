"""
Real market data via yfinance.

Note: this sandbox's network is restricted to package registries, so these
functions can't be exercised here - they're written to be run on your own
machine, where yfinance works normally. The rest of the library only
depends on getting a DataFrame with ['Open','High','Low','Close','Volume']
columns and a datetime index, so any other data source (a broker API,
Alpaca, ccxt for crypto, a CSV export) drops in just as easily.
"""

from __future__ import annotations
import pandas as pd


def fetch_historical(ticker: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
    """
    Download historical OHLCV data.

    period: e.g. '6mo', '1y', '2y', '5y', 'max'
    interval: e.g. '1d', '1h', '15m' (intraday intervals have shorter
        max history - yfinance enforces this)
    """
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


def fetch_latest(ticker: str, interval: str = '1d', lookback_period: str = '1y') -> pd.DataFrame:
    """
    Fetch a rolling window of the most recent bars - what a live monitor
    calls on every poll. `lookback_period` controls how much history comes
    back with it (the scanner needs enough bars before the newest one to
    find pivots and a preceding trend).
    """
    return fetch_historical(ticker, period=lookback_period, interval=interval)
