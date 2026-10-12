"""
Swing point (pivot) detection.

Every chart pattern in this project is defined as a set of rules about a
sequence of alternating swing highs and swing lows (the "1, 2, 3, 4..."
points Bulkowski marks by hand). This module turns raw OHLC bars into that
sequence using a percentage-reversal zigzag algorithm: a swing point is only
confirmed once price has reversed by at least `pct_threshold` from the most
recent extreme in the opposite direction. This filters out normal day-to-day
noise and keeps only moves large enough to matter.
"""

from __future__ import annotations
import pandas as pd


def find_zigzag_pivots(df: pd.DataFrame, pct_threshold: float = 0.03) -> pd.DataFrame:
    """
    Identify alternating swing highs and swing lows in an OHLC DataFrame.

    Parameters
    ----------
    df : DataFrame with columns ['Open', 'High', 'Low', 'Close'] and a
        datetime-like index.
    pct_threshold : minimum reversal size (as a fraction, e.g. 0.03 = 3%)
        required to confirm a swing point. Smaller values find more, noisier
        pivots; larger values find fewer, more significant ones.

    Returns
    -------
    DataFrame with columns ['idx', 'date', 'price', 'kind'] where `idx` is
    the integer bar position in `df`, `kind` is 'high' or 'low', sorted
    chronologically. Consecutive rows always alternate kind.
    """
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    dates = df.index
    n = len(df)

    if n < 3:
        return pd.DataFrame(columns=['idx', 'date', 'price', 'kind'])

    pivots = []
    direction = None  # None until the first swing is confirmed, then 'up'/'down'

    # Confirmation arithmetic runs entirely on Close (both candidates below
    # use the same series) - mixing High for the up-candidate and Low for
    # the down-candidate seems more "accurate" but lets a single bar's
    # intrabar spread masquerade as a multi-bar swing, producing spurious
    # back-to-back pivots of the same kind. Once a pivot bar is identified,
    # we report its actual High (for a 'high') or Low (for a 'low'), not
    # its Close - that part still reflects the real touch.
    up_idx, up_price = 0, close[0]      # candidate for the next high pivot
    down_idx, down_price = 0, close[0]  # candidate for the next low pivot

    for i in range(1, n):
        c = close[i]

        if direction != 'down':
            if c > up_price:
                up_price, up_idx = c, i
            if (up_price - c) / up_price >= pct_threshold:
                pivots.append((up_idx, dates[up_idx], high[up_idx], 'high'))
                direction = 'down'
                down_price, down_idx = c, i
                continue

        if direction != 'up':
            if c < down_price:
                down_price, down_idx = c, i
            if (c - down_price) / down_price >= pct_threshold:
                pivots.append((down_idx, dates[down_idx], low[down_idx], 'low'))
                direction = 'up'
                up_price, up_idx = c, i

    out = pd.DataFrame(pivots, columns=['idx', 'date', 'price', 'kind'])
    out = out.drop_duplicates(subset='idx').sort_values('idx').reset_index(drop=True)
    return out


def prior_trend(df: pd.DataFrame, end_idx: int, lookback: int = 15) -> str:
    """
    Rough classification of the trend leading INTO a bar, used to check the
    "must be preceded by a downtrend / uptrend" requirement each pattern has.
    Compares the close `lookback` bars earlier to the close at `end_idx`.
    """
    start_idx = max(0, end_idx - lookback)
    if start_idx >= end_idx:
        return 'flat'
    start_close = df['Close'].iloc[start_idx]
    end_close = df['Close'].iloc[end_idx]
    change = (end_close - start_close) / start_close
    if change <= -0.02:
        return 'down'
    if change >= 0.02:
        return 'up'
    return 'flat'
