"""
Pattern matchers.

Each detector takes the zigzag pivot sequence (see pivots.py) plus the
underlying OHLC DataFrame, and returns a list of PatternMatch objects for
every place the pattern's geometric rules are satisfied. A match carries
everything needed to plot it: the defining points, fitted trendlines, and
(if price has moved past the formation) the breakout details.

Breakout confirmation always uses the CLOSING price against the boundary,
never an intrabar wick, matching the standard rule that a mere poke through
a level doesn't count.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class PatternMatch:
    pattern: str                    # 'broadening_bottom' | 'head_and_shoulders_top'
    points: list                    # list of (idx, date, price, kind) defining the pattern
    lines: dict                     # named trendlines, each (x1, y1, x2, y2)
    breakout: dict | None = None    # {'idx','date','price','direction'} or None if not yet broken out
    target: float | None = None     # measured-move price target
    failed: bool | None = None      # True/False once we know; None if undetermined yet
    meta: dict = field(default_factory=dict)


def _fit_line(xs, ys):
    """Least-squares line through points; returns (slope, intercept)."""
    slope, intercept = np.polyfit(xs, ys, 1)
    return slope, intercept


def _line_y_at(slope, intercept, x):
    return slope * x + intercept


def _find_breakout(df: pd.DataFrame, after_idx: int, level_fn, direction: str, max_bars: int = 60):
    """
    Scan forward from `after_idx` for the first CLOSE that crosses the
    boundary defined by level_fn(bar_idx) -> price, in the given direction
    ('up' or 'down'). Returns a breakout dict or None if not (yet) found.
    """
    n = len(df)
    end = min(n, after_idx + 1 + max_bars)
    for i in range(after_idx + 1, end):
        level = level_fn(i)
        close = df['Close'].iloc[i]
        if direction == 'up' and close > level:
            return {'idx': i, 'date': df.index[i], 'price': close, 'direction': 'up'}
        if direction == 'down' and close < level:
            return {'idx': i, 'date': df.index[i], 'price': close, 'direction': 'down'}
    return None


def _check_failure(df: pd.DataFrame, breakout: dict, threshold: float = 0.05, max_bars: int = 40) -> bool | None:
    """
    A confirmed breakout is still classed a "failure" if price doesn't move
    at least `threshold` (5% by default) beyond the breakout price before
    reversing. Returns True (failed), False (held), or None (not enough
    bars yet to tell).
    """
    n = len(df)
    start = breakout['idx']
    end = min(n, start + max_bars)
    if end - start < max_bars and end == n:
        pass  # we'll still answer with what we have; caller can re-check later
    price0 = breakout['price']
    best = price0
    for i in range(start, end):
        c = df['Close'].iloc[i]
        if breakout['direction'] == 'up':
            best = max(best, c)
            if (best - price0) / price0 >= threshold:
                return False
        else:
            best = min(best, c)
            if (price0 - best) / price0 >= threshold:
                return False
    if end - start < max_bars:
        return None  # too soon to judge
    return True


def detect_broadening_bottom(pivots: pd.DataFrame, df: pd.DataFrame,
                              min_touches: int = 4, trend_lookback: int = 15) -> list[PatternMatch]:
    """
    A broadening bottom is: alternating swing points, preceded by a
    downtrend, where each swing low is LOWER than the one before it and
    each swing high is HIGHER than the one before it (diverging support and
    resistance lines - a widening funnel rather than a converging triangle).
    """
    matches = []
    kinds = pivots['kind'].tolist()

    for start in range(len(pivots) - min_touches + 1):
        window = pivots.iloc[start:start + min_touches].reset_index(drop=True)
        seq = window['kind'].tolist()

        if any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
            continue  # must strictly alternate

        lows = window[window.kind == 'low']
        highs = window[window.kind == 'high']
        if len(lows) < 2 or len(highs) < 2:
            continue

        low_prices = lows['price'].tolist()
        high_prices = highs['price'].tolist()
        widening = all(low_prices[i] > low_prices[i + 1] for i in range(len(low_prices) - 1)) and \
                   all(high_prices[i] < high_prices[i + 1] for i in range(len(high_prices) - 1))
        if not widening:
            continue

        first_idx = int(window.iloc[0]['idx'])
        from .pivots import prior_trend
        if prior_trend(df, first_idx, trend_lookback) != 'down':
            continue

        # Fit the two diverging trendlines
        r_slope, r_int = _fit_line(highs['idx'].values, highs['price'].values)   # resistance (rising)
        s_slope, s_int = _fit_line(lows['idx'].values, lows['price'].values)     # support (falling)

        last_idx = int(window.iloc[-1]['idx'])
        highest_high = high_prices[-1] if window.iloc[-1]['kind'] == 'high' else max(high_prices)
        lowest_low = min(low_prices)
        formation_height = max(high_prices) - lowest_low

        # Breakout: close beyond the horizontal line extended from the
        # highest high (up) or lowest low (down) - the standard fallback
        # when price rides a diagonal without ever cleanly piercing it.
        up_breakout = _find_breakout(df, last_idx, lambda i: max(high_prices), 'up')
        down_breakout = _find_breakout(df, last_idx, lambda i: lowest_low, 'down')

        breakout = None
        if up_breakout and down_breakout:
            breakout = up_breakout if up_breakout['idx'] < down_breakout['idx'] else down_breakout
        else:
            breakout = up_breakout or down_breakout

        target = None
        failed = None
        if breakout:
            if breakout['direction'] == 'up':
                target = (window.iloc[-1]['price'] + formation_height if window.iloc[-1]['kind'] == 'low'
                          else max(high_prices) + formation_height)
            else:
                target = min(low_prices) - formation_height
            failed = _check_failure(df, breakout)

        matches.append(PatternMatch(
            pattern='broadening_bottom',
            points=list(window.itertuples(index=False, name=None)),
            lines={
                'resistance': (highs['idx'].iloc[0], _line_y_at(r_slope, r_int, highs['idx'].iloc[0]),
                               highs['idx'].iloc[-1], _line_y_at(r_slope, r_int, highs['idx'].iloc[-1])),
                'support': (lows['idx'].iloc[0], _line_y_at(s_slope, s_int, lows['idx'].iloc[0]),
                            lows['idx'].iloc[-1], _line_y_at(s_slope, s_int, lows['idx'].iloc[-1])),
            },
            breakout=breakout,
            target=target,
            failed=failed,
            meta={'formation_height': formation_height},
        ))

    return matches


def detect_head_and_shoulders_top(pivots: pd.DataFrame, df: pd.DataFrame,
                                   shoulder_tol: float = 0.15, neckline_tol: float = 0.06,
                                   trend_lookback: int = 15) -> list[PatternMatch]:
    """
    A head-and-shoulders top is 5 consecutive alternating pivots
    high-low-high-low-high where: the middle high (head) is taller than
    both outer highs (shoulders); the two shoulders are roughly comparable
    heights; the two troughs (neckline points) are roughly level; and the
    whole thing is preceded by an uptrend.
    """
    matches = []

    for start in range(len(pivots) - 4):
        window = pivots.iloc[start:start + 5].reset_index(drop=True)
        seq = window['kind'].tolist()
        if seq != ['high', 'low', 'high', 'low', 'high']:
            continue

        l_sh, t1, head, t2, r_sh = window['price'].tolist()

        if not (head > l_sh and head > r_sh):
            continue
        shoulder_diff = abs(l_sh - r_sh) / max(l_sh, r_sh)
        if shoulder_diff > shoulder_tol:
            continue
        neckline_diff = abs(t1 - t2) / max(t1, t2)
        if neckline_diff > neckline_tol:
            continue

        first_idx = int(window.iloc[0]['idx'])
        from .pivots import prior_trend
        if prior_trend(df, first_idx, trend_lookback) != 'up':
            continue

        n_slope, n_int = _fit_line(
            [window.iloc[1]['idx'], window.iloc[3]['idx']],
            [t1, t2],
        )
        last_idx = int(window.iloc[-1]['idx'])
        breakout = _find_breakout(df, last_idx, lambda i: _line_y_at(n_slope, n_int, i), 'down')

        target = None
        failed = None
        if breakout:
            neckline_at_break = _line_y_at(n_slope, n_int, breakout['idx'])
            target = neckline_at_break - (head - (t1 + t2) / 2)
            failed = _check_failure(df, breakout)

        matches.append(PatternMatch(
            pattern='head_and_shoulders_top',
            points=list(window.itertuples(index=False, name=None)),
            lines={'neckline': (window.iloc[1]['idx'], t1, window.iloc[3]['idx'], t2)},
            breakout=breakout,
            target=target,
            failed=failed,
            meta={},
        ))

    return matches


def scan(df: pd.DataFrame, pct_threshold: float = 0.03) -> list[PatternMatch]:
    """Run every detector over one OHLC DataFrame and return all matches."""
    from .pivots import find_zigzag_pivots
    pivots = find_zigzag_pivots(df, pct_threshold)
    matches = []
    matches += detect_broadening_bottom(pivots, df)
    matches += detect_head_and_shoulders_top(pivots, df)
    return matches
