"""
Composite patterns: diamond, horn, pipe.

These were named in the original family taxonomy but never built - they're
"composite" because each is a tight variant or combination of primitives
that already exist elsewhere in this library, not a new detection method:

- A diamond is a broadening formation (diverging lines) immediately
  followed by a converging one - drawing all four boundary lines traces a
  rough diamond/rhombus. Detected here by checking that the width between
  paired touches increases then decreases across a window, then fitting
  the narrowing second half the same way formations.py fits a triangle.
- A horn is a double top/bottom (extremes.py) with its two touches
  unusually close together in time - same detector, tighter spacing.
- A pipe is an even tighter version of the same idea - two adjacent,
  narrow (Adam-shaped) spikes only a few bars apart, typically on weekly
  charts in the book but expressed here in bars, whatever timeframe the
  data is in.
"""

from __future__ import annotations
import pandas as pd

from .patterns import PatternMatch, _find_breakout, _check_failure
from .pivots import prior_trend
from .extremes import detect_double_extreme, classify_touch_shape


def detect_diamond(pivots: pd.DataFrame, df: pd.DataFrame, kind: str = 'top',
                    trend_lookback: int = 15) -> list[PatternMatch]:
    """
    5 alternating pivots (touch, mid, touch, mid, touch - e.g. high, low,
    high, low, high for a top). Diamonds widen then narrow, so the leg
    between pivots should grow then shrink - the biggest swing should sit
    in the middle of the window, not at either end. Breakout follows the
    same convention as every other reversal pattern here: a top resolves
    down, a bottom resolves up, confirmed against the level of the last
    "mid" touch (the narrowing pattern's final neckline).
    """
    matches = []
    touch_kind = 'high' if kind == 'top' else 'low'
    other_kind = 'low' if kind == 'top' else 'high'
    expected = [touch_kind, other_kind, touch_kind, other_kind, touch_kind]
    required_trend = 'up' if kind == 'top' else 'down'
    direction = 'down' if kind == 'top' else 'up'

    for start in range(len(pivots) - 4):
        window = pivots.iloc[start:start + 5].reset_index(drop=True)
        if window['kind'].tolist() != expected:
            continue

        prices = window['price'].tolist()
        legs = [abs(prices[i + 1] - prices[i]) for i in range(4)]
        if legs.index(max(legs)) not in (1, 2):
            continue  # the widest swing must sit in the middle, not at either end

        first_idx = int(window.iloc[0]['idx'])
        if prior_trend(df, first_idx, trend_lookback) != required_trend:
            continue

        last_idx = int(window.iloc[-1]['idx'])
        confirmation_level = prices[3]  # the last "mid" touch - the narrowing neckline
        breakout = _find_breakout(df, last_idx, lambda i: confirmation_level, direction)

        target = None
        failed = None
        height = max(legs)
        if breakout:
            target = breakout['price'] - height if direction == 'down' else breakout['price'] + height
            failed = _check_failure(df, breakout)

        matches.append(PatternMatch(
            pattern=f'diamond_{kind}',
            points=list(window.itertuples(index=False, name=None)),
            lines={'confirmation': (window.iloc[3]['idx'], confirmation_level,
                                     last_idx, confirmation_level)},
            breakout=breakout,
            target=target,
            failed=failed,
            meta={'legs': legs},
        ))

    return matches


def detect_horn(pivots: pd.DataFrame, df: pd.DataFrame, kind: str = 'top',
                 max_bars_between: int = 15, tolerance: float = 0.04,
                 trend_lookback: int = 15) -> list[PatternMatch]:
    """A double top/bottom with its two touches unusually close together
    in time - same shape as extremes.detect_double_extreme(), just with a
    tight spacing requirement added and renamed."""
    base = detect_double_extreme(pivots, df, kind=kind, tolerance=tolerance,
                                  trend_lookback=trend_lookback)
    matches = []
    for m in base:
        span = m.points[-1][0] - m.points[0][0]
        if span > max_bars_between:
            continue
        m.pattern = f'horn_{kind}'
        matches.append(m)
    return matches


def detect_pipe(pivots: pd.DataFrame, df: pd.DataFrame, kind: str = 'top',
                 max_bars_between: int = 8, tolerance: float = 0.03,
                 trend_lookback: int = 15) -> list[PatternMatch]:
    """An even tighter horn: two adjacent, narrow (Adam-shaped) spikes
    only a few bars apart. Bulkowski describes these on weekly charts
    specifically (2 weeks apart); expressed here in bars, so a couple of
    bars is the equivalent spacing whatever timeframe the data is in."""
    base = detect_double_extreme(pivots, df, kind=kind, tolerance=tolerance,
                                  trend_lookback=trend_lookback)
    touch_kind = 'high' if kind == 'top' else 'low'
    matches = []
    for m in base:
        span = m.points[-1][0] - m.points[0][0]
        if span > max_bars_between:
            continue
        shape1 = classify_touch_shape(df, m.points[0][0], touch_kind)
        shape2 = classify_touch_shape(df, m.points[-1][0], touch_kind)
        if shape1 != 'Adam' or shape2 != 'Adam':
            continue  # a pipe is narrow on both sides, not a rounded touch
        m.pattern = f'pipe_{kind}'
        matches.append(m)
    return matches
