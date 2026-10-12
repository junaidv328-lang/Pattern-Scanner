"""
Multi-touch extremes: double tops/bottoms, triple tops/bottoms, and
head-and-shoulders (simple and complex).

These patterns share a different shape than the two-line formations in
formations.py: instead of fitting a boundary LINE through several points,
they're about N peaks (or troughs) landing at a similar price level, with
a horizontal "confirmation line" at the level of the point(s) between
them. Breakout is a close beyond that line - the same closing-price rule
as everywhere else in this library.

Head-and-shoulders is the one exception where the touches are NOT all
equal height - the middle one is taller (top) or lower (bottom) than the
others - so it gets its own rule. That rule is written once, generalized
to any number of shoulders per side, covering both the simple 3-peak
version and Bulkowski's "complex" multi-shoulder version with the same
code, the same way formations.py collapsed 16 named patterns into one
parameterized detector.

Double and triple tops/bottoms are further classified per touch as "Adam"
(a narrow, spike-like touch - often a single bar) or "Eve" (a wide,
rounded touch spanning several bars), matching Bulkowski's naming
(Adam & Eve, Eve & Eve, Adam & Adam...) - see classify_touch_shape().
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .patterns import PatternMatch, _fit_line, _line_y_at, _find_breakout, _check_failure
from .pivots import prior_trend


def classify_touch_shape(df: pd.DataFrame, idx: int, kind: str,
                          tolerance_pct: float = 0.015, window: int = 4) -> str:
    """
    'Adam' if only a bar or two near this touch come close to its extreme
    price (a narrow spike); 'Eve' if several bars linger near it (a wide,
    rounded touch). Counts bars within `tolerance_pct` of the extreme,
    looking `window` bars either side.
    """
    n = len(df)
    lo = max(0, idx - window)
    hi = min(n, idx + window + 1)
    local = df.iloc[lo:hi]
    if kind == 'high':
        extreme = df['High'].iloc[idx]
        near = (local['High'] >= extreme * (1 - tolerance_pct)).sum()
    else:
        extreme = df['Low'].iloc[idx]
        near = (local['Low'] <= extreme * (1 + tolerance_pct)).sum()
    return 'Eve' if near >= 4 else 'Adam'


def detect_double_extreme(pivots: pd.DataFrame, df: pd.DataFrame, kind: str = 'top',
                           tolerance: float = 0.04, trend_lookback: int = 15) -> list[PatternMatch]:
    """
    kind='top': two highs at a similar level with one low between them
    (peak, trough, peak) - a double top. kind='bottom': the mirror image
    (trough, peak, trough) - a double bottom. Preceded by an uptrend
    (tops) or downtrend (bottoms), same as every reversal pattern here.
    """
    matches = []
    touch_kind = 'high' if kind == 'top' else 'low'
    other_kind = 'low' if kind == 'top' else 'high'
    required_trend = 'up' if kind == 'top' else 'down'
    direction = 'down' if kind == 'top' else 'up'

    for start in range(len(pivots) - 2):
        window = pivots.iloc[start:start + 3].reset_index(drop=True)
        if window['kind'].tolist() != [touch_kind, other_kind, touch_kind]:
            continue

        p1, mid, p2 = window.iloc[0], window.iloc[1], window.iloc[2]
        diff = abs(p1['price'] - p2['price']) / max(p1['price'], p2['price'])
        if diff > tolerance:
            continue

        first_idx = int(p1['idx'])
        if prior_trend(df, first_idx, trend_lookback) != required_trend:
            continue

        confirmation_level = mid['price']
        last_idx = int(p2['idx'])
        breakout = _find_breakout(df, last_idx, lambda i: confirmation_level, direction)

        shape1 = classify_touch_shape(df, int(p1['idx']), touch_kind)
        shape2 = classify_touch_shape(df, int(p2['idx']), touch_kind)
        name = f"double_{kind}_{shape1.lower()}_{shape2.lower()}"

        target = None
        failed = None
        height = abs(p1['price'] - confirmation_level)
        if breakout:
            target = breakout['price'] - height if direction == 'down' else breakout['price'] + height
            failed = _check_failure(df, breakout)

        matches.append(PatternMatch(
            pattern=name,
            points=list(window.itertuples(index=False, name=None)),
            lines={'confirmation': (first_idx, confirmation_level, last_idx, confirmation_level)},
            breakout=breakout,
            target=target,
            failed=failed,
            meta={'shapes': (shape1, shape2)},
        ))

    return matches


def detect_triple_extreme(pivots: pd.DataFrame, df: pd.DataFrame, kind: str = 'top',
                           tolerance: float = 0.04, trend_lookback: int = 15) -> list[PatternMatch]:
    """Same idea as detect_double_extreme, with three touches instead of two."""
    matches = []
    touch_kind = 'high' if kind == 'top' else 'low'
    other_kind = 'low' if kind == 'top' else 'high'
    required_trend = 'up' if kind == 'top' else 'down'
    direction = 'down' if kind == 'top' else 'up'

    for start in range(len(pivots) - 4):
        window = pivots.iloc[start:start + 5].reset_index(drop=True)
        expected = [touch_kind, other_kind, touch_kind, other_kind, touch_kind]
        if window['kind'].tolist() != expected:
            continue

        touches = [window.iloc[0], window.iloc[2], window.iloc[4]]
        prices = [t['price'] for t in touches]
        if (max(prices) - min(prices)) / max(prices) > tolerance:
            continue

        first_idx = int(touches[0]['idx'])
        if prior_trend(df, first_idx, trend_lookback) != required_trend:
            continue

        between_prices = [window.iloc[1]['price'], window.iloc[3]['price']]
        confirmation_level = min(between_prices) if kind == 'top' else max(between_prices)
        last_idx = int(touches[-1]['idx'])
        breakout = _find_breakout(df, last_idx, lambda i: confirmation_level, direction)

        target = None
        failed = None
        height = abs(prices[0] - confirmation_level)
        if breakout:
            target = breakout['price'] - height if direction == 'down' else breakout['price'] + height
            failed = _check_failure(df, breakout)

        matches.append(PatternMatch(
            pattern=f'triple_{kind}',
            points=list(window.itertuples(index=False, name=None)),
            lines={'confirmation': (first_idx, confirmation_level, last_idx, confirmation_level)},
            breakout=breakout,
            target=target,
            failed=failed,
            meta={},
        ))

    return matches


def detect_head_and_shoulders(pivots: pd.DataFrame, df: pd.DataFrame, kind: str = 'top',
                               max_shoulders: int = 3, shoulder_tol: float = 0.15,
                               neckline_tol: float = 0.06, trend_lookback: int = 15) -> list[PatternMatch]:
    """
    kind='top': a taller HEAD in the middle with 1+ shoulders of roughly
    comparable height on each side (troughs forming the neckline).
    kind='bottom': the mirror image (a lower head, with the neckline
    formed by the peaks between the shoulders).

    `max_shoulders` bounds how many shoulders per side to look for.
    max_shoulders=1 is the simple, textbook 3-peak pattern; anything more
    is Bulkowski's "complex" head-and-shoulders with multiple shoulders -
    same rule, just a longer window.
    """
    matches = []
    touch_kind = 'high' if kind == 'top' else 'low'
    other_kind = 'low' if kind == 'top' else 'high'
    required_trend = 'up' if kind == 'top' else 'down'
    direction = 'down' if kind == 'top' else 'up'

    for n_each_side in range(1, max_shoulders + 1):
        size = 4 * n_each_side + 1  # n=1 -> 5 pivots, n=2 -> 9, n=3 -> 13...
        expected = [touch_kind if i % 2 == 0 else other_kind for i in range(size)]

        for start in range(len(pivots) - size + 1):
            window = pivots.iloc[start:start + size].reset_index(drop=True)
            if window['kind'].tolist() != expected:
                continue

            touch_prices = [window.iloc[i]['price'] for i in range(0, size, 2)]
            trough_prices = [window.iloc[i]['price'] for i in range(1, size, 2)]
            trough_idxs = [int(window.iloc[i]['idx']) for i in range(1, size, 2)]
            head_pos = len(touch_prices) // 2
            head = touch_prices[head_pos]
            shoulders = touch_prices[:head_pos] + touch_prices[head_pos + 1:]

            if kind == 'top':
                if not all(head > s for s in shoulders):
                    continue
            else:
                if not all(head < s for s in shoulders):
                    continue
            if shoulders and (max(shoulders) - min(shoulders)) / max(shoulders) > shoulder_tol:
                continue
            if (max(trough_prices) - min(trough_prices)) / max(trough_prices) > neckline_tol:
                continue

            first_idx = int(window.iloc[0]['idx'])
            if prior_trend(df, first_idx, trend_lookback) != required_trend:
                continue

            n_slope, n_int = _fit_line(np.array(trough_idxs), np.array(trough_prices))
            last_idx = int(window.iloc[-1]['idx'])
            breakout = _find_breakout(df, last_idx, lambda i: _line_y_at(n_slope, n_int, i), direction)

            target = None
            failed = None
            avg_trough = sum(trough_prices) / len(trough_prices)
            if breakout:
                neckline_at_break = _line_y_at(n_slope, n_int, breakout['idx'])
                height = abs(head - avg_trough)
                target = neckline_at_break - height if direction == 'down' else neckline_at_break + height
                failed = _check_failure(df, breakout)

            name = (f'head_and_shoulders_{kind}' if n_each_side == 1
                    else f'complex_head_and_shoulders_{kind}')

            matches.append(PatternMatch(
                pattern=name,
                points=list(window.itertuples(index=False, name=None)),
                lines={'neckline': (trough_idxs[0], trough_prices[0], trough_idxs[-1], trough_prices[-1])},
                breakout=breakout,
                target=target,
                failed=failed,
                meta={'n_shoulders_each_side': n_each_side},
            ))

    return matches
