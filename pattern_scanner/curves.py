"""
Curved shapes: rounding bottoms/tops and cup with handle.

These patterns are smooth bowl or dome shapes rather than a sequence of
sharp zigzag pivots, so they need a different primitive entirely: fit a
parabola (quadratic curve) to a sliding window of closing prices, and
check both that it opens the right way (upward for a bottom, downward for
a top) and that it actually fits well (a high R²) - a genuinely curved
run of prices, not just noisy chop that happens to trend one way then
the other.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .patterns import PatternMatch, _find_breakout, _check_failure


def _fit_quadratic(y: np.ndarray) -> tuple[np.ndarray, float]:
    """Fits y = a*x^2 + b*x + c over x = 0..len(y)-1. Returns (coeffs, r_squared)."""
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 2)
    fitted = np.polyval(coeffs, x)
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coeffs, r_squared


def detect_rounding(df: pd.DataFrame, kind: str = 'bottom', min_window: int = 20, max_window: int = 90,
                     step: int = 5, min_r_squared: float = 0.6, vertex_margin: float = 0.2
                     ) -> list[PatternMatch]:
    """
    Slides across `df` and, at each start position, fits a parabola to
    Close prices at a few different window sizes, keeping the
    best-fitting one that's a genuine bowl (kind='bottom') or dome
    (kind='top'): the right curvature sign, a decent fit (R²), and a
    vertex that falls roughly in the middle of the window rather than at
    an edge (which would just be a trend, not a turn).

    Confirmation ("breakout") is a close back above (bottom) or below
    (top) the price level the curve started at - the rim, in cup terms -
    matching Bulkowski's rounding-pattern convention.
    """
    matches = []
    n = len(df)
    claimed_until = -1

    for start in range(0, n - min_window):
        if start < claimed_until:
            continue

        best = None
        for window_size in range(min_window, max_window + 1, step):
            end = start + window_size
            if end > n:
                break
            y = df['Close'].iloc[start:end].values
            coeffs, r2 = _fit_quadratic(y)
            a, b, c = coeffs
            if r2 < min_r_squared:
                continue
            if kind == 'bottom' and a <= 0:
                continue
            if kind == 'top' and a >= 0:
                continue
            vertex_x = -b / (2 * a)
            if not (window_size * vertex_margin <= vertex_x <= window_size * (1 - vertex_margin)):
                continue
            if best is None or r2 > best[1]:
                best = (window_size, r2, vertex_x)

        if best is None:
            continue
        window_size, r2, vertex_x = best
        first_idx = start
        last_idx = start + window_size - 1
        vertex_idx = int(start + round(vertex_x))
        rim_level = df['Close'].iloc[first_idx]
        direction = 'up' if kind == 'bottom' else 'down'
        breakout = _find_breakout(df, last_idx, lambda i: rim_level, direction)

        depth = abs(df['Close'].iloc[vertex_idx] - rim_level)
        target = None
        failed = None
        if breakout:
            target = breakout['price'] + depth if direction == 'up' else breakout['price'] - depth
            failed = _check_failure(df, breakout)

        matches.append(PatternMatch(
            pattern=f'rounding_{kind}',
            points=[
                (first_idx, df.index[first_idx], df['Close'].iloc[first_idx], kind),
                (vertex_idx, df.index[vertex_idx], df['Close'].iloc[vertex_idx], kind),
                (last_idx, df.index[last_idx], df['Close'].iloc[last_idx], kind),
            ],
            lines={'rim': (first_idx, rim_level, last_idx, rim_level)},
            breakout=breakout,
            target=target,
            failed=failed,
            meta={'r_squared': r2, 'window_size': window_size, 'depth': depth},
        ))
        claimed_until = (breakout['idx'] if breakout else last_idx) + 1

    return matches


def detect_scallops(df: pd.DataFrame, kind: str = 'ascending', **rounding_kwargs) -> list[PatternMatch]:
    """
    Repeated rounding bottoms, each higher than the last (kind='ascending'
    - a staircase of 'J' shapes), or repeated rounding tops each lower
    than the last (kind='descending'). Built on detect_rounding() rather
    than a new curve-fit - a scallop sequence is just consecutive rounding
    matches whose rims trend the right direction, chained together.
    """
    sub_kind = 'bottom' if kind == 'ascending' else 'top'
    rounds = sorted(detect_rounding(df, kind=sub_kind, **rounding_kwargs),
                     key=lambda m: m.points[0][0])
    matches = []
    i = 0
    while i < len(rounds):
        chain = [rounds[i]]
        j = i + 1
        while j < len(rounds):
            prev_rim = chain[-1].lines['rim'][1]
            cur_rim = rounds[j].lines['rim'][1]
            improves = (cur_rim > prev_rim) if kind == 'ascending' else (cur_rim < prev_rim)
            if improves:
                chain.append(rounds[j])
                j += 1
            else:
                break
        if len(chain) >= 2:
            points = [p for m in chain for p in m.points]
            last = chain[-1]
            matches.append(PatternMatch(
                pattern=f'scallop_{kind}',
                points=points,
                lines={},
                breakout=last.breakout,
                target=last.target,
                failed=last.failed,
                meta={'n_scallops': len(chain)},
            ))
            i = j if j > i + 1 else i + 1
        else:
            i += 1
    return matches


def detect_cup_with_handle(df: pd.DataFrame, handle_max_bars: int = 20,
                            handle_max_retrace: float = 0.5, **rounding_kwargs) -> list[PatternMatch]:
    """
    A cup with handle is a rounding bottom (the "cup") followed by a
    smaller pullback (the "handle") near the cup's right rim, then a
    breakout above the rim. Built on top of detect_rounding() rather than
    re-deriving the curve fit - the cup IS a rounding bottom, this just
    adds the handle check and re-does the breakout search from after it.
    """
    cups = detect_rounding(df, kind='bottom', **rounding_kwargs)
    matches = []

    for cup in cups:
        cup_last_idx = cup.points[-1][0]
        rim_level = cup.lines['rim'][1]
        cup_depth = cup.meta['depth']
        if cup_depth <= 0:
            continue

        end = min(len(df), cup_last_idx + handle_max_bars + 1)
        handle_df = df.iloc[cup_last_idx:end]
        if len(handle_df) < 3:
            continue
        handle_low = handle_df['Low'].min()
        handle_low_pos = int(handle_df['Low'].values.argmin())
        handle_low_idx = cup_last_idx + handle_low_pos

        handle_retrace = (rim_level - handle_low) / cup_depth
        if handle_retrace <= 0 or handle_retrace > handle_max_retrace:
            continue  # no real pullback, or the "handle" is as deep as the cup itself

        breakout = _find_breakout(df, handle_low_idx, lambda i: rim_level, 'up')
        target = None
        failed = None
        if breakout:
            target = breakout['price'] + cup_depth
            failed = _check_failure(df, breakout)

        matches.append(PatternMatch(
            pattern='cup_with_handle',
            points=cup.points + [(handle_low_idx, df.index[handle_low_idx], handle_low, 'handle')],
            lines={'rim': cup.lines['rim']},
            breakout=breakout,
            target=target,
            failed=failed,
            meta={'cup_r_squared': cup.meta['r_squared'], 'handle_retrace': handle_retrace},
        ))

    return matches
