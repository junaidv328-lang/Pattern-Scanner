"""
Leg-structure patterns: measured moves, three rising valleys, three
falling peaks.

Unlike the two-line formations (fit a boundary line) or the multi-touch
extremes (compare touches at a similar price level), these patterns are
about comparing the MAGNITUDE of consecutive zigzag legs to each other:
is the second leg of a move roughly the same size as the first? are three
consecutive lows each higher than the last?
"""

from __future__ import annotations
import pandas as pd

from .patterns import PatternMatch, _find_breakout


def detect_measured_move(pivots: pd.DataFrame, df: pd.DataFrame, direction: str = 'up',
                          min_leg_pct: float = 0.08, leg_similarity_tol: float = 0.5,
                          min_retracement: float = 0.15, max_retracement: float = 0.75
                          ) -> list[PatternMatch]:
    """
    direction='up': low (A) -> high (B) [first leg] -> low (C, above A)
    [corrective phase - a PARTIAL retracement of the first leg] -> high
    (D) [second leg, roughly the same size as the first]. direction='down'
    is the mirror image. Unlike the reversal patterns elsewhere in this
    library, a measured move is a trend-CONTINUATION structure - there's
    no separate breakout to confirm, the pattern is just read once the
    second leg matches the first closely enough.
    """
    matches = []
    touch_kind = 'low' if direction == 'up' else 'high'
    other_kind = 'high' if direction == 'up' else 'low'

    for start in range(len(pivots) - 3):
        window = pivots.iloc[start:start + 4].reset_index(drop=True)
        if window['kind'].tolist() != [touch_kind, other_kind, touch_kind, other_kind]:
            continue

        a, b, c, d = window['price'].tolist()
        if direction == 'up':
            leg1, correction, leg2 = b - a, b - c, d - c
        else:
            leg1, correction, leg2 = a - b, c - b, c - d

        if leg1 <= 0 or leg2 <= 0 or correction <= 0:
            continue
        if leg1 / a < min_leg_pct:
            continue
        retracement_pct = correction / leg1
        if not (min_retracement <= retracement_pct <= max_retracement):
            continue
        if abs(leg2 - leg1) / leg1 > leg_similarity_tol:
            continue

        c_idx, c_price = int(window.iloc[2]['idx']), c
        target = c_price + leg1 if direction == 'up' else c_price - leg1

        matches.append(PatternMatch(
            pattern=f'measured_move_{direction}',
            points=list(window.itertuples(index=False, name=None)),
            lines={
                'leg1': (window.iloc[0]['idx'], a, window.iloc[1]['idx'], b),
                'leg2': (window.iloc[2]['idx'], c, window.iloc[3]['idx'], d),
            },
            breakout=None,
            target=target,
            failed=None,
            meta={'leg1_height': leg1, 'leg2_height': leg2, 'retracement_pct': retracement_pct},
        ))

    return matches


def detect_three_rising_valleys(pivots: pd.DataFrame, df: pd.DataFrame) -> list[PatternMatch]:
    """Three consecutive, strictly HIGHER lows (a single rising support
    line, not a level-matched touch like a triple bottom), with a
    breakout above the highest of the two intervening highs."""
    matches = []
    for start in range(len(pivots) - 4):
        window = pivots.iloc[start:start + 5].reset_index(drop=True)
        if window['kind'].tolist() != ['low', 'high', 'low', 'high', 'low']:
            continue
        lows = [window.iloc[0]['price'], window.iloc[2]['price'], window.iloc[4]['price']]
        if not (lows[0] < lows[1] < lows[2]):
            continue
        highs = [window.iloc[1]['price'], window.iloc[3]['price']]

        last_idx = int(window.iloc[-1]['idx'])
        confirmation_level = max(highs)
        breakout = _find_breakout(df, last_idx, lambda i: confirmation_level, 'up')

        matches.append(PatternMatch(
            pattern='three_rising_valleys',
            points=list(window.itertuples(index=False, name=None)),
            lines={'confirmation': (window.iloc[1]['idx'], confirmation_level,
                                     last_idx, confirmation_level)},
            breakout=breakout,
            target=(breakout['price'] + (lows[2] - lows[0]) if breakout else None),
            failed=None,
            meta={},
        ))
    return matches


def detect_three_falling_peaks(pivots: pd.DataFrame, df: pd.DataFrame) -> list[PatternMatch]:
    """Mirror image of detect_three_rising_valleys: three consecutive,
    strictly LOWER highs, breakout below the lowest intervening low."""
    matches = []
    for start in range(len(pivots) - 4):
        window = pivots.iloc[start:start + 5].reset_index(drop=True)
        if window['kind'].tolist() != ['high', 'low', 'high', 'low', 'high']:
            continue
        highs = [window.iloc[0]['price'], window.iloc[2]['price'], window.iloc[4]['price']]
        if not (highs[0] > highs[1] > highs[2]):
            continue
        lows = [window.iloc[1]['price'], window.iloc[3]['price']]

        last_idx = int(window.iloc[-1]['idx'])
        confirmation_level = min(lows)
        breakout = _find_breakout(df, last_idx, lambda i: confirmation_level, 'down')

        matches.append(PatternMatch(
            pattern='three_falling_peaks',
            points=list(window.itertuples(index=False, name=None)),
            lines={'confirmation': (window.iloc[1]['idx'], confirmation_level,
                                     last_idx, confirmation_level)},
            breakout=breakout,
            target=(breakout['price'] - (highs[0] - highs[2]) if breakout else None),
            failed=None,
            meta={},
        ))
    return matches
