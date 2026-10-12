"""
Gap patterns: breakaway, runaway (continuation), and exhaustion gaps, plus
island reversals.

Unlike every other detector in this library, these don't start from
zigzag pivots - a gap is a bar-to-bar event (today's range doesn't
overlap yesterday's at all), so the whole series gets scanned bar by bar
instead. Classifying WHICH kind of gap it is still leans on the zigzag
pivots, though, since that's what tells us where the gap sits relative
to the current trend.
"""

from __future__ import annotations
import pandas as pd

from .patterns import PatternMatch
from .pivots import find_zigzag_pivots, prior_trend


def find_gaps(df: pd.DataFrame, min_gap_pct: float = 0.01) -> list[dict]:
    """
    Every bar-to-bar gap at least `min_gap_pct` wide: today's Low sitting
    entirely above yesterday's High (a gap up), or today's High sitting
    entirely below yesterday's Low (a gap down). Raw events, not yet
    classified into a named pattern.
    """
    gaps = []
    for i in range(1, len(df)):
        prev_high = df['High'].iloc[i - 1]
        prev_low = df['Low'].iloc[i - 1]
        today_high = df['High'].iloc[i]
        today_low = df['Low'].iloc[i]

        if today_low > prev_high:
            size = (today_low - prev_high) / prev_high
            if size >= min_gap_pct:
                gaps.append({'idx': i, 'date': df.index[i], 'direction': 'up',
                             'from': prev_high, 'to': today_low, 'size_pct': size})
        elif today_high < prev_low:
            size = (prev_low - today_high) / prev_low
            if size >= min_gap_pct:
                gaps.append({'idx': i, 'date': df.index[i], 'direction': 'down',
                             'from': prev_low, 'to': today_high, 'size_pct': size})
    return gaps


def classify_gaps(df: pd.DataFrame, gaps: list[dict] | None = None, pct_threshold: float = 0.03,
                   min_gap_pct: float = 0.01, trend_lookback: int = 15) -> list[PatternMatch]:
    """
    Classifies each raw gap by WHERE it sits in the current swing (using
    ordinary zigzag pivots at `pct_threshold` to know how far into the
    swing we already are):

    - breakaway: within a handful of bars of the swing's last pivot - the
      gap that kicks a new leg off
    - exhaustion: many bars into an already-extended swing (>=15% moved
      since the last pivot) - often precedes a reversal, a "blow-off"
    - runaway (continuation): in the middle of an established swing -
      neither just starting nor obviously overextended
    """
    if gaps is None:
        gaps = find_gaps(df, min_gap_pct)
    pivots = find_zigzag_pivots(df, pct_threshold)
    matches = []

    for gap in gaps:
        prior_pivots = pivots[pivots['idx'] < gap['idx']]
        if len(prior_pivots) == 0:
            continue
        last_pivot = prior_pivots.iloc[-1]
        bars_since = gap['idx'] - int(last_pivot['idx'])
        move_since_pct = abs(df['Close'].iloc[gap['idx']] - last_pivot['price']) / last_pivot['price']

        if bars_since <= 5:
            kind = 'breakaway'
        elif move_since_pct >= 0.15:
            kind = 'exhaustion'
        else:
            kind = 'runaway'

        matches.append(PatternMatch(
            pattern=f'{kind}_gap',
            points=[(gap['idx'], gap['date'], gap['to'], gap['direction'])],
            lines={},
            breakout=None,
            target=None,
            failed=None,
            meta={'gap': gap, 'bars_since_pivot': bars_since, 'move_since_pivot_pct': move_since_pct},
        ))

    return matches


def detect_island_reversal(df: pd.DataFrame, max_island_bars: int = 10,
                            min_gap_pct: float = 0.01) -> list[PatternMatch]:
    """
    Two gaps in OPPOSITE directions within a short span isolate a small
    cluster of bars - the "island" - cut off from the price action on
    both sides. A gap up then a gap down (bracketing a brief high plateau)
    is an island top; the mirror is an island bottom.
    """
    gaps = find_gaps(df, min_gap_pct)
    matches = []

    for i in range(len(gaps) - 1):
        g1, g2 = gaps[i], gaps[i + 1]
        if g1['direction'] == g2['direction']:
            continue
        span = g2['idx'] - g1['idx']
        if span < 1 or span > max_island_bars:
            continue

        kind = 'island_top' if g1['direction'] == 'up' else 'island_bottom'
        matches.append(PatternMatch(
            pattern=kind,
            points=[(g1['idx'], g1['date'], g1['to'], 'gap'),
                    (g2['idx'], g2['date'], g2['to'], 'gap')],
            lines={},
            breakout={'idx': g2['idx'], 'date': g2['date'], 'price': df['Close'].iloc[g2['idx']],
                      'direction': 'down' if kind == 'island_top' else 'up'},
            target=None,
            failed=None,
            meta={'island_bars': span},
        ))

    return matches
