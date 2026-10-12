"""
Two-line formations: the "diverging / converging / parallel" family.

Every pattern here is the same underlying shape - two boundary lines fitted
through a window of alternating swing highs and swing lows - differing only
in three properties:

  1. the SLOPE of each line (rising / falling / flat)
  2. whether the lines DIVERGE, CONVERGE, or stay roughly PARALLEL
  3. a few optional preconditions (a required prior trend, a "flagpole"
     precondition for flag/pennant, a max duration for short-lived patterns)

That means broadening bottoms, right-angled broadening formations,
broadening wedges, all three triangle types, rising/falling wedges,
rectangles, flags, and pennants are all ONE detector fed a table of
definitions, rather than 15 separate pieces of geometry code.

Breakout confirmation still uses the closing price, same as the rest of
the library. For converging formations (triangles, wedges, pennants) the
search is capped at the lines' apex - Bulkowski notes patterns rarely
resolve meaningfully past the point the boundaries would cross, so a
"breakout" found beyond that point isn't treated as one.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .patterns import PatternMatch, _fit_line, _line_y_at, _check_failure
from .pivots import prior_trend, find_zigzag_pivots

FLAT_THRESHOLD = 0.0015      # slope smaller than this (as %/bar) counts as flat
CONVERGENCE_TOLERANCE = 0.15  # gap must change by more than this fraction to count


@dataclass
class FormationDef:
    name: str
    upper_slope: str            # 'up' | 'down' | 'flat'
    lower_slope: str            # 'up' | 'down' | 'flat'
    convergence: str            # 'diverging' | 'converging' | 'parallel'
    required_prior_trend: str | None = None   # 'up' | 'down' | None (no requirement)
    needs_flagpole: bool = False
    flagpole_min_pct: float = 0.15
    flagpole_lookback: int = 15
    max_duration_bars: int | None = None


FORMATION_DEFS = [
    # Flag/pennant first - they're the same geometry as rectangle/triangle
    # but with extra preconditions, so they must be checked before the
    # more generic definitions below or they'd never get a chance to match.
    FormationDef('flag_bullish', upper_slope='down', lower_slope='down', convergence='parallel',
                  required_prior_trend='up', needs_flagpole=True, max_duration_bars=15),
    FormationDef('flag_bearish', upper_slope='up', lower_slope='up', convergence='parallel',
                  required_prior_trend='down', needs_flagpole=True, max_duration_bars=15),
    FormationDef('pennant', upper_slope='down', lower_slope='up', convergence='converging',
                  needs_flagpole=True, max_duration_bars=15),

    FormationDef('broadening_bottom', upper_slope='up', lower_slope='down',
                  convergence='diverging', required_prior_trend='down'),
    FormationDef('broadening_top', upper_slope='up', lower_slope='down',
                  convergence='diverging', required_prior_trend='up'),
    FormationDef('broadening_right_angled_ascending', upper_slope='up', lower_slope='flat',
                  convergence='diverging'),
    FormationDef('broadening_right_angled_descending', upper_slope='flat', lower_slope='down',
                  convergence='diverging'),
    FormationDef('broadening_wedge_ascending', upper_slope='up', lower_slope='up',
                  convergence='diverging'),
    FormationDef('broadening_wedge_descending', upper_slope='down', lower_slope='down',
                  convergence='diverging'),

    FormationDef('triangle_ascending', upper_slope='flat', lower_slope='up', convergence='converging'),
    FormationDef('triangle_descending', upper_slope='down', lower_slope='flat', convergence='converging'),
    FormationDef('triangle_symmetrical', upper_slope='down', lower_slope='up', convergence='converging'),
    FormationDef('wedge_rising', upper_slope='up', lower_slope='up', convergence='converging'),
    FormationDef('wedge_falling', upper_slope='down', lower_slope='down', convergence='converging'),

    FormationDef('rectangle_top', upper_slope='flat', lower_slope='flat',
                  convergence='parallel', required_prior_trend='up'),
    FormationDef('rectangle_bottom', upper_slope='flat', lower_slope='flat',
                  convergence='parallel', required_prior_trend='down'),
]


def _classify_slope(slope: float, avg_price: float) -> str:
    pct_per_bar = slope / avg_price
    if abs(pct_per_bar) < FLAT_THRESHOLD:
        return 'flat'
    return 'up' if pct_per_bar > 0 else 'down'


def _classify_convergence(gap_start: float, gap_end: float) -> str | None:
    if gap_start <= 0 or gap_end <= 0:
        return None  # lines already crossed inside the window - not a valid formation
    if gap_end > gap_start * (1 + CONVERGENCE_TOLERANCE):
        return 'diverging'
    if gap_end < gap_start * (1 - CONVERGENCE_TOLERANCE):
        return 'converging'
    return 'parallel'


def _apex_x(r_slope, r_int, s_slope, s_int):
    if abs(r_slope - s_slope) < 1e-9:
        return None
    return (s_int - r_int) / (r_slope - s_slope)


def _find_breakout_two_line(df, last_idx, r_slope, r_int, s_slope, s_int,
                             apex_x=None, max_bars=80):
    n = len(df)
    end = min(n, last_idx + 1 + max_bars)
    if apex_x is not None:
        end = min(end, int(apex_x) + 1)
    for i in range(last_idx + 1, end):
        upper_val = _line_y_at(r_slope, r_int, i)
        lower_val = _line_y_at(s_slope, s_int, i)
        close = df['Close'].iloc[i]
        if close > upper_val:
            return {'idx': i, 'date': df.index[i], 'price': close, 'direction': 'up'}
        if close < lower_val:
            return {'idx': i, 'date': df.index[i], 'price': close, 'direction': 'down'}
    return None


def detect_two_line_formations(pivots: pd.DataFrame, df: pd.DataFrame,
                                defs=FORMATION_DEFS, window_sizes=(4, 5, 6),
                                trend_lookback: int = 15) -> list[PatternMatch]:
    # Smaller windows first: in testing, starting with the largest window
    # tends to greedily swallow an adjacent, unrelated pivot (often the
    # very breakout candle from a prior formation) into the fit, which
    # skews the line slopes and misclassifies the pattern. Smaller windows
    # first, combined with claiming through the breakout bar below, gave
    # the most reliable results - see the README's "window selection" note
    # for the known remaining edge cases.
    matches = []
    accepted_ranges = []  # (start_idx, end_idx) already claimed, to de-dupe overlapping windows

    for size in window_sizes:
        for start in range(len(pivots) - size + 1):
            window = pivots.iloc[start:start + size].reset_index(drop=True)
            seq = window['kind'].tolist()
            if any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
                continue  # must strictly alternate

            highs = window[window.kind == 'high']
            lows = window[window.kind == 'low']
            if len(highs) < 2 or len(lows) < 2:
                continue

            first_idx = int(window.iloc[0]['idx'])
            last_idx = int(window.iloc[-1]['idx'])

            if any(first_idx < e and last_idx > s for s, e in accepted_ranges):
                continue  # overlaps a formation we already accepted

            r_slope, r_int = _fit_line(highs['idx'].values, highs['price'].values)
            s_slope, s_int = _fit_line(lows['idx'].values, lows['price'].values)
            avg_price = window['price'].mean()

            upper_class = _classify_slope(r_slope, avg_price)
            lower_class = _classify_slope(s_slope, avg_price)

            gap_start = _line_y_at(r_slope, r_int, first_idx) - _line_y_at(s_slope, s_int, first_idx)
            gap_end = _line_y_at(r_slope, r_int, last_idx) - _line_y_at(s_slope, s_int, last_idx)
            if gap_start <= 0 or gap_end <= 0:
                continue
            if upper_class == 'flat' and lower_class == 'flat':
                # Two flat lines are parallel by construction - small gap
                # fluctuation here is just touch-to-touch noise, not a real
                # widening or narrowing formation.
                convergence = 'parallel'
            else:
                convergence = _classify_convergence(gap_start, gap_end)
            if convergence is None:
                continue

            trend = prior_trend(df, first_idx, trend_lookback)
            duration = last_idx - first_idx

            for d in defs:
                if d.needs_flagpole:
                    continue  # handled by detect_flags_and_pennants() instead - see its docstring
                if d.upper_slope != upper_class or d.lower_slope != lower_class:
                    continue
                if d.convergence != convergence:
                    continue
                if d.required_prior_trend is not None and trend != d.required_prior_trend:
                    continue
                if d.max_duration_bars is not None and duration > d.max_duration_bars:
                    continue

                apex_x = _apex_x(r_slope, r_int, s_slope, s_int) if convergence == 'converging' else None
                breakout = _find_breakout_two_line(df, last_idx, r_slope, r_int, s_slope, s_int, apex_x)

                target = None
                failed = None
                height = abs(max(gap_start, gap_end))
                if breakout:
                    target = breakout['price'] + height if breakout['direction'] == 'up' \
                        else breakout['price'] - height
                    failed = _check_failure(df, breakout)

                matches.append(PatternMatch(
                    pattern=d.name,
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
                    meta={'height': height, 'convergence': convergence},
                ))
                # Claim through the breakout bar itself, not just the last
                # touch - otherwise the breakout candle (often a fresh local
                # extreme) gets reused as the first "touch" of a bogus
                # follow-on formation.
                claim_end = breakout['idx'] if breakout else last_idx
                accepted_ranges.append((first_idx, claim_end))
                break  # first matching definition wins for this window

    return matches


def detect_flags_and_pennants(df: pd.DataFrame, pole_pct_threshold: float = 0.03,
                               flagpole_lookback: int = 40, flagpole_min_pct: float = 0.15,
                               channel_min_bars: int = 3, channel_max_bars: int = 15,
                               channel_pct_threshold: float = 0.01,
                               max_search_bars: int = 80) -> list[PatternMatch]:
    """
    Flags and pennants don't go through detect_two_line_formations() above,
    on purpose. That detector works from CONFIRMED zigzag pivots at one
    threshold for the whole series - fine for broadening/triangle/rectangle
    formations, which span dozens of bars, but flags are short (a handful
    to ~15 bars) and tight (a percent or two wide). Finding them needs a
    zigzag threshold small enough to catch that tight channel, and running
    the WHOLE series at that threshold would flood everything else with
    noise pivots. So instead this scans directly for the thing that defines
    a flag: a sharp "flagpole" move, followed immediately by a short, tight
    consolidation.

    The flagpole itself is found from ordinary zigzag pivots (at
    `pole_pct_threshold`, the normal swing scale) rather than a raw
    percent-move scan - a pivot only confirms once price genuinely
    reverses, so consecutive pivots reliably bracket the pole's actual
    start and end, instead of an arbitrary bar where a running total
    happens to first cross the threshold. The channel right after the
    pole is then read at a much smaller, LOCAL threshold, since it's
    naturally tight.

    `flagpole_lookback` bounds how many bars the pole's pivot-to-pivot span
    is allowed to take (guards against matching a multi-month trend as a
    "pole") - the sharpness itself comes from `flagpole_min_pct` on that
    span, not from a tight bar count, since a real pole can include a
    quiet run-up before the sharp part with no intervening reversal.

    Direction logic: a flagpole up followed by a channel sloping down (or
    flat) is a bullish flag (continuation up); the mirror is a bearish
    flag; a channel that CONVERGES after either direction of pole is a
    pennant.
    """
    matches = []
    n = len(df)
    claimed_until = -1

    swing_pivots = find_zigzag_pivots(df, pole_pct_threshold)
    if len(swing_pivots) < 2:
        return matches

    for k in range(1, len(swing_pivots)):
        pole_start_idx = int(swing_pivots.iloc[k - 1]['idx'])
        pole_end_idx = int(swing_pivots.iloc[k]['idx'])
        if pole_end_idx <= claimed_until:
            continue
        if pole_end_idx - pole_start_idx > flagpole_lookback:
            continue
        pole_start_price = swing_pivots.iloc[k - 1]['price']
        pole_end_price = swing_pivots.iloc[k]['price']
        change = (pole_end_price - pole_start_price) / pole_start_price
        if abs(change) < flagpole_min_pct:
            continue
        pole_direction = 'up' if change > 0 else 'down'
        i = pole_end_idx

        # Channel: pivots computed locally, at a much smaller threshold,
        # only within the bars right after the pole.
        end = min(n, i + channel_max_bars + 1)
        channel_df = df.iloc[i:end]
        channel_pivots = find_zigzag_pivots(channel_df, channel_pct_threshold)
        if len(channel_pivots) < 4:
            continue
        channel_pivots = channel_pivots.copy()
        channel_pivots['idx'] = channel_pivots['idx'] + i  # back to global bar positions

        window = channel_pivots.iloc[:4].reset_index(drop=True)
        seq = window['kind'].tolist()
        if seq[0] == seq[1] or seq[1] == seq[2] or seq[2] == seq[3]:
            continue
        highs = window[window.kind == 'high']
        lows = window[window.kind == 'low']
        if len(highs) < 2 or len(lows) < 2:
            continue

        first_idx = int(window.iloc[0]['idx'])
        last_idx = int(window.iloc[-1]['idx'])
        if last_idx - first_idx > channel_max_bars:
            continue

        r_slope, r_int = _fit_line(highs['idx'].values, highs['price'].values)
        s_slope, s_int = _fit_line(lows['idx'].values, lows['price'].values)
        avg_price = window['price'].mean()
        upper_class = _classify_slope(r_slope, avg_price)
        lower_class = _classify_slope(s_slope, avg_price)

        gap_start = _line_y_at(r_slope, r_int, first_idx) - _line_y_at(s_slope, s_int, first_idx)
        gap_end = _line_y_at(r_slope, r_int, last_idx) - _line_y_at(s_slope, s_int, last_idx)
        if gap_start <= 0 or gap_end <= 0:
            continue
        if upper_class == 'flat' and lower_class == 'flat':
            convergence = 'parallel'
        else:
            convergence = _classify_convergence(gap_start, gap_end)
        if convergence is None:
            continue

        name = None
        if convergence == 'converging':
            name = 'pennant'
        elif convergence == 'parallel':
            if pole_direction == 'up' and upper_class != 'up' and lower_class != 'up':
                name = 'flag_bullish'
            elif pole_direction == 'down' and upper_class != 'down' and lower_class != 'down':
                name = 'flag_bearish'
        if name is None:
            continue

        apex_x = _apex_x(r_slope, r_int, s_slope, s_int) if convergence == 'converging' else None
        breakout = _find_breakout_two_line(df, last_idx, r_slope, r_int, s_slope, s_int,
                                            apex_x, max_bars=max_search_bars)

        target = None
        failed = None
        height = abs(max(gap_start, gap_end))
        if breakout:
            target = breakout['price'] + height if breakout['direction'] == 'up' \
                else breakout['price'] - height
            failed = _check_failure(df, breakout)

        matches.append(PatternMatch(
            pattern=name,
            points=list(window.itertuples(index=False, name=None)),
            lines={
                'resistance': (highs['idx'].iloc[0], _line_y_at(r_slope, r_int, highs['idx'].iloc[0]),
                               highs['idx'].iloc[-1], _line_y_at(r_slope, r_int, highs['idx'].iloc[-1])),
                'support': (lows['idx'].iloc[0], _line_y_at(s_slope, s_int, lows['idx'].iloc[0]),
                            lows['idx'].iloc[-1], _line_y_at(s_slope, s_int, lows['idx'].iloc[-1])),
                'flagpole': (pole_start_idx, pole_start_price, pole_end_idx, pole_end_price),
            },
            breakout=breakout,
            target=target,
            failed=failed,
            meta={'height': height, 'convergence': convergence, 'pole_direction': pole_direction},
        ))
        claimed_until = breakout['idx'] if breakout else last_idx

    return matches
