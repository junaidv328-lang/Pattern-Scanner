"""
Synthetic OHLC data with deliberately embedded patterns.

This sandbox environment can't reach Yahoo Finance or any live market data
API (only a small allowlist of package registries is reachable), so this
module builds a realistic-looking price series with a known broadening
bottom and a known head-and-shoulders top stitched in, purely so the
detection and plotting logic can be exercised end to end. On your own
machine, data_sources.py's yfinance-based loader replaces this entirely.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _build_from_anchors(anchors, seed: int, noise_std: float = 0.6,
                         start_date: str = '2023-01-02') -> pd.DataFrame:
    """Shared helper: piecewise-linear interpolation through (bar, price)
    anchors, plus light noise so bars aren't perfectly straight."""
    rng = np.random.default_rng(seed)
    xs = np.array([a[0] for a in anchors])
    ys = np.array([a[1] for a in anchors])
    n = anchors[-1][0] + 1
    x_full = np.arange(n)
    close = np.interp(x_full, xs, ys)

    close = close + rng.normal(0, noise_std, size=n)
    close[0] = anchors[0][1]

    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] + rng.normal(0, noise_std * 0.67, size=n - 1)

    spread = np.abs(rng.normal(0.9, 0.4, size=n)) + 0.2
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = (rng.integers(80_000, 220_000, size=n)).astype(float)

    dates = pd.bdate_range(start_date, periods=n)
    return pd.DataFrame(
        {'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume},
        index=dates,
    )


def _reflect_anchors(anchors, axis: float):
    """Mirror anchors vertically around `axis` (axis - price). Turns a
    rising line into a falling one and vice versa while preserving every
    percentage swing exactly - useful for deriving a pattern's mirror
    image (e.g. a descending triangle from an ascending one) without
    re-deriving all the slope/threshold arithmetic by hand."""
    return [(bar, axis - price) for bar, price in anchors]


def build_synthetic_ohlc(seed: int = 7) -> pd.DataFrame:
    # (bar index, price) anchor points the series is piecewise-linearly
    # interpolated through. Comments show which pattern piece each is.
    anchors = [
        (0, 110),
        (40, 70),    # prior downtrend into the broadening bottom
        (55, 82),    # first high
        (70, 60),    # lower low
        (85, 92),    # higher high (must be a genuine local peak for the zigzag to register it)
        (90, 86),    # small pullback confirming that peak
        (95, 105),   # upward breakout
        (120, 140),  # continued uptrend / prior uptrend into H&S
        (130, 152),  # left shoulder
        (140, 128),  # neckline point 1
        (150, 168),  # head
        (160, 130),  # neckline point 2
        (170, 148),  # right shoulder
        (180, 113),  # downward breakout (below neckline)
        (200, 88),   # continued downtrend
        (220, 90),   # settle / tail
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_family1(seed: int = 11) -> pd.DataFrame:
    """
    Four two-line formations stitched together: ascending triangle,
    symmetrical triangle, rectangle bottom, and an ascending broadening
    wedge - one full loop through the diverging/converging/parallel family.
    """
    anchors = [
        (0, 80),
        (8, 95),     # ascending triangle: high 1
        (16, 86),    # low 1
        (24, 98),    # high 2 (flat resistance)
        (32, 90),    # low 2 (rising support)
        (42, 112),   # breakout up
        (48, 108), (54, 110),           # quiet plateau - separates formations
        (64, 130),   # symmetrical triangle: high 1
        (74, 114),   # low 1
        (84, 126),   # high 2 (lower than high 1 - descending upper line)
        (94, 120),   # low 2 (higher than low 1 - ascending lower line)
        (97, 126),   # small bounce confirming low 2 as a genuine pivot
        (104, 100),  # breakout down through the rising support line
        (110, 96), (116, 98),           # quiet plateau
        (126, 80),   # rectangle bottom: low 1
        (136, 91),   # high 1
        (146, 81),   # low 2
        (156, 90),   # high 2
        (166, 82),   # low 3
        (173, 101),  # breakout up
        (179, 103), (185, 101),         # quiet plateau
        (195, 112),  # broadening wedge ascending: start
        (205, 125),  # high 1
        (215, 113),  # low 1
        (225, 140),  # high 2 (higher high, upper line rising fast)
        (235, 120),  # low 2 (higher low, but rising slower - lines diverge)
        (245, 146),
        (255, 166),  # breakout up through the extended upper line
        (260, 160),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_triangle_ascending(seed: int = 21) -> pd.DataFrame:
    anchors = [
        (0, 80), (10, 98), (20, 84), (30, 98), (40, 88),
        (45, 93),   # small bounce confirming the low at 40 as a genuine pivot
        (55, 110),  # breakout up through the flat resistance
        (65, 118),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_triangle_symmetrical(seed: int = 22) -> pd.DataFrame:
    anchors = [
        (0, 130), (10, 110), (20, 124), (30, 114),
        (33, 118),  # small bounce confirming the low as a genuine pivot
        (43, 105),  # breakout down through the rising support line
        (53, 90),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_rectangle_bottom(seed: int = 23) -> pd.DataFrame:
    anchors = [
        (0, 130), (20, 95), (25, 80),   # prior downtrend into low 1
        (35, 91), (45, 81), (55, 90), (65, 82),
        (72, 101),  # breakout up
        (80, 110),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_broadening_wedge_ascending(seed: int = 24) -> pd.DataFrame:
    anchors = [
        (0, 108), (10, 128), (20, 118), (30, 148), (40, 130),
        (55, 182),  # breakout up through the extended (faster-rising) upper line
        (65, 172),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_triangle_descending(seed: int = 25) -> pd.DataFrame:
    # Mirror image of the (proven) ascending triangle: falling resistance,
    # flat support, breaks down.
    base = [
        (0, 80), (10, 98), (20, 84), (30, 98), (40, 88),
        (45, 93), (55, 110), (65, 118),
    ]
    return _build_from_anchors(_reflect_anchors(base, axis=200), seed)


def build_synthetic_wedge_rising(seed: int = 26) -> pd.DataFrame:
    anchors = [
        (0, 102), (10, 126), (20, 118), (30, 132),
        (45, 100),  # breakout down - the lower line rises faster, so the
                    # lines converge and the wedge resolves downward
        (55, 90),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_wedge_falling(seed: int = 27) -> pd.DataFrame:
    # Mirror image of the rising wedge: both lines fall, upper falls
    # faster, breaks up.
    base = [
        (0, 102), (10, 126), (20, 118), (30, 132), (45, 100), (55, 90),
    ]
    return _build_from_anchors(_reflect_anchors(base, axis=230), seed)


def build_synthetic_rectangle_top(seed: int = 28) -> pd.DataFrame:
    anchors = [
        (0, 60), (15, 100), (20, 90),   # prior uptrend into high 1, with a
                                         # pullback so it doesn't chain
                                         # straight into the channel's own low
        (25, 111), (35, 101), (45, 111), (55, 102), (65, 110),
        (72, 85),   # breakout down
        (80, 75),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_broadening_right_angled_ascending(seed: int = 29) -> pd.DataFrame:
    anchors = [
        (0, 80), (10, 95), (20, 81), (30, 105), (40, 90),
        (55, 130),  # breakout up through the rising resistance line
        (65, 125),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_broadening_right_angled_descending(seed: int = 30) -> pd.DataFrame:
    # Mirror image: flat resistance, falling support, breaks down.
    base = [
        (0, 80), (10, 95), (20, 81), (30, 105), (40, 90), (55, 130), (65, 125),
    ]
    return _build_from_anchors(_reflect_anchors(base, axis=200), seed)


def build_synthetic_broadening_wedge_descending(seed: int = 31) -> pd.DataFrame:
    # Mirror image of the (proven) ascending broadening wedge.
    base = [
        (0, 108), (10, 128), (20, 118), (30, 148), (40, 130), (55, 182), (65, 172),
    ]
    return _build_from_anchors(_reflect_anchors(base, axis=250), seed)


def build_synthetic_flag_bearish(seed: int = 4) -> pd.DataFrame:
    # Mirror image of the (proven) bullish flag: pole down, channel rising,
    # breaks down.
    base = [
        (0, 100), (20, 100), (28, 125), (30, 124), (33, 120),
        (36, 124.5), (39, 118), (42, 135), (50, 150),
    ]
    return _build_from_anchors(_reflect_anchors(base, axis=225), seed, noise_std=0.15)


def build_synthetic_pennant(seed: int = 5) -> pd.DataFrame:
    anchors = [
        (0, 100), (20, 100),  # quiet lead-in
        (28, 125),   # flagpole up
        (33, 119),   # channel low 1
        (36, 122),   # channel high 1 (below the pole peak - converging)
        (39, 120.5),  # channel low 2 (higher - converging toward the other line)
        (42, 133),   # breakout up
        (50, 145),
    ]
    return _build_from_anchors(anchors, seed, noise_std=0.15)


def build_synthetic_double_top(seed: int = 40) -> pd.DataFrame:
    anchors = [
        (0, 80), (20, 120),  # prior uptrend into peak 1
        (30, 100),  # trough (confirmation line)
        (40, 118),  # peak 2, within tolerance of peak 1
        (50, 90),   # breakout down through the confirmation line
        (60, 82),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_double_bottom(seed: int = 41) -> pd.DataFrame:
    # Mirror image of the double top.
    base = [
        (0, 80), (20, 120), (30, 100), (40, 118), (50, 90), (60, 82),
    ]
    return _build_from_anchors(_reflect_anchors(base, axis=200), seed)


def build_synthetic_triple_top(seed: int = 42) -> pd.DataFrame:
    anchors = [
        (0, 70), (15, 110),  # prior uptrend into peak 1
        (25, 95),   # trough 1
        (35, 108),  # peak 2
        (45, 93),   # trough 2 (lower of the two - becomes the confirmation line)
        (55, 109),  # peak 3
        (65, 80),   # breakout down
        (75, 72),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_head_and_shoulders_bottom(seed: int = 43) -> pd.DataFrame:
    anchors = [
        (0, 140),   # prior downtrend
        (15, 90),   # left shoulder (low)
        (25, 110),  # neckline point 1
        (35, 70),   # head (lower low)
        (45, 108),  # neckline point 2
        (55, 92),   # right shoulder (low)
        (65, 125),  # breakout up through the neckline
        (75, 135),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_complex_head_and_shoulders_top(seed: int = 44) -> pd.DataFrame:
    anchors = [
        (0, 80),
        (10, 125),  # left shoulder 2
        (18, 110),  # trough 1
        (26, 136),  # left shoulder 1 (taller, closer to the head)
        (34, 116),  # trough 2
        (42, 152),  # head (tallest)
        (50, 114),  # trough 3
        (58, 134),  # right shoulder 1
        (66, 110),  # trough 4
        (74, 123),  # right shoulder 2
        (82, 95),   # breakout down through the (sloped) neckline
        (90, 85),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_measured_move_up(seed: int = 50) -> pd.DataFrame:
    anchors = [
        (0, 80),    # A: leg 1 starts
        (15, 110),  # B: leg 1 ends (+30)
        (25, 98),   # C: corrective phase - partial retracement (~40% of leg 1)
        (40, 128),  # D: leg 2 ends, a similar-sized move to leg 1 (+30)
        (48, 125),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_measured_move_down(seed: int = 51) -> pd.DataFrame:
    base = [(0, 80), (15, 110), (25, 98), (40, 128), (48, 125)]
    return _build_from_anchors(_reflect_anchors(base, axis=200), seed)


def build_synthetic_three_rising_valleys(seed: int = 52) -> pd.DataFrame:
    anchors = [
        (0, 80), (10, 100),  # low 1, high 1
        (20, 88),   # low 2 (higher than low 1)
        (30, 102),  # high 2
        (40, 95),   # low 3 (higher than low 2)
        (50, 120),  # breakout above the confirmation line
        (58, 128),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_three_falling_peaks(seed: int = 53) -> pd.DataFrame:
    base = [(0, 80), (10, 100), (20, 88), (30, 102), (40, 95), (50, 120), (58, 128)]
    return _build_from_anchors(_reflect_anchors(base, axis=200), seed)


def _inject_gap(df: pd.DataFrame, idx: int, gap_pct: float) -> pd.DataFrame:
    """
    Shifts every bar from `idx` onward by a constant amount so that bar
    idx's range sits entirely clear of bar idx-1's range - a genuine price
    gap. Plain interpolation can't produce this on its own: since each
    bar's Open is derived from the previous bar's Close, the ranges always
    stay continuous. positive gap_pct gaps up, negative gaps down.
    """
    df = df.copy()
    prev_high = df['High'].iloc[idx - 1]
    prev_low = df['Low'].iloc[idx - 1]
    if gap_pct > 0:
        shift = prev_high * (1 + gap_pct) - df['Low'].iloc[idx]
    else:
        shift = prev_low * (1 + gap_pct) - df['High'].iloc[idx]
    for col in ['Open', 'High', 'Low', 'Close']:
        df.loc[df.index[idx:], col] = df[col].iloc[idx:] + shift
    return df


def build_synthetic_breakaway_gap(seed: int = 60) -> pd.DataFrame:
    anchors = [(0, 100), (20, 90), (22, 92), (40, 130)]
    df = _build_from_anchors(anchors, seed)
    return _inject_gap(df, 22, 0.04)  # gap up 2 bars after the pivot low at 20


def build_synthetic_exhaustion_gap(seed: int = 61) -> pd.DataFrame:
    anchors = [(0, 100), (20, 80), (60, 150), (65, 152)]
    df = _build_from_anchors(anchors, seed)
    return _inject_gap(df, 60, 0.04)  # gap up 40 bars / +87% into an extended run


def build_synthetic_island_top(seed: int = 62) -> pd.DataFrame:
    anchors = [(0, 100), (20, 120), (21, 121), (23, 120.5), (25, 121.5), (26, 121), (35, 90)]
    df = _build_from_anchors(anchors, seed)
    df = _inject_gap(df, 21, 0.03)   # gap up into the island
    df = _inject_gap(df, 26, -0.03)  # gap down out of the island
    return df


def _ohlc_from_close(close, seed: int, noise_std: float = 0.5,
                      start_date: str = '2023-01-02') -> pd.DataFrame:
    """Like _build_from_anchors, but starting from an already-built Close
    series rather than sparse anchors - used for the curved patterns,
    which need a genuinely smooth run of prices that piecewise-linear
    anchor interpolation can't produce (too few segments to look curved)."""
    rng = np.random.default_rng(seed)
    n = len(close)
    close = np.asarray(close) + rng.normal(0, noise_std, size=n)

    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] + rng.normal(0, noise_std * 0.5, size=n - 1)

    spread = np.abs(rng.normal(0.6, 0.25, size=n)) + 0.15
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = (rng.integers(80_000, 220_000, size=n)).astype(float)

    dates = pd.bdate_range(start_date, periods=n)
    return pd.DataFrame(
        {'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume},
        index=dates,
    )


def build_synthetic_rounding_bottom(seed: int = 70, n_cup: int = 60, depth: float = 25,
                                     start_price: float = 100, tail_bars: int = 15) -> pd.DataFrame:
    vertex = n_cup / 2
    a = depth / (vertex ** 2)
    x_cup = np.arange(n_cup)
    cup_close = a * (x_cup - vertex) ** 2 + (start_price - depth)
    # tail: a roughly LINEAR continuation past the rim, not more of the same
    # parabola - otherwise the curve fit just keeps extending into it and
    # leaves no bars free for a breakout search.
    tail_close = np.linspace(start_price + 1, start_price + depth * 0.7, tail_bars)
    close = np.concatenate([cup_close, tail_close])
    return _ohlc_from_close(close, seed)


def build_synthetic_rounding_top(seed: int = 71, n_dome: int = 60, height: float = 25,
                                  start_price: float = 100, tail_bars: int = 15) -> pd.DataFrame:
    vertex = n_dome / 2
    a = -height / (vertex ** 2)
    x_dome = np.arange(n_dome)
    dome_close = a * (x_dome - vertex) ** 2 + (start_price + height)
    tail_close = np.linspace(start_price - 1, start_price - height * 0.7, tail_bars)
    close = np.concatenate([dome_close, tail_close])
    return _ohlc_from_close(close, seed)


def build_synthetic_cup_with_handle(seed: int = 72, n_cup: int = 60, depth: float = 25,
                                     start_price: float = 100, handle_bars: int = 10,
                                     handle_depth: float = 6, tail_bars: int = 15) -> pd.DataFrame:
    vertex = n_cup / 2
    a = depth / (vertex ** 2)
    x_cup = np.arange(n_cup)
    cup_close = a * (x_cup - vertex) ** 2 + (start_price - depth)

    # handle: a smaller, roughly linear pullback right after the cup's rim,
    # shallow relative to the cup's own depth
    handle_vertex = handle_bars / 2
    b = handle_depth / (handle_vertex ** 2)
    x_handle = np.arange(handle_bars)
    handle_close = b * (x_handle - handle_vertex) ** 2 + (start_price - handle_depth)

    tail_close = np.linspace(start_price + 2, start_price + depth * 0.8, tail_bars)

    close = np.concatenate([cup_close, handle_close, tail_close])
    return _ohlc_from_close(close, seed, noise_std=0.4)


def build_synthetic_diamond_top(seed: int = 80) -> pd.DataFrame:
    anchors = [
        (0, 70), (8, 56),     # small early wiggle so it's a separate pivot,
                               # not the diamond's own first touch
        (18, 80), (28, 110),  # lead-in uptrend into high 1
        (36, 100),   # low 1
        (44, 130),   # high 2 (widest point of the diamond)
        (52, 108),   # low 2 (narrowing back in)
        (60, 118),   # high 3 (narrowing)
        (68, 90),    # breakout down through the low 2 confirmation level
        (76, 80),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_diamond_bottom(seed: int = 84) -> pd.DataFrame:
    base = [
        (0, 70), (8, 56), (18, 80), (28, 110), (36, 100),
        (44, 130), (52, 108), (60, 118), (68, 90), (76, 80),
    ]
    return _build_from_anchors(_reflect_anchors(base, axis=195), seed)


def build_synthetic_horn_top(seed: int = 81) -> pd.DataFrame:
    anchors = [
        (0, 80), (15, 120),   # prior uptrend into peak 1
        (18, 112),  # trough (confirmation line) - close in time to both peaks
        (21, 119),  # peak 2, close in time to peak 1
        (30, 95),   # breakout down
        (38, 85),
    ]
    return _build_from_anchors(anchors, seed)


def build_synthetic_pipe_bottom(seed: int = 82) -> pd.DataFrame:
    base = [(0, 80), (15, 120), (18, 112), (21, 119), (30, 95), (38, 85)]
    return _build_from_anchors(_reflect_anchors(base, axis=200), seed)


def build_synthetic_scallop_ascending(seed: int = 83) -> pd.DataFrame:
    n1, depth1, rim1 = 35, 12, 100
    n2, depth2, rim2 = 35, 12, 116

    v1 = n1 / 2
    a1 = depth1 / (v1 ** 2)
    x1 = np.arange(n1)
    close1 = a1 * (x1 - v1) ** 2 + (rim1 - depth1)

    bridge = np.linspace(close1[-1] + 0.5, rim2 - depth2, 6)  # connects scallop 1's end to scallop 2's start

    v2 = n2 / 2
    a2 = depth2 / (v2 ** 2)
    x2 = np.arange(n2)
    close2 = a2 * (x2 - v2) ** 2 + (rim2 - depth2)

    tail = np.linspace(rim2 + 2, rim2 + 20, 15)
    close = np.concatenate([close1, bridge, close2, tail])
    return _ohlc_from_close(close, seed, noise_std=0.4)


def build_synthetic_flag(seed: int = 3) -> pd.DataFrame:
    """
    A standalone flagpole + tight down-sloping channel + breakout. Kept
    separate from build_synthetic_family1() because flags are short and
    tight by nature - detecting one needs a much smaller zigzag threshold
    than the broader formations, so it's tested with its own scan.
    """
    anchors = [
        (0, 100),
        (20, 100),   # quiet lead-in so the prior-trend check has context
        (28, 125),   # flagpole: a sharp rise
        (30, 124),   # flag channel: high 1
        (33, 120),   # low 1
        (36, 124.5), # high 2 (slightly lower - keeps the channel roughly
                     # parallel to the support side rather than converging)
        (39, 118),   # low 2 (lower - descending channel, parallel to high side)
        (42, 135),   # breakout up through the channel
        (50, 150),
    ]
    return _build_from_anchors(anchors, seed, noise_std=0.15)
