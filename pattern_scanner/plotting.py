"""
Plotting.

A small self-contained candlestick renderer (so we're not fighting a
third-party charting library's overlay API) plus a function to draw a
PatternMatch on top of it: the pivot points, fitted trendlines, and the
breakout marker if one has occurred.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd


def plot_candles(ax, df: pd.DataFrame, up_color='#3B8B4A', down_color='#C0392B', width=0.6):
    for i in range(len(df)):
        o, h, l, c = df['Open'].iloc[i], df['High'].iloc[i], df['Low'].iloc[i], df['Close'].iloc[i]
        color = up_color if c >= o else down_color
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=2)
        y0, height = (o, c - o) if c >= o else (c, o - c)
        height = max(height, (h - l) * 0.01)  # keep doji bodies visible
        ax.add_patch(Rectangle((i - width / 2, y0), width, height,
                                facecolor=color, edgecolor=color, zorder=3))
    ax.set_xlim(-1, len(df))


def _fmt_dates(ax, df, step=None):
    step = step or max(1, len(df) // 10)
    ticks = list(range(0, len(df), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([pd.Timestamp(df.index[t]).strftime('%Y-%m-%d') for t in ticks],
                        rotation=45, ha='right', fontsize=8)


def plot_match(df: pd.DataFrame, match, pad: int = 10, ax=None, title: str | None = None):
    """
    Plot the OHLC bars around a PatternMatch with the pattern's points,
    trendlines, and breakout overlaid.
    """
    idxs = [p[0] for p in match.points]
    lo = max(0, min(idxs) - pad)
    hi = min(len(df), max(idxs) + pad + (match.breakout['idx'] - max(idxs) if match.breakout else 0) + 5)
    window_df = df.iloc[lo:hi]
    offset = lo

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5.5))

    plot_candles(ax, window_df)

    # Pivot points + connecting shape
    xs = [p[0] - offset for p in match.points]
    ys = [p[2] for p in match.points]
    ax.plot(xs, ys, color='#555555', linewidth=1.3, zorder=4)
    ax.scatter(xs, ys, color='#555555', s=25, zorder=5)
    for (idx, date, price, kind) in match.points:
        ax.annotate(kind, (idx - offset, price),
                    textcoords="offset points", xytext=(0, 10 if kind == 'high' else -14),
                    ha='center', fontsize=8, color='#555555')

    # Trendlines
    for name, (x1, y1, x2, y2) in match.lines.items():
        # extend the line to the breakout (or to the right edge) so the
        # projected boundary that triggered the breakout is visible
        end_x = (match.breakout['idx'] if match.breakout else hi - 1) - offset
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        end_y = y1 + slope * (end_x - (x1 - offset))
        ax.plot([x1 - offset, end_x], [y1, end_y], linestyle='--', color='#888888', linewidth=1, zorder=1)
        ax.text(end_x, end_y, f' {name}', fontsize=8, color='#888888', va='center')

    # Breakout marker
    if match.breakout:
        bx = match.breakout['idx'] - offset
        by = match.breakout['price']
        color = '#3B8B4A' if match.breakout['direction'] == 'up' else '#C0392B'
        ax.scatter([bx], [by], color=color, s=60, zorder=6, marker='*')
        ax.annotate('breakout', (bx, by), textcoords="offset points",
                     xytext=(6, 6), fontsize=9, color=color, fontweight='bold')
        if match.target:
            ax.axhline(match.target, color=color, linestyle=':', linewidth=1, alpha=0.7)
            ax.text(len(window_df) - 1, match.target, f' target {match.target:.2f}',
                    fontsize=8, color=color, va='center')

    status = ''
    if match.breakout:
        status = f" | breakout {match.breakout['direction']}" + (
            '' if match.failed is None else (' (failed)' if match.failed else ' (held)'))
    ax.set_title(title or f"{match.pattern.replace('_', ' ').title()}{status}", fontsize=11)
    _fmt_dates(ax, window_df)
    ax.set_ylabel('Price')
    ax.grid(alpha=0.2)
    return ax
