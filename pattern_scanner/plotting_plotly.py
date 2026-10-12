"""
plotting_plotly.py
────────────────────
Interactive counterpart to plotting.py, for the Streamlit app specifically.
The demo_*.py scripts still use plotting.py's matplotlib renderer to save
static PNGs - a crosshair only means anything on something you can
actually hover, so this is a separate module rather than a mode flag on
the matplotlib one, and it's the only place in this project that needs
plotly as a dependency.

The x-axis here uses each bar's actual date (and time, for intraday data)
as a CATEGORY label rather than a continuous datetime axis. Two reasons:
it keeps the same "no gaps for weekends/holidays" behavior the matplotlib
version gets from using plain bar-index positions, and - the thing this
module got wrong on the first pass - it makes the real date/time show up
directly in the crosshair hover, since the x-value the cursor is
snapping to *is* the date string, not a bar-index number.

Kept as a close visual mirror of plot_match() in plotting.py (same
colors, same layout logic) so the two don't feel like different products.
"""

from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go

UP_COLOR = '#3B8B4A'
DOWN_COLOR = '#C0392B'
LINE_COLOR = '#555555'
TREND_COLOR = '#888888'


def _date_format(index: pd.DatetimeIndex) -> str:
    """'%Y-%m-%d' for daily-or-coarser data, '%Y-%m-%d %H:%M' if the data
    is intraday - inferred from the median gap between bars rather than
    assumed, so this works the same whether it's fed daily bars or 5m ones."""
    if len(index) < 2:
        return '%Y-%m-%d'
    median_gap = index.to_series().diff().dropna().median()
    return '%Y-%m-%d %H:%M' if median_gap < pd.Timedelta(days=1) else '%Y-%m-%d'


def _date_labels(index: pd.DatetimeIndex, fmt: str) -> list[str]:
    return [pd.Timestamp(d).strftime(fmt) for d in index]


def _pos_to_label(pos, labels: list[str]) -> str:
    """Maps an integer bar position to its date-string category label,
    clamping to the chart's own bounds - a computed line/marker position
    should always land on a real bar, but this guards against an
    off-by-one at the edge rather than letting it throw."""
    i = max(0, min(len(labels) - 1, int(round(pos))))
    return labels[i]


def _crosshair_layout(fig: go.Figure, title: str, n_bars: int, height: int = 520):
    """Applies Plotly's spike-line crosshair to both axes."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        yaxis_title='Price',
        xaxis=dict(
            type='category',
            showspikes=True, spikemode='across', spikesnap='cursor',
            spikedash='dot', spikethickness=1, spikecolor=TREND_COLOR,
            nticks=min(10, n_bars),
            rangeslider_visible=False, showgrid=True, gridcolor='#eeeeee',
        ),
        yaxis=dict(
            showspikes=True, spikemode='across', spikesnap='cursor',
            spikedash='dot', spikethickness=1, spikecolor=TREND_COLOR,
            showgrid=True, gridcolor='#eeeeee',
        ),
        hovermode='x unified',
        height=height,
        margin=dict(l=50, r=90, t=50, b=40),
        plot_bgcolor='white',
        showlegend=False,
    )
    return fig


def plot_candles_plotly(df: pd.DataFrame, title: str = '') -> go.Figure:
    """Plain candlestick chart with a crosshair - used for the overview
    view before any scan has run, and when a scan finds nothing."""
    fmt = _date_format(df.index)
    labels = _date_labels(df.index, fmt)
    fig = go.Figure(go.Candlestick(
        x=labels,
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
        increasing_fillcolor=UP_COLOR, decreasing_fillcolor=DOWN_COLOR,
        name='Price',
    ))
    return _crosshair_layout(fig, title, len(df))


def plot_match_plotly(df: pd.DataFrame, match, pad: int = 10, title: str | None = None) -> go.Figure:
    """
    Interactive version of plotting.plot_match(): candlesticks with the
    pattern's points, fitted trendlines, and breakout marker overlaid, and
    a crosshair that follows the cursor showing the exact price/date.
    """
    idxs = [p[0] for p in match.points]
    lo = max(0, min(idxs) - pad)
    hi = min(len(df), max(idxs) + pad + (match.breakout['idx'] - max(idxs) if match.breakout else 0) + 5)
    window_df = df.iloc[lo:hi]
    offset = lo

    fmt = _date_format(window_df.index)
    labels = _date_labels(window_df.index, fmt)

    fig = go.Figure(go.Candlestick(
        x=labels,
        open=window_df['Open'], high=window_df['High'],
        low=window_df['Low'], close=window_df['Close'],
        increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
        increasing_fillcolor=UP_COLOR, decreasing_fillcolor=DOWN_COLOR,
        name='Price',
    ))

    # Pivot points + connecting shape
    xs = [_pos_to_label(p[0] - offset, labels) for p in match.points]
    ys = [p[2] for p in match.points]
    kinds = [p[3] for p in match.points]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='lines+markers+text',
        line=dict(color=LINE_COLOR, width=1.5),
        marker=dict(size=7, color=LINE_COLOR),
        text=[str(k) for k in kinds],
        textposition=['top center' if k == 'high' else 'bottom center' for k in kinds],
        textfont=dict(size=10, color=LINE_COLOR),
        hoverinfo='skip',
    ))

    # Trendlines - extended to the breakout (or the right edge) same as the matplotlib version
    end_x = (match.breakout['idx'] if match.breakout else hi - 1) - offset
    for name, (x1, y1, x2, y2) in match.lines.items():
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        end_y = y1 + slope * (end_x - (x1 - offset))
        fig.add_trace(go.Scatter(
            x=[_pos_to_label(x1 - offset, labels), _pos_to_label(end_x, labels)],
            y=[y1, end_y], mode='lines',
            line=dict(color=TREND_COLOR, width=1, dash='dash'),
            hoverinfo='skip',
        ))
        fig.add_annotation(x=_pos_to_label(end_x, labels), y=end_y, text=name, showarrow=False,
                            font=dict(size=10, color=TREND_COLOR), xanchor='left')

    # Breakout marker + target line
    if match.breakout:
        bx = _pos_to_label(match.breakout['idx'] - offset, labels)
        by = match.breakout['price']
        color = UP_COLOR if match.breakout['direction'] == 'up' else DOWN_COLOR
        fig.add_trace(go.Scatter(
            x=[bx], y=[by], mode='markers+text',
            marker=dict(symbol='star', size=14, color=color),
            text=['breakout'], textposition='top center',
            textfont=dict(size=11, color=color),
            hoverinfo='skip',
        ))
        if match.target:
            fig.add_hline(y=match.target, line_dash='dot', line_color=color, opacity=0.7,
                          annotation_text=f'target {match.target:.2f}',
                          annotation_position='right', annotation_font_color=color)

    status = ''
    if match.breakout:
        status = f" | breakout {match.breakout['direction']}" + (
            '' if match.failed is None else (' (failed)' if match.failed else ' (held)'))
    return _crosshair_layout(fig, title or f"{match.pattern.replace('_', ' ').title()}{status}",
                              len(window_df))


def plot_points_plotly(df: pd.DataFrame, match, pad: int = 15) -> go.Figure:
    """
    For patterns with no fitted lines (gaps/islands) - plain candlesticks
    with the pattern's own points marked, same fallback render_match_chart()
    uses on the matplotlib side.
    """
    idxs = [p[0] for p in match.points]
    lo = max(0, min(idxs) - pad)
    hi = min(len(df), max(idxs) + pad)
    window_df = df.iloc[lo:hi]
    offset = lo

    fmt = _date_format(window_df.index)
    labels = _date_labels(window_df.index, fmt)

    fig = plot_candles_plotly(window_df, match.pattern.replace('_', ' ').title())
    xs = [_pos_to_label(p[0] - offset, labels) for p in match.points]
    ys = [p[2] for p in match.points]
    kinds = [str(p[3]) for p in match.points]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode='markers+text',
        marker=dict(symbol='star', size=12, color=DOWN_COLOR),
        text=kinds, textposition='top center',
        textfont=dict(size=10, color=DOWN_COLOR),
        hoverinfo='skip',
    ))
    return fig
