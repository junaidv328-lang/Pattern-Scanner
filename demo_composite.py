"""
Demo: the composite family (diamond, horn, pipe) plus scallops.

These are tight variants or compositions of primitives that already exist
elsewhere in this library, not new detection methods - see composite.py's
and curves.py's docstrings for how each one is built from what's already
there.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pattern_scanner import synthetic_data as sd
from pattern_scanner.pivots import find_zigzag_pivots
from pattern_scanner.composite import detect_diamond, detect_horn, detect_pipe
from pattern_scanner.curves import detect_scallops
from pattern_scanner.plotting import plot_match

EXAMPLES = [
    ('diamond_top', sd.build_synthetic_diamond_top, lambda p, df: detect_diamond(p, df, kind='top')),
    ('diamond_bottom', sd.build_synthetic_diamond_bottom, lambda p, df: detect_diamond(p, df, kind='bottom')),
    ('horn_top', sd.build_synthetic_horn_top, lambda p, df: detect_horn(p, df, kind='top')),
    ('pipe_bottom', sd.build_synthetic_pipe_bottom, lambda p, df: detect_pipe(p, df, kind='bottom')),
]

for label, builder, detector in EXAMPLES:
    df = builder()
    pivots = find_zigzag_pivots(df, 0.03)
    matches = detector(pivots, df)
    print(f"{label}: {len(matches)} match(es) -> {[m.pattern for m in matches]}")
    if not matches:
        print("  no match found - skipping chart")
        continue
    target = matches[0]
    if target.breakout:
        status = f"{target.breakout['direction']} @ {target.breakout['price']:.2f}"
        status += '' if target.failed is None else (' (FAILED)' if target.failed else ' (held)')
        print(f"  {status}")
    ax = plot_match(df, target)
    plt.tight_layout()
    out_path = Path(__file__).parent / f"composite_{target.pattern}.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  saved {out_path.name}")

# scallops: no fitted lines, so plot candles with each rounding leg's rim marked
from pattern_scanner.plotting import plot_candles, _fmt_dates

df = sd.build_synthetic_scallop_ascending()
matches = detect_scallops(df, kind='ascending')
print(f"scallop_ascending: {len(matches)} match(es)")
if matches:
    m = matches[0]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_candles(ax, df)
    for (idx, dt, price, kind) in m.points:
        ax.scatter([idx], [price], color='#3B8B4A', s=40, zorder=6)
    ax.set_title(f"Scallop Ascending ({m.meta['n_scallops']} scallops)")
    _fmt_dates(ax, df)
    ax.set_ylabel('Price')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out_path = Path(__file__).parent / "composite_scallop_ascending.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  saved {out_path.name}")
