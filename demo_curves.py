"""
Demo: the curved shapes family (rounding bottom/top, cup with handle).

These use a genuinely different primitive from every other detector here:
a quadratic curve fit over Close prices, rather than zigzag pivots at all.
See curves.py's docstring for why.

Note: detect_cup_with_handle() is called with a narrower min/max window
than its defaults here, deliberately matched to the synthetic cup's known
length. The generic curve-fit window search can otherwise settle on a
shorter, still-well-fitting sub-window of a symmetric parabola (parabolas
look parabolic at many scales) - a real tuning consideration, not a bug,
noted in the README.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pattern_scanner import synthetic_data as sd
from pattern_scanner.curves import detect_rounding, detect_cup_with_handle
from pattern_scanner.plotting import plot_match

EXAMPLES = [
    ('rounding_bottom', sd.build_synthetic_rounding_bottom,
     lambda df: detect_rounding(df, kind='bottom')),
    ('rounding_top', sd.build_synthetic_rounding_top,
     lambda df: detect_rounding(df, kind='top')),
    ('cup_with_handle', sd.build_synthetic_cup_with_handle,
     lambda df: detect_cup_with_handle(df, min_window=50, max_window=65, step=2)),
]

for label, builder, detector in EXAMPLES:
    df = builder()
    matches = detector(df)
    print(f"{label}: {len(matches)} match(es) -> {[m.pattern for m in matches]}")
    if not matches:
        print("  no match found - skipping chart")
        continue
    target = matches[0]
    if target.breakout:
        status = f"{target.breakout['direction']} @ {target.breakout['price']:.2f}"
        status += '' if target.failed is None else (' (FAILED)' if target.failed else ' (held)')
        print(f"  {status}  (r2={target.meta.get('r_squared', target.meta.get('cup_r_squared', 0)):.3f})")
    ax = plot_match(df, target)
    plt.tight_layout()
    out_path = Path(__file__).parent / f"curve_{target.pattern}.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  saved {out_path.name}")
