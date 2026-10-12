"""
Demo: the multi-touch extremes family (double/triple tops & bottoms,
head-and-shoulders simple and complex).

Runs each detector against one isolated synthetic example and saves an
annotated chart. Double and triple tops/bottoms cover the Adam/Eve naming
automatically via classify_touch_shape() - the pattern name reflects
whichever shape each touch actually has in the data.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pattern_scanner import synthetic_data as sd
from pattern_scanner.pivots import find_zigzag_pivots
from pattern_scanner.extremes import detect_double_extreme, detect_triple_extreme, detect_head_and_shoulders
from pattern_scanner.plotting import plot_match

EXAMPLES = [
    ('double_top', sd.build_synthetic_double_top, lambda p, df: detect_double_extreme(p, df, kind='top')),
    ('double_bottom', sd.build_synthetic_double_bottom, lambda p, df: detect_double_extreme(p, df, kind='bottom')),
    ('triple_top', sd.build_synthetic_triple_top, lambda p, df: detect_triple_extreme(p, df, kind='top')),
    ('head_and_shoulders_bottom', sd.build_synthetic_head_and_shoulders_bottom,
     lambda p, df: detect_head_and_shoulders(p, df, kind='bottom', max_shoulders=1)),
    ('complex_head_and_shoulders_top', sd.build_synthetic_complex_head_and_shoulders_top,
     lambda p, df: detect_head_and_shoulders(p, df, kind='top', max_shoulders=2)),
]

for label, builder, detector in EXAMPLES:
    df = builder()
    pivots = find_zigzag_pivots(df, 0.03)
    matches = detector(pivots, df)
    print(f"{label}: {len(matches)} match(es) -> {[m.pattern for m in matches]}")
    # for the complex H&S example, plot the largest (most-touches) match
    target = max(matches, key=lambda m: len(m.points)) if matches else None
    if target is None:
        print("  no match found - skipping chart")
        continue
    if target.breakout:
        status = f"{target.breakout['direction']} @ {target.breakout['price']:.2f}"
        status += '' if target.failed is None else (' (FAILED)' if target.failed else ' (held)')
        print(f"  {target.pattern}: {status}")
    ax = plot_match(df, target)
    plt.tight_layout()
    out_path = Path(__file__).parent / f"extremes_{target.pattern}.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  saved {out_path.name}")
