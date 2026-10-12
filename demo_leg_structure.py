"""
Demo: the leg-structure family (measured moves, three rising valleys,
three falling peaks).

These compare the MAGNITUDE of consecutive zigzag legs to each other,
rather than fitting a boundary line or matching touches at a similar
level. Measured moves don't have a breakout to confirm - they're a
trend-continuation structure that's read once the second leg matches the
first, not a reversal pattern waiting for price to cross a line.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pattern_scanner import synthetic_data as sd
from pattern_scanner.pivots import find_zigzag_pivots
from pattern_scanner.leg_structure import (
    detect_measured_move, detect_three_rising_valleys, detect_three_falling_peaks,
)
from pattern_scanner.plotting import plot_match

EXAMPLES = [
    ('measured_move_up', sd.build_synthetic_measured_move_up,
     lambda p, df: detect_measured_move(p, df, direction='up')),
    ('measured_move_down', sd.build_synthetic_measured_move_down,
     lambda p, df: detect_measured_move(p, df, direction='down')),
    ('three_rising_valleys', sd.build_synthetic_three_rising_valleys,
     lambda p, df: detect_three_rising_valleys(p, df)),
    ('three_falling_peaks', sd.build_synthetic_three_falling_peaks,
     lambda p, df: detect_three_falling_peaks(p, df)),
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
    elif target.meta:
        print(f"  {target.meta}")
    ax = plot_match(df, target)
    plt.tight_layout()
    out_path = Path(__file__).parent / f"leg_{target.pattern}.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  saved {out_path.name}")
