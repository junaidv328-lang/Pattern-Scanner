"""
Demo: the two-line formation family (diverging / converging / parallel).

Runs the generalized detector against one isolated synthetic example per
formation and saves an annotated chart for each. 11 of the 16 definitions
in FORMATION_DEFS are covered here through detect_two_line_formations().
The other 3 - flag_bullish, flag_bearish, pennant - are short, tight
patterns anchored to a sharp prior move, so they get their own dedicated
scan, detect_flags_and_pennants(), further down this script (see its
docstring in formations.py for why). broadening_bottom and broadening_top
aren't re-tested here since patterns.py's original, hand-built detector
already covers that geometry with the book's more precise dual-breakout
nuance - see demo_historical.py. That's all 16 definitions verified.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pattern_scanner import synthetic_data as sd
from pattern_scanner.pivots import find_zigzag_pivots
from pattern_scanner.formations import detect_two_line_formations
from pattern_scanner.plotting import plot_match

EXAMPLES = {
    'triangle_ascending': sd.build_synthetic_triangle_ascending,
    'triangle_descending': sd.build_synthetic_triangle_descending,
    'triangle_symmetrical': sd.build_synthetic_triangle_symmetrical,
    'wedge_rising': sd.build_synthetic_wedge_rising,
    'wedge_falling': sd.build_synthetic_wedge_falling,
    'rectangle_bottom': sd.build_synthetic_rectangle_bottom,
    'rectangle_top': sd.build_synthetic_rectangle_top,
    'broadening_wedge_ascending': sd.build_synthetic_broadening_wedge_ascending,
    'broadening_wedge_descending': sd.build_synthetic_broadening_wedge_descending,
    'broadening_right_angled_ascending': sd.build_synthetic_broadening_right_angled_ascending,
    'broadening_right_angled_descending': sd.build_synthetic_broadening_right_angled_descending,
}

for expected, builder in EXAMPLES.items():
    df = builder()
    pivots = find_zigzag_pivots(df, 0.03)
    matches = detect_two_line_formations(pivots, df)
    print(f"{expected}: {len(matches)} match(es) -> {[m.pattern for m in matches]}")
    target = next((m for m in matches if m.pattern == expected), matches[0] if matches else None)
    if target is None:
        print("  no match found - skipping chart")
        continue
    if target.breakout:
        status = f"{target.breakout['direction']} @ {target.breakout['price']:.2f}"
        status += '' if target.failed is None else (' (FAILED)' if target.failed else ' (held)')
        print(f"  {status}")
    ax = plot_match(df, target)
    plt.tight_layout()
    out_path = Path(__file__).parent / f"family1_{target.pattern}.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  saved {out_path.name}")

print()
print("Flag/pennant sub-family - detected with their own dedicated scan")
print("(detect_flags_and_pennants), which reads flagpole + channel directly")
print("instead of going through the general pivot-window scanner:")
from pattern_scanner.formations import detect_flags_and_pennants

FLAG_EXAMPLES = {
    'flag_bullish': sd.build_synthetic_flag,
    'flag_bearish': sd.build_synthetic_flag_bearish,
    'pennant': sd.build_synthetic_pennant,
}
for expected, builder in FLAG_EXAMPLES.items():
    df = builder()
    matches = detect_flags_and_pennants(df)
    print(f"{expected}: {len(matches)} match(es) -> {[m.pattern for m in matches]}")
    target = next((m for m in matches if m.pattern == expected), matches[0] if matches else None)
    if target is None:
        print("  no match found - skipping chart")
        continue
    if target.breakout:
        status = f"{target.breakout['direction']} @ {target.breakout['price']:.2f}"
        status += '' if target.failed is None else (' (FAILED)' if target.failed else ' (held)')
        print(f"  {status}")
    ax = plot_match(df, target)
    plt.tight_layout()
    out_path = Path(__file__).parent / f"family1_{target.pattern}.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  saved {out_path.name}")
