"""
Demo: the gap family (breakaway/exhaustion/runaway gaps, island reversals).

These are the only detectors in this library that don't start from zigzag
pivots - a gap is a bar-to-bar event. See gaps.py's docstring for why
that also makes them the only place a "synthetic gap" has to be injected
directly rather than emerging from interpolation.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pattern_scanner import synthetic_data as sd
from pattern_scanner.gaps import find_gaps, classify_gaps, detect_island_reversal
from pattern_scanner.plotting import plot_candles, _fmt_dates

EXAMPLES = [
    ('breakaway_gap', sd.build_synthetic_breakaway_gap, 'gap'),
    ('exhaustion_gap', sd.build_synthetic_exhaustion_gap, 'gap'),
    ('island_top', sd.build_synthetic_island_top, 'island'),
]

for label, builder, mode in EXAMPLES:
    df = builder()
    if mode == 'gap':
        matches = classify_gaps(df)
    else:
        matches = detect_island_reversal(df)
    print(f"{label}: {len(matches)} match(es) -> {[m.pattern for m in matches]}")
    if not matches:
        print("  no match found - skipping chart")
        continue
    target = matches[0]
    print(f"  {target.meta}")

    # gaps/islands don't carry fitted lines, so plot candles directly with
    # a marker at each gap point rather than using plot_match's line overlay
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_candles(ax, df)
    for (idx, date, price, kind) in target.points:
        ax.scatter([idx], [price], color='#C0392B', s=70, zorder=6, marker='*')
        ax.annotate(kind, (idx, price), textcoords="offset points", xytext=(6, 6),
                    fontsize=9, color='#C0392B', fontweight='bold')
    ax.set_title(target.pattern.replace('_', ' ').title())
    _fmt_dates(ax, df)
    ax.set_ylabel('Price')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out_path = Path(__file__).parent / f"gap_{target.pattern}.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"  saved {out_path.name}")
