"""
Demo: historical scan.

Runs the scanner against a synthetic OHLC series with a known broadening
bottom and a known head-and-shoulders top embedded in it, and saves an
annotated chart for each match found.

To run this against real data instead, swap the synthetic_data import for:
    from pattern_scanner.data_sources import fetch_historical
    df = fetch_historical('AAPL', period='2y', interval='1d')
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pattern_scanner.synthetic_data import build_synthetic_ohlc
from pattern_scanner.patterns import scan
from pattern_scanner.plotting import plot_match

df = build_synthetic_ohlc()
print(f"Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

matches = scan(df, pct_threshold=0.05)
print(f"Found {len(matches)} pattern match(es):")
for m in matches:
    status = 'no breakout yet'
    if m.breakout:
        status = f"{m.breakout['direction']} breakout @ {m.breakout['price']:.2f} on {m.breakout['date'].date()}"
        if m.failed is not None:
            status += ' (FAILED - 5% rule)' if m.failed else ' (held)'
    print(f"  - {m.pattern}: {status}")

for i, m in enumerate(matches):
    ax = plot_match(df, m)
    plt.tight_layout()
    out_path = Path(__file__).parent / f"pattern_{i}_{m.pattern}.png"
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Saved {out_path}")
