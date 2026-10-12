"""
Demo: live monitoring.

This can't actually run in the sandbox that built this project (no access
to Yahoo Finance from there), but on your own machine with internet access,
this is a complete working example. It polls once an hour for new bars on
a daily chart and prints an alert the moment a pattern's breakout is newly
confirmed.

Usage:
    python demo_live.py AAPL
    python demo_live.py AAPL --poll-seconds 300 --interval 1h
"""
import argparse
from pattern_scanner.live_monitor import run_live_monitor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker')
    parser.add_argument('--interval', default='1d', help="e.g. 1d, 1h, 15m")
    parser.add_argument('--poll-seconds', type=int, default=3600)
    parser.add_argument('--pct-threshold', type=float, default=0.03,
                         help="zigzag reversal threshold, e.g. 0.03 = 3%%")
    args = parser.parse_args()

    print(f"Watching {args.ticker} ({args.interval} bars) every {args.poll_seconds}s. Ctrl+C to stop.")
    run_live_monitor(args.ticker, interval=args.interval, poll_seconds=args.poll_seconds,
                      pct_threshold=args.pct_threshold)
