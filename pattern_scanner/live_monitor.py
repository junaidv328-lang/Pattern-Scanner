"""
Live monitoring.

Polls for the latest bars on an interval, re-runs the scanner on the tail
of the data, and prints an alert the first time a pattern's breakout is
newly confirmed. A `seen` set keyed on (ticker, pattern, breakout bar date)
stops the same event from alerting twice on every subsequent poll.

This is polling-based (checks "did the picture change" every N seconds),
not a tick-level feed - appropriate for daily/hourly bars where you don't
need sub-second reaction time. For intraday scalping timeframes you'd want
a websocket feed instead, but the detection logic underneath is identical.
"""

from __future__ import annotations
import time
from datetime import datetime

from .data_sources import fetch_latest
from .patterns import scan


def run_live_monitor(ticker: str, interval: str = '1d', poll_seconds: int = 3600,
                      pct_threshold: float = 0.03, on_alert=None, max_iterations: int | None = None):
    """
    Blocking loop. Call with on_alert=your_function to do something other
    than print (send a Telegram message, write to a DB, etc.) - it receives
    (ticker, PatternMatch).
    """
    seen = set()
    iterations = 0

    while True:
        try:
            df = fetch_latest(ticker, interval=interval)
            matches = scan(df, pct_threshold=pct_threshold)
            for m in matches:
                if not m.breakout:
                    continue
                key = (ticker, m.pattern, str(m.breakout['date']))
                if key in seen:
                    continue
                seen.add(key)
                if on_alert:
                    on_alert(ticker, m)
                else:
                    _default_alert(ticker, m)
        except Exception as e:
            print(f"[{datetime.now()}] {ticker}: poll failed - {e}")

        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            break
        time.sleep(poll_seconds)


def _default_alert(ticker: str, match):
    direction = match.breakout['direction']
    failed_note = '' if match.failed is None else (' [already failed the 5% rule]' if match.failed else '')
    target_note = f", target {match.target:.2f}" if match.target else ""
    print(f"[{datetime.now()}] {ticker}: {match.pattern} - {direction} breakout confirmed "
          f"at {match.breakout['date']} @ {match.breakout['price']:.2f}{target_note}{failed_note}")
