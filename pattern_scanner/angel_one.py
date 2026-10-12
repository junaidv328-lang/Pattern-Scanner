"""
angel_one.py
─────────────
Angel One SmartAPI data source - an alternative to yfinance in
data_sources.py, for pulling real NSE/BSE/MCX candles to test the
detectors against.

Follows the same auth and chunking conventions as your other Angel
One-integrated apps (see the Trend Backtest Lab's data_fetch.py): TOTP
login via pyotp + SmartConnect, then getCandleData in chunks sized to
each interval's documented day limit, kept a little under the published
cap as safety margin.

Needs a real Angel One trading account with API access enabled (API key,
client code, password/MPIN, and a TOTP secret from enabling TOTP in your
account settings) - this can't be exercised in a sandboxed environment,
the same way this project's yfinance path can't either. See
streamlit_app.py for how this gets wired into the UI (st.secrets first,
manual entry fallback).
"""

from __future__ import annotations
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

INSTRUMENT_MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"

# Angel One's documented max days per request, by interval - kept a little
# under the published cap as safety margin, same convention as your other
# apps' INTERVAL_MAP.
INTERVAL_MAP = {
    '1m': ('ONE_MINUTE', 25),
    '3m': ('THREE_MINUTE', 55),
    '5m': ('FIVE_MINUTE', 90),
    '10m': ('TEN_MINUTE', 90),
    '15m': ('FIFTEEN_MINUTE', 180),
    '30m': ('THIRTY_MINUTE', 180),
    '1h': ('ONE_HOUR', 350),
    '1d': ('ONE_DAY', 1800),
}

# Index tokens don't reliably surface through searchScrip (indices aren't
# "tradeable" instruments the way stocks are, so the scrip search often
# comes up empty or ambiguous for them) - these are fixed, known-good
# tokens rather than something to search for. Same values used across
# your other Angel One apps.
COMMON_INDICES = {
    "NIFTY 50": {"exchange": "NSE", "token": "99926000"},
    "BANKNIFTY": {"exchange": "NSE", "token": "99926009"},
    "SENSEX": {"exchange": "BSE", "token": "99919000"},
}


def _toml_escape(s: str) -> str:
    return s.replace('\\', '\\\\').replace('"', '\\"')


def save_credentials(api_key: str, client_id: str, password: str, totp_key: str,
                      app_dir: str | Path) -> Path:
    """
    Writes/updates the [angel_one] section of .streamlit/secrets.toml next
    to the app, so future runs auto-fill the login form via st.secrets -
    the same file your other Angel One apps' deployments read from.

    Local use only: on Streamlit Community Cloud the filesystem is
    ephemeral, so this persists only until the app restarts or redeploys
    there - use the platform's own Secrets manager for anything that needs
    to survive a redeploy. Also note this writes credentials in PLAIN TEXT,
    same as secrets.toml always has - fine for a personal local file kept
    out of version control (it already is, via .gitignore, the same as
    your other apps), not something to commit or share.
    """
    path = Path(app_dir) / ".streamlit" / "secrets.toml"
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = path.read_text() if path.exists() else ""
    # Drop any existing [angel_one] section (up to the next section header
    # or end of file) before appending the fresh one, so re-saving doesn't
    # duplicate or leave stale keys behind.
    existing = re.sub(r'\n?\[angel_one\].*?(?=\n\[|\Z)', '', existing, flags=re.DOTALL).rstrip()

    block = (
        f'[angel_one]\n'
        f'api_key = "{_toml_escape(api_key)}"\n'
        f'client_id = "{_toml_escape(client_id)}"\n'
        f'password = "{_toml_escape(password)}"\n'
        f'totp_key = "{_toml_escape(totp_key)}"\n'
    )
    combined = (existing + '\n\n' + block) if existing else block
    path.write_text(combined.strip() + '\n')
    return path


def check_dependencies() -> list[str]:
    """Which of the optional Angel One packages are missing from the
    current environment, if any. Used by the UI to show a clear diagnostic
    before someone hits a bare ImportError on login."""
    missing = []
    try:
        import pyotp  # noqa: F401
    except ImportError:
        missing.append('pyotp')
    try:
        from SmartApi import SmartConnect  # noqa: F401
    except ImportError:
        missing.append('smartapi-python')
    return missing


def login(api_key: str, client_id: str, password: str, totp_secret: str):
    """Returns (smart_connect_obj, error_message). error_message is None on success."""
    missing = check_dependencies()
    if missing:
        pkgs = ' '.join(missing)
        return None, (f"Missing package(s): {', '.join(missing)}. Run `pip install {pkgs}` "
                       f"(or `pip install -r requirements.txt`) in the same environment "
                       f"you're running Streamlit from, then restart the app.")
    try:
        import pyotp
        from SmartApi import SmartConnect

        totp_code = pyotp.TOTP(totp_secret).now()
        obj = SmartConnect(api_key=api_key)
        resp = obj.generateSession(client_id, password, totp_code)
        if resp.get('status'):
            return obj, None
        return None, resp.get('message', 'Login failed - unknown error')
    except Exception as e:
        return None, str(e)


def search_instrument(obj, exchange: str, search_text: str):
    """
    Angel One's LIVE search endpoint (searchScrip). Kept for reference/use
    outside Streamlit, but in practice this often returns "no matching
    trading symbols" even for symbols that clearly exist - a widely
    reported quirk of this endpoint, not specific to any one query.
    fetch_instrument_master() + search_instrument_master() below is the
    more reliable path (same technique your continuous_contract.py already
    uses for MCX futures chains) and is what streamlit_app.py actually
    uses now.

    Returns (list_of_matches, error_message_or_None); each match is a dict
    with at least 'tradingsymbol' and 'symboltoken'.
    """
    try:
        resp = obj.searchScrip(exchange, search_text)
        if resp.get('status'):
            return resp.get('data', []) or [], None
        return [], resp.get('message', 'search failed')
    except Exception as e:
        return [], str(e)


def fetch_instrument_master() -> pd.DataFrame:
    """
    Angel One's full tradeable-instrument list (every stock, index-linked
    derivative, future, and option, across every exchange segment) - large
    (tens of MB), so callers should cache this rather than re-fetch it per
    search. In streamlit_app.py this is wrapped in st.cache_data(ttl=86400)
    so it's only actually downloaded once a day.
    """
    resp = requests.get(INSTRUMENT_MASTER_URL, timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def search_instrument_master(master: pd.DataFrame, exchange: str, search_text: str,
                              limit: int = 25) -> list[dict]:
    """
    Case-insensitive substring search across the 'symbol' and 'name'
    columns of an already-fetched instrument master, filtered to one
    exchange segment. This is a local pandas filter, not a network call -
    no rate limits, no live-endpoint quirks, and it actually finds real
    symbols reliably. Returns up to `limit` matches, shaped the same as
    search_instrument()'s results (dicts with 'tradingsymbol'/'symboltoken')
    so both can feed the same UI code.
    """
    if master.empty or not search_text.strip():
        return []
    text = search_text.strip().upper()
    seg = master[master['exch_seg'] == exchange]
    if seg.empty:
        return []
    mask = seg['symbol'].astype(str).str.upper().str.contains(text, na=False, regex=False) | \
           seg['name'].astype(str).str.upper().str.contains(text, na=False, regex=False)
    hits = seg[mask].head(limit)
    return [{'tradingsymbol': row['symbol'], 'symboltoken': str(row['token']),
             'name': row.get('name', '')} for _, row in hits.iterrows()]


def _fetch_chunk(obj, exchange, token, angel_interval, from_dt, to_dt, max_retries=3):
    """Single chunked call with retry/backoff on rate-limit responses."""
    params = {
        'exchange': exchange,
        'symboltoken': token,
        'interval': angel_interval,
        'fromdate': from_dt.strftime('%Y-%m-%d %H:%M'),
        'todate': to_dt.strftime('%Y-%m-%d %H:%M'),
    }
    last_err = None
    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep([2, 5][min(attempt - 1, 1)])
        try:
            resp = obj.getCandleData(params)
            if resp.get('status'):
                return resp.get('data', []), None
            msg = str(resp.get('message', '')).lower()
            if 'rate' in msg or 'throttle' in msg:
                last_err = resp.get('message')
                continue
            return [], resp.get('message', 'fetch failed')
        except Exception as e:
            last_err = str(e)
            time.sleep(1)
    return [], last_err or 'failed after retries'


def fetch_angel_one_history(obj, exchange: str, token: str, interval_label: str,
                             from_date, to_date, progress_cb=None):
    """
    from_date / to_date: python date objects. Chunks the request into
    pieces no longer than the interval's day limit - Angel One caps how
    much history you can request in a single call, and the cap gets much
    tighter at shorter intervals (30 days for 1-minute data vs. ~5 years
    for daily).

    Returns (df, error_message_or_None). df has a datetime index and
    Open/High/Low/Close/Volume columns - the same shape every other data
    source in this project produces, so it drops straight into
    find_zigzag_pivots() and every detector unchanged.
    """
    if interval_label not in INTERVAL_MAP:
        return pd.DataFrame(), f"Unknown interval: {interval_label}"
    angel_interval, max_days = INTERVAL_MAP[interval_label]

    start = datetime.combine(from_date, datetime.min.time().replace(hour=9, minute=0))
    end = datetime.combine(to_date, datetime.min.time().replace(hour=23, minute=59))
    if end <= start:
        return pd.DataFrame(), "Empty date range."

    chunks = []
    errors = []
    cursor = start
    total_days = max((end - start).days, 1)
    n_chunks_est = max(1, -(-total_days // max_days))
    done = 0

    while cursor < end:
        chunk_end = min(cursor + timedelta(days=max_days), end)
        rows, err = _fetch_chunk(obj, exchange, token, angel_interval, cursor, chunk_end)
        if err and not rows:
            errors.append(err)
        if rows:
            chunks.extend(rows)
        cursor = chunk_end + timedelta(minutes=1)
        done += 1
        if progress_cb:
            progress_cb(min(done / n_chunks_est, 1.0))
        time.sleep(0.35)

    if not chunks:
        if errors:
            return pd.DataFrame(), f"No data returned - last API message: {errors[-1]}"
        return pd.DataFrame(), "No data returned for this symbol/range (empty response, no error given)."

    df = pd.DataFrame(chunks, columns=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    df = df.drop_duplicates(subset='datetime').sort_values('datetime').set_index('datetime')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(), None
