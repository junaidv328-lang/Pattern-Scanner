"""
Streamlit UI for pattern_scanner.

Run with:
    streamlit run streamlit_app.py

Wraps every detector built so far behind a simple flow: pick a data
source (a synthetic example, an uploaded CSV, a live yfinance ticker, or
a real Angel One symbol/date-range/timeframe), pick which pattern
families to scan for, and browse the results with an annotated chart per
match - so a real chart's detections can be eyeballed against the real
candles, not just trusted blind. All the actual detection logic lives in
the pattern_scanner package - this file is UI only.
"""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from pattern_scanner.pivots import find_zigzag_pivots
from pattern_scanner.patterns import detect_broadening_bottom
from pattern_scanner.formations import detect_two_line_formations, detect_flags_and_pennants
from pattern_scanner.extremes import detect_double_extreme, detect_triple_extreme, detect_head_and_shoulders
from pattern_scanner.leg_structure import (
    detect_measured_move, detect_three_rising_valleys, detect_three_falling_peaks,
)
from pattern_scanner.gaps import classify_gaps, detect_island_reversal
from pattern_scanner.curves import detect_rounding, detect_cup_with_handle, detect_scallops
from pattern_scanner.composite import detect_diamond, detect_horn, detect_pipe
from pattern_scanner import angel_one
from pattern_scanner.plotting_plotly import plot_candles_plotly, plot_match_plotly, plot_points_plotly
from pattern_scanner import synthetic_data as sd

st.set_page_config(page_title="Chart Pattern Scanner", layout="wide", page_icon="📈")


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_instrument_master() -> pd.DataFrame:
    """Angel One's full instrument list, cached for a day - it's tens of
    MB, so this should only actually hit the network once daily regardless
    of how many searches get run in between."""
    return angel_one.fetch_instrument_master()


FAMILIES = {
    "Two-line formations (triangles, wedges, rectangles, broadening)": "two_line",
    "Flags & pennants": "flags",
    "Multi-touch extremes (double/triple tops & bottoms, head & shoulders)": "extremes",
    "Leg structure (measured moves, rising valleys / falling peaks)": "legs",
    "Gaps & islands": "gaps",
    "Curved shapes (rounding, cup with handle, scallops)": "curves",
    "Composite (diamond, horn, pipe)": "composite",
}

SYNTHETIC_EXAMPLES = {
    "Broadening bottom + head & shoulders top": sd.build_synthetic_ohlc,
    "Ascending triangle": sd.build_synthetic_triangle_ascending,
    "Descending triangle": sd.build_synthetic_triangle_descending,
    "Symmetrical triangle": sd.build_synthetic_triangle_symmetrical,
    "Rising wedge": sd.build_synthetic_wedge_rising,
    "Falling wedge": sd.build_synthetic_wedge_falling,
    "Rectangle bottom": sd.build_synthetic_rectangle_bottom,
    "Rectangle top": sd.build_synthetic_rectangle_top,
    "Broadening wedge ascending": sd.build_synthetic_broadening_wedge_ascending,
    "Broadening wedge descending": sd.build_synthetic_broadening_wedge_descending,
    "Right-angled broadening ascending": sd.build_synthetic_broadening_right_angled_ascending,
    "Right-angled broadening descending": sd.build_synthetic_broadening_right_angled_descending,
    "Bullish flag": sd.build_synthetic_flag,
    "Bearish flag": sd.build_synthetic_flag_bearish,
    "Pennant": sd.build_synthetic_pennant,
    "Double top": sd.build_synthetic_double_top,
    "Double bottom": sd.build_synthetic_double_bottom,
    "Triple top": sd.build_synthetic_triple_top,
    "Head & shoulders bottom": sd.build_synthetic_head_and_shoulders_bottom,
    "Complex head & shoulders top": sd.build_synthetic_complex_head_and_shoulders_top,
    "Measured move up": sd.build_synthetic_measured_move_up,
    "Measured move down": sd.build_synthetic_measured_move_down,
    "Three rising valleys": sd.build_synthetic_three_rising_valleys,
    "Three falling peaks": sd.build_synthetic_three_falling_peaks,
    "Breakaway gap": sd.build_synthetic_breakaway_gap,
    "Exhaustion gap": sd.build_synthetic_exhaustion_gap,
    "Island top": sd.build_synthetic_island_top,
    "Rounding bottom": sd.build_synthetic_rounding_bottom,
    "Rounding top": sd.build_synthetic_rounding_top,
    "Cup with handle": sd.build_synthetic_cup_with_handle,
    "Scallop ascending": sd.build_synthetic_scallop_ascending,
    "Diamond top": sd.build_synthetic_diamond_top,
    "Diamond bottom": sd.build_synthetic_diamond_bottom,
    "Horn top": sd.build_synthetic_horn_top,
    "Pipe bottom": sd.build_synthetic_pipe_bottom,
}


def run_scan(df: pd.DataFrame, pct_threshold: float, families: set[str]) -> list:
    pivots = find_zigzag_pivots(df, pct_threshold)
    matches = []

    if "two_line" in families:
        matches += detect_two_line_formations(pivots, df)
        matches += detect_broadening_bottom(pivots, df)
    if "flags" in families:
        matches += detect_flags_and_pennants(df)
    if "extremes" in families:
        matches += detect_double_extreme(pivots, df, kind='top')
        matches += detect_double_extreme(pivots, df, kind='bottom')
        matches += detect_triple_extreme(pivots, df, kind='top')
        matches += detect_triple_extreme(pivots, df, kind='bottom')
        matches += detect_head_and_shoulders(pivots, df, kind='top')
        matches += detect_head_and_shoulders(pivots, df, kind='bottom')
    if "legs" in families:
        matches += detect_measured_move(pivots, df, direction='up')
        matches += detect_measured_move(pivots, df, direction='down')
        matches += detect_three_rising_valleys(pivots, df)
        matches += detect_three_falling_peaks(pivots, df)
    if "gaps" in families:
        matches += classify_gaps(df, pct_threshold=pct_threshold)
        matches += detect_island_reversal(df)
    if "curves" in families:
        matches += detect_rounding(df, kind='bottom')
        matches += detect_rounding(df, kind='top')
        matches += detect_cup_with_handle(df)
        matches += detect_scallops(df, kind='ascending')
        matches += detect_scallops(df, kind='descending')
    if "composite" in families:
        matches += detect_diamond(pivots, df, kind='top')
        matches += detect_diamond(pivots, df, kind='bottom')
        matches += detect_horn(pivots, df, kind='top')
        matches += detect_horn(pivots, df, kind='bottom')
        matches += detect_pipe(pivots, df, kind='top')
        matches += detect_pipe(pivots, df, kind='bottom')

    return matches


def render_overview_chart(df: pd.DataFrame, title: str):
    return plot_candles_plotly(df, title)


def render_match_chart(df: pd.DataFrame, match):
    if match.lines:
        return plot_match_plotly(df, match)
    # gaps/islands don't carry fitted lines - mark the points on plain candles instead
    return plot_points_plotly(df, match)


# ------------------------------------------------------------------ sidebar
st.sidebar.header("1. Data")
source = st.sidebar.radio("Source", ["Synthetic example", "Upload CSV", "Live ticker (yfinance)",
                                      "Angel One (live)"])

df = None
data_label = ""

if source == "Synthetic example":
    choice = st.sidebar.selectbox("Example", list(SYNTHETIC_EXAMPLES.keys()))
    df = SYNTHETIC_EXAMPLES[choice]()
    data_label = choice

elif source == "Upload CSV":
    st.sidebar.caption("Needs a date column plus Open, High, Low, Close (Volume optional).")
    uploaded = st.sidebar.file_uploader("OHLC CSV", type="csv")
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        date_col = next((c for c in raw.columns if c.lower() in ('date', 'datetime', 'timestamp')),
                         raw.columns[0])
        raw[date_col] = pd.to_datetime(raw[date_col])
        raw = raw.set_index(date_col).sort_index()
        raw = raw.rename(columns={c: c.title() for c in raw.columns})
        missing = [c for c in ['Open', 'High', 'Low', 'Close'] if c not in raw.columns]
        if missing:
            st.sidebar.error(f"Missing columns: {missing}")
        else:
            df = raw
            data_label = uploaded.name

elif source == "Live ticker (yfinance)":
    st.sidebar.caption("Needs real internet access to Yahoo Finance - won't work from a "
                        "sandboxed/offline environment.")
    ticker = st.sidebar.text_input("Ticker", "AAPL")
    period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y", "5y", "max"], index=2)
    interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m", "15m"], index=0)
    if st.sidebar.button("Fetch"):
        try:
            from pattern_scanner.data_sources import fetch_historical
            st.session_state['live_df'] = fetch_historical(ticker, period=period, interval=interval)
            st.session_state['live_label'] = f"{ticker} ({interval})"
        except Exception as e:
            st.sidebar.error(f"Fetch failed: {e}")
    if 'live_df' in st.session_state:
        df = st.session_state['live_df']
        data_label = st.session_state['live_label']

else:  # Angel One (live)
    st.sidebar.caption("Needs a real Angel One trading account with API access enabled - won't "
                        "work from a sandboxed/offline environment. Same TOTP auth pattern as "
                        "your other Angel One apps: secrets.toml first, manual entry fallback.")

    missing_deps = angel_one.check_dependencies()
    if missing_deps:
        st.sidebar.error(
            f"Missing package(s) in this environment: **{', '.join(missing_deps)}**.\n\n"
            f"Run this in the same environment you launch Streamlit from, then restart the app:\n\n"
            f"`pip install {' '.join(missing_deps)}`\n\n"
            f"(or `pip install -r requirements.txt` — they're in there. If you already ran that "
            f"and still see this, it likely installed into a different Python environment than "
            f"the one running `streamlit run`.)"
        )

    def _angel_secrets():
        try:
            if "angel_one" in st.secrets:
                sec = st.secrets["angel_one"]
                return {"api_key": sec.get("api_key", ""), "client_id": sec.get("client_id", ""),
                        "password": sec.get("password", ""), "totp_key": sec.get("totp_key", "")}
        except Exception:
            pass
        return {}

    creds = _angel_secrets()
    with st.sidebar.expander("Login", expanded=('angel_obj' not in st.session_state)):
        api_key = st.text_input("API key", value=creds.get("api_key", ""), type="password")
        client_id = st.text_input("Client ID", value=creds.get("client_id", ""))
        password = st.text_input("Password / MPIN", value=creds.get("password", ""), type="password")
        totp_key = st.text_input("TOTP secret", value=creds.get("totp_key", ""), type="password")
        remember = st.checkbox(
            "💾 Remember these credentials on this machine",
            value=False,
            help="Saves to .streamlit/secrets.toml next to the app, in plain text - the same file "
                 "st.secrets reads from, so next time the app starts these fields auto-fill. Local "
                 "use only: on Streamlit Community Cloud this resets on redeploy since the "
                 "filesystem there is temporary; use the platform's own Secrets manager instead "
                 "for anything deployed.",
        )

        if st.button("Login", type="primary"):
            if remember:
                try:
                    saved_path = angel_one.save_credentials(api_key, client_id, password, totp_key,
                                                              app_dir=Path(__file__).parent)
                    st.info(f"Credentials saved to `{saved_path}`.")
                except Exception as e:
                    st.warning(f"Couldn't save credentials: {e}")
            obj, err = angel_one.login(api_key, client_id, password, totp_key)
            if err:
                st.error(f"Login failed: {err}")
            else:
                st.session_state['angel_obj'] = obj
                st.success("Logged in")

    if 'angel_obj' in st.session_state:
        obj = st.session_state['angel_obj']

        quick_pick = st.sidebar.selectbox(
            "Quick pick: common index",
            ["-- none, search below --"] + list(angel_one.COMMON_INDICES.keys()),
            help="Indices don't reliably show up in symbol search (they aren't tradeable "
                 "instruments the way stocks are), so these are fixed, known-good tokens - "
                 "picking one here skips search and the exchange dropdown entirely.",
        )

        picked = None
        exchange = None
        if quick_pick != "-- none, search below --":
            info = angel_one.COMMON_INDICES[quick_pick]
            exchange = info["exchange"]
            picked = {"tradingsymbol": quick_pick, "symboltoken": info["token"]}
            st.sidebar.caption(f"Using **{quick_pick}** → exchange {exchange}, token {info['token']}")
        else:
            exchange = st.sidebar.selectbox("Exchange", ["NSE", "BSE", "MCX", "NFO", "BFO", "CDS"])

            search_text = st.sidebar.text_input("Symbol search (e.g. RELIANCE)")
            if st.sidebar.button("Search") and search_text:
                try:
                    with st.spinner("Loading instrument list (cached daily after the first time)..."):
                        master = _cached_instrument_master()
                    st.session_state['angel_results'] = angel_one.search_instrument_master(
                        master, exchange, search_text)
                    if not st.session_state['angel_results']:
                        st.sidebar.warning(f"No matches for '{search_text}' on {exchange}. Check "
                                            f"the exchange segment - equity is NSE/BSE, futures/"
                                            f"options are NFO/BFO/CDS, commodities are MCX.")
                except Exception as e:
                    st.sidebar.error(f"Couldn't load the instrument list: {e}")

            results = st.session_state.get('angel_results', [])
            if results:
                options = {f"{r.get('tradingsymbol', '?')}  ·  token {r.get('symboltoken', '?')}": r
                           for r in results}
                picked_label = st.sidebar.selectbox("Match", list(options.keys()))
                picked = options[picked_label]

        manual_token = st.sidebar.text_input(
            "...or a symbol token directly",
            help="Must be the numeric token Angel One assigns internally (e.g. 99926000), not "
                 "the trading symbol name (e.g. NIFTY50 won't work here).",
        )
        if manual_token and not manual_token.strip().isdigit():
            st.sidebar.warning(
                "That doesn't look like a numeric token - Angel One needs the numeric symbol "
                "token (e.g. 99926000 for NIFTY 50), not the trading symbol name. Use the quick "
                "pick above for indices, or Search for anything else."
            )

        col_a, col_b = st.sidebar.columns(2)
        from_date = col_a.date_input("From", value=date.today() - timedelta(days=365))
        to_date = col_b.date_input("To", value=date.today())
        interval_label = st.sidebar.selectbox("Timeframe", list(angel_one.INTERVAL_MAP.keys()), index=7)

        if st.sidebar.button("Fetch from Angel One", type="primary"):
            token = (picked or {}).get('symboltoken') or manual_token.strip()
            if not token:
                st.sidebar.error("Pick a quick-pick index, search and pick a match, or enter a "
                                  "token directly, first.")
            elif not token.isdigit():
                st.sidebar.error(f"'{token}' isn't a valid numeric symbol token - fetch not "
                                  f"attempted. See the note above the token field.")
            else:
                with st.spinner("Fetching..."):
                    fdf, err = angel_one.fetch_angel_one_history(obj, exchange, token, interval_label,
                                                                  from_date, to_date)
                if err:
                    st.sidebar.error(err)
                else:
                    st.session_state['angel_df'] = fdf
                    label = (picked or {}).get('tradingsymbol', token)
                    st.session_state['angel_label'] = f"{label} ({interval_label}, Angel One)"

    if 'angel_df' in st.session_state:
        df = st.session_state['angel_df']
        data_label = st.session_state['angel_label']

st.sidebar.header("2. Scan settings")
pct_threshold = st.sidebar.slider("Zigzag threshold (%)", 1.0, 8.0, 3.0, 0.5) / 100
family_labels = st.sidebar.multiselect("Pattern families", list(FAMILIES.keys()),
                                        default=list(FAMILIES.keys()))
families = {FAMILIES[f] for f in family_labels}

scan_clicked = st.sidebar.button("🔍 Scan for patterns", type="primary", disabled=(df is None))

# --------------------------------------------------------------------- main
st.title("📈 Chart Pattern Scanner")
st.caption("Automated detection of Bulkowski-style chart patterns from OHLC data")

if df is None:
    st.info("Choose a data source in the sidebar to get started.")
    st.stop()

if scan_clicked:
    with st.spinner("Scanning..."):
        st.session_state['matches'] = run_scan(df, pct_threshold, families)
        st.session_state['scanned_df'] = df
        st.session_state['scanned_label'] = data_label

if 'matches' not in st.session_state:
    st.subheader(data_label)
    st.plotly_chart(render_overview_chart(df, data_label), width='stretch')
    st.info("Click **Scan for patterns** in the sidebar to run the detectors.")
    st.stop()

matches = st.session_state['matches']
df = st.session_state['scanned_df']
st.subheader(st.session_state['scanned_label'])
st.success(f"Found {len(matches)} pattern match(es)")

if not matches:
    st.plotly_chart(render_overview_chart(df, data_label), width='stretch')
    st.stop()

rows = []
for i, m in enumerate(matches):
    status = "no breakout yet"
    if m.breakout:
        status = f"{m.breakout['direction']} @ {m.breakout['price']:.2f}"
        if m.failed is not None:
            status += " (failed)" if m.failed else " (held)"
    rows.append({
        "#": i,
        "Pattern": m.pattern.replace('_', ' ').title(),
        "Breakout": status,
        "Target": f"{m.target:.2f}" if m.target else "-",
    })

st.dataframe(pd.DataFrame(rows).set_index("#"), width='stretch')

selected = st.selectbox(
    "View chart for:",
    range(len(matches)),
    format_func=lambda i: f"{rows[i]['Pattern']} ({rows[i]['Breakout']})",
)

st.plotly_chart(render_match_chart(df, matches[selected]), width='stretch')

with st.expander("Match details"):
    m = matches[selected]
    st.json({
        "pattern": m.pattern,
        "breakout": {k: str(v) for k, v in m.breakout.items()} if m.breakout else None,
        "target": m.target,
        "failed": m.failed,
        "meta": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                 for k, v in m.meta.items()},
    })
