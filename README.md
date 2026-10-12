# pattern_scanner

A prototype that detects chart patterns (currently: broadening bottom,
head and shoulders top) directly from OHLC data and draws them on a
candlestick chart — no manual eyeballing required. Works in both a
historical scan mode and a live polling/alert mode.

## How it works

1. **`pivots.py`** — turns raw OHLC bars into an alternating sequence of
   swing highs and swing lows using a percentage-reversal zigzag algorithm.
   This is the foundation every pattern rule is built on. Confirmation logic
   runs on Close prices (to avoid a single bar's intrabar spread being
   mistaken for a real multi-bar swing); the reported price at each pivot is
   the bar's actual High or Low.
2. **`patterns.py`** — one detector per pattern, each just a set of
   geometric checks against a window of consecutive pivots:
   - **Broadening bottom**: alternating pivots, preceded by a downtrend,
     where swing lows keep getting lower and swing highs keep getting
     higher (diverging trendlines, fitted by least-squares regression).
   - **Head and shoulders top**: 5 alternating pivots (high-low-high-low-high)
     where the middle high is tallest, the two outer highs are comparable,
     and the two troughs are roughly level (the neckline).

   Both detectors confirm breakout only on a **closing** price beyond the
   boundary (never an intrabar wick), compute a measured-move price target,
   and check whether price has moved at least 5% past the breakout before
   reversing (the standard "failure" filter).
3. **`plotting.py`** — a small self-contained candlestick renderer plus an
   overlay function that draws a match's swing points, trendlines, and
   breakout marker on top.
4. **`data_sources.py`** — yfinance-based historical + "latest bars" loading.
   Swap this out for any other source (broker API, ccxt, a CSV) — everything
   downstream just needs a DataFrame with `Open/High/Low/Close/Volume` and a
   datetime index.
5. **`live_monitor.py`** — polls `data_sources.fetch_latest` on an interval,
   re-runs the scanner on the tail of the data, and alerts (prints, or calls
   your own `on_alert` callback) the first time each breakout is confirmed.
   A `seen` set prevents re-alerting the same event on every subsequent poll.
6. **`synthetic_data.py`** — generates an OHLC series with a known
   broadening bottom and a known head-and-shoulders top stitched in. This
   sandbox can't reach Yahoo Finance (only package registries are
   reachable), so this is what `demo_historical.py` was actually tested
   against. On your machine, swap in `data_sources.fetch_historical('AAPL')`
   and it works the same way against real data.

## Running it

```bash
pip install -r requirements.txt

# historical scan on synthetic data (works anywhere, no internet needed)
python demo_historical.py
python demo_family1.py
python demo_extremes.py
python demo_leg_structure.py
python demo_gaps.py
python demo_curves.py
python demo_composite.py

# interactive web UI - browse any synthetic example, an uploaded CSV, a
# live yfinance ticker, or a real Angel One symbol/date-range/timeframe,
# scan, and click through the results (needs internet for the live-data
# options; Angel One additionally needs a real trading account with API
# access - see the "Angel One live data" section below)
streamlit run streamlit_app.py

# live monitoring on a real ticker (needs internet)
python demo_live.py AAPL --poll-seconds 3600
```

`demo_historical.py` prints every match found and saves an annotated PNG
per match. The two included in this delivery
(`pattern_0_broadening_bottom.png`, `pattern_1_head_and_shoulders_top.png`)
were generated this way against the synthetic data. Each of the other
`demo_*.py` scripts does the same for its own family, saving charts
prefixed accordingly (`family1_*.png`, `extremes_*.png`, `leg_*.png`,
`gap_*.png`, `curve_*.png`, `composite_*.png`).

### The web UI (`streamlit_app.py`)

A thin UI layer over the same detectors the demo scripts use - no new
pattern logic lives here. Pick a data source in the sidebar (28 of the
synthetic examples built while developing this library are available
directly, so you can browse every verified pattern type without needing
real data), choose which families to scan, hit **Scan for patterns**, and
click through the results table to see each match's annotated chart.
Uploading your own CSV needs a date column plus Open/High/Low/Close
(Volume optional, column names case-insensitive). The live-ticker option
uses the same `data_sources.fetch_historical()` as `demo_live.py`, so it
needs real internet access the way that script does.

Charts in the app are interactive Plotly candlesticks (`plotting_plotly.py`)
with a genuine crosshair - hovering shows a dashed line following the
cursor on both axes, with the exact price and date, plus the OHLC values
for whatever candle you're over. Pan and zoom work too, since that comes
for free with Plotly. This is a separate module from `plotting.py`'s
matplotlib renderer, which the `demo_*.py` scripts still use to save
static PNGs - a crosshair only means something on something you can
actually hover, so a static image was never going to give you one; the
two renderers are kept visually consistent (same colors, same layout
logic) but are genuinely different code for that reason.

## The two-line formation family (`formations.py`)

Broadening formations, triangles, wedges, and rectangles are all the same
underlying shape - two boundary lines fitted through alternating swing
highs and lows - differing only in the slope of each line and whether the
lines diverge, converge, or stay parallel. `detect_two_line_formations()`
implements this once, as a single detector fed a table (`FORMATION_DEFS`)
of ~15 named patterns, each just a combination of:

- `upper_slope` / `lower_slope`: `'up'`, `'down'`, or `'flat'`
- `convergence`: `'diverging'`, `'converging'`, or `'parallel'`
- optional preconditions: a required prior trend, a "flagpole" precondition
  for flag/pennant, a max duration for short-lived patterns

Adding a new pattern from this family is a matter of adding one `FormationDef`
entry, not writing new geometry code. **All 16 definitions are now verified.**
11 through `detect_two_line_formations()` with a dedicated isolated synthetic
example each (see `demo_family1.py` and the `family1_*.png` charts):
`triangle_ascending`, `triangle_descending`, `triangle_symmetrical`,
`wedge_rising`, `wedge_falling`, `rectangle_bottom`, `rectangle_top`,
`broadening_wedge_ascending`, `broadening_wedge_descending`,
`broadening_right_angled_ascending`, and `broadening_right_angled_descending`
all detect the correct pattern name, boundary lines, and breakout direction.
`broadening_bottom` / `broadening_top` are covered by the original hand-built
detector in `patterns.py` rather than re-tested through this generalized
engine, since that logic already matches the book's more precise
dual-breakout-method nuance.

Several of these were derived from an already-working example by reflecting
its prices vertically (`_reflect_anchors()` in `synthetic_data.py`) rather
than re-deriving the slope/threshold arithmetic by hand - e.g. the falling
wedge is just the rising wedge's prices mirrored, and it worked correctly
on the first try. That symmetry is a useful trick worth knowing about if
you add more patterns yourself: geometry that works for one direction
usually works for its mirror image with zero changes to the detector.

**`flag_bullish`, `flag_bearish`, and `pennant`** are the last 3, and they
needed a genuinely different approach rather than a synthetic-data tweak.
They don't go through `detect_two_line_formations()` at all - they have
their own function, `detect_flags_and_pennants()`. The reason: that
function works from confirmed zigzag pivots at ONE threshold for the whole
series, which is right for patterns spanning dozens of bars, but flags are
short (a handful to ~15 bars) and tight (a percent or two wide) - finding
one needs a much smaller threshold, and running the whole series at that
scale would flood everything else with noise pivots. The fix: scan
directly for a sharp "flagpole" move using the *normal*-scale zigzag
pivots (a pivot only confirms once price genuinely reverses, so consecutive
pivots reliably bracket the pole's real start and end - no guessing which
bar the "true" peak is), then compute pivots at a much smaller, *local*
threshold only within the short window right after the pole. A channel
that stays roughly parallel to the pole's own slope is a flag; one that
converges is a pennant. All three are confirmed working end to end
(`family1_flag_bullish.png`, `family1_flag_bearish.png`, `family1_pennant.png`).

**Known limitation - window selection.** `detect_two_line_formations()`
can still occasionally find a second, technically-valid match alongside
the intended one when formations sit close together - `rectangle_top`'s
isolated example reports a harmless extra `wedge_rising` match from its
own lead-in trend, for instance (the real `rectangle_top` is still found
correctly too). The general fix - scoring every candidate window and
solving for the best non-overlapping set across the whole series, instead
of greedily accepting the first valid match - remains a good next
improvement for cleaning up scans of long, real price histories with many
adjacent formations, even though every individual pattern type in this
family is now verified in isolation.

## The multi-touch extremes family (`extremes.py`)

Double tops/bottoms, triple tops/bottoms, and head-and-shoulders all share
a different shape than formations.py's two-line family: instead of fitting
a boundary LINE, they're about N peaks (or troughs) landing at a similar
price level, with a horizontal "confirmation line" at the level of the
point(s) between them.

- `detect_double_extreme()` / `detect_triple_extreme()` cover double and
  triple tops/bottoms in one parameterized detector each (`kind='top'` or
  `'bottom'`). Each touch is automatically classified "Adam" (a narrow,
  spike-like touch) or "Eve" (a wide, rounded touch spanning several bars)
  via `classify_touch_shape()`, so the pattern name - e.g.
  `double_top_adam_eve` - falls out of the data rather than being a
  separate detector per combination. That's Adam & Adam, Adam & Eve, Eve &
  Adam, and Eve & Eve all from one function.
- `detect_head_and_shoulders()` generalizes the original simple 3-peak
  detector to both directions (`kind='top'` or `'bottom'`) and any number
  of shoulders per side (`max_shoulders`) - `max_shoulders=1` is the
  textbook version, anything higher is Bulkowski's "complex"
  multi-shoulder variant, using the exact same rule.

Verified with isolated synthetic examples for `double_top`, `double_bottom`,
`triple_top`, `head_and_shoulders_bottom`, and `complex_head_and_shoulders_top`
(2 shoulders per side) - see `demo_extremes.py` and the `extremes_*.png`
charts. The complex example also correctly reports the simple
`head_and_shoulders_top` nested inside it as a second, valid match (using
just the innermost shoulder pair) - that's a real sub-pattern, not a bug,
the same way a rectangle can sit inside a larger consolidation.

## The leg-structure family (`leg_structure.py`)

Measured moves and three rising valleys/falling peaks compare the
MAGNITUDE of consecutive zigzag legs to each other, rather than fitting a
boundary line or matching touches at a similar level.

- `detect_measured_move()` looks for low→high→low→high (or the mirror):
  a first leg, a partial retracement (15-75% of the first leg), then a
  second leg of similar size. Unlike every reversal pattern elsewhere in
  this library, this is a trend-CONTINUATION structure with no breakout
  to confirm - it's read once the second leg matches the first closely
  enough, not once price crosses a line.
- `detect_three_rising_valleys()` / `detect_three_falling_peaks()` look
  for three strictly increasing lows (or decreasing highs) - a single
  rising (or falling) support/resistance line, simpler than a full
  triangle since only one side needs to trend.

Verified with isolated synthetic examples for all four - see
`demo_leg_structure.py` and the `leg_*.png` charts.

## The gap family (`gaps.py`)

The only detectors in this library that don't start from zigzag pivots -
a gap is a bar-to-bar event (today's range doesn't overlap yesterday's at
all), so `find_gaps()` scans every consecutive pair of bars directly.
`classify_gaps()` then labels each gap **breakaway** (within a few bars of
the last swing pivot - the gap that kicks off a new leg), **exhaustion**
(many bars into an already-extended swing - often precedes a reversal),
or **runaway/continuation** (mid-swing, neither of the above), using
ordinary zigzag pivots to know where in the swing the gap sits.
`detect_island_reversal()` looks for two opposite-direction gaps within a
short span, isolating a small cluster of bars.

Because plain anchor interpolation always keeps each bar's range
continuous with the last (Open is derived from the previous Close), a
genuine gap can't emerge from it on its own - `synthetic_data.py` has a
dedicated `_inject_gap()` helper that shifts a bar (and everything after
it) by a fixed amount to create one on purpose. Verified with isolated
examples for `breakaway_gap`, `exhaustion_gap`, and `island_top` - see
`demo_gaps.py` and the `gap_*.png` charts.

## The curved shapes family (`curves.py`)

Rounding bottoms/tops and cup with handle are smooth bowl/dome shapes
rather than a sequence of sharp zigzag pivots, so they need a genuinely
different primitive: `detect_rounding()` fits a parabola (quadratic) to a
sliding window of Close prices, keeping fits with the right curvature
sign (opens upward for a bottom, downward for a top), a decent R², and a
vertex that falls roughly in the middle of the window rather than at an
edge. `detect_cup_with_handle()` builds on top of it - the cup IS a
rounding bottom - by additionally requiring a shallow pullback (the
handle) right after the rim before confirming a breakout above it.

**Known tuning note.** A symmetric parabola looks parabolic at many
sub-window scales, so the generic window search can settle on a shorter,
still-well-fitting piece of a longer curve rather than the "obviously
intended" full shape - `demo_curves.py`'s cup-with-handle example calls
`detect_cup_with_handle()` with an explicit, narrower window range for
this reason (see the comment at the top of that script). Not a
correctness bug - every fit found is genuinely a good parabola - just
something to be aware of when scanning real data with the wide default
range.

Verified with isolated synthetic examples for all three - see
`demo_curves.py` and the `curve_*.png` charts. Scallops (repeated rounding
turns in a staircase) aren't implemented yet - a natural extension of
`detect_rounding()` for anyone who wants to pick it up, but genuinely more
work (composing several rounding matches together) than a synthetic-data
tweak.

## The composite family (`composite.py`, plus scallops in `curves.py`)

Diamond, horn, and pipe were named in the original family taxonomy but
left unbuilt until now - each is a tight variant or combination of
primitives that already exist elsewhere in this library, not a new
detection method:

- `detect_diamond()` looks for 5 alternating pivots (touch, mid, touch,
  mid, touch) where the swing between pivots grows then shrinks - the
  widest leg has to sit in the middle of the window, not at either end.
  (An earlier version tried to fit and pair diverging/converging boundary
  lines the way formations.py does for triangles - that pairing logic
  turned out to be wrong whenever the window could legally start on
  either a high or a low, giving nonsensical width comparisons. The
  leg-magnitude version above is simpler and doesn't have that bug.)
  Breakout and target use the same closing-price and measured-move
  conventions as everywhere else.
- `detect_horn()` is `extremes.detect_double_extreme()` with an added
  constraint that the two touches land close together in time - literally
  the same detector, just tighter.
- `detect_pipe()` is the same idea again, tighter still, with the added
  requirement that both touches are "Adam"-shaped (narrow spikes, via
  `classify_touch_shape()`) rather than rounded.
- `detect_scallops()` (in `curves.py`, next to the rounding detector it's
  built on) chains consecutive rounding-bottom matches into a staircase
  where each one's rim sits higher than the last (`kind='ascending'`), or
  the mirror for rounding tops (`kind='descending'`).

Verified with isolated synthetic examples for `diamond_top`,
`diamond_bottom`, `horn_top`, `pipe_bottom`, and `scallop_ascending` - see
`demo_composite.py` and the `composite_*.png` charts.

## Angel One live data (`angel_one.py`)

An alternative to the yfinance path in `data_sources.py`, for pulling real
NSE/BSE/MCX candles instead of synthetic or Yahoo Finance data - wired into
`streamlit_app.py` as an "Angel One (live)" data source alongside the
others. Needs a real Angel One trading account with API access enabled
(API key, client code, password/MPIN, and a TOTP secret from enabling TOTP
in account settings).

The auth and chunking follow the same conventions used across Angel
One-integrated apps generally: TOTP login via `pyotp` + `SmartConnect`,
then `getCandleData` in chunks sized to each interval's documented day
limit (kept a little under the published cap as safety margin). Since
Angel One's historical endpoint takes a symbol *token*, not a ticker,
`search_instrument()` wraps `searchScrip()` so the UI can resolve a name
like "RELIANCE" to its token before fetching.

This can't be exercised end-to-end in a sandboxed environment, the same
way the yfinance path can't - but the code path was verified as far as
it's possible to: with the real `smartapi-python` package installed and a
login attempt actually made, it reached a real network call to
`apiconnect.angelone.in` and failed only because sandbox network egress
blocks that host, not because of a bug. Invalid credentials against a
real network also fail with a clear, visible error rather than a crash.
On your own machine with a real account, this should just work.

**Using it**: in the Streamlit sidebar, pick "Angel One (live)", log in
(or fill in `secrets.toml` under an `[angel_one]` section with `api_key`,
`client_id`, `password`, `totp_key` and it'll be picked up automatically,
same as your other Angel One apps), pick an exchange, search for a symbol
and select the match, choose a date range and timeframe, and fetch. The
scan then runs on that real data exactly the way it runs on synthetic
data - which is the actual point: pick real data, run the scan, look at
the annotated chart, and judge for yourself whether the detection is
right, rather than trusting it blind.

**Remembering credentials.** The login form has a "💾 Remember these
credentials on this machine" checkbox - checking it and clicking Login
writes them to `.streamlit/secrets.toml` next to the app (creating the
file/folder if needed, merging cleanly with any other sections already in
there rather than overwriting them). Next time the app starts, `st.secrets`
picks that file up automatically and the form auto-fills - no retyping.
This save happens independently of whether the login itself succeeds, so
it also works as a way to just persist what you've typed even while
you're still troubleshooting a login issue. It's a local-machine
convenience, the same file-based pattern your other Angel One apps
already use for `secrets.toml` — plain text, not encrypted, so keep the
file out of version control (a `.gitignore` entry for `.streamlit/` is
standard for exactly this reason) — and on Streamlit Community Cloud it
only persists until that instance restarts or redeploys, since the
filesystem there is temporary; use the platform's own Secrets manager for
anything actually deployed.

**Troubleshooting "No module named 'pyotp'" (or 'SmartApi').** This means
whatever Python environment is actually running `streamlit run` doesn't
have that package installed - usually because `pip install -r
requirements.txt` was run in a different environment/venv than the one
Streamlit is launched from, or requirements.txt was updated after the
last install. The sidebar now shows a dependency diagnostic with the
exact `pip install ...` command the moment you select "Angel One (live)",
so this should be self-explanatory going forward rather than surfacing as
a raw import error after clicking Login.

**Troubleshooting "nothing seems to happen" on fetch.** The most common
cause: the symbol token field needs Angel One's internal *numeric* token
(e.g. `99926000` for NIFTY 50), not the trading symbol name - typing
`NIFTY50` there looks reasonable but silently fails against the API.
Two fixes now in place for this: a **quick-pick dropdown for NIFTY 50 /
BANKNIFTY / SENSEX** with their known-good tokens baked in (indices don't
reliably show up in symbol search anyway, since they aren't tradeable
instruments the way stocks are, so search isn't the right tool for them
regardless), and the manual token field now warns immediately if what's
typed isn't numeric, plus refuses to attempt the fetch at all rather than
sending a request that's going to fail quietly. For anything other than
these three indices, use Search rather than guessing a token by hand.

**Troubleshooting "Search successful. No matching trading symbols found"
even for symbols that clearly exist.** This is Angel One's *live*
`searchScrip` endpoint, and it's genuinely unreliable in practice - a
widely reported quirk, not specific to any query you type. Symbol search
no longer uses it: it now downloads Angel One's full instrument master
(the same file, and the same technique, your `continuous_contract.py`
already uses for MCX futures chains) and searches that locally instead -
no live-endpoint quirks, no rate limits, and it actually finds real
symbols. It's a large file, so the first search in a session takes a few
seconds while it downloads; after that it's cached for the rest of the
day. The old live-search function (`search_instrument()`) is still in
`angel_one.py` if you ever want it, just not what the UI uses anymore.

## Tuning

- `pct_threshold` (in `scan()` / `find_zigzag_pivots()`) — how big a reversal
  (as a fraction, e.g. 0.03 = 3%) is needed to confirm a swing point. Lower
  = more, noisier pivots; higher = fewer, more significant ones. Daily bars
  on a typical stock: 3–6% is a reasonable starting range. Intraday bars
  will usually want something smaller.
- `shoulder_tol` / `neckline_tol` (head and shoulders) — how much the two
  shoulders / two neckline points are allowed to differ, as a fraction.
- `min_touches` (broadening bottom) — minimum alternating pivots required;
  4 is the loosest valid formation, 5+ requires an extra touch.

## Known limitations (this is a prototype)

- Chart patterns are inherently fuzzy — Bulkowski's own eye is doing a lot
  of subjective work deciding what "counts." These rules are a reasonable
  first approximation, not a perfect replica of his judgment.
- Only 2 of the dozens of patterns in the book are implemented so far.
  Adding another is a matter of writing one more `detect_*` function in
  `patterns.py` following the same shape.
- The live monitor is polling-based, not a tick-level feed — fine for
  daily/hourly bars, not for sub-second reaction on intraday scalping
  timeframes.
- No position sizing, order execution, or brokerage integration — this
  detects and alerts, it doesn't trade.
