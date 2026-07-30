"""
Phase 5: interactive visualization for the Phase 3/4 portfolio backtest.

Layout (a 2-column grid): left half is one symbol's candlesticks +
entry/exit/weight-adjust markers, switchable via the topbar. The right half
is split top/bottom: top is "Portfolio Return" (equity + drawdown,
crosshair-synced to the left pane); bottom is a scrollable Sharpe/return
stats table (left) and a per-day portfolio weight pie chart (right), both
driven by whatever bar is under the cursor.
"""

import json
from collections import defaultdict
from typing import Callable, cast

import pandas as pd
from lightweight_charts import AbstractChart, Chart

from src.data.higher_tf_indicators import session_window_starts
from src.data.resample import resample_ohlcv
from src.engine.portfolio_backtest import Trade, WeightAdjustment
from src.engine.portfolio_pullback_backtest import PortfolioBacktestResult
from src.rules.base import Side

OPEN_COLOR = "#2196F3"
TP_COLOR = "#26a69a"
SL_COLOR = "#ef5350"
WA_COLOR = "#ffca28"
FLIP_COLOR = "#ab47bc"
PIE_COLORS = [
    OPEN_COLOR, TP_COLOR, SL_COLOR, WA_COLOR, FLIP_COLOR, "#8d6e63", "#26c6da", "#789262", "#ff8a65",
]
# Same system-ui stack the page body (and so the topbar symbol switcher) uses
# by default - the legend/table/custom widgets each had their own font
# choice (legend defaults to Monaco, custom divs were set to Monaco too),
# so none of them matched the topbar. Reuse this everywhere instead.
UI_FONT = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, '
    'Cantarell, "Helvetica Neue", sans-serif'
)


def _to_ns(time_col: pd.Series) -> pd.Series:
    """
    lightweight-charts converts the 'time' column via
    `astype('int64') // 10**9`, which assumes nanosecond resolution. pandas
    now parses timestamps as datetime64[us] by default, so without forcing ns
    here that division is off by 1000x and every bar collapses near epoch 0
    (renders as a 1970 axis with all bars/markers piled on top of each other).
    """
    return time_col.dt.as_unit("ns")


def _ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    out = df[cols].rename(columns={"timestamp": "time"})
    out["time"] = _to_ns(out["time"])
    return out


def _macd_series(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """Same formula as rules.entry.MACDGoldenCross, but the full series (for
    charting) rather than just the last two bars (for signal detection)."""
    closes = df["close"]
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return pd.DataFrame(
        {
            "time": _to_ns(df["timestamp"]),
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line,
        }
    )


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Same formula as rules.exit.atr, but the full rolling series (for
    charting) rather than just the latest value (for exit-rule evaluation)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.rolling(period).mean()
    return pd.DataFrame({"time": _to_ns(df["timestamp"]), "atr": atr})


def _by_symbol(items: list) -> dict[str, list]:
    by_symbol = defaultdict(list)
    for item in items:
        by_symbol[item.symbol].append(item)
    return by_symbol


def _set_markers(
    chart: Chart,
    aligned_df: pd.DataFrame,
    trades: list[Trade],
    weight_adjustments: list[WeightAdjustment],
    marker_time: Callable[[int], pd.Timestamp] | None = None,
) -> None:
    """
    `marker_time`, if given, overrides how a bar index maps to the time a
    marker is drawn at (default: that bar's own timestamp in `aligned_df`).
    Part D's higher-timeframe panes display resampled candles, not the raw
    5m bars a trade's entry_bar_idx/exit_bar_idx index into - a marker
    placed at the raw 5m timestamp wouldn't land on any bar that exists in
    that pane's series, so those callers snap it to the containing
    higher-timeframe candle's own start time instead (see
    `show_multi_timeframe_backtest`).
    """
    chart.clear_markers()
    times = aligned_df["timestamp"]

    def time_at(i: int) -> pd.Timestamp:
        return times.iloc[i] if marker_time is None else marker_time(i)

    # A FLIP's exit (closing the old side) and the very next trade's entry
    # (opening the reversed side) land on the same bar - drawing both an
    # "Open ..." arrow and a "FLIP ..." arrow there is redundant since the
    # flip already *is* the new open. Collapse the pair into one purple
    # arrow labeled with the direction just opened.
    skip_open = set()
    for idx, t in enumerate(trades):
        if t.reason == "FLIP" and idx + 1 < len(trades):
            nxt = trades[idx + 1]
            if nxt.entry_bar_idx == t.exit_bar_idx and nxt.side is not t.side:
                skip_open.add(idx + 1)

    markers = []
    for idx, t in enumerate(trades):
        is_long = t.side is Side.LONG
        if idx not in skip_open:
            markers.append(
                {
                    "time": time_at(t.entry_bar_idx),
                    "position": "below" if is_long else "above",
                    "shape": "arrow_up" if is_long else "arrow_down",
                    "color": OPEN_COLOR,
                    "text": "Open Long" if is_long else "Open Short",
                }
            )

        if t.reason == "FLIP" and idx + 1 < len(trades) and (idx + 1) in skip_open:
            new_side_long = trades[idx + 1].side is Side.LONG
            markers.append(
                {
                    "time": time_at(t.exit_bar_idx),
                    "position": "below" if new_side_long else "above",
                    "shape": "arrow_up" if new_side_long else "arrow_down",
                    "color": FLIP_COLOR,
                    "text": "FLIP Long" if new_side_long else "FLIP Short",
                }
            )
        else:
            side_label = "Long" if is_long else "Short"
            markers.append(
                {
                    "time": time_at(t.exit_bar_idx),
                    "position": "above" if is_long else "below",
                    "shape": "arrow_down" if is_long else "arrow_up",
                    "color": {"TP": TP_COLOR, "SL": SL_COLOR, "WA": WA_COLOR, "FLIP": FLIP_COLOR}[t.reason],
                    "text": f"{t.reason} {side_label}",
                }
            )
    for wa in weight_adjustments:
        markers.append(
            {
                "time": time_at(wa.bar_idx),
                "position": "inside",
                "shape": "circle",
                "color": WA_COLOR,
                "text": "WA",
            }
        )
    markers.sort(key=lambda m: m["time"])
    if markers:
        chart.marker_list(markers)


_DOT_REASONS = {"open": OPEN_COLOR, "TP": TP_COLOR, "SL": SL_COLOR, "WA": WA_COLOR, "FLIP": FLIP_COLOR}


def _create_trade_dot_series(main_chart: AbstractChart) -> dict[str, "Line"]:
    """
    One small-dot scatter series per event color (open/TP/SL/WA/FLIP),
    plotted directly on the candlestick pane at the *actual* fill price -
    unlike the arrow markers (which snap to a fixed position above/below
    whichever bar the trade lands on), these show exactly where within a
    bar's high/low range a trade actually filled. Most useful on the
    higher-timeframe tiles, where one displayed candle can span many 5m
    fills (the engine always executes at 5m grain - see
    `src/rules/higher_tf.py`'s module docstring) and a single "above/below
    the bar" arrow can't distinguish them.
    """
    series = {}
    for reason, color in _DOT_REASONS.items():
        line = main_chart.create_line(
            reason, color=color, width=1, price_line=False, price_label=False
        )
        line.run_script(
            f"{line.id}.series.applyOptions({{lineVisible: false, pointMarkersVisible: true, "
            f"pointMarkersRadius: 3, lastValueVisible: false, crosshairMarkerVisible: false}})"
        )
        # These are a plotting aid, not a real named series - drop their
        # auto-added legend row so the legend still just shows OHLC/volume.
        # Wrapped in try/catch: an uncaught JS exception here kills the rest
        # of the queued script batch in the pywebview thread (silently
        # blacking out the whole window), and item.row's parent isn't
        # reliably legend.div itself (nested wrapper elements observed in
        # practice), so removeChild can throw NotFoundError.
        line.run_script(f"""
            try {{
                const item = {main_chart.id}.legend._lines.find((l) => l.series == {line.id}.series)
                if (item) {{
                    {main_chart.id}.legend._lines = {main_chart.id}.legend._lines.filter((l) => l != item)
                    if (item.row && item.row.parentNode) {{
                        item.row.parentNode.removeChild(item.row)
                    }}
                }}
            }} catch (e) {{}}
        """)
        series[reason] = line
    return series


def _set_trade_dots(
    dot_series: dict[str, "Line"],
    trades: list[Trade],
    marker_time: Callable[[int], pd.Timestamp] | None,
    times: "pd.Series[pd.Timestamp]",
) -> None:
    def time_at(i: int) -> pd.Timestamp:
        return times.iloc[i] if marker_time is None else marker_time(i)

    rows_by_reason: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        rows_by_reason["open"].append({"time": time_at(t.entry_bar_idx), "open": t.entry_price})
        rows_by_reason[t.reason].append({"time": time_at(t.exit_bar_idx), t.reason: t.exit_price})

    for reason, line in dot_series.items():
        rows = rows_by_reason.get(reason, [])
        if not rows:
            line.set(None)
            continue
        df = pd.DataFrame(rows)
        df["time"] = _to_ns(pd.to_datetime(df["time"]))
        # Explicit per-point color, same convention already used for the
        # MACD histogram bars (`hist_df["color"] = ...`) - relying solely on
        # the series' single color set at creation time left some points
        # (reported: the first few on each chart, only visible once zoomed
        # in far enough) rendering black instead of their real color.
        df["color"] = _DOT_REASONS[reason]
        line.set(df)


def _build_stats_table(chart: Chart, sharpe_stats: dict[str, float], portfolio_return: float) -> None:
    table = chart.create_table(
        width=0.25,
        height=0.5,
        headings=("Metric", "Value"),
        widths=(0.6, 0.4),
        position="right",
        # The JS table always attaches a click listener to every row/cell,
        # regardless of whether func is set. Without a func, its callbackName
        # stays null and clicking fires "null_~_<rowId>" back to python -
        # there's no handler named "null", so that KeyError crashes the app.
        func=lambda _row: None,
    )
    table.new_row("Portfolio Return", f"{portfolio_return:.2%}")
    for symbol, sharpe in sharpe_stats.items():
        label = "Portfolio Sharpe" if symbol == "Portfolio" else f"{symbol} Sharpe"
        table.new_row(label, "n/a" if pd.isna(sharpe) else f"{sharpe:.2f}")

    # The table's row area is meant to scroll (overflow-y: auto is set on its
    # wrapper div) once rows exceed the fixed height, but that wrapper is a
    # child of a column flex container without flex/min-height set on it -
    # the default min-height:auto on flex children means it just grows to
    # fit its content and overflows instead of clipping+scrolling. Patch
    # that wrapper directly, and bump the font size while we're in there
    # (the table's own text was left at the library's hardcoded 12px).
    table.run_script(
        f"""
        {table.id}._div.style.fontSize = '15px'
        const scrollArea = {table.id}._div.querySelector('div')
        scrollArea.style.flex = '1 1 auto'
        scrollArea.style.minHeight = '0'
        """
    )


def _build_weight_pie(chart: Chart, symbols: list[str]) -> list[str]:
    """
    A CSS conic-gradient pie (no chart-library support for pie charts) with a
    color-swatch legend below it, both updated on hover by _wire_weight_hover.
    Floated right, in front of the stats table in DOM order, so it claims the
    rightmost quarter of the window and pushes the stats table to its left.
    """
    colors = [PIE_COLORS[i % len(PIE_COLORS)] for i in range(len(symbols))]
    var = chart.id
    legend_rows_js = "\n".join(
        f"""
        {{
            const row = document.createElement('div');
            row.style.display = 'flex';
            row.style.alignItems = 'center';
            row.style.gap = '6px';
            row.style.margin = '2px 0';
            const dot = document.createElement('span');
            dot.style.width = '10px';
            dot.style.height = '10px';
            dot.style.minWidth = '10px';
            dot.style.borderRadius = '50%';
            dot.style.background = '{color}';
            const label = document.createElement('span');
            label.innerText = '{symbol}: -';
            row.appendChild(dot);
            row.appendChild(label);
            {var}.pieLegendDiv.appendChild(row);
            {var}.pieLegendLabels.push(label);
        }}
        """
        for symbol, color in zip(symbols, colors)
    )
    chart.run_script(
        f"""
        {var}.pieContainer = document.createElement('div')
        {var}.pieContainer.style.width = '25%'
        {var}.pieContainer.style.height = '50%'
        {var}.pieContainer.style.float = 'right'
        {var}.pieContainer.style.boxSizing = 'border-box'
        {var}.pieContainer.style.display = 'flex'
        {var}.pieContainer.style.flexDirection = 'column'
        {var}.pieContainer.style.alignItems = 'center'
        {var}.pieContainer.style.justifyContent = 'center'
        {var}.pieContainer.style.overflowY = 'auto'
        {var}.pieContainer.style.backgroundColor = '#121417'
        {var}.pieContainer.style.color = '#d1d4dc'
        {var}.pieContainer.style.fontFamily = '{UI_FONT}'
        {var}.pieContainer.style.fontSize = '14px'
        {var}.pieContainer.style.padding = '8px'

        {var}.pieTitle = document.createElement('div')
        {var}.pieTitle.innerText = 'Weight'
        {var}.pieTitle.style.fontSize = '16px'
        {var}.pieTitle.style.marginBottom = '8px'
        {var}.pieContainer.appendChild({var}.pieTitle)

        {var}.pieDiv = document.createElement('div')
        {var}.pieDiv.style.width = '180px'
        {var}.pieDiv.style.height = '180px'
        {var}.pieDiv.style.minHeight = '180px'
        {var}.pieDiv.style.borderRadius = '50%'
        {var}.pieDiv.style.background = '#333'
        {var}.pieContainer.appendChild({var}.pieDiv)

        {var}.pieLegendDiv = document.createElement('div')
        {var}.pieLegendDiv.style.marginTop = '8px'
        {var}.pieContainer.appendChild({var}.pieLegendDiv)
        {var}.pieLegendLabels = []

        {legend_rows_js}

        window.containerDiv.appendChild({var}.pieContainer)
        """
    )
    return colors


def _build_return_title(chart: AbstractChart, equity_chart: AbstractChart, title: str = "Portfolio Return") -> None:
    chart.run_script(
        f"""
        {equity_chart.id}.returnTitle = document.createElement('div')
        {equity_chart.id}.returnTitle.innerText = '{title}'
        {equity_chart.id}.returnTitle.style.position = 'absolute'
        {equity_chart.id}.returnTitle.style.top = '4px'
        {equity_chart.id}.returnTitle.style.left = '8px'
        {equity_chart.id}.returnTitle.style.zIndex = '3000'
        {equity_chart.id}.returnTitle.style.color = '#d1d4dc'
        {equity_chart.id}.returnTitle.style.fontSize = '16px'
        {equity_chart.id}.returnTitle.style.fontFamily = '{UI_FONT}'
        {equity_chart.id}.returnTitle.style.pointerEvents = 'none'
        {equity_chart.id}.div.appendChild({equity_chart.id}.returnTitle)
        """
    )


def _create_legend_div(pane: AbstractChart, attr_name: str, font_size: int = 14) -> None:
    """
    An empty top-left, absolutely-positioned text div on `pane` (stashed as
    `{pane.id}.{attr_name}`), mirroring the main chart's OHLCV legend style.
    Content is filled in later by `_wire_indicator_hover` rather than by a
    per-pane crosshair subscription here, so the same value can be pushed in
    from hovering *any* of the synced panes, not just this one.
    """
    pane.run_script(
        f"""
        {pane.id}.{attr_name} = document.createElement('div')
        {pane.id}.{attr_name}.style.position = 'absolute'
        {pane.id}.{attr_name}.style.top = '4px'
        {pane.id}.{attr_name}.style.left = '8px'
        {pane.id}.{attr_name}.style.zIndex = '3000'
        {pane.id}.{attr_name}.style.color = '#d1d4dc'
        {pane.id}.{attr_name}.style.fontSize = '{font_size}px'
        {pane.id}.{attr_name}.style.fontFamily = '{UI_FONT}'
        {pane.id}.{attr_name}.style.pointerEvents = 'none'
        {pane.id}.div.appendChild({pane.id}.{attr_name})
        """
    )


def _wire_indicator_hover(
    chart: AbstractChart,
    macd_chart: AbstractChart,
    atr_chart: AbstractChart,
    indicator_lookup: dict[int, dict[str, float]],
) -> None:
    """
    Fills in the MACD/ATR legend divs (see `_create_legend_div`) from
    whichever of the three synced panes - main candlestick chart, MACD pane,
    or ATR pane - the cursor happens to be over.

    `create_subchart(sync=True)`'s built-in sync only mirrors the visual
    crosshair/native OHLCV legend between panes (see the similar comment on
    `_wire_weight_hover`); it doesn't forward custom
    `subscribeCrosshairMove` callbacks. Also, `param.seriesData` from one
    chart's crosshair event can only look up series that live on that same
    chart, so hovering the main chart can't directly read the MACD/ATR
    series values - each pane instead just reports the hovered time back to
    Python, which looks the values up in `indicator_lookup` (kept in sync
    with whichever symbol is currently displayed by `_set_macd`/`_set_atr`)
    and pushes the formatted text to both legend divs.

    This also fixes the same ambiguity `_create_legend_div` was written to
    route around: with fast/slow/histogram sharing one price scale, magnet
    crosshair's "snap to whichever series is pixel-closest" favors the
    fast/slow lines over the histogram, so a native per-series label isn't
    reliable here regardless of which pane is hovered.
    """
    handler_name = f"indicator_hover_{macd_chart.id.rsplit('.', 1)[-1]}"

    def on_move(time_str):
        if time_str in (None, "null", "undefined", ""):
            return
        try:
            t = int(float(time_str))
        except (TypeError, ValueError):
            return
        values = indicator_lookup.get(t, {})
        macd_v, signal_v, hist_v, atr_v = (
            values.get("macd"), values.get("signal"), values.get("histogram"), values.get("atr")
        )

        if (
            macd_v is not None and signal_v is not None and hist_v is not None
            and not any(pd.isna(v) for v in (macd_v, signal_v, hist_v))
        ):
            hist_color = TP_COLOR if hist_v >= 0 else SL_COLOR
            macd_html = (
                f"Fast: {macd_v:.3f}  Slow: {signal_v:.3f}  "
                f"<span style='color: {hist_color}'>Hist: {hist_v:.3f}</span>"
            )
        else:
            macd_html = ""
        atr_html = f"ATR: {atr_v:.3f}" if atr_v is not None and not pd.isna(atr_v) else ""

        macd_chart.run_script(f"{macd_chart.id}.macdLegend.innerHTML = `{macd_html}`")
        atr_chart.run_script(f"{atr_chart.id}.atrLegend.innerHTML = `{atr_html}`")

    chart.win.handlers[handler_name] = on_move
    subscribe_script = f"""
        chartObj.subscribeCrosshairMove(param => {{
            if (!param.time) return;
            window.callbackFunction(`{handler_name}_~_${{param.time}}`)
        }})
    """
    chart.run_script(subscribe_script.replace("chartObj", f"{chart.id}.chart"))
    macd_chart.run_script(subscribe_script.replace("chartObj", f"{macd_chart.id}.chart"))
    atr_chart.run_script(subscribe_script.replace("chartObj", f"{atr_chart.id}.chart"))


def _wire_return_legend(equity_chart: AbstractChart, bar_return_hist) -> None:
    """
    Mirrors the main chart's top-left OHLCV legend, but for this pane's
    per-bar return: a small text box, top-left, updated on hover, colored
    green/red to match the bar's sign. Third line, below the "Portfolio
    Return" title, the "Portfolio:" total line, and the "Baseline:" line
    (see `_wire_portfolio_legend`/`_wire_baseline_legend`) so none overlap.
    """
    equity_chart.run_script(
        f"""
        {equity_chart.id}.returnLegend = document.createElement('div')
        {equity_chart.id}.returnLegend.style.position = 'absolute'
        {equity_chart.id}.returnLegend.style.top = '68px'
        {equity_chart.id}.returnLegend.style.left = '8px'
        {equity_chart.id}.returnLegend.style.zIndex = '3000'
        {equity_chart.id}.returnLegend.style.color = '#d1d4dc'
        {equity_chart.id}.returnLegend.style.fontSize = '14px'
        {equity_chart.id}.returnLegend.style.fontFamily = '{UI_FONT}'
        {equity_chart.id}.returnLegend.style.pointerEvents = 'none'
        {equity_chart.id}.div.appendChild({equity_chart.id}.returnLegend)

        {equity_chart.id}.chart.subscribeCrosshairMove(param => {{
            const bar = param.time && param.seriesData.get({bar_return_hist.id}.series)
            if (bar) {{
                const v = bar.value
                {equity_chart.id}.returnLegend.innerText = `Daily Return: ${{v >= 0 ? '+' : ''}}${{v.toFixed(2)}}`
                {equity_chart.id}.returnLegend.style.color = v >= 0 ? '{TP_COLOR}' : '{SL_COLOR}'
            }} else {{
                {equity_chart.id}.returnLegend.innerText = ''
            }}
        }})
        """
    )


def _wire_portfolio_legend(equity_chart: AbstractChart, equity_line: "Line") -> None:
    """
    First line below the "Portfolio Return" title (top: 28px) - the
    portfolio equity curve's own dollar value at the hovered point, same
    top-left hover-text treatment as `_wire_return_legend`/`_wire_baseline_legend`.
    """
    equity_chart.run_script(
        f"""
        {equity_chart.id}.portfolioLegend = document.createElement('div')
        {equity_chart.id}.portfolioLegend.style.position = 'absolute'
        {equity_chart.id}.portfolioLegend.style.top = '28px'
        {equity_chart.id}.portfolioLegend.style.left = '8px'
        {equity_chart.id}.portfolioLegend.style.zIndex = '3000'
        {equity_chart.id}.portfolioLegend.style.color = '{OPEN_COLOR}'
        {equity_chart.id}.portfolioLegend.style.fontSize = '14px'
        {equity_chart.id}.portfolioLegend.style.fontFamily = '{UI_FONT}'
        {equity_chart.id}.portfolioLegend.style.pointerEvents = 'none'
        {equity_chart.id}.div.appendChild({equity_chart.id}.portfolioLegend)

        {equity_chart.id}.chart.subscribeCrosshairMove(param => {{
            const point = param.time && param.seriesData.get({equity_line.id}.series)
            if (point) {{
                {equity_chart.id}.portfolioLegend.innerText = `Portfolio: ${{point.value.toFixed(2)}}`
            }} else {{
                {equity_chart.id}.portfolioLegend.innerText = ''
            }}
        }})
        """
    )


def _wire_baseline_legend(equity_chart: AbstractChart, benchmark_line: "Line") -> None:
    """
    Same top-left hover-text treatment as `_wire_return_legend`'s "Return:"
    label, one row further down (48px vs 28px) so the two don't overlap -
    shows the buy&hold-with-daily-rebalanced-weights benchmark line's value
    at the hovered point.
    """
    equity_chart.run_script(
        f"""
        {equity_chart.id}.baselineLegend = document.createElement('div')
        {equity_chart.id}.baselineLegend.style.position = 'absolute'
        {equity_chart.id}.baselineLegend.style.top = '48px'
        {equity_chart.id}.baselineLegend.style.left = '8px'
        {equity_chart.id}.baselineLegend.style.zIndex = '3000'
        {equity_chart.id}.baselineLegend.style.color = '#787b86'
        {equity_chart.id}.baselineLegend.style.fontSize = '14px'
        {equity_chart.id}.baselineLegend.style.fontFamily = '{UI_FONT}'
        {equity_chart.id}.baselineLegend.style.pointerEvents = 'none'
        {equity_chart.id}.div.appendChild({equity_chart.id}.baselineLegend)

        {equity_chart.id}.chart.subscribeCrosshairMove(param => {{
            const point = param.time && param.seriesData.get({benchmark_line.id}.series)
            if (point) {{
                {equity_chart.id}.baselineLegend.innerText = `Baseline: ${{point.value.toFixed(2)}}`
            }} else {{
                {equity_chart.id}.baselineLegend.innerText = ''
            }}
        }})
        """
    )


def _wire_weight_hover(
    chart: AbstractChart,
    equity_chart: AbstractChart,
    weights_history: pd.DataFrame,
    symbols: list[str],
    pie_colors: list[str],
) -> None:
    """
    Subscribes to both the candlestick and the equity/return chart's
    crosshair move (hovering either one should drive the pie the same way -
    sync_crosshairs_only only syncs the visual crosshair/legend between the
    two charts, it doesn't forward custom subscribeCrosshairMove callbacks
    like this one, so each chart needs its own subscription). Whenever the
    hovered bar's calendar date changes, rewrites the weight pie/legend
    (fixed symbol order, matching each symbol's color) with that day's
    per-symbol weights.
    """
    handler_name = f"wa_hover_{chart.id.rsplit('.', 1)[-1]}"
    state: dict[str, object] = {"last_date": None}

    def on_move(time_str):
        if time_str in (None, "null", "undefined", ""):
            return
        try:
            hovered_date = pd.Timestamp(int(float(time_str)), unit="s", tz="UTC").date()
        except (TypeError, ValueError):
            return
        if hovered_date == state["last_date"] or hovered_date not in weights_history.index:
            return
        state["last_date"] = hovered_date

        day_weights = cast(
            "dict[str, float]", cast(pd.Series, weights_history.loc[hovered_date]).to_dict()
        )

        cum = 0.0
        stops = []
        for symbol, color in zip(symbols, pie_colors):
            start = cum
            cum += day_weights.get(symbol, 0.0) * 100
            stops.append(f"{color} {start:.2f}% {cum:.2f}%")
        if cum < 100:
            stops.append(f"#333 {cum:.2f}% 100%")
        legend_updates = "\n".join(
            f"{chart.id}.pieLegendLabels[{i}].innerText = '{symbol}: {day_weights.get(symbol, 0.0):.1%}'"
            for i, symbol in enumerate(symbols)
        )
        chart.run_script(
            f"""
            {chart.id}.pieDiv.style.background = 'conic-gradient({", ".join(stops)})'
            {legend_updates}
            """
        )

    chart.win.handlers[handler_name] = on_move
    subscribe_script = f"""
        chartObj.subscribeCrosshairMove(param => {{
            if (!param.time) return;
            window.callbackFunction(`{handler_name}_~_${{param.time}}`)
        }})
    """
    chart.run_script(subscribe_script.replace("chartObj", f"{chart.id}.chart"))
    equity_chart.run_script(subscribe_script.replace("chartObj", f"{equity_chart.id}.chart"))


def show_portfolio_backtest(
    aligned: dict[str, pd.DataFrame],
    trades: list[Trade],
    weight_adjustments: list[WeightAdjustment],
    equity_curve: pd.Series,
    weights_history: pd.DataFrame,
    sharpe_stats: dict[str, float],
    portfolio_return: float,
) -> None:
    symbols = list(aligned.keys())
    trades_by_symbol = _by_symbol(trades)
    weight_adjustments_by_symbol = _by_symbol(weight_adjustments)

    # The library's own default window is 800x600 - too small for this grid
    # to stay legible, and a "restore down" from maximized falls back to it.
    # Open large enough from the start that the restored size still fits.
    # Left column height is split 0.6 main / 0.2 MACD / 0.2 ATR so the two
    # indicator subcharts (added below) have room to stack under the main
    # candlestick chart without encroaching on the right column.
    chart = Chart(
        title="Strategy Backtest Terminal", toolbox=True, inner_width=0.5, inner_height=0.6,
        width=1600, height=1000, maximize=True,
    )
    chart.legend(visible=True, font_size=16, font_family=UI_FONT)
    # The dashed "last value" price line every series draws by default,
    # extending clear across the pane from the latest bar leftward, gets in
    # the way once several panes/series are on screen at once - the legends
    # already show the current value, so the line itself is just clutter.
    # Disabled everywhere below (candlesticks, volume, and every indicator
    # series) while leaving the axis label (lastValueVisible) alone.
    chart.price_line(line_visible=False)
    chart.run_script(f"{chart.id}.volumeSeries.applyOptions({{priceLineVisible: false}})")
    # Bars here are 5-minute intraday, not daily - without time_visible the
    # timescale falls back to a daily/business-day tick formatter, which
    # mislabels sub-day data (e.g. showing a bogus 1970 year on ticks) and
    # makes the whole session look aggregated into a single tick.
    chart.time_scale(time_visible=True, seconds_visible=False)
    # Default crosshair mode floats at the raw mouse pixel, so the horizontal
    # line (and its right-axis value label) doesn't line up with the actual
    # bar's price/volume under the cursor - magnet snaps both to the nearest
    # data point's real value instead.
    chart.crosshair(mode="magnet")
    # Volume lives on its own "volume_scale" price scale, which is invisible
    # by default (no axis) - without a visible axis there's nowhere for the
    # crosshair to attach a value label, so it never shows one on hover.
    chart.run_script(f'{chart.id}.chart.priceScale("volume_scale").applyOptions({{visible: true}})')

    # Left column, below the main chart: MACD and ATR indicator panes, each
    # toggleable from the topbar. sync_crosshairs_only=False so the timescale
    # (zoom/pan), not just the crosshair, stays in lockstep with the main
    # chart in both directions. Hidden by default (display: none) until
    # their topbar button is toggled.
    macd_chart = chart.create_subchart(
        position="left", width=0.5, height=0.2, sync=True, sync_crosshairs_only=False
    )
    macd_chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
    # With three series (fast/slow lines + histogram) sharing one price
    # scale, magnet's horizontal line snaps to whichever is pixel-closest to
    # the mouse - in practice almost always the fast/slow lines, since the
    # histogram is compressed near zero by its scale margins. There's no
    # per-series way to pin that, so the horizontal line is disabled here
    # and replaced with an explicit Fast/Slow/Hist text readout instead
    # (_wire_macd_legend) - the vertical line still tracks time as normal.
    macd_chart.crosshair(mode="magnet", horz_visible=False)
    macd_line = macd_chart.create_line("macd", color=OPEN_COLOR, price_line=False)
    signal_line = macd_chart.create_line("signal", color=WA_COLOR, price_line=False)
    macd_hist = macd_chart.create_histogram(
        "histogram", color=TP_COLOR, scale_margin_top=0.8, scale_margin_bottom=0.0, price_line=False
    )
    _create_legend_div(macd_chart, "macdLegend")
    # The library lays every pane out with CSS float, which packs each new
    # float into whatever gap is free first rather than strictly under the
    # element above it - with the right column (equity/table/pie) appended
    # later in the DOM, the space beside the (now-shorter) main chart is
    # still empty when this pane is created, so it floats up into the right
    # column instead of sitting below the main chart, pushing the real right
    # column content down. Taking it out of the float flow entirely with a
    # pinned position leaves the main/equity/table/pie layout untouched.
    macd_chart.run_script(
        f"""
        {macd_chart.id}.wrapper.style.float = 'none'
        {macd_chart.id}.wrapper.style.position = 'absolute'
        {macd_chart.id}.wrapper.style.left = '0%'
        {macd_chart.id}.wrapper.style.display = 'none'
        """
    )

    atr_chart = chart.create_subchart(
        position="left", width=0.5, height=0.2, sync=True, sync_crosshairs_only=False
    )
    atr_chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
    atr_chart.crosshair(mode="magnet")
    atr_line = atr_chart.create_line("atr", color=OPEN_COLOR, price_line=False)
    _create_legend_div(atr_chart, "atrLegend")
    atr_chart.run_script(
        f"""
        {atr_chart.id}.wrapper.style.float = 'none'
        {atr_chart.id}.wrapper.style.position = 'absolute'
        {atr_chart.id}.wrapper.style.left = '0%'
        {atr_chart.id}.wrapper.style.display = 'none'
        """
    )

    # Keyed by unix seconds (matching the crosshair's `param.time`) so
    # `_wire_indicator_hover` can look up Fast/Slow/Hist/ATR for whatever
    # symbol is currently displayed, regardless of which pane triggered the
    # hover. All symbols share the same aligned timestamps, so switching
    # symbols just overwrites values in place rather than needing a reset.
    indicator_lookup: dict[int, dict[str, float]] = {}

    def _set_macd(df: pd.DataFrame) -> None:
        macd_df = _macd_series(df)
        macd_line.set(macd_df[["time", "macd"]])
        signal_line.set(macd_df[["time", "signal"]])
        hist_df = macd_df[["time", "histogram"]].copy()
        hist_df["color"] = [TP_COLOR if v >= 0 else SL_COLOR for v in hist_df["histogram"]]
        macd_hist.set(hist_df)
        seconds = macd_df["time"].astype("int64") // 10**9
        for t, macd_v, signal_v, hist_v in zip(seconds, macd_df["macd"], macd_df["signal"], macd_df["histogram"]):
            indicator_lookup.setdefault(int(t), {}).update(macd=macd_v, signal=signal_v, histogram=hist_v)

    def _set_atr(df: pd.DataFrame) -> None:
        atr_df = _atr_series(df)
        atr_line.set(atr_df)
        seconds = atr_df["time"].astype("int64") // 10**9
        for t, atr_v in zip(seconds, atr_df["atr"]):
            indicator_lookup.setdefault(int(t), {})["atr"] = atr_v

    _wire_indicator_hover(chart, macd_chart, atr_chart, indicator_lookup)

    # Indicator panes stack in the order they were opened - whichever is
    # toggled on first takes the slot right below the main chart, and a
    # second one opened afterwards lands below that. Closing one and
    # reopening it later re-appends it to the end of this list, so it drops
    # below any pane that's stayed open the whole time.
    indicator_charts = {"macd": macd_chart, "atr": atr_chart}
    open_order: list[str] = []
    PANE_TOPS = ("60%", "80%")

    def _reposition_panes() -> None:
        for i, key in enumerate(open_order):
            pane = indicator_charts[key]
            pane.run_script(f"{pane.id}.wrapper.style.top = '{PANE_TOPS[i]}'")

    def _make_toggle(key: str, widget_name: str):
        sub_chart = indicator_charts[key]

        def on_toggle(c):
            visible = c.topbar[widget_name].value
            if key in open_order:
                open_order.remove(key)
            if visible:
                open_order.append(key)
            sub_chart.run_script(
                f"{sub_chart.id}.wrapper.style.display = '{{}}'".format("block" if visible else "none")
            )
            _reposition_panes()

        return on_toggle

    chart.topbar.button("macd_toggle", "MACD", toggle=True, func=_make_toggle("macd", "macd_toggle"))
    chart.topbar.button("atr_toggle", "ATR", toggle=True, func=_make_toggle("atr", "atr_toggle"))

    # Top-right quadrant: portfolio return (equity + drawdown).
    equity_chart = chart.create_subchart(
        position="right", width=0.5, height=0.5, sync=True, sync_crosshairs_only=True
    )
    _build_return_title(chart, equity_chart)
    equity_chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
    equity_chart.crosshair(mode="magnet")
    equity_line = equity_chart.create_line("equity", color=OPEN_COLOR, price_line=False)
    # Per-bar P&L of the portfolio equity curve (not drawdown, which was an
    # unlabeled, always-red, hard-to-read blob) - green when that bar gained,
    # red when it lost, so the sign is legible at a glance.
    bar_return_hist = equity_chart.create_histogram(
        "bar_return", color=TP_COLOR, scale_margin_top=0.7, scale_margin_bottom=0.0, price_line=False
    )

    equity_df = equity_curve.rename("equity").reset_index()
    equity_df.columns = ["time", "equity"]
    equity_df["time"] = _to_ns(equity_df["time"])
    equity_line.set(equity_df)

    bar_return = equity_curve.diff().fillna(0.0)
    bar_return_df = bar_return.rename("bar_return").reset_index()
    bar_return_df.columns = ["time", "bar_return"]
    bar_return_df["time"] = _to_ns(bar_return_df["time"])
    bar_return_df["color"] = [TP_COLOR if v >= 0 else SL_COLOR for v in bar_return_df["bar_return"]]
    bar_return_hist.set(bar_return_df)
    _wire_return_legend(equity_chart, bar_return_hist)
    # create_subchart(sync=True) queues a one-time visible-range copy from
    # the main chart as a run_last script, which runs after everything else
    # on page load - including a plain equity_chart.fit() call, clobbering
    # it back out of scope. Queue this fit as run_last too so it actually
    # runs last and wins.
    equity_chart.run_script(f"{equity_chart.id}.chart.timeScale().fitContent()", run_last=True)

    # Bottom-right quadrant, split in two: pie (rightmost) is created first so
    # it claims the far-right quarter; the stats table then lands to its left.
    pie_colors = _build_weight_pie(chart, symbols)
    _build_stats_table(chart, sharpe_stats, portfolio_return)
    _wire_weight_hover(chart, equity_chart, weights_history, symbols, pie_colors)

    def on_symbol_change(c):
        symbol = c.topbar["symbol"].value
        chart.set(_ohlc(aligned[symbol]))
        _set_markers(chart, aligned[symbol], trades_by_symbol[symbol], weight_adjustments_by_symbol[symbol])
        chart.fit()
        _set_macd(aligned[symbol])
        _set_atr(aligned[symbol])
        macd_chart.fit()
        atr_chart.fit()

    chart.topbar.switcher("symbol", tuple(symbols), default=symbols[0], func=on_symbol_change)

    chart.set(_ohlc(aligned[symbols[0]]))
    _set_markers(chart, aligned[symbols[0]], trades_by_symbol[symbols[0]], weight_adjustments_by_symbol[symbols[0]])
    chart.fit()
    _set_macd(aligned[symbols[0]])
    _set_atr(aligned[symbols[0]])
    macd_chart.fit()
    atr_chart.fit()

    chart.show(block=True)


# --- v1.1 Part D: multi-timeframe strategy comparison -----------------------
#
# 4 rows, one per independently-backtested strategy variant (5m/15m/30m/1h -
# see src/run_phaseD.py and src/rules/higher_tf.py). Each row pairs that
# variant's own candlestick+MACD+ATR (left, driven by that timeframe's own
# indicators) with that variant's own equity/Sharpe/weights (right) - these
# are 4 different strategies' results side by side, not one strategy shown
# 4 different ways.
#
# Every pane is pinned with absolute positioning (not the library's default
# CSS float) for the same reason the single-timeframe view's MACD/ATR panes
# are: floats pack into whatever gap is free first rather than strictly
# under/beside a specific element, which breaks down fast once there are
# more than a couple of panes fighting for the same rows.

# 2x2 tile grid - one tile per timeframe - rather than 4 stacked rows, so
# each tile actually has room to be read. Tile (row, col): 5m top-left,
# 15m top-right, 30m bottom-left, 1h bottom-right. Each tile is itself split
# left/right: chart half (candlestick + MACD + ATR) in the left half of the
# window, summary half (equity + return/Sharpe/pie) in the right half - so
# the window is a 2x2 grid of chart-tiles beside a 2x2 grid of summary-tiles.
_TIMEFRAME_GRID = {"5m": (0, 0), "15m": (0, 1), "30m": (1, 0), "1h": (1, 1)}
_TIMEFRAME_ORDER = ("5m", "15m", "30m", "1h")
_ROW_H = 0.5
_COL_W = 0.25  # chart-half tile width (2 cols across the 0.5-wide chart half)
_MAIN_H = 0.30
_IND_H = 0.10  # main + macd + atr = 0.30 + 0.10 + 0.10 = _ROW_H
_SUMMARY_COL_W = 0.25  # summary-half tile width (2 cols across the other 0.5)
_SUMMARY_LEFT0 = 0.5
# Within a tile's summary half, equity chart (top) and the
# return/Sharpe/pie panel (bottom) are stacked - giving them the same box
# would just hide the panel behind the chart's canvas.
_EQUITY_H = 0.26
_SUMMARY_H = _ROW_H - _EQUITY_H


def _pin(pane: AbstractChart, top_frac: float, left_frac: float) -> None:
    """Absolute-position `pane` at (top_frac, left_frac); width/height stay
    whatever was passed to its constructor/create_subchart call (already
    correct, since those set both the CSS % and the actual canvas pixels)."""
    pane.run_script(
        f"""
        {pane.id}.wrapper.style.float = 'none'
        {pane.id}.wrapper.style.position = 'absolute'
        {pane.id}.wrapper.style.top = '{top_frac * 100}%'
        {pane.id}.wrapper.style.left = '{left_frac * 100}%'
        """
    )


def _resample_indicators_to_display(live_indicators: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Downsamples the live (recomputed every 5m tick) higher-TF indicator
    series to one row per higher-timeframe *display* candle - taking each
    window's last live value, which by the time a window's final 5m bar
    prints has converged to what a plain closed-candle computation would
    give anyway. This matches `resample_ohlcv`'s bar count/timestamps
    exactly, which is what lets the MACD/ATR panes use the library's native
    index-based `sync=True` against the resampled candlestick series -
    otherwise their bar counts differ (many 5m rows vs few higher-TF rows)
    and native sync misaligns.
    """
    starts = session_window_starts(live_indicators, rule)
    out = live_indicators.assign(_window=starts).groupby("_window", as_index=False, sort=True).last()
    out["time"] = _to_ns(out["_window"])
    return out[["time", "macd", "signal", "histogram", "atr"]]


def _combined_indicator_df(df_5m: pd.DataFrame) -> pd.DataFrame:
    """Uniform {time, macd, signal, histogram, atr} shape, computed fresh from native 5m bars."""
    macd_df = _macd_series(df_5m)
    atr_df = _atr_series(df_5m)
    out = macd_df.copy()
    out["atr"] = atr_df["atr"]
    return out


def _apply_indicators(macd_line, signal_line, macd_hist, atr_line, combined: pd.DataFrame) -> None:
    macd_line.set(combined[["time", "macd"]])
    signal_line.set(combined[["time", "signal"]])
    hist_df = combined[["time", "histogram"]].copy()
    hist_df["color"] = [TP_COLOR if v >= 0 else SL_COLOR for v in hist_df["histogram"]]
    macd_hist.set(hist_df)
    atr_line.set(combined[["time", "atr"]])


def _indicator_lookup_from_df(combined: pd.DataFrame) -> dict[int, dict[str, float]]:
    seconds = combined["time"].astype("int64") // 10**9
    lookup: dict[int, dict[str, float]] = {}
    for t, macd_v, signal_v, hist_v, atr_v in zip(
        seconds, combined["macd"], combined["signal"], combined["histogram"], combined["atr"]
    ):
        lookup[int(t)] = {"macd": macd_v, "signal": signal_v, "histogram": hist_v, "atr": atr_v}
    return lookup


def _set_value_lookup(chart: AbstractChart, seconds: "pd.Series[int]", values: "pd.Series[float]") -> None:
    """
    Stashes {unix_seconds: value} on `chart`, keyed as `.valueLookup` -
    each entry in `_wire_synced_crosshair`'s group reads its OWN value here
    for whatever time it's asked to show a crosshair at, rather than
    borrowing a value computed for a different chart's series/price scale.
    """
    lookup = {int(t): float(v) for t, v in zip(seconds, values) if pd.notna(v)}
    chart.run_script(f"{chart.id}.valueLookup = {json.dumps(lookup)}")


def _set_floor_map(chart: AbstractChart, src_seconds: "pd.Series[int]", dst_seconds: "pd.Series[int]") -> None:
    """
    Stashes {unix_seconds -> unix_seconds} on `chart` as `.floorMap`,
    mapping any timestamp on the shared 5m grid to the start of the bar on
    `chart`'s own (possibly coarser) grid that contains it - e.g. a 15m
    chart's floorMap sends 09:47 to 09:45, and 1h's sends it to 09:30.
    Charts already on the 5m grid (macd/atr panes, equity, the 5m tile's own
    main chart) don't need one - `_wire_synced_crosshair` falls back to
    treating the hovered time as already-aligned when `.floorMap` is unset.
    """
    mapping = {int(s): int(d) for s, d in zip(src_seconds, dst_seconds)}
    chart.run_script(f"{chart.id}.floorMap = {json.dumps(mapping)}")


def _sync_timescale_only(chart_a: AbstractChart, chart_b: AbstractChart) -> None:
    """
    Two-way visible-range mirror between `chart_a` and `chart_b` (e.g. a
    main chart and its own ATR subpane), WITHOUT the library's native
    `create_subchart(sync=True)` crosshair pairing.

    That native pairing (see `AbstractChart.win.create_subchart`'s JS-side
    `syncCharts`) runs its own independent crosshair-propagation system in
    parallel with `_wire_synced_crosshair` - toggling which of the pair's
    `subscribeCrosshairMove` listeners is active based on `mouseover`, and
    calling `setCrosshairPosition` using the hovered series' raw value/time
    with no floor-map translation. Layered on top of our own broadcast (which
    already handles this exact pair correctly, same as every other
    non-natively-paired combination), the two systems fight over the same
    parent/child pair - this was the confirmed cause of the crosshair going
    missing specifically between an ATR pane and its own main chart while
    every other pane combination worked. Wiring the timescale link by hand
    here keeps the zoom/pan lockstep (why `sync=True` was used) without its
    conflicting crosshair half.
    """
    for src, dst in ((chart_a, chart_b), (chart_b, chart_a)):
        src.run_script(
            f"""
            {src.id}.chart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                if (!range || {src.id}.rangeSyncing) return
                {src.id}.rangeSyncing = true
                {dst.id}.chart.timeScale().setVisibleLogicalRange(range)
                {src.id}.rangeSyncing = false
            }})
            """
        )


def _wire_synced_crosshair(
    entries: list[tuple[AbstractChart, str]],
    legend_divs: dict[str, tuple[str, str]] | None = None,
    native_legend_ids: set[str] | None = None,
) -> None:
    """
    `entries`: (chart, series_js_ref) pairs, where series_js_ref is the JS
    expression for the chart's own POPULATED series - a subchart's default
    candlestick series (`{chart.id}.series`) is unused/empty when the pane
    only hosts a custom line/histogram, so callers must pass that series
    object's own id instead (e.g. `f"{macd_line.id}.series"`).

    Hovering any entry broadcasts the hovered time to every other entry.
    Each destination first floors the incoming time through its own
    `.floorMap` (identity if unset) to land on a bar that actually exists
    on its series, then reads ITS OWN value there via `.valueLookup`. This
    is what lets a 15m hover land cleanly on the enclosing 1h bar (and vice
    versa) instead of a native index- or exact-time-based sync silently
    dropping the crosshair whenever no bar exists at the exact hovered
    instant (the failure mode of the library's built-in `sync=` option
    across panes with different bar density).

    `legend_divs`: optional `{chart.id: (div_attr, label)}` - for entries
    listed here, the broadcast also writes `label: dstValue` straight into
    that div, using the SAME `dstValue` already resolved via `.valueLookup`
    above, instead of leaving it to a separate `param.seriesData`-based
    listener (e.g. `_wire_single_line_legend`) to independently re-derive
    the value. That separate path requires an exact bar to exist at the
    landed time on that series; `dstValue` is already guaranteed non-undefined
    here (that's the branch condition), so writing it directly avoids a
    second, more fragile lookup for panes that need a text readout kept in
    sync with the same broadcast.

    `native_legend_ids`: chart ids that have the library's own native
    `legend(visible=True)` OHLCV/MA box. That box's own internal handler
    (`{chart.id}.legend.legendHandler`) normally only redraws from a real
    mouse event, using `param.seriesData.get(series)` - an exact-time
    lookup that silently comes up empty for a landed `dstTime` that isn't
    precisely a real bar's own timestamp on that pane (the OHLC row's
    failure mode observed here; the MA rows happened to still resolve).
    The library's OWN native cross-pane sync (`create_subchart(sync=True)`)
    sidesteps this exact problem by calling `legendHandler({time}, true)` -
    the `true` makes it convert time -> chart coordinate -> logical index
    and read the bar `byIndex` instead of by exact time match, which is
    robust to the same floor/broadcast landing this whole module already
    relies on. Calling that same entry point here (found in the bundled
    lightweight-charts JS) reuses the native box's own rendering untouched -
    same colors, same layout - it's just driven by our broadcast instead of
    a mouse event.
    """
    legend_divs = legend_divs or {}
    native_legend_ids = native_legend_ids or set()
    for chart, _ in entries:
        others = [(c, s) for c, s in entries if c.id != chart.id]
        clear_others_js = "\n".join(
            f"""
            {c.id}.chart.clearCrosshairPosition()
            {f"{c.id}.{legend_divs[c.id][0]}.innerText = '{legend_divs[c.id][1]}'" if c.id in legend_divs else ''}
            {f"{c.id}.legend.legendHandler({{}}, true)" if c.id in native_legend_ids else ''}
            """
            for c, _ in others
        )
        update_others_js = "\n".join(
            f"""
            {{
                const dstTime = {c.id}.floorMap ? {c.id}.floorMap[time] : time
                const dstValue = dstTime !== undefined && {c.id}.valueLookup ? {c.id}.valueLookup[dstTime] : undefined
                if (dstValue !== undefined) {{
                    {c.id}.chart.setCrosshairPosition(dstValue, dstTime, {s})
                    {f"{c.id}.{legend_divs[c.id][0]}.innerText = '{legend_divs[c.id][1]}: ' + dstValue.toFixed(3)" if c.id in legend_divs else ''}
                    {f"{c.id}.legend.legendHandler({{time: dstTime}}, true)" if c.id in native_legend_ids else ''}
                }} else {{
                    {c.id}.chart.clearCrosshairPosition()
                    {f"{c.id}.{legend_divs[c.id][0]}.innerText = '{legend_divs[c.id][1]}'" if c.id in legend_divs else ''}
                    {f"{c.id}.legend.legendHandler({{}}, true)" if c.id in native_legend_ids else ''}
                }}
            }}
            """
            for c, s in others
        )
        chart.run_script(
            f"""
            {chart.id}.chart.subscribeCrosshairMove(param => {{
                if ({chart.id}.crosshairBroadcasting) return
                {chart.id}.crosshairBroadcasting = true
                if (!param.time) {{
                    {clear_others_js}
                }} else {{
                    const time = param.time
                    {update_others_js}
                }}
                {chart.id}.crosshairBroadcasting = false
            }})
            """
        )


def _build_row_summary(
    main_chart: AbstractChart,
    equity_chart: AbstractChart,
    top_frac: float,
    left_frac: float,
    width_frac: float,
    height_frac: float,
    title: str,
    return_label: str,
    sharpe_stats: dict[str, float],
    portfolio_return: float,
    weights_history: pd.DataFrame,
    symbols: list[str],
) -> None:
    """
    Per-row summary: title, portfolio return/Sharpe (static), and a weight
    pie/legend that updates on hover - mirroring `_wire_weight_hover` from
    the single-timeframe view, but scoped to this row: div refs are stashed
    on `equity_chart.id` (unique per row) instead of the shared root, so
    hovering one row's chart only moves that row's pie.
    """
    var = equity_chart.id
    colors = [PIE_COLORS[i % len(PIE_COLORS)] for i in range(len(symbols))]

    legend_rows_js = "\n".join(
        f"""
        {{
            const row = document.createElement('div')
            row.style.display = 'flex'
            row.style.alignItems = 'center'
            row.style.gap = '4px'
            const dot = document.createElement('span')
            dot.style.width = '8px'
            dot.style.height = '8px'
            dot.style.minWidth = '8px'
            dot.style.borderRadius = '50%'
            dot.style.background = '{color}'
            const label = document.createElement('span')
            label.innerText = '{symbol}: -'
            row.appendChild(dot)
            row.appendChild(label)
            {var}.pieLegendDiv.appendChild(row)
            {var}.pieLegendLabels.push(label)
        }}
        """
        for symbol, color in zip(symbols, colors)
    )

    sharpe_rows_html = "".join(
        f"<div>{'Portfolio' if s == 'Portfolio' else s} Sharpe: {'n/a' if pd.isna(v) else f'{v:.2f}'}</div>"
        for s, v in sharpe_stats.items()
    )

    equity_chart.run_script(
        f"""
        {var}.summaryDiv = document.createElement('div')
        {var}.summaryDiv.style.position = 'absolute'
        {var}.summaryDiv.style.top = '{top_frac * 100}%'
        {var}.summaryDiv.style.left = '{left_frac * 100}%'
        {var}.summaryDiv.style.width = '{width_frac * 100}%'
        {var}.summaryDiv.style.height = '{height_frac * 100}%'
        {var}.summaryDiv.style.boxSizing = 'border-box'
        {var}.summaryDiv.style.overflow = 'visible'
        {var}.summaryDiv.style.padding = '6px 8px'
        {var}.summaryDiv.style.backgroundColor = '#121417'
        {var}.summaryDiv.style.color = '#d1d4dc'
        {var}.summaryDiv.style.fontFamily = '{UI_FONT}'
        {var}.summaryDiv.style.fontSize = '13px'
        {var}.summaryDiv.style.lineHeight = '1.5'
        {var}.summaryDiv.style.display = 'flex'
        {var}.summaryDiv.style.flexDirection = 'column'
        {var}.summaryDiv.style.borderLeft = '1px solid rgb(70,70,70)'

        {var}.summaryTitle = document.createElement('div')
        {var}.summaryTitle.innerText = '{title}'
        {var}.summaryTitle.style.fontSize = '16px'
        {var}.summaryTitle.style.marginBottom = '4px'
        {var}.summaryDiv.appendChild({var}.summaryTitle)

        {var}.summaryReturn = document.createElement('div')
        {var}.summaryReturn.innerText = '{return_label}: {portfolio_return:.2%}'
        {var}.summaryDiv.appendChild({var}.summaryReturn)

        // Grid, not flex - two fixed columns side by side (Sharpe stats on
        // the left, pie+legend on the right) that never wrap onto separate
        // rows regardless of how narrow the tile is, unlike a flex row
        // (which can end up rendering as stacked if the flex-basis/width
        // math for its children doesn't resolve the way expected).
        {var}.summaryDetails = document.createElement('div')
        {var}.summaryDetails.style.display = 'grid'
        {var}.summaryDetails.style.gridTemplateColumns = '1fr auto'
        {var}.summaryDetails.style.columnGap = '14px'
        {var}.summaryDetails.style.alignItems = 'start'
        {var}.summaryDetails.style.marginTop = '4px'
        {var}.summaryDiv.appendChild({var}.summaryDetails)

        {var}.summarySharpe = document.createElement('div')
        {var}.summarySharpe.innerHTML = `{sharpe_rows_html}`
        {var}.summaryDetails.appendChild({var}.summarySharpe)

        {var}.summaryPieRow = document.createElement('div')
        {var}.summaryPieRow.style.display = 'flex'
        {var}.summaryPieRow.style.alignItems = 'center'
        {var}.summaryPieRow.style.gap = '8px'
        {var}.summaryDetails.appendChild({var}.summaryPieRow)

        {var}.pieDiv = document.createElement('div')
        {var}.pieDiv.style.width = '64px'
        {var}.pieDiv.style.height = '64px'
        {var}.pieDiv.style.minWidth = '64px'
        {var}.pieDiv.style.borderRadius = '50%'
        {var}.pieDiv.style.background = '#333'
        {var}.summaryPieRow.appendChild({var}.pieDiv)

        {var}.pieLegendDiv = document.createElement('div')
        {var}.pieLegendDiv.style.fontSize = '12px'
        {var}.pieLegendDiv.style.display = 'grid'
        {var}.pieLegendDiv.style.gridTemplateColumns = 'repeat(2, auto)'
        {var}.pieLegendDiv.style.columnGap = '10px'
        {var}.pieLegendLabels = []
        {legend_rows_js}
        {var}.summaryPieRow.appendChild({var}.pieLegendDiv)

        {var}.summaryDiv.appendChild({var}.summaryPieRow)
        window.containerDiv.appendChild({var}.summaryDiv)
        """
    )
    _wire_weight_hover(equity_chart, main_chart, weights_history, symbols, colors)


def show_multi_timeframe_backtest(symbols: list[str], variants: dict[str, dict]) -> None:
    """
    `variants[tf]` (tf in "5m"/"15m"/"30m"/"1h") must have keys: candles
    (dict[symbol -> OHLCV df, already resampled to that timeframe for
    display]), rule (None for "5m", else the pandas resample rule string),
    trades, weight_adjustments, equity_curve, weights_history, sharpe_stats,
    portfolio_return, live_indicators (dict[symbol -> DataFrame] for
    non-"5m" timeframes, None for "5m" - see `src/run_phaseD.py`).
    """
    # `root` itself hosts no series or content - it's purely the top-level
    # window/topbar container that every tile's own chart is a subchart of.
    # Using root itself as the 5m tile's main chart (as an earlier version
    # of this function did, to avoid one "wasted" default candlestick
    # series) made 5m's macd/atr `sync=True` mouseover-driven pane-swap
    # listener attach to root.wrapper - which, being the DOM ancestor every
    # other tile's subchart also lives under, received a bubbled mouseover
    # from literally anywhere on the page, not just the 5m tile. That
    # constant cross-tile "active pane" reassignment was the likely cause
    # of 5m's zoom/pan feeling locked. Giving every tile (5m included) its
    # own real subchart keeps each tile's sync wiring scoped to just its
    # own wrapper.
    root = Chart(
        title="Multi-Timeframe Strategy Comparison", toolbox=False,
        inner_width=_COL_W, inner_height=_MAIN_H,
        width=1800, height=1100, maximize=True,
    )
    _pin(root, 0.0, 0.0)

    rows: dict[str, dict] = {}
    trades_by_symbol_tf: dict[str, dict] = {}
    wa_by_symbol_tf: dict[str, dict] = {}

    # Floor maps only depend on timestamps (all symbols share the same
    # aligned 5m index - `align_to_common_index` in run_phaseD.py), so
    # compute them once here rather than per symbol.
    ref_5m = variants["5m"]["candles"][symbols[0]]
    ref_seconds = _to_ns(ref_5m["timestamp"]).astype("int64") // 10**9
    floor_maps: dict[str, "pd.Series[int]"] = {}
    for tf in _TIMEFRAME_ORDER:
        rule = variants[tf]["rule"]
        if rule is None:
            continue
        starts = session_window_starts(ref_5m, rule)
        floor_maps[tf] = _to_ns(starts).astype("int64") // 10**9

    for tf in _TIMEFRAME_ORDER:
        v = variants[tf]
        grid_row, grid_col = _TIMEFRAME_GRID[tf]
        main_top = grid_row * _ROW_H
        chart_left = grid_col * _COL_W
        summary_left = _SUMMARY_LEFT0 + grid_col * _SUMMARY_COL_W
        macd_top = main_top + _MAIN_H
        atr_top = macd_top + _IND_H

        main_chart = root.create_subchart(position="left", width=_COL_W, height=_MAIN_H)
        main_chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
        main_chart.crosshair(mode="magnet")
        main_chart.legend(visible=True, font_size=12, font_family=UI_FONT)
        main_chart.price_line(line_visible=False)
        main_chart.run_script(f"{main_chart.id}.volumeSeries.applyOptions({{priceLineVisible: false}})")
        _pin(main_chart, main_top, chart_left)
        dot_series = _create_trade_dot_series(main_chart)
        if tf in floor_maps:
            _set_floor_map(main_chart, ref_seconds, floor_maps[tf])

        # Every row's MACD/ATR panes are fed a *resampled* indicator series
        # (`_resample_indicators_to_display`) with exactly one row per
        # displayed candle - same bar count as the main chart's candlestick
        # series - so the library's native index-based sync (zoom + pan +
        # crosshair) lines up exactly, same as the 5m row always had.
        macd_chart = main_chart.create_subchart(
            position="left", width=_COL_W, height=_IND_H, sync=True, sync_crosshairs_only=False
        )
        macd_chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
        macd_chart.crosshair(mode="magnet", horz_visible=False)
        macd_line = macd_chart.create_line("macd", color=OPEN_COLOR, price_line=False)
        signal_line = macd_chart.create_line("signal", color=WA_COLOR, price_line=False)
        macd_hist = macd_chart.create_histogram(
            "histogram", color=TP_COLOR, scale_margin_top=0.8, scale_margin_bottom=0.0, price_line=False
        )
        _create_legend_div(macd_chart, "macdLegend")
        _pin(macd_chart, macd_top, chart_left)

        atr_chart = main_chart.create_subchart(
            position="left", width=_COL_W, height=_IND_H, sync=True, sync_crosshairs_only=False
        )
        atr_chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
        atr_chart.crosshair(mode="magnet")
        atr_line = atr_chart.create_line("atr", color=OPEN_COLOR, price_line=False)
        _create_legend_div(atr_chart, "atrLegend")
        _pin(atr_chart, atr_top, chart_left)

        # macd_chart/atr_chart now plot the *resampled* indicator series
        # (one row per displayed candle, see `_resample_indicators_to_display`)
        # rather than the raw 5m-grid live series, so - like main_chart -
        # they need the same floor map for `_wire_synced_crosshair` to land
        # a cross-tile hover on the right bar.
        if tf in floor_maps:
            _set_floor_map(macd_chart, ref_seconds, floor_maps[tf])
            _set_floor_map(atr_chart, ref_seconds, floor_maps[tf])

        # Indicator panes stack in the order they're opened, right below the
        # main chart - mirrors the single-timeframe view's toggle behavior.
        # Both start open (visible) here, unlike the single-timeframe view.
        indicator_charts = {"macd": macd_chart, "atr": atr_chart}
        pane_tops = (macd_top, atr_top)
        open_order: list[str] = ["macd", "atr"]

        def _reposition_panes(indicator_charts=indicator_charts, pane_tops=pane_tops, open_order=open_order) -> None:
            for i, key in enumerate(open_order):
                pane = indicator_charts[key]
                pane.run_script(f"{pane.id}.wrapper.style.top = '{pane_tops[i] * 100}%'")

        def _make_toggle(key: str, widget_name: str, indicator_charts=indicator_charts, open_order=open_order):
            sub_chart = indicator_charts[key]

            def on_toggle(c):
                visible = c.topbar[widget_name].value
                if key in open_order:
                    open_order.remove(key)
                if visible:
                    open_order.append(key)
                sub_chart.run_script(
                    f"{sub_chart.id}.wrapper.style.display = '{{}}'".format("block" if visible else "none")
                )
                _reposition_panes()

            return on_toggle

        # Scoped to this row's own main_chart.topbar (each AbstractChart has
        # its own TopBar, rendered into that chart's own wrapper) rather than
        # root.topbar - the shared root topbar only spans the top-left
        # tile's width, so cramming all 4 rows' buttons into it would wrap
        # and cover that one tile's candlesticks instead of sitting cleanly
        # in each row.
        main_chart.topbar.button("macd_toggle", "MACD", toggle=True, func=_make_toggle("macd", "macd_toggle"))
        main_chart.topbar.button("atr_toggle", "ATR", toggle=True, func=_make_toggle("atr", "atr_toggle"))

        # No native sync (see the macd/atr comment above for why) - crosshair
        # alignment is handled uniformly by `_wire_synced_crosshair` below.
        equity_chart = main_chart.create_subchart(
            position="right", width=_SUMMARY_COL_W, height=_EQUITY_H, sync=False
        )
        equity_chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
        equity_chart.crosshair(mode="magnet")
        equity_line = equity_chart.create_line("equity", color=OPEN_COLOR, price_line=False)
        bar_return_hist = equity_chart.create_histogram(
            "bar_return", color=TP_COLOR, scale_margin_top=0.7, scale_margin_bottom=0.0, price_line=False
        )
        _build_return_title(main_chart, equity_chart, title=f"Portfolio Return ({tf})")
        _pin(equity_chart, main_top, summary_left)

        equity_df = v["equity_curve"].rename("equity").reset_index()
        equity_df.columns = ["time", "equity"]
        equity_df["time"] = _to_ns(equity_df["time"])
        equity_line.set(equity_df)
        # Portfolio-level, not per-symbol - set once rather than in
        # render_symbol.
        _set_value_lookup(
            equity_chart, equity_df["time"].astype("int64") // 10**9, equity_df["equity"]
        )

        bar_return = v["equity_curve"].diff().fillna(0.0)
        bar_return_df = bar_return.rename("bar_return").reset_index()
        bar_return_df.columns = ["time", "bar_return"]
        bar_return_df["time"] = _to_ns(bar_return_df["time"])
        bar_return_df["color"] = [TP_COLOR if x >= 0 else SL_COLOR for x in bar_return_df["bar_return"]]
        bar_return_hist.set(bar_return_df)
        _wire_return_legend(equity_chart, bar_return_hist)
        equity_chart.run_script(f"{equity_chart.id}.chart.timeScale().fitContent()", run_last=True)

        _build_row_summary(
            main_chart, equity_chart, main_top + _EQUITY_H, summary_left, _SUMMARY_COL_W, _SUMMARY_H,
            title=f"{tf} variant",
            return_label="Return",
            sharpe_stats=v["sharpe_stats"],
            portfolio_return=v["portfolio_return"],
            weights_history=v["weights_history"],
            symbols=symbols,
        )

        trades_by_symbol_tf[tf] = _by_symbol(v["trades"])
        wa_by_symbol_tf[tf] = _by_symbol(v["weight_adjustments"])

        indicator_lookup: dict[int, dict[str, float]] = {}
        rows[tf] = dict(
            main_chart=main_chart, macd_chart=macd_chart, atr_chart=atr_chart,
            macd_line=macd_line, signal_line=signal_line, macd_hist=macd_hist, atr_line=atr_line,
            equity_chart=equity_chart, equity_line=equity_line,
            indicator_lookup=indicator_lookup, dot_series=dot_series,
        )
        _wire_indicator_hover(main_chart, macd_chart, atr_chart, indicator_lookup)

    crosshair_entries: list[tuple[AbstractChart, str]] = []
    for tf in _TIMEFRAME_ORDER:
        row = rows[tf]
        crosshair_entries.append((row["main_chart"], f"{row['main_chart'].id}.series"))
        crosshair_entries.append((row["macd_chart"], f"{row['macd_line'].id}.series"))
        crosshair_entries.append((row["atr_chart"], f"{row['atr_line'].id}.series"))
        crosshair_entries.append((row["equity_chart"], f"{row['equity_line'].id}.series"))
    _wire_synced_crosshair(crosshair_entries)

    def render_symbol(symbol: str) -> None:
        for tf in _TIMEFRAME_ORDER:
            v = variants[tf]
            row = rows[tf]
            df_display = v["candles"][symbol]
            live = v["live_indicators"][symbol] if v["live_indicators"] is not None else None

            row["main_chart"].set(_ohlc(df_display))
            _set_value_lookup(
                row["main_chart"],
                _to_ns(df_display["timestamp"]).astype("int64") // 10**9,
                df_display["close"],
            )

            if v["rule"] is None:
                marker_time = None
            else:
                # entry_bar_idx/exit_bar_idx index into the 5m-aligned series
                # (the engine always executes at 5m grain, regardless of
                # variant) - snap each to the higher-TF window it falls in
                # so the marker lands on a bar that actually exists in the
                # resampled display candles.
                assert live is not None
                starts = session_window_starts(live, v["rule"])
                marker_time = lambda i, _starts=starts: _starts.iloc[i]

            _set_markers(
                row["main_chart"], df_display,
                trades_by_symbol_tf[tf][symbol], wa_by_symbol_tf[tf][symbol],
                marker_time=marker_time,
            )
            _set_trade_dots(
                row["dot_series"], trades_by_symbol_tf[tf][symbol], marker_time, df_display["timestamp"]
            )
            row["main_chart"].fit()

            combined = (
                _combined_indicator_df(df_display) if live is None
                else _resample_indicators_to_display(live, v["rule"])
            )
            _apply_indicators(row["macd_line"], row["signal_line"], row["macd_hist"], row["atr_line"], combined)
            row["macd_chart"].fit()
            row["atr_chart"].fit()
            row["indicator_lookup"].clear()
            row["indicator_lookup"].update(_indicator_lookup_from_df(combined))

            combined_seconds = combined["time"].astype("int64") // 10**9
            _set_value_lookup(row["macd_chart"], combined_seconds, combined["macd"])
            _set_value_lookup(row["atr_chart"], combined_seconds, combined["atr"])

    def on_symbol_change(c):
        render_symbol(c.topbar["symbol"].value)

    root.topbar.switcher("symbol", tuple(symbols), default=symbols[0], func=on_symbol_change)
    render_symbol(symbols[0])

    root.show(block=True)


# --- 回调入场策略: 单ticker前端 --------------------------------------------
#
# 复用Part D的2x2周期格布局(5m/15m/30m/2h取代1h), 但这里只有*一个*策略
# (PullbackEntryEngine)的交易结果，四个格子只是把同一批交易画在四种不同的
# K线聚合粒度上，不是4个独立策略的对比 - 所以markers/equity/Sharpe在四个
# tile之间是共享的同一份数据，不像Part D那样每个tile各自独立。
# MACD整块不画(信号源已经完全是MA级联，不再是过滤条件，画出来没有意义)；
# MA5/20/50直接叠加在每个tile自己的主图K线上(不像MACD那样单独开一行)，
# ATR面板保留(仍然是止损/止盈内部计算依赖的东西)。

_PB_TIMEFRAME_GRID = {"5m": (0, 0), "15m": (0, 1), "30m": (1, 0), "2h": (1, 1)}
_PB_TIMEFRAME_ORDER = ("5m", "15m", "30m", "2h")
_PB_RESAMPLE_RULE = {"5m": None, "15m": "15min", "30m": "30min", "2h": "2h"}
_PB_ROW_H = 0.5
_PB_COL_W = 0.25
_PB_MAIN_H = 0.38  # 没有MACD这一行了，主图分到比Part D更多的高度
_PB_ATR_H = 0.12  # main + atr = 0.38 + 0.12 = _PB_ROW_H
_PB_SUMMARY_LEFT = 0.5

_MA_LINE_COLORS = {5: "#64b5f6", 20: "#ffb74d", 50: "#ba68c8"}
PROTECTIVE_SL_COLOR = "#7e57c2"
_PB_REASON_COLOR = {"SL": SL_COLOR, "TP": TP_COLOR, "PROTECTIVE_SL": PROTECTIVE_SL_COLOR}
_PB_DOT_COLORS = {"open": OPEN_COLOR, **_PB_REASON_COLOR}
# Marker text labels. PROTECTIVE_SL is abbreviated because every stop marker now
# carries a fill-quality suffix and "PROTECTIVE_SL(G) 50%" is unreadable sitting
# on a candle; the color already separates it (purple vs red).
_PB_REASON_LABEL = {"SL": "SL", "TP": "TP", "PROTECTIVE_SL": "PSL"}
# Admin-only overlays.
PULLBACK_COLOR = "#fdd835"  # 回调 confirmation marker
COOLDOWN_COLOR = "rgba(239, 83, 80, 0.13)"  # cooldown background band


def _wire_single_line_legend(chart: AbstractChart, line: "Line", label: str, div_attr: str) -> None:
    """
    Fills in a legend div (see `_create_legend_div`) with `label` plus the
    hovered value of `line`'s own series, read directly via this chart's own
    `param.seriesData` - `line` lives on `chart` itself so there's no
    cross-chart lookup limitation to route around (unlike `_wire_indicator_hover`,
    which needs a python roundtrip because MACD/ATR live on *different*
    panes than the one being hovered). Also used by `_wire_synced_crosshair`
    broadcasting a `setCrosshairPosition` onto this chart - that fires the
    same `subscribeCrosshairMove` event as a real hover, so the label updates
    when any *other* synced pane is hovered too, not just this one.
    """
    chart.run_script(f"{chart.id}.{div_attr}.innerText = '{label}'")
    chart.run_script(f"""
        {chart.id}.chart.subscribeCrosshairMove(param => {{
            const point = param.time && param.seriesData.get({line.id}.series)
            if (point) {{
                {chart.id}.{div_attr}.innerText = `{label}: ${{point.value.toFixed(3)}}`
            }} else {{
                {chart.id}.{div_attr}.innerText = '{label}'
            }}
        }})
    """)


def _ma_lines_df(df: pd.DataFrame, periods: tuple[int, ...] = (5, 20, 50)) -> dict[int, pd.DataFrame]:
    time_ns = _to_ns(df["timestamp"])
    return {p: pd.DataFrame({"time": time_ns, "value": df["close"].rolling(p).mean()}) for p in periods}


def _create_pullback_dot_series(main_chart: AbstractChart) -> dict[str, "Line"]:
    """Same pattern as `_create_trade_dot_series`, keyed by the pullback engine's own reason set."""
    series = {}
    for reason, color in _PB_DOT_COLORS.items():
        line = main_chart.create_line(reason, color=color, width=1, price_line=False, price_label=False)
        line.run_script(
            f"{line.id}.series.applyOptions({{lineVisible: false, pointMarkersVisible: true, "
            f"pointMarkersRadius: 3, lastValueVisible: false, crosshairMarkerVisible: false}})"
        )
        line.run_script(f"""
            try {{
                const item = {main_chart.id}.legend._lines.find((l) => l.series == {line.id}.series)
                if (item) {{
                    {main_chart.id}.legend._lines = {main_chart.id}.legend._lines.filter((l) => l != item)
                    if (item.row && item.row.parentNode) {{
                        item.row.parentNode.removeChild(item.row)
                    }}
                }}
            }} catch (e) {{}}
        """)
        series[reason] = line
    return series


def _set_pullback_markers(
    chart: Chart, display_df: pd.DataFrame, trades: list,
    marker_time: Callable[[int], pd.Timestamp] | None,
    pullback_points: list | None = None,
) -> None:
    """
    Entry/exit arrows, plus - when `pullback_points` is passed (admin mode) -
    a circle on every 回调 confirmation bar. Stop exits carry a fill-quality
    suffix: "(G)" means the bar gapped through the stop and the fill was taken
    at the bar's open rather than the stop line. No suffix = filled at the line.
    """
    chart.clear_markers()
    times = display_df["timestamp"]

    def time_at(i: int) -> pd.Timestamp:
        return times.iloc[i] if marker_time is None else marker_time(i)

    entry_time_at = exit_time_at = time_at

    markers = []
    seen_entries: set[int] = set()
    for t in trades:
        is_long = t.direction == "long"
        if t.entry_bar_idx not in seen_entries:
            seen_entries.add(t.entry_bar_idx)
            # `signal_type` ("open"/"reentry") isn't distinguished on the
            # marker text - `had_prior_batch` in the engine is currently a
            # one-way flag (true for every batch after the first, regardless
            # of direction or why the prior one closed), so it doesn't carry
            # the "post-take-profit 回补" meaning the field is meant to have
            # yet. Every entry just reads "Open Long/Short" until that's
            # fixed at the source.
            markers.append({
                "time": entry_time_at(t.entry_bar_idx),
                "position": "below" if is_long else "above",
                "shape": "arrow_up" if is_long else "arrow_down",
                "color": OPEN_COLOR,
                "text": f"Open {'Long' if is_long else 'Short'}",
            })
        suffix = "(G)" if getattr(t, "gapped", False) else ""
        markers.append({
            "time": exit_time_at(t.exit_bar_idx),
            "position": "above" if is_long else "below",
            "shape": "arrow_down" if is_long else "arrow_up",
            "color": _PB_REASON_COLOR[t.reason],
            "text": f"{_PB_REASON_LABEL[t.reason]}{suffix} {t.size_fraction:.0%}",
        })

    for p in pullback_points or []:
        # Circle, not an arrow, so a 回调 confirmation is never mistaken for an
        # entry/exit. It marks the bar the pullback was *confirmed* on - the
        # entry trigger, if it comes at all, is a later bar.
        markers.append({
            "time": time_at(p["bar_idx"]),
            "position": "inside",
            "shape": "circle",
            "color": PULLBACK_COLOR,
            "text": f"Pullback {'L' if p['direction'] == 'long' else 'S'}",
        })

    markers.sort(key=lambda m: m["time"])
    if markers:
        chart.marker_list(markers)


def _set_pullback_dots(
    dot_series: dict[str, "Line"], trades: list,
    marker_time: Callable[[int], pd.Timestamp] | None, times: "pd.Series[pd.Timestamp]",
) -> None:
    def time_at(i: int) -> pd.Timestamp:
        return times.iloc[i] if marker_time is None else marker_time(i)

    entry_time_at = exit_time_at = time_at

    rows_by_reason: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        rows_by_reason["open"].append({"time": entry_time_at(t.entry_bar_idx), "open": t.entry_price})
        rows_by_reason[t.reason].append({"time": exit_time_at(t.exit_bar_idx), t.reason: t.exit_price})

    for reason, line in dot_series.items():
        rows = rows_by_reason.get(reason, [])
        if not rows:
            line.set(None)
            continue
        df = pd.DataFrame(rows)
        df["time"] = _to_ns(pd.to_datetime(df["time"]))
        df["color"] = _PB_DOT_COLORS[reason]
        line.set(df)


def _cooldown_hist_df(
    df_5m: pd.DataFrame, spans: list, df_display: pd.DataFrame, rule: str | None,
) -> pd.DataFrame | None:
    """
    Full-height background bars on every display candle overlapping a cooldown
    stretch (the TradingView session-shading look), as ONE histogram series per
    tile rather than one `vertical_span` per stretch - a two-year run has
    hundreds of stretches and a series apiece would mean hundreds of JS series
    per tile.

    `spans` are inclusive (start, end) 5m bar indices. On a higher timeframe a
    candle is shaded if ANY 5m bar inside it was cooling, binned by the same
    `session_window_starts` the markers use. None = nothing to draw.
    """
    if not spans:
        return None
    cooling = pd.Series(False, index=range(len(df_5m)))
    for start, end in spans:
        cooling.iloc[start:end + 1] = True
    if rule is None:
        flags = cooling.to_numpy()
    else:
        # groupby on the tz-aware Series (not .to_numpy(), which would drop the
        # tz and then never match df_display's tz-aware timestamps).
        starts = session_window_starts(df_5m, rule)
        per_window = cooling.groupby(starts).max()
        flags = df_display["timestamp"].map(per_window).fillna(False).to_numpy()
    rows = pd.DataFrame({"time": _to_ns(df_display["timestamp"]), "cooldown": flags.astype(float)})
    rows = rows[rows["cooldown"] > 0]
    return rows.reset_index(drop=True) if len(rows) else None


def _build_admin_button(
    root: Chart, top: float, left_px: int, height: float, on_toggle: "Callable[[str], None]",
) -> None:
    """
    Admin toggle for the diagnostic overlays (回调 confirmations + cooldown
    bands). Built as a custom pinned element rather than `topbar.button`
    because this layout never creates a library topbar at all - the ticker
    picker is itself a custom widget (`_build_ticker_search`), so a topbar
    button would have nowhere visible to render. Anchored in px off that
    input's fixed 200px width so it sits beside it at any window size.
    """
    handler_name = f"admin_toggle_{root.id.rsplit('.', 1)[-1]}"
    root.win.handlers[handler_name] = on_toggle
    root.run_script(f"""
        const adminWrap = document.createElement('div')
        adminWrap.style.position = 'absolute'
        adminWrap.style.top = '{top * 100}%'
        adminWrap.style.left = '{left_px}px'
        adminWrap.style.height = '{height * 100}%'
        adminWrap.style.zIndex = '4002'
        adminWrap.style.display = 'flex'
        adminWrap.style.alignItems = 'center'
        adminWrap.style.fontFamily = '{UI_FONT}'

        const adminBtn = document.createElement('div')
        adminBtn.innerText = 'Admin'
        adminBtn.style.cursor = 'pointer'
        adminBtn.style.userSelect = 'none'
        adminBtn.style.padding = '5px 16px'
        adminBtn.style.borderRadius = '4px'
        adminBtn.style.fontSize = '13px'
        let adminOn = false
        const paintAdmin = () => {{
            adminBtn.style.background = adminOn ? '{OPEN_COLOR}' : '#1e222d'
            adminBtn.style.color = adminOn ? '#ffffff' : '#d1d4dc'
            adminBtn.style.border = '1px solid ' + (adminOn ? '{OPEN_COLOR}' : '#2a2e39')
        }}
        paintAdmin()
        adminBtn.addEventListener('click', () => {{
            adminOn = !adminOn
            paintAdmin()
            window.callbackFunction(`{handler_name}_~_${{adminOn ? '1' : '0'}}`)
        }})
        adminWrap.appendChild(adminBtn)
        window.containerDiv.appendChild(adminWrap)
    """)


def show_pullback_backtest(
    symbol: str,
    df_5m: pd.DataFrame,
    trades: list,
    equity_curve: pd.Series,
    sharpe: float,
    total_return: float,
) -> None:
    """
    `df_5m`: raw 5m OHLCV for `symbol` (15m/30m/2h display candles are
    resampled from it here). `trades`: the single shared list of
    `engine.pullback_backtest.Trade` produced by one `PullbackEntryEngine`
    run - entry_bar_idx/exit_bar_idx index into `df_5m` (execution is always
    5m-grain, same convention as Part D), and are snapped onto the
    containing higher-timeframe candle for the 15m/30m/2h tiles.
    """
    root = Chart(
        title=f"Pullback Strategy - {symbol}", toolbox=False,
        inner_width=_PB_COL_W, inner_height=_PB_MAIN_H,
        width=1800, height=1100, maximize=True,
    )
    _pin(root, 0.0, 0.0)

    display_candles = {
        tf: (df_5m if rule is None else resample_ohlcv(df_5m, rule))
        for tf, rule in _PB_RESAMPLE_RULE.items()
    }

    ref_seconds = _to_ns(df_5m["timestamp"]).astype("int64") // 10**9
    floor_maps: dict[str, "pd.Series[int]"] = {}
    for tf, rule in _PB_RESAMPLE_RULE.items():
        if rule is None:
            continue
        starts = session_window_starts(df_5m, rule)
        floor_maps[tf] = _to_ns(starts).astype("int64") // 10**9

    crosshair_entries: list[tuple[AbstractChart, str]] = []
    atr_legend_targets: dict[str, tuple[str, str]] = {}
    native_legend_ids: set[str] = set()
    equity_chart_ref: AbstractChart | None = None

    for tf in _PB_TIMEFRAME_ORDER:
        grid_row, grid_col = _PB_TIMEFRAME_GRID[tf]
        main_top = grid_row * _PB_ROW_H
        chart_left = grid_col * _PB_COL_W
        atr_top = main_top + _PB_MAIN_H

        main_chart = root.create_subchart(position="left", width=_PB_COL_W, height=_PB_MAIN_H)
        main_chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
        main_chart.crosshair(mode="magnet")
        main_chart.legend(visible=True, font_size=12, font_family=UI_FONT)
        native_legend_ids.add(main_chart.id)
        main_chart.price_line(line_visible=False)
        main_chart.run_script(f"{main_chart.id}.volumeSeries.applyOptions({{priceLineVisible: false}})")
        _pin(main_chart, main_top, chart_left)

        df_display = display_candles[tf]
        main_chart.set(_ohlc(df_display))
        ma_lines = {}
        for period, color in _MA_LINE_COLORS.items():
            line = main_chart.create_line(f"MA{period}", color=color, width=1, price_line=False)
            line.set(_ma_lines_df(df_display, (period,))[period].rename(columns={"value": f"MA{period}"}))
            ma_lines[period] = line

        dot_series = _create_pullback_dot_series(main_chart)
        if tf in floor_maps:
            _set_floor_map(main_chart, ref_seconds, floor_maps[tf])
            # Snap to whichever higher-timeframe candle actually contains
            # the 5m bar's own timestamp, matching what's drawn on the 5m
            # tile itself.
            starts = session_window_starts(df_5m, _PB_RESAMPLE_RULE[tf])
            marker_time = lambda i, _starts=starts: _starts.iloc[i]
        else:
            marker_time = None

        _set_pullback_markers(main_chart, df_display, trades, marker_time)
        _set_pullback_dots(dot_series, trades, marker_time, df_display["timestamp"])
        main_chart.fit()

        atr_chart = main_chart.create_subchart(
            position="left", width=_PB_COL_W, height=_PB_ATR_H, sync=False
        )
        atr_chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
        atr_chart.crosshair(mode="magnet")
        atr_line = atr_chart.create_line("atr", color=OPEN_COLOR, price_line=False)
        _create_legend_div(atr_chart, "atrLegend")
        _pin(atr_chart, atr_top, chart_left)
        _sync_timescale_only(main_chart, atr_chart)

        atr_df = _atr_series(df_display)
        atr_line.set(atr_df)
        _wire_single_line_legend(atr_chart, atr_line, "ATR", "atrLegend")
        atr_legend_targets[atr_chart.id] = ("atrLegend", "ATR")
        if tf in floor_maps:
            _set_floor_map(atr_chart, ref_seconds, floor_maps[tf])
        atr_chart.fit()

        crosshair_entries.append((main_chart, f"{main_chart.id}.series"))
        crosshair_entries.append((atr_chart, f"{atr_line.id}.series"))
        _set_value_lookup(
            main_chart, _to_ns(df_display["timestamp"]).astype("int64") // 10**9, df_display["close"]
        )
        atr_seconds = atr_df["time"].astype("int64") // 10**9
        _set_value_lookup(atr_chart, atr_seconds, atr_df["atr"])

    # 右半边: 权益曲线(上) + 统计表(下) - 单ticker没有多品种权重饼图，直接
    # 省掉(Week 6接入多品种组合层之后再补)。
    equity_chart = root.create_subchart(
        position="right", width=_PB_COL_W * 2, height=_PB_ROW_H, sync=False
    )
    equity_chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
    equity_chart.crosshair(mode="magnet")
    equity_line = equity_chart.create_line("equity", color=OPEN_COLOR, price_line=False)
    bar_return_hist = equity_chart.create_histogram(
        "bar_return", color=TP_COLOR, scale_margin_top=0.7, scale_margin_bottom=0.0, price_line=False
    )
    _build_return_title(root, equity_chart, title=f"{symbol} Pullback Strategy Return")
    _pin(equity_chart, 0.0, _PB_SUMMARY_LEFT)

    equity_df = equity_curve.rename("equity").reset_index()
    equity_df.columns = ["time", "equity"]
    equity_df["time"] = _to_ns(equity_df["time"])
    equity_line.set(equity_df)
    _set_value_lookup(equity_chart, equity_df["time"].astype("int64") // 10**9, equity_df["equity"])

    bar_return = equity_curve.diff().fillna(0.0)
    bar_return_df = bar_return.rename("bar_return").reset_index()
    bar_return_df.columns = ["time", "bar_return"]
    bar_return_df["time"] = _to_ns(bar_return_df["time"])
    bar_return_df["color"] = [TP_COLOR if x >= 0 else SL_COLOR for x in bar_return_df["bar_return"]]
    bar_return_hist.set(bar_return_df)
    _wire_return_legend(equity_chart, bar_return_hist)
    # No `run_last=True` here (unlike show_multi_timeframe_backtest's equity
    # pane): that chart's equity_chart is sync=True, which queues a
    # library-internal visible-range copy from the main chart as its own
    # run_last script, so fitContent had to be run_last too just to win that
    # race. This equity_chart is sync=False - no such race exists, and
    # queuing this run_last would instead put it dead last in the *entire*
    # page's script batch (after all 4 grid tiles + table + crosshair
    # wiring), so a single JS exception anywhere earlier in that batch would
    # silently drop it, leaving the pane on the library's default small
    # initial bar window instead of the full curve. Running it immediately,
    # right after the data it depends on is set, avoids that.
    equity_chart.run_script(f"{equity_chart.id}.chart.timeScale().fitContent()")
    crosshair_entries.append((equity_chart, f"{equity_line.id}.series"))

    table = root.create_table(
        width=_PB_COL_W * 2, height=_PB_ROW_H, headings=("Metric", "Value"), widths=(0.6, 0.4),
        position="right", func=lambda _row: None,
    )
    table.new_row("Total Return", f"{total_return:.2%}")
    table.new_row("Sharpe", "n/a" if pd.isna(sharpe) else f"{sharpe:.2f}")
    n_batches = len({t.entry_bar_idx for t in trades})
    wins = sum(1 for i in {t.entry_bar_idx for t in trades} if
               sum(t.pnl_pct * t.size_fraction for t in trades if t.entry_bar_idx == i) > 0)
    table.new_row("Batches", str(n_batches))
    table.new_row("Win rate", f"{wins / n_batches:.1%}" if n_batches else "n/a")
    table.run_script(
        f"""
        {table.id}._div.style.fontSize = '15px'
        const scrollArea = {table.id}._div.querySelector('div')
        scrollArea.style.flex = '1 1 auto'
        scrollArea.style.minHeight = '0'
        """
    )
    # `_pin` assumes `.wrapper` (chart panes) - a Table uses `._div` instead,
    # so calling `_pin(table, ...)` throws (`undefined.wrapper.style`), which
    # silently kills every run_script queued after it in the same batch -
    # including `_wire_synced_crosshair` below. Pin the table's own div
    # directly instead.
    table.run_script(
        f"""
        {table.id}._div.style.float = 'none'
        {table.id}._div.style.position = 'absolute'
        {table.id}._div.style.top = '{_PB_ROW_H * 100}%'
        {table.id}._div.style.left = '{_PB_SUMMARY_LEFT * 100}%'
        """
    )

    _wire_synced_crosshair(
        crosshair_entries, legend_divs=atr_legend_targets, native_legend_ids=native_legend_ids
    )

    root.show(block=True)


_PF_TOPBAR_H = 0.06


def _build_ticker_search(
    root: Chart, symbols: list[str], default: str,
    top: float, left: float, width: float, height: float,
    on_select: "Callable[[str], None]",
) -> None:
    """
    Search-as-you-type ticker picker for the portfolio view's left-side
    ticker switch. The library's own `topbar.menu` is a fixed button list
    with no text filtering and no way to update its options after creation,
    so this is a small custom text input + filtered dropdown instead - scales
    to more than 9 tickers later without becoming a wall of buttons.

    Selecting an option calls `on_select(symbol)` in Python via the same
    `window.callbackFunction` handler-registration pattern used elsewhere in
    this module (e.g. `_wire_indicator_hover`).
    """
    handler_name = f"ticker_search_{root.id.rsplit('.', 1)[-1]}"
    root.win.handlers[handler_name] = on_select
    options_js = "\n".join(
        f"""
        {{
            const opt = document.createElement('div')
            opt.innerText = '{s}'
            opt.style.padding = '4px 8px'
            opt.style.cursor = 'pointer'
            opt.addEventListener('mouseenter', () => opt.style.background = '#2a2e39')
            opt.addEventListener('mouseleave', () => opt.style.background = '')
            // mousedown (not click) + preventDefault so the input's own
            // blur handler doesn't hide the list before the selection
            // registers - blur fires on mousedown, before click would.
            opt.addEventListener('mousedown', (e) => {{
                e.preventDefault()
                input.value = '{s}'
                list.style.display = 'none'
                window.callbackFunction(`{handler_name}_~_{s}`)
            }})
            list.appendChild(opt)
        }}
        """
        for s in symbols
    )
    root.run_script(f"""
        const searchBox = document.createElement('div')
        searchBox.style.position = 'absolute'
        searchBox.style.top = '{top * 100}%'
        searchBox.style.left = '{left * 100}%'
        searchBox.style.width = '{width * 100}%'
        searchBox.style.height = '{height * 100}%'
        searchBox.style.zIndex = '4000'
        searchBox.style.boxSizing = 'border-box'
        searchBox.style.padding = '6px 8px'
        searchBox.style.fontFamily = '{UI_FONT}'

        const input = document.createElement('input')
        input.value = '{default}'
        input.placeholder = 'Search ticker...'
        input.style.width = '200px'
        input.style.padding = '4px 8px'
        input.style.background = '#1e222d'
        input.style.color = '#d1d4dc'
        input.style.border = '1px solid #2a2e39'
        input.style.borderRadius = '4px'
        input.style.fontFamily = '{UI_FONT}'
        input.style.fontSize = '13px'
        searchBox.appendChild(input)

        const list = document.createElement('div')
        list.style.position = 'absolute'
        list.style.top = '100%'
        list.style.left = '8px'
        list.style.width = '200px'
        list.style.maxHeight = '220px'
        list.style.overflowY = 'auto'
        list.style.background = '#1e222d'
        list.style.border = '1px solid #2a2e39'
        list.style.borderRadius = '4px'
        list.style.color = '#d1d4dc'
        list.style.fontSize = '13px'
        list.style.display = 'none'
        list.style.zIndex = '4001'
        searchBox.appendChild(list)

        {options_js}

        input.addEventListener('focus', () => {{
            for (const opt of list.children) opt.style.display = ''
            list.style.display = 'block'
        }})
        input.addEventListener('input', () => {{
            const q = input.value.toLowerCase()
            for (const opt of list.children) {{
                opt.style.display = opt.innerText.toLowerCase().includes(q) ? '' : 'none'
            }}
            list.style.display = 'block'
        }})
        input.addEventListener('blur', () => {{
            list.style.display = 'none'
        }})

        window.containerDiv.appendChild(searchBox)
    """)


def _build_weight_pie_pinned(
    chart: Chart, symbols: list[str], top: float, left: float, width: float, height: float,
) -> list[str]:
    """
    Same conic-gradient pie + color-swatch legend as `_build_weight_pie`,
    pinned at an explicit position instead of floated - this whole layout is
    `_pin`-based throughout (like `show_pullback_backtest`), not float-based
    like `show_portfolio_backtest`, so mixing in a floated element here would
    make its size depend on unrelated DOM order instead of these fractions.
    """
    colors = [PIE_COLORS[i % len(PIE_COLORS)] for i in range(len(symbols))]
    var = chart.id
    legend_rows_js = "\n".join(
        f"""
        {{
            const row = document.createElement('div');
            row.style.display = 'flex';
            row.style.alignItems = 'center';
            row.style.gap = '6px';
            row.style.margin = '2px 0';
            const dot = document.createElement('span');
            dot.style.width = '10px';
            dot.style.height = '10px';
            dot.style.minWidth = '10px';
            dot.style.borderRadius = '50%';
            dot.style.background = '{color}';
            const label = document.createElement('span');
            label.innerText = '{symbol}: -';
            row.appendChild(dot);
            row.appendChild(label);
            {var}.pieLegendDiv.appendChild(row);
            {var}.pieLegendLabels.push(label);
        }}
        """
        for symbol, color in zip(symbols, colors)
    )
    chart.run_script(
        f"""
        {var}.pieContainer = document.createElement('div')
        {var}.pieContainer.style.position = 'absolute'
        {var}.pieContainer.style.top = '{top * 100}%'
        {var}.pieContainer.style.left = '{left * 100}%'
        {var}.pieContainer.style.width = '{width * 100}%'
        {var}.pieContainer.style.height = '{height * 100}%'
        {var}.pieContainer.style.boxSizing = 'border-box'
        {var}.pieContainer.style.display = 'flex'
        {var}.pieContainer.style.flexDirection = 'column'
        {var}.pieContainer.style.alignItems = 'center'
        {var}.pieContainer.style.justifyContent = 'center'
        {var}.pieContainer.style.overflowY = 'auto'
        {var}.pieContainer.style.backgroundColor = '#121417'
        {var}.pieContainer.style.color = '#d1d4dc'
        {var}.pieContainer.style.fontFamily = '{UI_FONT}'
        {var}.pieContainer.style.fontSize = '14px'
        {var}.pieContainer.style.padding = '8px'

        {var}.pieTitle = document.createElement('div')
        {var}.pieTitle.innerText = 'Weight'
        {var}.pieTitle.style.fontSize = '16px'
        {var}.pieTitle.style.marginBottom = '8px'
        {var}.pieContainer.appendChild({var}.pieTitle)

        {var}.pieDiv = document.createElement('div')
        {var}.pieDiv.style.width = '160px'
        {var}.pieDiv.style.height = '160px'
        {var}.pieDiv.style.minHeight = '160px'
        {var}.pieDiv.style.borderRadius = '50%'
        {var}.pieDiv.style.background = '#333'
        {var}.pieContainer.appendChild({var}.pieDiv)

        {var}.pieLegendDiv = document.createElement('div')
        {var}.pieLegendDiv.style.marginTop = '8px'
        {var}.pieContainer.appendChild({var}.pieLegendDiv)
        {var}.pieLegendLabels = []

        {legend_rows_js}

        window.containerDiv.appendChild({var}.pieContainer)
        """
    )
    return colors


def _fmt_pct(v: float, signed: bool = False) -> str:
    if pd.isna(v):
        return "n/a"
    return f"{v:+.1%}" if signed else f"{v:.1%}"


def _fmt_ratio(v: float) -> str:
    return "n/a" if pd.isna(v) else f"{v:.2f}"


def _build_portfolio_stats_tables(
    root: Chart, result: "PortfolioBacktestResult",
    top: float, left: float, width: float, height: float,
) -> None:
    """
    Three "pages" toggled by a row of buttons above the table (not a native
    tab widget - a plain dropdown/tab metaphor was judged less obvious for
    the end user than explicit labeled buttons). Page 1 and 3 each reuse the
    same pinned-Portfolio-row-plus-scrollable-tickers layout (the library's
    Table widget has no sticky-first-row option, so a frozen top row is a
    second, separate, non-scrolling Table stacked above the scrolling one) -
    page 2 has no meaningful "Portfolio" aggregate row (long/short is a
    per-ticker breakdown), so it's just one scrollable table.
    """
    button_h = height * 0.07
    body_top = top + button_h
    body_height = height - button_h
    portfolio_h = body_height * 0.16
    tickers_h = body_height - portfolio_h

    # ---- page 1: Return & Sharpe ----
    headings1 = ("Ticker", "Return", "Sharpe", "Win rate", "Batches")
    widths1 = (0.28, 0.22, 0.18, 0.18, 0.14)

    p1_portfolio = root.create_table(
        width=width, height=portfolio_h, headings=headings1, widths=widths1,
        position="right", func=lambda _row: None,
    )
    p1_portfolio.new_row(
        "Portfolio", f"{result.portfolio_return:.2%}",
        "n/a" if pd.isna(result.portfolio_sharpe) else f"{result.portfolio_sharpe:.2f}", "-", "-",
    )
    p1_portfolio.run_script(
        f"""
        {p1_portfolio.id}._div.style.fontSize = '14px'
        {p1_portfolio.id}._div.style.fontWeight = 'bold'
        {p1_portfolio.id}._div.style.float = 'none'
        {p1_portfolio.id}._div.style.position = 'absolute'
        {p1_portfolio.id}._div.style.top = '{body_top * 100}%'
        {p1_portfolio.id}._div.style.left = '{left * 100}%'
        """
    )

    p1_tickers = root.create_table(
        width=width, height=tickers_h, headings=headings1, widths=widths1,
        position="right", func=lambda _row: None,
    )
    for symbol in sorted(result.per_ticker):
        r = result.per_ticker[symbol]
        p1_tickers.new_row(
            symbol, f"{r.total_return:.2%}",
            "n/a" if pd.isna(r.sharpe) else f"{r.sharpe:.2f}",
            _fmt_pct(r.win_rate), str(r.n_batches),
        )
    p1_tickers.run_script(
        f"""
        {p1_tickers.id}._div.style.fontSize = '13px'
        const scrollArea1 = {p1_tickers.id}._div.querySelector('div')
        scrollArea1.style.flex = '1 1 auto'
        scrollArea1.style.minHeight = '0'
        {p1_tickers.id}._div.style.float = 'none'
        {p1_tickers.id}._div.style.position = 'absolute'
        {p1_tickers.id}._div.style.top = '{(body_top + portfolio_h) * 100}%'
        {p1_tickers.id}._div.style.left = '{left * 100}%'
        """
    )

    # ---- page 2: Long/Short breakdown ----
    headings2 = ("Ticker", "Win rate (L/S)", "Payoff (L/S)", "Batches (Long/Short)", "Contribution % (L/S)")
    widths2 = (0.16, 0.22, 0.20, 0.22, 0.20)

    p2_tickers = root.create_table(
        width=width, height=body_height, headings=headings2, widths=widths2,
        position="right", func=lambda _row: None,
    )
    for symbol in sorted(result.per_ticker):
        r = result.per_ticker[symbol]
        long_, short_ = r.long, r.short
        p2_tickers.new_row(
            symbol,
            f"{_fmt_pct(long_.win_rate)} / {_fmt_pct(short_.win_rate)}",
            f"{_fmt_ratio(long_.payoff_ratio)} / {_fmt_ratio(short_.payoff_ratio)}",
            f"{r.n_batches}({long_.n_batches}/{short_.n_batches})",
            f"{_fmt_pct(long_.contribution_pct, signed=True)} / {_fmt_pct(short_.contribution_pct, signed=True)}",
        )
    p2_tickers.run_script(
        f"""
        {p2_tickers.id}._div.style.fontSize = '13px'
        const scrollArea2 = {p2_tickers.id}._div.querySelector('div')
        scrollArea2.style.flex = '1 1 auto'
        scrollArea2.style.minHeight = '0'
        {p2_tickers.id}._div.style.float = 'none'
        {p2_tickers.id}._div.style.position = 'absolute'
        {p2_tickers.id}._div.style.top = '{body_top * 100}%'
        {p2_tickers.id}._div.style.left = '{left * 100}%'
        {p2_tickers.id}._div.style.display = 'none'
        """
    )

    # ---- page 3: Benchmark (plain buy&hold, per ticker) ----
    headings3 = ("Ticker", "Buy&Hold", "B&H Sharpe", "Up/Down Days", "Avg Move (Up/Down/Total)")
    widths3 = (0.16, 0.16, 0.16, 0.22, 0.30)

    p3_tickers = root.create_table(
        width=width, height=body_height, headings=headings3, widths=widths3,
        position="right", func=lambda _row: None,
    )
    for symbol in sorted(result.per_ticker):
        b = result.per_ticker[symbol].benchmark
        total_days = b.up_days + b.down_days
        p3_tickers.new_row(
            symbol,
            _fmt_pct(b.buy_hold_return),
            _fmt_ratio(b.buy_hold_sharpe),
            f"{total_days}({b.up_days}/{b.down_days})",
            f"{_fmt_pct(b.avg_up, signed=True)} / {_fmt_pct(b.avg_down, signed=True)} / {_fmt_pct(b.avg_total, signed=True)}",
        )
    p3_tickers.run_script(
        f"""
        {p3_tickers.id}._div.style.fontSize = '13px'
        const scrollArea3 = {p3_tickers.id}._div.querySelector('div')
        scrollArea3.style.flex = '1 1 auto'
        scrollArea3.style.minHeight = '0'
        {p3_tickers.id}._div.style.float = 'none'
        {p3_tickers.id}._div.style.position = 'absolute'
        {p3_tickers.id}._div.style.top = '{body_top * 100}%'
        {p3_tickers.id}._div.style.left = '{left * 100}%'
        {p3_tickers.id}._div.style.display = 'none'
        """
    )

    # ---- page-toggle buttons ----
    root.run_script(f"""
        const statsBtnBar = document.createElement('div')
        statsBtnBar.style.position = 'absolute'
        statsBtnBar.style.top = '{top * 100}%'
        statsBtnBar.style.left = '{left * 100}%'
        statsBtnBar.style.width = '{width * 100}%'
        statsBtnBar.style.height = '{button_h * 100}%'
        statsBtnBar.style.display = 'flex'
        statsBtnBar.style.gap = '8px'
        statsBtnBar.style.alignItems = 'center'
        statsBtnBar.style.zIndex = '4000'
        statsBtnBar.style.boxSizing = 'border-box'
        statsBtnBar.style.fontFamily = '{UI_FONT}'

        const mkStatsBtn = (label) => {{
            const b = document.createElement('button')
            b.innerText = label
            b.style.padding = '3px 10px'
            b.style.fontSize = '12px'
            b.style.fontFamily = '{UI_FONT}'
            b.style.border = '1px solid #2a2e39'
            b.style.borderRadius = '4px'
            b.style.cursor = 'pointer'
            b.style.color = '#d1d4dc'
            return b
        }}
        const statsBtn1 = mkStatsBtn('收益与Sharpe')
        const statsBtn2 = mkStatsBtn('多空对比')
        const statsBtn3 = mkStatsBtn('Benchmark')
        statsBtnBar.appendChild(statsBtn1)
        statsBtnBar.appendChild(statsBtn2)
        statsBtnBar.appendChild(statsBtn3)
        window.containerDiv.appendChild(statsBtnBar)

        const statsPage1 = [{p1_portfolio.id}._div, {p1_tickers.id}._div]
        const statsPage2 = [{p2_tickers.id}._div]
        const statsPage3 = [{p3_tickers.id}._div]
        const allStatsPages = [statsPage1, statsPage2, statsPage3]
        const allStatsBtns = [statsBtn1, statsBtn2, statsBtn3]
        const showStatsPage = (n) => {{
            allStatsPages.forEach((pageDivs, i) => {{
                pageDivs.forEach(d => d.style.display = i === n ? 'flex' : 'none')
                allStatsBtns[i].style.background = i === n ? '#2a2e39' : '#1e222d'
            }})
        }}
        statsBtn1.addEventListener('click', () => showStatsPage(0))
        statsBtn2.addEventListener('click', () => showStatsPage(1))
        statsBtn3.addEventListener('click', () => showStatsPage(2))
        showStatsPage(0)
    """)


def show_portfolio_pullback_backtest(
    dfs: dict[str, pd.DataFrame], result: "PortfolioBacktestResult",
) -> None:
    """
    Week 6: multi-ticker version of `show_pullback_backtest`.

    Left half builds ONE set of 4 tiles (5m/15m/30m/2h + ATR) and re-renders
    it in place - candles/MA/markers/dots/floor-maps swapped out, not
    rebuilt - whenever a different ticker is picked from the search box
    (`_build_ticker_search`). Building 9 separate tile sets up front would be
    9x the panes/DOM/crosshair-wiring for something only one of which is
    ever visible at a time.

    Right half is portfolio-level and does NOT change with the left-side
    ticker pick: the combined portfolio equity curve, the per-day
    inverse-ATR weight pie, and a stats table (Portfolio pinned on top,
    tickers below it alphabetically - see `_build_portfolio_stats_tables`).
    """
    symbols = sorted(dfs.keys())
    root = Chart(
        title="Pullback Strategy - Portfolio", toolbox=False,
        inner_width=_PB_COL_W, inner_height=_PB_MAIN_H,
        width=1800, height=1100, maximize=True,
    )
    _pin(root, 0.0, 0.0)

    scale = 1 - _PF_TOPBAR_H
    row_h = _PB_ROW_H * scale
    main_h = _PB_MAIN_H * scale
    atr_h = _PB_ATR_H * scale

    crosshair_entries: list[tuple[AbstractChart, str]] = []
    atr_legend_targets: dict[str, tuple[str, str]] = {}
    native_legend_ids: set[str] = set()

    tiles: dict[str, dict] = {}
    for tf in _PB_TIMEFRAME_ORDER:
        grid_row, grid_col = _PB_TIMEFRAME_GRID[tf]
        main_top = _PF_TOPBAR_H + grid_row * row_h
        chart_left = grid_col * _PB_COL_W
        atr_top = main_top + main_h

        main_chart = root.create_subchart(position="left", width=_PB_COL_W, height=main_h)
        main_chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
        main_chart.crosshair(mode="magnet")
        main_chart.legend(visible=True, font_size=12, font_family=UI_FONT)
        native_legend_ids.add(main_chart.id)
        main_chart.price_line(line_visible=False)
        main_chart.run_script(f"{main_chart.id}.volumeSeries.applyOptions({{priceLineVisible: false}})")
        _pin(main_chart, main_top, chart_left)

        ma_lines = {}
        for period, color in _MA_LINE_COLORS.items():
            ma_lines[period] = main_chart.create_line(f"MA{period}", color=color, width=1, price_line=False)
        dot_series = _create_pullback_dot_series(main_chart)

        atr_chart = main_chart.create_subchart(position="left", width=_PB_COL_W, height=atr_h, sync=False)
        atr_chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
        atr_chart.crosshair(mode="magnet")
        atr_line = atr_chart.create_line("atr", color=OPEN_COLOR, price_line=False)
        _create_legend_div(atr_chart, "atrLegend")
        _pin(atr_chart, atr_top, chart_left)
        _sync_timescale_only(main_chart, atr_chart)
        _wire_single_line_legend(atr_chart, atr_line, "ATR", "atrLegend")
        atr_legend_targets[atr_chart.id] = ("atrLegend", "ATR")

        crosshair_entries.append((main_chart, f"{main_chart.id}.series"))
        crosshair_entries.append((atr_chart, f"{atr_line.id}.series"))

        # Cooldown background band (admin overlay). Own price scale with zero
        # margins so a constant value paints the full tile height behind the
        # candles; kept out of the legend like the dot series.
        cooldown_hist = main_chart.create_histogram(
            "cooldown", color=COOLDOWN_COLOR, price_line=False, price_label=False,
            scale_margin_top=0.0, scale_margin_bottom=0.0,
        )
        cooldown_hist.run_script(f"""
            try {{
                const item = {main_chart.id}.legend._lines.find((l) => l.series == {cooldown_hist.id}.series)
                if (item) {{
                    {main_chart.id}.legend._lines = {main_chart.id}.legend._lines.filter((l) => l != item)
                    if (item.row && item.row.parentNode) {{
                        item.row.parentNode.removeChild(item.row)
                    }}
                }}
            }} catch (e) {{}}
        """)

        tiles[tf] = {
            "main_chart": main_chart, "atr_chart": atr_chart,
            "ma_lines": ma_lines, "dot_series": dot_series, "atr_line": atr_line,
            "cooldown_hist": cooldown_hist,
        }

    # Which symbol is showing and whether admin overlays are on. `render_ctx`
    # caches each tile's display frame + bar-index->time mapping so toggling
    # admin only redraws markers/bands - re-running render_symbol would reset
    # candles and zoom on every click.
    view_state = {"symbol": symbols[0], "admin": False}
    render_ctx: dict[str, tuple] = {}

    def draw_overlays() -> None:
        symbol = view_state["symbol"]
        admin = view_state["admin"]
        ticker = result.per_ticker[symbol]
        df_5m = dfs[symbol].reset_index(drop=True)
        for tf in _PB_TIMEFRAME_ORDER:
            if tf not in render_ctx:
                continue
            df_display, marker_time = render_ctx[tf]
            t = tiles[tf]
            _set_pullback_markers(
                t["main_chart"], df_display, ticker.trades, marker_time,
                pullback_points=ticker.pullback_points if admin else None,
            )
            _set_pullback_dots(t["dot_series"], ticker.trades, marker_time, df_display["timestamp"])
            t["cooldown_hist"].set(
                _cooldown_hist_df(df_5m, ticker.cooldown_spans, df_display, _PB_RESAMPLE_RULE[tf])
                if admin else None
            )

    def render_symbol(symbol: str) -> None:
        view_state["symbol"] = symbol
        df_5m = dfs[symbol].reset_index(drop=True)
        display_candles = {
            tf: (df_5m if rule is None else resample_ohlcv(df_5m, rule))
            for tf, rule in _PB_RESAMPLE_RULE.items()
        }
        ref_seconds = _to_ns(df_5m["timestamp"]).astype("int64") // 10**9
        floor_maps: dict[str, "pd.Series[int]"] = {}
        for tf, rule in _PB_RESAMPLE_RULE.items():
            if rule is None:
                continue
            starts = session_window_starts(df_5m, rule)
            floor_maps[tf] = _to_ns(starts).astype("int64") // 10**9

        for tf in _PB_TIMEFRAME_ORDER:
            t = tiles[tf]
            main_chart, atr_chart = t["main_chart"], t["atr_chart"]
            df_display = display_candles[tf]

            main_chart.set(_ohlc(df_display))
            for period, line in t["ma_lines"].items():
                line.set(_ma_lines_df(df_display, (period,))[period].rename(columns={"value": f"MA{period}"}))

            if tf in floor_maps:
                _set_floor_map(main_chart, ref_seconds, floor_maps[tf])
                starts = session_window_starts(df_5m, _PB_RESAMPLE_RULE[tf])
                marker_time = lambda i, _starts=starts: _starts.iloc[i]
            else:
                marker_time = None

            render_ctx[tf] = (df_display, marker_time)
            main_chart.fit()

            atr_df = _atr_series(df_display)
            t["atr_line"].set(atr_df)
            if tf in floor_maps:
                _set_floor_map(atr_chart, ref_seconds, floor_maps[tf])
            atr_seconds = atr_df["time"].astype("int64") // 10**9
            _set_value_lookup(atr_chart, atr_seconds, atr_df["atr"])
            _set_value_lookup(
                main_chart, _to_ns(df_display["timestamp"]).astype("int64") // 10**9, df_display["close"]
            )
            atr_chart.fit()

        draw_overlays()

    def on_admin_toggle(value: str) -> None:
        view_state["admin"] = value == "1"
        draw_overlays()

    _build_ticker_search(
        root, symbols, symbols[0], top=0.0, left=0.0, width=_PB_COL_W * 2, height=_PF_TOPBAR_H,
        on_select=render_symbol,
    )
    # 224px = the search input's fixed 200px + its 8px padding + a gap, so the
    # button sits immediately right of the ticker box at any window width.
    _build_admin_button(root, top=0.0, left_px=224, height=_PF_TOPBAR_H, on_toggle=on_admin_toggle)
    render_symbol(symbols[0])

    # 右半边: 组合权益曲线 + 权重饼图 + 统计表 - 不随左边切换的ticker变化。
    equity_chart = root.create_subchart(position="right", width=_PB_COL_W * 2, height=row_h, sync=False)
    equity_chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
    equity_chart.crosshair(mode="magnet")
    equity_line = equity_chart.create_line("equity", color=OPEN_COLOR, price_line=False)
    # Buy&hold-with-daily-rebalanced-weights benchmark - same weights_history
    # as the strategy, zero entry/exit timing, isolates how much of the
    # strategy's return comes from the vol-weighted rotation itself.
    benchmark_line = equity_chart.create_line(
        "benchmark", color="#787b86", style="dashed", width=1, price_line=False, price_label=False,
    )
    bar_return_hist = equity_chart.create_histogram(
        "bar_return", color=TP_COLOR, scale_margin_top=0.7, scale_margin_bottom=0.0, price_line=False
    )
    _build_return_title(root, equity_chart, title="Portfolio Return")
    _pin(equity_chart, _PF_TOPBAR_H, _PB_SUMMARY_LEFT)

    equity_curve = result.portfolio_equity_curve
    equity_df = equity_curve.rename("equity").reset_index()
    equity_df.columns = ["time", "equity"]
    equity_df["time"] = _to_ns(equity_df["time"])
    equity_line.set(equity_df)
    _set_value_lookup(equity_chart, equity_df["time"].astype("int64") // 10**9, equity_df["equity"])
    _wire_portfolio_legend(equity_chart, equity_line)

    benchmark_df = result.benchmark_equity_curve.rename("benchmark").reset_index()
    benchmark_df.columns = ["time", "benchmark"]
    benchmark_df["time"] = _to_ns(benchmark_df["time"])
    benchmark_line.set(benchmark_df)
    _wire_baseline_legend(equity_chart, benchmark_line)

    bar_return = equity_curve.diff().fillna(0.0)
    bar_return_df = bar_return.rename("bar_return").reset_index()
    bar_return_df.columns = ["time", "bar_return"]
    bar_return_df["time"] = _to_ns(bar_return_df["time"])
    bar_return_df["color"] = [TP_COLOR if x >= 0 else SL_COLOR for x in bar_return_df["bar_return"]]
    bar_return_hist.set(bar_return_df)
    _wire_return_legend(equity_chart, bar_return_hist)
    equity_chart.run_script(f"{equity_chart.id}.chart.timeScale().fitContent()")
    crosshair_entries.append((equity_chart, f"{equity_line.id}.series"))

    bottom_top = _PF_TOPBAR_H + row_h
    table_w = _PB_COL_W * 2 * 0.62
    pie_w = _PB_COL_W * 2 * 0.38
    pie_left = _PB_SUMMARY_LEFT + table_w

    _build_portfolio_stats_tables(
        root, result, top=bottom_top, left=_PB_SUMMARY_LEFT, width=table_w, height=row_h
    )
    pie_colors = _build_weight_pie_pinned(root, symbols, top=bottom_top, left=pie_left, width=pie_w, height=row_h)
    _wire_weight_hover(root, equity_chart, result.weights_history, symbols, pie_colors)

    _wire_synced_crosshair(
        crosshair_entries, legend_divs=atr_legend_targets, native_legend_ids=native_legend_ids
    )

    root.show(block=True)
