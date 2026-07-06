"""
Phase 5: interactive visualization for the Phase 3/4 portfolio backtest.

Layout (a 2-column grid): left half is one symbol's candlesticks +
entry/exit/weight-adjust markers, switchable via the topbar. The right half
is split top/bottom: top is "Portfolio Return" (equity + drawdown,
crosshair-synced to the left pane); bottom is a scrollable Sharpe/return
stats table (left) and a per-day portfolio weight pie chart (right), both
driven by whatever bar is under the cursor.
"""

from collections import defaultdict
from typing import cast

import pandas as pd
from lightweight_charts import AbstractChart, Chart

from src.engine.portfolio_backtest import Trade, WeightAdjustment
from src.rules.base import Side

OPEN_COLOR = "#2196F3"
TP_COLOR = "#26a69a"
SL_COLOR = "#ef5350"
WA_COLOR = "#ffca28"
PIE_COLORS = [OPEN_COLOR, TP_COLOR, SL_COLOR, WA_COLOR, "#ab47bc", "#8d6e63", "#26c6da", "#789262"]
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
) -> None:
    chart.clear_markers()
    times = aligned_df["timestamp"]
    markers = []
    for t in trades:
        is_long = t.side is Side.LONG
        markers.append(
            {
                "time": times.iloc[t.entry_bar_idx],
                "position": "below" if is_long else "above",
                "shape": "arrow_up" if is_long else "arrow_down",
                "color": OPEN_COLOR,
                "text": "Open Long" if is_long else "Open Short",
            }
        )
        side_label = "Long" if is_long else "Short"
        markers.append(
            {
                "time": times.iloc[t.exit_bar_idx],
                "position": "above" if is_long else "below",
                "shape": "arrow_down" if is_long else "arrow_up",
                "color": {"TP": TP_COLOR, "SL": SL_COLOR, "WA": WA_COLOR}[t.reason],
                "text": f"{t.reason} {side_label}",
            }
        )
    for wa in weight_adjustments:
        markers.append(
            {
                "time": times.iloc[wa.bar_idx],
                "position": "inside",
                "shape": "circle",
                "color": WA_COLOR,
                "text": "WA",
            }
        )
    markers.sort(key=lambda m: m["time"])
    if markers:
        chart.marker_list(markers)


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


def _build_return_title(chart: AbstractChart, equity_chart: AbstractChart) -> None:
    chart.run_script(
        f"""
        {equity_chart.id}.returnTitle = document.createElement('div')
        {equity_chart.id}.returnTitle.innerText = 'Portfolio Return'
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


def _wire_return_legend(equity_chart: AbstractChart, bar_return_hist) -> None:
    """
    Mirrors the main chart's top-left OHLCV legend, but for this pane's
    per-bar return: a small text box, top-left, updated on hover, colored
    green/red to match the bar's sign. Stacked just below the "Portfolio
    Return" title so the two don't overlap.
    """
    equity_chart.run_script(
        f"""
        {equity_chart.id}.returnLegend = document.createElement('div')
        {equity_chart.id}.returnLegend.style.position = 'absolute'
        {equity_chart.id}.returnLegend.style.top = '28px'
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
                {equity_chart.id}.returnLegend.innerText = `Return: ${{v >= 0 ? '+' : ''}}${{v.toFixed(2)}}`
                {equity_chart.id}.returnLegend.style.color = v >= 0 ? '{TP_COLOR}' : '{SL_COLOR}'
            }} else {{
                {equity_chart.id}.returnLegend.innerText = ''
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
    macd_chart.crosshair(mode="magnet")
    macd_line = macd_chart.create_line("macd", color=OPEN_COLOR)
    signal_line = macd_chart.create_line("signal", color=WA_COLOR)
    macd_hist = macd_chart.create_histogram("histogram", color=TP_COLOR, scale_margin_top=0.8, scale_margin_bottom=0.0)
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
    atr_line = atr_chart.create_line("atr", color=OPEN_COLOR)
    atr_chart.run_script(
        f"""
        {atr_chart.id}.wrapper.style.float = 'none'
        {atr_chart.id}.wrapper.style.position = 'absolute'
        {atr_chart.id}.wrapper.style.left = '0%'
        {atr_chart.id}.wrapper.style.display = 'none'
        """
    )

    def _set_macd(df: pd.DataFrame) -> None:
        macd_df = _macd_series(df)
        macd_line.set(macd_df[["time", "macd"]])
        signal_line.set(macd_df[["time", "signal"]])
        hist_df = macd_df[["time", "histogram"]].copy()
        hist_df["color"] = [TP_COLOR if v >= 0 else SL_COLOR for v in hist_df["histogram"]]
        macd_hist.set(hist_df)

    def _set_atr(df: pd.DataFrame) -> None:
        atr_line.set(_atr_series(df))

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
    equity_line = equity_chart.create_line("equity", color=OPEN_COLOR)
    # Per-bar P&L of the portfolio equity curve (not drawdown, which was an
    # unlabeled, always-red, hard-to-read blob) - green when that bar gained,
    # red when it lost, so the sign is legible at a glance.
    bar_return_hist = equity_chart.create_histogram(
        "bar_return", color=TP_COLOR, scale_margin_top=0.7, scale_margin_bottom=0.0
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
