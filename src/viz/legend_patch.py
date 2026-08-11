# -*- coding: utf-8 -*-
"""
把 lightweight-charts 自带 OHLCV 图例里的百分比改写成"上一根 close 到这一根
close"。

原先放在 `src.viz.chart` 里 (v2.1/phaseF 用), 但那个模块会连带 import 整套
回测栈 (engine / rules / data.resample)。v3.0 的分钟线浏览器只要这一个补丁,
不该为此拖进回测依赖, 所以抽到这里, 两边共用同一份实现。
"""
from __future__ import annotations

from lightweight_charts import AbstractChart

# 总开关: 关掉就回到库自带的 (close-open)/open。
PATCH_LEGEND_PCT = True


def patch_legend_percent(chart: AbstractChart) -> None:
    r"""
    Rewrite the % in the library's built-in OHLCV legend to close-to-close.

    lightweight-charts computes it as `(close - open) / open` - the hovered
    bar's own body - which makes every gap invisible: a bar that opens 3% above
    the prior close and then drifts down 0.1% reads "-0.10 %". Overnight gaps
    are exactly what the exit model cares about here, so the legend should show
    the conventional `(close - prev_close) / prev_close` instead.

    Implemented as an ADDITIONAL crosshair subscriber that rewrites just that
    one number in the already-rendered HTML. The library subscribes its own
    `legendHandler` by reference in the Legend constructor
    (`chart.subscribeCrosshairMove(this.legendHandler)`), so reassigning that
    property has no effect - the chart still holds the original reference.
    Subscribers fire in registration order and ours is registered later, so by
    the time it runs the legend text is already on screen and only needs the
    percentage swapped. The rendered string is otherwise untouched: same slot,
    same `|` separator, same sign, same 2dp, same `%`.

    Patching from here (rather than editing the installed package) keeps the
    change inside this repo, version-controlled, and safe from a
    `pip install --upgrade`. The first bar of a series has no predecessor and
    keeps the library's intrabar value.
    """
    if not PATCH_LEGEND_PCT:
        return
    chart.run_script(rf"""
    {chart.id}.chart.subscribeCrosshairMove(param => {{
        try {{
            const L = {chart.id}.legend
            if (!L || !L.percentEnabled || !L.candle || !param || !param.time) return
            const series = L.handler.series
            const cur = param.seriesData.get(series)
            if (!cur || cur.close === undefined) return
            const ts = L.handler.chart.timeScale()
            const coord = ts.timeToCoordinate(param.time)
            if (coord === null || coord === undefined) return
            const logical = ts.coordinateToLogical(coord.valueOf())
            if (logical === null || logical === undefined) return
            const prev = series.dataByIndex(logical.valueOf() - 1)
            if (!prev || !prev.close) return
            const pct = (cur.close - prev.close) / prev.close * 100
            const txt = `${{pct >= 0 ? '+' : ''}}${{pct.toFixed(2)}} %`
            L.candle.innerHTML = L.candle.innerHTML.replace(/[+-]?\d+(?:\.\d+)? %/, txt)
        }} catch (err) {{ console.log('legend pct patch:', err) }}
    }})
    """)
