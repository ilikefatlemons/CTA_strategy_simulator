# -*- coding: utf-8 -*-
"""
lineA-03 的标注型测试窗口: **两页各四格** x 三层 (主图 / ATR / MACD)。

    页1 决策层   15m  30m        页2 执行层   1m   5m
                 2h   1d                      15m  30m

六个周期, 但**瓦片只建六个不是八个** —— 15m/30m 两页都出现, 换页只挪位置, 不复制
series。分两页而不是 3x2 一屏六格: 那样格高掉三分之一, 两个副图各只剩 ~61px。

这个窗口的用途是**对着 K 线核对策略的每一步**, 不是看收益。所以:

  * 左上角一行: 两个页面 pill ｜ 三个指标开关 MA / ATR / MACD —— 纯 JS, 零 Python 往返
  * 右上角六个标注开关: 统计 / 回调 / 大周期反转 / 冷静期 / 入场双重条件 / 止损·吊灯线
  * **默认全关, 图上只有 Entry / 出场两种箭头**

------------------------------------------------------------- 四个库的坑 --

沿用 `chart_rbreaker.py:13-38` 与 `chart_pullback.py:1-45` 记下的那几条:

1. **所有 `run_script` 体被拼成一个字符串, 一次 `evaluate_js` 送进去**
   (`abstract.py:54-58`)。顶层 `const` 会跨脚本撞名; 一个未捕获异常会静默杀掉它
   **之后**的全部脚本, 表现为黑窗。所以每段都裹在裸 `{ }` 里 (块级作用域), 跨脚本
   状态一律挂 `window.la3`。已被 `chart_minute.build_ticker_search` 占用的顶层名,
   不许再用: `box` `input` `list` `active` `setActive` `visibleOpts` `pick` `openList`。

2. **marker 必须自己按时间升序排**, 而且**不能走库的 `marker_list`** —— 它把时间
   floor 到猜出来的 `_interval` 网格。乱序本身**不报错**, 是被 `visibleTimedValues`
   的二分静默漏画。两条都见 `lwc_helpers.set_markers_exact`。

3. **K 线帧不许带 `volume` 列**。带上库会自动建一个成交量副图, 把三层版面挤扁
   (`abstract.py:558-563`)。

4. **时间列必须是纳秒** (`lwc_helpers.to_ns`) —— pandas 默认 `datetime64[us]`, 库做
   `astype('int64')//10**9`, 不转会整体塌到 1970。

另外: **永远不重建图表对象** (重建会丢掉缩放状态), 只重新 `set()`。

--------------------------------------------------------------- 出场配色 --

出场箭头**按结果着色而不是只按理由**:

    固定止损        红      (必然是亏的)
    吊灯 · 盈利     绿
    吊灯 · 亏损     琥珀    ← 这一类是这套装配自带的性质, 不是 bug

吊灯结构上是追踪止损: 浮盈在 +1R 到 +2R 之间时它生效但坐在入场价下方, 那一段被它
打掉的交易结构性地亏损 (几何见 `src/engine/lineA_03_backtest.py` 的模块 docstring)。
把它单独染一个颜色, 是为了在图上一眼看见这件事而不是被"绿色=止盈"骗过去。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from lightweight_charts import Chart

from src.data.v3_timeframes import TFBars
from src.engine.lineA_03_backtest import 止损, 回测结果
from src.performance.lineA_03_stats import (
    统计, 面板副标题, 面板行, 面板标题, 面板脚注,
)
from src.viz.chart_minute import build_minute_chart, build_ticker_search
from src.viz.legend_patch import patch_legend_percent
from src.viz.lwc_helpers import (
    UI_FONT_CJK, create_legend_div, drop_legend_row, invalidate_crosshair_snap,
    pin, price_dp, set_markers_exact, sync_timescale_only, to_ns,
    wire_single_line_legend, wire_synced_crosshair,
)

# ---------------------------------------------------------------- 版面 ----
# 顶部两行浮层 (品种下拉 + 顶栏开关) 留位。TOP 与 pill 的字号/内边距耦合。
ROWS, COLS = 2, 2
TOP = 0.085
ROW_H = (1.0 - TOP) / ROWS
COL_W = 1.0 / COLS
ATR_H = ROW_H * 0.20
MACD_H = ROW_H * 0.20

# 六个周期, **两页各四格**, 格子大小与三层结构一个字都不动。
#
# 为什么不做 3x2 一屏六格: 那样 ROW_H 掉到 0.305, 两个副图各只剩 ~61px (1000px 窗口),
# 已经贴着可读下限; 主图从 274px 缩到 183px。分两页则格高完全不变。
#
# **15m / 30m 两页都出现**, 它们是两页的接缝 —— 决策层要看回调与风险尺, 执行层要看
# 信号落到哪根。但**瓦片只建六个不是八个**: 15m/30m 各自只有一份 series, 换页只挪位置。
# 复制成八格意味着多推一份 15m+30m 的全部数据, 十字线条目也从 24 涨到 32。
周期集: tuple[str, ...] = ("1m", "5m", "15m", "30m", "2h", "1d")
页名 = {1: "决策层", 2: "执行层"}
# {tf: {页号: (行, 列)}} —— 没有某页的键就是那一页不显示这一格
GRID: dict[str, dict[int, tuple[int, int]]] = {
    "15m": {1: (0, 0), 2: (1, 0)},
    "30m": {1: (0, 1), 2: (1, 1)},
    "2h":  {1: (1, 0)},
    "1d":  {1: (1, 1)},
    "1m":  {2: (0, 0)},
    "5m":  {2: (0, 1)},
}
格标签 = {"1m": "1 分钟", "5m": "5 分钟", "15m": "15 分钟",
          "30m": "30 分钟", "2h": "2 小时", "1d": "日线"}

MA周期 = (21, 55)
MA颜色 = {21: "#64b5f6", 55: "#ffb74d"}
ATR颜色 = "#2196F3"
DIF颜色, DEA颜色 = "#42a5f5", "#ffa726"
柱正, 柱负 = "#26a69a", "#ef5350"

入场色 = "#2196F3"
止损色 = "#ef5350"
吊灯盈色 = "#26a69a"
吊灯亏色 = "#ffa726"
回调色 = "#fdd835"
反转色 = "#ab47bc"
锁1色 = "#ff7043"
锁2色 = "#29b6f6"
冷静色 = "rgba(239, 83, 80, 0.13)"
固定止损线色 = "#ff00d9"      # 与固定止损箭头同色
点半径 = 1.5                  # 逐根圆点的半径, 见 `止损吊灯帧` 为什么不画折线
吊灯线色 = "#39a626"          # 与吊灯箭头同色

TEXT, MUTED, BORDER = "#d1d4dc", "#8b93a7", "#2a2e39"
选中底, 空闲底 = "#2196F3", "#1e222d"
面板底 = "rgba(30,34,45,0.94)"


@dataclass
class 瓦片:
    # 全是库的 AbstractChart / Line / Histogram, 没有公开的类型 stub
    tf: str
    main: Any
    atr面: Any
    macd面: Any
    ma线: dict[int, Any] = field(default_factory=dict)
    atr线: Any = None
    dif线: Any = None
    dea线: Any = None
    macd柱: Any = None
    冷静带: Any = None
    固定止损线: Any = None
    吊灯线: Any = None

    @property
    def 所在页(self) -> tuple[int, ...]:
        return tuple(sorted(GRID[self.tf]))

    def 顶(self, 页: int) -> float:
        return TOP + GRID[self.tf][页][0] * ROW_H

    def 左(self, 页: int) -> float:
        return GRID[self.tf][页][1] * COL_W

    # 建图时的初始落位 = 它最早出现的那一页。启动时 `applyPanes()` 会按当前页重排,
    # 所以这里落在哪一页都不影响最终版面 —— 但必须落在**某一页的合法坐标**上,
    # 否则第一帧会看到格子叠在一起。
    @property
    def top(self) -> float:
        return self.顶(self.所在页[0])

    @property
    def left(self) -> float:
        return self.左(self.所在页[0])


# ------------------------------------------------------------- 建面板 ----
def 建瓦片(root: Chart, 风险周期: str) -> dict[str, 瓦片]:
    """
    六格 x 三层, **必须在 `root.show()` 之前**建好。六个格子一次全建出来, 靠
    `applyPanes()` 按当前页决定谁显示 —— 换页不重建任何图表对象 (重建会丢缩放状态)。

    `风险周期` 那一格的 ATR 图例会标注「策略读这条」—— 其余各格的 ATR 只是陪看,
    不标出来的话很容易以为每条都参与判定。
    """
    出: dict[str, 瓦片] = {}
    for tf in 周期集:
        t = 瓦片(tf=tf, main=None, atr面=None, macd面=None)
        主 = root.create_subchart(position="left", width=COL_W, height=ROW_H)
        主.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
        主.crosshair(mode="normal")
        主.legend(visible=True, font_size=11, font_family=UI_FONT_CJK)
        主.price_line(line_visible=False, title="")
        主.candle_style(
            up_color="#26a69a", down_color="#ef5350",
            wick_up_color="#26a69a", wick_down_color="#ef5350",
            border_up_color="#26a69a", border_down_color="#ef5350",
        )
        # 图例百分比改成 close→close: 时段边界的跳空恰恰是四格对读时要看的东西
        patch_legend_percent(主)
        t.main = 主

        for 期 in MA周期:
            t.ma线[期] = 主.create_line(f"MA{期}", color=MA颜色[期], width=1,
                                        price_line=False, price_label=False)

        # 有效止损的两条腿, 画成**逐根圆点**而不是折线 —— 平仓处必须真断开, 而
        # LWC 的 line 渲染器在 whitespace 处**不断开**(它会画一条停在旧价位的横线)。
        # 完整理由与浏览器实测见 `止损吊灯帧` 的 docstring。
        t.固定止损线 = 主.create_line(
            "固定止损", color=固定止损线色, width=1,
            price_line=False, price_label=False)
        t.吊灯线 = 主.create_line(
            "吊灯", color=吊灯线色, width=1, price_line=False, price_label=False)
        for 线 in (t.固定止损线, t.吊灯线):
            主.run_script(
                f"{线.id}.series.applyOptions({{lineVisible: false, "
                f"pointMarkersVisible: true, pointMarkersRadius: {点半径}, "
                f"crosshairMarkerVisible: false}})")
        # 两行读数交给我们自己的图例 div —— 库自带的那份在空仓(whitespace)时只会
        # 留一行空白, 而这里要显示「未开仓」, 并且要跟着右上角开关一起显示/隐藏。
        drop_legend_row(主, t.固定止损线)
        drop_legend_row(主, t.吊灯线)
        create_legend_div(主, 'stopLegend', font_size=11, font_family=UI_FONT_CJK)
        主.run_script(
            f"{主.id}.stopLegend.style.top = '';"
            f"{主.id}.stopLegend.style.bottom = '30px';"
            f"{主.id}.stopLegend.style.display = 'none'")
        _止损图例(主, t.固定止损线, t.吊灯线)

        # 冷静期底色: 值恒 1.0 + 独立价格轴 + 零边距 -> 铺满整格高度。
        # 整段历史几百个区间用**一条**直方图, 不是几百个 vertical_span
        # (chart_pullback.py:156-160 / chart.py:2485-2499 的既有做法)。
        t.冷静带 = 主.create_histogram(
            "冷静期", color=冷静色, price_line=False, price_label=False,
            scale_margin_top=0.0, scale_margin_bottom=0.0)
        drop_legend_row(主, t.冷静带)

        atr面 = 主.create_subchart(position="left", width=COL_W, height=ATR_H, sync=False)
        atr面.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
        atr面.crosshair(mode="normal")
        t.atr线 = atr面.create_line("atr", color=ATR颜色, width=1,
                                    price_line=False, price_label=False)
        create_legend_div(atr面, "atrLegend", font_family=UI_FONT_CJK)
        标 = f"{格标签[tf]} ATR"
        if tf == 风险周期:
            标 += "  ← 策略读这条"
        wire_single_line_legend(atr面, t.atr线, 标, "atrLegend", digits=3)
        t.atr面 = atr面

        macd面 = 主.create_subchart(position="left", width=COL_W, height=MACD_H, sync=False)
        macd面.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
        # 三条 series 共用一个价格轴时 magnet 十字线的水平线会吸到 DIF/DEA 而不是柱,
        # 没有 per-series 办法钉住 —— 所以关掉水平线, 改用文字读数
        # (chart.py:712-718 记着同一件事)。
        # 十字线用 normal (不是 magnet), 水平线保留 —— 悬停时要能读出 MACD 的纵轴
        # 位置。v1.1 那份关掉水平线是因为它用了 magnet, 磁吸会把横线吸到 DIF/DEA
        # 而不是柱上 (chart.py:712-718)。normal 模式没有这个问题。
        macd面.crosshair(mode="normal")

        # **三条 series 必须都在 `right` 轴上。**
        #
        # 库里 `Histogram.__init__` 把 `priceScaleId` 写死成它自己的 id, 那是一条
        # **overlay 轴** —— lightweight-charts 只渲染 `left` / `right` 两条边轴,
        # overlay 轴既不画刻度也不能拖。所以:
        #   * 什么都不做  -> 线在 right(能拖)、柱在 overlay, 两者各自缩放, 拖右边栏
        #                    只动线不动柱
        #   * 让线去就柱  -> 三者同轴了, 但 right 上没有 series, **整条纵轴消失**,
        #                    刻度没了也拖不动 (踩过)
        #   * 让柱来就线  -> 对的那个。`applyOptions` 支持改 priceScaleId, 库里
        #                    `Lh()` 显式处理它 (`!==this._n.priceScaleId && Hl(this,i)`,
        #                    即把 series 迁到另一条轴)。
        t.macd柱 = macd面.create_histogram(
            "柱", color=柱正, price_line=False, price_label=False)
        t.macd柱.run_script(f"""{{
            try {{
                {t.macd柱.id}.series.applyOptions({{
                    priceScaleId: 'right',
                    // 直方图默认 `priceFormat: {{type:"volume"}}`, 会把 0.123 显示
                    // 成 "0" 之类。同轴之后它就是整条轴的格式, 必须改成价格格式。
                    priceFormat: {{type: 'price', precision: 3, minMove: 0.001}},
                }})
            }} catch (e) {{ console.log('macd hist -> right:', e) }}
        }}""")
        # 两条线用默认轴, 也就是 right —— 与柱同轴, 于是一起缩放
        t.dif线 = macd面.create_line("DIF", color=DIF颜色, width=1,
                                     price_line=False, price_label=False)
        t.dea线 = macd面.create_line("DEA", color=DEA颜色, width=1,
                                     price_line=False, price_label=False)
        # 边距设在 right 轴上 (不是设在柱那条已经废弃的 overlay 轴上)
        macd面.run_script(f"""{{
            try {{
                {macd面.id}.chart.priceScale('right').applyOptions({{
                    scaleMargins: {{top: 0.12, bottom: 0.12}}
                }})
            }} catch (e) {{ console.log('macd right margins:', e) }}
        }}""")

        create_legend_div(macd面, "macdLegend", font_family=UI_FONT_CJK)
        # 面板叫 **MACD**, 三个值分开列 —— 不叫 DIF。面板上有三条东西
        # (DIF / DEA / 柱), 拿其中一条的名字当面板名会误导; 而在通达信一系的口径里
        # 「MACD」指的还是柱, 更容易读错。
        _macd图例(macd面, t.macd柱, t.dif线, t.dea线, f"{格标签[tf]} MACD", "macdLegend")
        t.macd面 = macd面

        pin(主, t.top, t.left)
        pin(atr面, t.top + ROW_H - ATR_H - MACD_H, t.left)
        pin(macd面, t.top + ROW_H - MACD_H, t.left)
        # 缩放/平移主图时两个副图跟着走 —— 否则上下三层的时间轴各走各的, 对不齐。
        # 不用库原生 `create_subchart(sync=True)`: 它会另跑一套十字线配对, 和自建
        # 广播抢同一对父子图, 实测导致副图与自己主图之间十字线消失
        # (`lwc_helpers.py:344-350`)。
        sync_timescale_only(主, atr面)
        sync_timescale_only(主, macd面)
        出[tf] = t
    return 出


def _止损图例(pane, 固, 吊) -> None:
    """
    主图**本格**悬停时的两行止损读数（鼠标就在这一格上）。

    别的格子悬停时走 `wire_synced_crosshair` 的广播路径 —— `setCrosshairPosition`
    **不触发** `subscribeCrosshairMove` (`lwc_helpers.py:88-96`)，所以两条路都得有。
    两边的文案必须逐字一致，否则同一根 K 线在「悬停本格」和「悬停别格」时读数不同。

    空仓时显示「未开仓」而不是留空 —— 留空会让人以为是数据缺失。
    """
    pane.run_script(f"""{{
        try {{
            const el = {pane.id}.stopLegend
            const 画 = (p) => {{
                if (!p || p.time === undefined) {{ el.innerText = ''; return }}
                const a = p.seriesData.get({固.id}.series)
                const b = p.seriesData.get({吊.id}.series)
                const f = (x, n) => (x && x.value !== undefined)
                    ? x.value.toFixed(2) : {json.dumps("未开仓")}
                el.innerText = {json.dumps("固定止损")} + ': ' + f(a)
                    + '    ' + {json.dumps("吊灯")} + ': ' + f(b)
            }}
            el.innerText = ''
            {pane.id}.chart.subscribeCrosshairMove(画)
        }} catch (e) {{ console.log('stop legend:', e) }}
    }}""")


def _macd图例(pane, 柱, dif, dea, 标签: str, attr: str, 位数: int = 3) -> None:
    """
    MACD 面板**本格**悬停时的图例（鼠标就在这一格上）。

    别的格子悬停时走的是 `wire_synced_crosshair` 的广播路径 —— 因为
    `setCrosshairPosition` **不触发** `subscribeCrosshairMove`
    (`lwc_helpers.py:88-96`)，那条路的多值渲染由 helper 的 `extras` 负责。
    两条路的文案必须一致，否则同一根 K 线在「悬停本格」和「悬停别格」时读数格式不同。
    """
    pane.run_script(f"""{{
        try {{
            const el = {pane.id}.{attr}
            const 名 = {json.dumps(标签)}
            const 画 = (p) => {{
                if (!p || p.time === undefined) {{ el.innerText = 名; return }}
                const a = p.seriesData.get({dif.id}.series)
                const b = p.seriesData.get({dea.id}.series)
                const c = p.seriesData.get({柱.id}.series)
                const f = (x) => (x >= 0 ? '+' : '') + x.toFixed({位数})
                el.innerText = 名
                    + (a && a.value !== undefined ? '  DIF ' + f(a.value) : '')
                    + (b && b.value !== undefined ? '  DEA ' + f(b.value) : '')
                    + (c && c.value !== undefined ? '  柱 ' + f(c.value) : '')
            }}
            el.innerText = 名
            {pane.id}.chart.subscribeCrosshairMove(画)
        }} catch (e) {{ console.log('macd legend:', e) }}
    }}""")


def 注册瓦片(root: Chart, 瓦片们: dict[str, 瓦片]) -> None:
    """
    句柄挂到 `window.la3.tiles`。**必须在 `建骨架` 之后调** —— 脚本按调用顺序拼成
    一个 blob, `window.la3` 早于它存在才行。
    """
    条 = ", ".join(
        "{}: {{main: {}, atrPane: {}, macdPane: {}, mas: [{}], cd: {}, "
        "stopLines: [{}, {}], stopLegend: {}.stopLegend, pos: {}}}".format(
            json.dumps(tf), t.main.id, t.atr面.id, t.macd面.id,
            ", ".join(str(l.id) for l in t.ma线.values()), t.冷静带.id,
            t.固定止损线.id, t.吊灯线.id, t.main.id,
            # 每页一份 (top, left)。JSON 的键一律是字符串, JS 侧用 String(page) 查。
            json.dumps({str(页): [t.顶(页), t.左(页)] for 页 in t.所在页}),
        )
        for tf, t in 瓦片们.items()
    )
    root.run_script(f"""{{
        try {{
            window.la3.tiles = {{{条}}}
            window.la3.layout = {{rowH: {ROW_H}, atrH: {ATR_H}, macdH: {MACD_H}}}
        }} catch (e) {{ console.log('la3 tiles:', e) }}
    }}""")


def 联动十字线(root: Chart, 瓦片们: dict[str, 瓦片]) -> None:
    """
    四格主图 + 八个副图互相联动 —— **四周期对读正是这个窗口存在的理由**。

    走自建广播而不是库原生 `sync=`: 原生配对会另跑一套十字线传播, 和自定义广播抢同
    一对父子图, 实测导致副图与它自己的主图之间十字线消失 (`lwc_helpers.py:344-350`)。

    副图必须传**有数据的那个 series** (`atr线.series` / `dif线.series`), 不是
    `pane.series` —— 副图自动建的 K 线 series 是空的。
    """
    面板: list[dict] = []
    for tf, t in 瓦片们.items():
        面板.append({"pane": t.main, "series_js": f"{t.main.id}.series", "group": tf,
                    "legend": "native", "mas": list(t.ma线.values())})
        # 主图第二个图例块: 两条止损线的读数。空仓时显示「未开仓」(`empty`),
        # 价格不加 `+` 前缀 (`sign=False`)。与 `_止损图例` 的文案逐字一致。
        面板.append({"pane": t.main, "series_js": f"{t.main.id}.series", "group": tf,
                    "legend": "div", "div_attr": "stopLegend", "label": "",
                    "extras": [
                        {"name": "固定止损", "line": t.固定止损线, "digits": 2,
                         "empty": "未开仓", "sign": False},
                        {"name": "吊灯", "line": t.吊灯线, "digits": 2,
                         "empty": "未开仓", "sign": False}]})
        面板.append({"pane": t.atr面, "series_js": f"{t.atr线.id}.series", "group": tf,
                    "legend": "div", "div_attr": "atrLegend",
                    "label": f"{格标签[tf]} ATR", "digits": 3})
        # extras: 面板叫 MACD, 但要同时报 DIF / DEA / 柱 三个读数。
        # 与 `_macd图例` 那条本格路径的文案保持一致。
        面板.append({"pane": t.macd面, "series_js": f"{t.dif线.id}.series", "group": tf,
                    "legend": "div", "div_attr": "macdLegend",
                    "label": f"{格标签[tf]} MACD", "digits": 3,
                    "extras": [{"name": "DIF", "line": t.dif线, "digits": 3},
                               {"name": "DEA", "line": t.dea线, "digits": 3},
                               {"name": "柱", "line": t.macd柱, "digits": 3}]})
    wire_synced_crosshair(
        root, 面板, ns="window.la3",
        snap_sources={tf: f"{t.main.id}.series" for tf, t in 瓦片们.items()},
    )


# --------------------------------------------------------------- 数据 ----
def 驱动到1m(基: TFBars) -> tuple[np.ndarray, np.ndarray]:
    """
    **驱动周期**的下标 -> 它的**首 / 末** 1m 成分下标。

    引擎的所有下标 (交易、观测点、逐根数组) 都在驱动周期的下标空间里, 而各格 K 线
    与 1m 的映射是 `TFBars.bin_of` (1m -> 本格箱), 所以要落到别的格上得先换回 1m。

    `基` 就是驱动周期的 `TFBars`, **由调用方传进来, 不写死 `tf["15m"]`**。驱动时钟
    换成 1m 时 (二稿 L15 的 TODO 2), 这个函数退化成恒等映射, 调用点一个字不用改。

    **首用于入场与状态类事件, 末用于出场。** 入场成交在这根驱动 bar 的 open, 落点
    就是它的第一分钟; 而出场只知道「发生在这根之内」, 具体哪一分钟不可知 —— 标在
    末端才不会声称知道得比实际多。比驱动周期粗的格子上两者几乎总落进同一个箱,
    差别只出现在比它细的格子上。
    """
    首 = np.flatnonzero(np.r_[True, 基.bin_of[1:] != 基.bin_of[:-1]])
    末 = np.r_[首[1:] - 1, len(基.bin_of) - 1]
    return 首, 末


def 起箱of(tfb: TFBars, 结果: 回测结果, 基: TFBars) -> int:
    """暖机段切点: `结果.暖机根数` 是**驱动周期**的根数, 先换回 1m 再问本格。"""
    首, _ = 驱动到1m(基)
    if 结果.暖机根数 >= len(首):
        return tfb.n_bars
    return tfb.first_bin_at_or_after(int(首[结果.暖机根数]))


def K线帧(tfb: TFBars, 起箱: int) -> pd.DataFrame:
    """**不带 volume 列** —— 见模块 docstring 第 3 条坑。"""
    b = tfb.bars.iloc[起箱:]
    out = b[["open", "high", "low", "close"]].copy()
    out.insert(0, "time", to_ns(pd.Series(b.index)).to_numpy())
    return out.reset_index(drop=True)


def MA帧(tfb: TFBars, 期: int, 起箱: int) -> pd.DataFrame:
    """**在全帧(含暖机段)上滚完再切显示段** —— 首根可见 K 线上 MA55 就已经是热的。"""
    ma = tfb.bars["close"].rolling(期).mean().to_numpy("float64")
    idx = tfb.bars.index[起箱:]
    return pd.DataFrame({"time": to_ns(pd.Series(idx)).to_numpy(),
                         f"MA{期}": ma[起箱:]})


def ATR帧(tfb: TFBars, 起箱: int, 期: int = 14) -> pd.DataFrame:
    from src.indicators import atr as atr_series
    a = np.asarray(atr_series(tfb.bars, 期), dtype="float64")
    idx = tfb.bars.index[起箱:]
    return pd.DataFrame({"time": to_ns(pd.Series(idx)).to_numpy(), "atr": a[起箱:]})


def MACD帧(tfb: TFBars, 起箱: int, 快=12, 慢=26, 信号=9):
    """
    返回 `(DIF 帧, DEA 帧, 柱帧)`。柱是**逐点着色**的 —— 正绿负红。

    不逐点着色的话会有点渲染成黑色 (`chart.py:252-257` 记着)。
    """
    from src.strategy.lineA_03 import macd线
    dif, dea = macd线(tfb.bars["close"].to_numpy("float64"), 快, 慢, 信号)
    t = np.asarray(to_ns(pd.Series(tfb.bars.index[起箱:])))
    柱 = dif[起箱:] - dea[起箱:]
    柱帧 = pd.DataFrame({"time": t, "柱": 柱})
    柱帧["color"] = np.where(柱 >= 0, 柱正, 柱负)
    return (pd.DataFrame({"time": t, "DIF": dif[起箱:]}),
            pd.DataFrame({"time": t, "DEA": dea[起箱:]}),
            柱帧)


def 冷静期帧(结果: 回测结果, tfb: TFBars, 起箱: int,
           基: TFBars) -> pd.DataFrame | None:
    """
    一根合成 K 线只要**包含任意一根**处于冷静期的 1m bar 就整根涂色。
    `None` 表示没什么可画。

    `结果.冷静期区间` 是 **15m** 下标的闭区间, 先经 `首/末` 换回 1m 再按本格分箱。
    """
    if not 结果.冷静期区间:
        return None
    首, 末 = 驱动到1m(基)
    n = len(tfb.bin_of)
    冷 = np.zeros(n, dtype=bool)
    for a, b in 结果.冷静期区间:
        冷[int(首[a]):int(末[b]) + 1] = True
    if not 冷.any():
        return None
    每箱 = np.bincount(tfb.bin_of, weights=冷.astype("float64"),
                       minlength=tfb.n_bars) > 0
    idx = tfb.bars.index[起箱:]
    旗 = 每箱[起箱:]
    if not 旗.any():
        return None
    out = pd.DataFrame({"time": to_ns(pd.Series(idx)).to_numpy(),
                        "冷静期": np.where(旗, 1.0, np.nan)})
    return out.dropna(subset=["冷静期"])


def 止损吊灯帧(结果: 回测结果, tfb: TFBars, 起箱: int,
             基: TFBars) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    两条**贯穿整笔**的线, 各自都从入场画到出场:

        固定止损   入场时冻结, 整笔恒定                    -> 一条水平线
        吊灯       极值 ∓ 3·ATR₀, **不带地板**, 从入场就画  -> 一条单调线

    读法: **有效止损 = 两条里更靠上的那条(多头; 空头取更靠下的)。** 入场那一刻吊灯
    在固定止损**下方**(更差), 完全不起作用; 浮盈涨到 +1R 时两条线**相交**, 此后吊灯
    接管。交叉点因此一眼可见 —— 那正是引擎 docstring 里那三个临界点的几何。

    两条都直接取自引擎逐 1m 记的 `固定止损线` / `吊灯原线`, 图层不自己算。

    ---------------------------------------------------------------------
    为什么画成**逐根圆点**而不是线 (2026-08-26 改)
    ---------------------------------------------------------------------
    平仓之后必须**断开**。而 lightweight-charts 的 line series **不会**在
    whitespace 处断开 —— `js_data` 把 NaN 剔成只剩 time 的 whitespace 行是对的,
    但渲染器 (`walkLine`) 只是 `for(...) lineTo(...)`, **没有任何断口判断**, 它把
    洞两侧的两个真实点直接连起来。浏览器里逐像素验过:

        lineType=1 (阶梯)  -> 洞里画出一条**停在旧价位的横线**, 再在洞右缘竖跳
        lineType=0 (直线)  -> 洞里画出一条斜线
        两者都是桥, 不是断口

    单条 series 能拿到真断口的唯一办法是关掉连线、只画点:
    `lineVisible: false` + `pointMarkersVisible: true`。每根 bar 一个点, 点与点之间
    没有任何连接, 所以空仓的箱天然是空的。(另一条路是每笔交易一条 series ——
    实测 200 条 series 建站 63ms, 可行, 但 4 格 x 2 线 x 上百笔 = 八百多条, 换来的
    只是好看一点的折线, 不值。)

    ---------------------------------------------------------------------
    采样口径
    ---------------------------------------------------------------------
    每根合成 bar 取**箱内最后一根有仓位的 1m** 的值 —— 与 `close` 取末根同构。
    所以 bar b 上那个点 = 这根 bar 收盘那一刻的线位。要核对「有没有用到当根数据」,
    拿 **bar b 的点** 和 **bar b+1 的 low/high** 比: 前者在后者开始之前就已定死。

    三条注意:
      * bar 内部线还在逐分钟抬, 点显示的是这根 bar **末尾**的值, 也就是这根 bar
        期间的**最高**一档 (多头)。周期越粗越粗糙, 5m 那格最准。
      * 一根箱里出现两笔交易时 (2h 上可能), 取的是后一笔。
      * **末尾未平仓那一批两条线照画, 但没有任何箭头** —— 两条线取自引擎逐根写的
        数组, 尾段照写; 而 markers 只从 `结果.交易` 生成, 那一批刻意不进 `交易`
        (不造合成腿)。所以图最右边会出现「有止损线、没有入场箭头」的一段, 这是
        设计不是漏画, 面板上的「末尾未平仓 N 批」就是它。
    """
    # 引擎的两条线是 **15m** 索引的。先经 `基.bin_of` 铺回 1m —— 这只是**显示层的
    # 索引换算**(粗值在它自己的成分上重复), 不是把 15m 的判定"细化"到 1m: 一根
    # 15m 内部这条线是常数, 因为引擎一根 15m 才动它一次。
    固全 = 结果.固定止损线[基.bin_of]
    吊全 = 结果.吊灯原线[基.bin_of]
    n = len(固全)
    时 = np.asarray(to_ns(pd.Series(tfb.bars.index[起箱:])))
    if n == 0 or tfb.n_bars <= 起箱:
        空 = np.full(len(时), np.nan)
        return (pd.DataFrame({"time": 时, "固定止损": 空}),
                pd.DataFrame({"time": 时, "吊灯": 空}))

    固, 吊 = 固全, 吊全
    b = tfb.bin_of
    首 = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    序 = np.arange(n)

    def 箱末(arr: np.ndarray) -> np.ndarray:
        """每个箱里最后一个非 NaN 的值; 整箱空仓 -> NaN (库会剔成 whitespace 断口)。"""
        末 = np.maximum.reduceat(np.where(np.isnan(arr), -1, 序), 首)
        出 = np.full(len(首), np.nan)
        有 = 末 >= 0
        出[有] = arr[末[有]]
        return 出

    return (pd.DataFrame({"time": 时, "固定止损": 箱末(固)[起箱:]}),
            pd.DataFrame({"time": 时, "吊灯": 箱末(吊)[起箱:]}))


# --------------------------------------------------------------- 标记 ----
def 标记(结果: 回测结果, tfb: TFBars, 起箱: int, 基: TFBars,
        回调: bool = False, 反转: bool = False, 锁: bool = False) -> list[dict]:
    """
    **15m 下标** -> 1m 下标 -> 所属合成 K 线的末端标签。所以同一个标记会在四张图上
    各自落到对应的那根 —— 这正是四周期对读要的效果。

    入场与三类状态点走 `首`(那根驱动 bar 的第一分钟), 出场走 `末` —— 理由见 `驱动到1m`。

    出场箭头按**结果**着色, 不是只按理由 —— 见模块 docstring「出场配色」。
    """
    snap = tfb.bars.index.to_numpy()[tfb.bin_of]
    首, 末 = 驱动到1m(基)
    下限 = tfb.bars.index[起箱] if tfb.n_bars > 起箱 else None
    out: list[dict] = []

    def 加(下标: int, 用末: bool = False, **kw) -> None:
        t = snap[int(末[下标] if 用末 else 首[下标])]
        if 下限 is None or t >= 下限:
            out.append({"time": pd.Timestamp(t), **kw})

    for t in 结果.交易:
        多头 = t.方向 == "多"
        dp = price_dp(t.入场价)
        加(t.入场下标,
           position="below" if 多头 else "above",
           shape="arrow_up" if 多头 else "arrow_down",
           color=入场色,
           text=f"{'多' if 多头 else '空'} @{t.入场价:.{dp}f}")
        if t.理由 == 止损:
            色, 标 = 止损色, "止损"
        else:
            赚 = t.净收益 > 0
            色, 标 = (吊灯盈色 if 赚 else 吊灯亏色), ("吊灯" if 赚 else "吊灯亏")
        跳 = "(G)" if t.跳空成交 else ""
        加(t.出场下标, 用末=True,
           position="above" if 多头 else "below",
           shape="arrow_down" if 多头 else "arrow_up",
           color=色, text=f"{标}{跳} {t.净收益:+.2%}")

    if 回调:
        for p in 结果.回调闩锁点:
            加(p["下标"], position="inside", shape="circle", color=回调色,
               text="回调" + ("多" if p["方向"] == 1 else "空"))
    if 反转:
        for p in 结果.大周期翻转点:
            加(p["下标"], position="inside", shape="square", color=反转色,
               text="大周期转" + ("多" if p["方向"] == 1 else "空"))
    if 锁:
        for p in 结果.锁1解锁点:
            加(p["下标"], position="inside", shape="circle", color=锁1色, text="锁1")
        for p in 结果.锁2解锁点:
            加(p["下标"], position="inside", shape="circle", color=锁2色, text="锁2")

    out.sort(key=lambda m: m["time"])
    return out


# --------------------------------------------------------------- 推送 ----
def 推全部(瓦片们: dict[str, 瓦片], tf: dict[str, TFBars], 结果: 回测结果,
          参数) -> None:
    """全量重画。切品种时调。"""
    基 = tf[结果.参数.驱动周期]
    for 名 in 周期集:
        t, tfb = 瓦片们[名], tf[名]
        起 = 起箱of(tfb, 结果, 基)
        t.main.set(K线帧(tfb, 起))
        for 期, 线 in t.ma线.items():
            线.set(MA帧(tfb, 期, 起))
        t.atr线.set(ATR帧(tfb, 起, 参数.ATR周期))
        d, e, h = MACD帧(tfb, 起, 参数.MACD快, 参数.MACD慢, 参数.MACD信号)
        t.dif线.set(d)
        t.dea线.set(e)
        t.macd柱.set(h)
        # 冷静期 / 止损两条线的数据都一次性推好, 之后开关只改 visible —— 纯 JS 零往返
        t.冷静带.set(冷静期帧(结果, tfb, 起, 基))
        固帧, 吊帧 = 止损吊灯帧(结果, tfb, 起, 基)
        t.固定止损线.set(固帧)
        t.吊灯线.set(吊帧)
        t.main.fit()
        t.atr面.fit()
        t.macd面.fit()
    # 时间轴换过了 -> 作废十字线的 snap 缓存。必须排在四格 set() 之后。
    invalidate_crosshair_snap(瓦片们[周期集[0]].main, "window.la3")
    推标记(瓦片们, tf, 结果)


def 推标记(瓦片们: dict[str, 瓦片], tf: dict[str, TFBars], 结果: 回测结果,
         回调: bool = False, 反转: bool = False, 锁: bool = False) -> None:
    """
    只换 markers —— 不动 K 线, 因此不重置缩放。切标注开关时调。

    **不用库的 `marker_list`**: 它会把时间向下取整到猜出来的 `_interval` 网格,
    于是所有被时段断口截短的 bar (30m 的 10:15、2h 的 02:30/11:30/15:00 …) 上的
    标记都被挪到更早的一根, 看起来就像用了未来数据。见 `set_markers_exact`。
    """
    基 = tf[结果.参数.驱动周期]
    for 名 in 周期集:
        t, tfb = 瓦片们[名], tf[名]
        起 = 起箱of(tfb, 结果, 基)
        set_markers_exact(t.main, 标记(结果, tfb, 起, 基, 回调, 反转, 锁))


# --------------------------------------------------------------- 面板 ----
def 面板html(s: 统计) -> str:
    行 = "".join(
        f'<div style="display:flex;justify-content:space-between;gap:22px">'
        f'<span style="color:{MUTED}">{k}</span><span>{v}</span></div>'
        for k, v in 面板行(s)
    )
    脚 = "".join(
        f'<div style="color:{MUTED};font-size:10px;line-height:1.5;margin-top:4px">'
        f'⚠ {x}</div>' for x in 面板脚注(s)
    )
    return (
        f'<div style="font-weight:600;font-size:14px">{面板标题(s)}</div>'
        f'<div style="color:{MUTED};font-size:11px;margin:2px 0 8px;line-height:1.5">'
        f'{面板副标题(s)}</div>'
        f'{行}'
        f'<div style="border-top:1px solid {BORDER};margin-top:8px;padding-top:5px">'
        f'{脚}</div>'
    )


def 建面板(chart: Chart) -> None:
    chart.run_script(f"""{{
        const p = document.createElement('div')
        p.style.cssText = 'position:absolute;top:52px;right:12px;z-index:4003;'
            + 'padding:10px 14px;border-radius:6px;background:{面板底};'
            + 'border:1px solid {BORDER};color:{TEXT};font-size:13px;line-height:1.85;'
            + 'pointer-events:none;font-variant-numeric:tabular-nums;'
            + 'min-width:300px;max-width:460px'
        p.style.fontFamily = {json.dumps(UI_FONT_CJK)}
        p.style.display = window.la3.stats ? 'block' : 'none'
        window.la3.panel = p
        window.containerDiv.appendChild(p)
    }}""")


def 推面板(chart: Chart, html: str) -> None:
    chart.run_script(
        f"try {{ window.la3.panel.innerHTML = {json.dumps(html)} }} catch (e) {{}}")


# ----------------------------------------------------------------- JS ----
def 建骨架(chart: Chart) -> None:
    """
    `window.la3` —— 跨脚本共享状态。**必须第一个调**, 后面所有脚本都碰它。

    根图整屏隐藏: 它只做容器 (搜索框与面板都挂 `window.containerDiv`), 真正要看的
    是四格瓦片。
    """
    chart.run_script(f"""{{
        window.la3 = {{
            ma: true, atr: true, macd: true,          // 左上角三个指标开关
            stats: true,                              // 右上角统计面板
            回调: false, 反转: false, 冷静: false, 锁: false,   // 标注, **默认全关**
            止损线: false,                            // 固定止损 + 吊灯 两条线
            page: 1,                                  // 1 决策层 / 2 执行层
            range: null,                              // `_限可见区间` 算出来的可见窗口
            tiles: {{}}, layout: {{}}, panel: null, boxes: [], searchBoxW: 0,
        }}

        // 三层版面: 主图 / ATR / MACD 自上而下。关掉哪一层, 下面的往上移、主图长高。
        // 与 chart_rbreaker.applyPB 的差别: 那份只有一个副图所以只改高度就够了,
        // 这里有两个, 关掉 ATR 时 MACD 必须**上移**, 否则中间留一个空洞。
        window.la3.applyPanes = () => {{
            try {{
                const L = window.la3.layout
                const mainH = L.rowH - (window.la3.atr ? L.atrH : 0)
                                     - (window.la3.macd ? L.macdH : 0)
                const P = String(window.la3.page)
                for (const k in window.la3.tiles) {{
                    const t = window.la3.tiles[k]
                    const xy = t.pos[P]
                    // 本页没有这一格 -> 整格三层一起藏起来。**必须连主图一起藏**,
                    // 否则它还占着上一页的坑位, 和本页的格子叠在一起。
                    if (!xy) {{
                        t.main.wrapper.style.display = 'none'
                        t.atrPane.wrapper.style.display = 'none'
                        t.macdPane.wrapper.style.display = 'none'
                        continue
                    }}
                    const x = xy[1] * 100
                    let y = xy[0]
                    t.main.wrapper.style.display = ''
                    t.main.wrapper.style.left = x + '%'
                    t.main.scale.height = mainH
                    t.main.wrapper.style.top = (y * 100) + '%'
                    y += mainH
                    t.atrPane.wrapper.style.left = x + '%'
                    t.atrPane.wrapper.style.display = window.la3.atr ? '' : 'none'
                    if (window.la3.atr) {{
                        t.atrPane.wrapper.style.top = (y * 100) + '%'
                        y += L.atrH
                    }}
                    t.macdPane.wrapper.style.left = x + '%'
                    t.macdPane.wrapper.style.display = window.la3.macd ? '' : 'none'
                    if (window.la3.macd) {{
                        t.macdPane.wrapper.style.top = (y * 100) + '%'
                        y += L.macdH
                    }}
                    // reSize 必须排在 display 置空之后 —— 隐藏元素量出来的盒子是 0,
                    // 先 reSize 再显示会把图画成零宽。
                    t.main.reSize()
                    if (window.la3.atr) t.atrPane.reSize()
                    if (window.la3.macd) t.macdPane.reSize()
                    for (const ln of t.mas) ln.series.applyOptions({{visible: window.la3.ma}})
                    t.cd.series.applyOptions({{visible: window.la3.冷静}})
                    for (const ln of t.stopLines)
                        ln.series.applyOptions({{visible: window.la3.止损线}})
                    // 图例跟着开关一起显示/隐藏 —— 关掉线却留着读数会误导
                    t.stopLegend.style.display = window.la3.止损线 ? '' : 'none'
                }}
            }} catch (e) {{ console.log('la3.applyPanes', e) }}
        }}

        // 换页时把可见窗口补回去。**这一步不能省**: `_限可见区间` 是在建图那一批
        // 脚本里对**所有**面板发的 setVisibleRange, 而当时非首页的格子是隐藏的、
        // 盒子宽度为 0, 那次调用对它们是空转。不补的话切到第二页会看到全区间。
        window.la3.applyRange = () => {{
            try {{
                const r = window.la3.range
                if (!r) return
                const P = String(window.la3.page)
                for (const k in window.la3.tiles) {{
                    const t = window.la3.tiles[k]
                    if (!t.pos[P]) continue
                    for (const p of [t.main, t.atrPane, t.macdPane]) {{
                        try {{
                            p.chart.timeScale().setVisibleRange({{from: r.from, to: r.to}})
                        }} catch (e) {{}}
                    }}
                }}
            }} catch (e) {{ console.log('la3.applyRange', e) }}
        }}

        {chart.id}.wrapper.style.display = 'none'
    }}""")


def 建顶栏开关(chart: Chart) -> None:
    """
    左上角一整行 pill: `[决策层][执行层] ｜ [MA][ATR][MACD]`。

    **页面开关和指标开关刻意共用一行、一个容器。** 另起一行要动 `TOP`, 而 `TOP` 和
    pill 字号/内边距、以及下面四格的起始高度全都耦合着; 另开一个容器则要自己算横向
    偏移 (得先量出这一行的宽度)。共用一行两样都不用碰。

    **纯 JS, 零 Python 往返** —— 状态存 `window.la3.*`, `paint()` 重绘样式,
    再调 `applyPanes()`。换页额外调一次 `applyRange()` 把可见窗口补给新露出来的格子。

    照抄 `chart_rbreaker._build_lines_toggle` (:483-567)。`box` 是
    `build_ticker_search` 在同一批脚本里声明的顶层 const —— 位置**实测取它的高度**
    而不是写死, `chart_rbreaker.py:490-496` 记着这个坐标踩过一次高度相关的遮挡 bug。

    **这个函数必须排在 `注册瓦片` 之后**: 它结尾要调一次 `applyPanes()` 把非当前页
    的格子藏起来 —— Python 侧 `pin()` 只按「最早出现的那一页」落位, 不调这一下,
    第一帧会看到 1m/5m 叠在 2h/1d 上面。
    """
    页1标签, 页2标签 = json.dumps(f"1 {页名[1]}"), json.dumps(f"2 {页名[2]}")
    chart.run_script(f"""{{
        const w = document.createElement('div')
        w.style.cssText = 'position:absolute;left:10px;z-index:4002;display:flex;'
            + 'align-items:center;gap:6px;pointer-events:none'
        w.style.top = '46px'
        try {{ w.style.top = (box.getBoundingClientRect().height + 2) + 'px' }} catch (e) {{}}
        w.style.fontFamily = {json.dumps(UI_FONT_CJK)}

        const mk = (label) => {{
            const e = document.createElement('div')
            e.innerText = label
            // 外框是 pointer-events:none, 每个 pill 必须显式打开
            e.style.cssText = 'cursor:pointer;user-select:none;border-radius:4px;'
                + 'pointer-events:auto;font-size:12px;padding:4px 12px'
            w.appendChild(e)
            return e
        }}
        const b页1 = mk({页1标签}), b页2 = mk({页2标签})
        const 隔 = document.createElement('div')
        隔.style.cssText = 'width:1px;height:16px;background:{BORDER};margin:0 4px'
        w.appendChild(隔)
        const bMa = mk('MA'), bAtr = mk('ATR'), bMacd = mk('MACD')

        const one = (el, on) => {{
            el.style.background = on ? '{选中底}' : '{空闲底}'
            el.style.color = on ? '#ffffff' : '{TEXT}'
            el.style.border = '1px solid ' + (on ? '{选中底}' : '{BORDER}')
        }}
        const paint = () => {{
            // 页面是**单选**, 指标是各自独立的开关
            one(b页1, window.la3.page === 1); one(b页2, window.la3.page === 2)
            one(bMa, window.la3.ma); one(bAtr, window.la3.atr); one(bMacd, window.la3.macd)
        }}
        const 切 = (键, el) => el.addEventListener('click', () => {{
            window.la3[键] = !window.la3[键]
            paint()
            window.la3.applyPanes()
        }})
        切('ma', bMa); 切('atr', bAtr); 切('macd', bMacd)

        const 翻页 = (页, el) => el.addEventListener('click', () => {{
            if (window.la3.page === 页) return
            window.la3.page = 页
            paint()
            window.la3.applyPanes()
            window.la3.applyRange()
        }})
        翻页(1, b页1); 翻页(2, b页2)

        paint()
        window.containerDiv.appendChild(w)
        // 建图时 pin() 只按「最早出现的那一页」落位, 这一下把非当前页的格子藏掉
        window.la3.applyPanes()
    }}""")


def 建标注开关(chart: Chart, handler: str) -> None:
    """
    右上角六个 pill, **互相独立**(不是三向互斥), 点一次开、再点一次关。

      统计       纯 JS 改 panel.style.display, 零往返
      冷静期     纯 JS 改冷静期直方图的 visible —— 数据一次性推好, 不重推
      止损/吊灯线 同上, 两条 series 一起开关
      回调 / 大周期反转 / 入场双重条件
              改变 marker 集合, 必须重算 -> **合成一次往返**, 三个 flag 用 `;;;`
              拼一起 (`util.py:32-36` 按 `;;;` 拆成位置参数)

    容器与配色照抄 `chart_rbreaker._build_top_right_bar` (:570-632)。
    """
    chart.run_script(f"""{{
        const bar = document.createElement('div')
        bar.style.cssText = 'position:absolute;top:10px;right:12px;z-index:4000;'
            + 'display:flex;gap:8px;align-items:center'
        bar.style.fontFamily = {json.dumps(UI_FONT_CJK)}

        const mk = (label) => {{
            const e = document.createElement('div')
            e.innerText = label
            e.style.cssText = 'cursor:pointer;user-select:none;padding:5px 13px;'
                + 'border-radius:4px;font-size:12px;border:1px solid {BORDER};'
                + 'color:{TEXT};background:{空闲底}'
            bar.appendChild(e)
            return e
        }}
        const bStats = mk({json.dumps("统计")})
        const bPb = mk({json.dumps("回调")})
        const bRev = mk({json.dumps("大周期反转")})
        const bCd = mk({json.dumps("冷静期")})
        const bLock = mk({json.dumps("入场双重条件")})
        const bStop = mk({json.dumps("止损/吊灯线")})

        const paint = () => {{
            const 表 = [[bStats, window.la3.stats], [bPb, window.la3.回调],
                        [bRev, window.la3.反转], [bCd, window.la3.冷静],
                        [bLock, window.la3.锁], [bStop, window.la3.止损线]]
            for (const [e, on] of 表) {{
                e.style.background = on ? '{选中底}' : '{空闲底}'
                e.style.color = on ? '#ffffff' : '{TEXT}'
                e.style.border = '1px solid ' + (on ? '{选中底}' : '{BORDER}')
            }}
        }}
        const 推标注 = () => window.callbackFunction(
            `{handler}_~_${{window.la3.回调 ? 1 : 0}};;;`
            + `${{window.la3.反转 ? 1 : 0}};;;${{window.la3.锁 ? 1 : 0}}`)

        bStats.addEventListener('click', () => {{
            window.la3.stats = !window.la3.stats
            try {{ window.la3.panel.style.display = window.la3.stats ? 'block' : 'none' }}
            catch (e) {{}}
            paint()
        }})
        // 冷静期 / 止损线都只是 series 的可见性 —— 数据在 `推全部` 里一次推好,
        // 所以这两个开关是纯 JS 零往返。改变 marker 集合的那三个才需要回 Python。
        for (const [键, el] of [['冷静', bCd], ['止损线', bStop]]) {{
            el.addEventListener('click', () => {{
                window.la3[键] = !window.la3[键]
                paint()
                window.la3.applyPanes()
            }})
        }}
        for (const [键, el] of [['回调', bPb], ['反转', bRev], ['锁', bLock]]) {{
            el.addEventListener('click', () => {{
                window.la3[键] = !window.la3[键]
                paint()
                推标注()
            }})
        }}

        paint()
        window.containerDiv.appendChild(bar)
        window.la3.applyPanes()          // 初始版面 + 冷静期默认隐藏
    }}""")


# --------------------------------------------------------------- 入口 ----
def show_lineA_03(
    symbols: list[str], 加载: Callable[[str], tuple],
    default: str = "AU", 可见交易日: int = 60, debug: bool = False,
) -> None:
    """
    `加载(品种) -> (瓦片用的 tf 字典, 回测结果, 统计, 参数)`。

    数据准备留在入口脚本里, 这一层只管画 —— 与 `chart_rbreaker` 把 `render` 传进
    `build_ticker_search` 是同一个分工。
    """
    symbols = sorted(symbols)
    视图: dict = {"symbol": default}
    # 先播种缓存再建瓦片 —— 建瓦片要知道风险周期是哪一格 (那一格的 ATR 图例要标
    # 「策略读这条」)。不播种的话 `画全部` 会把同一个品种再加载一遍。
    缓存: dict[str, tuple] = {default: 加载(default)}

    def 取(sym: str):
        if sym not in 缓存:
            缓存[sym] = 加载(sym)
        return 缓存[sym]

    瓦片们 = 建瓦片(chart := build_minute_chart(
        title="lineA-03 · 多周期回调（标注型测试窗口）", debug=debug),
        缓存[default][3].风险周期)

    def 画全部(sym: str | None = None) -> None:
        if sym:
            视图["symbol"] = sym
        tf, 结果, 统, 参数 = 取(视图["symbol"])
        推全部(瓦片们, tf, 结果, 参数)
        推面板(chart, 面板html(统))
        _限可见区间(瓦片们, tf, 可见交易日)

    def 换标注(回调: str, 反转: str, 锁: str) -> None:
        tf, 结果, _, _ = 取(视图["symbol"])
        推标记(瓦片们, tf, 结果, 回调 == "1", 反转 == "1", 锁 == "1")

    # 顺序有讲究: 建骨架 必须在任何东西碰 window.la3 之前
    build_ticker_search(chart, symbols, default, 画全部)
    建骨架(chart)
    注册瓦片(chart, 瓦片们)
    联动十字线(chart, 瓦片们)
    handler = f"la3_notes_{chart.id.rsplit('.', 1)[-1]}"
    chart.win.handlers[handler] = 换标注
    建面板(chart)
    建顶栏开关(chart)
    建标注开关(chart, handler)

    画全部(default)
    chart.show(block=True)


def _限可见区间(瓦片们: dict[str, 瓦片], tf: dict[str, TFBars], 交易日: int) -> None:
    """
    只显示最后 N 个交易日。**统计仍然跑全区间** —— 两个口径是分开的。

    不调库的 `set_visible_range`: 它生成的 `setVisibleRange` 是**裸的**, 没有
    try/catch, 而这批脚本拼成一个 blob 一次执行, 任何未捕获异常都会静默杀掉它之后
    的全部脚本。这里自己发一段等价的、包了 try 的。
    """
    b = tf[周期集[0]].bars
    天 = b["trading_date"].drop_duplicates()
    if len(天) <= 交易日:
        return
    掩 = (b["trading_date"] >= 天.iloc[-交易日]).to_numpy()
    起 = pd.Timestamp(b.index.to_numpy()[掩][0])
    止 = pd.Timestamp(b.index.to_numpy()[-1])
    if pd.isna(起) or pd.isna(止):
        return
    f, t2 = 起.timestamp(), 止.timestamp()
    for 瓦 in 瓦片们.values():
        for 面 in (瓦.main, 瓦.atr面, 瓦.macd面):
            面.run_script(f"""{{
                try {{
                    {面.id}.chart.timeScale().setVisibleRange({{from: {f}, to: {t2}}})
                }} catch (e) {{ console.log('la3 range:', e) }}
            }}""")
    # 存一份给 `applyRange` —— 上面这一轮对**隐藏页**的格子是空转 (盒子宽度为 0),
    # 换页时要拿这个值给刚露出来的格子补一次。
    第一 = next(iter(瓦片们.values()))
    第一.main.run_script(
        f"{{ try {{ window.la3.range = {{from: {f}, to: {t2}}} }} catch (e) {{}} }}")
