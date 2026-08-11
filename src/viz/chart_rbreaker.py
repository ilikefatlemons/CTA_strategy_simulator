# -*- coding: utf-8 -*-
"""
v3.0 R-Breaker 突破策略的图表叠加层。

在分钟 K 线浏览器上加四样东西:
  1. 六条 R-Breaker 水平线 (逐交易日阶梯), 左上角一个开关统一开/关。
  2. 右上角 Breakout / Reversal 互斥按钮。Breakout 显示 Bbreak / Sbreak /
     追踪止损线 + 成交箭头; Reversal 显示四条反转线 + 它自己的止损线和箭头。
     两个策略互相独立, 各跑各的回测、各有各的统计面板。
  3. 入场/出场箭头。蓝=入场, 绿=盈利离场, 红=亏损离场。
  4. 中文统计面板, 由「策略结果统计」按钮开关。

---------------------------------------------------------------------------
lightweight-charts 的三个坑 (都已在 v4.1.3 的 bundle 里逐行核对过)
---------------------------------------------------------------------------
(1) **所有 run_script 体在 show() 前会被拼成一个字符串, 送进单次
    evaluate_js** (abstract.py:54-58)。所以顶层 `const` 会跨脚本冲突, 而且
    任何一处未捕获异常会静默杀掉这批脚本的剩余部分 —— 窗口直接变黑。
    对策: 每个脚本体裹在裸 `{ }` 块里, 需要跨脚本存活的挂到 `window.rb`,
    碰库内部的一律 try/catch。
    注意 `chart_minute.build_ticker_search` 已经占用了顶层的
    `box`/`input`/`list`/`active` 和 `setActive`/`visibleOpts`/`pick`/`openList`。

(2) **NaN 不会画出断口, 而是画出一座桥。** `util.py:42` 的 `js_data` 会逐条
    剔除值为 NaN 的键, 于是那一行序列化成 `{"time": …}`; 数据层把它归类成
    whitespace 行, 再被 `Ks()` 谓词整个滤出序列; 渲染器于是无条件地跨过去
    直接连线。所以"只在持仓期间画止损线"必须靠**逐点颜色**, 不能靠 NaN。
    逐点 `color` 在这个 bundle 里是支持的 (行工厂读 `n.color`), 库自己的
    成交量副图就在用 (abstract.py:561-562)。

(3) **一个点的颜色管的是"从它出发"的那一段。** 反编译 `walkLine` (压缩后
    叫 `dt`) 后确认: 样式变化时先用**旧**样式 stroke 掉已累积的路径。
      * lineType 0 (Simple): 整段 h-1 -> h 都用 h-1 的颜色。干净。
      * lineType 1 (WithSteps): 水平段用 h-1 的颜色, 但 x_h 处那一小段
        **垂直跳变用 h 的颜色**。
    所以: 六条线用阶梯 (日界不该有斜线), **止损线用 Simple** —— 否则把
    每笔最后一个点涂透明只能藏掉水平的"桥", x 处还会留下一根竖线。
    1 分钟的点距下, Simple 和 WithSteps 在笔内看不出区别。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from lightweight_charts import Chart

from src.data.v3_sessions import (
    CLEAN_DIR, DEFAULT_SESSION_MODE, SESSION_MODES, Prepared, PreparedBase,
    apply_mode, prepare_base,
)
from src.data.v3_timeframes import build_timeframes
from src.engine.pullback_v3_backtest import run_pullback_v3_backtest
from src.engine.rbreaker_backtest import TRAIL, RBreakerResult, run_rbreaker_backtest
from src.engine.rbreaker_reversal_backtest import run_rbreaker_reversal_backtest
from src.performance.pullback_stats import compute_stats as pb_compute_stats
from src.performance.pullback_stats import format_stats as pb_format_stats
from src.performance.rbreaker_stats import (
    RBreakerStats, compute_stats, format_stats, panel_rows, panel_subtitle, panel_title,
)
from src.strategy.pullback import PullbackConfig, PullbackParams
from src.strategy.rbreaker import (
    LINE_NAMES, RBreakerConfig, RBreakerParams, ReversalConfig,
)
from src.viz.chart_minute import build_minute_chart, build_ticker_search
from src.viz.chart_pullback import build_pullback_tiles
from src.viz.chart_pullback import draw_markers as draw_pb_markers
from src.viz.chart_pullback import push_tiles, register_tiles, wire_tile_crosshairs
from src.viz.chart_pullback import stats_html as pb_stats_html
from src.viz.lwc_helpers import UI_FONT_CJK, drop_legend_row, price_dp, to_ns

# 回调策略的暖机段: 趋势方向要日线 MA50, 所以显示窗口开始前必须先有 50 根日线。
# 60 = 50 + 10 根余量。
PB_WARMUP_DAYS = 60

# 视觉层级: 策略真正读的三条线用饱和色 + 2px 实线; 只做展示的四条降到细的
# 冷色。Senter/Benter 是追踪止损的地板, 算半功能, 给紫色虚线。
LINE_SPEC: dict[str, tuple[str, str, int]] = {          # name -> (color, style, width)
    "Bbreak": ("#26a69a", "solid", 2),
    "Sbreak": ("#ef5350", "solid", 2),
    "Senter": ("#7e57c2", "dashed", 1),
    "Benter": ("#7e57c2", "dashed", 1),
    "Ssetup": ("#78909c", "dotted", 1),
    "Bsetup": ("#78909c", "dotted", 1),
}
TRAIL_NAME = "trail"          # 突破策略的追踪止损
TRAIL_REV_NAME = "trail_rev"  # 反转策略的追踪止损 (两者永不同时可见)
TRAIL_COLOR = "#ff9800"
ENTRY_COLOR = "#2196F3"
WIN_COLOR = "#26a69a"
LOSS_COLOR = "#ef5350"
HIDDEN = "rgba(0,0,0,0)"     # 全透明 —— 用来藏掉不该出现的那一段

ACTIVE_BG = "#2196F3"
IDLE_BG = "#1e222d"
BORDER = "#2a2e39"
TEXT = "#d1d4dc"
MUTED = "#8b93a7"


# ------------------------------------------------------------- 线的数据 ----
def step_rows(prepared: Prepared, name: str) -> pd.DataFrame:
    """
    六条线之一的阶梯数据: **每个时段只发一个点**, 打在该时段首根 bar 上,
    末尾再补一个窗口末根的点让最后一段的线延伸到右边缘。

    阶梯渲染器会把前值一直保持到下一个点的 x, 所以「时段 k 首根 -> 时段 k+1
    首根」恰好是一条平线 + 一次垂直跳变。

    **按时段发点而不是按 trading_date**, 因为策略用的是 `Prepared.line_idx`
    (按时段钉住的线下标)。`day_night` 模式下时段跨两个 trading_date, 若这里
    仍按 trading_date 发点, 21:00 处会画出一次策略根本没用过的跳变 —— 图和
    实际执行对不上。对其余四种模式, 同一 trading_date 的两段取值相同, 渲染
    结果与旧版逐字节一致。

    开销: 6 × (时段数 + 1) 个点, 而不是 6 × 85000。按 `js_data` 实测 8.5 万行
    要 16.3 MB 算, 每次切品种省掉约 25 MB 的 JS 字符串。

    ---------------------------------------------------------------------
    隐藏点的**取值**必须向后填充, 不能向前填充
    ---------------------------------------------------------------------
    阶梯模式 (lineType 1) 下, 一个点的颜色只管**从它出发的水平段**; `x_h` 处
    那一小段**垂直跳变用的是点 h 自己的颜色** (见模块 docstring 第 (3) 条)。

    于是「隐藏点后面跟一个可见点」会画出一根**光秃秃的竖线** —— 水平段被前一个
    隐藏点涂没了, 只剩下跳变那一竖。视觉上就是每段开头一个 L / 反 L。

    对策: 隐藏点的值取**下一个可见点**的值 (bfill), 这样那根竖线高度为零,
    什么也画不出来。ffill 恰好相反, 会把高度撑到两个日子的线距 —— 那就是
    「只做夜盘」下六条线全变 L 形的原因。

    `valid=False` 的时段颜色涂成全透明 —— 那天没有合法参照, 不该画出一条
    陈旧的线。**但不可交易的时段照常画线**: 线是「昨日投影」这个日历事实的
    属性, 与本模式下能不能交易无关。「只做夜盘」时白盘段仍然要看得见这一天的
    六条线, 只是不会有成交箭头。
    """
    df, ln = prepared.df, prepared.lines
    if df.empty or ln.empty:
        return pd.DataFrame(columns=["time", name, "color"])

    first_pos = np.flatnonzero(df["is_session_first"].to_numpy())
    idx = prepared.line_idx[first_pos]
    shown = ln["valid"].to_numpy(bool)[idx]
    # 先把隐藏点挖空, 再 bfill —— 隐藏点因此继承**下一个**可见点的值。
    vals = pd.Series(ln[name].to_numpy()[idx]).where(shown)
    filled = vals.bfill().ffill()

    times = list(df.index.to_numpy()[first_pos])
    values = list(filled.to_numpy())
    colors = [LINE_SPEC[name][0] if v else HIDDEN for v in shown]
    # 收尾哨兵: 让最后一个时段的线一直画到窗口右边缘。
    times.append(df.index[-1])
    values.append(values[-1])
    colors.append(colors[-1])

    out = pd.DataFrame({"time": pd.to_datetime(times), name: values, "color": colors})
    out["time"] = to_ns(out["time"])
    return out.dropna(subset=[name])


def trail_rows(prepared: Prepared, result: RBreakerResult,
               name: str = TRAIL_NAME) -> pd.DataFrame:
    """
    追踪止损线: 只在持仓的那些根上发点。

    每笔交易的**最后一个点**涂成全透明, 这样"从它出发"的那一段 —— 也就是
    通往下一笔的那座桥 —— 不可见, 而笔内每一段照常可见。见模块 docstring 的
    第 (3) 条: 这个技巧只在 lineType 0 下干净, 所以这条线不设 lineType。
    """
    df = prepared.df
    inpos = result.in_position
    if df.empty or not inpos.any():
        return pd.DataFrame(columns=["time", name, "color"])

    idx = np.flatnonzero(inpos)
    out = pd.DataFrame({
        "time": df.index.to_numpy()[idx],
        name: result.trail_stop[idx],
        "color": TRAIL_COLOR,
    })
    last_bars = {t.exit_bar_idx for t in result.trades}
    out.loc[np.isin(idx, list(last_bars)), "color"] = HIDDEN
    out["time"] = to_ns(pd.to_datetime(out["time"]))
    return out


# --------------------------------------------------------------- 成交标记 ----
def build_markers(result: RBreakerResult) -> list[dict]:
    """
    入场蓝箭头 `Long @ 价` / `Short @ 价`; 离场箭头按**盈亏符号**着绿/红, 文字按
    **离场原因**分 `Stop ±x.xx%` / `End Session ±x.xx%`。所以收盘平仓的那笔也是
    绿或红。

    离场**刻意不带仓位百分比** —— R-Breaker 永远整仓进出, `100%` 是恒定值、没有
    信息量。Archive-Pullback 那边带, 因为一批会拆成两条腿
    (`chart_pullback.markers`)。两个视图的离场文案因此不同构, 是有意的。

    必须自己按时间排序: `marker_list` (abstract.py:249-273) 不排序, 而
    `setMarkers` 对乱序输入会出错 —— chart.py:180 就是为此排的。
    """
    markers: list[dict] = []
    for t in result.trades:
        long = t.direction == "long"
        markers.append({
            "time": t.entry_time,
            "position": "below" if long else "above",
            "shape": "arrow_up" if long else "arrow_down",
            "color": ENTRY_COLOR,
            "text": f"{'Long' if long else 'Short'} @ "
                    f"{t.entry_price:.{price_dp(t.entry_price)}f}",
        })
        pnl = t.pnl_net
        verb = "Stop" if t.reason == TRAIL else "End Session"
        markers.append({
            "time": t.exit_time,
            "position": "above" if long else "below",
            "shape": "arrow_down" if long else "arrow_up",
            "color": WIN_COLOR if pnl > 0 else LOSS_COLOR,
            "text": f"{verb} {pnl:+.2%}",
        })
    markers.sort(key=lambda m: m["time"])
    return markers


# --------------------------------------------------------------- 面板 HTML ----
def stats_html(stats: RBreakerStats) -> str:
    rows = "".join(
        f'<div style="display:flex;justify-content:space-between;gap:22px">'
        f'<span style="color:{MUTED}">{k}</span><span>{v}</span></div>'
        for k, v in panel_rows(stats)
    )
    return (
        f'<div style="font-weight:600;font-size:14px">{panel_title(stats)}</div>'
        f'<div style="color:{MUTED};font-size:11px;margin:2px 0 8px">{panel_subtitle(stats)}</div>'
        f'{rows}'
    )


# ------------------------------------------------------------------- JS ----
def _bootstrap(chart: Chart, default_mode: str, gap: int = 8) -> None:
    chart.run_script(f"""{{
        window.rb = {{
            linesOn: false,      // 左上角开关: 六条线
            mode: 'breakout',    // 右上角三向互斥: 'breakout' | 'reversal' | 'pullback' | null
            mode2: {json.dumps(default_mode)},   // 时段模式, 与 mode 正交
            modeLabels: {json.dumps(SESSION_MODES, ensure_ascii=True)},
            statsOn: true,       // 统计面板
            series: {{}},
            statsPanel: null,
            // Archive-Pullback 那一套。tiles / btns 由后面的构造函数填。
            pb: {{on: false, ma: true, atr: true, notes: false,
                  tiles: {{}}, btns: {{}}, layout: {{}}}},
            searchBoxW: 0, modeBox: null, notesBox: null, maBox: null,
        }}
        window.rb.applyVis = () => {{
            try {{
                const all = window.rb.linesOn
                const brk = window.rb.mode === 'breakout'
                const rev = window.rb.mode === 'reversal'
                const pb = window.rb.mode === 'pullback'
                // 突破读 Bbreak/Sbreak; 反转读 Ssetup/Senter/Benter/Bsetup。
                // 左上角「指标线 六条」独立叠加, 所以 all 与 brk/rev 是 or 关系。
                // 两条止损线永不同时可见, 各自对应一个策略。
                const vis = {{
                    Bbreak: all || brk,  Sbreak: all || brk,
                    Ssetup: all || rev,  Bsetup: all || rev,
                    Senter: all || rev,  Benter: all || rev,
                    trail: brk,          trail_rev: rev,
                }}
                // 回调视图下整张 root 图都藏起来了, 八条线一律关掉 —— 免得切回
                // Breakout 的瞬间闪一下上一个状态。
                for (const k in vis) {{
                    const s = window.rb.series[k]
                    if (s) s.series.applyOptions({{visible: pb ? false : vis[k]}})
                }}
            }} catch (e) {{ console.log('rb.applyVis', e) }}
        }}

        // 顶部那排浮层的横向排布:
        //   [ticker 搜索框][时段选择器][额外注释][MA][ATR][root 的 OHLCV 图例]
        // 后三节只在回调视图下出现, 所以图例的左偏移要跟着重算。
        // display:none 的元素测出来宽度是 0, 所以不在回调视图时它们自动不占位,
        // 这一段不需要判断当前是哪个视图。
        window.rb.layoutTopBar = () => {{
            try {{
                const bw = window.rb.searchBoxW
                let x = bw + window.rb.modeBox.getBoundingClientRect().width
                for (const el of [window.rb.notesBox, window.rb.maBox]) {{
                    if (!el) continue
                    el.style.left = x + 'px'
                    x += el.getBoundingClientRect().width
                }}
                const legend = {chart.id}.legend
                if (legend && legend.div) legend.div.style.left = (x + {gap}) + 'px'
            }} catch (e) {{ console.log('rb.layoutTopBar', e) }}
        }}

        // 回调视图的整块开合。纯 JS —— MA / ATR 两个开关因此零往返。
        window.rb.applyPB = () => {{
            try {{
                const pb = window.rb.pb
                pb.on = window.rb.mode === 'pullback'
                // root 整屏图与四格互斥。隐藏 root 时它的 OHLCV 图例(是 div 的
                // 子节点)跟着消失, 而挂在 containerDiv 上的浮层照常可见。
                {chart.id}.wrapper.style.display = pb.on ? 'none' : ''
                if (!pb.on) {chart.id}.reSize()
                for (const k in pb.tiles) {{
                    const t = pb.tiles[k]
                    t.main.wrapper.style.display = pb.on ? '' : 'none'
                    t.atrPane.wrapper.style.display = (pb.on && pb.atr) ? '' : 'none'
                    if (!pb.on) continue
                    // ATR 关掉 -> 主图长高填满整行 (top 不变, 只改高度)
                    t.main.scale.height = pb.atr ? pb.layout.mainH : pb.layout.rowH
                    t.main.reSize()
                    if (pb.atr) t.atrPane.reSize()
                    for (const ln of t.mas) ln.series.applyOptions({{visible: pb.ma}})
                }}
                // 「指标线 六条」与「MA / ATR」互斥显示 —— 回调策略不读那六条线。
                // MA/ATR 整盒开合 (它是 flex 容器, 两个按钮之间有 gap), 不是逐个
                // 按钮开合 —— 否则盒子还在原位占着 layoutTopBar 的一节。
                if (pb.btns.lines) pb.btns.lines.style.display = pb.on ? 'none' : ''
                if (window.rb.maBox) {{
                    window.rb.maBox.style.display = pb.on ? 'flex' : 'none'
                }}
                if (window.rb.notesBox) {{
                    window.rb.notesBox.style.display = pb.on ? '' : 'none'
                }}
                if (pb.paintBtns) pb.paintBtns()
                window.rb.layoutTopBar()
            }} catch (e) {{ console.log('rb.applyPB', e) }}
        }}
    }}""")


def _register_series(chart: Chart, series: dict) -> None:
    entries = ", ".join(f"{name!r}: {ln.id}" for name, ln in series.items())
    chart.run_script(f"{{ window.rb.series = {{{entries}}} }}")


def _build_session_picker(chart: Chart, handler: str, default_mode: str) -> None:
    """
    ticker 搜索框**右边**的时段模式选择器。

    刻意不复用 `chart_minute.build_ticker_search` —— 那个函数被纯浏览器共用,
    而且它在顶层声明了 `box`/`input`/`list`/`active` 和 `setActive`/`visibleOpts`/
    `pick`/`openList`, 再实例化一份必然重名冲突 (所有脚本会被拼成一个 blob 送进
    单次 evaluate_js)。这里整段裹在裸 `{ }` 块里, 只有 5 个固定选项、点击展开,
    不需要键盘导航。

    **同时把库自带的 OHLCV 图例再往右推。** `build_ticker_search` 末尾把图例挪到
    `left = 搜索框宽 + 8`(chart_minute.py:217-218), 而本选择器正好占了那个位置。
    实际的排布交给 `window.rb.layoutTopBar()` —— 「额外注释」按钮还要插在中间,
    而它是后建的。`box` 是前一个脚本的顶层 const, 同一批 evaluate_js 里可见。
    """
    opts_js = "\n".join(
        f"""
        {{
            const o = document.createElement('div')
            o.innerText = {json.dumps(label)}
            o.dataset.mode = {json.dumps(key)}
            o.style.cssText = 'padding:4px 10px;cursor:pointer;white-space:nowrap'
            o.addEventListener('mouseenter', () => o.style.background = '{BORDER}')
            o.addEventListener('mouseleave', () => o.style.background =
                (o.dataset.mode === window.rb.mode2 ? '{BORDER}' : ''))
            o.addEventListener('mousedown', (e) => {{ e.preventDefault(); pickMode(o) }})
            modeList.appendChild(o)
        }}
        """
        for key, label in SESSION_MODES.items()
    )
    chart.run_script(f"""{{
        const modeBox = document.createElement('div')
        modeBox.style.cssText = 'position:absolute;top:0%;z-index:4000;padding:8px 0;'
            + 'pointer-events:none'
        modeBox.style.fontFamily = {json.dumps(UI_FONT_CJK)}

        const modeBtn = document.createElement('div')
        modeBtn.style.cssText = 'padding:5px 9px;border-radius:4px;font-size:13px;'
            + 'background:rgba(30,34,45,0.92);color:{TEXT};border:1px solid {BORDER};'
            + 'cursor:pointer;user-select:none;pointer-events:auto;white-space:nowrap'
        modeBox.appendChild(modeBtn)

        const modeList = document.createElement('div')
        modeList.style.cssText = 'position:absolute;top:calc(100% - 4px);left:0;'
            + 'background:rgba(30,34,45,0.97);border:1px solid {BORDER};border-radius:4px;'
            + 'color:{TEXT};font-size:13px;display:none;z-index:4001;pointer-events:auto'
        modeBox.appendChild(modeList)

        const paintMode = () => {{
            const cur = window.rb.mode2
            modeBtn.innerText = '时段: ' + (window.rb.modeLabels[cur] || cur) + ' ▾'
            for (const o of modeList.children) {{
                o.style.background = (o.dataset.mode === cur) ? '{BORDER}' : ''
            }}
        }}
        const pickMode = (o) => {{
            modeList.style.display = 'none'
            if (o.dataset.mode === window.rb.mode2) return
            window.rb.mode2 = o.dataset.mode
            paintMode()
            window.callbackFunction(`{handler}_~_${{o.dataset.mode}}`)
        }}
        modeBtn.addEventListener('click', () => {{
            modeList.style.display = modeList.style.display === 'none' ? 'block' : 'none'
        }})
        modeBtn.addEventListener('blur', () => {{ modeList.style.display = 'none' }})

        {opts_js}

        window.rb.mode2 = {json.dumps(default_mode)}
        paintMode()
        window.containerDiv.appendChild(modeBox)

        // 放到搜索框右边, 再把 OHLCV 图例推到本选择器右边。
        // `modeBox` 和 `box` 都是各自脚本块里的 const, 后面的脚本看不见 —— 挂到
        // window.rb 上, 好让「额外注释」按钮和 layoutTopBar 能接着排下去。
        // (既有模式: window.rb.statsPanel。)
        try {{
            const bw = box.getBoundingClientRect().width
            modeBox.style.left = bw + 'px'
            window.rb.searchBoxW = bw
            window.rb.modeBox = modeBox
            window.rb.layoutTopBar()
        }} catch (e) {{ console.log('session picker layout:', e) }}
    }}""")


def _build_extra_notes_button(chart: Chart, handler: str) -> None:
    """
    时段选择器**右边**的「额外注释」开关。只在 Archive-Pullback 视图下出现。

    打开后四张图上同时画出: 回调确认的黄色圆点 + 冷静期的整格红色背景带 ——
    也就是 phaseF 那个 Admin 叠加层 (`chart.py:1733-1778` / `2514-2532`) 的对应物。

    走一次 Python 往返: markers 必须经 `chart.marker_list` (库自己管着
    `chart.markers`, 直接调 `setMarkers` 会和 `clear_markers` 脱节)。
    """
    chart.run_script(f"""{{
        const notesBox = document.createElement('div')
        notesBox.style.cssText = 'position:absolute;top:0%;z-index:4000;padding:8px 0;'
            + 'pointer-events:none;display:none'
        notesBox.style.fontFamily = {json.dumps(UI_FONT_CJK)}

        const b = document.createElement('div')
        b.innerText = {json.dumps("额外注释")}
        b.style.cssText = 'padding:5px 9px;border-radius:4px;font-size:13px;'
            + 'cursor:pointer;user-select:none;pointer-events:auto;white-space:nowrap;'
            + 'border:1px solid {BORDER}'
        const paint = () => {{
            const on = window.rb.pb.notes
            b.style.background = on ? '{ACTIVE_BG}' : 'rgba(30,34,45,0.92)'
            b.style.color = on ? '#ffffff' : '{TEXT}'
            b.style.border = '1px solid ' + (on ? '{ACTIVE_BG}' : '{BORDER}')
        }}
        b.addEventListener('click', () => {{
            window.rb.pb.notes = !window.rb.pb.notes
            paint()
            window.callbackFunction(`{handler}_~_${{window.rb.pb.notes ? 1 : 0}}`)
        }})
        paint()
        notesBox.appendChild(b)
        window.containerDiv.appendChild(notesBox)
        window.rb.notesBox = notesBox
        window.rb.layoutTopBar()
    }}""")


def _build_stats_panel(chart: Chart) -> None:
    """面板只建一次, 切品种/切模式时只换 innerHTML。"""
    chart.run_script(f"""{{
        const p = document.createElement('div')
        p.style.cssText = 'position:absolute;top:48px;right:12px;z-index:4003;'
            + 'padding:10px 14px;border-radius:6px;background:rgba(30,34,45,0.94);'
            + 'border:1px solid {BORDER};color:{TEXT};font-size:13px;line-height:1.85;'
            + 'pointer-events:none;font-variant-numeric:tabular-nums;min-width:210px'
        p.style.fontFamily = {json.dumps(UI_FONT_CJK)}
        p.style.display = window.rb.statsOn ? 'block' : 'none'
        window.rb.statsPanel = p
        window.containerDiv.appendChild(p)
    }}""")


def _build_lines_toggle(chart: Chart) -> None:
    """
    左上角的图层开关。**两个容器**, 按视图互斥显示:

      * `w`      搜索框正下方 (y≈46px) —— 只放「指标线 六条」, 只在 R-Breaker 视图出现
      * `maBox`  顶栏那一行 —— 放「MA」「ATR」(默认都开), 只在 Archive-Pullback 出现

    **为什么 MA/ATR 不能留在搜索框下面**: 那个位置 (left:10px, y 46~73px,
    z-index 4002) 正好压住 1m tile 的原生 OHLC 图例 —— tile 钉在 `top:5%`,
    图例是 `.legend{{top:10px;left:10px;z-index:3000}}`, 于是图例的 OHLC 那行落在
    y = 0.05*H+10 ~ +30。窗口高 1000px 时两者重叠约 13px, 760px 时完全盖住,
    1400px 时又不重叠 —— 所以这个 bug 会随窗口高度出现和消失, 别人复现不了很正常。
    R-Breaker 视图下没这个问题: 那时 root 满屏, 它的图例已被 `layoutTopBar` 推到
    顶栏右端, 46px 那一带只有 K 线。

    MA / ATR 都是**纯 JS**: series 已经 set 过了, 只需 applyOptions({{visible}});
    ATR 还要改一下主图高度并 reSize。两者都不需要回 Python。
    """
    chart.run_script(f"""{{
        const w = document.createElement('div')
        w.style.cssText = 'position:absolute;left:10px;z-index:4002;display:flex;'
            + 'align-items:center;gap:6px'
        w.style.top = '46px'
        // 搜索框是 shrink-to-fit 的绝对定位块, 实测它的高度比写死像素稳。
        // `box` 是 build_ticker_search 在同一批脚本里声明的顶层 const。
        try {{ w.style.top = (box.getBoundingClientRect().height + 2) + 'px' }} catch (e) {{}}
        w.style.fontFamily = {json.dumps(UI_FONT_CJK)}

        // 顶栏那一节。样式与 `_build_extra_notes_button` 的 notesBox 逐字对齐,
        // 好让 layoutTopBar 一视同仁。left 由 layoutTopBar 填。
        const maBox = document.createElement('div')
        maBox.style.cssText = 'position:absolute;top:0%;z-index:4000;padding:8px 0;'
            + 'pointer-events:none;display:none;align-items:center;gap:6px'
        maBox.style.fontFamily = {json.dumps(UI_FONT_CJK)}

        const mk = (label, pad, parent) => {{
            const e = document.createElement('div')
            e.innerText = label
            // pointer-events:auto 是必须的 —— maBox 外框是 pointer-events:none
            // (和 notesBox 一样, 免得整条透明带挡住底下的 K 线), 不显式打开的话
            // MA/ATR 会变成点不动的装饰。
            e.style.cssText = 'cursor:pointer;user-select:none;border-radius:4px;'
                + 'pointer-events:auto;font-size:13px;padding:5px ' + pad + 'px'
            parent.appendChild(e)
            return e
        }}
        const bLines = mk({json.dumps("指标线 六条")}, 14, w)
        const bMa = mk('MA', 14, maBox)
        const bAtr = mk('ATR', 14, maBox)

        const paintOne = (el, on) => {{
            el.style.background = on ? '{ACTIVE_BG}' : '{IDLE_BG}'
            el.style.color = on ? '#ffffff' : '{TEXT}'
            el.style.border = '1px solid ' + (on ? '{ACTIVE_BG}' : '{BORDER}')
        }}
        const paint = () => {{
            paintOne(bLines, window.rb.linesOn)
            paintOne(bMa, window.rb.pb.ma)
            paintOne(bAtr, window.rb.pb.atr)
        }}
        window.rb.pb.btns = {{lines: bLines, ma: bMa, atr: bAtr}}
        window.rb.pb.paintBtns = paint

        bLines.addEventListener('click', () => {{
            window.rb.linesOn = !window.rb.linesOn
            paint()
            window.rb.applyVis()
        }})
        bMa.addEventListener('click', () => {{
            window.rb.pb.ma = !window.rb.pb.ma
            paint()
            window.rb.applyPB()
        }})
        bAtr.addEventListener('click', () => {{
            window.rb.pb.atr = !window.rb.pb.atr
            paint()
            window.rb.applyPB()
        }})

        paint()
        window.containerDiv.appendChild(w)
        window.containerDiv.appendChild(maBox)
        window.rb.maBox = maBox
        window.rb.layoutTopBar()
    }}""")


def _build_top_right_bar(chart: Chart, handler: str) -> None:
    """
    右上角: 策略结果统计 / **Archive-Pullback** / Breakout / Reversal。

    后三个**三向互斥**, 再点一次当前选中的就全关。Archive-Pullback 刻意排在
    「策略结果统计」和 Breakout 之间。
    """
    chart.run_script(f"""{{
        const bar = document.createElement('div')
        bar.style.cssText = 'position:absolute;top:10px;right:12px;z-index:4000;'
            + 'display:flex;gap:8px;align-items:center'
        bar.style.fontFamily = {json.dumps(UI_FONT_CJK)}

        const mk = (label) => {{
            const e = document.createElement('div')
            e.innerText = label
            e.style.cssText = 'cursor:pointer;user-select:none;padding:5px 14px;'
                + 'border-radius:4px;font-size:13px;border:1px solid {BORDER};'
                + 'color:{TEXT};background:{IDLE_BG}'
            bar.appendChild(e)
            return e
        }}
        const bStats = mk({json.dumps("策略结果统计")})
        const bPb = mk('Archive-Pullback')
        const bBrk = mk('Breakout')
        const bRev = mk('Reversal')

        const paint = () => {{
            const on = [
                [bStats, window.rb.statsOn],
                [bPb, window.rb.mode === 'pullback'],
                [bBrk, window.rb.mode === 'breakout'],
                [bRev, window.rb.mode === 'reversal'],
            ]
            for (const [e, active] of on) {{
                e.style.background = active ? '{ACTIVE_BG}' : '{IDLE_BG}'
                e.style.color = active ? '#ffffff' : '{TEXT}'
                e.style.border = '1px solid ' + (active ? '{ACTIVE_BG}' : '{BORDER}')
            }}
        }}
        const setMode = (m) => {{
            // 三向互斥, 且再点一次当前选中的就全关。
            window.rb.mode = (window.rb.mode === m) ? null : m
            paint()
            window.rb.applyVis()
            window.rb.applyPB()
            window.callbackFunction(`{handler}_~_${{window.rb.mode || 'none'}}`)
        }}
        bPb.addEventListener('click', () => setMode('pullback'))
        bBrk.addEventListener('click', () => setMode('breakout'))
        bRev.addEventListener('click', () => setMode('reversal'))
        bStats.addEventListener('click', () => {{
            window.rb.statsOn = !window.rb.statsOn
            try {{ window.rb.statsPanel.style.display = window.rb.statsOn ? 'block' : 'none' }}
            catch (e) {{}}
            paint()
        }})

        paint()
        window.rb.applyVis()
        window.rb.applyPB()
        window.containerDiv.appendChild(bar)
    }}""")


def _push_html(chart: Chart, html: str) -> None:
    chart.run_script(
        f"try {{ window.rb.statsPanel.innerHTML = {json.dumps(html)} }} catch (e) {{}}"
    )


# ------------------------------------------------------------------ app ----
def show_rbreaker_chart(
    symbols: list[str], default: str | None = None,
    start: str | None = None, end: str | None = None,
    params: RBreakerParams = RBreakerParams(),
    config: RBreakerConfig = RBreakerConfig(),
    rev_config: ReversalConfig = ReversalConfig(),
    session_mode: str = DEFAULT_SESSION_MODE,
    clean_dir: str = CLEAN_DIR,
    debug: bool = False,
    pb_params: PullbackParams = PullbackParams(),
    pb_config: PullbackConfig = PullbackConfig(),
    pb_warmup_days: int = PB_WARMUP_DAYS,
) -> None:
    """
    整屏一分钟 K 线 + R-Breaker 叠加层 + 中文统计面板, 外加一个可切换的
    **Archive-Pullback** 四格视图。

    三个维度互相正交:
      * ticker 搜索框         —— 品种
      * 它右边的时段选择器     —— 5 种时段划分
      * 右上角 Archive-Pullback / Breakout / Reversal —— 3 个策略

    只建一张 root 图 + 八个隐藏面板, 全部在 `show()` **之前**建好 (所有 run_script
    会被拼成一次 evaluate_js)。之后:
      * 切品种 / 切时段 -> `render` (跑两个 R-Breaker 回测 + 重设八条线,
        回调那半边按需惰性跑)
      * 切策略        -> `draw_overlays` (只换 markers 和面板文字)
      * MA / ATR      -> 纯 JS, 零往返
      * 额外注释      -> 一次往返 (markers 必须经 chart.marker_list)
    重建图表会丢掉缩放状态, 所以永远不重建。

    **回调那一半用自己的 `PreparedBase`** (`warmup_days=pb_warmup_days`), 与
    R-Breaker 那份分开: 它的趋势方向要日线 MA50, 得多留 60 个交易日的暖机段;
    把暖机段掺进 R-Breaker 的帧会给它的权益曲线接上一段零收益前缀, 夏普就变了。
    代价是一个品种读两次 parquet (实测 ~0.12s 一次)。
    """
    if session_mode not in SESSION_MODES:
        raise ValueError(f"未知的时段模式 {session_mode!r}; 可选: {list(SESSION_MODES)}")
    symbols = sorted(symbols)
    default = default or symbols[0]

    chart = build_minute_chart(title="v3.0 · R-Breaker 突破", debug=debug)

    # 七条叠加线, 建一次。全部踢出图例 —— 否则 OHLCV 那行下面会多堆七行,
    # 而这些线的值逐日恒定, 看图就知道, 不需要图例复述。
    series = {}
    for name in LINE_NAMES:
        color, style, width = LINE_SPEC[name]
        ln = chart.create_line(name, color=color, style=style, width=width,
                               price_line=False, price_label=False)
        # lineType 1 = WithSteps。日界处是垂直跳变而不是斜线。
        ln.run_script(f"""try {{
            {ln.id}.series.applyOptions({{lineType: 1, lastValueVisible: false,
                priceLineVisible: false, crosshairMarkerVisible: false}})
        }} catch (e) {{ console.log('line opts:', e) }}""")
        drop_legend_row(chart, ln)
        series[name] = ln

    # 两条止损线: 突破一条、反转一条。各自在 render() 里 set 一次, 可见性完全
    # 由 JS 的 applyVis 决定 —— 切模式零往返。
    # 刻意**不设** lineType: 见模块 docstring 第 (3) 条, 只有 Simple 下
    # "把每笔最后一个点涂透明"才能干净地藏掉笔与笔之间的连线。
    for tname in (TRAIL_NAME, TRAIL_REV_NAME):
        t = chart.create_line(tname, color=TRAIL_COLOR, style="solid", width=2,
                              price_line=False, price_label=False)
        t.run_script(f"""try {{
            {t.id}.series.applyOptions({{lastValueVisible: false,
                priceLineVisible: false, crosshairMarkerVisible: false}})
        }} catch (e) {{ console.log('trail opts:', e) }}""")
        drop_legend_row(chart, t)
        series[tname] = t

    # 贵的一半按**品种**缓存 (读 parquet + 清洗 + 复权 + 六条线, 与模式无关);
    # 便宜的一半按 **(品种, 时段模式)** 缓存 (分组 + 两个回测, 约 0.15s)。
    base_cache: dict[str, PreparedBase] = {}
    cache: dict[tuple[str, str], tuple[Prepared, RBreakerResult, RBreakerStats,
                                       RBreakerResult, RBreakerStats]] = {}
    # 回调那一半, 同样两层缓存, 但**惰性**: 用户不点 Archive-Pullback 就一次都不跑。
    pb_base_cache: dict[str, PreparedBase] = {}
    pb_cache: dict[tuple[str, str], tuple] = {}
    tiles = build_pullback_tiles(chart)

    view: dict = {"symbol": default, "session_mode": session_mode,
                  "mode": "breakout", "notes": False, "pb_shown": None}

    def _key() -> tuple[str, str]:
        return (str(view["symbol"]), str(view["session_mode"]))

    def _pullback():
        """(prep, tf, result, stats)，按 (品种, 时段模式) 缓存。窗口内无数据时 None。"""
        key = _key()
        if key not in pb_cache:
            sym, smode = key
            if sym not in pb_base_cache:
                pb_base_cache[sym] = prepare_base(
                    sym, start, end, params, config, clean_dir,
                    warmup_days=pb_warmup_days)
            prep = apply_mode(pb_base_cache[sym], smode)
            if prep.df.empty:
                return None
            tf = build_timeframes(prep.df)
            res = run_pullback_v3_backtest(prep, tf, pb_params, pb_config)
            pb_cache[key] = (prep, tf, res,
                             pb_compute_stats(res, prep.mode_label, pb_params))
        return pb_cache[key]

    def show_pullback() -> None:
        """
        四格视图的绘制。(品种, 时段模式) 没变时只重画 markers 和背景带 ——
        重推 K 线会把缩放状态清掉。
        """
        ctx = _pullback()
        if ctx is None:
            print(f"[pullback] {view['symbol']}: 窗口内无数据")
            return
        prep, tf, res, stats = ctx
        if view["pb_shown"] != _key():
            push_tiles(tiles, tf, res, prep.warmup_bars, view["notes"])
            view["pb_shown"] = _key()
        else:
            draw_pb_markers(tiles, tf, res, prep.warmup_bars, view["notes"])
        _push_html(chart, pb_stats_html(stats))
        print(pb_format_stats(stats))

    def draw_overlays() -> None:
        """便宜: 只换 markers 和面板文字。切策略时调, 不重跑回测也不重设线。"""
        mode = view.get("mode")
        if mode == "pullback":
            show_pullback()
            return
        if _key() not in cache:
            return
        _prep, brk_res, brk_stats, rev_res, rev_stats = cache[_key()]
        chart.clear_markers()
        # 两个都关时面板仍显示突破的结果(默认视图), 但不画 markers。
        result, stats = (rev_res, rev_stats) if mode == "reversal" else (brk_res, brk_stats)
        if mode in ("breakout", "reversal"):
            markers = build_markers(result)
            if markers:
                chart.marker_list(markers)
        _push_html(chart, stats_html(stats))

    def render(symbol: str | None = None, mode: str | None = None) -> None:
        """跑两个回测 + 重设八条线。切品种或切时段模式时调。"""
        if symbol is not None:
            view["symbol"] = symbol
        if mode is not None:
            view["session_mode"] = mode
        sym, smode = _key()
        if sym not in base_cache:
            base_cache[sym] = prepare_base(sym, start, end, params, config, clean_dir)
        if (sym, smode) not in cache:
            prep = apply_mode(base_cache[sym], smode)
            brk_res = run_rbreaker_backtest(prep, params, config)
            rev_res = run_rbreaker_reversal_backtest(prep, params, rev_config)
            cache[(sym, smode)] = (prep, brk_res, compute_stats(brk_res, "突破", prep.mode_label),
                                   rev_res, compute_stats(rev_res, "反转", prep.mode_label))
        prep, brk_res, brk_stats, rev_res, rev_stats = cache[(sym, smode)]
        symbol = sym
        if prep.df.empty:
            print(f"[rbreaker] {symbol}: 窗口内无数据")
            return

        chart.set(prep.ohlcv())
        chart.fit()
        # 八条全部重设 —— 否则上一个品种的数据会残留在没有新数据的那几条上。
        for name in LINE_NAMES:
            series[name].set(step_rows(prep, name))
        series[TRAIL_NAME].set(trail_rows(prep, brk_res, TRAIL_NAME))
        series[TRAIL_REV_NAME].set(trail_rows(prep, rev_res, TRAIL_REV_NAME))
        draw_overlays()

        span = f"{prep.df.index[0]:%Y-%m-%d} ~ {prep.df.index[-1]:%Y-%m-%d}"
        print(f"[rbreaker] {symbol}: {len(prep.df):,} bars  {span}"
              f"  dropped_runs={len(prep.dropped_runs)}")
        print(format_stats(brk_stats))
        print(format_stats(rev_stats))

    def on_strategy(mode: str) -> None:
        view["mode"] = None if mode == "none" else mode
        draw_overlays()

    def on_session_mode(mode: str) -> None:
        render(mode=mode)

    def on_notes(value: str) -> None:
        """「额外注释」: 回调确认圆点 + 冷静期背景带。只重画覆盖层, 不动 K 线。"""
        view["notes"] = value == "1"
        if view.get("mode") != "pullback":
            return
        ctx = _pullback()
        if ctx is not None:
            prep, tf, res, _ = ctx
            draw_pb_markers(tiles, tf, res, prep.warmup_bars, view["notes"])

    suffix = chart.id.rsplit(".", 1)[-1]
    strat_handler = f"rb_mode_{suffix}"
    sess_handler = f"rb_sessmode_{suffix}"
    notes_handler = f"rb_notes_{suffix}"
    chart.win.handlers[strat_handler] = on_strategy
    chart.win.handlers[sess_handler] = on_session_mode
    chart.win.handlers[notes_handler] = on_notes

    # 顺序要紧:
    #   build_ticker_search 先跑并声明顶层 const `box`;
    #   _bootstrap 建 window.rb (后面所有脚本都往它上面挂);
    #   _build_session_picker 靠 `box` 定位并把宽度存进 window.rb.searchBoxW;
    #   _build_extra_notes_button 再接在时段选择器右边;
    #   _build_top_right_bar 放最后, 它末尾会调 applyVis/applyPB 把初始状态刷上去。
    build_ticker_search(chart, symbols, default, render)
    _bootstrap(chart, session_mode)
    _register_series(chart, series)
    register_tiles(chart, tiles)          # 必须在 _bootstrap 之后 —— 它挂 window.rb
    wire_tile_crosshairs(chart, tiles)    # 同上
    _build_session_picker(chart, sess_handler, session_mode)
    _build_extra_notes_button(chart, notes_handler)
    _build_stats_panel(chart)
    _build_lines_toggle(chart)
    _build_top_right_bar(chart, strat_handler)

    render(default)
    chart.show(block=True)
