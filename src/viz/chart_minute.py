# -*- coding: utf-8 -*-
"""
v3.0 一分钟 K 线浏览器。

整屏一张 K 线图, 无 MA / 无 ATR / 无成交量副图 —— 只有蜡烛。左上角浮一个
可搜索的 ticker 选择器 (phaseF `_build_ticker_search` 的增强版, 补了键盘
上下键 + Enter 选中)。

复权口径 (见 data/v3.0/data_quality_checklist.md D2/D8):
  源数据是**后复权**, 复权价 = close / K, 锚在上市首约, 历史值固定不变。
  直接画 close/K 的话纵轴会严重偏离真实价 (铁矿石是真实价的 8.6 倍), 所以
  展示层再乘一个常数 K0 = 所取窗口**第一根** bar 的 K:

      adj = close / K * K0

  乘常数不改变任何连续性, 窗口首根恰好落在真实价上, 且 K0 只依赖窗口起点 ——
  补新数据时 K0 不变, 存储层永不重算。**不要乘 K_last**, 那是前复权。
"""
from __future__ import annotations

import os
from typing import Callable

import pandas as pd
from lightweight_charts import Chart

from src.viz.legend_patch import patch_legend_percent
from src.viz.lwc_helpers import UI_FONT  # noqa: F401  (原来定义在这里, 调用点不动)

CLEAN_DIR = r"D:\2026_Summer\TradingApp\data\v3.0\clean_1m"


# ---------------------------------------------------------------- data ----
def load_minute(symbol: str, start: str | None = None, end: str | None = None,
                adjust: bool = True, clean_dir: str = CLEAN_DIR) -> pd.DataFrame:
    """
    读一个品种的分钟 parquet, 切窗口, 按需复权。

    返回列: time / open / high / low / close / volume —— 这是 lightweight-charts
    `set()` 的口径, 带上 volume 它就会自动画出成交量副图。

    注意 volume **不参与复权**: K 是价格调整因子, 手数是手数, 乘上去没有意义。

    复权公式和 `src.data.v3_sessions.adjust_prices` 是同一个 —— 那边给策略层
    用, 需要显式传 k0 (因为回测帧带着 padding, 不能拿 padding 首根当锚)。
    这里刻意不 import 过去: 纯浏览器不该为了三行算术就挂上整个 strategy 包。
    改动其中一个时记得同步另一个。
    """
    path = os.path.join(clean_dir, f"{symbol}.parquet")
    df = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume", "K"])
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end) + pd.Timedelta(days=1)]
    if df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    out = df[["open", "high", "low", "close"]].copy()
    if adjust:
        k = df["K"].to_numpy()
        k0 = k[0]                      # 窗口首根 -> 后复权, 增量安全
        out = out.div(k, axis=0).mul(k0)
    out["volume"] = df["volume"].to_numpy()
    out.insert(0, "time", df.index)
    return out.reset_index(drop=True)


# ------------------------------------------------------------ selector ----
def build_ticker_search(
    root: Chart, symbols: list[str], default: str, on_select: "Callable[[str], None]",
    top: float = 0.0, left: float = 0.0, legend_gap: int = 8,
) -> None:
    """
    phaseF `_build_ticker_search` 的增强版, 四点不同:

      1. 键盘上下键在候选里移动、Enter 选中、Esc 收起 (原版只能用鼠标点)。
      2. 容器 `pointerEvents: none`, 只有输入框和列表本身接收鼠标事件 ——
         否则这个浮层会挡住它覆盖区域内的 K 线拖拽/缩放。
      3. 候选列表按 DOM 顺序维护一个 activeIndex, 键盘移动时调用
         scrollIntoView, 所以长列表 (59 个品种) 也能滚到可见区。
      4. 建完之后把库自带的 OHLCV 图例挪到搜索框右侧 (间隔 `legend_gap` 像素),
         否则它默认的 top:10px/left:10px 正好和搜索框重叠 —— phaseF 里图例在
         各自的小图内部, 不存在这个冲突。
    """
    handler = f"ticker_pick_{root.id.rsplit('.', 1)[-1]}"
    root.win.handlers[handler] = on_select
    opts_js = "\n".join(
        f"""
        {{
            const opt = document.createElement('div')
            opt.innerText = {s!r}
            opt.dataset.sym = {s!r}
            opt.style.padding = '4px 10px'
            opt.style.cursor = 'pointer'
            opt.style.whiteSpace = 'nowrap'
            opt.addEventListener('mouseenter', () => setActive(opt))
            opt.addEventListener('mousedown', (e) => {{ e.preventDefault(); pick(opt) }})
            list.appendChild(opt)
        }}
        """
        for s in symbols
    )
    root.run_script(f"""
        const box = document.createElement('div')
        box.style.position = 'absolute'
        box.style.top = '{top * 100}%'
        box.style.left = '{left * 100}%'
        box.style.zIndex = '4000'
        box.style.padding = '8px 10px'
        box.style.fontFamily = '{UI_FONT}'
        box.style.pointerEvents = 'none'          // 不挡住底下的 K 线

        const input = document.createElement('input')
        input.value = {default!r}
        input.placeholder = 'Search ticker...'
        input.spellcheck = false
        input.style.width = '150px'
        input.style.padding = '5px 9px'
        input.style.background = 'rgba(30,34,45,0.92)'
        input.style.color = '#d1d4dc'
        input.style.border = '1px solid #2a2e39'
        input.style.borderRadius = '4px'
        input.style.fontFamily = '{UI_FONT}'
        input.style.fontSize = '13px'
        input.style.outline = 'none'
        input.style.pointerEvents = 'auto'
        box.appendChild(input)

        const list = document.createElement('div')
        list.style.position = 'absolute'
        list.style.top = 'calc(100% - 4px)'
        list.style.left = '10px'
        list.style.width = '150px'
        list.style.maxHeight = '260px'
        list.style.overflowY = 'auto'
        list.style.background = 'rgba(30,34,45,0.97)'
        list.style.border = '1px solid #2a2e39'
        list.style.borderRadius = '4px'
        list.style.color = '#d1d4dc'
        list.style.fontSize = '13px'
        list.style.display = 'none'
        list.style.zIndex = '4001'
        list.style.pointerEvents = 'auto'
        box.appendChild(list)

        let active = null
        function setActive(opt) {{
            if (active) active.style.background = ''
            active = opt
            if (active) {{
                active.style.background = '#2a2e39'
                active.scrollIntoView({{block: 'nearest'}})
            }}
        }}
        function visibleOpts() {{
            return Array.from(list.children).filter(o => o.style.display !== 'none')
        }}
        function pick(opt) {{
            if (!opt) return
            input.value = opt.dataset.sym
            list.style.display = 'none'
            input.blur()
            window.callbackFunction(`{handler}_~_${{opt.dataset.sym}}`)
        }}
        function openList(resetFilter) {{
            if (resetFilter) for (const o of list.children) o.style.display = ''
            list.style.display = 'block'
            const vis = visibleOpts()
            setActive(vis.find(o => o.dataset.sym === input.value) || vis[0] || null)
        }}

        {opts_js}

        input.addEventListener('focus', () => {{ input.select(); openList(true) }})
        input.addEventListener('input', () => {{
            const q = input.value.trim().toLowerCase()
            for (const o of list.children) {{
                o.style.display = o.dataset.sym.toLowerCase().includes(q) ? '' : 'none'
            }}
            list.style.display = 'block'
            setActive(visibleOpts()[0] || null)
        }})
        input.addEventListener('keydown', (e) => {{
            const vis = visibleOpts()
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {{
                e.preventDefault()
                if (list.style.display === 'none') {{ openList(true); return }}
                const i = vis.indexOf(active)
                const n = vis.length
                if (!n) return
                setActive(vis[(i + (e.key === 'ArrowDown' ? 1 : -1) + n) % n])
            }} else if (e.key === 'Enter') {{
                e.preventDefault()
                pick(active || vis[0])
            }} else if (e.key === 'Escape') {{
                list.style.display = 'none'
                input.blur()
            }}
        }})
        input.addEventListener('blur', () => {{ setTimeout(() => list.style.display = 'none', 120) }})

        window.containerDiv.appendChild(box)

        // 库自带的 OHLCV 图例是 .legend {{ position:absolute; top:10px; left:10px }}
        // (lightweight_charts/js/styles.css:217), 和这个搜索框重叠。
        //
        // 往下推是不行的: 候选列表展开时高达 260px, 图例照样被盖住。所以改为
        // 移到搜索框**右侧**, top 保持 10px 不动 —— 那个高度正好和输入框垂直
        // 居中对齐, 而且列表是朝下展开的, 右边永远不会被它覆盖。
        //
        // 用实测宽度而不是写死像素: 搜索框是 shrink-to-fit 的绝对定位块, 宽度
        // = 输入框宽 + 左右内边距, 以后改输入框宽度图例会自动跟着让位。行内样式
        // 优先级高于样式表, 库本身只动 display/maxWidth, 不会把 left 覆盖掉。
        try {{
            const legend = {root.id}.legend
            if (legend && legend.div) {{
                const w = box.getBoundingClientRect().width
                legend.div.style.left = (w + {legend_gap}) + 'px'
            }}
        }} catch (e) {{ console.log('legend offset:', e) }}
    """)


# --------------------------------------------------------------- chart ----
def build_minute_chart(title: str = "v3.0 · 1-minute candles", debug: bool = False) -> Chart:
    """
    建好一张配置完毕的分钟 K 线图 (深色蜡烛 + 底部成交量副图 + 收盘到收盘的
    图例百分比), 但**不塞数据、不建任何浮层**。

    抽出来是为了让纯浏览器 (`show_minute_candles`) 和 R-Breaker 叠加层
    (`src.viz.chart_rbreaker`) 用同一份图表配置 —— 两边的视觉一致是靠构造
    保证的, 不是靠两处各自维护一份参数。

    `debug=True` 会打开 webview 的 devtools, 调 JS 时用。
    """
    chart = Chart(title=title, toolbox=False,
                  width=1800, height=1000, maximize=True, debug=debug)
    chart.legend(visible=True, font_size=13, font_family=UI_FONT)
    chart.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
    chart.crosshair(mode="normal")
    chart.price_line(line_visible=False, title="")
    chart.candle_style(
        up_color="#26a69a", down_color="#ef5350",
        wick_up_color="#26a69a", wick_down_color="#ef5350",
        border_up_color="#26a69a", border_down_color="#ef5350",
    )
    # 成交量副图。scale_margin_top=0.82 把它压在底部约 18% 的高度里, 蜡烛仍占
    # 绝大部分屏幕; 颜色比蜡烛淡一档, 免得抢视觉重心。
    chart.volume_config(
        scale_margin_top=0.82, scale_margin_bottom=0.0,
        up_color="rgba(38,166,154,0.5)", down_color="rgba(239,83,80,0.5)",
    )
    # 图例里的百分比改成"上一根 close -> 这一根 close"。库自带的是
    # (close-open)/open, 即单根 bar 的实体涨跌 —— 那样任何跳空都看不见, 一根
    # 高开 3% 再回落 0.1% 的 bar 会显示 "-0.10 %"。
    patch_legend_percent(chart)
    return chart


def show_minute_candles(
    symbols: list[str], default: str | None = None,
    start: str | None = None, end: str | None = None,
    adjust: bool = True, clean_dir: str = CLEAN_DIR,
) -> None:
    """
    整屏一分钟 K 线 (含成交量副图) + 左上角可搜索 ticker 选择器。

    只建一张图, 切 ticker 时用 `chart.set()` 换数据而不是重建图表 —— 重建会
    丢掉缩放状态, 且 59 个品种各建一张的话 DOM 会爆。
    """
    symbols = sorted(symbols)
    default = default or symbols[0]

    chart = build_minute_chart()

    cache: dict[str, pd.DataFrame] = {}

    def render(symbol: str) -> None:
        if symbol not in cache:
            cache[symbol] = load_minute(symbol, start, end, adjust, clean_dir)
        df = cache[symbol]
        if df.empty:
            print(f"[chart] {symbol}: 窗口内无数据")
            return
        chart.set(df)
        chart.fit()
        span = f"{df['time'].iloc[0]:%Y-%m-%d} ~ {df['time'].iloc[-1]:%Y-%m-%d}"
        print(f"[chart] {symbol}: {len(df):,} bars  {span}")

    build_ticker_search(chart, symbols, default, render)
    render(default)
    chart.show(block=True)
