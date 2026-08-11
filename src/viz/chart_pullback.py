# -*- coding: utf-8 -*-
"""
v3.0 回调策略 (Archive-Pullback) 的图层: 2x2 四张 K 线 + 每张下面一个 ATR 副图。

右上角 `Archive-Pullback` 选中时, 整屏的那张 1m 图隐藏, 换成这八个面板:

    +-------------------+-------------------+
    |   1m  (MA5/20)    |   5m  (MA5/20)    |
    |   ATR             |   ATR             |
    +-------------------+-------------------+
    |  30m  (MA5/20)    |  1d (MA5/20/50)   |
    |   ATR             |   ATR             |
    +-------------------+-------------------+

MA 的配置**就是策略实际读的那些**: 1m/5m/30m 用快中两线 (MA5/MA20) 找时机,
日线用三线严格排列 (MA5/MA20/MA50) 定方向。

⚠ **四个 ATR 副图里只有 30m 那个进决策** (A6: 入场止损距离 + 吊灯都读最后一根
已收盘 30m 的 ATR(14))。1m/5m/1d 的 ATR 纯展示, 放在那里是为了让四张图版式一致、
也方便肉眼比较各周期的波动尺度。

---------------------------------------------------------------------------
为什么隐藏 root 另建四组面板, 而不是把 root 缩成 1m 那一格
---------------------------------------------------------------------------
复用 root 能省一次 1m 数据推送 (实测 8.5 万行 ~ 0.36s / 16MB, 见
`run_v3_rbreaker_chart.py:34-36`)。但代价是把这个视图和 R-Breaker 的五样内部状态
焊死: 八条线的可见性、成交量副图、被 `build_ticker_search` 和
`_build_session_picker` **两次**平移过的 OHLCV 图例 (它是 `{root.id}.div` 的子
节点, 会跟着 root 一起缩)、resize 往返、以及 `patch_legend_percent` 的十字线订阅。

0.36 秒是**一次性、有界、实测过**的代价, 且按 (品种, 时段模式) 缓存后再次切换免费;
那五处耦合是无界的。仓库自己就有先例: `_build_session_picker` 刻意不复用
`build_ticker_search`, 理由同构 (`chart_rbreaker.py:268-275`)。

---------------------------------------------------------------------------
lightweight-charts 的坑 (与 chart_rbreaker.py 抬头那三条同源)
---------------------------------------------------------------------------
* 所有 `run_script` 体在 `show()` 前会被拼成**一个**字符串送进单次 `evaluate_js`
  (abstract.py:54-58)。所以八个面板、所有 series、所有按钮都必须在 `show()`
  **之前**建好, 建完立刻 `display:none`; 首次激活时才 `set()` 数据。
* marker 必须自己按时间排序 —— `marker_list` (abstract.py:249-273) 不排序, 而
  `setMarkers` 对乱序输入会出错。
* K 线数据**不带 volume 列**: `AbstractChart.set` 一看到 volume 就会建成交量副图
  (abstract.py:558-563)。四分之一屏下面已经压了一条 ATR, 再加成交量带纯属噪音。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightweight_charts import Chart

from src.data.v3_timeframes import PB_TFS, TFBars
from src.engine.pullback_v3_backtest import (
    END_SESSION, PROTECTIVE_SL, SL, TP, PullbackResult,
)
from src.indicators import atr as atr_series
from src.performance.pullback_stats import (
    PullbackStats, panel_rows, panel_subtitle, panel_title,
)
from src.viz.legend_patch import patch_legend_percent
from src.viz.lwc_helpers import (
    UI_FONT_CJK, create_legend_div, drop_legend_row, invalidate_crosshair_snap, pin,
    price_dp, set_visible, sync_timescale_only, to_ns, wire_single_line_legend,
    wire_synced_crosshair,
)

# 策略实际读的均线组合, 一格一格对上。
MA_PERIODS: dict[str, tuple[int, ...]] = {
    "1m": (5, 20), "5m": (5, 20), "30m": (5, 20), "1d": (5, 20, 50),
}
MA_COLORS = {5: "#64b5f6", 20: "#ffb74d", 50: "#ba68c8"}      # chart.py:1544
TILE_LABEL = {"1m": "1 分钟", "5m": "5 分钟", "30m": "30 分钟", "1d": "日线"}

ENTRY_COLOR = "#2196F3"
# 离场按**原因**着色 (R-Breaker 那边是按盈亏符号着色 —— 两个视图刻意不同: 这里
# 一批可能有两条腿, 用颜色区分腿的性质比区分赚赔有用)。收盘平仓取蓝灰: 它既不是
# 策略认输也不是策略兑现, 只是交易所关门了, 不该染上红绿任何一边的语气。
# **这两个 dict 是无守卫下标** (markers() 里直接 REASON_COLOR[t.reason]) ——
# 引擎新增出场原因时漏了这里就是 KeyError。
REASON_COLOR = {SL: "#ef5350", TP: "#26a69a", PROTECTIVE_SL: "#7e57c2",
                END_SESSION: "#78909c"}
REASON_LABEL = {SL: "SL", TP: "TP", PROTECTIVE_SL: "PSL",     # chart.py:1551
                END_SESSION: "End Session"}                   # 与 R-Breaker 同名
PULLBACK_COLOR = "#fdd835"                                    # chart.py:1553
COOLDOWN_COLOR = "rgba(239, 83, 80, 0.13)"                    # chart.py:1554
ATR_COLOR = "#2196F3"
MUTED = "#8b93a7"

# 版面。丢掉组合层之后四张图吃满整屏 —— 每格半宽 (phaseF 是 0.25, 因为右半屏被
# 组合权益/饼图/统计表占着)。TOP 给顶部那排浮层 (ticker 搜索框、时段选择器、
# 额外注释、右上角按钮条) 留位置; 它和按钮的字号/内边距耦合, 所以收成一个常量。
TOP = 0.05
ROW_H = (1.0 - TOP) / 2          # 0.475
COL_W = 0.5
MAIN_FRAC = 0.72                 # 主图占一行的比例, 其余给 ATR
MAIN_H = ROW_H * MAIN_FRAC       # 0.342
ATR_H = ROW_H - MAIN_H           # 0.133
GRID = {"1m": (0, 0), "5m": (0, 1), "30m": (1, 0), "1d": (1, 1)}


@dataclass
class PullbackTile:
    tf: str
    main: object                      # AbstractChart
    atr_pane: object                  # AbstractChart
    ma_lines: dict[int, object] = field(default_factory=dict)
    atr_line: object = None
    cooldown: object = None

    @property
    def main_top(self) -> float:
        return TOP + GRID[self.tf][0] * ROW_H

    @property
    def left(self) -> float:
        return GRID[self.tf][1] * COL_W


# ------------------------------------------------------------- 建面板 ----
def build_pullback_tiles(root: Chart) -> dict[str, PullbackTile]:
    """
    八个面板 + 所有 series, **必须在 `root.show()` 之前**调 (见模块 docstring)。
    建完立刻全部隐藏; 数据由 `push_tiles` 在首次激活时才推。
    """
    tiles: dict[str, PullbackTile] = {}
    for tf in PB_TFS:
        row, col = GRID[tf]
        main_top = TOP + row * ROW_H
        left = col * COL_W

        main = root.create_subchart(position="left", width=COL_W, height=MAIN_H)
        main.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
        main.crosshair(mode="normal")
        main.legend(visible=True, font_size=11, font_family=UI_FONT_CJK)
        main.price_line(line_visible=False, title="")
        main.candle_style(
            up_color="#26a69a", down_color="#ef5350",
            wick_up_color="#26a69a", wick_down_color="#ef5350",
            border_up_color="#26a69a", border_down_color="#ef5350",
        )
        # 图例里的百分比改成"上一根 close -> 这一根 close" —— 否则任何跳空都看不见,
        # 而这四张图的跳空(时段边界)恰恰是要看的东西。
        patch_legend_percent(main)
        pin(main, main_top, left)

        ma_lines = {}
        for period in MA_PERIODS[tf]:
            ma_lines[period] = main.create_line(
                f"MA{period}", color=MA_COLORS[period], width=1,
                price_line=False, price_label=False)

        # 冷静期背景带: 独立价格轴 + 零边距, 常量 1.0 因此铺满整格高度 (TradingView
        # 的时段底色那种效果)。整段历史几百个区间, 用一条直方图而不是几百个
        # vertical_span —— 与 chart.py:2485-2499 同一个做法。
        cooldown = main.create_histogram(
            "cooldown", color=COOLDOWN_COLOR, price_line=False, price_label=False,
            scale_margin_top=0.0, scale_margin_bottom=0.0)
        drop_legend_row(main, cooldown)

        atr_pane = main.create_subchart(
            position="left", width=COL_W, height=ATR_H, sync=False)
        atr_pane.time_scale(time_visible=True, seconds_visible=False, border_visible=True)
        atr_pane.crosshair(mode="normal")
        atr_line = atr_pane.create_line("atr", color=ATR_COLOR, width=1,
                                        price_line=False, price_label=False)
        create_legend_div(atr_pane, "atrLegend", font_family=UI_FONT_CJK)
        pin(atr_pane, main_top + MAIN_H, left)
        sync_timescale_only(main, atr_pane)
        wire_single_line_legend(
            atr_pane, atr_line, f"{TILE_LABEL[tf]} ATR", "atrLegend", digits=3)

        # 建完即隐藏 —— 默认视图是 R-Breaker 那张整屏图。
        set_visible(main, False)
        set_visible(atr_pane, False)

        tiles[tf] = PullbackTile(tf=tf, main=main, atr_pane=atr_pane,
                                 ma_lines=ma_lines, atr_line=atr_line,
                                 cooldown=cooldown)
    return tiles


def register_tiles(root: Chart, tiles: dict[str, PullbackTile]) -> None:
    """
    把面板句柄挂到 `window.rb.pb.tiles` 上, 供纯 JS 的 MA / ATR 开关使用。

    **必须在 `_bootstrap` 之后调** —— 所有 run_script 体按调用顺序拼成一个 blob,
    `window.rb` 是 `_bootstrap` 建的, 早于它跑就只会静默进 catch, 于是四格永远
    不显示。所以这一步刻意从 `build_pullback_tiles` 里拆出来。

    整段裹在裸 `{ }` 块里并且只往 `window.rb` 上挂 —— 顶层 `const` 会跨脚本冲突。
    """
    entries = ", ".join(
        "{}: {{main: {}, atrPane: {}, mas: [{}], cooldown: {}}}".format(
            json.dumps(tf), t.main.id, t.atr_pane.id,  # type: ignore[attr-defined]
            ", ".join(f"{ln.id}" for ln in t.ma_lines.values()),  # type: ignore[attr-defined]
            t.cooldown.id,  # type: ignore[attr-defined]
        )
        for tf, t in tiles.items()
    )
    root.run_script(f"""{{
        try {{
            window.rb.pb.tiles = {{{entries}}}
            window.rb.pb.layout = {{
                top: {TOP}, rowH: {ROW_H}, mainH: {MAIN_H}, atrH: {ATR_H},
            }}
        }} catch (e) {{ console.log('pb register:', e) }}
    }}""")


def wire_tile_crosshairs(root: Chart, tiles: dict[str, PullbackTile]) -> None:
    """
    八个面板互相联动十字线: 悬停任意一格, 另外七格在**对应时刻**同时出现十字线,
    四个 OHLC 图例和四个 ATR 读数一起刷新。

    四周期对读正是这个视图存在的理由 —— 缺了它, 四张图只是四张孤立的图。

    与 `register_tiles` 一样必须在 `_bootstrap` 之后调 (要 `window.rb.pb` 已存在)。

    两个容易踩的点:
      * 副图传的是 `atr_line.series` 而不是 `atr_pane.series` —— 副图自动建的那个
        K 线 series 是空的 (`chart.py:1042-1046` 记着同一件事)。
      * snap 每个**周期**一本, 主图和它的 ATR 副图共用: 同一个 tile 上所有 series
        都由 `bars.index[start_bin:]` 切出, 时间轴逐点相同, 所以数组下标 == 逻辑
        下标, 两个面板可以共享同一本。
    """
    panes: list[dict] = []
    for tf, t in tiles.items():
        panes.append({
            "pane": t.main, "series_js": f"{t.main.id}.series", "group": tf,
            "legend": "native", "mas": list(t.ma_lines.values()),
        })
        panes.append({
            "pane": t.atr_pane, "series_js": f"{t.atr_line.id}.series",  # type: ignore[attr-defined]
            "group": tf, "legend": "div", "div_attr": "atrLegend",
            "label": f"{TILE_LABEL[tf]} ATR", "digits": 3,
        })
    wire_synced_crosshair(
        root, panes, ns="window.rb.pb",
        snap_sources={tf: f"{t.main.id}.series" for tf, t in tiles.items()},
    )


# --------------------------------------------------------------- 数据 ----
def tile_frame(tfb: TFBars, start_bin: int) -> pd.DataFrame:
    """
    某个周期的显示 K 线, 已切掉暖机段。

    **不带 volume 列** —— 见模块 docstring 第三条坑。
    """
    bars = tfb.bars.iloc[start_bin:]
    out = bars[["open", "high", "low", "close"]].copy()
    out.insert(0, "time", to_ns(pd.Series(bars.index)).to_numpy())
    return out.reset_index(drop=True)


def ma_frame(tfb: TFBars, period: int, start_bin: int) -> pd.DataFrame:
    """
    均线。**在全帧(含暖机段)上滚完再切显示段** —— 所以第一根可见 K 线上 MA50
    就已经是热的, 而不是从窗口起点重新暖机。这正是暖机 padding 的目的。
    """
    ma = tfb.bars["close"].rolling(period).mean()
    idx = tfb.bars.index[start_bin:]
    return pd.DataFrame({"time": to_ns(pd.Series(idx)).to_numpy(),
                         f"MA{period}": ma.to_numpy()[start_bin:]})


def atr_frame(tfb: TFBars, start_bin: int, period: int = 14) -> pd.DataFrame:
    """同理: ATR 在全帧上滚完再切, 首根可见 K 线的 ATR 已经是热的。"""
    a = atr_series(tfb.bars, period)
    idx = tfb.bars.index[start_bin:]
    return pd.DataFrame({"time": to_ns(pd.Series(idx)).to_numpy(),
                         "atr": np.asarray(a, dtype="float64")[start_bin:]})


def cooldown_frame(
    result: PullbackResult, tfb: TFBars, start_bin: int,
) -> pd.DataFrame | None:
    """
    冷静期背景带。某根合成 K 线只要**包含任意一根**处于冷静期的 1m bar 就整根涂色
    —— 与 `chart.py:1726` 的 `cooling.groupby(starts).max()` 同语义, 但用
    `np.bincount` 而不是 groupby。`None` 表示没什么可画。
    """
    if not result.cooldown_spans:
        return None
    n = len(tfb.bin_of)
    cooling = np.zeros(n, dtype=bool)
    for s, e in result.cooldown_spans:
        cooling[s:e + 1] = True
    if not cooling.any():
        return None
    per_bin = np.bincount(tfb.bin_of, weights=cooling.astype("float64"),
                          minlength=tfb.n_bars) > 0
    idx = tfb.bars.index[start_bin:]
    flags = per_bin[start_bin:]
    if not flags.any():
        return None
    out = pd.DataFrame({"time": to_ns(pd.Series(idx)).to_numpy(),
                        "cooldown": np.where(flags, 1.0, np.nan)})
    return out.dropna(subset=["cooldown"])


def markers(result: PullbackResult, tfb: TFBars, start_bin: int,
            notes: bool) -> list[dict]:
    """
    成交箭头 + (「额外注释」打开时) 回调确认圆点。

    1m 下标 -> 所属合成 K 线的**末端标签**, 靠 `bin_of` 一次映射。高周期上多个
    事件可能落在同一根 K 线上 —— 与老 phaseF 行为相同, 接受。

    文案:
        入场   `Long @ 3521.00` / `Short @ 3521.00`
        离场   `<原因>[(G)] <仓位%> <收益%>`, 例如 `TP 50% +4.00%`
               `End Session 50% +0.31%`, `SL(G) 100% -1.53%`

    两个百分比**含义不同**, 不要相加:
      * 仓位% = 这条腿占整批的比例 (`size_fraction`), 同一批各腿加起来是 100%
      * 收益% = **单位仓位**相对入场价的涨跌 (`pnl_net`), 与仓位比例无关 ——
        所以 `TP 50% +4.00%` 读作"平掉一半, 那个价位比入场价高 4%",
        它对整批权益的贡献是 0.5 x 4% = 2%。

    离场箭头按**出场原因**着色 (红止损 / 绿止盈 / 紫护盈 / 灰收盘), 而不是按盈亏 ——
    与 R-Breaker 那边刻意不同, 见 REASON_COLOR 上面那段。跳空成交加 `(G)` 后缀:
    那一笔是按 bar 的 open 成交的, 不是按止损线 (止盈和收盘平仓永远不会带 `(G)`)。

    2R 止盈那一根正好是时段末根时, 同一批会在**同一根**上产生 TP 和 End Session
    两个离场箭头。lightweight-charts 会把同时刻同侧的标记叠起来画; 高周期上本来
    就有多个事件挤一根的情况 (见上), 所以这里不做合并 —— 两条腿是真的各自成交过。
    """
    snap = tfb.bars.index.to_numpy()[tfb.bin_of]
    lo = tfb.bars.index[start_bin] if tfb.n_bars > start_bin else None
    out: list[dict] = []

    def add(bar_idx: int, **kw) -> None:
        t = snap[bar_idx]
        if lo is not None and t >= lo:
            out.append({"time": pd.Timestamp(t), **kw})

    seen: set[int] = set()
    for t in result.trades:
        is_long = t.direction == "long"
        if t.entry_bar_idx not in seen:
            seen.add(t.entry_bar_idx)
            add(t.entry_bar_idx,
                position="below" if is_long else "above",
                shape="arrow_up" if is_long else "arrow_down",
                color=ENTRY_COLOR,
                text=f"{'Long' if is_long else 'Short'} @ "
                     f"{t.entry_price:.{price_dp(t.entry_price)}f}")
        suffix = "(G)" if t.gapped else ""
        add(t.exit_bar_idx,
            position="above" if is_long else "below",
            shape="arrow_down" if is_long else "arrow_up",
            color=REASON_COLOR[t.reason],
            text=f"{REASON_LABEL[t.reason]}{suffix} "
                 f"{t.size_fraction:.0%} {t.pnl_net:+.2%}")

    if notes:
        for p in result.pullback_points:
            # 圆点而不是箭头 —— 回调**确认**永远不该被看成入场。触发(如果来的话)
            # 是更晚的一根。与 chart.py:1659-1669 同一个理由。
            add(p["bar_idx"], position="inside", shape="circle",
                color=PULLBACK_COLOR,
                text=f"Pullback {'L' if p['direction'] == 'long' else 'S'}")

    out.sort(key=lambda m: m["time"])
    return out


# --------------------------------------------------------------- 推送 ----
def push_tiles(
    tiles: dict[str, PullbackTile], tf: dict[str, TFBars],
    result: PullbackResult, warmup_bars: int, notes: bool,
) -> None:
    """四张图的全量重画。切品种 / 切时段模式时调。"""
    for name in PB_TFS:
        tile, tfb = tiles[name], tf[name]
        start = tfb.first_bin_at_or_after(warmup_bars)
        main = tile.main
        main.set(tile_frame(tfb, start))                      # type: ignore[attr-defined]
        for period, line in tile.ma_lines.items():
            line.set(ma_frame(tfb, period, start))            # type: ignore[attr-defined]
        tile.atr_line.set(atr_frame(tfb, start))              # type: ignore[attr-defined]
        tile.cooldown.set(cooldown_frame(result, tfb, start) if notes else None)  # type: ignore[attr-defined]
        main.fit()                                            # type: ignore[attr-defined]
        tile.atr_pane.fit()                                   # type: ignore[attr-defined]
    # 时间轴换过了 -> 作废十字线的 snap 缓存, 下次悬停重建。必须排在四张图
    # `set()` 之后 (脚本按调用顺序执行)。`draw_markers` 不动时间轴, 所以它里面
    # 不需要这一句。
    invalidate_crosshair_snap(tiles[PB_TFS[0]].main, "window.rb.pb")
    draw_markers(tiles, tf, result, warmup_bars, notes)


def draw_markers(
    tiles: dict[str, PullbackTile], tf: dict[str, TFBars],
    result: PullbackResult, warmup_bars: int, notes: bool,
) -> None:
    """只换 markers 和冷静期带 —— 切「额外注释」时调, 不动 K 线也不重置缩放。"""
    for name in PB_TFS:
        tile, tfb = tiles[name], tf[name]
        start = tfb.first_bin_at_or_after(warmup_bars)
        tile.main.clear_markers()                             # type: ignore[attr-defined]
        ms = markers(result, tfb, start, notes)
        if ms:
            tile.main.marker_list(ms)                         # type: ignore[attr-defined]
        tile.cooldown.set(cooldown_frame(result, tfb, start) if notes else None)  # type: ignore[attr-defined]


# --------------------------------------------------------------- 面板 ----
def stats_html(stats: PullbackStats) -> str:
    """与 `chart_rbreaker.stats_html` 同一版式, 好让三个策略的面板并排可比。"""
    rows = "".join(
        f'<div style="display:flex;justify-content:space-between;gap:22px">'
        f'<span style="color:{MUTED}">{k}</span><span>{v}</span></div>'
        for k, v in panel_rows(stats)
    )
    return (
        f'<div style="font-weight:600;font-size:14px">{panel_title(stats)}</div>'
        f'<div style="color:{MUTED};font-size:11px;margin:2px 0 8px">'
        f'{panel_subtitle(stats)}</div>{rows}'
    )
