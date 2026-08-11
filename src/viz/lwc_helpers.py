# -*- coding: utf-8 -*-
"""
lightweight-charts 的通用小工具。

**这个模块必须保持无依赖**（只 import pandas / lightweight_charts），因为
v3.0 的图表要复用它而不能拖进整套 v2.1 回测栈。`src/viz/chart.py` 里同名的
私有函数保持原样不动 —— 那边 import 了 `engine`/`rules`/`data.resample`。
仓库对这个问题已有先例：`chart.py:392-394` 把 `_patch_legend_percent` 抽到
`legend_patch.py`，理由完全相同。
"""
from __future__ import annotations

import json

import pandas as pd

# 页面 body 默认用的 system-ui 栈。纯拉丁。
UI_FONT = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, '
    'Cantarell, "Helvetica Neue", sans-serif'
)

# 带中日韩字形的版本。"Segoe UI" 没有汉字, Chromium 会自己回退 —— 在
# Windows 上通常落到 SimSun(宋体), 13px 下衬线糊成一团。所以显式点名。
#
# 约束: **字体名里不能出现撇号**。这个字符串会被插进单引号的 JS 字符串
# 字面量 (`el.style.fontFamily = '...'`), 一个撇号就能把整批脚本打断。
UI_FONT_CJK = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", '
    '"Microsoft YaHei UI", "Microsoft YaHei", '        # Windows 11 zh-CN 默认无衬线
    '"PingFang SC", "Hiragino Sans GB", '              # macOS
    '"Noto Sans CJK SC", "Source Han Sans SC", '       # 开源兜底
    'Roboto, "Helvetica Neue", sans-serif'
)


def to_ns(time_col: pd.Series) -> pd.Series:
    """
    lightweight-charts 用 `astype('int64') // 10**9` 转 'time' 列, 隐含假设
    纳秒精度。pandas 现在默认把时间戳解析成 datetime64[us], 不强制转 ns 的话
    那个除法会差 1000 倍, 所有 bar 塌到 epoch 0 附近 (轴显示 1970, 蜡烛和
    marker 全叠在一起)。

    与 `src/viz/chart.py:44` 的 `_to_ns` 同一份逻辑。
    """
    return time_col.dt.as_unit("ns")


def wire_synced_crosshair(root, panes: list[dict], ns: str,
                          snap_sources: dict[str, str]) -> None:
    r"""
    一组面板之间的十字线广播: 悬停任何一个, 其余每个都在**对应时刻**画出十字线
    并刷新自己的图例。

    ---------------------------------------------------------------------
    对齐规则: 一次 lower-bound, 不需要任何映射表
    ---------------------------------------------------------------------
    前提是所有网格的标签口径统一为 **bar 末端**, 且粗周期的箱是细周期下标空间上
    一个**连续、稠密、非降**的划分 (`v3_timeframes.py` 的 `_assert_shape` 保证)。
    于是对任意两个网格 X、Y 和 X 上的任意标签 t:

        t 在 Y 上所属的那根 = Y_labels[lowerBound(Y_labels, t)]      (最小的 >= t)

    双向都成立, 不必区分细->粗还是粗->细。粗->细时落在那根粗 bar 的**最后一分钟**,
    这正是想要的。已在 RB/UR/SA/AG/CU/T/B 七个品种上逐 bar 与 `bin_of` 推出的真值
    比对过, 12 个有向配对零处不一致。

    这条规则替代了 `chart.py:989` 的 `_set_floor_map` —— 那边每个面板要推一本
    `{时刻 -> 时刻}` 字典。

    ---------------------------------------------------------------------
    三个必须照做的细节 (每一条都对应库里一个会炸的地方)
    ---------------------------------------------------------------------
    1. **图例走 byTime 分支, 不走 byIndex。** `chart.py:1108` 用的是
       `legendHandler({time}, true)`。那条分支在 bundle 里是

           if(e){ let i=ts.timeToCoordinate(t.time); i&&(o=ts.coordinateToLogical(i));
                  o&&(s=series.dataByIndex(o)) }
           ...
           this._lines.forEach(i => { s = e&&o ? ... : t.seriesData.get(i.series) })

       `timeToCoordinate` 返回 **0**(最左边那根) 或 `coordinateToLogical` 返回
       **0**(第一根) 都是 falsy, 于是 `o` 保持 null, 最后落到
       `t.seriesData.get(...)` —— 而 byIndex 调用传的 `t` 里根本没有 `seriesData`,
       直接 `TypeError`。主图有 MA 行 (`_lines` 非空) 时必炸。
       所以这里自己拼一个真的 `seriesData` Map, 走 `e=false` 那条路。

    2. **`setCrosshairPosition` 不触发 `subscribeCrosshairMove`。**
       bundle 里 `td(...)` 调 `Qc(l,r,null,n,!0)`, 而 `Qc` 的最后一个参数为真时
       **跳过** 派发 (`e||this.xc.m(...)`)。三个后果:
         * 不存在广播风暴, 不需要重入标志;
         * `wire_single_line_legend` 那类订阅**不会**自己跑 —— 所以副图图例必须
           在这里手写, 那不是优化而是必需;
         * `legend_patch.patch_legend_percent` 同样不会跑 —— 不补的话被动面板的
           百分比会退回库自带的 (close-open)/open, 和主动面板显示的口径不一致。
           所以下面把那条改写原样重放了一遍。

    3. **越界和空洞都要显式挡住。** `j >= 长度` 时 `Y[j]` 是 undefined,
       `setCrosshairPosition(v, undefined, s)` 会在库内部 `ensureNotNull` 抛错;
       `dataByIndex` 对 whitespace 行 (NaN 被 `util.js_data` 剔成只剩 time) 返回
       null。两者都 clear 掉那一格, 且**每个目标面板各自 try** —— 一格失败不许
       拖垮其余的。

    参数
    ----
    panes         每个面板一个 dict:
                  `pane` / `series_js` (**必须是有数据的那个 series** —— 副图自动
                  建的 K 线 series 是空的) / `group` (共用哪本 snap) /
                  `legend`: "native" | "div" | None /
                  `mas`: 主图上要一起刷的线 (native 用) /
                  `div_attr` `label` `digits` (div 用)
    ns            JS 命名空间表达式, 例如 `window.rb.pb`。必须已经存在。
    snap_sources  {group: 取 `.data()` 的 series JS 表达式}
    """
    src = ", ".join(f"{json.dumps(g)}: {js}" for g, js in snap_sources.items())
    entries = ", ".join(
        "{{group: {g}, id: {i}, chart: {i}.chart, series: {s}, legend: {lg},"
        " legendObj: {i}.legend, mas: [{mas}], div: {div}, label: {lb}, digits: {d}}}"
        .format(
            g=json.dumps(p["group"]), i=p["pane"].id, s=p["series_js"],
            lg=json.dumps(p.get("legend")),
            mas=", ".join(f"{ln.id}.series" for ln in p.get("mas", ())),
            div=(f'{p["pane"].id}.{p["div_attr"]}' if p.get("div_attr") else "null"),
            lb=json.dumps(p.get("label", "")), d=int(p.get("digits", 3)),
        )
        for p in panes
    )
    subs = "\n            ".join(
        f"{p['pane'].id}.chart.subscribeCrosshairMove(pp => "
        f"N.xhairQueue({p['pane'].id}, pp && pp.time !== undefined ? pp.time : null))"
        for p in panes
    )
    # 用占位符替换而不是 %% 或 f-string: 这段 JS 里有正则 `[+-]?\d+(?:\.\d+)? %`
    # 和 `+ ' %'`, `%` 格式化会当成转换符; 而 `{ }` 满天飞, f-string 要全部翻倍。
    js = r"""{
        try {
            const N = __NS__
            N.xhairSnapSrc = {__SRC__}
            N.xhairPanes = [__ENTRIES__]
            N.xhairSnap = null
            N.xhairFrame = 0
            // 最小的 j 使 a[j] >= t
            N.xhairLB = (a, t) => {
                let lo = 0, hi = a.length
                while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < t) lo = m + 1; else hi = m }
                return lo
            }
            // 从 series.data() 现取。零 Python 载荷, 而且结构上不可能与 series
            // 实际持有的数据不一致。1m 帧十几万根时这一趟也就几毫秒, 且只在
            // 数据换过之后的第一次悬停跑。
            N.xhairBuild = () => {
                const s = {}
                for (const g in N.xhairSnapSrc) {
                    const d = N.xhairSnapSrc[g].data()
                    const a = new Float64Array(d.length)
                    for (let i = 0; i < d.length; i++) a[i] = d[i].time
                    s[g] = a
                }
                N.xhairSnap = s
            }
            const clear = (q) => {
                q.chart.clearCrosshairPosition()
                if (q.legend === 'native') q.legendObj.legendHandler({}, true)
                else if (q.legend === 'div' && q.div) q.div.innerText = q.label
            }
            N.xhairApply = (srcId, t) => {
                if (!N.xhairSnap) N.xhairBuild()
                for (const q of N.xhairPanes) {
                    if (q.id === srcId) continue
                    try {
                        const Y = N.xhairSnap[q.group]
                        if (!Y || !Y.length) continue
                        if (t === null) { clear(q); continue }
                        const j = N.xhairLB(Y, t)
                        if (j >= Y.length) { clear(q); continue }
                        const bar = q.series.dataByIndex(j)
                        if (!bar) { clear(q); continue }
                        const v = (bar.close !== undefined) ? bar.close : bar.value
                        if (v === undefined || v === null) { clear(q); continue }
                        q.chart.setCrosshairPosition(v, Y[j], q.series)
                        if (q.legend === 'div') {
                            if (q.div) q.div.innerText = q.label + ': ' + v.toFixed(q.digits)
                            continue
                        }
                        if (q.legend !== 'native') continue
                        // 自建 seriesData -> 走 legendHandler 的 byTime 分支
                        const m = new Map()
                        m.set(q.series, bar)
                        for (const s of q.mas) { const b = s.dataByIndex(j); if (b) m.set(s, b) }
                        q.legendObj.legendHandler({time: Y[j], seriesData: m}, false)
                        // legend_patch 的收盘到收盘 % 不会自己跑, 这里重放一遍
                        const L = q.legendObj
                        if (L && L.percentEnabled && L.candle && bar.close !== undefined) {
                            const prev = q.series.dataByIndex(j - 1)
                            if (prev && prev.close) {
                                const pct = (bar.close - prev.close) / prev.close * 100
                                L.candle.innerHTML = L.candle.innerHTML.replace(
                                    /[+-]?\d+(?:\.\d+)? %/,
                                    (pct >= 0 ? '+' : '') + pct.toFixed(2) + ' %')
                            }
                        }
                    } catch (e) {}
                }
            }
            // rAF 合并: 高轮询率的鼠标一帧能来好几次, 而每次要刷 N-1 个面板的
            // innerHTML。顺带给整段找了个统一的 try 落点。
            N.xhairQueue = (srcPane, t) => {
                N.xhairSrcId = srcPane.id
                N.xhairT = t
                if (N.xhairFrame) return
                N.xhairFrame = requestAnimationFrame(() => {
                    N.xhairFrame = 0
                    try { N.xhairApply(N.xhairSrcId, N.xhairT) }
                    catch (e) { console.log('xhair apply:', e) }
                })
            }
            __SUBS__
        } catch (e) { console.log('wire_synced_crosshair:', e) }
    }"""
    root.run_script(js.replace("__NS__", ns).replace("__SRC__", src)
                    .replace("__ENTRIES__", entries).replace("__SUBS__", subs))


def invalidate_crosshair_snap(pane, ns: str) -> None:
    """
    数据换过之后作废 snap, 下一次悬停会重建。

    必须在 `series.set(...)` **之后**调 —— 所有 run_script 体按调用顺序拼成一个
    字符串执行。只换 markers 不换 K 线时不需要调 (时间轴没动)。
    """
    pane.run_script(f"{{ try {{ {ns}.xhairSnap = null }} catch (e) {{}} }}")


def price_dp(price: float) -> int:
    """
    marker 文案里价格该保留几位小数。

    国内商品期货的最小变动价位跨度极大 —— 螺纹 1 元/吨 (报价 3000+), 而白银
    1 元/千克、玉米淀粉 1 元/吨 (报价 2000+), 燃油 1 元/吨。三位小数对四位数的
    报价是噪音, 两位小数对两位数的报价又丢精度。以 100 为界是够用的近似。

    三个策略的 marker 共用这一条, 免得同一根 K 线上入场箭头和离场箭头的小数位
    不一样。
    """
    return 2 if abs(price) >= 100 else 3


def drop_legend_row(chart, series) -> None:
    """
    把一个辅助 series 从图表自带的 OHLCV 图例里摘掉。

    用途: 六条 R-Breaker 线 + 止损线如果都进图例, 会在 OHLCV 那行下面堆出
    七行, 把图顶部占掉一大片 —— 而这些线的值是逐日常数, 看图就知道, 不需要
    图例复述。

    整段包 try/catch: 这里摸的是库的内部结构 (`legend._lines`), 而所有
    `run_script` 体在 `show()` 前会被拼成**一个**字符串送进单次
    `evaluate_js`(abstract.py:54-58), 任何一处未捕获异常都会静默杀掉这批
    脚本的剩余部分, 表现为整个窗口变黑。
    """
    chart.run_script(f"""
        try {{
            const item = {chart.id}.legend._lines.find((l) => l.series == {series.id}.series)
            if (item) {{
                {chart.id}.legend._lines = {chart.id}.legend._lines.filter((l) => l != item)
                if (item.row && item.row.parentNode) {{
                    item.row.parentNode.removeChild(item.row)
                }}
            }}
        }} catch (e) {{ console.log('drop_legend_row:', e) }}
    """)


# ---------------------------------------------------------------- 版面 ----
def pin(pane, top_frac: float, left_frac: float) -> None:
    """
    把 `pane` 绝对定位到 (top_frac, left_frac)。宽高保持构造时传入的值 —— 那两个
    参数已经同时设好了 CSS 百分比和画布像素。

    与 `src/viz/chart.py:918` 的 `_pin` 同一份逻辑。整个多格版面都靠它: 库自带的
    布局是 float 流式的, 想要 2x2 的格子必须自己钉。
    """
    pane.run_script(f"""
        {pane.id}.wrapper.style.float = 'none'
        {pane.id}.wrapper.style.position = 'absolute'
        {pane.id}.wrapper.style.top = '{top_frac * 100}%'
        {pane.id}.wrapper.style.left = '{left_frac * 100}%'
    """)


def set_visible(pane, visible: bool) -> None:
    """显示/隐藏一个面板。隐藏时它的图例(是 `div` 的子节点)也跟着消失。"""
    show = "''" if visible else "'none'"
    resize = f"{pane.id}.reSize()" if visible else ""
    pane.run_script(f"""
        try {{
            {pane.id}.wrapper.style.display = {show}
            {resize}
        }} catch (e) {{ console.log('set_visible:', e) }}
    """)


def create_legend_div(pane, attr_name: str, font_size: int = 13,
                      font_family: str = UI_FONT) -> None:
    """
    在 `pane` 左上角挂一个空的绝对定位文字块, 存成 `{pane.id}.{attr_name}`。
    与 `src/viz/chart.py:398` 的 `_create_legend_div` 同一份逻辑。
    """
    pane.run_script(f"""
        try {{
            const el = document.createElement('div')
            el.style.cssText = 'position:absolute;top:4px;left:8px;z-index:3000;'
                + 'color:#d1d4dc;font-size:{font_size}px;pointer-events:none'
            el.style.fontFamily = {json.dumps(font_family)}
            {pane.id}.{attr_name} = el
            {pane.id}.div.appendChild(el)
        }} catch (e) {{ console.log('create_legend_div:', e) }}
    """)


def wire_single_line_legend(pane, line, label: str, attr_name: str,
                            digits: int = 2) -> None:
    """
    把 `{attr_name}` 那个文字块接上 `line` 的悬停值, 显示 `标签: 值`。

    读的是本图自己的 `param.seriesData` —— `line` 就长在 `pane` 上, 不存在跨图
    查值的问题。与 `src/viz/chart.py:1557` 的 `_wire_single_line_legend` 同构。
    """
    pane.run_script(f"""
        try {{
            {pane.id}.{attr_name}.innerText = {json.dumps(label)}
            {pane.id}.chart.subscribeCrosshairMove(param => {{
                const pt = param.time && param.seriesData.get({line.id}.series)
                {pane.id}.{attr_name}.innerText = pt
                    ? {json.dumps(label)} + ': ' + pt.value.toFixed({digits})
                    : {json.dumps(label)}
            }})
        }} catch (e) {{ console.log('wire_single_line_legend:', e) }}
    """)


def sync_timescale_only(chart_a, chart_b) -> None:
    """
    两张图之间**只**做可见区间(缩放/平移)的双向镜像, 不带库自带
    `create_subchart(sync=True)` 的十字线配对。

    与 `src/viz/chart.py:1003` 同一份逻辑, 那边的 docstring 记着为什么必须这样:
    原生配对会另跑一套十字线传播, 和自定义广播抢同一对父子图, **实测导致 ATR
    副图与它自己的主图之间十字线消失**, 而其它任何组合都正常。
    """
    for src, dst in ((chart_a, chart_b), (chart_b, chart_a)):
        src.run_script(f"""
            try {{
                {src.id}.chart.timeScale().subscribeVisibleLogicalRangeChange(range => {{
                    if (!range || {src.id}.rangeSyncing) return
                    {src.id}.rangeSyncing = true
                    {dst.id}.chart.timeScale().setVisibleLogicalRange(range)
                    {src.id}.rangeSyncing = false
                }})
            }} catch (e) {{ console.log('sync_timescale_only:', e) }}
        """)
