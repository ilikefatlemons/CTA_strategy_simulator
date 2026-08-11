# -*- coding: utf-8 -*-
"""
前端接线的**无窗口**冒烟测试。

lightweight-charts 的坑 (1): 所有 `run_script` 体在 `show()` 前会被拼成**一个**
字符串送进单次 `evaluate_js` (abstract.py:54-58)。后果是:

  * 任何一处语法错误 -> 整批脚本作废 -> 窗口直接变黑, 而且 Python 侧毫无提示;
  * 顶层 `const` 会跨脚本冲突, 同一个名字声明两次也是整批作废。

这两类错误没有任何 Python 测试能抓到 —— 所以这里把整批脚本原样取出来 (`Chart`
构造完但不 `show()`, 脚本都还躺在 `win.scripts` 里), 交给 `node --check` 过一遍
语法, 再检查几个关键符号确实被写进去了。

顺序也在这里钉: `window.rb` 由 `_bootstrap` 建, 而 `register_tiles` /
`_build_session_picker` / 各个按钮都往它上面挂 —— 谁早于它跑, 谁就静默进 catch,
表现为"按钮点了没反应"。
"""

import json
import os
import re
import shutil
import subprocess

import pytest

from src.data.v3_sessions import CLEAN_DIR

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CLEAN_DIR, "RB.parquet")),
    reason="需要 data/v3.0/clean_1m —— 先跑 python -m src.data.prepare_v3_minute",
)


@pytest.fixture(scope="module")
def win():
    """建好整个窗口但不 show()。脚本都还躺在 `win.scripts` 里, 回调也已注册。"""
    from lightweight_charts.chart import Chart

    from src.viz import chart_rbreaker

    captured: dict = {}
    original_show = Chart.show

    def fake_show(self, block: bool = False):     # noqa: ARG001
        captured["win"] = self.win

    Chart.show = fake_show                        # type: ignore[method-assign]
    try:
        chart_rbreaker.show_rbreaker_chart(
            ["RB"], default="RB", start="2025-07-01", end="2025-09-01")
    finally:
        Chart.show = original_show                # type: ignore[method-assign]
    return captured["win"]


@pytest.fixture(scope="module")
def blob(win) -> str:
    return "\n".join([*win.scripts, *win.final_scripts])


def test_the_whole_script_batch_is_valid_javascript(blob, tmp_path):
    """
    一处语法错误就会把整批脚本干掉, 窗口全黑。`node --check` 是唯一能在不开窗口的
    情况下抓到它的办法。
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("没有 node, 跳过 JS 语法检查")
    path = tmp_path / "blob.js"
    # 库自己的脚本里有 `_~_~RETURN~_~_` 这种非 JS 的哨兵前缀, 剔掉再检查。
    src = "\n".join(ln for ln in blob.splitlines() if "_~_~RETURN~_~_" not in ln)
    path.write_text(src, encoding="utf-8")
    proc = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[:4000]


def test_no_duplicate_top_level_const(blob):
    """
    顶层 `const` 跨脚本冲突同样会作废整批。`build_ticker_search` 已经占了
    `box`/`input`/`list`/`active`, 新代码必须裹在裸 `{ }` 块里。
    """
    names = re.findall(r"^\s{0,8}const\s+([A-Za-z_$][\w$]*)\s*=", blob, re.MULTILINE)
    dupes = {n for n in names if names.count(n) > 1}
    # 缩进 <= 8 的 const 未必真在顶层(嵌在 { } 里的也可能), 所以只对
    # build_ticker_search 已知占用的那几个名字较真。
    reserved = {"box", "input", "list"} & dupes
    assert not reserved, f"和 build_ticker_search 的顶层 const 重名: {reserved}"


def _cjk(label: str) -> str:
    """
    中文标签是用 `json.dumps` 写进脚本的, 默认 `ensure_ascii=True` -> 变成
    `\\u7b56\\u7565...`。所以搜索时也得用同一个编码, 不能拿原文去 find。
    """
    return json.dumps(label)


def test_archive_pullback_button_sits_between_stats_and_breakout(blob):
    """需求 1: 位置在「策略结果统计」和 Breakout 之间。"""
    i_stats = blob.index(f"mk({_cjk('策略结果统计')})")
    i_pb = blob.index("mk('Archive-Pullback')")
    i_brk = blob.index("mk('Breakout')")
    assert i_stats < i_pb < i_brk


@pytest.mark.parametrize("token", [
    "mk('Archive-Pullback')", _cjk("额外注释"), _cjk("指标线 六条"),
    "mk('MA', 14, maBox)", "mk('ATR', 14, maBox)",
    "window.rb.applyPB", "window.rb.layoutTopBar", "window.rb.pb.tiles",
    "window.rb.pb.btns", "window.rb.pb.paintBtns", "window.rb.maBox",
])
def test_key_symbols_are_present(blob, token):
    assert token in blob, f"{token} 没进脚本"


def test_ma_atr_box_is_a_separate_top_bar_node(blob):
    """
    需求 2 的机器版本: MA/ATR 不能再和「指标线 六条」共用搜索框下面那个盒子 ——
    那个位置 (y≈46px) 会压住 1m tile 的原生 OHLC 图例。

    三件事一起钉:
      * MA/ATR 挂在 maBox 上, 而 maBox 是 top:0% 的顶栏节点, 不是 46px 那个
      * maBox 参与 layoutTopBar 的横向链条
      * 按钮显式打开 pointer-events —— maBox 外框是 pointer-events:none,
        不打开的话按钮点不动 (静默失效, 屏幕上看不出来)
    """
    i_w = blob.index("const w = document.createElement('div')")
    i_ma = blob.index("const maBox = document.createElement('div')")
    seg = blob[i_ma:blob.index("const bLines")]
    assert "top:0%" in seg, "maBox 不在顶栏那一行"
    assert "'46px'" in blob[i_w:i_ma], "46px 那个盒子应该只剩「指标线 六条」"
    assert "pointer-events:auto" in blob[blob.index("const mk ="):i_ma + 4000]
    assert "[window.rb.notesBox, window.rb.maBox]" in blob, "maBox 没进 layoutTopBar"
    # 整盒开合, 不是逐按钮 —— 否则空盒子仍占着 layoutTopBar 的一节
    assert "window.rb.maBox.style.display = pb.on ? 'flex' : 'none'" in blob


# ------------------------------------------------- 跨 tile 十字线联动 ----
def test_all_eight_panes_broadcast_their_crosshair(blob):
    """八个面板 (四主图 + 四 ATR 副图) 都要订阅, 否则悬停某一格时其余不跟随。"""
    assert blob.count("N.xhairQueue(") == 8


def test_crosshair_uses_the_by_time_legend_branch_not_by_index(blob):
    """
    库的 `legendHandler(t, true)` (byIndex) 在 `timeToCoordinate` 返回 0 或
    `coordinateToLogical` 返回 0 时会掉进 `t.seriesData.get(...)`, 而 byIndex
    调用根本不传 seriesData —— 主图有 MA 行时必 TypeError。所以刷新必须走
    byTime 分支并自带一个真的 Map。

    只有**清空**那一路可以用 `legendHandler({}, true)`: 它在 `!t.time` 处就早退了。
    """
    apply = blob[blob.index("N.xhairApply ="):blob.index("N.xhairQueue =")]
    assert "legendHandler({time: Y[j], seriesData: m}, false)" in apply
    assert "legendHandler({}, true)" in blob          # clear 那一路
    assert "legendHandler({time: dstTime}, true)" not in blob
    assert "new Map()" in apply


def test_crosshair_guards_every_way_the_library_can_throw(blob):
    apply = blob[blob.index("N.xhairApply ="):blob.index("N.xhairQueue =")]
    assert "if (j >= Y.length)" in apply, "越界会让 setCrosshairPosition 在库内部抛错"
    assert "if (!bar)" in apply, "whitespace 行 dataByIndex 返回 null"
    assert "if (!Y || !Y.length)" in apply, "空面板 (窗口内无数据) 也会被悬停到"
    # 每个目标面板各自 try —— 一格失败不许拖垮其余七格
    assert apply.count("try {") >= 1 and apply.count("catch (e) {}") >= 1


def test_crosshair_replays_the_close_to_close_percent(blob):
    """
    `setCrosshairPosition` 不派发 `subscribeCrosshairMove`, 所以
    `legend_patch.patch_legend_percent` 那个订阅在被动面板上不会跑 —— 不补的话
    主动面板显示收盘到收盘 %, 另外七格显示库自带的 (close-open)/open。
    """
    apply = blob[blob.index("N.xhairApply ="):blob.index("N.xhairQueue =")]
    assert "percentEnabled" in apply
    assert "(bar.close - prev.close) / prev.close * 100" in apply
    assert r"/[+-]?\d+(?:\.\d+)? %/" in apply


def test_crosshair_snap_is_built_in_js_not_pushed(blob):
    """1m 帧可达 14 万根 —— 推 JSON 数组要 1.5 MB, 从 series.data() 现取是零载荷。"""
    assert ".data()" in blob and "Float64Array" in blob
    assert "N.xhairSnap = null" in blob


def test_crosshair_is_raf_coalesced(blob):
    assert "requestAnimationFrame" in blob
    assert "N.xhairFrame" in blob


def test_atr_panes_broadcast_their_line_series_not_the_empty_candle_series(blob):
    """副图自动建的 K 线 series 是空的 —— 传它的话 dataByIndex 永远返回 null。"""
    wire = blob[blob.index("N.xhairPanes = ["):blob.index("N.xhairSnap = null")]
    assert wire.count("legend: \"div\"") == 4
    # div 面板的 series 必须来自一个 line 句柄, 而不是面板自己的 .series
    for chunk in wire.split("legend: \"div\"")[1:]:
        pass
    assert wire.count("atrLegend") == 4


def test_bootstrap_runs_before_anything_touches_window_rb(blob):
    """
    `window.rb = {` 必须出现在所有 `window.rb.xxx = ` 赋值之前, 否则那些赋值
    静默进 catch, 按钮点了没反应而控制台只有一行 log。
    """
    i_boot = blob.index("window.rb = {")
    for token in ("window.rb.pb.tiles =", "window.rb.modeBox =",
                  "window.rb.notesBox =", "window.rb.pb.btns ="):
        assert blob.index(token) > i_boot, token


def test_three_way_mode_switch_is_wired(blob):
    for m in ("'pullback'", "'breakout'", "'reversal'"):
        assert f"setMode({m})" in blob


def test_ma_and_atr_toggles_are_pure_js(blob):
    """
    MA / ATR 不该产生 Python 往返 —— 它们的 click 里只有 applyPB, 没有
    callbackFunction。额外注释相反, 必须有 (markers 得走 chart.marker_list)。
    """
    ma_click = blob[blob.index("bMa.addEventListener"):blob.index("bAtr.addEventListener")]
    assert "callbackFunction" not in ma_click
    assert "applyPB" in ma_click
    notes = blob[blob.index(f"b.innerText = {_cjk('额外注释')}"):]
    assert "callbackFunction" in notes[:2000]


def test_eight_pullback_panes_are_created(blob):
    """
    四张主图 + 四个 ATR 副图, 全部在 show() 之前建好、pin 成绝对定位、立刻隐藏。
    尺寸也一并核对: 主图 0.5 x MAIN_H, ATR 0.5 x ATR_H。
    """
    from src.viz.chart_pullback import ATR_H, COL_W, MAIN_H

    handlers = re.findall(r'new Lib\.Handler\("[^"]+", ([\d.]+), ([\d.]+),', blob)
    assert len(handlers) == 9, "1 张 root + 8 个回调面板"
    dims = [(round(float(w), 6), round(float(h), 6)) for w, h in handlers[1:]]
    assert dims.count((round(COL_W, 6), round(MAIN_H, 6))) == 4
    assert dims.count((round(COL_W, 6), round(ATR_H, 6))) == 4
    assert blob.count("wrapper.style.display = 'none'") >= 8
    assert blob.count("wrapper.style.position = 'absolute'") >= 8


# ------------------------------------------- 回调路径的端到端(无窗口) ----
def _handler(win, prefix: str):
    from lightweight_charts.abstract import Window
    keys = [k for k in Window.handlers if k.startswith(prefix)]
    assert keys, f"没有注册 {prefix}* 回调"
    return Window.handlers[keys[-1]]


def _new_scripts(win, fn) -> str:
    """跑一段回调, 返回它新推进来的那些脚本。"""
    before = len(win.scripts)
    fn()
    return "\n".join(win.scripts[before:])


def test_clicking_archive_pullback_pushes_data_to_all_eight_panes(win):
    """
    模拟点一次 Archive-Pullback: Python 侧必须跑完回测并把 K 线/MA/ATR 推进四格。
    这一步把 `_pullback` -> `push_tiles` 整条链走通, 是 JS 语法检查覆盖不到的一半。
    """
    out = _new_scripts(win, lambda: _handler(win, "rb_mode_")("pullback"))
    assert out.count("setData(") >= 4 + 9 + 4, "K 线 4 + MA 9 + ATR 4 都该被推一遍"
    assert "statsPanel.innerHTML" in out
    assert json.dumps("回调入场 (Archive)")[1:-1] in out or "Archive" in out
    # 时间轴换了 -> 十字线的 snap 必须作废, 否则二分会查在旧的标签数组上。
    # 关键是它排在**四张主图**都 set 完之后 (snap 从 main.series.data() 取);
    # 后面 draw_markers 还会 set 四条冷静期直方图, 那个不影响时间轴。
    i_inv = out.index("window.rb.pb.xhairSnap = null")
    assert out[:i_inv].count("setData(") >= 4 + 9 + 4, \
        "作废排得太早 —— 主图还没 set 完, snap 会立刻按旧数据重建"


def test_extra_notes_toggle_pushes_markers_and_bands(win):
    """额外注释走一次 Python 往返: markers + 冷静期背景带, 但**不重推 K 线**。"""
    _handler(win, "rb_mode_")("pullback")
    on = _new_scripts(win, lambda: _handler(win, "rb_notes_")("1"))
    assert "setMarkers" in on, "回调确认圆点没画出来"
    assert on.count("setData(") <= 4, "只该重设 4 条冷静期直方图, 不该重推 K 线"
    off = _new_scripts(win, lambda: _handler(win, "rb_notes_")("0"))
    assert "setMarkers" in off


def test_switching_back_to_breakout_redraws_the_root(win):
    _handler(win, "rb_mode_")("pullback")
    out = _new_scripts(win, lambda: _handler(win, "rb_mode_")("breakout"))
    assert "setMarkers" in out and "statsPanel.innerHTML" in out


def test_session_mode_switch_works_while_pullback_is_active(win):
    """切时段时四格必须整体重画, 而不是拿着上一个模式的结果不放。"""
    _handler(win, "rb_mode_")("pullback")
    out = _new_scripts(win, lambda: _handler(win, "rb_sessmode_")("night"))
    assert out.count("setData(") >= 4 + 9 + 4
    _handler(win, "rb_sessmode_")("split")


def test_python_side_dispatch_covers_pullback():
    """
    `draw_overlays` 必须认得 'pullback' 这个 mode —— 否则点了按钮 JS 那边切了视图,
    Python 这边还在往 root 图上画 R-Breaker 的 markers。
    """
    import inspect

    from src.viz import chart_rbreaker
    src = inspect.getsource(chart_rbreaker.show_rbreaker_chart)
    assert 'mode == "pullback"' in src
    assert "show_pullback()" in src
    assert "pb_warmup_days" in src
