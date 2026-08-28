# -*- coding: utf-8 -*-
"""
lineA-03 图表层 —— `src/viz/chart_lineA_03.py`。

**不开窗口。** 分两半:

  前半  验推给 lightweight-charts 的那些表 (K 线帧 / MA / ATR / MACD / 冷静期 / marker)
  后半  把整批 `run_script` 取出来交给 `node --check`, 再检查关键符号与**顺序**

后半那一套照抄 `tests/test_pullback_v3_wiring.py`: 所有脚本在 `show()` 前被拼成
**一个**字符串送进单次 `evaluate_js` (`abstract.py:54-58`), 于是任何一处语法错误或
顶层 `const` 重名都会让整批作废、窗口变黑, 而 Python 侧毫无提示。没有别的办法能在不
开窗口的情况下抓到它。

按钮的视觉、版面、可见性仍然验不了 —— 那部分只能手工核对, 清单在计划的 §六。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.data.paths import CLEAN_DIR, CLEAN_DIR_HINT
from src.data.v3_sessions import derive_segments, prepare
from src.data.v3_timeframes import build_timeframes
from src.engine.lineA_03_backtest import 计算周期, 跑回测
from src.performance.lineA_03_stats import 算统计
from src.viz import chart_lineA_03 as C

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CLEAN_DIR, "AU.parquet")),
    reason=f"需要 {CLEAN_DIR_HINT} —— 先跑 python -m src.data.prepare_v3_minute",
)


@pytest.fixture(scope="module")
def 数据():
    prep = prepare("AU", start="2024-01-01", end="2026-07-29", warmup_days=60)
    tf = build_timeframes(prep.df, segments=derive_segments(prep.df), tfs=计算周期)
    结果 = 跑回测(prep, tf)
    return prep, tf, 结果


@pytest.fixture(scope="module")
def 参数(数据):
    return 数据[2].参数


@pytest.fixture(scope="module")
def 钟(数据):
    """驱动周期的 `TFBars` —— 引擎所有下标都活在它的下标空间里, 图层换算的基。"""
    _, tf, 结果 = 数据
    return tf[结果.参数.驱动周期]


# =========================================================== 表 =============
@pytest.mark.parametrize("名", C.周期集)
def test_K线帧切干净了暖机段(数据, 名, 钟):
    _, tf, 结果 = 数据
    tfb = tf[名]
    起 = C.起箱of(tfb, 结果, 钟)
    df = C.K线帧(tfb, 起)
    assert 0 < len(df) == tfb.n_bars - 起
    # 首根可见 K 线的**末端标签** >= 显示窗口首根 1m 的时刻
    assert tfb.bars.index[起] >= 数据[0].df.index[结果.暖机根数]


@pytest.mark.parametrize("名", C.周期集)
def test_K线帧不带volume列(数据, 名, 钟):
    """带上 volume 库会自动建成交量副图, 把三层版面挤扁。"""
    _, tf, 结果 = 数据
    起 = C.起箱of(tf[名], 结果, 钟)
    assert list(C.K线帧(tf[名], 起).columns) == ["time", "open", "high", "low", "close"]


@pytest.mark.parametrize("名", C.周期集)
def test_MA首根可见K线上就已经是热的(数据, 名, 钟):
    """在全帧(含暖机段)上滚完再切显示段 —— 否则首根 MA55 是 NaN。"""
    _, tf, 结果 = 数据
    起 = C.起箱of(tf[名], 结果, 钟)
    for 期 in C.MA周期:
        df = C.MA帧(tf[名], 期, 起)
        assert len(df) == tf[名].n_bars - 起
        assert not pd.isna(df[f"MA{期}"].iloc[0]), f"{名} MA{期} 首根是 NaN"


@pytest.mark.parametrize("名", C.周期集)
def test_ATR与MACD与K线逐元素对齐(数据, 名, 参数, 钟):
    _, tf, 结果 = 数据
    起 = C.起箱of(tf[名], 结果, 钟)
    k = C.K线帧(tf[名], 起)
    a = C.ATR帧(tf[名], 起, 参数.ATR周期)
    d, e, h = C.MACD帧(tf[名], 起, 参数.MACD快, 参数.MACD慢, 参数.MACD信号)
    for df in (a, d, e, h):
        assert np.array_equal(df["time"].to_numpy(), k["time"].to_numpy())
    assert not pd.isna(a["atr"].iloc[0])


@pytest.mark.parametrize("名", C.周期集)
def test_MACD柱逐点着色(数据, 名, 参数, 钟):
    """不逐点着色的话会有点渲染成黑色 (`chart.py:252-257` 记着)。"""
    _, tf, 结果 = 数据
    起 = C.起箱of(tf[名], 结果, 钟)
    _, _, h = C.MACD帧(tf[名], 起, 参数.MACD快, 参数.MACD慢, 参数.MACD信号)
    assert "color" in h.columns
    assert set(h["color"].unique()) <= {C.柱正, C.柱负}
    assert (h.loc[h["柱"] >= 0, "color"] == C.柱正).all()
    assert (h.loc[h["柱"] < 0, "color"] == C.柱负).all()


def test_冷静期涂的箱与区间独立重算的结果一致(数据, 钟):
    _, tf, 结果 = 数据
    tfb = tf["2h"]
    起 = C.起箱of(tfb, 结果, 钟)
    df = C.冷静期帧(结果, tfb, 起, 钟)
    assert df is not None and (df["冷静期"] == 1.0).all()
    # 冷静期区间是 **15m** 下标的闭区间, 先换回 1m 再问 2h 的箱号
    首, 末 = C.驱动到1m(钟)
    应涂 = set()
    for a, b in 结果.冷静期区间:
        段 = tfb.bin_of[int(首[a]):int(末[b]) + 1]
        应涂 |= {int(x) for x in np.unique(段) if x >= 起}
    实涂 = {int(np.flatnonzero(tfb.bars.index == t)[0]) for t in df["time"]}
    assert 实涂 == 应涂


def test_没有冷静期时返回None(数据, 钟):
    import dataclasses
    _, tf, 结果 = 数据
    空 = dataclasses.replace(结果, 冷静期区间=[])
    assert C.冷静期帧(空, tf["2h"], 0, 钟) is None


# ========================================================= 标记 =============
@pytest.mark.parametrize("名", C.周期集)
def test_标记已排序且都落在真实K线上(数据, 名, 钟):
    _, tf, 结果 = 数据
    起 = C.起箱of(tf[名], 结果, 钟)
    ms = C.标记(结果, tf[名], 起, 钟, 回调=True, 反转=True, 开关=True)
    assert ms
    时 = [m["time"] for m in ms]
    assert 时 == sorted(时), "marker_list 不排序, setMarkers 收到乱序会直接报错"
    合法 = set(tf[名].bars.index[起:])
    assert all(t in 合法 for t in 时)


def test_默认只有入场与出场箭头(数据, 钟):
    """**默认全关** —— 这是这个窗口的硬要求。"""
    _, tf, 结果 = 数据
    起 = C.起箱of(钟, 结果, 钟)
    ms = C.标记(结果, 钟, 起, 钟)
    assert ms
    assert {m["shape"] for m in ms} <= {"arrow_up", "arrow_down"}
    assert not any(m["shape"] in ("circle", "square") for m in ms)


def test_三个标注开关各自只加自己那一类(数据, 钟):
    _, tf, 结果 = 数据
    tfb, 起 = 钟, C.起箱of(钟, 结果, 钟)
    基 = C.标记(结果, tfb, 起, 钟)
    for 键, 色 in (("回调", C.回调色), ("反转", C.反转色)):
        开 = C.标记(结果, tfb, 起, 钟, **{键: True})
        加 = [m for m in 开 if m.get("color") == 色]
        assert 加, f"{键} 打开后一个标记都没加"
        assert len(开) == len(基) + len(加)
    开开关 = C.标记(结果, tfb, 起, 钟, 开关=True)
    锁标 = [m for m in 开开关 if m.get("color") in (C.开关1色, C.开关2色)]
    assert 锁标 and len(开开关) == len(基) + len(锁标)
    # 两个锁必须用不同颜色, 否则「双重条件」看不出是哪一个先开
    assert {m["color"] for m in 锁标} == {C.开关1色, C.开关2色}


def test_出场箭头按结果着色而不是只按理由(数据, 钟):
    """
    吊灯是追踪止损, 可盈可亏。绿/琥珀分开染, 才不会被「绿色=止盈」骗过去。
    """
    _, tf, 结果 = 数据
    起 = C.起箱of(钟, 结果, 钟)
    ms = C.标记(结果, 钟, 起, 钟)
    色 = {m["color"] for m in ms}
    assert C.止损色 in 色 and C.吊灯盈色 in 色 and C.吊灯亏色 in 色, (
        f"三种出场配色没都出现: {色}")
    文 = [m["text"] for m in ms]
    assert any(t.startswith("吊灯亏") for t in 文)


def test_标记不伸进暖机段(数据, 钟):
    _, tf, 结果 = 数据
    for 名 in C.周期集:
        起 = C.起箱of(tf[名], 结果, 钟)
        ms = C.标记(结果, tf[名], 起, 钟, 回调=True, 反转=True, 开关=True)
        assert all(m["time"] >= tf[名].bars.index[起] for m in ms)


# ========================================================= 版面 =============
def test_版面分数铺满屏幕():
    assert C.TOP + C.ROWS * C.ROW_H == pytest.approx(1.0)
    assert C.COLS * C.COL_W == pytest.approx(1.0)
    assert C.ATR_H + C.MACD_H < C.ROW_H, "两个副图占满了整行, 主图没地方了"
    assert 0 < C.TOP < 0.15


def test_四个格位恰好铺满网格且不重叠():
    """
    格子绑的是**位置**不是周期。四个格位必须恰好填满 ROWS x COLS —— 少一格屏幕留洞,
    重一格两张图永久叠在一起 (换周期只重推数据, 不重建图表对象)。
    """
    满 = {(r, c) for r in range(C.ROWS) for c in range(C.COLS)}
    assert set(C.格坐标) == set(C.格位), "格坐标与格位对不上"
    坐标 = [C.格坐标[位] for 位 in C.格位]
    assert len(坐标) == len(满) == len(C.格位)
    assert set(坐标) == 满, f"格位坐标不铺满: {sorted(坐标)}"
    assert len(set(坐标)) == len(坐标), "有两个格位落在同一个坑位"


def test_默认周期合法且四格可重复():
    assert set(C.默认格位周期) == set(C.格位), "默认周期没给全每个格位"
    for 位, tf in C.默认格位周期.items():
        assert tf in C.周期集, f"{位} 的默认周期 {tf} 不在下拉框里"
    # 允许重复是这套模型的目的, 所以**不许**断言四格互不相同


def test_面板html带着三条声明(数据, 钟):
    _, tf, 结果 = 数据
    每日 = tf["2h"].n_bars / 数据[0].df["trading_date"].nunique()
    html = C.面板html(算统计(结果, 每日))
    # 「交易日等分」这一条: 2026-08-26 前是「箱长 30~120 分钟不等 … 等权平均」,
    # 改成交易日等分之后箱长齐了, 要声明的换成「末根是余数 + 那一根横跨夜→日断口」。
    for 片 in ("可交易日", "累计净收益", "离场 固定止损/吊灯", "毛", "涨跌停",
              "交易日等分", "横跨夜→日断口"):
        assert 片 in html, f"面板里缺「{片}」"


# ==================================================== 接线 (node --check) ===
@pytest.fixture(scope="module")
def win(数据):
    """建好整个窗口但不 show()。脚本都还躺在 `win.scripts` 里, 回调也已注册。"""
    from lightweight_charts.chart import Chart

    prep, tf, 结果 = 数据
    每日 = tf["2h"].n_bars / prep.df["trading_date"].nunique()
    统 = 算统计(结果, 每日)

    抓: dict = {}
    原 = Chart.show
    Chart.show = lambda self, block=False: 抓.update(win=self.win)   # noqa: ARG005
    try:
        C.show_lineA_03(["AU"], lambda _s: (tf, 结果, 统, 结果.参数), default="AU")
    finally:
        Chart.show = 原
    return 抓["win"]


@pytest.fixture(scope="module")
def blob(win) -> str:
    return "\n".join([*win.scripts, *win.final_scripts])


def test_整批脚本是合法的javascript(blob, tmp_path):
    """一处语法错误就把整批脚本干掉、窗口全黑, 而 Python 侧毫无提示。"""
    node = shutil.which("node")
    if node is None:
        pytest.skip("没有 node, 跳过 JS 语法检查")
    路径 = tmp_path / "blob.js"
    # 库自己的脚本里有 `_~_~RETURN~_~_` 这种非 JS 的哨兵前缀, 剔掉再检查
    路径.write_text(
        "\n".join(l for l in blob.splitlines() if "_~_~RETURN~_~_" not in l),
        encoding="utf-8")
    r = subprocess.run([node, "--check", str(路径)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:4000]


def test_没有和ticker搜索框重名的顶层const(blob):
    """`build_ticker_search` 占了 `box`/`input`/`list`/`active`, 重名会作废整批。"""
    名 = re.findall(r"^\s{0,8}const\s+([A-Za-z_$][\w$]*)\s*=", blob, re.MULTILINE)
    重 = {n for n in 名 if 名.count(n) > 1}
    assert not ({"box", "input", "list", "active"} & 重)


def test_建骨架排在任何东西碰window_la3之前(blob):
    """破了这条, 后面所有脚本静默进 catch —— 四格不显示、按钮点了没反应, 且不报错。"""
    建 = blob.find("window.la3 = {")
    首 = blob.find("window.la3")
    assert 建 >= 0 and 建 == 首, f"有脚本早于 建骨架 就碰了 window.la3: {blob[首:首+160]!r}"


def test_pill与下拉框都建了(blob):
    """中文标签是 json.dumps 写进脚本的 (ensure_ascii -> \\uXXXX), 要用同样的编码去找。"""
    for 标 in ("统计", "回调", "大周期反转", "冷静期", "入场双重条件", "止损/吊灯线(1m)"):
        assert json.dumps(标) in blob, 标
    for 标 in ("'MA'", "'ATR'", "'MACD'"):
        assert 标 in blob, 标
    for 位 in C.格位:
        assert json.dumps(位) in blob, f"下拉框少了格位 {位}"
    for tf in C.周期集:
        assert f"<option value='{tf}'>" in blob, f"下拉框选项少了 {tf}"


def test_换周期的接线都在(blob):
    """
    换周期只重推数据、**不重建图表对象**(重建会丢缩放状态)。四件事缺一不可:

      1. 四个下拉框各带一次 Python 往返 —— 换周期要换整套数据, 纯 JS 做不到
      2. 瓦片带 tf / atrLabel / macdLabel / hasStop, 且这些是**可变**的
      3. 图例标签在渲染时**重读** window.la3, 不是烤进 JS 常量 —— 否则换周期后
         标签会停在旧周期上
      4. applyRange 存在 —— 新推的数据会把时间轴弹回全区间, 要补一次
    """
    assert blob.count("addEventListener('change'") >= 1, "下拉框没接 change"
    assert "callbackFunction" in blob
    for 键 in ("tf:", "atrLabel:", "macdLabel:", "hasStop:"):
        assert 键 in blob, f"瓦片没导出 {键}"
    assert ".atrLabel || ''" in blob, "ATR 图例的标签不是运行时重读的"
    assert ".macdLabel || ''" in blob, "MACD 图例的标签不是运行时重读的"
    assert "window.la3.applyRange" in blob
    # 止损图例四格都建, 靠 hasStop 决定谁显示
    assert "window.la3.止损线 && t.hasStop" in blob, "止损图例没按 hasStop 过滤"


def test_格位坐标是从Python写死进JS的(blob):
    """四格位置固定不变, JS 侧只读不算 —— 位置的真源在 Python 的 `格坐标`。"""
    assert "top:" in blob and "left:" in blob
    assert "wrapper.style.left = x + '%'" in blob
    assert blob.count("wrapper.style.left = x + '%'") >= 3, "三层都要摆 left"


def test_三层版面的重排逻辑在(blob):
    """关掉 ATR 时 MACD 必须**上移**, 不能只改高度 —— 否则中间留一个空洞。"""
    assert "applyPanes" in blob
    assert "atrPane.wrapper.style.top" in blob
    assert "macdPane.wrapper.style.top" in blob
    assert "scale.height" in blob and "reSize()" in blob


def test_标注开关合成一次往返(blob):
    """三个 flag 用 `;;;` 拼一起, `parse_event_message` 按它拆成位置参数。"""
    assert ";;;" in blob
    assert "callbackFunction" in blob



def test_三层的时间轴是同步的(blob):
    """
    缩放/平移主图时 ATR 与 MACD 必须跟着走, 否则上下三层各走各的、对不齐。

    走 `sync_timescale_only` 的双向镜像, **不用**库原生 `create_subchart(sync=True)`:
    原生配对会另跑一套十字线传播, 和自建广播抢同一对父子图, 实测导致副图与它自己的
    主图之间十字线消失 (`lwc_helpers.py:344-350`)。
    """
    n = blob.count("subscribeVisibleLogicalRangeChange")
    # 每格两对 (主↔ATR, 主↔MACD), 每对双向 = 4 次订阅。**四个格位**, 与能选的周期
    # 数无关 —— 格子绑位置不绑周期。
    期望 = len(C.格位) * 2 * 2
    assert n == 期望, f"时间轴同步的订阅数是 {n}, 期望 {期望}"
    assert "rangeSyncing" in blob, "少了重入守卫, 两张图会互相触发到爆栈"


def test_MACD面板不叫DIF(blob):
    """
    面板上有三条东西 (DIF / DEA / 柱), 拿其中一条的名字当面板名会误导; 而在通达信
    一系的口径里「MACD」指的还是柱, 更容易读错。所以面板叫 MACD, 三个值分开列。
    """
    # 图例标签是**动态**的, blob 里只会出现四个默认周期的那几个
    for 格 in [C.格标签[C.默认格位周期[位]] for 位 in C.格位]:
        assert json.dumps(f"{格} MACD") in blob, f"{格} 的 MACD 面板没有叫 MACD"
        assert json.dumps(f"{格} DIF") not in blob, f"{格} 的面板还标着 DIF"
    # 三个读数都在
    for 名 in ("DIF", "DEA", "柱"):
        assert json.dumps(名) in blob or f"'  {名} '" in blob, 名


def test_MACD三条series都在可见的right轴上(blob):
    """
    lightweight-charts 只渲染 `left` / `right` 两条边轴; 自定义 `priceScaleId` 是
    **overlay 轴**, 既不画刻度也不能拖。而库里 `Histogram.__init__` 把
    `priceScaleId` 写死成它自己的 id。

    于是有三种状态, 只有第三种是对的:
      * 什么都不做 -> 线在 right、柱在 overlay, 拖右边栏只动线不动柱
      * 让线去就柱 -> 三者同轴了, 但 right 上没 series, **整条纵轴消失**(踩过)
      * 让柱来就线 -> 对。柱 applyOptions 迁到 'right', 线用默认轴

    这条同时钉住「一起缩放」和「纵轴还在、能拖」两件事。
    """
    n = len(C.格位)
    # 每格一个柱, 每个柱后面都跟一句迁轴
    assert blob.count('"柱"') >= n
    assert blob.count("priceScaleId: 'right'") == n, "柱没被迁到 right 轴, 或次数不对"
    # 线一律用默认轴 —— 不许出现指向 overlay 的 priceScaleId
    assert not re.search(r'createLineSeries\(.*?priceScaleId:\s*"window\.', blob, re.S), \
        "有 line 被挂到了 overlay 轴上, 那条轴不画刻度也不能拖"
    # 边距设在 right 上
    assert blob.count("priceScale('right').applyOptions") == n
    # 直方图默认是 volume 格式, 同轴后必须改成价格格式
    assert blob.count("type: 'price', precision: 3") == n


def test_MACD的十字线保留水平线(blob):
    """
    悬停时要能读出 MACD 的纵轴位置, 跟 ATR 一样。v1.1 那份关掉水平线是因为它用了
    magnet 模式(磁吸会把横线吸到 DIF/DEA 而不是柱), normal 模式没这个问题。
    """
    关掉的 = re.findall(r"horzLine:\s*\{[^}]*visible:\s*false", blob)
    assert not 关掉的, f"有 {len(关掉的)} 处把十字线的水平线关掉了"


def test_十字线广播会跳过鼠标所在那一格(blob):
    """
    鼠标所在的那一格已经有自己的**原生**十字线在跟着光标走。广播如果再往它身上
    `setCrosshairPosition` 一条钉在 bar close 上, 两条会互相打架、肉眼可见地频闪,
    而且两条的纵轴位置不同(一条在光标处, 一条在收盘价处)。

    跳过靠 `q.id === srcId`, 而这一句要成立, **两边必须是同一种东西**:
      * `q.id`   取 Handler 的 `id` 字段 —— 字符串
      * `srcId`  来自 `srcPane.id`      —— 同一个字符串
    曾经 `q.id` 写的是 `{pane.id}`, 那是 JS **对象**, 与字符串永远不相等, 于是这句
    从来没生效过。这条测试就是防它退回去。
    """
    ids = re.findall(r"\{group: [^}]*?id: ([^,]+), chart:", blob)
    assert ids, "没找到任何十字线面板条目"
    坏 = [x for x in ids if not x.endswith(".id")]
    assert not 坏, f"这些条目的 id 是对象而不是字符串, 跳过源面板会失效: {坏[:3]}"
    assert "N.xhairSrcId = srcPane.id" in blob
    # 面板条目 = 四格 x 四条: 主图(native 图例) / ATR / MACD / 主图的止损读数(div 图例)。
    # **主图占两条是刻意的**: 一条走库自带的 OHLC+MA 图例, 一条走我们自己的
    # `stopLegend` div。同一张图挂两次 `subscribeCrosshairMove` 是幂等的 (同一个值、
    # 同一个时刻、同一条 series), 而广播那边靠 `q.id === srcId` 把两条一起跳过。
    # 每格四条: 主图原生图例 / 主图止损读数 / ATR / MACD。**止损那条四格都挂**,
    # 但只有当前显示驱动周期的那一格会显示 (applyPanes 读 hasStop)。
    期望 = len(C.格位) * 4
    assert len(ids) == 期望, f"面板数是 {len(ids)}, 期望 {期望}"

def test_python侧handler注册了(win):
    名 = [k for k in win.handlers if k.startswith("la3_notes_")]
    assert len(名) == 1, 名


def test_根图被隐藏(blob):
    """根图只做容器 (搜索框与面板挂 containerDiv), 真正要看的是四格。"""
    assert re.search(r"window\.\w+\.wrapper\.style\.display = 'none'", blob)


# ==================================================== 十字线对齐 ============
def _每根的首个1m(tfb) -> np.ndarray:
    """每根合成 bar 的首个 1m 成分下标, 一次算完整条。

    原来是 `_首根1m(tfb, b) = flatnonzero(bin_of == b)[0]`, **每调一次扫一遍整条
    bin_of**(15 万根)。周期集从四个涨到六个之后, 1m 当源那几组会变成 1.5e5 x 1.5e5
    次比较, 单组就要几分钟。`bin_of` 是稠密非降的 (`v3_timeframes._assert_shape`
    保证), 所以取相邻不等的位置即可, 结果与逐根扫描逐位相同。
    """
    return np.flatnonzero(np.r_[True, tfb.bin_of[1:] != tfb.bin_of[:-1]])


@pytest.mark.parametrize("源", C.周期集)
@pytest.mark.parametrize("靶", C.周期集)
def test_十字线按开盘对齐(数据, 源, 靶, 钟):
    """
    光标停在源网格的某根 bar 上, 靶网格应当高亮**包含这根 bar 开盘时刻**的那一根。

    真值来自 `bin_of` (1m 下标 -> 各周期箱号), 与 JS 里那条纯二分规则毫无关系,
    所以能当独立裁判。

    规则: 标签在 bar 末端 => 第 k 根的开盘时刻就是**第 k-1 根的标签**;
    标签 e 的 bar 覆盖 `[e-周期, e)`, 含 s 当且仅当 `e > s` => 要 upperBound。

    曾经用的是 `lowerBound(Y, X[k])` —— 那是按**收盘**对齐, 粗->细时落在那根粗 bar
    的最后一分钟。AU 实测 30m->15m 上 749 根错 709 根。
    """
    if 源 == 靶:
        pytest.skip("同一网格无需对齐")
    _, tf, _ = 数据
    X, Y = tf[源], tf[靶]
    Xlab, Ylab = X.bars.index.to_numpy(), Y.bars.index.to_numpy()

    # 整条向量化, **不抽样** —— 覆盖与逐根循环完全相同, 只是快几个数量级。
    首 = _每根的首个1m(X)
    assert len(首) == X.n_bars, "bin_of 不是稠密非降的, 向量化前提破了"
    真 = Y.bin_of[首].astype(np.int64)
    s = np.r_[Xlab[0] - np.timedelta64(1, "s"), Xlab[:-1]]
    got = np.searchsorted(Ylab, s, side="right").astype(np.int64)
    错 = int((got != 真).sum())
    # 只允许窗口第一根差 —— 那一根没有上一根标签可用, 退回旧行为
    assert 错 <= 1, f"{源} -> {靶}: {X.n_bars} 根里 {错} 根对不上"


def test_十字线对齐的JS实现是upperBound取上一根标签(blob):
    """结构断言, 防止 JS 那一侧退回按收盘对齐。"""
    assert "N.xhairUB" in blob, "少了 upperBound"
    assert "a[m] <= t" in blob, "xhairUB 写成了 lowerBound"
    assert "N.xhairUB(Y, s)" in blob, "对齐没走 upperBound"
    assert "s = X[k - 1]" in blob, "没有把末端标签换算成开盘时刻"
    assert "N.xhairLB(Y, t)" not in blob, "还留着按收盘对齐的老路径"


# ============================== marker 时间戳不许被 floor ====================
def _所有setMarkers载荷(blob: str) -> list[list[dict]]:
    """blob 里每一次 setMarkers 的 JSON 载荷（空的不算）。"""
    出 = []
    for m in re.finditer(r"(window\.\w+)\.series\.setMarkers\(", blob):
        pat = re.compile(
            re.escape(f"{m.group(1)}.series.setMarkers(") + r"(\[.*?\])\)", re.S)
        找 = pat.findall(blob)
        if 找:
            载 = json.loads(找[-1])
            if 载:
                出.append(载)
    return 出


def test_marker时间戳精确落在bar上_不被floor(数据, blob):
    """
    库的 `marker_list` 会把时间**向下取整到猜出来的 `_interval` 网格**
    (`SeriesCommon._single_datetime_format`), 而 K 线数据走的是精确路径
    (`_df_datetime_format` -> `astype('int64')//10**9`)。两条路不一致, 于是标签不
    落在网格上的 bar, 它上面的标记会被悄悄挪到**更早**的一根 —— 看起来就像信号
    用了未来数据。

    中国期货必然踩中: 小节休息/午休把合成 bar 截短, 截短的 bar 标签自然偏离网格。
    实测 2026-07-28 10:01 触发的信号, 30m 上应在 10:15 被写成 10:00, 2h 上应在
    11:30 也被写成 10:00 —— 两个都 floor 到了**产生信号的那根 bar**, 看起来就是
    未来函数。(floor 到一个不存在的时刻反而无害: JS 侧 `timeToIndex(findNearest)`
    会向后吸附回下一根真实 bar。有害的恰恰是落在一根真实 bar 上的这一类。)

    所以图层走 `lwc_helpers.set_markers_exact` 直接写精确 UTC 秒。这条断言:
    **发出去的每一个 marker 时间戳, 都精确等于某一根 bar 的时间戳。**
    """
    _, tf, _ = 数据
    合法 = {名: {int(pd.Timestamp(t).value // 10 ** 9) for t in tf[名].bars.index}
          for 名 in C.周期集}
    载荷们 = _所有setMarkers载荷(blob)
    assert 载荷们, "一次非空的 setMarkers 都没发出去"
    for 载 in 载荷们:
        命中 = [名 for 名, s in 合法.items() if all(x["time"] in s for x in 载)]
        assert 命中, (
            "有一批 marker 的时间戳不落在任何一个周期的 bar 上 —— 多半是被 floor 了。"
            f" 首个: {pd.Timestamp(载[0]['time'], unit='s')}")


def test_这条测试不是空真_确实有偏离网格的bar(数据, 钟):
    """
    守卫上一条: 若所有 bar 标签都恰好在 `_interval` 网格上, floor 与不 floor 毫无
    区别, 上面那条就是空真。这里断言 30m / 2h 上**确实存在**偏离网格的 bar。
    """
    _, tf, _ = 数据
    偏 = {}
    for 名, iv in (("1m", 60), ("5m", 300), ("15m", 900),
                   ("30m", 1800), ("2h", 7200), ("1d", 86400)):
        秒 = np.array([int(pd.Timestamp(t).value // 10 ** 9) for t in tf[名].bars.index])
        偏[名] = int((秒 % iv != 0).sum())
    assert 偏["1m"] == 0 and 偏["5m"] == 0 and 偏["15m"] == 0, f"1m/5m/15m 本该全在网格上: {偏}"
    assert 偏["30m"] > 0, "30m 上没有被截短的 bar —— 分箱口径变了?"
    assert 偏["2h"] > 0, "2h 上没有被截短的 bar —— 分箱口径变了?"
    # 1d 标在当日最后一分钟 (收盘时刻), 几乎不可能落在 UTC 天网格上
    assert 偏["1d"] > 0, "1d 全落在 86400 秒网格上 —— 标签口径变了?"


@pytest.mark.parametrize("模块名", ["src.viz.chart_lineA_03", "src.viz.chart_pullback"])
def test_图层没有再走库的marker_API(模块名):
    """
    结构断言: 走库的 marker API 就会被 floor, 不许退回去。

    两个模块都查 —— `chart_pullback` 的 30m 格每天的 10:15 同样中招, 而且
    `chart_rbreaker.py` 里的回调四格是直接调它的 `draw_markers`。

    也禁 `marker_list` / `.marker(` / `remove_marker`: 它们是**追加**语义, 而
    `set_markers_exact` 是**全量替换**, 同一个 pane 上混用会静默吞掉一批
    (或者在 `remove_marker` 处 KeyError)。
    """
    import importlib
    import inspect
    源 = inspect.getsource(importlib.import_module(模块名))
    for 禁 in (r"\.marker_list\(", r"\.clear_markers\(",
              r"\.marker\(", r"\.remove_marker\("):
        assert not re.search(禁, 源), f"{模块名} 里又出现了 {禁}"
    assert re.search(r"set_markers_exact\(", 源)


def test_未知的position或shape必须当场炸():
    """
    库对这两个都是静默失败, 后果只能靠肉眼发现:
    * `marker_position` 查不到返回 `None` -> JS 定位 switch 无 default -> 钉在 y=0
    * `marker_shape` 查不到原样透传   -> JS 绘制 switch 无 default -> 只剩文字
    """
    from src.viz.lwc_helpers import set_markers_exact

    class 假图:
        id = "window.fake"

        def run_script(self, s):
            pass

    正常 = {"time": pd.Timestamp("2026-07-28 10:15"), "position": "inside",
          "shape": "circle", "color": "#fff", "text": ""}
    with pytest.raises(ValueError, match="position"):
        set_markers_exact(假图(), [{**正常, "position": "belowBar"}])
    with pytest.raises(ValueError, match="shape"):
        set_markers_exact(假图(), [{**正常, "shape": "triangle"}])


def test_set_markers_exact本身不floor也不容忍乱序():
    """直接测那个 helper, 不经过图表。"""
    from src.viz.lwc_helpers import set_markers_exact

    发出 = []

    class 假图:
        id = "window.fake"

        def __init__(self):
            self.markers = {}

        def run_script(self, s):
            发出.append(s)

    图 = 假图()
    # 10:15 在 1800s 网格上是 20.5 格 —— 库会 floor 成 10:00, 这里必须原样
    t = pd.Timestamp("2026-07-28 10:15:00")
    set_markers_exact(图, [{"time": t, "position": "inside",
                            "shape": "square", "color": "#ab47bc", "text": "x"}])
    载 = json.loads(re.search(r"setMarkers\((\[.*\])\)", 发出[-1], re.S).group(1))
    assert 载[0]["time"] == int(t.value // 10 ** 9)
    assert 载[0]["position"] == "inBar" and 载[0]["shape"] == "square"

    with pytest.raises(ValueError, match="升序"):
        set_markers_exact(图, [
            {"time": pd.Timestamp("2026-07-28 10:15"), "position": "inside",
             "shape": "circle", "color": "#fff", "text": ""},
            {"time": pd.Timestamp("2026-07-28 10:00"), "position": "inside",
             "shape": "circle", "color": "#fff", "text": ""},
        ])


# ================================================ 止损线 / 吊灯线 =============
def test_两条止损线的值都能在1m原始序列里找到出处(数据, 钟):
    """
    图上每一个点都必须是引擎逐 1m 记的 `止损线` 里真实出现过的值, 不能是图层自己
    算出来的 —— 图层一旦自己算, 它和引擎就可能不一致, 而这两条线的**全部意义**
    就是拿来核对引擎。
    """
    _, tf, 结果 = 数据
    for 名 in C.周期集:
        tfb = tf[名]
        起 = C.起箱of(tfb, 结果, 钟)
        固帧, 吊帧, 阻帧 = C.止损吊灯帧(结果, tfb, 起, 钟)
        assert len(固帧) == len(吊帧) == tfb.n_bars - 起
        for 帧, 列 in ((固帧, "固定止损"), (吊帧, "吊灯")):
            v = 帧[列].to_numpy("float64")
            有 = ~np.isnan(v)
            assert 有.any(), f"{名} 的「{列}」一个点都没有"
            for k in np.flatnonzero(有)[:400]:
                b = 起 + int(k)
                成员 = np.flatnonzero(tfb.bin_of == b)
                # 引擎数组是 **15m** 索引的, 铺回 1m 再按本格的成员取值
                源列 = (结果.固定止损线 if 列 == "固定止损" else 结果.吊灯原线)
                源 = set(源列[钟.bin_of][成员[0]:成员[-1] + 1])
                assert v[k] in 源, f"{名} bin {b} 的「{列}」={v[k]} 不在 1m 序列里"


def test_两条线同生同死_且吊灯从入场就画(数据, 钟):
    """
    2026-08-26 改: 吊灯**从入场就画**, 包括它还压在固定止损更差一侧的那一段 ——
    否则看不见两条线什么时候相交, 而交叉点正是「吊灯开始接管」的时刻。

    所以这里钉的是两条:
      * 两条线的有值区间**逐根相同** (同生同死) —— 一条有值另一条没有就是 bug
      * 入场附近吊灯必然在固定止损的**更差**一侧 (多头更低), 也就是确实画出了
        那段「不起作用」的吊灯; 全程都更优的话说明又退回只画接管之后了
    """
    _, tf, 结果 = 数据
    for 名 in C.周期集:
        tfb = tf[名]
        起 = C.起箱of(tfb, 结果, 钟)
        固帧, 吊帧, 阻帧 = C.止损吊灯帧(结果, tfb, 起, 钟)
        a = 固帧["固定止损"].to_numpy("float64")
        b = 吊帧["吊灯"].to_numpy("float64")
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{名} 两条线的有值区间不一致"
        两者都有 = ~np.isnan(a)
        assert 两者都有.any(), f"{名} 上两条线一个点都没有"
        # 必须存在「吊灯更差」的那一段, 否则说明又退回了只画接管之后
        多头 = [t for t in 结果.交易 if t.方向 == "多"]
        assert 多头, "夹具里没有多头"
        更差 = (b < a) & 两者都有
        assert 更差.any(), (
            f"{名} 上吊灯从来没有低于固定止损 —— 入场那一段的吊灯没画出来")


def test_空仓的箱两条线都留断口(数据, 钟):
    """
    整箱没有仓位 -> NaN -> 库的 `js_data` 把它剔成只剩 time 的 whitespace 行。

    ⚠ **whitespace 本身不产生视觉断口** —— LWC 的 line 渲染器 (`walkLine`) 只是
    `for(...) lineTo(...)`, 没有断口判断, 会把洞两侧的点直接连起来 (浏览器逐像素
    验过)。视觉断口来自 `lineVisible: false + pointMarkersVisible: true` 那套画法,
    由 `test_止损线画成逐根圆点而不是折线` 守着。这条测试守的是**数据侧**: 空仓的
    箱必须是 NaN, 否则连点画法也会在那儿多画一个点。
    """
    _, tf, 结果 = 数据
    tfb = 钟
    起 = C.起箱of(tfb, 结果, 钟)
    固帧, _, _ = C.止损吊灯帧(结果, tfb, 起, 钟)
    v = 固帧["固定止损"].to_numpy("float64")
    assert np.isnan(v).any(), "一个空仓的箱都没有 —— 这条测试是空真的"
    # 「该有线」的那些驱动 bar: 持仓区间去掉入场根 —— `一根bar只做一个动作` 开着时
    # 本根不进出场块, 所以入场那一根本来就没有线 (引擎侧由
    # `test_三条止损线满足取大取小的恒等式` 钉死)。
    有线 = np.zeros(len(结果.止损线), dtype=bool)
    for t in 结果.交易:
        起根 = t.入场下标 + (1 if 结果.开关.一根bar只做一个动作 else 0)
        有线[起根:t.出场下标 + 1] = True
    有线 &= 结果.可成交          # 竞价根不出场 -> 出场块没跑 -> 没有线
    有线1m = 有线[钟.bin_of]
    for k in np.flatnonzero(np.isnan(v))[:400]:
        成员 = np.flatnonzero(tfb.bin_of == 起 + int(k))
        assert not 有线1m[成员].any(), f"bin {起 + int(k)} 有仓位却被留了断口"


def test_止损线只画在驱动周期那一格(数据):
    """
    粗周期上每个箱只留一个值(箱末), 一根 30m 里有两笔交易时前一笔的平仓价位会被后一笔
    盖掉 —— 那是画错不是粗糙, 拿它核对策略会得出错误结论。所以只画驱动格。
    """
    _, _, 结果 = 数据
    该画 = [名 for 名 in C.周期集 if C.该画止损线(名, 结果)]
    assert 该画 == [结果.参数.驱动周期], f"该画止损线的格子是 {该画}"
    assert len(该画) == 1, "只能有一格画止损线"


def test_吊灯受阻那条只在被阀门挡住的根上有值(数据, 钟):
    """
    受阻那条是**吊灯那条的子集**: 同样的值, 只在 `吊灯受阻` 为真的根上保留。它建在
    吊灯之后、点位重合, 所以画在上面显示成深绿 —— 方便一眼看出哪些根的止盈被阀门挡了。
    """
    _, _, 结果 = 数据
    起 = C.起箱of(钟, 结果, 钟)
    _, 吊帧, 阻帧 = C.止损吊灯帧(结果, 钟, 起, 钟)
    吊 = 吊帧["吊灯"].to_numpy("float64")
    阻 = 阻帧["吊灯受阻"].to_numpy("float64")
    有阻 = ~np.isnan(阻)
    assert 有阻.any(), "一根受阻的都没有 —— 这条测试是空真的"
    assert np.array_equal(阻[有阻], 吊[有阻]), "受阻那条的值必须与吊灯那条相同"
    assert (~np.isnan(吊[有阻])).all(), "受阻的根上吊灯那条也必须有值(图例要读它)"
    真阻 = 结果.吊灯受阻[起:]
    assert np.array_equal(有阻, 真阻 & ~np.isnan(吊)), "受阻的根与引擎记的对不上"


def test_止损线画成逐根圆点而不是折线(blob):
    """
    **平仓处必须真断开。** 而 LWC 的 line 渲染器在 whitespace 处不断开 ——
    `walkLine` 只是 `for(...) lineTo(...)`, 没有任何断口判断, 它把洞两侧的两个真实
    点直接连起来 (浏览器逐像素验过: 阶梯线画出一条停在旧价位的横线, 直线画出一条
    斜线)。单条 series 拿到真断口的唯一办法是关掉连线、只画点。

    所以这里禁掉 `lineVisible: true` 路径, 并要求两个点标记选项都在。
    """
    for 片 in ("lineVisible: false", "pointMarkersVisible: true"):
        assert blob.count(片) >= 2 * len(C.周期集), f"缺「{片}」"
    assert "lineType: 1" not in blob, (
        "又用回阶梯线了 —— 它在平仓处会画一条停在旧价位的横线")


def test_止损线默认关掉且开关联动图例(blob):
    """线关掉了却把读数留在图例上会误导, 所以图例块跟着同一个开关走。"""
    assert "止损线: false" in blob, "止损线默认必须是关的"
    assert "stopLines" in blob, "两条线没挂进 window.la3.tiles"
    assert "visible: window.la3.止损线" in blob, "开关没接到 series 可见性上"
    assert "window.la3.止损线 && t.hasStop" in blob, (
        "图例块没跟着开关显示/隐藏, 或没按 hasStop 只留在显示驱动周期的那一格")


def test_空仓时图例显示未开仓而不是留空(blob):
    """
    两条路都要有这句: 悬停**本格**走 `_止损图例` 的 subscribeCrosshairMove,
    悬停**别格**走 `wire_synced_crosshair` 广播 (setCrosshairPosition 不触发订阅)。
    文案必须逐字一致, 否则同一根 K 线在两种悬停下读数不同。
    """
    import json as _j
    assert blob.count(_j.dumps("未开仓")) >= 2, (
        "「未开仓」占位没有同时出现在本格路径和广播路径上")
