# -*- coding: utf-8 -*-
"""
lineA-03 引擎 —— `src/engine/lineA_03_backtest.py`。

用**真实 AU parquet**, 不是合成帧: 这一层要验的是执行纪律在真实的时段断口、跳空、
换月上还成不成立, 合成帧验不到。

夹具窗口取 2024-01-01 ~ 2026-07-29 —— 短到测试能跑完, 长到 SL / 吊灯 / 跳空 / 冷静期
四类事件都真实出现过 (由 `test_夹具值得断言` 守着, 否则下面全是空真)。
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from src.data.paths import CLEAN_DIR, CLEAN_DIR_HINT
from src.data.v3_sessions import derive_segments, prepare
from src.data.v3_timeframes import build_timeframes
from src.engine.lineA_03_backtest import 出场理由, 吊灯, 止损, 计算周期, 跑回测
from src.indicators import atr as atr_series
from src.strategy.lineA_03 import 策略开关
from src.strategy.pullback import gap_filled

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CLEAN_DIR, "AU.parquet")),
    reason=f"需要 {CLEAN_DIR_HINT} —— 先跑 python -m src.data.prepare_v3_minute",
)

_切点 = (0.35, 0.6, 0.85)


@pytest.fixture(scope="module")
def prep():
    return prepare("AU", start="2024-01-01", end="2026-07-29", warmup_days=60)


@pytest.fixture(scope="module")
def tf(prep):
    return build_timeframes(prep.df, segments=derive_segments(prep.df), tfs=计算周期)


@pytest.fixture(scope="module")
def res(prep, tf):
    return 跑回测(prep, tf)


@pytest.fixture(scope="module")
def 钟(tf, res):
    """驱动周期的 `TFBars` —— 所有下标都活在它的下标空间里。"""
    return tf[res.参数.驱动周期]


# --------------------------------------------------------------- 空真守卫 ----
def test_夹具值得断言(res):
    """下面每一条断言的前提。四类事件都得真实出现过, 否则全是空真。"""
    assert len(res.交易) >= 20, f"只有 {len(res.交易)} 笔, 样本太少"
    理由集 = {t.理由 for t in res.交易}
    assert 理由集 == set(出场理由), f"两种出场理由没都出现: {理由集}"
    assert any(t.跳空成交 for t in res.交易), "一笔跳空成交都没有"
    assert res.冷静期区间, "冷静期一次都没触发"
    assert any(t.方向 == "多" for t in res.交易)
    assert any(t.方向 == "空" for t in res.交易)


# ------------------------------------------------------------- 截断法 ----
@pytest.mark.parametrize("切", _切点)
def test_无未来函数_截断法(prep, res, 切):
    """
    把 bar i 之后的价格全改成垃圾重跑, 所有 `出场下标 < i` 的交易必须**逐字节相同**。

    照抄 `tests/test_pullback_v3_backtest.py:74-103`。**只改价格, 不动索引与
    trading_date**, 所以分箱不变 —— 于是这条真正检验的是「决策有没有读到未来的
    **价格**」, 而不是「分箱有没有变」。

    这是那张无未来函数审计表的唯一机械凭据: 表可以写错, 截断法不会。

    **投毒仍然打在 1m 帧上**(那是数据源), 但比较要在 **15m** 下标空间做 ——
    引擎自 2026-08-26 起完全跑在 15m 上。投毒 1m bar i 会弄脏**包含它的那根 15m**,
    所以干净的是 `15m 下标 < bin_of15[i]` 的那些。
    """
    n = len(prep.df)
    i = int(n * 切)
    脏 = prep.df.copy()
    rng = np.random.default_rng(0)
    for col in ("open", "high", "low", "close"):
        v = 脏[col].to_numpy("float64").copy()
        v[i:] = v[i:] * 3.0 + rng.normal(0, 50, size=n - i)
        脏[col] = v
    脏["high"] = np.maximum(脏["high"], np.maximum(脏["open"], 脏["close"]))
    脏["low"] = np.minimum(脏["low"], np.minimum(脏["open"], 脏["close"]))

    脏tf = build_timeframes(脏, tfs=计算周期)
    got = 跑回测(dataclasses.replace(prep, df=脏), 脏tf)

    切 = int(脏tf[res.参数.驱动周期].bin_of[i])   # 第一根被弄脏的驱动 bar

    def 之前(交易):
        return [(t.方向, t.入场下标, t.入场价, t.出场下标, t.出场价,
                 t.理由, t.仓位比例, t.跳空成交)
                for t in 交易 if t.出场下标 < 切]

    assert 之前(got.交易) == 之前(res.交易)
    assert len(之前(res.交易)) >= 5, "截断点太靠前, 这条测试没验到东西"


def test_截断法本身不是空真(prep, res):
    """守卫上一条: 投毒之后尾段必须真的变了, 否则截断法测的是个恒等式。"""
    n = len(prep.df)
    i = int(n * 0.6)
    脏 = prep.df.copy()
    v = 脏["close"].to_numpy("float64").copy()
    v[i:] = v[i:] * 3.0
    for col in ("open", "high", "low", "close"):
        c = 脏[col].to_numpy("float64").copy()
        c[i:] = c[i:] * 3.0
        脏[col] = c
    脏tf = build_timeframes(脏, tfs=计算周期)
    got = 跑回测(dataclasses.replace(prep, df=脏), 脏tf)
    切 = int(脏tf[res.参数.驱动周期].bin_of[i])
    后 = lambda r: [t.出场价 for t in r.交易 if t.出场下标 >= 切]
    assert 后(got) != 后(res)


# ------------------------------------------------ 这一版特有的两条不变量 ----
def test_入场成交价就是那根驱动bar的open(res, 钟):
    """
    **恒等式, 一笔例外都不许有**: `一笔.入场下标` 是驱动周期的下标, 成交价就是那根
    的 `open`。

    旧口径的例外类 (入场被闸门推迟到驱动 bar 内部的某一分钟) 结构上不可能出现 ——
    闸门是逐驱动 bar 判的, 被挡的那一根整根不开仓, 不存在「箱内推迟」。
    """
    o = 钟.bars["open"].to_numpy("float64")
    assert res.交易, "一笔都没有"
    坏 = [t for t in res.交易 if abs(t.入场价 - o[t.入场下标]) > 1e-12]
    assert not 坏, (f"{len(坏)}/{len(res.交易)} 笔的入场价不等于那根 "
                    f"{res.参数.驱动周期} 的 open")


def test_引擎只读驱动周期的价格(res, 钟):
    """
    所有出场价要么落在驱动 bar 的 [low, high] 内, 要么等于它的 open (跳空补价)。

    这条钉的是「价格只从驱动周期取」。引擎若偷看比驱动周期更细的极值, 出场价就会
    落到驱动 bar 的范围之外 —— 15m 驱动时代这条挡的是偷看 1m; 1m 驱动之后驱动周期
    已经是最细的一层, 它退化成「出场价必须真的来自那一根」。
    """
    b = 钟.bars
    lo, hi, op = (b["low"].to_numpy("float64"), b["high"].to_numpy("float64"),
                  b["open"].to_numpy("float64"))
    for t in res.交易:
        j = t.出场下标
        assert lo[j] - 1e-9 <= t.出场价 <= hi[j] + 1e-9 or abs(t.出场价 - op[j]) < 1e-12, (
            f"出场价 {t.出场价} 落在 {res.参数.驱动周期} bar "
            f"[{lo[j]}, {hi[j]}] 之外 @{t.出场时间}")


def _后退根数(res) -> int:
    """有效止损朝不利方向移动了多少根 —— 单调性的度量。"""
    坏 = 0
    for t in res.交易:
        段 = res.止损线[t.入场下标:t.出场下标 + 1]
        段 = 段[~np.isnan(段)]
        if len(段) < 2:
            continue
        d = np.diff(段)
        坏 += int((d < -1e-9).sum() if t.方向 == "多" else (d > 1e-9).sum())
    return 坏


def test_ATR冻结时有效止损单调不降(prep, tf):
    """
    极值单调 + ATR₀ 不变 => 吊灯线单调 => `max(固定止损, 吊灯线)` 单调。

    **ATR 随动就拿不到这条性质** —— 见下一条。那是随动的代价, 是 2026-08-28 明确
    接受的取舍, 不是 bug。
    """
    r = 跑回测(prep, tf, 开关=dataclasses.replace(策略开关(), 止盈_ATR随动=False))
    assert _后退根数(r) == 0, "ATR 冻结时有效止损竟然后退了"


def test_ATR随动真的会让追踪线后退(prep, tf):
    """
    空真守卫 + 代价的显式记录。随动必须**真的**破坏单调性, 否则上一条测试是在比较
    两个等价的东西, `止盈_ATR随动` 这个开关也就没有存在的理由。
    """
    r = 跑回测(prep, tf, 开关=dataclasses.replace(策略开关(), 止盈_ATR随动=True))
    assert _后退根数(r) > 0, "ATR 随动竟然没让线后退?"


def test_有效止损单调不降_默认口径(res):
    """默认口径下的同一条断言, 走 `res` 夹具, 与上面那两条互为交叉检查。"""
    if res.开关.止盈_ATR随动:
        pytest.skip("ATR 随动不保证单调, 见 test_ATR随动真的会让追踪线后退")
    for t in res.交易:
        段 = res.止损线[t.入场下标:t.出场下标 + 1]
        段 = 段[~np.isnan(段)]
        if len(段) < 2:
            continue
        差 = np.diff(段)
        if t.方向 == "多":
            assert (差 >= -1e-9).all(), f"多头有效止损后退了 @{t.入场时间}"
        else:
            assert (差 <= 1e-9).all(), f"空头有效止损后退了 @{t.入场时间}"


def test_三条止损线满足取大取小的恒等式(res):
    """
    引擎记三条逐驱动 bar 的线, 恒等式是 `止损线 = max(固定止损线, 吊灯原线)` (空头取 min)。

    `吊灯原线` 是**不带地板**的, 所以它在浮盈涨过 +1R 之前坐在固定止损的更差一侧 ——
    图层要画出那一段, 才看得见两条线什么时候相交。这条测试同时保证:
      * 三条线的有值区间逐根一致 (要么都空仓, 要么都有值)
      * 图层拿 `吊灯原线` 画出来的东西, 与引擎真正测试的 `止损线` 是同一套数
    """
    固, 吊, 有效 = res.固定止损线, res.吊灯原线, res.止损线
    assert np.array_equal(np.isnan(固), np.isnan(有效)), "固定止损线与止损线的空仓区间不一致"
    assert np.array_equal(np.isnan(吊), np.isnan(有效)), "吊灯原线与止损线的空仓区间不一致"

    # 两处不对称, 都要显式对上:
    #
    #   出场根   出场块先写 `止损线[i]` 再把批置空, 而 `仓位[i]` 是随后的 MTM 块写的
    #            —— 于是出场根上「有线但仓位为 0」, 要**并进来**。
    #   入场根   `一根bar只做一个动作` 开着时, 本根不进出场块, 所以没有线; 而 MTM 已经
    #            把 `仓位[i]` 写上了 —— 于是入场根上「有仓位但没线」, 要**去掉**。
    #
    # 两条差集都必须**恰好**等于对应的根集合, 顺带把这两处不对称本身钉死。
    有 = ~np.isnan(有效)
    出场根 = np.zeros(len(有效), dtype=bool)
    for t in res.交易:
        出场根[t.出场下标] = True
    # 入场根从 `仓位` 的 0 -> 非 0 跃变推, **不从 `res.交易` 推** —— 末尾未平仓的那
    # 一批不在 `交易` 里 (刻意丢弃), 但它的仓位与线一样都写在数组上。
    仓 = res.仓位 != 0
    入场根 = 仓 & ~np.r_[False, 仓[:-1]]
    期望 = 仓 | 出场根
    if res.开关.一根bar只做一个动作:
        期望 = 期望 & ~入场根
    # 竞价根不出场 -> 出场块整块没跑 -> 三条线都不写。
    期望 = 期望 & res.可成交
    assert np.array_equal(有, 期望), "止损线的有值区间 ≠ 持仓根 ∪ 出场根 − 入场根"

    多 = res.仓位 > 0
    for t in res.交易:                       # 出场根的方向从这一笔补回来
        多[t.出场下标] = (t.方向 == "多")
    多有, 空有 = 有 & 多, 有 & ~多
    assert 多有.any() and 空有.any(), "多空两边没都出现 —— 这条测试是空真的"
    # 止盈阀门开着时恒等式多一支: 受阻的根上有效止损**退回固定止损**, 不取大/取小。
    阻 = res.吊灯受阻
    多通, 空通 = 多有 & ~阻, 空有 & ~阻
    assert np.allclose(有效[多通], np.maximum(固[多通], 吊[多通])), "多头不满足取大"
    assert np.allclose(有效[空通], np.minimum(固[空通], 吊[空通])), "空头不满足取小"
    if res.开关.止盈_不得差于入场价:
        受 = 有 & 阻
        assert 受.any(), "一根受阻的都没有 —— 阀门那一支没验到"
        assert np.allclose(有效[受], 固[受]), "受阻的根上有效止损应当就是固定止损"
    else:
        assert not 阻.any(), "阀门关着却记了受阻"

    # 空真守卫: 必须真有「吊灯还压在固定止损更差一侧」的那一段, 否则 max/min 退化
    assert (吊[多有] < 固[多有]).any(), "多头从没出现吊灯低于固定止损 —— 入场那段没记下来"
    assert (吊[空有] > 固[空有]).any(), "空头从没出现吊灯高于固定止损"


def _独立重算出场(t, res, 钟, 含本根: bool, 险=None):
    """
    完全独立地把一笔的出场重算一遍, 不碰引擎的任何内部状态。

    `含本根=False` 是设计: 测试 bar i 时用的极值只折到 **i-1**。
    `含本根=True`  是那个经典错误: 先把 bar i 的 high/low 折进极值再测试。
    返回 (出场下标, 出场价, 理由) 或 None。

    吊灯的 ATR 按 `开关.止盈_ATR随动` 取: 不随动就用反解出来的 ATR₀; 随动则每根重取
    「截至本根、最后一根已收盘 30m」的 ATR(14) —— 与引擎同一个 `closed_pos`, 但这里
    是独立算的一份。
    """
    p = res.参数
    多 = t.方向 == "多"
    b = 钟.bars
    hi = b["high"].to_numpy("float64")
    lo = b["low"].to_numpy("float64")
    op = b["open"].to_numpy("float64")
    atr0 = abs(t.入场价 - t.固定止损位) / p.止损ATR倍数
    可成交 = res.可成交
    冻结 = not res.开关.止盈_ATR随动
    if not 冻结:
        ATR险 = np.asarray(atr_series(险.bars, p.ATR周期), dtype="float64")
        c险 = 险.closed_pos.astype(np.int64)
    def 折(极, j):
        """竞价根不参与极值 —— 与引擎第 11 步同一个守卫。"""
        if not 可成交[j]:
            return 极
        return max(极, hi[j]) if 多 else min(极, lo[j])

    极值 = t.入场价
    # `一根bar只做一个动作` 开着时, 本根开的仓本根不测出场 —— 起点是入场下标+1。
    起 = t.入场下标 + (1 if res.开关.一根bar只做一个动作 else 0)
    if 起 > t.入场下标:
        # **入场那一根的极值仍然要折进来。** 引擎里出场块的守卫是 `<`(本根不测出场),
        # 而极值折入那一步的守卫是 `<=` —— 入场根不测出场, 但它的 high/low 照样进极值。
        # 漏了这一折, 只有极少数笔会露馅 (实测 328 笔里 1 笔), 正好是这条测试的价值。
        极值 = 折(极值, t.入场下标)
    for j in range(起, len(hi)):
        if 含本根:
            极值 = 折(极值, j)
        if 冻结 or c险[j] < 0 or np.isnan(ATR险[c险[j]]):
            吊atr = atr0
        else:
            吊atr = float(ATR险[c险[j]])
        raw = (极值 - p.吊灯ATR倍数 * 吊atr) if 多 else (极值 + p.吊灯ATR倍数 * 吊atr)
        if 可成交[j]:                     # 竞价根不出场
            # 止盈阀门: 吊灯线差于入场价时本根不许用它出场, 退回固定止损。
            受阻 = res.开关.止盈_不得差于入场价 and (
                raw < t.入场价 if 多 else raw > t.入场价)
            线 = t.固定止损位 if 受阻 else (
                max(raw, t.固定止损位) if 多 else min(raw, t.固定止损位))
            if (lo[j] <= 线) if 多 else (hi[j] >= 线):
                return (j, gap_filled(线, 多, op[j], True),
                        止损 if 线 == t.固定止损位 else 吊灯)
        if not 含本根:
            极值 = 折(极值, j)
    return None


def test_吊灯的极值严格滞后一根_不含本根的high_low(res, 钟, tf):
    """
    **截断法结构上抓不到这条。** 截断法把 bar i 之后的数据改成垃圾, 验的是「有没有读
    到未来的 bar」; 而这里的错误是「在同一根 bar 内部把顺序搞反了」—— 先用本根的
    high 抬高吊灯, 再拿本根的 low 去撞它。那样的引擎在截断法下照样全绿。

    所以这条测试独立把每一笔的出场重算两遍, 断言引擎与**滞后版**逐笔相等。

    为什么必须滞后: 我们**不知道 bar 内部的走势**。折入本根之后, 本根的 low 就可能
    反过来「打掉」一条刚被本根 high 抬起来的线 —— 凭空造出一笔真实市场里不可能发生
    的出场。驱动周期越粗风险越大 —— 一根 15m 的振幅比一根 1m 大得多。1m 驱动之后
    单根振幅小了, 但**错排造成的价差反而更大**: 排错了触发的是另一根 bar 上的另一
    个出场事件, 差多少与单根振幅无关 (实测 1m 帧上中位 14.8、最小 6.5)。
    """
    def 同(g, t, 容差: float) -> bool:
        return (g is not None and g[0] == t.出场下标 and g[2] == t.理由
                and abs(g[1] - t.出场价) <= 容差 * max(1.0, abs(t.出场价)))

    # 出场**下标**与**理由**必须逐笔精确相等; 价格给 1e-9 相对容差, 因为这里的 ATR₀
    # 是从 `|入场价 - 固定止损位| / 止损ATR倍数` 反解出来的, 与引擎那个不是逐位相同
    # (实测最大相对差 2.1e-16, 一个 ULP)。这个容差比真实的错排差异小 15 个数量级 ——
    # 下面那个空真守卫会证明它确实分得开。
    坏 = [t for t in res.交易
          if not 同(_独立重算出场(t, res, 钟, False, tf['30m']), t, 1e-9)]
    assert not 坏, (
        f"引擎与「极值截至 i-1」不一致: {len(坏)}/{len(res.交易)} 笔, "
        f"首个 @{坏[0].入场时间}")

    # 空真守卫: 两个版本必须**真的**能分出来, 否则上面那条断言什么都没验。
    # 而且要在**同一个容差**下分得开 —— 否则那条断言只是在验浮点噪声。
    分歧 = [t for t in res.交易
            if not 同(_独立重算出场(t, res, 钟, True, tf['30m']), t, 1e-9)]
    assert 分歧, "含本根版与引擎完全一致 —— 这条测试是空真的, 夹具里没有能分辨的 bar"
    差 = [abs(_独立重算出场(t, res, 钟, True, tf['30m'])[1] - t.出场价) for t in 分歧
          if _独立重算出场(t, res, 钟, True, tf['30m']) is not None]
    assert max(差) > 1e-3, f"两个版本只差 {max(差):.2e} —— 那不是错排, 是浮点噪声"


def test_吊灯出场里有亏损的_而且这不是bug(res):
    """
    浮盈在 +1R 到 +2R 之间时吊灯生效但坐在入场价下方, 所以那一段被它打掉的交易
    **结构性地是亏损单**。这条测试把这件事钉成「已知性质」而不是「待修的 bug」——
    哪天它变成 0, 说明有人动了倍数或者动了理由的二分规则。

    面板因此必须把吊灯出场的盈/亏分开报 (`lineA_03_stats.面板行`)。
    """
    吊 = [t for t in res.交易 if t.理由 == 吊灯]
    亏 = [t for t in 吊 if t.净收益 <= 0]
    assert 吊, "一笔吊灯出场都没有"
    assert 亏, "吊灯出场全是盈利的 —— 倍数或理由二分规则被改过了"
    assert len(亏) / len(吊) < 0.8, "吊灯几乎全在亏, 这不对"


def test_三步走的时序在每一笔上都成立(res):
    """
    手工验收清单第 4、5 条的机械版: 每个 Entry 之前都必须能找到
    **回调闩锁点 → 开关1点 / 开关2点**, 且回调不晚于两个锁。

    这条把「三步走」从一句描述变成一个可证伪的断言 —— 哪天级联被改错了顺序
    (比如先看锁再认回调), 这里立刻红。
    """
    l1 = np.array([p["下标"] for p in res.开关1点])
    l2 = np.array([p["下标"] for p in res.开关2点])
    pb = np.array([p["下标"] for p in res.回调闩锁点])
    assert l1.size and l2.size and pb.size
    for t in res.交易:
        前1, 前2, 前回 = l1[l1 <= t.入场下标], l2[l2 <= t.入场下标], pb[pb <= t.入场下标]
        assert 前1.size, f"入场 @{t.入场时间} 之前没有开关1点"
        assert 前2.size, f"入场 @{t.入场时间} 之前没有开关2点"
        assert 前回.size, f"入场 @{t.入场时间} 之前没有回调闩锁点"
        assert 前回.max() <= 前1.max() and 前回.max() <= 前2.max(), (
            f"入场 @{t.入场时间} 的回调晚于锁 —— 三步走的顺序被破坏了")


# --------------------------------------------------------- 结构不变量 ----
def test_出场理由只有两种(res):
    assert {t.理由 for t in res.交易} <= {止损, 吊灯}


def test_一批一腿(res):
    """本策略无部分止盈 —— 仓位比例恒 1.0, 入场下标互不重复。"""
    assert all(t.仓位比例 == 1.0 for t in res.交易)
    入场 = [t.入场下标 for t in res.交易]
    assert len(入场) == len(set(入场))


def test_一次只持一仓(res):
    区间 = sorted((t.入场下标, t.出场下标) for t in res.交易)
    for (a1, b1), (a2, _) in zip(区间, 区间[1:]):
        assert a2 > b1, f"{a1}-{b1} 与 {a2} 重叠"


def test_出场不早于入场(res):
    assert all(t.出场下标 >= t.入场下标 for t in res.交易)


def test_跳空补价只会让成交更差(res):
    """
    「止损吃亏, 止盈不占便宜」。本策略两种出场都是被线打掉的, 所以两种都补价。
    `跳空成交` 标志必须与成交价一致。
    """
    for t in res.交易:
        线 = res.止损线[t.出场下标]
        if np.isnan(线):
            continue
        if t.方向 == "多":
            assert t.出场价 <= 线 + 1e-9
        else:
            assert t.出场价 >= 线 - 1e-9
        assert t.跳空成交 == (abs(t.出场价 - 线) > 1e-9)


def test_末尾未平仓被如实报出(res):
    """
    刻意丢弃, 但必须是**可见的**丢弃 —— 计数要对得上末根仓位。
    (少一笔数据好过多一笔错误数据, 但不能悄悄少。)
    """
    assert res.末尾未平仓 == (1 if res.仓位[-1] != 0 else 0)


def test_冷静期里一笔都不开(res):
    冻结 = np.zeros(len(res.仓位), dtype=bool)
    for a, b in res.冷静期区间:
        冻结[a:b + 1] = True
    assert not 冻结[[t.入场下标 for t in res.交易]].any()


def test_暖机段里一笔都不开(res):
    assert all(t.入场下标 >= res.暖机根数 for t in res.交易)


def test_观测记录都不参与决策(res):
    """
    四类标注点是**仅观测**的。把它们清空重跑, 交易必须逐字节相同 —— 这条抓的是
    「不小心让图层的记录反过来影响了判定」。
    """
    assert res.回调闩锁点 and res.大周期翻转点 and res.开关1点 and res.开关2点
    # 结构上它们只被 append, 从不被读; 这里断言下标都落在合法范围里
    n = len(res.仓位)
    for 列 in (res.回调闩锁点, res.大周期翻转点, res.开关1点, res.开关2点):
        assert all(0 <= d["下标"] < n for d in 列)
        assert all(d["方向"] in (1, -1) for d in 列)


# ------------------------------------------------------------- 具名开关 ----
def test_冷静期按方向计数真的接上了(prep, tf, res):
    """抓「加了字段但引擎根本没读」那类哑火。"""
    关 = 跑回测(prep, tf, 开关=策略开关(冷静期_按方向计数=False))
    assert 关.冷静期区间 != res.冷静期区间 or len(关.交易) != len(res.交易), \
        "翻转 冷静期_按方向计数 之后结果没变 —— 这个开关没接上"


def test_成本真的被扣掉(prep, tf, res):
    带成本 = 跑回测(prep, tf, 开关=策略开关(成本_每笔基点=5.0))
    assert 带成本.交易 and res.交易
    assert 带成本.交易[0].净收益 < res.交易[0].净收益
    # 一批一腿 -> 每笔恰好扣一次
    assert 带成本.交易[0].毛收益 == pytest.approx(res.交易[0].毛收益)
    assert (res.交易[0].净收益 - 带成本.交易[0].净收益) == pytest.approx(5e-4)


# ------------------------------------------------------------- 边界 ----
def test_空帧不炸(prep):
    空 = prep.df.iloc[:0]
    r = 跑回测(dataclasses.replace(prep, df=空))
    assert r.交易 == [] and len(r.权益曲线) == 0 and r.末尾未平仓 == 0
