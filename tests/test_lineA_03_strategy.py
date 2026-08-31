# -*- coding: utf-8 -*-
"""
lineA-03 原语层 —— `src/strategy/lineA_03.py`。

用合成数据而不是读 parquet: 失败时指向逻辑而不是数据。价格刻意落在 0.5 一档的粗糙
网格上, 与 `tests/test_pullback_v3_states.py:43-45` 同一个理由 —— 离散网格会制造大量
`收盘 == MA21` 的平局, 而平局分支是最容易在实现里被改掉的。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators import macd as 参考macd
from src.strategy.lineA_03 import (
    多, 空, 未定, 三线状态, 吊灯线, 固定止损, 大周期状态, 完全反向,
    macd线, 冷静期该解除, 粘住, 策略参数, 金叉死叉, 动量开关, 动量开关段,
)


def _游走(n: int = 600, seed: int = 7, 波动: float = 0.6, 起点: float = 500.0):
    rng = np.random.default_rng(seed)
    px = 起点
    out = np.empty(n, dtype="float64")
    for i in range(n):
        px = round((px + rng.normal(0, 波动)) * 2) / 2      # 0.5 一档
        out[i] = px
    return out


@pytest.fixture(scope="module")
def 收盘():
    return _游走()


# ---------------------------------------------------------------- MACD ----
def test_macd_与_indicators_逐位相同(收盘):
    """
    刻意不调 `src/indicators.py::macd`（它读 `df["timestamp"]` 这一列，而 v3 数据层
    的 `TFBars.bars` 是 DatetimeIndex 且没有这一列，直接喂会 KeyError）。
    公式必须逐位相同 —— 这条测试就是那个保证。
    """
    dif, dea = macd线(收盘)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=len(收盘), freq="15min"),
        "close": 收盘,
    })
    ref = 参考macd(df)
    assert np.array_equal(dif, ref["macd"].to_numpy())
    assert np.array_equal(dea, ref["signal"].to_numpy())


# ------------------------------------------------------------ 三线排列 ----
def test_三线严格不等_平局判未定():
    """三元严格不等, 不许有等号 —— 恒定序列上不能翻出多或空。"""
    平 = np.full(300, 100.0)
    assert (三线状态(平) == 未定).all()
    assert (大周期状态(平) == 未定).all()


def test_粘住只在开头留未定(收盘):
    生 = 三线状态(收盘)
    st = 粘住(生)
    非未定 = np.flatnonzero(st != 未定)
    assert 非未定.size, "夹具太短, 一次干净排列都没出现"
    assert (st[非未定[0]:] != 未定).all(), "未定只能出现在序列开头一段连续区间里"


def test_粘住期间逐位保持上一根(收盘):
    """排列混乱的那些根, 输出必须与上一根**逐位相同**。"""
    生 = 三线状态(收盘)
    st = 粘住(生)
    混乱 = np.flatnonzero(生 == 未定)
    混乱 = 混乱[混乱 > 0]
    assert 混乱.size
    assert np.array_equal(st[混乱], st[混乱 - 1])


def test_粘住真的改变了输出(收盘):
    """抓哑火: 如果粘性是空操作, 上面两条都会空真地通过。"""
    生 = 三线状态(收盘)
    assert not np.array_equal(生, 粘住(生))
    assert (粘住(生) == 未定).sum() < (生 == 未定).sum()


def test_完全反向要求排列干净():
    """「未定」不算反向 —— 排列不干净就不是回调。"""
    小 = np.array([多, 空, 未定], dtype=np.int8)
    assert 完全反向(小, 多).tolist() == [False, True, False]
    assert 完全反向(小, 空).tolist() == [True, False, False]
    assert 完全反向(小, 未定).tolist() == [False, False, False]


# ------------------------------------------------------------- 锁 2 ----
def test_开关2的两层暖机守卫(收盘):
    """
    第一层: 守卫之内不输出锁。
    第二层: 守卫之内**不累积状态** —— 少了这一层, 一次发生在 MACD 垃圾区里的金叉
    会把标志置上, 守卫刚放开锁就凭空开着。实测就踩过这个坑。
    """
    dif, dea = macd线(收盘)
    for 守卫 in (20, 35, 60):
        多开, 空开 = 动量开关(dif, dea, 守卫)
        assert not 多开[:守卫].any() and not 空开[:守卫].any()      # 第一层
        金, 死 = 金叉死叉(dif, dea)
        for arr, 事件, 水下 in ((多开, 金, True), (空开, 死, False)):
            起 = np.flatnonzero(arr & ~np.r_[False, arr[:-1]])
            for b in 起:                                            # 第二层
                assert 事件[b], f"守卫={守卫} 锁在 b={b} 开了, 但那一根不是交叉根"
                if 水下:
                    assert dif[b] < 0 and dea[b] < 0
                else:
                    assert dif[b] > 0 and dea[b] > 0


def test_开关2是闩锁不是事件(收盘):
    """
    「两个锁都解锁直接触发入场」—— 锁是状态, 开了就一直开到反向交叉。
    如果实现成事件, 每个锁只会有孤立的单根为真。
    """
    dif, dea = macd线(收盘)
    多开, _ = 动量开关(dif, dea, 35)
    assert 多开.any()
    段长 = np.diff(np.flatnonzero(np.diff(np.r_[False, 多开, False])))[::2]
    assert 段长.max() > 1, "开关2 被实现成了单根事件, 不是运行段"


def test_开关2的段落止于反向交叉(收盘):
    """多头锁的每一段, 段内必须全程 DIF > DEA。"""
    dif, dea = macd线(收盘)
    多开, 空开 = 动量开关(dif, dea, 35)
    assert (dif[多开] > dea[多开]).all()
    assert (dif[空开] < dea[空开]).all()


def test_多空开不可能同时开(收盘):
    dif, dea = macd线(收盘)
    多开, 空开 = 动量开关(dif, dea, 35)
    assert not (多开 & 空开).any()


# ------------------------------------------------------ 止损 / 吊灯几何 ----
def test_固定止损就是纯ATR倍数():
    assert 固定止损(100.0, True, 2.0, 1.5) == pytest.approx(97.0)
    assert 固定止损(100.0, False, 2.0, 1.5) == pytest.approx(103.0)


def test_吊灯永不比固定止损更差():
    """`地板` 是入场时冻结的固定止损 —— 剩余风险任何时刻都不超过它。"""
    rng = np.random.default_rng(1)
    for _ in range(500):
        入场 = float(rng.uniform(50, 500))
        atr = float(rng.uniform(0.1, 20))
        地板多 = 固定止损(入场, True, atr, 1.5)
        地板空 = 固定止损(入场, False, atr, 1.5)
        极值 = 入场 * float(rng.uniform(0.9, 1.1))
        assert 吊灯线(True, 极值, atr, 3.0, 地板多) >= 地板多 - 1e-9
        assert 吊灯线(False, 极值, atr, 3.0, 地板空) <= 地板空 + 1e-9


def test_吊灯的三个临界点():
    """
    记 R = 1.5 ATR₀。两条线共用同一个 ATR₀, 于是:

        极值 = 入场 + 1R  ->  吊灯恰好等于固定止损 (两条线在这里接上)
        极值 = 入场 + 2R  ->  吊灯恰好回到入场价   (保本)

    **所以浮盈在 +1R 到 +2R 之间时, 吊灯生效但坐在入场价下方** —— 那一段被吊灯打掉
    的交易结构性地是亏损单。这不是 bug, 是 1.5/3.0 这组倍数自带的性质。
    """
    入场, atr = 100.0, 2.0
    R = 1.5 * atr
    地板 = 固定止损(入场, True, atr, 1.5)
    assert 地板 == pytest.approx(入场 - R)

    assert 吊灯线(True, 入场 + R, atr, 3.0, 地板) == pytest.approx(地板)
    assert 吊灯线(True, 入场 + 2 * R, atr, 3.0, 地板) == pytest.approx(入场)
    中间 = 吊灯线(True, 入场 + 1.5 * R, atr, 3.0, 地板)
    assert 地板 < 中间 < 入场, "浮盈 +1.5R 时吊灯应当生效但仍在入场价下方"


def test_冻结ATR让吊灯线单调():
    """
    极值单调 + ATR₀ 不变 => 吊灯线单调不降。这是 v3 做不到的:
    v3 用实时 ATR, ATR 上升而极值不变时吊灯会**后退**
    (`tests/test_pullback_v3_backtest.py:106` 记着)。
    """
    入场, atr = 100.0, 2.0
    地板 = 固定止损(入场, True, atr, 1.5)
    极值 = np.maximum.accumulate(入场 + np.abs(np.random.default_rng(2).normal(0, 3, 400)))
    线 = np.array([吊灯线(True, float(e), atr, 3.0, 地板) for e in 极值])
    assert (np.diff(线) >= -1e-12).all()


# ---------------------------------------------------------------- 参数 ----
def test_参数拒绝乱序均线():
    with pytest.raises(ValueError, match="快线"):
        策略参数(快线=55, 慢线=21)
    with pytest.raises(ValueError, match="MACD快"):
        策略参数(MACD快=26, MACD慢=12)


# ============================================ 冷静期解除 R2′ (K2 / K3) ===
@pytest.mark.parametrize("前态,今态,应解除", [
    (未定, 多,   True),    # 从混乱走到干净
    (未定, 空,   True),
    (空,   多,   True),    # **直接反转也解除** —— K2 指名要的两行
    (多,   空,   True),
    (多,   多,   False),   # 一直干净不算跨越
    (空,   空,   False),
    (多,   未定, False),   # 今态必须干净
    (空,   未定, False),
    (未定, 未定, False),
])
def test_冷静期解除的真值表(前态, 今态, 应解除):
    assert 冷静期该解除(前态, 今态) is 应解除


def test_冷静期解除必须用不带粘性的态():
    """
    K3。粘性会把「未定」前向填成上一个方向, 于是 `未定 -> 干净同向` 这一跨越在粘性
    口径下比出来是「相同」, R2′ 直接失效。

    构造: 生 = [多, 未定, 多]。不带粘性时 未定->多 是跨越, 应解除;
    粘住之后变成 [多, 多, 多], 比出来处处相同, 一次都不解除。
    """
    生 = np.array([多, 未定, 多], dtype=np.int8)
    黏 = 粘住(生)
    assert list(黏) == [多, 多, 多], "粘住没有前向填充?"
    assert 冷静期该解除(生[1], 生[2]) is True, "不带粘性: 未定->多 必须解除"
    assert 冷静期该解除(黏[1], 黏[2]) is False, "粘性版给出了错误答案 —— 这正是 K3 要挡的"


# ================================================= 动量开关的段起点 =====
def test_动量开关段与薄包装逐位一致(收盘):
    dif, dea = macd线(收盘)
    多开, 空开 = 动量开关(dif, dea, 35)
    多开2, 空开2, _ = 动量开关段(dif, dea, 35)
    assert np.array_equal(多开, 多开2) and np.array_equal(空开, 空开2)


def test_段起点指向真正打开这一段的那根交叉(收盘):
    """
    段起点是 C2 的全部依据: 它决定「这一根的闩锁是不是解除之后才开的」。

    三条: 开着的根必有段起点、关着的恒为 -1、段起点那一根必须是一次**交叉**,
    且同一段内段起点恒定。
    """
    dif, dea = macd线(收盘)
    多开, 空开, 起 = 动量开关段(dif, dea, 35)
    金, 死 = 金叉死叉(dif, dea)
    开 = 多开 | 空开
    assert (起[开] >= 0).all(), "开着的根没有段起点"
    assert (起[~开] == -1).all(), "关着的根段起点不是 -1"
    for b in np.flatnonzero(开):
        s = int(起[b])
        assert 金[s] or 死[s], f"段起点 {s} 不是一次交叉"
        assert 开[s], "段起点那一根自己必须是开着的"
        assert 开[s:b + 1].all(), f"段 {s}..{b} 中间断过, 段起点不该跨过去"
    段变 = np.flatnonzero(开)
    assert len(set(起[段变].tolist())) > 1, "只有一段 —— 这条测试是空真的"
