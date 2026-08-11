"""
The five session-partition modes (`src/data/v3_sessions.py`).

「一个时段最多一笔 + 时段结束强制平仓」是策略的核心节流阀, 而「时段」怎么定义
直接决定了一天做几笔、持仓能多长、强平在哪一刻。这些测试钉住五种定义。

The two load-bearing tests:
  * `test_split_mode_reproduces_the_historical_derive_sessions` — mode 5 must be
    byte-identical to what shipped, or every existing result silently moves.
  * `test_day_night_pins_the_line_index_to_the_day_part` — mode 4 is the only
    mode whose session spans two trading_dates; if the line index were read per
    bar the stop would jump at 21:00 under an open position.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.v3_sessions import (
    DEFAULT_SESSION_MODE,
    SESSION_MODES,
    apply_session_mode,
    assert_segment_shape,
    derive_segments,
    derive_sessions,
    segment_is_night,
)
from src.engine.rbreaker_backtest import run_rbreaker_backtest
from src.engine.rbreaker_reversal_backtest import run_rbreaker_reversal_backtest

REF = (110.0, 90.0, 100.0)
TD1, TD2 = "2025-07-02", "2025-07-03"


def _night_and_day(raw_frame, bar_run, td, night_eve, day_date, price=100.0):
    """A full Chinese trading day: night session the previous evening, then day."""
    return (bar_run(f"{night_eve} 21:00", 121, td, price=price)
            + bar_run(f"{day_date} 09:01", 75, td, price=price)
            + bar_run(f"{day_date} 10:31", 60, td, price=price)
            + bar_run(f"{day_date} 13:31", 90, td, price=price))


def _day_only(bar_run, td, day_date, price=100.0):
    return (bar_run(f"{day_date} 09:00", 76, td, price=price)
            + bar_run(f"{day_date} 10:31", 60, td, price=price)
            + bar_run(f"{day_date} 13:31", 90, td, price=price))


# ------------------------------------------------------------ 夜/日 分类 ----
def test_last_segment_of_a_trading_date_is_the_day_segment(raw_frame, bar_run):
    """
    分类规则不读时钟: `trading_date` 总是以白盘收尾, 所以末段即白盘段。
    这条自动覆盖了无夜盘品种、中金所、2020 年 2-5 月全市场停夜盘。
    """
    df = raw_frame(_night_and_day(raw_frame, bar_run, TD1, "2025-07-01", TD1))
    seg = derive_segments(df)
    is_night = segment_is_night(df, seg)
    assert len(np.unique(seg)) == 2
    assert is_night[:121].all(), "the 21:00 run must be the night segment"
    assert not is_night[121:].any(), "everything after the 10h gap is the day segment"


def test_single_segment_day_is_classified_as_day(raw_frame, bar_run):
    """无夜盘品种 / 中金所 / 2020 年停夜盘期: 只有 1 段, 必须判为白盘。"""
    df = raw_frame(_day_only(bar_run, TD1, TD1))
    seg = derive_segments(df)
    assert not segment_is_night(df, seg).any()


def test_segment_shape_assertion_catches_a_misclassified_run(raw_frame, bar_run):
    """
    时钟只用来**校验**。构造一个 04:30 起的段跟在另一段后面 —— 它会被「末段即
    白盘」规则判成夜盘段, 而 04:30 不在 [20:00, 04:00) 里, 断言必须炸。

    真实世界里这对应「白盘段被 drop_dead_runs 整段清掉、只剩夜盘段」。
    """
    bars = (bar_run("2025-07-02 04:30", 30, TD1) + bar_run("2025-07-02 09:00", 30, TD1))
    df = raw_frame(bars)
    seg = derive_segments(df)
    with pytest.raises(ValueError, match="末段即白盘"):
        assert_segment_shape(df, seg, segment_is_night(df, seg), "TEST")


def test_normal_shapes_do_not_trip_the_segment_assertion(raw_frame, bar_run):
    for bars in (_night_and_day(raw_frame, bar_run, TD1, "2025-07-01", TD1),
                 _day_only(bar_run, TD1, TD1)):
        df = raw_frame(bars)
        seg = derive_segments(df)
        assert_segment_shape(df, seg, segment_is_night(df, seg), "TEST")


# ------------------------------------------------------------- 五种分组 ----
def _modes(raw_frame, bars):
    df = raw_frame(bars)
    return {m: apply_session_mode(df, m) for m in SESSION_MODES}


def test_split_mode_reproduces_the_historical_derive_sessions(raw_frame, bar_run):
    """
    模式 5 必须与历史行为逐值相同 —— 否则已经跑出来的所有结果都悄悄变了。
    """
    bars = (_night_and_day(raw_frame, bar_run, TD1, "2025-07-01", TD1)
            + _night_and_day(raw_frame, bar_run, TD2, "2025-07-02", TD2))
    df = raw_frame(bars)
    old = derive_sessions(df)
    new = apply_session_mode(df, "split")
    for col in ("session_id", "is_session_first", "is_session_last"):
        assert (old[col].to_numpy() == new[col].to_numpy()).all(), col
    assert new["tradable"].all()
    assert DEFAULT_SESSION_MODE == "split"


def test_split_gives_two_sessions_per_trading_date(raw_frame, bar_run):
    out = _modes(raw_frame, _night_and_day(raw_frame, bar_run, TD1, "2025-07-01", TD1))["split"]
    assert out["session_id"].nunique() == 2


def test_night_day_merges_the_whole_trading_date(raw_frame, bar_run):
    """模式 3: 夜+日合成一个时段, 强平只发生在 15:00。"""
    out = _modes(raw_frame, _night_and_day(raw_frame, bar_run, TD1, "2025-07-01", TD1))["night_day"]
    assert out["session_id"].nunique() == 1
    assert out["tradable"].all()
    last = out.index[out["is_session_last"]]
    assert list(last) == [pd.Timestamp(f"{TD1} 15:00")]
    first = out.index[out["is_session_first"]]
    assert list(first) == [pd.Timestamp("2025-07-01 21:00")]


def test_night_only_marks_just_the_night_segment_tradable(raw_frame, bar_run):
    out = _modes(raw_frame, _night_and_day(raw_frame, bar_run, TD1, "2025-07-01", TD1))["night"]
    assert out["session_id"].nunique() == 2, "分段不变, 只是白盘段不可交易"
    assert out["tradable"].to_numpy()[:121].all()
    assert not out["tradable"].to_numpy()[121:].any()


def test_day_only_marks_just_the_day_segment_tradable(raw_frame, bar_run):
    out = _modes(raw_frame, _night_and_day(raw_frame, bar_run, TD1, "2025-07-01", TD1))["day"]
    assert not out["tradable"].to_numpy()[:121].any()
    assert out["tradable"].to_numpy()[121:].all()


def test_night_only_on_a_symbol_without_a_night_session(raw_frame, bar_run):
    """
    无夜盘的 31 个品种在「只做夜盘」下应当整段不可交易 —— 决策 2 要求如实显示,
    而不是回退到日盘或隐藏该品种。
    """
    out = _modes(raw_frame, _day_only(bar_run, TD1, TD1))["night"]
    assert not out["tradable"].any()


def test_day_night_spans_two_trading_dates(raw_frame, bar_run):
    """
    模式 4: 白盘 T + 紧随其后的夜盘 (其 trading_date 是 T+1) 合成一个时段。
    这是**唯一**会跨 trading_date 的模式。
    """
    bars = (_day_only(bar_run, TD1, TD1)
            + bar_run(f"{TD1} 21:00", 121, TD2)      # 当晚夜盘, 属于次日
            + bar_run(f"{TD2} 09:01", 75, TD2))
    out = _modes(raw_frame, bars)["day_night"]
    sid = out["session_id"].to_numpy()
    td = out["trading_date"].to_numpy()

    # 白盘 TD1 (226 根) + 夜盘 TD2 (121 根) 必须同属一个时段
    assert sid[0] == sid[225] == sid[226] == sid[346], "day + following night not merged"
    assert sid[347] != sid[346], "the next day session must start a new session"
    spanning = pd.Series(td).groupby(sid).nunique()
    assert (spanning == 2).any(), "no session spans two trading_dates"
    # 强平落在夜盘收盘, 不是 15:00
    last = out.index[out["is_session_last"]]
    assert pd.Timestamp(f"{TD1} 23:00") in set(last)
    assert pd.Timestamp(f"{TD1} 15:00") not in set(last)


def test_day_night_friday_night_belongs_to_monday(raw_frame, bar_run):
    """
    周五白盘 + 周五晚夜盘 (其 trading_date 是下周一) 是一个时段。分组键靠
    「夜盘挂到前一个 trading_date」实现, 所以跨周末也对。
    """
    fri, mon = "2025-07-04", "2025-07-07"
    bars = (_day_only(bar_run, fri, fri)
            + bar_run(f"{fri} 21:00", 121, mon)      # 周五晚, trading_date = 周一
            + bar_run(f"{mon} 09:01", 75, mon))
    out = _modes(raw_frame, bars)["day_night"]
    sid = out["session_id"].to_numpy()
    assert sid[0] == sid[346], "Friday day + Friday night must be one session"
    assert sid[347] != sid[346]


def test_every_mode_produces_contiguous_sessions(raw_frame, bar_run):
    """时段必须是一段连续的 bar —— 回测循环完全依赖这条。"""
    bars = (_night_and_day(raw_frame, bar_run, TD1, "2025-07-01", TD1)
            + _night_and_day(raw_frame, bar_run, TD2, "2025-07-02", TD2))
    for mode, out in _modes(raw_frame, bars).items():
        sid = out["session_id"].to_numpy()
        assert (np.diff(sid) >= 0).all(), f"{mode}: session ids not monotone"
        n_runs = int((np.diff(sid, prepend=sid[0] - 1) != 0).sum())
        assert n_runs == len(np.unique(sid)), f"{mode}: a session is split in two"
        first = out["is_session_first"].to_numpy()
        last = out["is_session_last"].to_numpy()
        assert int(first.sum()) == int(last.sum()) == len(np.unique(sid)), mode
        assert last[-1], f"{mode}: the frame's final bar must force a flat"
        # tradable is constant inside a session
        assert (pd.Series(out["tradable"].to_numpy()).groupby(sid).nunique() == 1).all(), mode


def test_unknown_mode_raises(raw_frame, bar_run):
    df = raw_frame(_day_only(bar_run, TD1, TD1))
    with pytest.raises(ValueError, match="未知的时段模式"):
        apply_session_mode(df, "nonsense")


# ------------------------------------------------------------- 引擎行为 ----
def _breakout_day(bar_run, td, day_date, start_price=100.0):
    """A day session containing one clean upside breakout."""
    run = bar_run(f"{day_date} 09:01", 40, td, price=start_price)
    run[3] = (run[3][0], td, 100.0, 121.0, 100.0, 121.0, 10.0)     # > Bbreak 120.25
    return [(t, d, o, h, low, c, v) for (t, d, o, h, low, c, v) in run]


def test_night_day_allows_at_most_one_trade_per_trading_date(make_prepared, bar_run):
    """
    模式 3 把一天合成一个时段, 所以「每时段一笔」变成「每交易日一笔」——
    模式 5 下同样的数据可以做两笔。
    """
    night = bar_run("2025-07-01 21:00", 40, TD1, price=100.0)
    night[3] = (night[3][0], TD1, 100.0, 121.0, 100.0, 121.0, 10.0)
    day = _breakout_day(bar_run, TD1, TD1)

    split = run_rbreaker_backtest(make_prepared(night + day, {TD1: REF}, mode="split"))
    merged = run_rbreaker_backtest(make_prepared(night + day, {TD1: REF}, mode="night_day"))
    assert len(split.trades) == 2, "split should take the night and the day breakout"
    assert len(merged.trades) == 1, "night_day must throttle to one trade for the whole day"


def test_night_only_never_enters_in_the_day_session(make_prepared, bar_run):
    night = bar_run("2025-07-01 21:00", 40, TD1, price=100.0)
    day = _breakout_day(bar_run, TD1, TD1)          # the breakout is in the DAY part
    r = run_rbreaker_backtest(make_prepared(night + day, {TD1: REF}, mode="night"))
    assert r.trades == [], "entered a day-session breakout while in night-only mode"


def test_day_only_never_enters_in_the_night_session(make_prepared, bar_run):
    night = bar_run("2025-07-01 21:00", 40, TD1, price=100.0)
    night[3] = (night[3][0], TD1, 100.0, 121.0, 100.0, 121.0, 10.0)   # breakout at night
    day = bar_run(f"{TD1} 09:01", 40, TD1, price=100.0)
    r = run_rbreaker_backtest(make_prepared(night + day, {TD1: REF}, mode="day"))
    assert r.trades == []


def test_day_night_pins_the_line_index_to_the_day_part(make_prepared, bar_run):
    """
    模式 4 是唯一跨 trading_date 的模式。若线的下标按 bar 读, 21:00 一到就会
    跳到次日那组线, 持仓中的止损位随之平移 —— 与 `theo_entry` 不自洽。

    这里给两天**不同**的参照 OHLC, 所以两组线数值不同, 一跳就看得见。
    """
    day1 = _breakout_day(bar_run, TD1, TD1)
    night2 = bar_run(f"{TD1} 21:00", 60, TD2, price=121.0)
    day2 = bar_run(f"{TD2} 09:01", 30, TD2, price=121.0)
    refs = {TD1: REF, TD2: (150.0, 130.0, 140.0)}   # deliberately far apart

    p = make_prepared(day1 + night2 + day2, refs, mode="day_night")
    sid = p.df["session_id"].to_numpy()
    li = p.line_idx
    # the session holding both day1 and night2 must use ONE line index
    s0 = sid[0]
    assert (sid[:len(day1) + len(night2)] == s0).all(), "day1 + night2 are not one session"
    assert len(set(li[sid == s0])) == 1, "line_idx jumps inside the session"
    assert li[0] == 0, "pinned to the FIRST bar's trading_date (the day part)"

    r = run_rbreaker_backtest(p)
    assert len(r.trades) == 1
    t = r.trades[0]
    seg = r.trail_stop[t.entry_bar_idx:t.exit_bar_idx + 1]
    assert np.isfinite(seg).all()
    # a long's trail may only ratchet up; a mid-session line swap would break it
    assert (np.diff(seg) >= -1e-9).all(), "stop moved backwards - the lines swapped mid-trade"


def test_no_position_survives_a_session_boundary_in_any_mode(make_prepared, bar_run):
    night = bar_run("2025-07-01 21:00", 40, TD1, price=100.0)
    night[3] = (night[3][0], TD1, 100.0, 121.0, 100.0, 121.0, 10.0)
    day = _breakout_day(bar_run, TD1, TD1)
    for mode in SESSION_MODES:
        p = make_prepared(night + day, {TD1: REF}, mode=mode)
        sid = p.df["session_id"].to_numpy()
        trad = p.df["tradable"].to_numpy()
        for runner in (run_rbreaker_backtest, run_rbreaker_reversal_backtest):
            r = runner(p)
            assert r.position[-1] == 0, f"{mode}: ended holding"
            for t in r.trades:
                assert sid[t.entry_bar_idx] == sid[t.exit_bar_idx], f"{mode}: crossed a session"
                assert trad[t.entry_bar_idx], f"{mode}: entered an untradable session"
            assert len({t.session_id for t in r.trades}) == len(r.trades), f"{mode}: >1/session"


# ------------------------------------------------------------- 图表阶梯线 ----
def _two_night_days(bar_run):
    """两个完整交易日 (夜+日), 参照 OHLC 不同 -> 两天的线数值不同。"""
    return (bar_run("2025-07-01 21:00", 121, TD1) + bar_run(f"{TD1} 09:01", 75, TD1)
            + bar_run(f"{TD1} 21:00", 121, TD2) + bar_run(f"{TD2} 09:01", 75, TD2))


def test_step_lines_have_no_bare_vertical_in_any_mode(make_prepared, bar_run):
    """
    阶梯模式下一个点的颜色只管**从它出发的水平段**, 而 `x_h` 处的垂直跳变用的是
    **点 h 自己的**颜色。所以「隐藏点 + 可见点」会画出一根光秃秃的竖线 —— 视觉上
    就是每段开头一个 L / 反 L。

    不变量: 可见的垂直跳变必须挂在可见的水平段上。

        visible(h) and value(h-1) != value(h)  ==>  visible(h-1)

    这条曾经在「只做夜盘」模式下被破坏 (白盘段被涂透明, 于是 21:00 的跳变变成
    孤零零一根竖线), 六条线全成了 L 形。
    """
    from src.viz.chart_rbreaker import HIDDEN, step_rows

    refs = {TD1: REF, TD2: (150.0, 130.0, 140.0)}   # 两天线距拉开, 跳变才看得见
    for mode in SESSION_MODES:
        p = make_prepared(_two_night_days(bar_run), refs, mode=mode)
        for name in ("Bbreak", "Sbreak", "Ssetup", "Bsetup", "Senter", "Benter"):
            f = step_rows(p, name)
            v = f[name].to_numpy(float)
            vis = f["color"].to_numpy() != HIDDEN
            bare = vis[1:] & (v[1:] != v[:-1]) & ~vis[:-1]
            assert not bare.any(), f"{mode}/{name}: 出现了 {int(bare.sum())} 根光秃秃的竖线"


def test_step_lines_are_flat_inside_a_trading_date(make_prepared, bar_run):
    """
    除 `day_night` 外, 同一个 trading_date 的所有时段共用同一组线, 所以它们的
    阶梯点取值必须相同 —— 21:00 -> 09:01 之间不该有台阶, 线是平的。
    """
    from src.viz.chart_rbreaker import step_rows

    refs = {TD1: REF, TD2: (150.0, 130.0, 140.0)}
    for mode in ("split", "night", "day", "night_day"):
        p = make_prepared(_two_night_days(bar_run), refs, mode=mode)
        f = step_rows(p, "Bbreak")
        first_pos = np.flatnonzero(p.df["is_session_first"].to_numpy())
        td = p.df["trading_date"].to_numpy()[first_pos]
        v = f["Bbreak"].to_numpy(float)[:len(first_pos)]
        per_day = pd.Series(v).groupby(td).nunique()
        assert (per_day == 1).all(), f"{mode}: 交易日内部出现台阶"


def test_step_lines_stay_visible_in_untradable_sessions(make_prepared, bar_run):
    """
    决策 2: 不可交易的时段照常画线。线是「昨日投影」这个日历事实的属性, 与本
    模式下能不能交易无关 —— 「只做夜盘」时白盘段仍要看得见这一天的六条线,
    只是不会有成交箭头。
    """
    from src.viz.chart_rbreaker import HIDDEN, step_rows

    refs = {TD1: REF, TD2: REF}
    p = make_prepared(_two_night_days(bar_run), refs, mode="night")
    assert not p.df["tradable"].all(), "fixture 应该有不可交易的白盘段"
    f = step_rows(p, "Bbreak")
    assert (f["color"].to_numpy() != HIDDEN).all(), "有效日的线被藏起来了"


def test_tradable_days_reflects_the_mode(make_prepared, bar_run):
    """
    `可交易日` 在「只做夜盘」+ 无夜盘品种下必须是 0 —— 如实反映「这个品种在这个
    模式下无法交易」, 而不是假装有交易日却一笔不做。
    """
    bars = _day_only(bar_run, TD1, TD1) + _day_only(bar_run, TD2, TD2)
    refs = {TD1: REF, TD2: REF}
    assert make_prepared(bars, refs, mode="day").tradable_days == 2
    assert make_prepared(bars, refs, mode="split").tradable_days == 2
    assert make_prepared(bars, refs, mode="night").tradable_days == 0
