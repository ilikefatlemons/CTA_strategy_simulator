"""
Stats conventions (`src/performance/trade_stats.py`, `sharpe.py`,
`rbreaker_stats.py`).

Two things are pinned here that are easy to "fix" into being wrong later:
the `p <= 0` loss bucket (shared with v2.1), and Sharpe grouping by
`trading_date` rather than by calendar date.
"""

import numpy as np
import pandas as pd
import pytest

from src.engine.rbreaker_backtest import run_rbreaker_backtest
from src.performance.rbreaker_stats import compute_stats, panel_rows, panel_title
from src.performance.sharpe import (
    TRADING_DAYS_PER_YEAR_CN,
    daily_returns,
    session_returns,
    sharpe_ratio_cn,
)
from src.performance.trade_stats import max_drawdown, payoff, win_loss_avgs, win_rate

REF = (110.0, 90.0, 100.0)
TD = "2025-07-02"


# ------------------------------------------------------------ trade_stats ----
def test_a_flat_trade_counts_as_a_loss_not_a_win():
    """
    The `p <= 0` bucket is shared with v2.1 (`_win_loss_avgs`). It keeps win
    rate and payoff consistent with each other: a zero-P&L trade drags the
    average loss but never inflates the win rate.
    """
    pnls = [0.02, 0.0, -0.01]
    avg_win, avg_loss = win_loss_avgs(pnls)
    assert avg_win == pytest.approx(0.02)
    assert avg_loss == pytest.approx((0.0 + -0.01) / 2)
    assert win_rate(pnls) == pytest.approx(1 / 3)


def test_payoff_is_avg_win_over_abs_avg_loss():
    assert payoff([0.03, 0.01, -0.01, -0.01]) == pytest.approx(0.02 / 0.01)


def test_payoff_is_nan_when_one_side_is_missing():
    assert np.isnan(payoff([0.01, 0.02]))
    assert np.isnan(payoff([-0.01, -0.02]))
    assert np.isnan(payoff([]))


def test_win_rate_is_nan_on_an_empty_list():
    assert np.isnan(win_rate([]))


def test_max_drawdown_is_a_negative_fraction():
    curve = pd.Series([1.0, 1.2, 0.9, 1.1])
    assert max_drawdown(curve) == pytest.approx(0.9 / 1.2 - 1.0)
    assert max_drawdown(curve) < 0
    assert np.isnan(max_drawdown(pd.Series(dtype=float)))


def test_extracted_helpers_still_match_the_v21_call_sites():
    """
    `portfolio_pullback_backtest` imports these under their old private names.
    If the extraction ever drifts, v2.1's numbers move silently.
    """
    from src.engine import portfolio_pullback_backtest as ppb

    assert ppb._payoff is payoff
    assert ppb._win_loss_avgs is win_loss_avgs
    assert ppb._max_dd is max_drawdown


# ---------------------------------------------------------------- sharpe ----
def _night_curve():
    """
    One trading_date whose bars span two calendar dates: a 21:00-23:00 night
    session on 2025-07-01 plus a 09:01 day session on 2025-07-02, both carrying
    trading_date 2025-07-02. Plus a second, day-only trading date.
    """
    times = (
        list(pd.date_range("2025-07-01 21:00", periods=60, freq="1min"))
        + list(pd.date_range("2025-07-02 09:01", periods=60, freq="1min"))
        + list(pd.date_range("2025-07-03 09:01", periods=60, freq="1min"))
    )
    tds = ([pd.Timestamp("2025-07-02")] * 120) + ([pd.Timestamp("2025-07-03")] * 60)
    idx = pd.DatetimeIndex(times)
    eq = pd.Series(np.linspace(1.0, 1.1, len(times)), index=idx)
    return eq, pd.Series(tds, index=idx)


def test_sharpe_groups_by_trading_date_not_calendar_date():
    """
    Night bars carry a calendar date one day BEFORE their trading_date, so
    calendar grouping splits a trading day in two. Here 2 trading dates span 3
    calendar dates - the two groupings must disagree.
    """
    eq, td = _night_curve()
    assert td.nunique() == 2
    assert pd.DatetimeIndex(eq.index).normalize().nunique() == 3
    assert len(session_returns(eq, td)) == 1, "n_trading_dates - 1"
    assert len(daily_returns(eq)) == 2, "calendar grouping sees one day too many"


def test_session_returns_rejects_a_misaligned_trading_date():
    eq, td = _night_curve()
    with pytest.raises(ValueError):
        session_returns(eq, td.iloc[:-1])


def test_sharpe_annualises_with_243_sessions():
    """
    Measured, not assumed: RB has 4,017 trading dates over 16.57 years.
    A curve with a constant daily return has std 0, so build one with noise
    and check the ratio against a hand-computed annualisation.
    """
    rng = np.random.default_rng(7)
    n = 200
    tds = pd.date_range("2025-01-01", periods=n, freq="D")
    rets = rng.normal(0.001, 0.01, n)
    eq = pd.Series(np.cumprod(1 + rets), index=tds)
    got = sharpe_ratio_cn(eq, pd.Series(tds, index=tds))
    r = pd.Series(eq.to_numpy()).pct_change().dropna()
    want = r.mean() / r.std() * (TRADING_DAYS_PER_YEAR_CN ** 0.5)
    assert TRADING_DAYS_PER_YEAR_CN == 243
    assert got == pytest.approx(want, rel=1e-12)


def test_sharpe_is_nan_without_at_least_two_days():
    idx = pd.date_range("2025-07-02 09:00", periods=5, freq="1min")
    eq = pd.Series(np.ones(5), index=idx)
    td = pd.Series([pd.Timestamp("2025-07-02")] * 5, index=idx)
    assert np.isnan(sharpe_ratio_cn(eq, td))


def test_sharpe_is_nan_when_every_day_is_identical():
    idx = pd.date_range("2025-07-01", periods=10, freq="D")
    eq = pd.Series(np.ones(10), index=idx)
    assert np.isnan(sharpe_ratio_cn(eq, pd.Series(idx, index=idx)))


# ----------------------------------------------------------- panel rows ----
def _one_trade_result(make_prepared):
    times = pd.date_range(f"{TD} 09:00", periods=11, freq="1min")
    prices = ([(100.0, 100.0, 100.0, 100.0)] * 2
              + [(100.0, 121.0, 100.0, 121.0)]
              + [(121.0, 122.0, 121.0, 121.0)] * 8)
    bars = [(t, TD, o, h, low, c, 10.0) for t, (o, h, low, c) in zip(times, prices)]
    return run_rbreaker_backtest(make_prepared(bars, {TD: REF}))


def test_panel_has_exactly_the_nine_specified_rows(make_prepared):
    """
    Order is fixed and matches the reference layout, minus 平均R and 止损宽度,
    plus 策略夏普比.
    """
    stats = compute_stats(_one_trade_result(make_prepared))
    labels = [k for k, _ in panel_rows(stats)]
    assert labels == [
        "可交易日", "成交", "累计净收益", "最大回撤", "胜率",
        "盈亏比", "策略夏普比", "离场 止损/收盘", "平均持有",
    ]
    assert "平均R" not in labels and "止损宽度" not in labels
    assert panel_title(stats).endswith("R-Breaker 突破")


def test_panel_renders_na_rather_than_crashing_on_degenerate_input(make_prepared):
    """A single winning trade has no losing side, so payoff is nan."""
    stats = compute_stats(_one_trade_result(make_prepared))
    values = dict(panel_rows(stats))
    assert values["盈亏比"] == "n/a"
    assert all(v for v in values.values()), values


def test_max_drawdown_is_displayed_unsigned(make_prepared):
    stats = compute_stats(_one_trade_result(make_prepared))
    shown = dict(panel_rows(stats))["最大回撤"]
    assert not shown.startswith("-"), "the panel takes abs(); the stat itself stays negative"
    assert stats.max_drawdown <= 0


def test_exit_reason_counts_add_up(make_prepared):
    stats = compute_stats(_one_trade_result(make_prepared))
    assert stats.n_exit_trail + stats.n_exit_session == stats.n_trades
    assert stats.n_long + stats.n_short == stats.n_trades


def test_coefficients_are_carried_into_the_panel(make_prepared):
    """
    R-Breaker has no canonical published formula, so every reported result
    must state which reconstruction produced it.
    """
    from src.performance.rbreaker_stats import panel_subtitle

    stats = compute_stats(_one_trade_result(make_prepared))
    sub = panel_subtitle(stats)
    for token in ("f1=0.35", "f2=0.07", "f3=0.25", "d=3"):
        assert token in sub, sub
