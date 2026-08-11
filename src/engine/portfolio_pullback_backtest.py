"""
Week 6: multi-ticker portfolio layer on top of the single-ticker pullback
strategy (`engine.pullback_backtest`).

Each ticker runs its own independent `PullbackEntryEngine`/cooldown/signal
state - no cross-ticker coupling in the entry/exit logic itself. The
portfolio layer only decides how much of each ticker's own % return counts
each day, via a daily inverse-ATR-vol weight (`InverseVolatilitySizer` +
`rolling_daily_atr_vol`).

Weighting a ticker's OWN % return by that day's target weight is
mathematically the same as resizing an open position to the new weight at
the start of each day and letting it ride - % returns compose independently
of nominal position size, so this reproduces a real daily-resized share
ledger (like `engine.portfolio_backtest`'s) without needing to actually
track shares/cash. The one thing it doesn't capture is the cost of the
resize trades themselves, which `fee_pct` is a placeholder for (default 0.0
- no drag yet, wired in for when real per-trade costs are ready to plug in).
"""

from dataclasses import dataclass, field

import pandas as pd

from src.config import BacktestConfig
from src.engine.pullback_backtest import Trade, run_pullback_backtest
from src.engine.vol_estimator import daily_closes, rolling_daily_atr_vol
from src.performance.sharpe import sharpe_ratio
# Definitions moved to a leaf module so the v3.0 China-futures stack can reuse
# them without importing this file (which pulls in config/rules/resample).
# Imported under their old private names so nothing below this line changed.
from src.performance.trade_stats import (
    max_drawdown as _max_dd,
    payoff as _payoff,
    win_loss_avgs as _win_loss_avgs,
)
from src.rules.sizing import InverseVolatilitySizer


@dataclass
class DirectionStats:
    n_batches: int = 0
    win_rate: float = float("nan")
    payoff_ratio: float = float("nan")  # avg winning batch pnl_pct / abs(avg losing batch pnl_pct)
    # This direction's share of THIS TICKER's own gain/loss, as a fraction of
    # the ticker's starting capital - so long.contribution_pct +
    # short.contribution_pct == the ticker's total_return.
    #
    # It used to be a share of the PORTFOLIO's net profit, which inverts every
    # sign whenever the portfolio is down (a losing side divided by a negative
    # total reads as a positive contributor) and explodes as that total nears
    # zero. Dividing by starting capital keeps the denominator fixed and
    # positive, so a losing side always shows negative.
    contribution_pct: float = float("nan")


@dataclass
class BenchmarkStats:
    """Plain buy&hold stats for a single ticker - shown on the stats panel's
    third page, separate from the strategy's own performance."""
    buy_hold_return: float = float("nan")
    buy_hold_sharpe: float = float("nan")
    up_days: int = 0
    down_days: int = 0
    # Days whose close was EXACTLY unchanged. Counted separately so
    # up + down + flat reconciles to the day count - strict >0 / <0 tests
    # otherwise silently drop them (0-3 days per symbol here).
    flat_days: int = 0
    avg_up: float = float("nan")  # mean daily % move on up days
    avg_down: float = float("nan")  # mean daily % move on down days (negative)
    avg_total: float = float("nan")  # mean daily % move across every day (signed)


@dataclass
class TickerResult:
    trades: list[Trade]
    equity_curve: pd.Series
    sharpe: float
    total_return: float
    win_rate: float
    n_batches: int
    payoff_ratio: float = float("nan")  # across ALL batches, both directions
    avg_win: float = float("nan")  # mean winning batch, as a fraction of capital
    avg_loss: float = float("nan")  # mean losing batch (negative)
    max_drawdown: float = float("nan")  # worst peak-to-trough on this ticker's own curve
    benchmark: BenchmarkStats = field(default_factory=BenchmarkStats)
    long: DirectionStats = field(default_factory=DirectionStats)
    short: DirectionStats = field(default_factory=DirectionStats)
    # Observability only, for the chart's admin overlay (see PullbackBacktestResult).
    pullback_points: list = field(default_factory=list)
    cooldown_spans: list = field(default_factory=list)


@dataclass
class PortfolioBacktestResult:
    per_ticker: dict[str, TickerResult] = field(default_factory=dict)
    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_equity_curve: pd.Series = field(default_factory=pd.Series)
    portfolio_sharpe: float = float("nan")
    portfolio_return: float = float("nan")
    # Portfolio-level aggregates. These are deliberately NOT per-ticker values
    # combined: max drawdown is measured on the portfolio equity curve itself
    # (the worst the COMBINED book ever was - never max() of the per-ticker
    # drawdowns, which would be both wrong and pessimistic since tickers do not
    # bottom on the same day), and the benchmark stats come from the weighted
    # buy&hold curve rather than an average of the per-ticker benchmarks.
    portfolio_max_drawdown: float = float("nan")
    portfolio_win_rate: float = float("nan")
    portfolio_payoff: float = float("nan")
    portfolio_avg_win: float = float("nan")
    portfolio_avg_loss: float = float("nan")
    portfolio_n_batches: int = 0
    portfolio_long: DirectionStats = field(default_factory=DirectionStats)
    portfolio_short: DirectionStats = field(default_factory=DirectionStats)
    portfolio_benchmark: BenchmarkStats = field(default_factory=BenchmarkStats)
    # Same daily inverse-ATR weights as portfolio_equity_curve, but applied to
    # each ticker's plain buy-and-hold daily return instead of the strategy's
    # trade-driven return - isolates how much of the portfolio's return comes
    # from the vol-weighted rotation itself vs. the entry/exit timing skill.
    benchmark_equity_curve: pd.Series = field(default_factory=pd.Series)


def _batch_pnls(trades: list[Trade]) -> dict[int, tuple[str, float]]:
    """entry_bar_idx -> (direction, summed pnl_pct*size_fraction across that batch's legs)."""
    batches: dict[int, list[Trade]] = {}
    for t in trades:
        batches.setdefault(t.entry_bar_idx, []).append(t)
    return {
        idx: (legs[0].direction, sum(t.pnl_pct * t.size_fraction for t in legs))
        for idx, legs in batches.items()
    }


def _direction_by_day(df_5m: pd.DataFrame, trades: list[Trade], days: "pd.Index") -> dict:
    """
    date -> "long"/"short"/None for every date in `days`, derived from each
    batch's [entry_date, exit_date] span (exit_date = latest leg's exit,
    since a batch can close in two legs). Only one batch is ever open at a
    time (see `pullback_backtest.py`'s single-batch-at-a-time invariant), so
    these spans never overlap - each day maps to at most one direction.
    """
    batches: dict[int, list[Trade]] = {}
    for t in trades:
        batches.setdefault(t.entry_bar_idx, []).append(t)
    direction_by_day: dict = {}
    for entry_idx, legs in batches.items():
        entry_date = df_5m["timestamp"].iloc[entry_idx].date()
        exit_date = max(df_5m["timestamp"].iloc[t.exit_bar_idx].date() for t in legs)
        for d in days:
            if entry_date <= d <= exit_date:
                direction_by_day[d] = legs[0].direction
    return direction_by_day


def _benchmark_stats(daily: pd.Series, base: float | None = None) -> BenchmarkStats:
    """
    Buy&hold stats from a daily series. `base` is what the first day is measured
    against: the portfolio benchmark curve already has one day's return baked
    into its first point, so it needs its starting capital passed explicitly,
    whereas a plain price series measures from its own first close.

    sharpe_ratio only needs pct-change behaviour plus a date-like index, which a
    daily series already has - no synthetic equity curve required.
    """
    if not len(daily):
        return BenchmarkStats()
    prev = daily.shift(1)
    if base is not None:
        prev = prev.fillna(base)
    pct = (daily / prev - 1).dropna()
    start = float(base) if base is not None else float(daily.iloc[0])
    up, down, flat = pct[pct > 0], pct[pct < 0], pct[pct == 0]
    return BenchmarkStats(
        buy_hold_return=(float(daily.iloc[-1]) / start - 1) if start else float("nan"),
        buy_hold_sharpe=sharpe_ratio(daily),
        up_days=len(up), down_days=len(down), flat_days=len(flat),
        avg_up=up.mean() if len(up) else float("nan"),
        avg_down=down.mean() if len(down) else float("nan"),
        avg_total=pct.mean() if len(pct) else float("nan"),
    )


def _direction_contributions(
    df_5m: pd.DataFrame, trades: list[Trade], equity_curve: pd.Series, initial_capital: float,
) -> dict[str, float]:
    """
    Split this ticker's own gain/loss into long vs short, as a fraction of its
    STARTING CAPITAL - so the two returned values sum to the ticker's
    total_return (see DirectionStats.contribution_pct for why the denominator
    is not the portfolio's net profit).

    Exact by telescoping: each day's dollar change is equity(d) - equity(d-1),
    attributed to whichever direction held a position that day, and the daily
    changes sum to final - initial. Only days inside no batch's span are
    unattributed, and those are days with no position, i.e. zero change.
    """
    if equity_curve.empty:
        return {"long": float("nan"), "short": float("nan")}
    daily_eq = equity_curve.groupby(pd.DatetimeIndex(equity_curve.index).date).last()
    delta = daily_eq - daily_eq.shift(1).fillna(initial_capital)
    by_day = _direction_by_day(df_5m, trades, list(daily_eq.index))
    contrib = {"long": 0.0, "short": 0.0}
    for day, change in delta.items():
        direction = by_day.get(day)
        if change and direction is not None:
            contrib[direction] += float(change)
    return {k: v / initial_capital for k, v in contrib.items()}


def _direction_stats(batch_pnls: dict[int, tuple[str, float]], direction: str) -> DirectionStats:
    pnls = [pnl for d, pnl in batch_pnls.values() if d == direction]
    n = len(pnls)
    if n == 0:
        return DirectionStats(n_batches=0)
    win_rate = len([p for p in pnls if p > 0]) / n
    return DirectionStats(n_batches=n, win_rate=win_rate, payoff_ratio=_payoff(pnls))


def _ticker_result(df_5m: pd.DataFrame, initial_capital: float, config: BacktestConfig) -> TickerResult:
    result = run_pullback_backtest(df_5m, initial_capital=initial_capital, config=config)
    total_return = result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1
    sharpe = sharpe_ratio(result.equity_curve)
    batch_pnls = _batch_pnls(result.trades)
    n_batches = len(batch_pnls)
    wins = sum(1 for _d, pnl in batch_pnls.values() if pnl > 0)
    win_rate = wins / n_batches if n_batches else float("nan")
    benchmark = _benchmark_stats(daily_closes(df_5m))
    eq = result.equity_curve
    max_drawdown = _max_dd(eq)
    pnl_list = [pnl for _direction, pnl in batch_pnls.values()]
    avg_win, avg_loss = _win_loss_avgs(pnl_list)
    contrib = _direction_contributions(df_5m, result.trades, eq, initial_capital)
    long_stats = _direction_stats(batch_pnls, "long")
    short_stats = _direction_stats(batch_pnls, "short")
    long_stats.contribution_pct = contrib["long"]
    short_stats.contribution_pct = contrib["short"]
    return TickerResult(
        trades=result.trades, equity_curve=result.equity_curve, sharpe=sharpe,
        total_return=total_return, win_rate=win_rate, n_batches=n_batches,
        payoff_ratio=_payoff(pnl_list), avg_win=avg_win, avg_loss=avg_loss,
        max_drawdown=max_drawdown,
        benchmark=benchmark,
        long=long_stats, short=short_stats,
        pullback_points=result.pullback_points, cooldown_spans=result.cooldown_spans,
    )


def run_portfolio_pullback_backtest(
    dfs: dict[str, pd.DataFrame], fee_pct: float = 0.0, initial_capital: float = 10_000.0,
    config: BacktestConfig | None = None,
) -> PortfolioBacktestResult:
    config = config or BacktestConfig()
    symbols = list(dfs.keys())
    per_ticker = {s: _ticker_result(dfs[s], initial_capital, config) for s in symbols}

    daily_returns = {
        s: per_ticker[s].equity_curve.groupby(per_ticker[s].equity_curve.index.date).last().pct_change()
        for s in symbols
    }
    # Plain buy-and-hold close-to-close daily returns, for the benchmark
    # curve - same weights as the strategy, but no entry/exit timing at all.
    buy_hold_daily_returns = {s: daily_closes(dfs[s]).pct_change() for s in symbols}
    daily_vols = {s: rolling_daily_atr_vol(dfs[s]) for s in symbols}

    all_dates = sorted(set().union(*(set(daily_vols[s].dropna().index) for s in symbols)))
    sizer = InverseVolatilitySizer()

    # date -> "long"/"short"/None, per ticker - which direction (if any) was
    # actually in a position that day, so each day's dollar P&L can be
    # attributed to a direction exactly (see the contribution-dollar loop
    # below).
    direction_by_day = {
        s: _direction_by_day(dfs[s], per_ticker[s].trades, all_dates) for s in symbols
    }
    # Running dollar contribution per (ticker, direction) - filled in
    # alongside the main day loop below, using each day's actual
    # weight * return * prior-day-equity, which is an EXACT decomposition of
    # that day's dollar change in portfolio_equity_curve (by construction:
    # equity(d) - equity(d-1) = equity(d-1) * sum_s w_s(d) * r_s(d)). Summing
    # these across every ticker/direction therefore reproduces the
    # portfolio's total dollar profit exactly (fee drag, if any, is the only
    # unattributed residual) - unlike a naive sum of batch-level % pnl
    # against a compounded total return, which doesn't share a common
    # denominator and can blow up to numbers far past +-100%.
    contrib_dollar = {s: {"long": 0.0, "short": 0.0} for s in symbols}

    weight_rows = []
    port_returns = []
    benchmark_returns = []
    prev_weights: dict[str, float] = {}
    running_equity = initial_capital
    for d in all_dates:
        vols_today = {s: daily_vols[s].get(d) for s in symbols if pd.notna(daily_vols[s].get(d))}
        weights_today = sizer.weights(vols_today) if vols_today else {}
        turnover = sum(
            abs(weights_today.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in symbols
        )
        port_ret = sum(
            weights_today.get(s, 0.0) * daily_returns[s].get(d, 0.0)
            for s in symbols if pd.notna(daily_returns[s].get(d, 0.0))
        )
        port_ret -= fee_pct * turnover
        port_returns.append(port_ret)
        benchmark_returns.append(sum(
            weights_today.get(s, 0.0) * buy_hold_daily_returns[s].get(d, 0.0)
            for s in symbols if pd.notna(buy_hold_daily_returns[s].get(d, 0.0))
        ))

        equity_before = running_equity
        running_equity *= 1.0 + port_ret
        for s in symbols:
            r = daily_returns[s].get(d, 0.0)
            if not pd.notna(r) or r == 0.0:
                continue
            direction = direction_by_day[s].get(d)
            if direction is None:
                continue
            contrib_dollar[s][direction] += weights_today.get(s, 0.0) * r * equity_before

        # Plain `datetime.date` (not pd.Timestamp) - `_wire_weight_hover` in
        # chart.py (shared with the older portfolio_backtest.py engine, whose
        # weights_history is built the same way) looks up hovered dates via
        # `hovered_date not in weights_history.index` where hovered_date is
        # itself a plain `.date()` - a Timestamp-typed index would never
        # match.
        weight_rows.append({"date": d, **weights_today})
        prev_weights = weights_today

    dates_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in all_dates])
    weights_history = (
        pd.DataFrame(weight_rows).set_index("date") if weight_rows else pd.DataFrame()
    )
    port_ret_series = pd.Series(port_returns, index=dates_idx).fillna(0.0)
    portfolio_equity_curve = initial_capital * (1.0 + port_ret_series).cumprod()
    portfolio_sharpe = sharpe_ratio(portfolio_equity_curve)
    portfolio_return = portfolio_equity_curve.iloc[-1] / initial_capital - 1 if len(portfolio_equity_curve) else float("nan")

    benchmark_ret_series = pd.Series(benchmark_returns, index=dates_idx).fillna(0.0)
    benchmark_equity_curve = initial_capital * (1.0 + benchmark_ret_series).cumprod()

    # NOTE: contribution_pct is set per ticker in `_ticker_result`, against that
    # ticker's own starting capital. It is deliberately NOT a share of the
    # portfolio's net profit - that denominator flips every sign whenever the
    # portfolio is down and blows up as it approaches zero.

    # Pooled trade stats across every ticker: "of all the trades the strategy
    # took, how many won". Honest as stated, but note it is UNWEIGHTED - a trade
    # on a small-weight ticker counts the same as one on a large-weight ticker,
    # so this is a statement about the rule set, not about the P&L.
    pooled = dict(enumerate(
        pnl for s in symbols for pnl in _batch_pnls(per_ticker[s].trades).values()
    ))
    pooled_pnls = [pnl for _direction, pnl in pooled.values()]

    return PortfolioBacktestResult(
        per_ticker=per_ticker, weights_history=weights_history,
        portfolio_equity_curve=portfolio_equity_curve,
        portfolio_sharpe=portfolio_sharpe, portfolio_return=portfolio_return,
        portfolio_max_drawdown=_max_dd(portfolio_equity_curve),
        portfolio_win_rate=(
            len([p for p in pooled_pnls if p > 0]) / len(pooled_pnls)
            if pooled_pnls else float("nan")
        ),
        portfolio_payoff=_payoff(pooled_pnls),
        portfolio_avg_win=_win_loss_avgs(pooled_pnls)[0],
        portfolio_avg_loss=_win_loss_avgs(pooled_pnls)[1],
        portfolio_n_batches=len(pooled_pnls),
        portfolio_long=_direction_stats(pooled, "long"),
        portfolio_short=_direction_stats(pooled, "short"),
        portfolio_benchmark=_benchmark_stats(benchmark_equity_curve, base=initial_capital),
        benchmark_equity_curve=benchmark_equity_curve,
    )
