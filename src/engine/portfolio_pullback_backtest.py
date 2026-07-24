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

from src.engine.pullback_backtest import Trade, run_pullback_backtest
from src.engine.vol_estimator import daily_closes, rolling_daily_atr_vol
from src.performance.sharpe import sharpe_ratio
from src.rules.sizing import InverseVolatilitySizer


@dataclass
class DirectionStats:
    n_batches: int = 0
    win_rate: float = float("nan")
    payoff_ratio: float = float("nan")  # avg winning batch pnl_pct / abs(avg losing batch pnl_pct)
    # This direction's share of the PORTFOLIO's total profit, as a fraction
    # (e.g. 0.0654 = "6.54% of total profit") - matches win_rate's
    # fraction convention, not a pre-multiplied percentage.
    contribution_pct: float = float("nan")


@dataclass
class BenchmarkStats:
    """Plain buy&hold stats for a single ticker - shown on the stats panel's
    third page, separate from the strategy's own performance."""
    buy_hold_return: float = float("nan")
    buy_hold_sharpe: float = float("nan")
    up_days: int = 0
    down_days: int = 0
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
    benchmark: BenchmarkStats = field(default_factory=BenchmarkStats)
    long: DirectionStats = field(default_factory=DirectionStats)
    short: DirectionStats = field(default_factory=DirectionStats)


@dataclass
class PortfolioBacktestResult:
    per_ticker: dict[str, TickerResult] = field(default_factory=dict)
    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_equity_curve: pd.Series = field(default_factory=pd.Series)
    portfolio_sharpe: float = float("nan")
    portfolio_return: float = float("nan")
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


def _direction_stats(batch_pnls: dict[int, tuple[str, float]], direction: str) -> DirectionStats:
    pnls = [pnl for d, pnl in batch_pnls.values() if d == direction]
    n = len(pnls)
    if n == 0:
        return DirectionStats(n_batches=0)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else float("nan")
    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    payoff_ratio = (
        avg_win / abs(avg_loss) if wins and losses and avg_loss != 0 else float("nan")
    )
    return DirectionStats(n_batches=n, win_rate=win_rate, payoff_ratio=payoff_ratio)


def _ticker_result(df_5m: pd.DataFrame, initial_capital: float) -> TickerResult:
    result = run_pullback_backtest(df_5m, initial_capital=initial_capital)
    total_return = result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1
    sharpe = sharpe_ratio(result.equity_curve)
    batch_pnls = _batch_pnls(result.trades)
    n_batches = len(batch_pnls)
    wins = sum(1 for _d, pnl in batch_pnls.values() if pnl > 0)
    win_rate = wins / n_batches if n_batches else float("nan")
    closes = daily_closes(df_5m)
    buy_hold_return = closes.iloc[-1] / closes.iloc[0] - 1 if len(closes) else float("nan")
    # sharpe_ratio only needs pct-change behavior + a date-like index, which
    # a plain daily close series already has - no need for a synthetic
    # buy&hold equity curve.
    buy_hold_sharpe = sharpe_ratio(closes)
    daily_pct = closes.pct_change().dropna()
    up = daily_pct[daily_pct > 0]
    down = daily_pct[daily_pct < 0]
    benchmark = BenchmarkStats(
        buy_hold_return=buy_hold_return, buy_hold_sharpe=buy_hold_sharpe,
        up_days=len(up), down_days=len(down),
        avg_up=up.mean() if len(up) else float("nan"),
        avg_down=down.mean() if len(down) else float("nan"),
        avg_total=daily_pct.mean() if len(daily_pct) else float("nan"),
    )
    return TickerResult(
        trades=result.trades, equity_curve=result.equity_curve, sharpe=sharpe,
        total_return=total_return, win_rate=win_rate, n_batches=n_batches,
        benchmark=benchmark,
        long=_direction_stats(batch_pnls, "long"), short=_direction_stats(batch_pnls, "short"),
    )


def run_portfolio_pullback_backtest(
    dfs: dict[str, pd.DataFrame], fee_pct: float = 0.0, initial_capital: float = 10_000.0,
) -> PortfolioBacktestResult:
    symbols = list(dfs.keys())
    per_ticker = {s: _ticker_result(dfs[s], initial_capital) for s in symbols}

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

    total_profit_dollar = portfolio_equity_curve.iloc[-1] - initial_capital if len(portfolio_equity_curve) else 0.0
    if total_profit_dollar:
        for s in symbols:
            # Fraction (not already *100), matching win_rate/buy_hold_return's
            # convention - chart.py's `_fmt_pct` applies the `:.1%` format
            # spec, which multiplies by 100 itself.
            per_ticker[s].long.contribution_pct = contrib_dollar[s]["long"] / total_profit_dollar
            per_ticker[s].short.contribution_pct = contrib_dollar[s]["short"] / total_profit_dollar

    return PortfolioBacktestResult(
        per_ticker=per_ticker, weights_history=weights_history,
        portfolio_equity_curve=portfolio_equity_curve,
        portfolio_sharpe=portfolio_sharpe, portfolio_return=portfolio_return,
        benchmark_equity_curve=benchmark_equity_curve,
    )
