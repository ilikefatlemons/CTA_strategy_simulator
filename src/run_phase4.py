"""
Phase 4 entry point: same MACD entry / ATR TP-SL exit / cooldown re-entry
rules as Phase 3, now reporting Sharpe ratio at the portfolio level and at
the per-symbol level, to check whether vol-targeting actually flattens each
symbol's risk contribution.
"""

import pandas as pd

from src.engine.backtest import Strategy, run_backtest
from src.engine.portfolio_backtest import align_to_common_index, run_portfolio_backtest
from src.performance.sharpe import sharpe_ratio
from src.rules.entry import MACDGoldenCross
from src.rules.exit import ATRTakeProfitStopLoss
from src.rules.reentry import CooldownReentry
from src.rules.sizing import InverseVolatilitySizer

SYMBOLS = ["NVDA", "KO", "XOM", "JPM"]


def make_strategy() -> Strategy:
    return Strategy(
        entry_rule=MACDGoldenCross(),
        exit_rule=ATRTakeProfitStopLoss(),
        reentry_rule=CooldownReentry(),
    )


def main():
    dfs = {
        s: pd.read_csv(f"data/00-美股ETF历史/raw/{s}_5m.csv", parse_dates=["timestamp"]) for s in SYMBOLS
    }

    portfolio_result = run_portfolio_backtest(dfs, make_strategy(), InverseVolatilitySizer())
    portfolio_sharpe = sharpe_ratio(portfolio_result.equity_curve)

    print(f"portfolio Sharpe: {portfolio_sharpe:.2f}")
    print()

    # aligned to the same bars the portfolio run used, so entry/exit signals
    # match and the only difference left is position sizing, not the sample
    aligned = align_to_common_index(dfs)

    print("per-symbol Sharpe (same rules, same aligned bars, run standalone, full notional):")
    for s in SYMBOLS:
        single_result = run_backtest(aligned[s], make_strategy())
        single_sharpe = sharpe_ratio(single_result.equity_curve)
        print(f"  {s}: {single_sharpe:.2f}")


if __name__ == "__main__":
    main()
