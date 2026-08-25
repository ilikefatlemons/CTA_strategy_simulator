"""
Phase 3 entry point: run the same MACD entry / ATR TP-SL exit / cooldown
re-entry rules across all four symbols, sized by InverseVolatilitySizer with
daily rebalance (including resizing already-open positions).
"""

import pandas as pd

from src.engine.backtest import Strategy
from src.engine.portfolio_backtest import run_portfolio_backtest
from src.rules.entry import MACDGoldenCross
from src.rules.exit import ATRTakeProfitStopLoss
from src.rules.reentry import CooldownReentry
from src.rules.sizing import InverseVolatilitySizer

SYMBOLS = ["NVDA", "KO", "XOM", "JPM"]
TRADE_LOG_PATH = "data/phase3_trades.csv"
WEIGHTS_LOG_PATH = "data/phase3_weights.csv"


def main():
    dfs = {
        s: pd.read_csv(f"data/00-美股ETF历史/raw/{s}_5m.csv", parse_dates=["timestamp"]) for s in SYMBOLS
    }

    strategy = Strategy(
        entry_rule=MACDGoldenCross(),
        exit_rule=ATRTakeProfitStopLoss(),
        reentry_rule=CooldownReentry(),
    )
    sizer = InverseVolatilitySizer()

    result = run_portfolio_backtest(dfs, strategy, sizer)

    print(
        f"equity start: {result.equity_curve.iloc[0]:.2f}  "
        f"end: {result.equity_curve.iloc[-1]:.2f}  "
        f"return: {result.equity_curve.iloc[-1] / result.equity_curve.iloc[0] - 1:.2%}"
    )
    print()

    for s in SYMBOLS:
        sym_trades = [t for t in result.trades if t.symbol == s]
        n = len(sym_trades)
        wins = [t for t in sym_trades if t.pnl > 0]
        win_rate = len(wins) / n if n else 0.0
        mean_weight = result.weights_history[s].mean() if s in result.weights_history else 0.0
        print(f"{s}: trades={n}  win_rate={win_rate:.2%}  mean_weight={mean_weight:.2%}")

    trade_rows = [
        {
            "symbol": t.symbol,
            "side": t.side.name,
            "entry_bar_idx": t.entry_bar_idx,
            "entry_price": t.entry_price,
            "exit_bar_idx": t.exit_bar_idx,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
        }
        for t in result.trades
    ]
    pd.DataFrame(trade_rows).to_csv(TRADE_LOG_PATH, index=False)
    result.weights_history.to_csv(WEIGHTS_LOG_PATH)
    print(f"\ntrade log written to {TRADE_LOG_PATH}")
    print(f"weights history written to {WEIGHTS_LOG_PATH}")


if __name__ == "__main__":
    main()
