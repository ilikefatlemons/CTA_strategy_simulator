"""
Phase 1 (data source swap): fetch intraday bars from Alpaca's Market Data API
instead of yfinance, to get much deeper minute-bar history (years, not weeks).

Lands the same fixed schema the rest of the pipeline depends on:
timestamp, symbol, sector, open, high, low, close, volume.
"""

import os

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

load_dotenv()

SYMBOLS = {
    "NVDA": "Technology",
    "KO": "Consumer Staples",
    "XOM": "Energy",
    "JPM": "Financials",
}

TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)
LOOKBACK_DAYS = 730  # ~2 years; free/paper accounts get IEX-feed history this far back

OUT_DIR = "data/raw"


def get_client() -> StockHistoricalDataClient:
    api_key = os.environ["ALPACA_API_KEY"]
    api_secret = os.environ["ALPACA_API_SECRET"]
    return StockHistoricalDataClient(api_key, api_secret)


def fetch_symbol(client: StockHistoricalDataClient, symbol: str, sector: str) -> pd.DataFrame:
    start = pd.Timestamp.now("UTC") - pd.Timedelta(days=LOOKBACK_DAYS)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TIMEFRAME,
        start=start.to_pydatetime(),
        feed="iex",  # free-tier data feed
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    if df.empty:
        raise ValueError(f"Alpaca returned no data for {symbol}")

    df = df.reset_index()
    df = df.rename(
        columns={
            "timestamp": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    df["symbol"] = symbol
    df["sector"] = sector

    return df[["timestamp", "symbol", "sector", "open", "high", "low", "close", "volume"]]


def main():
    client = get_client()
    for symbol, sector in SYMBOLS.items():
        print(f"Fetching {symbol} ({sector}) [{TIMEFRAME}, last {LOOKBACK_DAYS}d]...")
        df = fetch_symbol(client, symbol, sector)
        out_path = f"{OUT_DIR}/{symbol}_5m.csv"
        df.to_csv(out_path, index=False)
        print(f"  -> {out_path}  ({len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()})")


if __name__ == "__main__":
    main()
