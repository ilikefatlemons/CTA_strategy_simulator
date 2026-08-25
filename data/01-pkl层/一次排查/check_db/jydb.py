"""
Connector for the JYDB (聚源) SQL Server instance.

Run it directly to smoke-test the connection:
    python data/03-pkl层/check_db/jydb.py

Or import it from a notebook / script:
    import sys; sys.path.append('data/03-pkl层/check_db')
    from jydb import read_sql
    df = read_sql("SELECT TOP 10 ContractCode, ContractName FROM Fut_ContractMain")

Environment overrides: FUT_JYDB_HOST / FUT_JYDB_USER / FUT_JYDB_PASSWORD / FUT_JYDB_DB.
Requires: pip install pymssql sqlalchemy
"""

import os
from urllib.parse import quote_plus

import pandas as pd
import pymssql
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# ── JYDB 连接参数（FUT_JYDB_* 环境变量可覆盖）────────────────────────────────
# HOST 支持两种写法：
#   '192.168.0.144'      内网直连（本机有路由时）
#   '127.0.0.1:11433'    SSH 反向隧道（跳板机跑：ssh -N -R 11433:192.168.0.144:1433 <本机>）
HOST = os.environ.get('FUT_JYDB_HOST', '192.168.0.144')
USERNAME = os.environ.get('FUT_JYDB_USER', 'readonly')
PASSWORD = os.environ.get('FUT_JYDB_PASSWORD', 'readonly')
DATABASE = os.environ.get('FUT_JYDB_DB', 'jydb')

# The server collation is Chinese_PRC and the text columns are varchar, not
# nvarchar, which puts the two directions of the connection in conflict:
#
#   charset='CP936'  Chinese reads back correctly, but Chinese in a WHERE clause
#                    matches NOTHING - and returns 0 rows rather than erroring,
#                    so a filter that silently finds nothing looks like real absence.
#   charset='UTF-8'  Chinese predicates match correctly, but returned Chinese is
#                    mojibake ('沪螺钢2708合约' -> '»¦ÂÝ¸Ö2708ºÏÔ¼').
#
# Correct queries matter more than pretty output, so we take UTF-8 and undo the
# mojibake on the way out (`_demojibake`). That gets both directions right.
CHARSET = 'UTF-8'

LOGIN_TIMEOUT = 10  # seconds to establish the TCP/login handshake
QUERY_TIMEOUT = 300  # seconds per query; JYDB's big tables are slow to scan

_engine: Engine | None = None


def _split_host(host: str = HOST) -> tuple[str, int]:
    """'192.168.0.144' -> ('192.168.0.144', 1433); '127.0.0.1:11433' -> (..., 11433)."""
    server, _, port = host.partition(':')
    return server, int(port or 1433)


def connect(database: str = DATABASE) -> pymssql.Connection:
    """Raw DB-API connection. Prefer `read_sql` unless you need a cursor."""
    server, port = _split_host()
    return pymssql.connect(
        server=server,
        port=str(port),  # pymssql's stub types `port` as str even though it accepts int
        user=USERNAME,
        password=PASSWORD,
        database=database,
        charset=CHARSET,
        login_timeout=LOGIN_TIMEOUT,
        timeout=QUERY_TIMEOUT,
    )


def get_engine(database: str = DATABASE) -> Engine:
    """Cached SQLAlchemy engine - this is what pandas wants to be handed."""
    global _engine
    if _engine is None or _engine.url.database != database:
        server, port = _split_host()
        _engine = create_engine(
            f"mssql+pymssql://{USERNAME}:{quote_plus(PASSWORD)}@{server}:{port}/{database}"
            f"?charset={CHARSET}",
            connect_args={"login_timeout": LOGIN_TIMEOUT, "timeout": QUERY_TIMEOUT},
            pool_pre_ping=True,  # the LAN link drops idle connections; reconnect instead of erroring
        )
    return _engine


def _demojibake(s: str) -> str:
    """Re-read a GBK byte string that FreeTDS handed us decoded as latin-1."""
    try:
        return s.encode('latin-1').decode('gb18030')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s  # pure ASCII, or already correct - leave it alone


def read_sql(sql: str, params=None, database: str = DATABASE, fix_text: bool = True) -> pd.DataFrame:
    """Run a query and return a DataFrame. Use %(name)s placeholders for params.

    Chinese in `params` is passed through as-is and matches correctly; Chinese in
    the result is repaired unless you pass fix_text=False.
    """
    df = pd.read_sql(sql, get_engine(database), params=params)
    if fix_text:
        # select_dtypes, not `dtypes == object`: pandas 3.0 gives text columns the
        # dedicated 'str' dtype, so an object-only check silently matches nothing.
        for col in df.select_dtypes(include=['object', 'string']).columns:
            df[col] = df[col].map(lambda v: _demojibake(v) if isinstance(v, str) else v)
    return df


def list_databases() -> list[str]:
    """Databases this login can actually open."""
    df = read_sql("SELECT name FROM sys.databases WHERE HAS_DBACCESS(name)=1 ORDER BY name",
                  database='master')
    return df['name'].tolist()


def list_tables(pattern: str = '%', database: str = DATABASE) -> pd.DataFrame:
    """Tables matching a LIKE pattern, with row counts, biggest first."""
    return read_sql(
        """
        SELECT t.name AS table_name, p.rows AS row_count
        FROM sys.tables t
        JOIN sys.partitions p ON p.object_id = t.object_id AND p.index_id IN (0, 1)
        WHERE t.name LIKE %(pattern)s
        ORDER BY p.rows DESC
        """,
        params={"pattern": pattern},
        database=database,
    )


def trading_sessions(contract_code: str, start: str = '2015-01-01',
                     end: str = '2030-01-01') -> pd.DataFrame:
    """Per-trading-day session boundaries for one contract, from Fut_TradingHours.

    One row per trading day. The night session ('continuous trade') sits on the
    PREVIOUS calendar evening but belongs to this trading day, so night_date is
    returned alongside night_beg/night_end - do not assume it is date - 1 day,
    because Monday's night session runs on the preceding Friday.

    night_beg is NULL on the first trading day after a public holiday: the evening
    before it was a holiday, so that trading day simply has no night session.
    Roughly 11 such days a year - the case a hardcoded session rule gets wrong.
    """
    return read_sql(
        """
        SELECT c.ContractCode                AS contract,
               h.CalendarDate                AS date,
               h.Week                        AS weekday,
               h.CallAuctionBeginTime        AS auction_beg,
               h.CallAuctionEndTime          AS auction_end,
               h.ContinuousTradeBDate        AS night_date,
               h.ContinuousTradeBTime        AS night_beg,
               h.ContinuousTradeETime        AS night_end,
               -- 1 = night session crosses midnight (ends 01:00/02:30), 2 = ends 23:00
               h.IfOvernightTrading          AS crosses_midnight,
               h.DayTradeFirstSessionBT      AS s1_beg,
               h.DayTradeFirstSessionET      AS s1_end,
               h.DayTradeSeconSessionBT      AS s2_beg,
               h.DayTradeSeconSessionET      AS s2_end,
               h.DayTradeThirdSessionBT      AS s3_beg,
               h.DayTradeThirdSessionET      AS s3_end
        FROM Fut_TradingHours h
        JOIN Fut_ContractMain c ON c.ContractInnerCode = h.ContractInnerCode
        WHERE c.ContractCode = %(code)s
          AND h.IfTradingDay = 1
          AND h.CalendarDate BETWEEN %(start)s AND %(end)s
        ORDER BY h.CalendarDate
        """,
        params={"code": contract_code, "start": start, "end": end},
    )


def describe(table: str, database: str = DATABASE) -> pd.DataFrame:
    """Column names and types for one table."""
    return read_sql(
        """
        SELECT name AS column_name, TYPE_NAME(user_type_id) AS data_type, max_length, is_nullable
        FROM sys.columns WHERE object_id = OBJECT_ID(%(table)s) ORDER BY column_id
        """,
        params={"table": table},
        database=database,
    )


if __name__ == "__main__":
    print(f"Connecting to {HOST} as {USERNAME} ...")
    print("Databases:", ", ".join(list_databases()))
    print()
    print("Largest Fut* tables in jydb:")
    print(list_tables('Fut%').head(8).to_string(index=False))
    print()
    print("Sample - SHFE rebar contracts, filtered by Chinese name:")
    print(read_sql("""
        SELECT TOP 5 ContractCode, ContractName, LastTradingDate
        FROM Fut_ContractMain WHERE ContractName LIKE %(p)s AND ExchangeCode = 10
        ORDER BY LastTradingDate DESC
    """, params={"p": "%沪螺钢%"}).to_string(index=False))
