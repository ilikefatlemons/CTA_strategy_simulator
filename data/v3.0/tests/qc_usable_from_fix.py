# -*- coding: utf-8 -*-
"""
重做 usable_from。

上一版算法是错的: 它取"滚动 padding 中位数最后一次 >=5% 的日子"之后作为起点,
于是历史中任何一段 60 日的坏窗口(例如 2015 年夜盘刚引入时夜盘成交稀疏)都会
把起点一路推到很后面 —— A 上市首年日均成交 9.9 万手却被判到 2015-06-01, 荒谬。

新算法只看**开头那一段**, 且判据改用用户真正关心的东西: 日成交量相对该品种
自身成熟期水平的比值。
  mature   = 最后 250 个 session 的日成交中位数
  fwd250   = 向后 250 个 session 的日成交中位数 (每个日期一个值)
  usable_from = 第一个满足 fwd250 >= 1% * mature 的日期
只用小的日频缓存, 不加载 6GB pkl。
"""
import os
import numpy as np
import pandas as pd

OUT = r"D:\2026_Summer\TradingApp\data\v3.0\test_data"
POOL = ['A','AG','AL','AP','AU','B','BC','BU','C','CF','CJ','CS','CU','CY','EB','EG',
        'FG','FU','HC','I','IC','IF','IH','J','JD','JM','L','LH','LU','M','MA','NI',
        'NR','OI','P','PB','PF','PG','PK','PP','RB','RM','RU','SA','SC','SF','SM',
        'SN','SP','SR','SS','T','TA','TF','TS','UR','V','Y','ZN']
pd.set_option('display.width', 240, 'display.max_rows', 400, 'display.max_columns', 40)

daily = pd.read_pickle(os.path.join(OUT, 'qc_daily_agg.pkl'))
daily = daily[daily.sym.isin(POOL)].sort_values(['sym', 'date'])
print(f'池内 {daily.sym.nunique()} 品种, {len(daily):,} 个 (品种,交易日)')

rows = []
for s, g in daily.groupby('sym', observed=True):
    g = g.reset_index(drop=True)
    v = g.volume.to_numpy(float)
    n = len(v)
    mature = np.median(v[-250:])
    # 向后 250 个 session 的成交中位数
    fwd = pd.Series(v[::-1]).rolling(250, min_periods=60).median()[::-1].to_numpy()
    ratio = fwd / mature if mature > 0 else np.full(n, np.nan)
    hit = np.flatnonzero(ratio >= 0.01)
    i0 = int(hit[0]) if len(hit) else n - 1
    y1 = v[:250]
    rows.append({
        'sym': s, 'first': g.date.iloc[0], 'last': g.date.iloc[-1],
        'usable_from': g.date.iloc[i0], 'sessions_all': n, 'sessions_lost': i0,
        'lost_pct': i0 / n,
        'vol_y1_med': np.median(y1),
        'oi_y1_med': np.median(g.oi_end.to_numpy()[:250]),
        'vol_mature': mature,
        'ratio_y1': np.median(y1) / mature if mature > 0 else np.nan,
    })
t = pd.DataFrame(rows).set_index('sym').sort_values('ratio_y1')
print('\n=== 全部 59 个池内品种, 按"首年日均成交 / 成熟期日均成交"升序 ===')
print(t.to_string(float_format=lambda x: f'{x:.6g}'))

print('\n=== 被判定存在早期死区的品种 (usable_from > first) ===')
flag = t[t.sessions_lost > 0].sort_values('sessions_lost', ascending=False)
print(flag.to_string(float_format=lambda x: f'{x:.6g}') if len(flag) else '  无')

print('\n=== 首年成交量占成熟期比值的分布 (看是否存在天然断层) ===')
r = t.ratio_y1.sort_values()
for q in [0, 1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 58]:
    if q < len(r):
        print(f'  第 {q+1:2d} 低: {r.index[q]:4s} {r.iloc[q]:.6f}')

for s in ['B', 'CY', 'AL', 'A', 'ZN', 'IF', 'PB', 'BC', 'TF', 'V', 'C']:
    g = daily[daily.sym == s].copy()
    if not len(g):
        continue
    g['year'] = pd.DatetimeIndex(g.date).year
    y = g.groupby('year').agg(sessions=('volume', 'size'), vol_med=('volume', 'median'),
                              oi_med=('oi_end', 'median'), zero_vol_days=('volume',
                                                                          lambda x: int((x == 0).sum())))
    print(f'\n--- {s} 逐年 ---')
    print(y.to_string(float_format=lambda x: f'{x:.0f}'))

t.to_csv(os.path.join(OUT, 'qc_F4_usable_from.csv'))
print('\n已覆盖 test_data/qc_F4_usable_from.csv')
