# -*- coding: utf-8 -*-
"""
用用户给的原始判据直接筛: "有时候一天都没有一手" —— 整日 volume == 0 的天数。
这比 padding 率或成交量比值都直接, 且没有阈值争议。
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

d = pd.read_pickle(os.path.join(OUT, 'qc_daily_agg.pkl'))
d = d[d.sym.isin(POOL)].sort_values(['sym', 'date']).reset_index(drop=True)
d['year'] = pd.DatetimeIndex(d.date).year

z = d.groupby('sym', observed=True).agg(
    sessions=('volume', 'size'),
    zero_days=('volume', lambda x: int((x == 0).sum())),
    tiny_days=('volume', lambda x: int((x < 100).sum())),
)
z['zero_pct'] = z.zero_days / z.sessions
z['tiny_pct'] = z.tiny_days / z.sessions
print('=== 池内 59 品种: 整日 volume==0 / <100 手 的天数 ===')
print(z[z.zero_days + z.tiny_days > 0].sort_values('tiny_days', ascending=False)
       .to_string(float_format=lambda x: f'{x:.5f}'))
print(f'\n完全没有"零成交日"也没有"<100手日"的品种: '
      f'{int(((z.zero_days == 0) & (z.tiny_days == 0)).sum())}/{len(z)}')

print('\n=== 有问题的品种, 逐年 ===')
bad = z.index[z.tiny_days > 0].tolist()
for s in bad:
    g = d[d.sym == s]
    y = g.groupby('year').agg(sessions=('volume', 'size'),
                              zero=('volume', lambda x: int((x == 0).sum())),
                              tiny=('volume', lambda x: int((x < 100).sum())),
                              vol_med=('volume', 'median'),
                              oi_med=('oi_end', 'median'))
    print(f'\n--- {s} ---')
    print(y.to_string(float_format=lambda x: f'{x:.0f}'))

print('\n=== 结论: 每个问题品种"最后一个 零成交日/极低成交日" ===')
rows = []
for s in bad:
    g = d[d.sym == s].reset_index(drop=True)
    m = (g.volume < 100)
    last_bad_i = int(np.flatnonzero(m.to_numpy()).max())
    after = g.date.iloc[last_bad_i + 1] if last_bad_i + 1 < len(g) else pd.NaT
    rows.append({'sym': s, 'first': g.date.iloc[0],
                 'last_tiny_day': g.date.iloc[last_bad_i],
                 'usable_from': after,
                 'sessions_all': len(g), 'sessions_lost': last_bad_i + 1,
                 'lost_pct': (last_bad_i + 1) / len(g)})
r = pd.DataFrame(rows).set_index('sym').sort_values('sessions_lost', ascending=False)
print(r.to_string(float_format=lambda x: f'{x:.4f}'))
r.to_csv(os.path.join(OUT, 'qc_F4_usable_from.csv'))
print('\n-> test_data/qc_F4_usable_from.csv (仅含真正需要裁剪的品种)')
