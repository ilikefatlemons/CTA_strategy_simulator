# -*- coding: utf-8 -*-
"""
收尾三个未定性的数字:
  (1) 33,300 根"日盘 bar 的 trading_date != 日历日"是什么
  (2) 2,367 根"非周六的凌晨盘 trading_date != 日历日"是什么
  (3) B8: 长假后首日仍带夜盘的 375 条, 其夜盘 bar 究竟落在哪个日历日
  (4) padding 改用近 250 个交易日重算 (全历史比例被早年低流动性污染)
"""
import os, time
import numpy as np
import pandas as pd

PKL = r"D:\2026_Summer\TradingApp\data\03-pkl层\all_symbol_min_full_main_close_k_1.pkl"
OUT = r"D:\2026_Summer\TradingApp\data\03-pkl层\排查证据"
EXCLUDED = ['ZC', 'JR', 'WH', 'RS', 'PM', 'RI', 'LR']
_t0 = time.time()
def hr(t):
    print('\n' + '=' * 92, flush=True); print(t, flush=True); print('=' * 92, flush=True)
pd.set_option('display.width', 240, 'display.max_columns', 60, 'display.max_rows', 300)

df = pd.read_pickle(PKL)
print(f'[{time.time()-_t0:.1f}s] loaded', flush=True)
sym_cat = df['underlying_symbol'].astype('category')
sc = sym_cat.cat.codes.to_numpy(); SYMS = np.asarray(sym_cat.cat.categories)
idx = df.index
mod = (idx.hour * 60 + idx.minute).to_numpy().astype(np.int32)
cal = idx.normalize().to_numpy()
wd = idx.dayofweek.to_numpy()
td = df['trading_date'].to_numpy()
vol = df['volume'].to_numpy(); tov = df['total_turnover'].to_numpy()
op = df['open'].to_numpy(); hi = df['high'].to_numpy()
lo = df['low'].to_numpy(); cl = df['close'].to_numpy()
POOL = [s for s in SYMS if s not in EXCLUDED]

is_eve = mod >= 21 * 60
is_am = mod <= 179
is_day = (~is_eve) & (~is_am)

# ---------------------------------------------------------------- (1)
hr('(1) 日盘 bar 的 trading_date != 日历日')
m = is_day & (td != cal)
print(f'命中 {int(m.sum()):,} 根')
sub = pd.DataFrame({'sym': SYMS[sc[m]], 'ts': idx[m], 'td': td[m],
                    'mod': mod[m], 'vol': vol[m]})
sub['cal'] = sub.ts.dt.normalize()
sub['delta_days'] = (sub.td - sub.cal).dt.days
print('\n按品种:'); print(sub.groupby('sym').size().sort_values(ascending=False).to_string())
print('\n按 (日历日 - trading_date) 差值:')
print(sub.delta_days.value_counts().sort_index().to_string())
print('\n按日历日 (前 20):')
print(sub.groupby('cal').size().sort_values(ascending=False).head(20).to_string())
print('\n时间分布 (小时):')
print(sub.groupby(sub.ts.dt.hour).size().to_string())
print('\n样例:'); print(sub.head(15).to_string(index=False))

# ---------------------------------------------------------------- (2)
hr('(2) 非周六的凌晨盘 trading_date != 日历日')
m2 = is_am & (td != cal) & (wd != 5)
print(f'命中 {int(m2.sum()):,} 根')
if m2.any():
    s2 = pd.DataFrame({'sym': SYMS[sc[m2]], 'ts': idx[m2], 'td': td[m2]})
    s2['cal'] = s2.ts.dt.normalize(); s2['wd'] = s2.ts.dt.dayofweek
    print('\n按品种:'); print(s2.groupby('sym').size().sort_values(ascending=False).to_string())
    print('\n按日历日:'); print(s2.groupby('cal').size().sort_values(ascending=False).head(20).to_string())
    print('\n日历日星期分布 (0=周一):'); print(s2.wd.value_counts().sort_index().to_string())
    print('\n样例:'); print(s2.head(12).to_string(index=False))

# ---------------------------------------------------------------- (3)
hr('(3) B8 -- 长假后首日的夜盘 bar 落在哪个日历日')
night = is_eve | is_am
for sym, tdate in [('A', '2024-10-08'), ('A', '2025-02-05'), ('AU', '2024-10-08'),
                   ('RB', '2026-02-24')]:
    code = int(np.flatnonzero(SYMS == sym)[0])
    mm = (sc == code) & (td == np.datetime64(tdate)) & night
    if not mm.any():
        print(f'\n{sym} {tdate}: 无夜盘 bar'); continue
    cals = pd.Series(cal[mm]).value_counts().sort_index()
    print(f'\n{sym} trading_date={tdate}: 夜盘 bar {int(mm.sum())} 根, 分布在日历日')
    print('   ' + ' | '.join(f'{pd.Timestamp(k).date()}({v})' for k, v in cals.items()))
    mmd = (sc == code) & (td == np.datetime64(tdate)) & is_day
    print(f'   同 trading_date 的日盘 bar {int(mmd.sum())} 根, '
          f'日历日 {sorted({str(pd.Timestamp(x).date()) for x in cal[mmd]})}')
    print(f'   夜盘成交量合计 {vol[mm].sum():,.0f} (日盘 {vol[mmd].sum():,.0f})')

# ---------------------------------------------------------------- (4)
hr('(4) padding 比例 -- 近 250 个交易日 vs 全历史')
pad = (vol == 0) & (tov == 0) & (op == hi) & (hi == lo) & (lo == cl)
alld = np.unique(td)
cut = alld[-250]
rec = td >= cut
print(f'窗口: {pd.Timestamp(cut).date()} ~ {pd.Timestamp(alld[-1]).date()}')
full = pd.Series(pad).groupby(sc).mean(); full.index = SYMS[full.index]
r = pd.Series(pad[rec]).groupby(sc[rec]).mean(); r.index = SYMS[r.index]
n_rec = pd.Series(np.ones(int(rec.sum()))).groupby(sc[rec]).size(); n_rec.index = SYMS[n_rec.index]
t = pd.DataFrame({'pad_all': full, 'pad_250d': r, 'n_bar_250d': n_rec})
t['in_pool'] = [s in POOL for s in t.index]
tp = t[t.in_pool].sort_values('pad_250d', ascending=False)
print(f'\n池内 {len(tp)} 品种, 按近 250 日 padding 比例排序:')
print(tp.to_string(float_format=lambda x: f'{x:.5f}'))
alive = tp[tp.n_bar_250d.notna()]
print(f'\n近 250 日仍有数据的池内品种: {len(alive)}')
for th in [0.05, 0.10, 0.20, 0.30]:
    hit = alive.index[alive.pad_250d > th].tolist()
    print(f'  padding > {th:.0%}: {len(hit)} 个  {hit}')
t.to_csv(os.path.join(OUT, 'qc_B1_padding_250d.csv'))
hr('DONE')
print(f'[{time.time()-_t0:.1f}s] done')
