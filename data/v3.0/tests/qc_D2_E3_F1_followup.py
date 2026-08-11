# -*- coding: utf-8 -*-
"""
Follow-up:
  (1) 为什么 contract 游程有 848,144 段而合约只有 4,041 个? -> 是否盘中来回跳 (D6/D7)
  (2) D2 用"干净换月点"(跨交易日 + 该日首根 bar) 复核 close/K 结论
  (3) K 的 NaN / <=0 分布
  (4) E3 修正候选乘数表 (补 16, 20000) + 按年乘数逐 bar 验隐含均价
  (5) F1 的 2019-06-10 异常日到底是什么
"""
import os, time
import numpy as np
import pandas as pd

PKL = r"D:\2026_Summer\TradingApp\data\v3.0\all_symbol_min_full_main_close_k_1.pkl"
OUT = r"D:\2026_Summer\TradingApp\data\v3.0\test_data"
_t0 = time.time()
def log(*a): print(f'[{time.time()-_t0:7.1f}s]', *a, flush=True)
def hr(t):
    print('\n' + '=' * 78, flush=True); print(t, flush=True); print('=' * 78, flush=True)

pd.set_option('display.width', 220, 'display.max_columns', 60, 'display.max_rows', 250)

log('loading ...')
df = pd.read_pickle(PKL)
sym_cat = df['underlying_symbol'].astype('category')
con_cat = df['contract'].astype('category')
sym_codes = sym_cat.cat.codes.to_numpy()
con_codes = con_cat.cat.codes.to_numpy()
SYMS = np.asarray(sym_cat.cat.categories)
CONS = np.asarray(con_cat.cat.categories)
ts  = df.index.to_numpy()
cl  = df['close'].to_numpy(); hi = df['high'].to_numpy(); lo = df['low'].to_numpy()
vol = df['volume'].to_numpy(); tov = df['total_turnover'].to_numpy(); Kv = df['K'].to_numpy()
d_codes, d_uniq = pd.factorize(df['trading_date'], sort=True)
d_uniq = pd.DatetimeIndex(d_uniq); yr = d_uniq.year.to_numpy()[d_codes]
log(f'{len(df):,} rows')

# ============================================================ (1) 游程诊断
hr('(1) contract 游程为何这么碎')
print(f'contract 列 NaN 数 (code==-1): {int((con_codes < 0).sum()):,}')

new_run = np.empty(len(con_codes), bool); new_run[0] = True
new_run[1:] = (con_codes[1:] != con_codes[:-1]) | (sym_codes[1:] != sym_codes[:-1])
run_start = np.flatnonzero(new_run)
run_len = np.diff(np.concatenate([run_start, [len(con_codes)]]))
print(f'游程数 {len(run_start):,} | 游程长度: median={np.median(run_len):.0f} '
      f'p90={np.quantile(run_len,.9):.0f} max={run_len.max():,}')

# 每个 symbol: 游程数 vs 唯一合约数; 换月是否跨交易日
chg = np.flatnonzero(new_run[1:]) + 1                       # 变更点(不含每个symbol首行)
chg = chg[sym_codes[chg] == sym_codes[chg - 1]]             # 只保留同 symbol 内的变更
cross_day = d_codes[chg] != d_codes[chg - 1]
print(f'\n同品种内 contract 变更点: {len(chg):,}   其中跨交易日: {cross_day.sum():,} '
      f'({cross_day.mean():.2%})  盘中变更: {(~cross_day).sum():,} ({(~cross_day).mean():.2%})')

rs = pd.DataFrame({'sym': SYMS[sym_codes[chg]], 'cross_day': cross_day})
tbl = rs.groupby('sym').agg(n_change=('cross_day', 'size'), n_cross=('cross_day', 'sum'))
tbl['n_intraday'] = tbl.n_change - tbl.n_cross
uc = pd.Series(con_codes).groupby(sym_codes).nunique(); uc.index = SYMS[uc.index]
tbl['n_contract'] = uc
tbl = tbl.sort_values('n_intraday', ascending=False)
print('\n每品种 contract 变更次数 (n_intraday 大 = 盘中来回跳):')
print(tbl.head(25).to_string())
tbl.to_csv(os.path.join(OUT, 'qc_D6_contract_switch_by_symbol.csv'))

# 是否同一合约反复出现 (振荡)
osc = len(run_start) - int(uc.sum())
print(f'\n游程数 - 唯一合约数(按品种汇总) = {osc:,}  (>0 说明同一合约被反复切回)')

for s in ['A', 'RB']:
    si = int(np.flatnonzero(SYMS == s)[0])
    m = np.flatnonzero(sym_codes == si)
    c = con_codes[m]
    k = np.flatnonzero(c[1:] != c[:-1]) + 1
    print(f'\n--- {s}: {len(k)} 次 contract 变更, 唯一合约 {len(np.unique(c))} 个; 前 12 次: ---')
    for j in k[:12]:
        print(f'   {pd.Timestamp(ts[m[j-1]])} {CONS[c[j-1]]:>8}  ->  '
              f'{pd.Timestamp(ts[m[j]])} {CONS[c[j]]:>8}   '
              f'同日={d_codes[m[j]]==d_codes[m[j-1]]}')

# ============================================================ (2) D2 复核
hr('(2) D2 复核 -- 只用干净换月点')
prev, nxt = chg - 1, chg
clean = cross_day & (cl[prev] > 0) & (cl[nxt] > 0) & np.isfinite(Kv[prev]) & \
        np.isfinite(Kv[nxt]) & (Kv[prev] > 0) & (Kv[nxt] > 0)
p, n = prev[clean], nxt[clean]
r_raw = np.log(cl[n] / cl[p]); rk = np.log(Kv[p] / Kv[n])
res = pd.DataFrame({
    'raw close': np.abs(r_raw), 'close * K': np.abs(r_raw - rk), 'close / K': np.abs(r_raw + rk),
}).describe(percentiles=[.5, .9, .99]).T[['50%', 'mean', '90%', '99%', 'max']]
print(f'干净换月点: {len(p):,}')
print(res.to_string(float_format=lambda x: f'{x:.6f}'))
if rk.std() > 0:
    print(f'\n回归 r_raw ~ rk  slope={np.polyfit(rk, r_raw, 1)[0]:+.4f}  '
          f'corr={np.corrcoef(rk, r_raw)[0,1]:+.4f}')
w = pd.DataFrame({'sym': SYMS[sym_codes[n]], 'raw': np.abs(r_raw),
                  'mul': np.abs(r_raw - rk), 'div': np.abs(r_raw + rk)}) \
      .groupby('sym')[['raw', 'mul', 'div']].median()
print('\n逐品种 winner 计票:', w.idxmin(axis=1).value_counts().to_dict())
print('非 div 的品种:', w.index[w.idxmin(axis=1) != 'div'].tolist())

# ============================================================ (3) K 可用性
hr('(3) K 的 NaN / 非正 分布')
kb = pd.DataFrame({'sym': SYMS[sym_codes], 'nan': np.isnan(Kv), 'nonpos': (Kv <= 0)}) \
       .groupby('sym').agg(n=('nan', 'size'), n_nan=('nan', 'sum'), n_nonpos=('nonpos', 'sum'))
kb['bad_pct'] = (kb.n_nan + kb.n_nonpos) / kb.n
bad = kb[kb.bad_pct > 0].sort_values('bad_pct', ascending=False)
print(f'K 有问题的品种 {len(bad)} / {len(kb)}:')
print(bad.to_string(float_format=lambda x: f'{x:.4f}'))
kb.to_csv(os.path.join(OUT, 'qc_D4_K_availability.csv'))
if (Kv <= 0).any():
    b = np.flatnonzero(Kv <= 0)[:10]
    print('\nK<=0 样例:')
    print(pd.DataFrame({'datetime': ts[b], 'sym': SYMS[sym_codes[b]],
                        'contract': CONS[con_codes[b]], 'K': Kv[b], 'close': cl[b]}).to_string())

# ============================================================ (4) E3 修正
hr('(4) E3 修正 -- 补齐候选乘数 + 按年验证')
CAND = np.array([1, 2, 5, 10, 15, 16, 20, 25, 30, 50, 60, 100, 200, 300, 400, 500,
                 1000, 2000, 5000, 10000, 20000])
valid = (vol > 0) & (cl > 0) & np.isfinite(tov) & (tov > 0)
mest = tov[valid] / vol[valid] / cl[valid]
gs, gy = sym_codes[valid], yr[valid]
med_yr = pd.Series(mest).groupby([gs, gy]).median()
snapped = med_yr.map(lambda x: CAND[np.argmin(np.abs(np.log(CAND / x)))] if np.isfinite(x) else np.nan)
relerr = (med_yr / snapped - 1).abs()
print(f'按(品种,年)反推: {len(med_yr)} 组 | 相对误差 >1% 的组: {(relerr > .01).sum()}')
if (relerr > .01).any():
    z = pd.DataFrame({'med': med_yr[relerr > .01], 'snap': snapped[relerr > .01],
                      'relerr': relerr[relerr > .01]})
    z.index = pd.MultiIndex.from_tuples([(SYMS[a], b) for a, b in z.index], names=['sym', 'year'])
    print(z.to_string(float_format=lambda x: f'{x:.6g}'))

lut = {}
for (a, b), v in snapped.items():
    lut[(a, b)] = v
mult_bar = np.array([lut.get((a, b), np.nan) for a, b in zip(gs, gy)])
implied = tov[valid] / vol[valid] / mult_bar
tol = 1e-6 + 5e-4 * np.abs(implied)
bad_px = (implied < lo[valid] - tol) | (implied > hi[valid] + tol)
vb = pd.DataFrame({'sym': SYMS[gs], 'bad': bad_px}).groupby('sym').agg(
    n=('bad', 'size'), n_bad=('bad', 'sum'))
vb['pct'] = vb.n_bad / vb.n
print(f'\n用"按年乘数"后, 隐含均价越界总比例: {bad_px.mean():.6%}  '
      f'(此前用单一乘数: 见 qc_E3_multiplier.csv)')
print('\n越界比例最高的 15 个品种:')
print(vb.sort_values('pct', ascending=False).head(15).to_string(float_format=lambda x: f'{x:.6g}'))
vb.to_csv(os.path.join(OUT, 'qc_E3_multiplier_fixed.csv'))
sm = snapped.copy()
sm.index = pd.MultiIndex.from_tuples([(SYMS[a], b) for a, b in sm.index], names=['sym', 'year'])
sm.unstack().to_csv(os.path.join(OUT, 'qc_E4_multiplier_by_year_fixed.csv'))

# ============================================================ (5) F1 异常日
hr('(5) F1 -- 2019-06-10 等异常日到底是什么')
daily = pd.read_pickle(os.path.join(OUT, 'qc_daily_agg.pkl'))
for d in ['2019-06-10', '2011-12-02', '2026-03-10']:
    d = pd.Timestamp(d)
    prv = daily.date[daily.date < d].max()
    a = daily[daily.date == prv].set_index('sym')
    b = daily[daily.date == d].set_index('sym')
    j = a.join(b, lsuffix='_prev', rsuffix='_now', how='inner')
    j['vol_ratio'] = j.volume_now / j.volume_prev
    j['bar_ratio'] = j.bars_now / j.bars_prev
    j['oi_ratio'] = j.oi_end_now / j.oi_end_prev
    print(f'\n--- {d.date()} (上一交易日 {prv.date()}) n={len(j)} ---')
    print(f'  volume ratio  median={j.vol_ratio.median():.3f}  '
          f'bars ratio median={j.bar_ratio.median():.3f}  '
          f'OI ratio median={j.oi_ratio.median():.3f}')
    print(f'  bars: 前 {a.bars.median():.0f} -> 今 {b.bars.median():.0f} (中位)')
    print(j[['bars_prev', 'bars_now', 'vol_ratio', 'bar_ratio', 'oi_ratio']]
          .sort_values('vol_ratio').head(8).to_string(float_format=lambda x: f'{x:.3f}'))

hr('DONE')
log('done')
