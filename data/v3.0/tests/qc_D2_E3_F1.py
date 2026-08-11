# -*- coding: utf-8 -*-
"""
排查清单前三项:
  D2  K 的方向与锚点      -- close*K 还是 close/K 使换月处连续?
  E3  成交额恒等式        -- turnover/volume/close 反推合约乘数, 验隐含均价 in [low,high]
  E4  乘数历史跳变
  F1  成交量/持仓量单双边口径台阶

输出: data/v3.0/test_data/qc_*.csv  +  stdout 报告
"""
import os, sys, time
import numpy as np
import pandas as pd

PKL = r"D:\2026_Summer\TradingApp\data\v3.0\all_symbol_min_full_main_close_k_1.pkl"
OUT = r"D:\2026_Summer\TradingApp\data\v3.0\test_data"
os.makedirs(OUT, exist_ok=True)

_t0 = time.time()
def log(*a):
    print(f'[{time.time()-_t0:7.1f}s]', *a, flush=True)

def hr(title):
    print('\n' + '=' * 78, flush=True)
    print(title, flush=True)
    print('=' * 78, flush=True)

pd.set_option('display.width', 200, 'display.max_columns', 50, 'display.max_rows', 200)

# ---------------------------------------------------------------- load
log('loading pickle ...')
df = pd.read_pickle(PKL)
log(f'loaded {len(df):,} rows')

sym_cat = df['underlying_symbol'].astype('category')
con_cat = df['contract'].astype('category')
sym_codes = sym_cat.cat.codes.to_numpy()
con_codes = con_cat.cat.codes.to_numpy()
SYMS = np.asarray(sym_cat.cat.categories)
CONS = np.asarray(con_cat.cat.categories)

ts   = df.index.to_numpy()
op   = df['open'].to_numpy()
hi   = df['high'].to_numpy()
lo   = df['low'].to_numpy()
cl   = df['close'].to_numpy()
vol  = df['volume'].to_numpy()
tov  = df['total_turnover'].to_numpy()
oi   = df['open_interest'].to_numpy()
Kv   = df['K'].to_numpy()

d_codes, d_uniq = pd.factorize(df['trading_date'], sort=True)
d_uniq = pd.DatetimeIndex(d_uniq)
yr_of_d = d_uniq.year.to_numpy()
yr = yr_of_d[d_codes]
log(f'{len(SYMS)} symbols, {len(CONS)} contracts, {len(d_uniq)} trading dates')

# 排序假设: 按 (symbol, datetime) 升序; 后面的游程切分依赖它
sorted_ok = bool((np.diff(sym_codes) >= 0).all())
key = sym_codes.astype(np.int64) * len(d_uniq) + d_codes
key_mono = bool((np.diff(key) >= 0).all())
log(f'sorted by symbol: {sorted_ok} | (symbol,date) key monotonic: {key_mono}')

# ================================================================ D2/D3/D4
hr('D2 / D3 / D4  --  复权因子 K')

# 合约游程 (同一 symbol 内 contract 连续不变的一段)
new_run = np.empty(len(con_codes), dtype=bool)
new_run[0] = True
new_run[1:] = (con_codes[1:] != con_codes[:-1]) | (sym_codes[1:] != sym_codes[:-1])
run_start = np.flatnonzero(new_run)
run_end = np.concatenate([run_start[1:] - 1, [len(con_codes) - 1]])
log(f'contract runs: {len(run_start):,}')

# --- D3: K 是否在合约游程内变化
k_chg = np.empty(len(Kv), dtype=bool)
k_chg[0] = False
k_chg[1:] = Kv[1:] != Kv[:-1]
k_bad_inside = k_chg & ~new_run
n_k_inside = int(k_bad_inside.sum())
print(f'D3  K 在合约游程内部发生变化的 bar 数: {n_k_inside:,}')
if n_k_inside:
    bs = np.flatnonzero(k_bad_inside)
    d3 = pd.DataFrame({'datetime': ts[bs], 'sym': SYMS[sym_codes[bs]],
                       'contract': CONS[con_codes[bs]],
                       'K_prev': Kv[bs - 1], 'K': Kv[bs]})
    print(d3.groupby('sym').size().sort_values(ascending=False).head(20))
    d3.to_csv(os.path.join(OUT, 'qc_D3_K_changes_inside_contract.csv'), index=False)

# --- D4: K 的可用性
print(f'D4  K NaN: {int(np.isnan(Kv).sum()):,} | K<=0: {int((Kv <= 0).sum()):,} | '
      f'K==1 恰好: {int((Kv == 1.0).sum()):,}  ({(Kv == 1.0).mean():.4%})')

# --- 换月点: 相邻游程且同一 symbol
i = np.arange(1, len(run_start))
prev_end = run_end[i - 1]
next_start = run_start[i]
same_sym = sym_codes[next_start] == sym_codes[prev_end]
prev_end = prev_end[same_sym]
next_start = next_start[same_sym]

c_old, c_new = cl[prev_end], cl[next_start]
k_old, k_new = Kv[prev_end], Kv[next_start]
good = (c_old > 0) & (c_new > 0) & np.isfinite(k_old) & np.isfinite(k_new) & (k_old > 0) & (k_new > 0)
prev_end, next_start = prev_end[good], next_start[good]
c_old, c_new, k_old, k_new = c_old[good], c_new[good], k_old[good], k_new[good]

r_raw = np.log(c_new / c_old)
rk = np.log(k_old / k_new)
jump_raw = r_raw
jump_mul = r_raw - rk          # adj = close * K
jump_div = r_raw + rk          # adj = close / K

# 基准: 同一游程内相邻 bar 的对数收益绝对值中位数 (取样避免全量排序)
step = max(1, len(cl) // 2_000_000)
samp = np.arange(1, len(cl), step)
samp = samp[~new_run[samp]]
base = np.abs(np.log(np.maximum(cl[samp], 1e-9) / np.maximum(cl[samp - 1], 1e-9)))
base = base[np.isfinite(base)]

print(f'\nD2  换月点数量: {len(r_raw):,}')
print(f'    K 是否在换月处必然变化: {float((k_old != k_new).mean()):.4%} 的换月点 K 发生了变化')
print('\n    |跳空| 分布 (对数, 越小越连续):')
rows = []
for name, j in [('raw  close', jump_raw), ('close * K', jump_mul), ('close / K', jump_div)]:
    a = np.abs(j)
    rows.append({'series': name, 'median': np.median(a), 'mean': a.mean(),
                 'p90': np.quantile(a, .90), 'p99': np.quantile(a, .99), 'max': a.max()})
rows.append({'series': '(基准)游程内相邻bar', 'median': np.median(base), 'mean': base.mean(),
             'p90': np.quantile(base, .90), 'p99': np.quantile(base, .99), 'max': base.max()})
print(pd.DataFrame(rows).set_index('series').to_string(float_format=lambda x: f'{x:.6f}'))

fin = np.isfinite(r_raw) & np.isfinite(rk)
if fin.sum() > 10 and rk[fin].std() > 0:
    slope = np.polyfit(rk[fin], r_raw[fin], 1)[0]
    corr = np.corrcoef(rk[fin], r_raw[fin])[0, 1]
    print(f'\n    回归 r_raw ~ rk   slope={slope:+.4f}  corr={corr:+.4f}'
          f'   (+1 => close*K ; -1 => close/K ; 0 => K 与换月无关)')

roll = pd.DataFrame({
    'sym': SYMS[sym_codes[next_start]],
    'at': ts[next_start],
    'from': CONS[con_codes[prev_end]], 'to': CONS[con_codes[next_start]],
    'close_old': c_old, 'close_new': c_new, 'K_old': k_old, 'K_new': k_new,
    'jump_raw': jump_raw, 'jump_mul': jump_mul, 'jump_div': jump_div,
})
roll.to_csv(os.path.join(OUT, 'qc_D2_rollovers.csv'), index=False)

per_sym = roll.assign(a_raw=roll.jump_raw.abs(), a_mul=roll.jump_mul.abs(),
                      a_div=roll.jump_div.abs()) \
    .groupby('sym')[['a_raw', 'a_mul', 'a_div']].median()
per_sym['winner'] = per_sym.idxmin(axis=1)
print('\n    逐品种 |跳空| 中位数 (winner = 最连续的口径):')
print(per_sym.to_string(float_format=lambda x: f'{x:.6f}'))
print('\n    winner 计票:', per_sym.winner.value_counts().to_dict())
per_sym.to_csv(os.path.join(OUT, 'qc_D2_per_symbol.csv'))

# --- 锚点: 每个 symbol 首/末游程的 K
anchor = []
for s in range(len(SYMS)):
    m = sym_codes[run_start] == s
    if not m.any():
        continue
    rs = run_start[m]
    anchor.append({'sym': SYMS[s], 'K_first': Kv[rs[0]], 'K_last': Kv[rs[-1]],
                   'K_min': np.nanmin(Kv[sym_codes == s]), 'K_max': np.nanmax(Kv[sym_codes == s]),
                   'n_runs': int(m.sum())})
anchor = pd.DataFrame(anchor).set_index('sym')
print('\n    锚点检查 (标准后复权应 K_last==1, 前复权应 K_first==1):')
print(f'    K_last==1 的品种数: {(np.abs(anchor.K_last - 1) < 1e-9).sum()} / {len(anchor)}')
print(f'    K_first==1 的品种数: {(np.abs(anchor.K_first - 1) < 1e-9).sum()} / {len(anchor)}')
print(anchor.head(15).to_string(float_format=lambda x: f'{x:.6f}'))
anchor.to_csv(os.path.join(OUT, 'qc_D2_anchor.csv'))

# ================================================================ E3 / E4
hr('E3 / E4  --  成交额恒等式与合约乘数')

valid = (vol > 0) & (cl > 0) & np.isfinite(tov) & (tov > 0)
print(f'可用于反推的 bar: {int(valid.sum()):,} / {len(vol):,} ({valid.mean():.2%})')
m_est = tov[valid] / vol[valid] / cl[valid]
g_sym = sym_codes[valid]
g_yr = yr[valid]

s = pd.Series(m_est)
mult_sym = s.groupby(g_sym).median()
mult_sym.index = SYMS[mult_sym.index]
mult_yr = s.groupby([g_sym, g_yr]).median().unstack()
mult_yr.index = SYMS[mult_yr.index]

CAND = np.array([1, 2, 5, 10, 15, 20, 25, 30, 50, 60, 100, 200, 300, 400, 500,
                 1000, 2000, 5000, 10000])
def snap(x):
    if not np.isfinite(x):
        return np.nan
    return CAND[np.argmin(np.abs(np.log(CAND / x)))]

e3 = pd.DataFrame({'mult_med': mult_sym})
e3['mult_snap'] = e3.mult_med.map(snap)
e3['rel_err'] = (e3.mult_med / e3.mult_snap - 1).abs()

# 用 snap 后的乘数验隐含均价是否落在 [low, high]
viol, nbar, imp_lo, imp_hi = {}, {}, {}, {}
snap_arr = e3.mult_snap.reindex(SYMS).to_numpy()
mult_per_bar = snap_arr[g_sym]
implied = tov[valid] / vol[valid] / mult_per_bar
lo_v, hi_v = lo[valid], hi[valid]
tol = 1e-6 + 5e-4 * np.abs(implied)
bad = (implied < lo_v - tol) | (implied > hi_v + tol)
vb = pd.Series(bad).groupby(g_sym).mean()
vb.index = SYMS[vb.index]
e3['px_out_of_range_pct'] = vb
e3['n_bars'] = pd.Series(np.ones(valid.sum())).groupby(g_sym).size().rename(
    index=lambda i: SYMS[i])
print('\n每品种反推乘数 + 隐含均价越界比例:')
print(e3.to_string(float_format=lambda x: f'{x:.6g}'))
e3.to_csv(os.path.join(OUT, 'qc_E3_multiplier.csv'))

print('\nE4  逐年乘数 (找台阶):')
print(mult_yr.round(2).to_string())
mult_yr.to_csv(os.path.join(OUT, 'qc_E4_multiplier_by_year.csv'))

ratio = mult_yr.max(axis=1) / mult_yr.min(axis=1)
jumped = ratio[ratio > 1.15].sort_values(ascending=False)
print(f'\nE4  年度乘数 max/min > 1.15 的品种 ({len(jumped)} 个):')
print(jumped.to_string(float_format=lambda x: f'{x:.3f}') if len(jumped) else '  无')

# ================================================================ F1
hr('F1  --  成交量/持仓量单双边口径台阶')

# 日聚合 (依赖 key 单调; 若不单调则退回 pandas groupby)
if key_mono:
    b = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate([[0], b])
    ends = np.concatenate([b, [len(key)]]) - 1
    gk = key[starts]
    dv = np.add.reduceat(vol, starts)
    dt_ = np.add.reduceat(tov, starts)
    daily = pd.DataFrame({
        'sym': SYMS[(gk // len(d_uniq)).astype(int)],
        'date': d_uniq[(gk % len(d_uniq)).astype(int)],
        'volume': dv, 'turnover': dt_, 'oi_end': oi[ends],
        'bars': ends - starts + 1, 'close': cl[ends],
    })
else:
    daily = df.groupby([sym_cat, df['trading_date']]).agg(
        volume=('volume', 'sum'), turnover=('total_turnover', 'sum'),
        oi_end=('open_interest', 'last'), bars=('close', 'size'),
        close=('close', 'last')).reset_index()
    daily.columns = ['sym', 'date', 'volume', 'turnover', 'oi_end', 'bars', 'close']
daily.to_pickle(os.path.join(OUT, 'qc_daily_agg.pkl'))
log(f'daily agg: {len(daily):,} rows -> qc_daily_agg.pkl')

def step_scan(field, label):
    piv = daily.pivot(index='date', columns='sym', values=field)
    lv = np.log(piv.where(piv > 0))
    d = lv.diff()
    res = pd.DataFrame({
        'med': d.median(axis=1),
        'n': d.notna().sum(axis=1),
        'frac_drop_40pct': (d < np.log(0.60)).sum(axis=1) / d.notna().sum(axis=1),
        'frac_rise_40pct': (d > np.log(1.0 / 0.60)).sum(axis=1) / d.notna().sum(axis=1),
    })
    res = res[res.n >= 20]
    print(f'\n--- {label}: 横截面 log 日变化中位数, 最负的 15 天 ---')
    print(res.nsmallest(15, 'med').to_string(float_format=lambda x: f'{x:.4f}'))
    print(f'\n--- {label}: 同日超过 40% 品种一起腰斩(-40%以上) 的日子 ---')
    su = res[res.frac_drop_40pct > 0.40].sort_values('frac_drop_40pct', ascending=False)
    print(su.head(15).to_string(float_format=lambda x: f'{x:.4f}') if len(su) else '  无')
    print(f'\n{label}: |中位数| 最大值 = {res.med.abs().max():.4f} '
          f'(口径由双边改单边应 ≈ {np.log(0.5):.4f})')
    res.to_csv(os.path.join(OUT, f'qc_F1_step_{field}.csv'))
    return res

r_vol = step_scan('volume', 'volume')
r_oi = step_scan('oi_end', 'open_interest')

# 若 volume 改口径而 turnover 未改, 乘数会翻倍 -> 已在 E4 覆盖; 这里直接看 turnover/volume 比
piv_r = daily.assign(r=daily.turnover / daily.volume.where(daily.volume > 0)) \
             .pivot(index='date', columns='sym', values='r')
dr = np.log(piv_r.where(piv_r > 0)).diff()
mr = dr.median(axis=1)
mr = mr[dr.notna().sum(axis=1) >= 20]
print(f'\nturnover/volume 比值的横截面 log 日变化, |中位数| 最大 = {mr.abs().max():.4f}')
print(mr.reindex(mr.abs().nlargest(8).index).to_string(float_format=lambda x: f'{x:.4f}'))

hr('DONE')
log('all outputs ->', OUT)
