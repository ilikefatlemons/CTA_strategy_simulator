# -*- coding: utf-8 -*-
"""
清单剩余项全查 + 每品种"有效起始日"
  E1 零价/NaN      E2 OHLC 逻辑    E5 量额一致   E6 负值    E7 OI 归零
  E8 单 bar 尖刺   E9 超停板跳空   A5 疫情夜盘   A6 夜盘引入  A7/A8 涨跌停封死
  B2 padding 对波动率的影响        B9 集合竞价 bar
  C2 重复时间戳    C3 盘中换月     C4 单调性     C6/F5 时段历史台阶
  D8 复权价偏离    F4 上市初期不可用 -> usable_from
"""
import os, time
import numpy as np
import pandas as pd

PKL = r"D:\2026_Summer\TradingApp\data\v3.0\all_symbol_min_full_main_close_k_1.pkl"
OUT = r"D:\2026_Summer\TradingApp\data\v3.0\test_data"
STEP1 = ['ZC', 'JR', 'WH', 'RS', 'PM', 'RI', 'LR']
STEP2 = ['BB', 'FB']
STEP3 = ['WR', 'RR']
DEFER = ['AO', 'BR', 'EC', 'IM', 'LC', 'PX', 'SH', 'SI', 'TL']
_t0 = time.time()
def log(*a): print(f'[{time.time()-_t0:6.1f}s]', *a, flush=True)
def hr(t):
    print('\n' + '=' * 94, flush=True); print(t, flush=True); print('=' * 94, flush=True)
pd.set_option('display.width', 250, 'display.max_columns', 60, 'display.max_rows', 400)

df = pd.read_pickle(PKL); log('loaded')
sym_cat = df['underlying_symbol'].astype('category')
con_cat = df['contract'].astype('category')
sc = sym_cat.cat.codes.to_numpy(); SYMS = np.asarray(sym_cat.cat.categories)
cc = con_cat.cat.codes.to_numpy(); CONS = np.asarray(con_cat.cat.categories)
op = df['open'].to_numpy(); hi = df['high'].to_numpy()
lo = df['low'].to_numpy(); cl = df['close'].to_numpy()
vol = df['volume'].to_numpy(); tov = df['total_turnover'].to_numpy()
oi = df['open_interest'].to_numpy(); Kv = df['K'].to_numpy()
idx = df.index
mod = (idx.hour * 60 + idx.minute).to_numpy().astype(np.int32)
ts = idx.to_numpy()
d_codes, d_uniq = pd.factorize(df['trading_date'], sort=True)
d_uniq = pd.DatetimeIndex(d_uniq); ND = len(d_uniq)
yr = d_uniq.year.to_numpy()[d_codes]
N = len(cl)

POOL = [s for s in SYMS if s not in STEP1 + STEP2 + STEP3 + DEFER]
pool_mask_sym = np.array([s in POOL for s in SYMS])
log(f'POOL_NOW={len(POOL)}  deferred={len(DEFER)}  excluded={len(STEP1+STEP2+STEP3)}')

def per_sym(vals, agg='sum'):
    r = getattr(pd.Series(vals).groupby(sc), agg)()
    r.index = SYMS[r.index]; return r

def show(series, title, topn=25):
    s = series[series > 0].sort_values(ascending=False)
    print(f'{title}: 命中品种 {len(s)}')
    if len(s):
        d = pd.DataFrame({'n': s})
        d['in_pool'] = [i in POOL for i in d.index]
        print(d.head(topn).to_string())

# (symbol, trading_date) 分组边界
key = sc.astype(np.int64) * ND + d_codes
assert (np.diff(key) >= 0).all(), 'key 非单调'
bnd = np.flatnonzero(np.diff(key)) + 1
gs = np.concatenate([[0], bnd]); ge = np.concatenate([bnd, [N]])
gkey = key[gs]; g_sym = (gkey // ND).astype(int); g_date = d_uniq[(gkey % ND).astype(int)]
gn = ge - gs
log(f'{len(gs):,} 个 (品种,交易日) 组')

# ================================================== E1 E2 E5 E6
hr('E1 / E2 / E5 / E6  --  数值一致性')
nan_px = np.isnan(op) | np.isnan(hi) | np.isnan(lo) | np.isnan(cl)
zero_px = (~nan_px) & ((op <= 0) | (hi <= 0) | (lo <= 0) | (cl <= 0))
print(f'E1 OHLC 含 NaN 的 bar: {int(nan_px.sum()):,} | OHLC 含 <=0 的 bar: {int(zero_px.sum()):,}')
show(per_sym(zero_px), 'E1 零价')
if zero_px.any():
    b = np.flatnonzero(zero_px)[:6]
    print(pd.DataFrame({'ts': ts[b], 'sym': SYMS[sc[b]], 'contract': CONS[cc[b]],
                        'o': op[b], 'h': hi[b], 'l': lo[b], 'c': cl[b],
                        'vol': vol[b], 'oi': oi[b]}).to_string(index=False))

ok = ~nan_px & ~zero_px
e2a = ok & (hi < lo)
e2b = ok & (hi < np.maximum(op, cl) - 1e-9)
e2c = ok & (lo > np.minimum(op, cl) + 1e-9)
print(f'\nE2 high<low: {int(e2a.sum()):,} | high<max(o,c): {int(e2b.sum()):,} | '
      f'low>min(o,c): {int(e2c.sum()):,}')
show(per_sym(e2a | e2b | e2c), 'E2 OHLC 逻辑破损')

e5 = ((vol == 0) & (tov > 0)) | ((vol > 0) & (tov == 0))
print(f'\nE5 量额矛盾 (一个为0另一个不为0): {int(e5.sum()):,}')
show(per_sym(e5), 'E5 量额矛盾')
e6 = (vol < 0) | (tov < 0) | (oi < 0)
print(f'E6 负值 (vol/turnover/OI): {int(e6.sum()):,}')
show(per_sym(e6), 'E6 负值')

# ================================================== E7 OI
hr('E7  --  持仓量归零 / NaN')
oi0 = (oi == 0) | np.isnan(oi)
t = pd.DataFrame({'oi0_pct': per_sym(oi0, 'mean'), 'n': per_sym(np.ones(N, np.int64))})
t['in_pool'] = [s in POOL for s in t.index]
print('OI 为 0 或 NaN 的比例 (>0.1% 的品种):')
print(t[t.oi0_pct > .001].sort_values('oi0_pct', ascending=False).to_string(
    float_format=lambda x: f'{x:.5f}'))
print(f'\n池内 {len(POOL)} 品种 OI 归零比例: max={t[t.in_pool].oi0_pct.max():.6f} '
      f'({t[t.in_pool].oi0_pct.idxmax()})')

# ================================================== E8 尖刺
hr('E8 / E9  --  单 bar 尖刺与超停板跳空')
same = np.zeros(N, bool)
same[1:] = (sc[1:] == sc[:-1]) & (cc[1:] == cc[:-1]) & (d_codes[1:] == d_codes[:-1])
good = ok & np.roll(ok, 1) & same
r = np.zeros(N)
np.divide(cl, np.roll(cl, 1), out=r, where=good)
r = np.where(good & (r > 0), np.log(r, where=(r > 0)), 0.0)
nxt = np.roll(r, -1); nxt_ok = np.roll(good, -1)
spike = (np.abs(r) > 0.03) & nxt_ok & (nxt * r < 0) & (np.abs(nxt) > 0.5 * np.abs(r))
print(f'E8 V 形尖刺 (单分钟>3% 且下一根反向回补过半): {int(spike.sum()):,}')
show(per_sym(spike), 'E8 尖刺')
if spike.any():
    b = np.flatnonzero(spike)[:8]
    print(pd.DataFrame({'ts': ts[b], 'sym': SYMS[sc[b]], 'prev': cl[b - 1], 'c': cl[b],
                        'next': cl[b + 1], 'vol': vol[b]}).to_string(index=False))
print(f'\n单分钟 |对数收益| 分布 (同合约同日内): p99.9={np.quantile(np.abs(r[good]),.999):.5f} '
      f'p99.99={np.quantile(np.abs(r[good]),.9999):.5f} max={np.abs(r[good]).max():.5f}')

dcl = cl[ge - 1]
dprev = np.concatenate([[np.nan], dcl[:-1]])
dsame = np.concatenate([[False], g_sym[1:] == g_sym[:-1]])
dr = np.where(dsame & (dprev > 0) & (dcl > 0), np.log(dcl / dprev), np.nan)
big = np.abs(dr) > 0.15
print(f'E9 日收盘对数变动 >15% 的 (品种,交易日): {int(np.nansum(big)):,}')
if np.nansum(big):
    bb = pd.DataFrame({'sym': SYMS[g_sym[big]], 'date': g_date[big], 'r': dr[big]})
    print(bb.groupby('sym').size().sort_values(ascending=False).head(12).to_string())
    print(bb.reindex(bb.r.abs().sort_values(ascending=False).index).head(8).to_string(index=False))

# ================================================== C2 C3 C4
hr('C2 / C3 / C4  --  重复时间戳、盘中换月、单调性')
dup = pd.DataFrame({'s': sc, 't': ts}).duplicated(['s', 't'])
print(f'C2 (品种,时间戳) 重复行: {int(dup.sum()):,}')
if dup.any():
    show(per_sym(dup.to_numpy()), 'C2 重复')
nc = pd.Series(cc).groupby(key).nunique()
multi = int((nc > 1).sum())
print(f'C3 同一 (品种,交易日) 内出现多个 contract 的组: {multi:,} / {len(gs):,}')
if multi:
    bad = nc[nc > 1].index.to_numpy()
    bs = (bad // ND).astype(int)
    d = pd.Series(bs).value_counts(); d.index = SYMS[d.index]
    dd = pd.DataFrame({'n_days': d}); dd['in_pool'] = [i in POOL for i in dd.index]
    print(dd.head(15).to_string())
mono = pd.Series(ts).groupby(sc).apply(lambda x: bool(x.is_monotonic_increasing))
mono.index = SYMS[mono.index]
print(f'C4 时间戳单调递增的品种: {int(mono.sum())}/{len(mono)}  '
      f'非单调: {mono.index[~mono].tolist()}')

# ================================================== A5 A6
hr('A5 / A6  --  疫情期夜盘停摆、夜盘引入时点')
night = (mod >= 21 * 60) | (mod <= 179)
gn_night = np.add.reduceat(night.astype(np.int64), gs)
nd = pd.DataFrame({'sym': SYMS[g_sym], 'date': g_date, 'n_night': gn_night,
                   'n_bar': gn, 'in_pool': [s in POOL for s in SYMS[g_sym]]})
cov = nd[nd.in_pool].groupby('date').apply(
    lambda g: (g.n_night > 0).mean(), include_groups=False)
w = cov['2020-01-01':'2020-06-01']
print('2020 年 1-5 月, 池内品种"当日有夜盘 bar"的比例 (按周):')
print(w.resample('W').mean().to_string(float_format=lambda x: f'{x:.3f}'))
first_night = nd[nd.n_night > 0].groupby('sym').date.min()
print(f'\nA6 各品种首个带夜盘的交易日 (前 20):')
print(first_night.sort_values().head(20).to_string())

# ================================================== A8 涨跌停封死
hr('A7 / A8  --  涨跌停封死 (整日无价格波动但有成交)')
dhi = np.maximum.reduceat(hi, gs); dlo = np.minimum.reduceat(lo, gs)
dvol = np.add.reduceat(vol, gs)
locked = (dhi == dlo) & (dvol > 0)
print(f'整日 high==low 且有成交的 (品种,交易日): {int(locked.sum()):,}')
lk = pd.DataFrame({'sym': SYMS[g_sym[locked]], 'date': g_date[locked],
                   'vol': dvol[locked]})
if len(lk):
    x = lk.groupby('sym').size().sort_values(ascending=False)
    d = pd.DataFrame({'n_days': x}); d['in_pool'] = [i in POOL for i in d.index]
    print(d.head(20).to_string())
    print('\n最近 8 例:'); print(lk.sort_values('date').tail(8).to_string(index=False))

# ================================================== B9 集合竞价
hr('B9  --  集合竞价 bar')
auc = np.isin(mod, [9 * 60, 21 * 60, 9 * 60 + 30])
first_of_grp = np.zeros(N, bool); first_of_grp[gs] = True
print(f'落在 09:00 / 21:00 / 09:30 的 bar: {int(auc.sum()):,}')
mv = np.median(vol[auc & (vol > 0)]); mv2 = np.median(vol[~auc & (vol > 0)])
print(f'竞价 bar 成交量中位数 {mv:,.0f}  vs  普通 bar {mv2:,.0f}  (倍数 {mv/mv2:.2f}x)')

# ================================================== C6 / F5 时段台阶
hr('C6 / F5  --  每日 bar 数的历史台阶')
bp = pd.DataFrame({'sym': SYMS[g_sym], 'year': pd.DatetimeIndex(g_date).year, 'n': gn})
pv = bp.groupby(['sym', 'year']).n.median().unstack()
pv_pool = pv.loc[[s for s in POOL if s in pv.index]]
nvar = pv_pool.apply(lambda r: r.dropna().nunique(), axis=1)
print(f'池内每日 bar 数在历史上出现过 >1 种取值的品种: {int((nvar>1).sum())}/{len(pv_pool)}')
print('\n变化最多的 15 个 (逐年每日 bar 数中位):')
print(pv_pool.loc[nvar.sort_values(ascending=False).head(15).index].to_string())
pv.to_csv(os.path.join(OUT, 'qc_F5_bars_per_day_by_year.csv'))

# ================================================== D8 复权价偏离
hr('D8  --  复权价对真实价的偏离幅度 (close/K)')
anchor = pd.read_csv(os.path.join(OUT, 'qc_D2_anchor.csv'), index_col=0)
a = anchor.loc[[s for s in POOL if s in anchor.index]].copy()
a['dev_max'] = (1 / a.K_min - 1).abs().combine((1 / a.K_max - 1).abs(), max)
print('复权价 = close/K, 偏离真实价最大的 15 个池内品种:')
print(a.sort_values('dev_max', ascending=False).head(15)[
    ['K_first', 'K_last', 'K_min', 'K_max', 'dev_max', 'n_runs']].to_string(
    float_format=lambda x: f'{x:.4f}'))

# ================================================== F4 有效起始日
hr('F4  --  每品种"有效起始日" (padding 长期回落到 5% 以下之后)')
pad = (vol == 0) & (tov == 0) & (op == hi) & (hi == lo) & (lo == cl)
gpad = np.add.reduceat(pad.astype(np.int64), gs) / gn
doi = oi[ge - 1]
dd = pd.DataFrame({'sym': SYMS[g_sym], 'date': g_date, 'pad': gpad,
                   'vol': dvol, 'oi': doi, 'bars': gn}).sort_values(['sym', 'date'])
rows = []
for s, sub in dd.groupby('sym', observed=True):
    roll = sub.pad.rolling(60, min_periods=30).median()
    bad = roll >= 0.05
    if bad.any():
        last_bad = sub.date[bad].max()
        after = sub.date[sub.date > last_bad]
        start = after.min() if len(after) else pd.NaT
    else:
        start = sub.date.iloc[0]
    lost = int((sub.date < start).sum()) if pd.notna(start) else len(sub)
    rows.append({'sym': s, 'first': sub.date.iloc[0], 'usable_from': start,
                 'sessions_all': len(sub), 'sessions_lost': lost,
                 'lost_pct': lost / len(sub),
                 'pad_first_year': sub.pad.head(250).median(),
                 'oi_first_year': sub.oi.head(250).median(),
                 'vol_first_year': sub.vol.head(250).median()})
uf = pd.DataFrame(rows).set_index('sym')
uf['in_pool'] = [s in POOL for s in uf.index]
up = uf[uf.in_pool].sort_values('sessions_lost', ascending=False)
print(f'池内 {len(up)} 品种, 按"被判为不可用的早期 session 数"排序 (前 25):')
print(up.head(25).to_string(float_format=lambda x: f'{x:.4f}'))
print(f'\n完全无早期不可用段的池内品种: {int((up.sessions_lost == 0).sum())}/{len(up)}')
print(f'损失 >10% 历史的池内品种: {up.index[up.lost_pct > .10].tolist()}')
uf.to_csv(os.path.join(OUT, 'qc_F4_usable_from.csv'))

print('\n--- B (豆二) 逐年明细 ---')
b = dd[dd.sym == 'B'].copy(); b['year'] = pd.DatetimeIndex(b.date).year
print(b.groupby('year').agg(sessions=('pad', 'size'), pad_med=('pad', 'median'),
                            oi_med=('oi', 'median'), vol_med=('vol', 'median')
                            ).to_string(float_format=lambda x: f'{x:.4f}'))

hr('DONE'); log('done')
