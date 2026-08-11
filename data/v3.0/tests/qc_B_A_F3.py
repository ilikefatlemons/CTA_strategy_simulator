# -*- coding: utf-8 -*-
"""
排查清单 4 / 5 / 6:
  4. B1 B3 B4 B5 B6 B7 B8 C1  -- padding、交易时段、夜盘归属
  5. A1 A2 A3                 -- 停更定性、缺失交易日
  6. F2 F3                    -- 合约代码格式 / 规格变更导致的历史拼接

注意: 全程只用整数 code 做掩码, 不要用 SYMS[sc]==s (会反复物化 6600 万元素的 object 数组)。
"""
import os, time
import numpy as np
import pandas as pd

PKL = r"D:\2026_Summer\TradingApp\data\v3.0\all_symbol_min_full_main_close_k_1.pkl"
OUT = r"D:\2026_Summer\TradingApp\data\v3.0\test_data"
EXCLUDED = ['ZC', 'JR', 'WH', 'RS', 'PM', 'RI', 'LR']          # Step 1 已排除
CFFEX = ['IF', 'IC', 'IH', 'IM', 'T', 'TF', 'TS', 'TL']        # 无小节休、无夜盘

_t0 = time.time()
def log(*a): print(f'[{time.time()-_t0:6.1f}s]', *a, flush=True)
def hr(t):
    print('\n' + '=' * 92, flush=True); print(t, flush=True); print('=' * 92, flush=True)
pd.set_option('display.width', 240, 'display.max_columns', 60, 'display.max_rows', 300)

df = pd.read_pickle(PKL); log('loaded')
sym_cat = df['underlying_symbol'].astype('category')
con_cat = df['contract'].astype('category')
sc = sym_cat.cat.codes.to_numpy(); SYMS = np.asarray(sym_cat.cat.categories)
cc = con_cat.cat.codes.to_numpy(); CONS = np.asarray(con_cat.cat.categories)
op = df['open'].to_numpy(); hi = df['high'].to_numpy(); lo = df['low'].to_numpy()
cl = df['close'].to_numpy(); vol = df['volume'].to_numpy(); tov = df['total_turnover'].to_numpy()
idx = df.index
mod = (idx.hour * 60 + idx.minute).to_numpy().astype(np.int32)
wd_ts = idx.dayofweek.to_numpy()
cal = idx.normalize().to_numpy()
td = df['trading_date'].to_numpy()

SID = {s: i for i, s in enumerate(SYMS)}
POOL = [s for s in SYMS if s not in EXCLUDED]
pool_codes = np.array([SID[s] for s in POOL])
cffex_codes = np.array([SID[s] for s in CFFEX if s in SID])
is_cffex = np.isin(sc, cffex_codes)                 # 整数掩码, 一次算完
recent = td >= np.datetime64('2024-08-01')
log(f'pool={len(POOL)}  excluded={len(EXCLUDED)}')

# 每个 symbol 的行区间 (数据已按 symbol 分块且有序)
b = np.flatnonzero(np.diff(sc)) + 1
seg_start = np.concatenate([[0], b]); seg_end = np.concatenate([b, [len(sc)]])
SEG = {int(sc[s]): (int(s), int(e)) for s, e in zip(seg_start, seg_end)}

def by_sym(vals, agg='mean'):
    r = getattr(pd.Series(vals).groupby(sc), agg)()
    r.index = SYMS[r.index]
    return r

# ================================================== C1 / B3 / B4
hr('C1 / B3 / B4  --  交易时段指纹 (近两年) 与休息时段是否被填充')
def blocks(minutes):
    u = np.unique(minutes)
    if len(u) == 0:
        return '(无)'
    brk = np.flatnonzero(np.diff(u) > 1)
    st = np.concatenate([[u[0]], u[brk + 1]]); en = np.concatenate([u[brk], [u[-1]]])
    return ' | '.join(f'{a//60:02d}:{a%60:02d}-{b//60:02d}:{b%60:02d}({b-a+1})'
                      for a, b in zip(st, en))

fp = {}
for code, (s, e) in sorted(SEG.items()):
    m = recent[s:e]
    fp[SYMS[code]] = blocks(mod[s:e][m]) if m.any() else '(近两年无数据)'
print(pd.Series(fp, name='fingerprint').to_string())

in_morning_brk = (~is_cffex) & (mod >= 10 * 60 + 16) & (mod <= 10 * 60 + 30)
in_lunch = np.where(is_cffex, (mod >= 11 * 60 + 31) & (mod <= 13 * 60),
                    (mod >= 11 * 60 + 31) & (mod <= 13 * 60 + 30))
print(f'\nB4 商品在小节休 10:16-10:30 的 bar 数: {int(in_morning_brk.sum()):,}')
if in_morning_brk.any():
    x = by_sym(in_morning_brk, 'sum'); print(x[x > 0].to_string())
print(f'B3 午休时段的 bar 数: {int(in_lunch.sum()):,}')
if in_lunch.any():
    x = by_sym(in_lunch, 'sum'); print(x[x > 0].to_string())

# ================================================== B1 padding
hr('B1  --  零成交 padding bar')
pad = (vol == 0) & (tov == 0) & (op == hi) & (hi == lo) & (lo == cl)
t = pd.DataFrame({'pad_pct': by_sym(pad), 'vol0_pct': by_sym(vol == 0),
                  'n': by_sym(np.ones(len(vol), np.int64), 'sum')})
t['in_pool'] = [s in POOL for s in t.index]
print(f'全库 padding: {int(pad.sum()):,} ({pad.mean():.4%}) | volume==0: '
      f'{int((vol==0).sum()):,} ({(vol==0).mean():.4%})')
print('\npadding 比例最高的 20 个:')
print(t.sort_values('pad_pct', ascending=False).head(20).to_string(
    float_format=lambda x: f'{x:.5f}'))
tp = t[t.in_pool]
print(f'\n池内 {len(tp)} 品种 padding: median={tp.pad_pct.median():.5f} '
      f'p90={tp.pad_pct.quantile(.9):.5f} max={tp.pad_pct.max():.5f} ({tp.pad_pct.idxmax()})')
print(f'池内 padding >1% 的品种: {tp.index[tp.pad_pct > .01].tolist()}')
print(f'池内 padding >5% 的品种: {tp.index[tp.pad_pct > .05].tolist()}')
t.to_csv(os.path.join(OUT, 'qc_B1_padding.csv'))

# ================================================== B5 夜盘
hr('B5  --  夜盘时段 (近两年)')
night = (mod >= 21 * 60) | (mod <= 179)
rows = []
for code, (s, e) in sorted(SEG.items()):
    m = recent[s:e] & night[s:e]
    if not m.any():
        rows.append({'sym': SYMS[code], 'has_night': False, 'night_start': '',
                     'night_end': '', 'n_night_bar': 0}); continue
    mm = mod[s:e][m]
    st = mm[mm >= 21 * 60].min() if (mm >= 21 * 60).any() else mm.min()
    en = mm[mm <= 179].max() if (mm <= 179).any() else mm.max()
    rows.append({'sym': SYMS[code], 'has_night': True,
                 'night_start': f'{st//60:02d}:{st%60:02d}',
                 'night_end': f'{en//60:02d}:{en%60:02d}', 'n_night_bar': int(m.sum())})
nt = pd.DataFrame(rows).set_index('sym')
nt['in_pool'] = [s in POOL for s in nt.index]
for en, grp in nt[nt.has_night].groupby('night_end'):
    print(f"  收于 {en} ({len(grp)} 个): {' '.join(sorted(grp.index))}")
print(f"\n无夜盘: {' '.join(sorted(nt.index[~nt.has_night]))}")
nt.to_csv(os.path.join(OUT, 'qc_B5_night_session.csv'))

# ================================================== B6 / B7
hr('B6 / B7  --  夜盘的 trading_date 归属')
td_wd = pd.DatetimeIndex(td).dayofweek.to_numpy()
is_eve = mod >= 21 * 60
is_am = mod <= 179
is_day = (~is_eve) & (~is_am)
print(f'B6 晚盘(21:00+) trading_date 未推到下一日: {int((is_eve & (td <= cal)).sum()):,}')
print(f'   凌晨(00:00-02:59) trading_date != 日历日: {int((is_am & (td != cal)).sum()):,}')
print(f'   日盘 trading_date != 日历日:              {int((is_day & (td != cal)).sum()):,}')
fri = is_eve & (wd_ts == 4)
print(f'\n周五晚盘 bar {int(fri.sum()):,} 的 trading_date 星期分布 (0=周一): '
      f'{pd.Series(td_wd[fri]).value_counts().sort_index().to_dict()}')
print(f'B7 trading_date 落在周末: {int(np.isin(td_wd, [5, 6]).sum()):,}')
print(f'   日历日落在周六且为凌晨盘 (正常): {int((np.isin(wd_ts,[5]) & is_am).sum()):,}')
print(f'   日历日落在周六但非凌晨盘 (异常): {int((np.isin(wd_ts,[5]) & ~is_am).sum()):,}')
print(f'   日历日落在周日: {int((wd_ts == 6).sum()):,}')

# ================================================== B8
hr('B8  --  长假前夜盘是否被伪造')
dn = pd.DataFrame({'s': sc, 'td': td, 'n': night}).groupby(['s', 'td'], observed=True).n.any()
dn = dn.reset_index()
dn['prev'] = dn.groupby('s').td.shift()
dn['gap'] = (dn.td - dn['prev']).dt.days
nightset = set(nt.index[nt.has_night])
dn['sym'] = SYMS[dn.s.to_numpy()]
h = dn[dn.sym.isin(nightset) & dn.gap.notna()]
bucket = pd.cut(h.gap, [0, 1, 3, 4, 6, 100],
                labels=['1天(正常)', '2-3天(周末)', '4天', '5-6天', '>6天(长假)'])
print('该交易日是否带夜盘, 按"距上一交易日间隔"分组:')
print(h.assign(b=bucket).groupby('b', observed=True).n.agg(['size', 'mean'])
       .rename(columns={'mean': 'has_night_rate'}).to_string(float_format=lambda x: f'{x:.4f}'))
bad8 = h[(h.gap > 6) & h.n]
print(f'\n长假(>6天)后首日仍带夜盘: {len(bad8)} 条')
if len(bad8):
    print(bad8[['sym', 'td', 'gap']].head(20).to_string(index=False))

# ================================================== A1 / A2
hr('A1 / A2  --  停更定性')
daily = pd.read_pickle(os.path.join(OUT, 'qc_daily_agg.pkl'))
last_all = daily.date.max()
rows = []
for s, sub in daily.groupby('sym', observed=True):
    sub = sub.sort_values('date')
    rows.append({'sym': s, 'first': sub.date.iloc[0], 'last': sub.date.iloc[-1],
                 'dead_days': (last_all - sub.date.iloc[-1]).days,
                 'bars_last': sub.bars.iloc[-1], 'bars_med': sub.bars.median(),
                 'vol_last5_ratio': sub.volume.tail(5).mean() / sub.volume.median(),
                 'oi_last_ratio': sub.oi_end.iloc[-1] / sub.oi_end.median(),
                 'oi_last': sub.oi_end.iloc[-1]})
a = pd.DataFrame(rows).set_index('sym'); a['cohort'] = a['first'].dt.year
dead = a[a.dead_days > 30].sort_values(['last', 'sym'])
print(f'停更品种 (dead_days>30): {len(dead)} 个')
print(dead.to_string(float_format=lambda x: f'{x:.3f}'))
print('\n按停更日分组 —— 看与"上市年份"是否相关, 以及停更时是否仍活着:')
for d, g in dead.groupby('last'):
    print(f"  {pd.Timestamp(d).date()}  n={len(g):2d}  上市年份 {sorted(g.cohort.unique())}  "
          f"末日OI/中位OI 中位={g.oi_last_ratio.median():.3f}  "
          f"末5日量/中位量 中位={g.vol_last5_ratio.median():.3f}")
    print(f"      {' '.join(sorted(g.index))}")
a.to_csv(os.path.join(OUT, 'qc_A1_truncation.csv'))

# ================================================== A3
hr('A3  --  缺失交易日 (全市场并集日历为基准, 只看各自存活区间)')
alld = np.sort(daily.date.unique())
miss = []
for s, sub in daily.groupby('sym', observed=True):
    lo_, hi_ = sub.date.min(), sub.date.max()
    expect = alld[(alld >= lo_) & (alld <= hi_)]
    gone = np.setdiff1d(expect, sub.date.to_numpy())
    miss.append({'sym': s, 'n_expect': len(expect), 'n_miss': len(gone),
                 'miss_pct': len(gone) / max(len(expect), 1),
                 'sample': ' '.join(str(pd.Timestamp(d).date()) for d in gone[:6])})
ms = pd.DataFrame(miss).set_index('sym').sort_values('n_miss', ascending=False)
ms['in_pool'] = [s in POOL for s in ms.index]
print(ms[ms.n_miss > 0].to_string(float_format=lambda x: f'{x:.5f}'))
print(f'\n零缺失: {int((ms.n_miss == 0).sum())}/{len(ms)} | '
      f'池内缺失>1% 的品种: {ms.index[(ms.miss_pct > .01) & ms.in_pool].tolist()}')
ms.to_csv(os.path.join(OUT, 'qc_A3_missing_sessions.csv'))

# ================================================== F2 / F3
hr('F2 / F3  --  合约代码格式与规格变更')
cinfo = []
for code, (s, e) in sorted(SEG.items()):
    sym = SYMS[code]
    u = np.unique(CONS[cc[s:e]])
    digits = sorted({len(c) - len(sym) for c in u})
    cinfo.append({'sym': sym, 'n_contract': len(u), 'digit_len': str(digits),
                  'first_contract': u.min(), 'last_contract': u.max()})
ci = pd.DataFrame(cinfo).set_index('sym'); ci['in_pool'] = [s in POOL for s in ci.index]
for d, g in ci.groupby('digit_len'):
    print(f"  年月位数 {d}: {' '.join(sorted(g.index))}")
mixed = ci[ci.digit_len.str.contains(',')]
print(f'\n位数不唯一 (历史换过代码格式) 的品种: {len(mixed)}')
if len(mixed):
    print(mixed.to_string())
ci.to_csv(os.path.join(OUT, 'qc_F2_contract_format.csv'))

mult = pd.read_csv(os.path.join(OUT, 'qc_E4_multiplier_by_year_fixed.csv'), index_col=0)
print('\nF3  乘数变更过的品种 (跨变更点历史不可比):')
for s in mult.index:
    row = mult.loc[s].dropna()
    if row.nunique() <= 1:
        continue
    v = row.to_numpy()
    yrs = row.index[np.flatnonzero(v[1:] != v[:-1]) + 1].tolist()
    print(f"  {s:4s} {' -> '.join(str(int(x)) for x in pd.unique(v)):<14s} "
          f"变更年份 {yrs}   {'[池内]' if s in POOL else '[Step1已排除]'}")

hr('DONE'); log('done')
