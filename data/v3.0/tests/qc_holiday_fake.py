# -*- coding: utf-8 -*-
"""元旦/长假 bar 定性: 伪造(零成交) 还是 真实跨夜盘?"""
import numpy as np, pandas as pd, time
PKL = r"D:\2026_Summer\TradingApp\data\v3.0\all_symbol_min_full_main_close_k_1.pkl"
_t0 = time.time()
df = pd.read_pickle(PKL); print(f'[{time.time()-_t0:.1f}s] loaded', flush=True)
sym_cat = df['underlying_symbol'].astype('category')
sc = sym_cat.cat.codes.to_numpy(); SYMS = np.asarray(sym_cat.cat.categories)
idx = df.index
mod = (idx.hour * 60 + idx.minute).to_numpy().astype(np.int32)
cal = idx.normalize().to_numpy(); wd = idx.dayofweek.to_numpy()
td = df['trading_date'].to_numpy(); vol = df['volume'].to_numpy()
op = df['open'].to_numpy(); hi = df['high'].to_numpy()
lo = df['low'].to_numpy(); cl = df['close'].to_numpy()
pad = (vol == 0) & (op == hi) & (hi == lo) & (lo == cl)
is_eve = mod >= 21*60; is_am = mod <= 179; is_day = (~is_eve) & (~is_am)

print('\n=== (1) 元旦"日盘" bar (td != cal) ===')
m = is_day & (td != cal)
print(f'n={int(m.sum()):,}  volume==0 比例={float((vol[m]==0).mean()):.6f}  '
      f'padding 比例={float(pad[m].mean()):.6f}  成交量合计={vol[m].sum():,.0f}')

print('\n=== (2) 非周六凌晨盘 (td != cal) ===')
m2 = is_am & (td != cal) & (wd != 5)
print(f'n={int(m2.sum()):,}  volume==0 比例={float((vol[m2]==0).mean()):.6f}  '
      f'padding 比例={float(pad[m2].mean()):.6f}  成交量合计={vol[m2].sum():,.0f}')
for d in sorted(set(pd.Timestamp(x).date() for x in cal[m2])):
    k = m2 & (cal == np.datetime64(str(d)))
    print(f'   {d}: n={int(k.sum()):,} vol0={float((vol[k]==0).mean()):.4f} '
          f'成交量={vol[k].sum():,.0f}')

print('\n=== 对照: 周六凌晨盘 (正常的周五夜盘延续) ===')
m3 = is_am & (wd == 5)
print(f'n={int(m3.sum()):,}  volume==0 比例={float((vol[m3]==0).mean()):.6f}  '
      f'成交量合计={vol[m3].sum():,.0f}')

print('\n=== (3) 受伪造日盘污染的 trading_date, 该日 bar 数是否翻倍 ===')
bad_td = np.unique(td[m])
for t in bad_td[:12]:
    k = (td == t) & (sc == sc[m][0])
    s = SYMS[sc[m][0]]
    print(f'   {s} trading_date={pd.Timestamp(t).date()}: bar={int(k.sum())} '
          f'(其中零成交 {int((vol[k]==0).sum())}), 日历日 '
          f'{sorted({str(pd.Timestamp(x).date()) for x in cal[k]})}')

print('\n=== (4) 全库: 长假前被取消的夜盘中, 伪造 bar 的规模 ===')
night = is_eve | is_am
dn = pd.DataFrame({'s': sc, 'td': td, 'night': night, 'nv': np.where(night, vol, 0)})
gg = dn.groupby(['s', 'td'], observed=True).agg(has_night=('night', 'any'),
                                                night_vol=('nv', 'sum'),
                                                night_bars=('night', 'sum'))
gg = gg.reset_index()
fake = gg[(gg.has_night) & (gg.night_vol == 0)]
print(f'有夜盘 bar 但夜盘成交量恒为 0 的 (品种,交易日) 组合: {len(fake):,}')
print(f'涉及 bar 数: {int(fake.night_bars.sum()):,}')
print(f'按交易日统计, 出现最多的 15 天:')
ft = fake.groupby('td').size().sort_values(ascending=False).head(15)
print('\n'.join(f'   {pd.Timestamp(k).date()}  {v} 个品种' for k, v in ft.items()))
print(f'\n[{time.time()-_t0:.1f}s] done')
