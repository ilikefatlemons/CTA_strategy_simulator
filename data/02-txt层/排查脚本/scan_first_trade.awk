# scan_first_trade.awk -- 统计「某合约某交易时段的第一笔成交」这件事的总体基数与出错率。
#
# 已知零价块的种子 100% 是「该合约该时段的第一笔成交」。但这只说明了充分性，
# 没说明必要性 —— 绝大多数首笔成交是正常的。要判断故障是随机还是有条件，
# 必须拿到分母：全市场一共有多少次「时段首笔成交」，其中多少次 OHLC 被写成 0。
#
# 同时记录两个可能的条件变量：
#   gap  —— 首笔成交距该时段开始隔了多少根 bar（1 = 开盘第一分钟就成交）
#   vol  —— 首笔成交的手数（用来看是不是只发生在清淡合约上）
#
# 时段：夜盘 = bartime >= 21:00 或 <= 02:59；其余为日盘。
#
# 输出两条流：
#   F 行 —— 逐 (结算日, 交易所, 品种, 时段) 汇总：首笔成交次数、其中 OHLC=0 的次数，
#           再按 gap 是否为 1 拆开
#   Z 行 —— 每一次 OHLC=0 的首笔成交的明细，用于给出可核查的案例

BEGINFILE {
    cday = FILENAME; sub(/^.*future_pricemin/, "", cday); sub(/\.txt$/, "", cday)
    delete seen; delete idx0; delete nf; delete nz; delete nf1; delete nz1
    cur = ""
}

FNR == 1 { off = ($0 ~ /Offset/) ? 1 : 0; FS = (off == 1) ? "\t" : ","; next }

{
    cid = $3; bt = $(5 + off); cl = $(6 + off) + 0; op = $(7 + off) + 0
    hi = $(8 + off) + 0; lo = $(9 + off) + 0; vol = $(10 + off) + 0
    vwp = $(12 + off) + 0
    if (cid != cur) { cur = cid; delete si; prevzero = 0; prevvol = -1 }

    sess = (bt >= "21:00" || bt <= "02:59") ? "N" : "D"
    k = cid SUBSEP sess
    si[sess]++                                  # 该合约在该时段的第 si 根 bar

    if (vol > 0 && !(k in seen)) {              # 该合约该时段的第一笔成交
        seen[k] = 1
        prod = cid; sub(/[0-9]+$/, "", prod)
        g = prod SUBSEP sess
        nf[g]++
        gap = si[sess]
        if (gap == 1) nf1[g]++
        z = (op <= 0 || hi <= 0 || lo <= 0 || cl <= 0)
        if (z) {
            nz[g]++
            if (gap == 1) nz1[g]++
            # cont=1 表示上一根已经是零价，这次只是块的延续，不是一次独立故障
            printf "Z,%s,%s,%s,%s,%s,%s,%s,%d,%.0f,%s,%d,%d\n",
                cday, $4, prod, cid, sess, $2, bt, gap, vol, vwp,
                (cl <= 0 ? 15 : 5), prevzero
        }
        mkt_of[g] = $4
    }
    prevzero = (cl <= 0); prevvol = vol
}

ENDFILE {
    for (g in nf) {
        split(g, a, SUBSEP)
        printf "F,%s,%s,%s,%s,%d,%d,%d,%d\n",
            cday, mkt_of[g], a[1], a[2], nf[g], nz[g]+0, nf1[g]+0, nz1[g]+0
    }
}
