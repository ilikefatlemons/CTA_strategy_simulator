# check_zero_seed_hypothesis.awk -- 检验「零价块的种子 = 该合约当节的第一笔成交」。
#
# 命中的话，缺陷就从笼统的"某些 bar 价格是 0"收敛到一个具体位置：
# 分钟 K 线构造器在开盘时对每个合约的第一笔成交没有初始化 OHLC。
#
# 注意块边界必须按「前一根是否也是 close<=0」来判定，不能按时段内首个零价 bar ——
# 跨夜盘到日盘的长块会在 09:01 处被误切成一个新块，把命中率压低。
#
# 时段划分：夜盘 = bartime >= 21:00 或 <= 02:59；其余为日盘。
# 输出：每个真实块一行 -> 结算日, 交易所, 合约, 时段, 种子时刻, 该节首笔成交时刻, 是否命中

BEGINFILE {
    cday = FILENAME; sub(/^.*future_pricemin/, "", cday); sub(/\.txt$/, "", cday)
    delete first; delete seed; delete seedsess; delete mkt_of
    cur = ""; prevzero = 0; nseed = 0
}

FNR == 1 { off = ($0 ~ /Offset/) ? 1 : 0; FS = (off == 1) ? "\t" : ","; next }

{
    cid = $3; bt = $(5 + off); cl = $(6 + off) + 0; vol = $(10 + off) + 0
    if (cid != cur) { cur = cid; prevzero = 0; mkt_of[cid] = $4 }

    sess = (bt >= "21:00" || bt <= "02:59") ? "N" : "D"
    k = cid SUBSEP sess
    if (vol > 0 && !(k in first)) first[k] = $2 " " bt

    if (cl <= 0 && !prevzero) {          # 真正的块起点：上一根还是正常的
        nseed++
        seed[nseed] = cid SUBSEP sess SUBSEP ($2 " " bt)
    }
    prevzero = (cl <= 0)
}

ENDFILE {
    for (i = 1; i <= nseed; i++) {
        split(seed[i], a, SUBSEP)
        k = a[1] SUBSEP a[2]
        f = (k in first) ? first[k] : "无成交"
        printf "%s,%s,%s,%s,%s,%s,%d\n", cday, mkt_of[a[1]], a[1], a[2], a[3], f, (f == a[3]) ? 1 : 0
    }
}
