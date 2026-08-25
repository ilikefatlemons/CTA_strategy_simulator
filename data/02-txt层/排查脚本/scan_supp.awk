# scan_supp.awk -- 补充扫描，只在少量指定日期上跑，补两个主扫描没带的量：
#   1) 盘前（bt < 09:00）的 bar 总数 —— 用来数 padding 日里 00:00~02:30 被伪造了多少根
#   2) 集合竞价那一根的成交量，以及所属交易时段的总成交量 —— 用来算竞价 bar 的成交量占比
#      （郑商所把竞价成交与开盘后首批连续成交并进同一根，占比会显著偏高）
#
# 输出粒度 (clearing_day, market, product)。用法同 scan_raw.awk。

BEGINFILE {
    delete pre_bars; delete pre_vol
    delete night_vol; delete day_vol
    delete v2100; delete v0900; delete v0930
    delete prods

    cday = FILENAME
    sub(/^.*future_pricemin/, "", cday)
    sub(/\.txt$/, "", cday)
}

FNR == 1 { off = ($0 ~ /Offset/) ? 1 : 0; FS = (off == 1) ? "\t" : ","; next }

{
    mkt = $4
    bt  = $(5 + off)
    vol = $(10 + off) + 0

    prod = $3
    sub(/[0-9]+$/, "", prod)
    k = mkt SUBSEP prod
    prods[k] = 1

    # 夜盘 = 21:00 之后到次日 09:00 之前；日盘 = 09:00 ~ 15:15
    if (bt >= "21:00" || bt < "09:00") { night_vol[k] += vol } else { day_vol[k] += vol }
    if (bt < "09:00") { pre_bars[k]++; pre_vol[k] += vol }

    if (bt == "21:00") v2100[k] += vol
    if (bt == "09:00") v0900[k] += vol
    if (bt == "09:30") v0930[k] += vol
}

ENDFILE {
    for (k in prods) {
        split(k, p, SUBSEP)
        printf "%s,%s,%s,%d,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n",
            cday, p[1], p[2],
            pre_bars[k]+0, pre_vol[k]+0,
            night_vol[k]+0, day_vol[k]+0,
            v2100[k]+0, v0900[k]+0, v0930[k]+0
    }
}
