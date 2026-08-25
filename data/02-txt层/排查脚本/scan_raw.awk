# scan_raw.awk -- 单遍扫描 data/02-txt层/原始数据/future_pricemin*.txt，输出三条聚合流。
#
# raw 有且仅有两套表头，以 2022-09-22 为界：
#   旧（<=20220921）TAB 分隔、14 列、第 5 列是 Offset
#   新（>=20220922）逗号分隔、13 列、无 Offset
# 用第一行是否含 "Offset" 判别，列偏移量 off 吸收这个差异。
#
# 流 A：粒度 (clearing_day, tdate, market, product)  时段结构 / 集合竞价 / padding / 负值 / vwap
# 流 B：粒度 (clearing_day, market, product)         逐合约全日成交额是否被凑成整万元
# 流 C：粒度 (clearing_day, market, contract)        盘前带量 bar 只有 1~3 根的合约，
#        带上该根的价量、其前一根零成交 padding 价、以及当日日盘首根的开盘价 —— 用来
#        判定这根到底是"夜盘尾部成交"还是"被错标的日盘集合竞价"。
#
# gawk 专用（BEGINFILE / ENDFILE / delete arr）。

BEGINFILE {
    delete bars;   delete ncon;    delete conseen
    delete svol;   delete sval
    delete minbt;  delete maxbt
    delete n0900;  delete n0900v;  delete n0900f
    delete n2100;  delete n2100v;  delete n2100f
    delete n0930;  delete n0930v;  delete n0930f
    delete npre;   delete vpre;    delete minpret; delete maxpret
    delete nnegval; delete nnegvol; delete nnegoi
    delete nzvol;  delete nvwapout; delete nvwapout1; delete maxdev
    delete cval;   delete cvalkey; delete cmkt; delete cprod
    delete pc_n;   delete pc_t;    delete pc_v;  delete pc_c
    delete zc_t;   delete zc_c
    delete d0_t;   delete d0_o;    delete d0_v

    # clearing day 取自文件名，不依赖第一列，避免个别行串列时污染 key
    cday = FILENAME
    sub(/^.*future_pricemin/, "", cday)
    sub(/\.txt$/, "", cday)
}

FNR == 1 {
    off = ($0 ~ /Offset/) ? 1 : 0
    FS  = (off == 1) ? "\t" : ","
    next
}

{
    tdate = $2
    cid   = $3
    mkt   = $4
    bt    = $(5  + off)
    cl    = $(6  + off) + 0
    op    = $(7  + off) + 0
    hi    = $(8  + off) + 0
    lo    = $(9  + off) + 0
    vol   = $(10 + off) + 0
    val   = $(11 + off) + 0
    vwp   = $(12 + off) + 0
    oi    = $(13 + off) + 0

    prod = cid
    sub(/[0-9]+$/, "", prod)

    k = tdate SUBSEP mkt SUBSEP prod

    bars[k]++
    svol[k] += vol
    sval[k] += val

    if (!((k SUBSEP cid) in conseen)) { conseen[k SUBSEP cid] = 1; ncon[k]++ }

    if (!(k in minbt) || bt < minbt[k]) minbt[k] = bt
    if (!(k in maxbt) || bt > maxbt[k]) maxbt[k] = bt

    flat = (op == hi && hi == lo && lo == cl)

    if (bt == "09:00") { n0900[k]++; if (vol > 0) { n0900v[k]++; if (flat) n0900f[k]++ } }
    if (bt == "21:00") { n2100[k]++; if (vol > 0) { n2100v[k]++; if (flat) n2100f[k]++ } }
    if (bt == "09:30") { n0930[k]++; if (vol > 0) { n0930v[k]++; if (flat) n0930f[k]++ } }

    # 盘前（09:00 之前）带成交量的 bar —— 正常夜盘品种到处都是，
    # 但在"夜盘已取消却被 padding"的交易日里，这就是被错标的日盘集合竞价。
    if (bt < "09:00" && vol > 0) {
        npre[k]++; vpre[k] += vol
        if (!(k in minpret) || bt < minpret[k]) minpret[k] = bt
        if (!(k in maxpret) || bt > maxpret[k]) maxpret[k] = bt

        pc_n[cid]++
        if (!(cid in pc_t) || bt > pc_t[cid]) { pc_t[cid] = bt; pc_v[cid] = vol; pc_c[cid] = cl }
    }
    # 盘前零成交 bar 的最后一根：padding 段的参考价（= 上一有效交易日收盘价）
    if (bt < "09:00" && vol == 0) {
        if (!(cid in zc_t) || bt > zc_t[cid]) { zc_t[cid] = bt; zc_c[cid] = cl }
    }
    # 当日日盘首根（09:00~10:00 内最早的一根），取其开盘价
    if (bt >= "09:00" && bt <= "10:00") {
        if (!(cid in d0_t) || bt < d0_t[cid]) { d0_t[cid] = bt; d0_o[cid] = op; d0_v[cid] = vol }
    }

    if (val < 0) nnegval[k]++
    if (vol < 0) nnegvol[k]++
    if (oi  < 0) nnegoi[k]++
    if (vol == 0) nzvol[k]++

    # vwap 必须落在当根的 [low, high] 内。落在外面说明这根的 value 被塞进了
    # 不属于它的成交额（郑商所"凑整万元"），且这种 bar 的 value 是正数，
    # 用 value<0 的过滤器抓不到。0.1% 阈值会混入小数舍入噪声，同时统计 1% 阈值。
    if (vol > 0 && vwp > 0 && lo > 0) {
        dev = (vwp > hi) ? (vwp - hi) / hi : ((vwp < lo) ? (lo - vwp) / lo : 0)
        if (dev > 1e-3) { nvwapout[k]++;  if (dev > maxdev[k]) maxdev[k] = dev }
        if (dev > 1e-2) nvwapout1[k]++
    }

    # 流 B：逐合约全日成交额（跨 tdate 合并，一个 clearing day 就是一个结算日）
    if (!(cid in cvalkey)) { cvalkey[cid] = 1; cmkt[cid] = mkt; cprod[cid] = prod }
    cval[cid] += val
}

ENDFILE {
    for (k in bars) {
        split(k, p, SUBSEP)
        printf "A,%s,%s,%s,%s,%d,%d,%.0f,%.0f,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.0f,%s,%s,%d,%d,%d,%d,%d,%d,%.6f\n",
            cday, p[1], p[2], p[3],
            bars[k], ncon[k], svol[k], sval[k], minbt[k], maxbt[k],
            n0900[k]+0, n0900v[k]+0, n0900f[k]+0,
            n2100[k]+0, n2100v[k]+0, n2100f[k]+0,
            n0930[k]+0, n0930v[k]+0, n0930f[k]+0,
            npre[k]+0, vpre[k]+0,
            (k in minpret) ? minpret[k] : "", (k in maxpret) ? maxpret[k] : "",
            nnegval[k]+0, nnegvol[k]+0, nnegoi[k]+0, nzvol[k]+0,
            nvwapout[k]+0, nvwapout1[k]+0, maxdev[k]+0
    }

    delete bcnt; delete bwhole; delete bzero; delete bsum
    for (cid in cvalkey) {
        g = cmkt[cid] SUBSEP cprod[cid]
        bsum[g] += cval[cid]
        if (cval[cid] == 0) { bzero[g]++; continue }   # 全日无成交额的合约不参与整万元判定
        bcnt[g]++
        r = cval[cid] / 10000.0
        d = r - int(r); if (d < 0) d = -d
        if (d < 1e-6 || d > 1 - 1e-6) bwhole[g]++
    }
    for (g in bsum) {
        split(g, q, SUBSEP)
        printf "B,%s,%s,%s,%d,%d,%d,%.0f\n", cday, q[1], q[2], bcnt[g]+0, bwhole[g]+0, bzero[g]+0, bsum[g]
    }

    # 流 C：盘前带量 bar 极少（1~3 根）的合约 —— 正常夜盘有几十上百根，
    # 只有 padding 日才会只剩孤零零一根。
    for (cid in pc_n) {
        if (pc_n[cid] > 3) continue
        printf "C,%s,%s,%s,%s,%d,%s,%d,%.4f,%s,%.4f,%s,%.4f,%d\n",
            cday, cmkt[cid], cprod[cid], cid, pc_n[cid],
            pc_t[cid], pc_v[cid], pc_c[cid],
            (cid in zc_t) ? zc_t[cid] : "", (cid in zc_c) ? zc_c[cid] : 0,
            (cid in d0_t) ? d0_t[cid] : "", (cid in d0_o) ? d0_o[cid] : 0,
            (cid in d0_v) ? d0_v[cid] : 0
    }
}
