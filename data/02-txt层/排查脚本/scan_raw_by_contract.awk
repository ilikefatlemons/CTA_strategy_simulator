# scan_raw_by_contract.awk -- 逐 (结算日, 合约) 扫描 raw，用于与二级数据（主力连续 pkl）精确对账。
#
# 为什么要按 contract 而不是按品种：v3.1 的扫描是 (结算日, TDATE, 交易所, 品种) 粒度，
# 想比对二级数据就只能用「品种总 bar 数 / 合约数」估算每合约 bar 数。交割日各合约的
# bar 数不齐，这个估算会造出几十天的假差异 —— 数据端拿任一条就能主张整套方法不可靠。
# 这里直接落到 contract 粒度，与 pkl 的 contract 列一一对应，差异要么是真的要么没有。
#
# 顺带补齐 v3.1 漏掉的两类硬错误计数（v3.0 在二级层查过，raw 侧一直是空白）：
#   n_ohlc_nonpos  OHLC 任一 <= 0
#   n_vol0_val     volume == 0 但 value != 0   \ 两者合称"量额矛盾"
#   n_val0_vol     volume != 0 但 value == 0   /
#
# 表头判别与 scan_raw.awk 一致：第一行含 Offset -> 旧格式 TAB 14 列，否则新格式逗号 13 列。

BEGINFILE {
    delete bars; delete svol; delete minbt; delete maxbt
    delete nonpos; delete v0val; delete val0v
    delete mkt_of; delete prod_of

    cday = FILENAME
    sub(/^.*future_pricemin/, "", cday)
    sub(/\.txt$/, "", cday)
}

FNR == 1 { off = ($0 ~ /Offset/) ? 1 : 0; FS = (off == 1) ? "\t" : ","; next }

{
    cid = $3
    bt  = $(5  + off)
    cl  = $(6  + off) + 0
    op  = $(7  + off) + 0
    hi  = $(8  + off) + 0
    lo  = $(9  + off) + 0
    vol = $(10 + off) + 0
    val = $(11 + off) + 0

    if (!(cid in bars)) {
        prod = cid; sub(/[0-9]+$/, "", prod)
        mkt_of[cid] = $4; prod_of[cid] = prod
    }

    bars[cid]++
    svol[cid] += vol
    if (!(cid in minbt) || bt < minbt[cid]) minbt[cid] = bt
    if (!(cid in maxbt) || bt > maxbt[cid]) maxbt[cid] = bt

    if (op <= 0 || hi <= 0 || lo <= 0 || cl <= 0) nonpos[cid]++
    if (vol == 0 && val != 0) v0val[cid]++
    if (vol != 0 && val == 0) val0v[cid]++
}

ENDFILE {
    for (cid in bars)
        printf "%s,%s,%s,%s,%d,%.0f,%s,%s,%d,%d,%d\n",
            cday, mkt_of[cid], prod_of[cid], cid,
            bars[cid], svol[cid], minbt[cid], maxbt[cid],
            nonpos[cid]+0, v0val[cid]+0, val0v[cid]+0
}
