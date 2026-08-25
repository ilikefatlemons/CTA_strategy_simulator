# scan_zero_price.awk -- 零价 K 线（OHLC 出现 0）的分型扫描。
#
# 实测有两种形态，性质完全不同，必须分开数：
#
#   形态 A「四价全 0」—— close 也是 0，于是「无成交沿用上一价」的逻辑把 0 一路复制
#       下去，直到下一根 close 正常的 bar 才复位。一次故障污染几十到几百根 K 线。
#       起点（种子）是一根真实成交的 bar：volume / value / vwap 都对，只有 OHLC 是 0。
#
#   形态 B「只有 open 和 low 为 0，high / close 正常」—— close 有效，不会传染，
#       是孤立的单根缺陷。
#
# 判定传染与否的关键是 close：close <= 0 才会被复制下去。所以块的定义用 close <= 0，
# 而不是"四价中任一 <= 0"。
#
# 依赖一个已验证的前提：raw 文件里同一合约的行严格连续成块且按时间升序
#   （新旧格式各抽查 0 个非连续块），因此可以顺序检测块边界。
#
# 输出三条流：
#   B 行 —— 形态 A 的块：起止、长度、块内成交 bar 数、种子的量额、块前/块后有效价
#   P 行 —— 形态 B 的孤立坏根：逐根，带 mask 标明哪几个字段为 0
#   S 行 —— 逐文件汇总，用于交叉核对
#
# mask 位：open=1, high=2, low=4, close=8（例：O 与 L 为 0 -> mask=5，四价全 0 -> 15）

BEGINFILE {
    cday = FILENAME
    sub(/^.*future_pricemin/, "", cday)
    sub(/\.txt$/, "", cday)
    cur = ""; inblk = 0; nblk = 0; nA = 0; nP = 0; nany = 0
}

FNR == 1 { off = ($0 ~ /Offset/) ? 1 : 0; FS = (off == 1) ? "\t" : ","; next }

function emit(next_open, at_end,   prod) {
    prod = cur; sub(/[0-9]+$/, "", prod)
    printf "B,%s,%s,%s,%s,%s,%s,%s,%s,%d,%d,%.0f,%.0f,%.0f,%s,%s,%s,%d,%d,%d\n",
        cday, mkt, prod, cur,
        b_st, b_sbt, b_et, b_ebt,
        b_n, b_nvol, b_vol, b_svol, b_sval, b_svwap,
        (b_prev == "" ? "" : b_prev), (next_open == "" ? "" : next_open),
        b_atstart, at_end, b_smask
    nblk++
}

{
    cid = $3
    if (cid != cur) {
        if (inblk) emit("", 1)
        cur = cid; mkt = $4; inblk = 0; lastc = ""; idx = 0
    }
    idx++

    tdate = $2
    bt  = $(5  + off)
    cl  = $(6  + off) + 0
    op  = $(7  + off) + 0
    hi  = $(8  + off) + 0
    lo  = $(9  + off) + 0
    vol = $(10 + off) + 0
    val = $(11 + off) + 0
    vwp = $(12 + off) + 0

    mask = (op <= 0 ? 1 : 0) + (hi <= 0 ? 2 : 0) + (lo <= 0 ? 4 : 0) + (cl <= 0 ? 8 : 0)
    if (mask) nany++

    if (cl <= 0) {                      # 形态 A：close 为 0 -> 会传染，按块处理
        nA++
        if (!inblk) {
            inblk = 1
            b_st = tdate; b_sbt = bt; b_n = 0; b_nvol = 0; b_vol = 0
            b_svol = vol; b_sval = val; b_svwap = vwp
            b_prev = lastc; b_atstart = (idx == 1); b_smask = mask
        }
        b_n++; b_et = tdate; b_ebt = bt
        if (vol > 0) { b_nvol++; b_vol += vol }
    } else {
        if (inblk) { emit(op, 0); inblk = 0 }
        lastc = cl
        if (mask) {                     # 形态 B：close 有效，孤立坏根，不传染
            nP++
            prod = cur; sub(/[0-9]+$/, "", prod)
            printf "P,%s,%s,%s,%s,%s,%s,%d,%s,%s,%s,%s,%.0f,%.0f,%s\n",
                cday, mkt, prod, cur, tdate, bt, mask,
                op, hi, lo, cl, vol, val, vwp
        }
    }
}

ENDFILE {
    if (inblk) emit("", 1)
    printf "S,%s,%d,%d,%d,%d\n", cday, nany, nA, nblk, nP
}
