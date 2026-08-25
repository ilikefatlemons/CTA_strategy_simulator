#!/usr/bin/env bash
# 全历史扫描 data/02-txt层/原始数据/*.txt -> data/02-txt层/排查证据/raw_scan_daily.csv
# 用法：bash data/02-txt层/排查脚本/run_scan.sh [并行度]   （默认 8）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RAW="$ROOT/data/02-txt层/原始数据"
OUT="$ROOT/data/02-txt层/排查证据"
TMP="$OUT/_parts"
NJOB="${1:-8}"

mkdir -p "$TMP"
rm -f "$TMP"/part_*.csv

# 只取 data/02-txt层/原始数据 顶层的 future_pricemin*.txt。
# raw/weekend/ 子目录下另有两个 2013 年的同名文件，不属于日度序列，单独处理，此处排除。
find "$RAW" -maxdepth 1 -name 'future_pricemin*.txt' | sort > "$TMP/filelist.txt"
NFILE=$(wc -l < "$TMP/filelist.txt")
echo "files: $NFILE   jobs: $NJOB"

# 按行轮转切成 NJOB 份，让每份的文件大小分布均匀（早年文件小、近年文件大）
awk -v n="$NJOB" -v d="$TMP" '{print > (d "/chunk_" (NR % n) ".txt")}' "$TMP/filelist.txt"

pids=()
for i in $(seq 0 $((NJOB - 1))); do
    [ -s "$TMP/chunk_$i.txt" ] || continue
    ( gawk -f "$ROOT/data/02-txt层/排查脚本/scan_raw.awk" $(cat "$TMP/chunk_$i.txt") > "$TMP/part_$i.csv" ) &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

{
    echo "stream,clearing_day,tdate,market,product,bars,n_contracts,sum_vol,sum_value,min_bt,max_bt,n0900,n0900v,n0900flat,n2100,n2100v,n2100flat,n0930,n0930v,n0930flat,n_pre_volbar,vol_pre,min_pre_voltime,max_pre_voltime,n_neg_value,n_neg_vol,n_neg_oi,n_zero_vol,n_vwap_out,n_vwap_out_1pct,max_vwap_dev"
    cat "$TMP"/part_*.csv | grep '^A,'
} > "$OUT/raw_scan_daily.csv"

{
    echo "stream,clearing_day,market,product,n_contracts,n_value_whole_10k,n_contracts_zero_value,sum_value"
    cat "$TMP"/part_*.csv | grep '^B,'
} > "$OUT/raw_scan_contract_value.csv"

{
    echo "stream,clearing_day,market,product,contract,n_pre_volbar,pre_bt,pre_vol,pre_close,pad_bt,pad_close,day0_bt,day0_open,day0_vol"
    cat "$TMP"/part_*.csv | grep '^C,'
} > "$OUT/raw_scan_prebar.csv"

echo "A rows: $(($(wc -l < "$OUT/raw_scan_daily.csv") - 1))"
echo "B rows: $(($(wc -l < "$OUT/raw_scan_contract_value.csv") - 1))"
echo "C rows: $(($(wc -l < "$OUT/raw_scan_prebar.csv") - 1))"
echo "done -> $OUT/raw_scan_daily.csv"
echo "     -> $OUT/raw_scan_contract_value.csv"
echo "     -> $OUT/raw_scan_prebar.csv"
