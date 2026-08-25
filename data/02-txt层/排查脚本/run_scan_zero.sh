#!/usr/bin/env bash
# 按块扫描零价 K 线 -> data/02-txt层/排查证据/zero_price_blocks.csv + zero_price_daily.csv
# 用法：bash data/02-txt层/排查脚本/run_scan_zero.sh [并行度]   （默认 12）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$ROOT/data/02-txt层/排查证据"
TMP="$OUT/_zparts"
NJOB="${1:-12}"

mkdir -p "$TMP"
rm -f "$TMP"/part_*.csv "$TMP"/chunk_*.txt

find "$ROOT/data/02-txt层/原始数据" -maxdepth 1 -name 'future_pricemin*.txt' | sort > "$TMP/filelist.txt"
echo "files: $(wc -l < "$TMP/filelist.txt")   jobs: $NJOB"
awk -v n="$NJOB" -v d="$TMP" '{print > (d "/chunk_" (NR % n) ".txt")}' "$TMP/filelist.txt"

pids=()
for i in $(seq 0 $((NJOB - 1))); do
    [ -s "$TMP/chunk_$i.txt" ] || continue
    ( gawk -f "$ROOT/data/02-txt层/排查脚本/scan_zero_price.awk" $(cat "$TMP/chunk_$i.txt") > "$TMP/part_$i.csv" ) &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

{
    echo "stream,clearing_day,market,product,contract,start_tdate,start_bt,end_tdate,end_bt,n_bars,n_bars_with_vol,sum_vol,seed_vol,seed_value,seed_vwap,prev_close,next_open,at_file_start,at_file_end,seed_mask"
    cat "$TMP"/part_*.csv | grep '^B,' | sort -t, -k2,2 -k5,5
} > "$OUT/zero_price_blocks.csv"

{
    echo "stream,clearing_day,market,product,contract,tdate,bartime,mask,open,high,low,close,volume,value,vwap"
    cat "$TMP"/part_*.csv | grep '^P,' | sort -t, -k2,2 -k5,5
} > "$OUT/zero_price_isolated.csv"

{
    echo "stream,clearing_day,n_bars_any_zero,n_bars_close_zero,n_blocks,n_isolated"
    cat "$TMP"/part_*.csv | grep '^S,' | sort -t, -k2,2
} > "$OUT/zero_price_daily.csv"

rm -rf "$TMP"
echo "blocks   (形态A): $(($(wc -l < "$OUT/zero_price_blocks.csv") - 1))"
echo "isolated (形态B): $(($(wc -l < "$OUT/zero_price_isolated.csv") - 1))"
echo "days:            $(($(wc -l < "$OUT/zero_price_daily.csv") - 1))"
