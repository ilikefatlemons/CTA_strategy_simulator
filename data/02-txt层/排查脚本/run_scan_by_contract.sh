#!/usr/bin/env bash
# 逐合约扫描 data/02-txt层/原始数据 -> data/02-txt层/排查证据/raw_scan_by_contract.csv
# 用法：bash data/02-txt层/排查脚本/run_scan_by_contract.sh [并行度]   （默认 12）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RAW="$ROOT/data/02-txt层/原始数据"
OUT="$ROOT/data/02-txt层/排查证据"
TMP="$OUT/_parts"
NJOB="${1:-12}"

mkdir -p "$TMP"
rm -f "$TMP"/part_*.csv "$TMP"/chunk_*.txt

find "$RAW" -maxdepth 1 -name 'future_pricemin*.txt' | sort > "$TMP/filelist.txt"
echo "files: $(wc -l < "$TMP/filelist.txt")   jobs: $NJOB"

# 轮转切分，让早年小文件与近年大文件均匀分布到各进程
awk -v n="$NJOB" -v d="$TMP" '{print > (d "/chunk_" (NR % n) ".txt")}' "$TMP/filelist.txt"

pids=()
for i in $(seq 0 $((NJOB - 1))); do
    [ -s "$TMP/chunk_$i.txt" ] || continue
    ( gawk -f "$ROOT/data/02-txt层/排查脚本/scan_raw_by_contract.awk" $(cat "$TMP/chunk_$i.txt") > "$TMP/part_$i.csv" ) &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

{
    echo "clearing_day,market,product,contract,bars,sum_vol,min_bt,max_bt,n_ohlc_nonpos,n_vol0_val,n_val0_vol"
    cat "$TMP"/part_*.csv
} > "$OUT/raw_scan_by_contract.csv"

rm -rf "$TMP"
echo "rows: $(($(wc -l < "$OUT/raw_scan_by_contract.csv") - 1))"
echo "done -> $OUT/raw_scan_by_contract.csv"
