#!/bin/bash
# W4 剩余批处理驱动（串行占用单卡）：S13 补 conditional → S14 网格 → S15 五臂 → S18 M3
# 依据：版本3 §5 P2.1/P2.2/§7/§8 M3；执行清单 S13–S15、S18。
cd "E:/文献/近两三年的故障诊断文献/rbop_diagnosis_v4" || exit 1
PY="E:/tool/anaconda/envs/fault-detect/python.exe"
export PYTHONUTF8=1
LOG=docs/evidence/remaining_w4_driver.log
echo "[$(date)] 驱动启动" >> "$LOG"

run() {
  local name="$1"; shift
  echo "[$(date)] === $name ===" >> "$LOG"
  "$PY" -W ignore scripts/run_sweep.py "$@" >> "$LOG" 2>&1
  echo "[$(date)] $name 退出码=$?" >> "$LOG"
}

run "S13-conditional" --split_mode paired --label_pool_ratio 0.5 --seeds 0 1 2 3 \
  --protocols transductive --label_ratios 0.99 --controls semi_supervised --lams 0.3 \
  --reprs logspec --alignment_modes conditional --tag align \
  --source_epochs 30 --transfer_epochs 30

run "S14a-grid-lr03-06" --split_mode paired --label_pool_ratio 0.5 --seeds 0 1 2 3 \
  --protocols transductive --label_ratios 0.3 0.6 --controls semi_supervised --lams 0.3 \
  --reprs logspec --tag grid --source_epochs 30 --transfer_epochs 30

run "S14b-pool0.3-lr0.99" --split_mode paired --label_pool_ratio 0.3 --seeds 0 1 2 3 \
  --protocols transductive --label_ratios 0.99 --controls semi_supervised --lams 0.3 \
  --reprs logspec --tag gridpool03 --source_epochs 30 --transfer_epochs 30

run "S15-arms" --split_mode paired --label_pool_ratio 0.5 --seeds 0 1 2 3 \
  --protocols transductive --label_ratios 0.99 --controls semi_supervised --lams 0.3 \
  --reprs logspec --arms B C D0.1 D0.3 E --source_epochs 30 --transfer_epochs 30

run "S18-M3-seeds100-103" --split_mode paired --label_pool_ratio 0.5 --seeds 100 101 102 103 \
  --protocols transductive --label_ratios 0.99 --controls semi_supervised --lams 0.3 \
  --reprs logspec --tag m3 --source_epochs 30 --transfer_epochs 30

echo "[$(date)] ALL_REMAINING_DONE" >> "$LOG"
