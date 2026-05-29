#!/usr/bin/env bash
set -euo pipefail

python3 scripts/val.py \
  --weights checkpoints/final_students/YOLOv8n-SigmoidTau-VOC-best.pt \
  --data configs/voc.yaml \
  --imgsz 640 \
  --batch ${BATCH:-16} \
  --device ${DEVICE:-0}
