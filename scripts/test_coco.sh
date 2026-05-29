#!/usr/bin/env bash
set -euo pipefail

python3 scripts/val.py \
  --weights checkpoints/final_students/YOLOv8n-SigmoidTau-COCO-best.pt \
  --data configs/coco.yaml \
  --imgsz 640 \
  --batch ${BATCH:-16} \
  --device ${DEVICE:-0}
