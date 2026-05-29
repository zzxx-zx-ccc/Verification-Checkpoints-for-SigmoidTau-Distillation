#!/usr/bin/env bash
set -euo pipefail

python3 scripts/val.py \
  --weights checkpoints/final_students/YOLOv8n-SigmoidTau-VisDrone-best.pt \
  --data configs/visdrone.yaml \
  --imgsz 960 \
  --batch ${BATCH:-16} \
  --device ${DEVICE:-0}
