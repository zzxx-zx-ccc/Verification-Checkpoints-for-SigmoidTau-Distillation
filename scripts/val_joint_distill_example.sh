#!/usr/bin/env bash
set -euo pipefail

# Use this only for a teacher-student joint distillation checkpoint.
# Final exported student checkpoints should be evaluated with scripts/test_*.sh.
python3 scripts/val.py \
  --model configs/yolov8n_sigmoid_tau_coco.yaml \
  --weights weights/joint_distill_checkpoint.pt \
  --data configs/coco.yaml \
  --imgsz 640 \
  --batch ${BATCH:-16} \
  --device ${DEVICE:-0} \
  --distill
