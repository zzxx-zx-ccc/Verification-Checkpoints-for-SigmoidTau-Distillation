#!/usr/bin/env bash
set -euo pipefail

python3 scripts/train_distill.py \
  --phase 0 \
  --data configs/voc.yaml \
  --model configs/yolov8n_sigmoid_tau_voc.yaml \
  --hyp configs/hyp_sigmoid_tau.yaml \
  --teacher ${TEACHER:-weights/yolov8s.pt} \
  --student ${STUDENT:-weights/yolov8n.pt} \
  --model-based ${MODEL_BASED:-weights/yolov8n.pt}

python3 scripts/train_distill.py \
  --phase 1 \
  --data configs/voc.yaml \
  --model configs/yolov8n_sigmoid_tau_voc.yaml \
  --hyp configs/hyp_sigmoid_tau.yaml \
  --teacher ${TEACHER:-weights/yolov8s.pt} \
  --student ${STUDENT:-weights/yolov8n.pt} \
  --model-based ${MODEL_BASED:-weights/yolov8n.pt} \
  --epochs ${EPOCHS:-50} \
  --imgsz 640 \
  --batch ${BATCH:-32} \
  --device ${DEVICE:-0} \
  --name sigmoid_tau_voc_tau3_50e \
  --final checkpoints/final_students/YOLOv8n-SigmoidTau-VOC-best.pt
