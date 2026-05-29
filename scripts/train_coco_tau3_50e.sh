#!/usr/bin/env bash
set -euo pipefail

# Phase 0 maps YOLOv8n/YOLOv8s weights into the joint teacher-student graph.
python3 scripts/train_distill.py \
  --phase 0 \
  --data configs/coco.yaml \
  --model configs/yolov8n_sigmoid_tau_coco.yaml \
  --hyp configs/hyp_sigmoid_tau.yaml \
  --teacher ${TEACHER:-weights/yolov8s.pt} \
  --student ${STUDENT:-weights/yolov8n.pt} \
  --model-based ${MODEL_BASED:-weights/yolov8n.pt}

python3 scripts/train_distill.py \
  --phase 1 \
  --data configs/coco.yaml \
  --model configs/yolov8n_sigmoid_tau_coco.yaml \
  --hyp configs/hyp_sigmoid_tau.yaml \
  --teacher ${TEACHER:-weights/yolov8s.pt} \
  --student ${STUDENT:-weights/yolov8n.pt} \
  --model-based ${MODEL_BASED:-weights/yolov8n.pt} \
  --epochs ${EPOCHS:-50} \
  --imgsz 640 \
  --batch ${BATCH:-32} \
  --device ${DEVICE:-0} \
  --name sigmoid_tau_coco_tau3_50e \
  --final checkpoints/final_students/YOLOv8n-SigmoidTau-COCO-best.pt
