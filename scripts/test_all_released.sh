#!/usr/bin/env bash
set -euo pipefail

BATCH=${BATCH:-16}
DEVICE=${DEVICE:-0}

python3 scripts/val.py --weights checkpoints/teachers/YOLOv8s-COCO-teacher.pt --data configs/coco.yaml --imgsz 640 --batch "$BATCH" --device "$DEVICE"
python3 scripts/val.py --weights checkpoints/baselines/YOLOv8n-COCO-standard-best.pt --data configs/coco.yaml --imgsz 640 --batch "$BATCH" --device "$DEVICE"
python3 scripts/val.py --weights checkpoints/final_students/YOLOv8n-SigmoidTau-COCO-best.pt --data configs/coco.yaml --imgsz 640 --batch "$BATCH" --device "$DEVICE"

python3 scripts/val.py --weights checkpoints/teachers/YOLOv8s-VOC-teacher.pt --data configs/voc.yaml --imgsz 640 --batch "$BATCH" --device "$DEVICE"
python3 scripts/val.py --weights checkpoints/baselines/YOLOv8n-VOC-standard-best.pt --data configs/voc.yaml --imgsz 640 --batch "$BATCH" --device "$DEVICE"
python3 scripts/val.py --weights checkpoints/final_students/YOLOv8n-SigmoidTau-VOC-best.pt --data configs/voc.yaml --imgsz 640 --batch "$BATCH" --device "$DEVICE"

python3 scripts/val.py --weights checkpoints/teachers/YOLOv8s-VisDrone-teacher.pt --data configs/visdrone.yaml --imgsz 960 --batch "$BATCH" --device "$DEVICE"
python3 scripts/val.py --weights checkpoints/baselines/YOLOv8n-VisDrone-standard-best.pt --data configs/visdrone.yaml --imgsz 960 --batch "$BATCH" --device "$DEVICE"
python3 scripts/val.py --weights checkpoints/final_students/YOLOv8n-SigmoidTau-VisDrone-best.pt --data configs/visdrone.yaml --imgsz 960 --batch "$BATCH" --device "$DEVICE"
