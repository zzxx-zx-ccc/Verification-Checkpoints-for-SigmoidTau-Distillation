"""Fallback runtime variables for SigmoidTau-Distillation.

This file is kept for compatibility with scripts or modules that import
`Globals.py` directly. Normal experiments should be configured through the
entry scripts and yaml files under `scripts/` and `configs/`.

The values below are lightweight default values for code execution and
verification, not a complete description of all paper experiments.
"""

# Basic runtime defaults
model_file = "configs/yolov8n_sigmoid_tau_coco.yaml"
dataset = "configs/coco.yaml"
epochs = 50
batch_size = 32
device = "0"

bool_save = True
save_period = 10
bool_resume = False

# Distillation switches
bool_distill = True
flag_train_output = 3
bool_val_output = True

# Teacher-student joint-graph mapping
scale_default = "n"
scale_student = "n"
scale_teacher = "s"

teacher_peer_list = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 45
]

student_peer_list = [
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46
]


hyp_cls_kd_mode = "sigmoid_tau"
hyp_cls_kd_t2 = False
hyp_use_partition_kd = True

hyp_teacher_conf_gate = 0.0
hyp_kd_log_interval = 50

# Weight paths
bool_load_student = True
path_student = "weights/yolov8n.pt"
path_teacher = "weights/yolov8s.pt"

path_student_prepare = "weights/prepare/yolov8_student_prepare.pth"
path_teacher_prepare = "weights/prepare/yolov8_teacher_prepare.pth"

model_based_file = "weights/yolov8n.pt"
final = "checkpoints/final_students/YOLOv8n-SigmoidTau-COCO-best.pt"

# 0: prepare mapped weights; 1: train with KD
phase = 1
