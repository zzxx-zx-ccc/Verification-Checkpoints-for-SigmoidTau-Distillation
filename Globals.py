"""Default runtime options for SigmoidTau-Distillation.

This file provides safe defaults for direct imports and simple verification runs.
Training scripts may override these values at runtime.

The released verification checkpoints are intended for validating the public
training and evaluation pipeline. They should not be interpreted as the complete
paper-production checkpoint set.
"""

# ---------------------------------------------------------------------------
# Basic runtime settings
# ---------------------------------------------------------------------------

model_file = "configs/yolov8n_sigmoid_tau_coco.yaml"
dataset = "configs/coco.yaml"

epochs = 50
batch_size = 32
device = "0"

bool_save = True
save_period = 10
bool_resume = False

# ---------------------------------------------------------------------------
# Distillation switches
# ---------------------------------------------------------------------------

bool_distill = True

# 1: teacher branch loss
# 2: student branch loss
# 3: native supervised loss + distillation loss
flag_train_output = 3

# During validation/export, Detect_Distill uses the student branch output.
bool_val_output = True

# ---------------------------------------------------------------------------
# Teacher-student joint graph settings
# ---------------------------------------------------------------------------

scale_default = "n"
scale_student = "n"
scale_teacher = "s"

# Layer indices of teacher and student branches in the joint model yaml.
teacher_peer_list = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 45
]

student_peer_list = [
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46
]

# ---------------------------------------------------------------------------
# Sigmoid-τ and partition-aware distillation settings
# ---------------------------------------------------------------------------

# Verification default. Paper-level hyperparameter settings should follow the
# manuscript and experiment scripts.
hyp_T = 3.0

# Distillation branch strengths.
hyp_cls_distill = 1.0
hyp_box_distill = 0.1
hyp_dfl_distill = 0.05


# Overall KD strength.
hyp_kd = 0.1

# Teacher-response weighting and compatibility fields used by loss.py.
hyp_w_t_cls = 0.99
hyp_w_t_box = 0.50
hyp_w_t_dfl = 0.50

# Classification KD mode.
hyp_cls_kd_mode = "sigmoid_tau"
hyp_cls_kd_t2 = False

# Enable partition-aware localization distillation.
hyp_use_partition_kd = True

# Optional filtering and logging.
hyp_teacher_conf_gate = 0.0
hyp_kd_log_interval = 50

# ---------------------------------------------------------------------------
# Weight paths
# ---------------------------------------------------------------------------

bool_load_student = True

# Official Ultralytics initialization weights.
path_student = "weights/yolov8n.pt"
path_teacher = "weights/yolov8s.pt"

# Temporary branch-mapped weights generated during preparation.
path_student_prepare = "weights/prepare/yolov8_student_prepare.pth"
path_teacher_prepare = "weights/prepare/yolov8_teacher_prepare.pth"

# Student base weight for export or initialization.
model_based_file = "weights/yolov8n.pt"

# Default verification checkpoint path.
final = "checkpoints/final_students/YOLOv8n-SigmoidTau-COCO-best.pt"

# Phase control:
# 0: map teacher/student weights into the joint graph
# 1: train with supervised detection loss and KD losses
phase = 1
