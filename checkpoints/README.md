### `checkpoints/README.md`

````markdown
# Checkpoints

Large `.pt` checkpoint files are not tracked in the source-code repository.

Verification checkpoints are released separately through GitHub Releases:

```text
https://github.com/zzxx-zx-ccc/Verification-Checkpoints-for-SigmoidTau-Distillation/releases/tag/v1.0.0-verification
````

These checkpoints are provided for validating the released training and evaluation pipeline. They are not intended to replace the complete paper-production checkpoints used for all manuscript tables.

## Expected folder layout

After downloading the release assets, place the checkpoint files as follows:

```text
checkpoints/
├── teachers/
│   ├── YOLOv8s-COCO-teacher.pt
│   ├── YOLOv8s-VOC-teacher.pt
│   └── YOLOv8s-VisDrone-teacher.pt
├── baselines/
│   ├── YOLOv8n-COCO-standard-best.pt
│   ├── YOLOv8n-VOC-standard-best.pt
│   └── YOLOv8n-VisDrone-standard-best.pt
└── final_students/
    ├── YOLOv8n-SigmoidTau-COCO-best.pt
    ├── YOLOv8n-SigmoidTau-VOC-best.pt
    └── YOLOv8n-SigmoidTau-VisDrone-best.pt
```

## Verification scope

The released checkpoint set is designed to verify:

* the YOLOv8s-to-YOLOv8n teacher-student distillation pipeline;
* Sigmoid-τ classification distillation;
* task-aligned foreground assignment and partition-aware localization distillation;
* validation on COCO, PASCAL VOC, and VisDrone2019.

The dataset paths in the yaml files should be modified according to the user's local dataset location before evaluation.

## Integrity check

SHA256 checksums are recorded in:

```text
checkpoints/checksums.sha256
```

Users may verify downloaded checkpoint files with:

```powershell
Get-FileHash .\checkpoints\teachers\YOLOv8s-COCO-teacher.pt -Algorithm SHA256
```

````

### `weights/README.md`

```markdown
# Weights

This folder is used for official YOLOv8 initialization weights and temporary converted weights generated during training preparation.

## Official initialization weights

Download the official Ultralytics YOLOv8 detection checkpoints and place them here when training from official initialization weights:

```text
weights/yolov8n.pt
weights/yolov8s.pt
````

* `yolov8n.pt` is used as the YOLOv8n student initialization.
* `yolov8s.pt` is used as the YOLOv8s teacher initialization when a dataset-specific teacher checkpoint is not provided.

These files are not included in the repository.

## Temporary preparation files

During the weight-mapping stage, the training script may generate temporary branch-mapped files:

```text
weights/prepare/yolov8_student_prepare.pth
weights/prepare/yolov8_teacher_prepare.pth
```

These files are generated artifacts and should not be committed to Git.

## Verification checkpoints

Dataset-specific verification checkpoints are stored under:

```text
checkpoints/teachers/
checkpoints/baselines/
checkpoints/final_students/
```

They are released separately through GitHub Releases rather than tracked in the source repository.

See the following files for checkpoint names, metrics, and usage instructions:

```text
checkpoints/README.md
docs/model_zoo.md
docs/released_checkpoint_metrics.md
results/released_checkpoint_metrics.csv
```

````

### `Globals.py`

```python
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
````

### `LICENSE_AND_ATTRIBUTION.md` 或 `NOTICE.md`

```markdown
# License and Attribution Notice

This repository provides research code for:

**Protocol-Consistent Sigmoid-τ Logit Distillation for Lightweight Object Detection**

The implementation is built on the Ultralytics YOLO codebase. Files derived from Ultralytics YOLO remain subject to the original Ultralytics license terms. Users should retain the original copyright and license notices when redistributing or modifying those files.

The authors' added research components, configuration files, scripts, and documentation are released for academic research and reproducibility, subject to compatibility with the license terms of the underlying Ultralytics codebase.

The released verification checkpoints are provided for research validation only. Users are responsible for ensuring that their use of this repository, datasets, pretrained models, and checkpoints complies with the corresponding licenses and terms of use.
```

### `RELEASE_CHANGES.md`

```markdown
# Release Changes

This package is aligned with the revised manuscript:

**Protocol-Consistent Sigmoid-τ Logit Distillation for Lightweight Object Detection**

## Main updates

- Organized the released code around protocol-consistent Sigmoid-τ logit distillation.
- Added documentation to clarify the relationship between the code implementation and the manuscript method.
- Separated source code from large checkpoint files.
- Documented verification checkpoints for COCO, PASCAL VOC, and VisDrone2019.
- Corrected checkpoint names and paths for teacher, baseline, and Sigmoid-τ student models.
- Replaced local machine dataset paths with repository yaml paths:
  - `configs/coco.yaml`
  - `configs/voc.yaml`
  - `configs/visdrone.yaml`
- Added released validation metrics and checksum records for verification.

## Included

- YOLOv8-based teacher-student distillation code.
- Sigmoid-τ classification distillation implementation.
- Partition-aware localization distillation implementation.
- Dataset configuration templates.
- Training and validation scripts.
- Documentation for checkpoint usage and released metrics.

## Not included

- Large `.pt` checkpoint files in Git history.
- Private training logs, local cache files, or temporary run directories.
- Complete paper-production checkpoints unless explicitly released later.

The released checkpoint set is intended for code verification and transparent validation, not as a replacement for the complete experimental records reported in the manuscript.
```

### `results/README.md`

````markdown
# Results

This folder stores released validation summaries and metric tables used for checkpoint verification.

The main released metric file is:

```text
results/released_checkpoint_metrics.csv
````

The metrics are provided to help users verify that the released checkpoints and evaluation scripts are working correctly.

These released checkpoint metrics are for code verification. They should not be confused with the complete paper-production results reported in the manuscript.

````

### `weights/prepare/README.md`

```markdown
# Prepared Weights

This folder is used for temporary branch-mapped weights generated during model preparation.

Typical generated files include:

```text
yolov8_student_prepare.pth
yolov8_teacher_prepare.pth
````

These files are intermediate artifacts and should not be committed to Git.

```
```
