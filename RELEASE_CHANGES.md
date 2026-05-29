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
results/README.md
# Results

This folder stores released validation summaries and metric tables used for checkpoint verification.

The main released metric file is:

```text
results/released_checkpoint_metrics.csv