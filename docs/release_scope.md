# Release Scope and Reproducibility Statement

This repository releases the code for the proposed Sigmoid-τ distillation pipeline and a transparent table of verification checkpoint metrics.

## What this release supports

- Training a YOLOv8n student under a frozen YOLOv8s teacher.
- Applying Sigmoid-τ classification distillation on teacher-positive positions.
- Applying partition-aware box/DFL localization distillation on teacher-only, student-only, and shared foreground positions.
- Exporting the trained student branch as a standard YOLOv8n-style checkpoint.
- Validating teacher, baseline, and distilled-student checkpoints on COCO2017, PASCAL VOC, and VisDrone2019.

## What this release does not claim

The checkpoint metrics in `results/released_checkpoint_metrics.csv` are for public code verification and transparency. They should not be described as the complete final journal results unless the corresponding full-training checkpoints are also released.

The manuscript's full experimental results were obtained under the paper's controlled experimental setting, including full training schedules, repeated COCO runs, hard-subset analysis, and cross-architecture verification. This code package is intended to make the method reproducible and inspectable, while keeping large model files separate from the source repository.

## Suggested wording for GitHub

> This repository contains the implementation of Protocol-Consistent Sigmoid-τ Logit Distillation. The released checkpoints are provided for code verification and transparent validation. They are not the complete set of paper-production checkpoints. The final paper results are reported in the manuscript; the metrics listed here correspond only to the uploaded public weights.
