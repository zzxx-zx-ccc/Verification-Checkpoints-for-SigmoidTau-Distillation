# Code Mapping to the Method

This file explains how the manuscript notation maps to the released implementation.

## Sigmoid-τ classification distillation

Manuscript notation:

```text
sigma_tau(z) = sigmoid(z - log(tau)) = 1 / (1 + tau * exp(-z))
L_cls^kd = MBCE_{i in F_t}(sigma_tau(z_t,i), sigma_tau(z_s,i))
```

Implementation:

```text
ultralytics/utils/loss.py
  v8DetectionLoss.sigmoid_T()
  v8DetectionLoss._cls_distill_losses()
  v8DetectionLoss.D_loss_calculate()
```

The default classification KD path uses `hyp_cls_kd_mode: sigmoid_tau`, which computes teacher probabilities through `sigmoid_T(teacher_logits, tau)` and applies the same logit shift to the student logits.

## Teacher/student foreground partitions

Manuscript notation:

```text
R_t = F_t \ F_s
R_s = F_s \ F_t
R_o = F_t ∩ F_s
```

Implementation:

```text
ultralytics/utils/loss.py
  rt = fg_mask_teacher & (~fg_mask_student)
  rs = fg_mask_student & (~fg_mask_teacher)
  ro = fg_mask_teacher & fg_mask_student
```

`fg_mask_teacher` and `fg_mask_student` are produced by the YOLOv8 task-aligned assigner in `loss_calculate()` for the teacher and student branches, respectively.

## Region-specific localization/DFL supervision

Manuscript design:

| Region | Supervision target | Role |
| --- | --- | --- |
| `R_t` | teacher box and DFL distribution | missing-positive recovery |
| `R_s` | ground-truth box and DFL distribution | assignment stabilization |
| `R_o` | 0.5 teacher + 0.5 ground truth | knowledge-constraint fusion |

Implementation:

```text
ultralytics/utils/loss.py
  v8DetectionLoss.D_loss_calculate()
    # box distillation: lines using rt, rs, ro
    # DFL distillation: lines using rt, rs, ro
```

The shared region uses an equal mixture of teacher-guided and ground-truth-guided localization discrepancies. No additional inference module is added.

## Joint graph and export

| Component | File |
| --- | --- |
| Teacher branch | `configs/yolov8n_sigmoid_tau_*.yaml`, `Detect_Teacher` |
| Student branch | `configs/yolov8n_sigmoid_tau_*.yaml`, `Detect_Student` |
| Training-time joint head | `Detect_Distill` |
| Student export | `scripts/train_distill.py`, `exchange_student_back()` |
