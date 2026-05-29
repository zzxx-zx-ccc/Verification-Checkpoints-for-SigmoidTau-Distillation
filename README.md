# Protocol-Consistent Sigmoid-τ Logit Distillation for Lightweight Object Detection

The implementation is built on Ultralytics YOLOv8. The exported student detector remains a standard YOLOv8n model, and the proposed distillation components are used only during training.

* **Sigmoid-τ classification distillation:** applies class-wise logit modulation under the Sigmoid/BCE detection protocol. The modulation is implemented as `sigma_tau(z) = sigmoid(z - log(tau))`.

* **Partition-aware localization distillation:** independently generates teacher and student foreground assignments using the native YOLOv8 task-aligned assigner. The foreground positions are divided into the teacher-only positive region `R_t`, the student-only positive region `R_s`, and the shared positive region `R_o`. These regions are used for region-specific box and DFL supervision.

No additional inference module is added to the final student model. Therefore, after exporting the student branch, the parameter count, FLOPs, and inference behavior remain the same as those of the corresponding YOLOv8n detector.

## Release scope

This repository is intended to provide a clean and reproducible implementation of the proposed distillation pipeline. The checkpoints listed in this release are **verification checkpoints** for code validation and transparency. They are not claimed to be the complete set of paper-production checkpoints.

The manuscript reports full experimental results on COCO2017, PASCAL VOC, VisDrone2019, challenging COCO subsets, and a YOLOv9 cross-architecture verification. By contrast, the checkpoint table included in this repository records the metrics of the public verification weights that will be uploaded separately to GitHub Releases or another persistent storage service.

## Important note about checkpoints

Large `.pt` files are not stored in this zip package or in the Git repository. Upload them separately, for example to GitHub Releases, Zenodo, Hugging Face, or an institutional download page. After upload, add the public links and SHA256 checksums in:

```text
docs/model_zoo.md
checkpoints/checksums.sha256
```

Expected checkpoint layout:

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

The validation metrics of these candidate/released checkpoints are recorded in:

```text
results/released_checkpoint_metrics.csv
docs/released_checkpoint_metrics.md
```

## Repository structure

```text
SigmoidTau-Distillation/
├── configs/
│   ├── coco.yaml
│   ├── voc.yaml
│   ├── visdrone.yaml
│   ├── hyp_sigmoid_tau.yaml
│   ├── yolov8n_sigmoid_tau_coco.yaml
│   ├── yolov8n_sigmoid_tau_voc.yaml
│   └── yolov8n_sigmoid_tau_visdrone.yaml
├── checkpoints/
│   ├── teachers/
│   ├── baselines/
│   └── final_students/
├── docs/
│   ├── code_mapping.md
│   ├── dataset_preparation.md
│   ├── model_zoo.md
│   ├── release_scope.md
│   └── released_checkpoint_metrics.md
├── results/
│   └── released_checkpoint_metrics.csv
├── scripts/
│   ├── train_distill.py
│   ├── train_coco_tau3_50e.sh
│   ├── train_voc_tau3_50e.sh
│   ├── train_visdrone_tau3_50e.sh
│   ├── val.py
│   ├── test_coco.sh
│   ├── test_voc.sh
│   ├── test_visdrone.sh
│   └── test_all_released.sh
├── ultralytics/
├── Globals.py
├── requirements.txt
└── LICENSE
```

## Environment

```bash
git clone https://github.com/your-lab/SigmoidTau-Distillation.git
cd SigmoidTau-Distillation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code requires a CUDA-enabled PyTorch environment compatible with the bundled Ultralytics YOLOv8 version. Use the same PyTorch/CUDA version for all baseline and distilled runs when comparing results.

## Dataset preparation

Download the datasets and convert them to YOLO detection format if needed. Then edit the `path:` field in each yaml file.

| Dataset | Local yaml | Input size used in this release |
| --- | --- | ---: |
| COCO2017 | `configs/coco.yaml` | 640 |
| PASCAL VOC | `configs/voc.yaml` | 640 |
| VisDrone2019 | `configs/visdrone.yaml` | 960 |

See `docs/dataset_preparation.md` for dataset layout examples.

## Official initialization weights

Download the official Ultralytics YOLOv8 checkpoints and place them under `weights/`:

```text
weights/yolov8n.pt
weights/yolov8s.pt
```

`yolov8n.pt` is used as the student initialization and export base. `yolov8s.pt` is used as the teacher initialization unless dataset-specific teacher checkpoints are used.

## Training quick-verification students

The public quick-verification scripts use a unified teacher-student pipeline and a 50-epoch schedule. The default public verification temperature is `tau=3.0`; adjust `configs/hyp_sigmoid_tau.yaml` if reproducing the manuscript's full-temperature ablation or full paper setting.

```bash
bash scripts/train_coco_tau3_50e.sh
bash scripts/train_voc_tau3_50e.sh
bash scripts/train_visdrone_tau3_50e.sh
```

Override device and batch size with environment variables:

```bash
DEVICE=0 BATCH=16 bash scripts/train_coco_tau3_50e.sh
```

## Evaluation

After downloading released weights into `checkpoints/`, run:

```bash
bash scripts/test_coco.sh
bash scripts/test_voc.sh
bash scripts/test_visdrone.sh
```

or validate all released/candidate checkpoints:

```bash
bash scripts/test_all_released.sh
```

For a CPU sanity check:

```bash
DEVICE=cpu BATCH=1 bash scripts/test_coco.sh
```

## Core implementation files

| File | Role |
| --- | --- |
| `ultralytics/utils/loss.py` | Adds Sigmoid-τ classification KD and partition-aware box/DFL KD on top of the native YOLOv8 loss. |
| `ultralytics/nn/modules/head.py` | Adds `Detect_Teacher`, `Detect_Student`, and `Detect_Distill` heads for joint teacher-student training and student-only validation/export. |
| `ultralytics/nn/tasks.py` | Parses the joint teacher-student model graph and routes teacher/student scales. |
| `configs/yolov8n_sigmoid_tau_*.yaml` | Defines the joint YOLOv8s teacher + YOLOv8n student graph. |
| `scripts/train_distill.py` | Maps teacher/student weights into the joint graph, trains with supervised + KD loss, and exports the student branch. |

See `docs/code_mapping.md` for the correspondence between the paper notation and implementation.

## License

This repository is based on Ultralytics YOLO and keeps the corresponding license terms. See `LICENSE` for details.

## Citation

Please update the bibliographic information after the article is formally published.

```bibtex

@article{shi2026sigmoidtau,
  title   = {Protocol-Consistent Sigmoid-τ Logit Distillation for Lightweight Object Detection},
  author  = {Shi, Xiangqun and Zhang, Xun and Zhang, Xian and Su, Yifan},
  journal = {Image and Vision Computing},
  year    = {2026},
  note    = {In press}
}
```
