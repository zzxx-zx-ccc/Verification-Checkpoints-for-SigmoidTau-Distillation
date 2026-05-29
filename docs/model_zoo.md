# Model Zoo

This page records the checkpoints associated with the public Sigmoid-τ distillation release.

## Important note

The released checkpoints are provided for code verification, validation transparency, and follow-up research. Some checkpoints are trained under a quick-verification setting and may not exactly match the final numbers reported in the manuscript because full paper experiments may use different schedules, repeated runs, and controlled environments.

Do not describe the uploaded verification checkpoints as the full journal result set unless the full-training paper checkpoints are also released.

## Recommended hosting

Do not store large `.pt` files directly in the Git repository. Recommended options are:

- GitHub Releases attached to this repository;
- Zenodo, preferably with a DOI;
- Hugging Face model repository;
- institutional or laboratory public download page.

After uploading, replace `Add public link` below with final download URLs and update `checkpoints/checksums.sha256`.

## Teacher checkpoints

| Checkpoint | Dataset | Input | Params | Validation mAP50 | Validation mAP50:95 | Link |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `YOLOv8s-COCO-teacher.pt` | COCO2017 | 640 | 11156544 | 0.6129 | 0.4474 | Add public link |
| `YOLOv8s-VOC-teacher.pt` | PASCAL VOC | 640 | 11133324 | 0.8624 | 0.6744 | Add public link |
| `YOLOv8s-VisDrone-teacher.pt` | VisDrone2019 | 960 | 11129454 | 0.4851 | 0.2998 | Add public link |

Place under:

```text
checkpoints/teachers/
```

## Baseline student checkpoints

| Checkpoint | Dataset | Input | Params | Validation mAP50 | Validation mAP50:95 | Link |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `YOLOv8n-COCO-standard-best.pt` | COCO2017 | 640 | 3151904 | 0.4949 | 0.3493 | Add public link |
| `YOLOv8n-VOC-standard-best.pt` | PASCAL VOC | 640 | 3009548 | 0.8192 | 0.6073 | Add public link |
| `YOLOv8n-VisDrone-standard-best.pt` | VisDrone2019 | 960 | 3007598 | 0.3991 | 0.2419 | Add public link |

Place under:

```text
checkpoints/baselines/
```

## Sigmoid-τ distilled student checkpoints

| Checkpoint | Dataset | Input | Params | Validation mAP50 | Validation mAP50:95 | Link |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `YOLOv8n-SigmoidTau-COCO-best.pt` | COCO2017 | 640 | 3151904 | 0.5188 | 0.3694 | Add public link |
| `YOLOv8n-SigmoidTau-VOC-best.pt` | PASCAL VOC | 640 | 3009548 | 0.8251 | 0.6205 | Add public link |
| `YOLOv8n-SigmoidTau-VisDrone-best.pt` | VisDrone2019 | 960 | 3007598 | 0.4133 | 0.2487 | Add public link |

Place under:

```text
checkpoints/final_students/
```

## Validation

```bash
bash scripts/test_all_released.sh
```

See `docs/released_checkpoint_metrics.md` and `results/released_checkpoint_metrics.csv` for the complete metric table.
