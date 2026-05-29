# Released Checkpoint Metrics

This table records the validation results of the candidate/released verification checkpoints. These metrics are included for code verification and result transparency. They should not be confused with the complete paper-production results unless the corresponding full-training paper checkpoints are explicitly released.

The same values are also stored in `results/released_checkpoint_metrics.csv`.

## Verification metrics

| Dataset | Checkpoint | P | R | mAP50 | mAP50:95 | Params | Weights | Data | Img |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| COCO | `YOLOv8s-COCO-teacher` | 0.6818 | 0.5623 | 0.6129 | 0.4474 | 11156544 | `checkpoints/teachers/YOLOv8s-COCO-teacher.pt` | `configs/coco.yaml` | 640 |
| COCO | `YOLOv8n-COCO-standard-best` | 0.6023 | 0.4537 | 0.4949 | 0.3493 | 3151904 | `checkpoints/baselines/YOLOv8n-COCO-standard-best.pt` | `configs/coco.yaml` | 640 |
| COCO | `YOLOv8n-SigmoidTau-COCO-best` | 0.6364 | 0.4762 | 0.5188 | 0.3694 | 3151904 | `checkpoints/final_students/YOLOv8n-SigmoidTau-COCO-best.pt` | `configs/coco.yaml` | 640 |
| VOC | `YOLOv8s-VOC-teacher` | 0.8418 | 0.8054 | 0.8624 | 0.6744 | 11133324 | `checkpoints/teachers/YOLOv8s-VOC-teacher.pt` | `configs/voc.yaml` | 640 |
| VOC | `YOLOv8n-VOC-standard-best` | 0.7934 | 0.7456 | 0.8192 | 0.6073 | 3009548 | `checkpoints/baselines/YOLOv8n-VOC-standard-best.pt` | `configs/voc.yaml` | 640 |
| VOC | `YOLOv8n-SigmoidTau-VOC-best` | 0.8117 | 0.7577 | 0.8251 | 0.6205 | 3009548 | `checkpoints/final_students/YOLOv8n-SigmoidTau-VOC-best.pt` | `configs/voc.yaml` | 640 |
| VisDrone | `YOLOv8s-VisDrone-teacher` | 0.5788 | 0.4576 | 0.4851 | 0.2998 | 11129454 | `checkpoints/teachers/YOLOv8s-VisDrone-teacher.pt` | `configs/visdrone.yaml` | 960 |
| VisDrone | `YOLOv8n-VisDrone-standard-best` | 0.4998 | 0.3917 | 0.3991 | 0.2419 | 3007598 | `checkpoints/baselines/YOLOv8n-VisDrone-standard-best.pt` | `configs/visdrone.yaml` | 960 |
| VisDrone | `YOLOv8n-SigmoidTau-VisDrone-best` | 0.5157 | 0.3893 | 0.4133 | 0.2487 | 3007598 | `checkpoints/final_students/YOLOv8n-SigmoidTau-VisDrone-best.pt` | `configs/visdrone.yaml` | 960 |

## Corrections applied before release

- The original VisDrone baseline row pointed to the Sigmoid-τ checkpoint path. It is corrected to `checkpoints/baselines/YOLOv8n-VisDrone-standard-best.pt`.
- The original VisDrone Sigmoid-τ row pointed to the baseline checkpoint path. It is corrected to `checkpoints/final_students/YOLOv8n-SigmoidTau-VisDrone-best.pt`.
- The typo `YYOLOv8n-SigmoidTau-VisDrone-best` is corrected to `YOLOv8n-SigmoidTau-VisDrone-best`.
- Absolute local paths such as `/root/autodl-tmp/...` and `/path/to/...` are replaced by repository yaml paths. Users should edit the `path:` field inside each yaml for their own machines.
