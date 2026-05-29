# Dataset Preparation

This repository expects standard YOLO detection-format datasets. The public configuration files contain relative paths only. Please download the datasets from their official sources and update the `path:` field in the corresponding yaml file.

## COCO2017

Official source: https://cocodataset.org/

Configuration file: `configs/coco.yaml`

Expected structure:

```text
datasets/coco/
├── images/
│   ├── train2017/
│   └── val2017/
├── labels/
│   ├── train2017/
│   └── val2017/
├── train2017.txt
└── val2017.txt
```

The official Ultralytics COCO yaml and download script can also be used as a reference.

## PASCAL VOC

Official source: http://host.robots.ox.ac.uk/pascal/VOC/

Configuration file: `configs/voc.yaml`

Expected structure after conversion to YOLO format:

```text
datasets/voc/
├── images/
│   ├── train2007/
│   ├── val2007/
│   └── train2012/
└── labels/
    ├── train2007/
    ├── val2007/
    └── train2012/
```

## VisDrone2019-DET

Official source: https://github.com/VisDrone/VisDrone-Dataset

Configuration file: `configs/visdrone.yaml`

Expected structure after conversion to YOLO format:

```text
datasets/visdrone2019/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── labels/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── labels/
└── VisDrone2019-DET-test-dev/
    └── images/
```

## Notes

- Do not commit datasets to Git.
- Keep dataset paths local and machine-independent in the public repository.
- If you release converted label files, state the conversion script and class order clearly.
