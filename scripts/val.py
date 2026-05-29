#!/usr/bin/env python3
import argparse
import sys
import types
import yaml


def load_yaml(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_globals(distill, hyp_cfg):
    g = types.ModuleType("Globals")
    g.bool_distill = bool(distill)
    g.flag_train_output = 3
    g.bool_val_output = True

    g.scale_default = hyp_cfg.get("scale_default", "n")
    g.scale_student = hyp_cfg.get("scale_student", "n")
    g.scale_teacher = hyp_cfg.get("scale_teacher", "s")
    g.teacher_peer_list = hyp_cfg.get(
        "teacher_peer_list",
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 45],
    )
    g.student_peer_list = hyp_cfg.get(
        "student_peer_list",
        [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46],
    )

    g.hyp_T = float(hyp_cfg.get("hyp_T", 3.0))
    g.hyp_cls_distill = float(hyp_cfg.get("hyp_cls_distill", 1.0))
    g.hyp_box_distill = float(hyp_cfg.get("hyp_box_distill", 0.1))
    g.hyp_dfl_distill = float(hyp_cfg.get("hyp_dfl_distill", 0.05))
    g.hyp_w_t_cls = float(hyp_cfg.get("hyp_w_t_cls", 0.99))
    g.hyp_w_t_box = float(hyp_cfg.get("hyp_w_t_box", 0.50))
    g.hyp_w_t_dfl = float(hyp_cfg.get("hyp_w_t_dfl", 0.50))
    g.hyp_cls_kd_mode = hyp_cfg.get("hyp_cls_kd_mode", "sigmoid_tau")
    g.hyp_cls_kd_t2 = bool(hyp_cfg.get("hyp_cls_kd_t2", False))
    g.hyp_kd = float(hyp_cfg.get("hyp_kd", 0.1))
    g.hyp_use_partition_kd = bool(hyp_cfg.get("hyp_use_partition_kd", True))

    return g


def parse_args():
    p = argparse.ArgumentParser(description="Validate a trained model")
    p.add_argument("--weights", required=True, help="Weights file (.pt)")
    p.add_argument("--data", required=True, help="Dataset yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--distill", action="store_true", help="Use Detect_Distill head")
    p.add_argument("--hyp", default="configs/hyp_sigmoid_tau.yaml")
    p.add_argument("--model", default=None, help="Optional model yaml for distill weights")
    return p.parse_args()


def main():
    args = parse_args()
    hyp_cfg = load_yaml(args.hyp)
    sys.modules["Globals"] = build_globals(args.distill, hyp_cfg)

    from ultralytics import YOLO

    model = YOLO(args.model if args.model else args.weights)
    if args.model:
        model.load(args.weights)

    model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device)


if __name__ == "__main__":
    main()
