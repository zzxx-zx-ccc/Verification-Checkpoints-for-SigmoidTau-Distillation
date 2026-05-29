#!/usr/bin/env python3
import argparse
import os
import sys
import types
from pathlib import Path

import yaml
import torch


def load_yaml(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_globals(args, hyp_cfg):
    g = types.ModuleType("Globals")

    # Core switches
    g.bool_distill = True
    g.flag_train_output = int(hyp_cfg.get("flag_train_output", 3))
    g.bool_val_output = bool(hyp_cfg.get("bool_val_output", True))

    # Model wiring
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

    # KD hyperparameters
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

    # Optional debug knobs
    g.hyp_teacher_conf_gate = float(hyp_cfg.get("hyp_teacher_conf_gate", 0.0))
    g.hyp_kd_log_interval = int(hyp_cfg.get("hyp_kd_log_interval", 50))

    # Paths and runtime
    g.model_file = args.model
    g.dataset = args.data
    g.bool_load_student = args.load_student
    g.path_student = args.student
    g.path_teacher = args.teacher
    g.path_student_prepare = args.student_prepare
    g.path_teacher_prepare = args.teacher_prepare
    g.model_based_file = args.model_based
    g.final = args.final
    g.phase = args.phase

    return g


def set_distill_runtime(enabled):
    try:
        import ultralytics.nn.tasks as tasks_mod
        tasks_mod.bool_distill = enabled
    except Exception:
        pass
    try:
        import ultralytics.utils.loss as loss_mod
        loss_mod.bool_distill = enabled
    except Exception:
        pass
    try:
        import ultralytics.engine.model as model_mod
        model_mod.bool_distill = enabled
    except Exception:
        pass
    try:
        import ultralytics.utils.ops as ops_mod
        ops_mod.bool_distill = enabled
    except Exception:
        pass


def normalize_omp_threads():
    omp_threads = os.environ.get("OMP_NUM_THREADS")
    if not omp_threads or not omp_threads.isdigit() or int(omp_threads) <= 0:
        os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() or 1))


def ensure_parent(path_str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def load_state_dict_with_report(model, weight_path, tag):
    print(f"[{tag}] loading: {weight_path}")
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(state_dict, strict=False)
    print(f"[{tag}] missing_keys: {len(result.missing_keys)}")
    print(f"[{tag}] unexpected_keys: {len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"[{tag}] first_missing: {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"[{tag}] first_unexpected: {result.unexpected_keys[:5]}")


def exchange_teacher(yolo_cls, path_teacher, path_teacher_prepare, teacher_peer_list):
    save_state = {}
    model_teacher = yolo_cls(path_teacher)
    for param_tensor in model_teacher.state_dict():
        old_layer_number = param_tensor.split(".")[2]
        new_layer_number = str(teacher_peer_list[int(old_layer_number)])
        save_state.update({
            param_tensor.replace("." + old_layer_number + ".", "." + new_layer_number + ".", 1):
                model_teacher.state_dict()[param_tensor]
        })
    torch.save(save_state, path_teacher_prepare)


def exchange_student(yolo_cls, path_student, path_student_prepare, student_peer_list):
    """Map a student checkpoint into the student branch of the joint distillation model.

    Some candidate checkpoints may contain extra layers or non-standard layer indices.
    For release recovery, we map valid YOLOv8n-style layers and skip unmappable tensors
    instead of stopping with IndexError.
    """
    save_state = {}
    skipped = []
    model_student = yolo_cls(path_student)
    state_dict = model_student.state_dict()

    for param_tensor, tensor in state_dict.items():
        parts = param_tensor.split(".")
        if len(parts) < 3 or not parts[2].isdigit():
            skipped.append(param_tensor)
            continue

        old_layer_number = parts[2]
        old_idx = int(old_layer_number)

        if old_idx >= len(student_peer_list):
            skipped.append(param_tensor)
            continue

        new_layer_number = str(student_peer_list[old_idx])
        save_state[
            param_tensor.replace("." + old_layer_number + ".", "." + new_layer_number + ".", 1)
        ] = tensor

    print(f"[exchange_student] mapped tensors: {len(save_state)}")
    print(f"[exchange_student] skipped tensors: {len(skipped)}")
    if skipped:
        print("[exchange_student] first skipped tensors:")
        for k in skipped[:20]:
            print("  -", k)

    torch.save(save_state, path_student_prepare)


def exchange_student_back(yolo_cls, model_based_train, model_after_train, path_save, student_peer_list):
    save_state = {}
    model_student = yolo_cls(model_based_train)
    for param_tensor in model_after_train.state_dict():
        old_layer_number = int(param_tensor.split(".")[2])
        if old_layer_number in student_peer_list:
            new_layer_number = str(student_peer_list.index(old_layer_number))
            save_state.update({
                param_tensor.replace("." + str(old_layer_number) + ".", "." + new_layer_number + ".", 1):
                    model_after_train.state_dict()[param_tensor]
            })
    model_student.load_state_dict(save_state, strict=False)
    model_student.save(path_save)


def ensure_phase1_files(globals_mod):
    missing = []
    if globals_mod.bool_load_student and not os.path.isfile(globals_mod.path_student_prepare):
        missing.append(globals_mod.path_student_prepare)
    if not os.path.isfile(globals_mod.path_teacher_prepare):
        missing.append(globals_mod.path_teacher_prepare)
    if not os.path.isfile(globals_mod.model_based_file):
        missing.append(globals_mod.model_based_file)
    if missing:
        msg = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError("Required files not found. Run phase=0 or fix paths:\n" + msg)


def freeze_teacher_layers(model, teacher_peer_list):
    net = model.model
    layers = getattr(net, "model", None)
    if layers is None:
        return
    for layer_idx in teacher_peer_list:
        if 0 <= layer_idx < len(layers):
            layers[layer_idx].eval()
            for param in layers[layer_idx].parameters():
                param.requires_grad = False


def parse_args():
    p = argparse.ArgumentParser(description="Train SigmoidTau distillation")
    p.add_argument("--data", default="configs/coco.yaml", help="Dataset yaml")
    p.add_argument("--model", default="configs/yolov8n_sigmoid_tau_coco.yaml", help="Model yaml")
    p.add_argument("--hyp", default="configs/hyp_sigmoid_tau.yaml", help="KD hyperparameter yaml")
    p.add_argument("--teacher", default="weights/yolov8s.pt", help="Teacher weights")
    p.add_argument("--student", default="weights/yolov8n.pt", help="Student weights")
    p.add_argument("--student-prepare", default="weights/prepare/yolov8_student_prepare.pth")
    p.add_argument("--teacher-prepare", default="weights/prepare/yolov8_teacher_prepare.pth")
    p.add_argument("--model-based", default="weights/yolov8n.pt", help="Base student model for export")
    p.add_argument("--final", default="checkpoints/final_students/YOLOv8n-SigmoidTau-COCO-best.pt", help="Exported student output")
    p.add_argument("--phase", type=int, default=1, choices=[0, 1])
    p.add_argument("--no-load-student", action="store_false", dest="load_student")
    p.set_defaults(load_student=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="sigmoid_tau_kd")
    p.add_argument("--close-mosaic", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    hyp_cfg = load_yaml(args.hyp)

    globals_mod = build_globals(args, hyp_cfg)
    sys.modules["Globals"] = globals_mod

    from ultralytics import YOLO

    normalize_omp_threads()
    set_distill_runtime(True)

    ensure_parent(globals_mod.path_student_prepare)
    ensure_parent(globals_mod.path_teacher_prepare)

    if args.phase == 0:
        if globals_mod.bool_load_student:
            exchange_student(YOLO, globals_mod.path_student, globals_mod.path_student_prepare, globals_mod.student_peer_list)
        exchange_teacher(YOLO, globals_mod.path_teacher, globals_mod.path_teacher_prepare, globals_mod.teacher_peer_list)
        print("[phase0] done.")
        return

    ensure_phase1_files(globals_mod)

    model = YOLO(globals_mod.model_file)
    if globals_mod.bool_load_student:
        load_state_dict_with_report(model, globals_mod.path_student_prepare, "student")
    load_state_dict_with_report(model, globals_mod.path_teacher_prepare, "teacher")
    freeze_teacher_layers(model, globals_mod.teacher_peer_list)

    model.train(
        data=globals_mod.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        save=True,
        save_period=max(1, min(10, args.epochs)),
        close_mosaic=args.close_mosaic,
        resume=args.resume,
        amp=args.amp,
        exist_ok=True,
        freeze=globals_mod.teacher_peer_list,
    )

    exchange_student_back(
        YOLO,
        globals_mod.model_based_file,
        model,
        globals_mod.final,
        globals_mod.student_peer_list,
    )

    model.val(data=globals_mod.dataset, imgsz=args.imgsz, batch=args.batch)


if __name__ == "__main__":
    main()
