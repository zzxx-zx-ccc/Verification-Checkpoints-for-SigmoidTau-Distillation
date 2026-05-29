import argparse
from pathlib import Path
import re
import torch


def unwrap_state_dict(obj):
    """Return a plain state_dict from a YOLO/torch checkpoint or module."""
    if isinstance(obj, dict):
        for key in ("ema", "model"):
            if key in obj and obj[key] is not None:
                m = obj[key]
                if hasattr(m, "state_dict"):
                    return m.float().state_dict() if hasattr(m, "float") else m.state_dict()
                if isinstance(m, dict):
                    return m
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    if hasattr(obj, "state_dict"):
        return obj.float().state_dict() if hasattr(obj, "float") else obj.state_dict()
    raise TypeError(f"Unsupported checkpoint type: {type(obj)}")


def map_student_key(k: str):
    """Map joint model student keys to normal YOLOv8n keys.

    Expected joint layout in this project:
      teacher: model.0 ... model.22
      student: model.23 ... model.44, and student Detect: model.46
    Therefore:
      model.23 -> model.0
      ...
      model.44 -> model.21
      model.46 -> model.22
    """
    m = re.match(r"^model\.(\d+)\.(.*)$", k)
    if not m:
        return None
    idx = int(m.group(1))
    rest = m.group(2)
    if 23 <= idx <= 44:
        return f"model.{idx - 23}.{rest}"
    if idx == 46:
        return f"model.22.{rest}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", required=True)
    ap.add_argument("--baseline", required=True, help="normal YOLOv8n.pt used as template")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    joint_path = Path(args.joint)
    baseline_path = Path(args.baseline)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    joint_ckpt = torch.load(joint_path, map_location="cpu", weights_only=False)
    base_ckpt = torch.load(baseline_path, map_location="cpu", weights_only=False)

    joint_sd = unwrap_state_dict(joint_ckpt)
    base_model = base_ckpt.get("model", None) if isinstance(base_ckpt, dict) else None
    if base_model is None or not hasattr(base_model, "state_dict"):
        raise RuntimeError("Baseline checkpoint must contain a normal YOLO model object under key 'model'.")

    base_model = base_model.float()
    base_sd = base_model.state_dict()

    mapped = {}
    used = 0
    skipped_shape = []
    for jk, v in joint_sd.items():
        nk = map_student_key(jk)
        if nk is None or nk not in base_sd:
            continue
        if tuple(v.shape) == tuple(base_sd[nk].shape):
            mapped[nk] = v.float() if torch.is_floating_point(v) else v
            used += 1
        else:
            skipped_shape.append((jk, nk, tuple(v.shape), tuple(base_sd[nk].shape)))

    if used == 0:
        raise RuntimeError("No student parameters were mapped. Check joint checkpoint layer layout.")

    missing, unexpected = base_model.load_state_dict(mapped, strict=False)

    out_ckpt = dict(base_ckpt) if isinstance(base_ckpt, dict) else {"model": base_model}
    out_ckpt["model"] = base_model
    out_ckpt["ema"] = None
    out_ckpt["train_args"] = out_ckpt.get("train_args", {})
    out_ckpt["student_only_extracted_from"] = str(joint_path)
    out_ckpt["student_only_note"] = "Student branch extracted from teacher-student joint/KD checkpoint."

    torch.save(out_ckpt, out_path)

    print(f"Mapped tensors: {used}")
    print(f"Missing tensors in template after partial load: {len(missing)}")
    print(f"Unexpected tensors: {len(unexpected)}")
    print(f"Shape-skipped tensors: {len(skipped_shape)}")
    if skipped_shape[:10]:
        print("First shape-skipped entries:")
        for x in skipped_shape[:10]:
            print("  ", x)
    print(f"Saved student-only checkpoint: {out_path}")

if __name__ == "__main__":
    main()
