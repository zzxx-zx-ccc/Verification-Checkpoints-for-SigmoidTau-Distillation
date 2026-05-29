import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def parse_yaml_simple(path):
    data = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip().strip("'").strip('"')
    return data

def list_images(root):
    return sorted([p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS])

def label_for_image(img_path, dataset_root):
    rel = img_path.relative_to(dataset_root)
    parts = list(rel.parts)
    if parts[0] == "images":
        parts[0] = "labels"
    return dataset_root.joinpath(*parts).with_suffix(".txt")

def check_label_file(label_path, nc):
    bad = []
    nobj = 0
    if not label_path.exists():
        return nobj, [f"missing label file: {label_path}"]
    txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return nobj, []
    for i, line in enumerate(txt.splitlines(), 1):
        ss = line.strip().split()
        if len(ss) != 5:
            bad.append(f"{label_path}:{i}: need 5 columns, got {len(ss)}")
            continue
        try:
            c = int(float(ss[0]))
            vals = [float(x) for x in ss[1:]]
        except Exception:
            bad.append(f"{label_path}:{i}: non-numeric value")
            continue
        if c < 0 or c >= nc:
            bad.append(f"{label_path}:{i}: class {c} out of range [0,{nc-1}]")
        if any(v < -1e-6 or v > 1 + 1e-6 for v in vals):
            bad.append(f"{label_path}:{i}: xywh not normalized to [0,1], vals={vals}")
        if vals[2] <= 0 or vals[3] <= 0:
            bad.append(f"{label_path}:{i}: width/height <= 0, vals={vals}")
        nobj += 1
    return nobj, bad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--nc", type=int, default=80)
    args = ap.parse_args()

    y = parse_yaml_simple(args.data)
    root = Path(y.get("path", "")).expanduser()
    train_dir = root / y.get("train", "images/train2017")
    val_dir = root / y.get("val", "images/val2017")

    print("DATA YAML:", args.data)
    print("DATA ROOT:", root)
    print("TRAIN DIR:", train_dir)
    print("VAL DIR:", val_dir)

    for p in [root, train_dir, val_dir, root / "labels/train2017", root / "labels/val2017"]:
        print(("OK   " if p.exists() else "MISS "), p)

    broken = []
    for p in root.rglob("*"):
        if p.is_symlink() and not p.exists():
            broken.append(str(p))
    print("Broken symlinks:", len(broken))
    for x in broken[:20]:
        print("  ", x)

    all_bad = []
    for split, img_dir in [("train", train_dir), ("val", val_dir)]:
        imgs = list_images(img_dir)
        missing = 0
        empty = 0
        nlabels = 0
        nobjects = 0
        bad = []

        for img in imgs:
            lab = label_for_image(img, root)
            if not lab.exists():
                missing += 1
                bad.append(f"missing label file: {lab}")
                continue
            nlabels += 1
            if lab.stat().st_size == 0:
                empty += 1
            nobj, b = check_label_file(lab, args.nc)
            nobjects += nobj
            bad.extend(b)

        print(f"\n[{split}]")
        print("images:", len(imgs))
        print("label files found:", nlabels)
        print("missing label files:", missing)
        print("empty label files:", empty)
        print("objects:", nobjects)
        print("bad label lines:", len(bad))

        for x in bad[:30]:
            print("  BAD:", x)
        all_bad.extend(bad)

    if broken or all_bad:
        print("\nWARNING: dataset check found problems. Please inspect BAD lines above.")
    else:
        print("\nDataset check passed: no broken symlinks or invalid label lines detected.")

if __name__ == "__main__":
    main()
