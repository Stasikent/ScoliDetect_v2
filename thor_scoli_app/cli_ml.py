# cli_ml.py
# -*- coding: utf-8 -*-
import os, glob, argparse, csv
import numpy as np, cv2

try:
    import pydicom
    _HAS_DCM = True
except Exception:
    _HAS_DCM = False

from auto_cobb_ml import auto_cobb_from_gray_ml


def _read_gray(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise RuntimeError(f"cannot read {path}")
        return im
    if ext in [".dcm",".dicom"]:
        if not _HAS_DCM:
            raise RuntimeError("pydicom required for DICOM")
        d = pydicom.dcmread(path)
        arr = d.pixel_array.astype(np.float32)
        arr -= arr.min()
        arr /= (arr.max() + 1e-6)
        return (arr * 255).astype(np.uint8)
    raise RuntimeError(f"unsupported: {path}")


def main():
    ap = argparse.ArgumentParser("Auto Cobb with optional classic-ML")
    ap.add_argument("data", help="file or folder")
    ap.add_argument("--ml", help="joblib model path from train_border_ml.py", default=None)
    ap.add_argument("--out", default="results_ml.csv")
    ap.add_argument("--save-overlays", default=None, help="folder to save overlays")
    ap.add_argument("--roi", type=float, default=0.18)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--w-grad", type=float, default=0.6)
    ap.add_argument("--w-ml", type=float, default=0.4)
    args = ap.parse_args()

    paths = []
    if os.path.isdir(args.data):
        for p in glob.glob(os.path.join(args.data, "*")):
            if os.path.splitext(p)[1].lower() in (".png",".jpg",".jpeg",".bmp",".tif",".tiff",".dcm",".dicom"):
                paths.append(p)
    else:
        paths = [args.data]

    if args.save_overlays and not os.path.exists(args.save_overlays):
        os.makedirs(args.save_overlays, exist_ok=True)

    rows = []
    print(f"Найдено файлов: {len(paths)}")
    for p in sorted(paths):
        name = os.path.basename(p)
        try:
            print(f"→ {name}: читаю и нормализую…")
            g = _read_gray(p)
            print(f"→ {name}: автоизмерение…")
            res = auto_cobb_from_gray_ml(g,
                                         roi_margin=args.roi,
                                         ml_model_path=args.ml,
                                         alpha=args.alpha,
                                         w_grad=args.w_grad,
                                         w_ml=args.w_ml)
            angle = res["angle_deg"]
            rows.append([name, f"{angle:.2f}"])
            if args.save_overlays:
                outp = os.path.join(args.save_overlays, os.path.splitext(name)[0] + "_overlay.png")
                cv2.imwrite(outp, res["overlay_bgr"])
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            rows.append([name, "error"])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "cobb_deg"])
        for r in rows:
            w.writerow(r)
    print(f"Готово: записано {len(rows)} строк в {args.out}.")


if __name__ == "__main__":
    main()
