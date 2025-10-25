# train_border_ml.py
# -*- coding: utf-8 -*-
# учебный классический ML-тренер пиксельного классификатора «граница позвоночника / не граница» внутри центрального ROI.
# Его задача — по размеченным маскам *_mask.png научить модель (GradientBoostingClassifier), которая затем используется в auto_cobb_ml.py
# чтобы улучшать «энергию» динамического программирования и тем самым точнее проводить центрлайн.
from __future__ import annotations
import os, glob, argparse, joblib, math
import numpy as np
import cv2

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

try:
    import pydicom
    _HAS_DCM = True
except Exception:
    _HAS_DCM = False


def _read_gray(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"failed to read image: {path}")
        return img
    if ext in [".dcm", ".dicom"]:
        if not _HAS_DCM:
            raise RuntimeError("pydicom is required for DICOM")
        d = pydicom.dcmread(path)
        arr = d.pixel_array.astype(np.float32)
        arr -= arr.min()
        arr /= (arr.max() + 1e-6)
        arr = (arr * 255).astype(np.uint8)
        return arr
    raise RuntimeError(f"unsupported image: {path}")


def _roi_box(w, h, margin_frac=0.18):
    cx = w // 2
    half = max(20, int(round(w * margin_frac)))
    x0 = max(0, cx - half)
    x1 = min(w - 1, cx + half)
    return x0, 0, x1, h - 1


def _features(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """векторизованные признаки для каждого пикселя ROI."""
    roi = gray[y0:y1+1, x0:x1+1].astype(np.float32) / 255.0
    h, w = roi.shape

                                                                                                                            # сглаживание лёгкое
    blur = cv2.GaussianBlur(roi, (0,0), 1.2)

                                                                                                                            # градиенты
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)

                                                                                                                            # локальная статистика
    k = max(3, int(round(min(h,w) * 0.01)) | 1)
    mean = cv2.blur(roi, (k,k))
    sqr = cv2.blur(roi*roi, (k,k))
    var = np.maximum(sqr - mean*mean, 0.0)
    std = np.sqrt(var)

                                                                                                                            # позиционные (центр ROI = 0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx - (w-1)/2.0) / max(1.0, (w-1)/2.0)
    yy = (yy - (h-1)/2.0) / max(1.0, (h-1)/2.0)

    feats = np.stack([roi, blur, gx, gy, mag, lap, mean, std, xx, yy], axis=-1)
    feats = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
    return feats


def _mask_for(img_path: str) -> str|None:
    base = os.path.splitext(img_path)[0]
    for suf in ["_mask.png", ".mask.png", "_mask.jpg"]:
        cand = base + suf
        if os.path.exists(cand):
            return cand
                                                                                                                            # также ищем по имени без расширения среди PNG
    candidates = glob.glob(base + "*mask*.png")
    return candidates[0] if candidates else None


def main():
    ap = argparse.ArgumentParser("Train classic ML border model")
    ap.add_argument("data_dir", help="directory with images + *_mask.png")
    ap.add_argument("--out", default="border_model.joblib", help="output model path")
    ap.add_argument("--roi", type=float, default=0.18, help="ROI half-width fraction")
    ap.add_argument("--max-pos", type=int, default=150_000, help="max positive samples")
    ap.add_argument("--max-neg", type=int, default=150_000, help="max negative samples")
    args = ap.parse_args()

    img_paths = sorted([p for p in glob.glob(os.path.join(args.data_dir, "*"))
                        if os.path.splitext(p)[1].lower() in (".png",".jpg",".jpeg",".bmp",".tif",".tiff",".dcm",".dicom")])

    Xs, ys = [], []

    for p in img_paths:
        mpath = _mask_for(p)
        if not mpath:
            print(f"[skip] no mask for {os.path.basename(p)}")
            continue
        img = _read_gray(p)
        msk = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            print(f"[skip] cannot read mask {mpath}")
            continue
        msk = (msk > 0).astype(np.uint8)
        h, w = img.shape
        x0,y0,x1,y1 = _roi_box(w,h,args.roi)

        feats = _features(img, x0,y0,x1,y1)
        roi_m = msk[y0:y1+1, x0:x1+1].reshape(-1)

        pos_idx = np.flatnonzero(roi_m == 1)
        neg_idx = np.flatnonzero(roi_m == 0)

        if len(pos_idx) == 0:
            print(f"[skip] empty positives in {p}")
            continue

                                                                                                                            # балансируем
        rng = np.random.default_rng(123)
        pos_sel = rng.choice(pos_idx, size=min(args.max_pos, len(pos_idx)), replace=False)
        neg_sel = rng.choice(neg_idx, size=min(args.max_neg, len(neg_idx)), replace=False)
        sel = np.concatenate([pos_sel, neg_sel])

        Xs.append(feats[sel])
        ys.append(roi_m[sel])

        print(f"[+] {os.path.basename(p)}: pos={len(pos_sel)}, neg={len(neg_sel)}")

    if not Xs:
        raise SystemExit("No training data found. Make sure *_mask.png exist.")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.uint8)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    clf = GradientBoostingClassifier(random_state=42, max_depth=3, n_estimators=250, learning_rate=0.08)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    print(classification_report(y_te, y_pred, digits=3))

    meta = {
        "roi_margin": args.roi,
        "feat_dim": X.shape[1],
        "scaler": None,                                                                                                         # оставлено для совместимости; признаки уже «в масштабе»
    }
    joblib.dump({"clf": clf, "meta": meta}, args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
