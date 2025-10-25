# -*- coding: utf-8 -*-
# утилита активного отбора снимков для ручной разметки: она читает DICOM-ы
# находит грубый ROI грудного отдела, оценивает «информативность» ROI
# выбирает k снимков по стратегии (случайно или по метрике)
# опционально сохраняет кропы и фиксирует выбор в CSV-манифесте.
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple
import random

import numpy as np
import pydicom
import cv2


def read_dcm_uint8(path: Path) -> np.ndarray:                                                           #приводит DICOM к 8-битному изображению для классических OpenCV-фильтров.
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)

                                                                                                        # применяем rescale (если есть)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + inter

                                                                                                        # windowing, если присутствуют
    ww = getattr(ds, "WindowWidth", None)
    wc = getattr(ds, "WindowCenter", None)
    try:
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = float(ww[0])
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = float(wc[0])
    except Exception:
        pass

    if ww is not None and wc is not None and ww > 0:
        lo = wc - ww / 2.0
        hi = wc + ww / 2.0
        arr = np.clip(arr, lo, hi)
    else:
                                                                                                        # min-max по кадру
        mi, ma = float(np.min(arr)), float(np.max(arr))
        if ma > mi:
            arr = (arr - mi) / (ma - mi)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        arr = arr * 1023.0                                                                              # псевдо-динамический диапазон перед CLAHE

                                                                                                        # нормализация в 0..255 uint8
    arr = arr.astype(np.float32)
    mi, ma = float(arr.min()), float(arr.max())
    if ma > mi:
        arr = (arr - mi) / (ma - mi)
    else:
        arr[:] = 0.0
    img = (arr * 255.0).clip(0, 255).astype(np.uint8)

                                                                                                        # инвертированные снимки
    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        img = 255 - img

    return img


def central_roi_ap(gray_u8: np.ndarray) -> Tuple[slice, slice]:                                         # грубо локализует центральный фрагмент грудного отдела 
    h, w = gray_u8.shape

                                                                                                        # контрастирование — только uint8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray_u8)

                                                                                                        # столб позвонков темнее → инвертируем и усредняем по строкам
    inv = 255 - g
    col = inv.mean(axis=0).astype(np.float32)
    col = cv2.GaussianBlur(col, (0, 0), 7)  # float32 ok

    x_peak = int(np.argmax(col))
    half = max(int(w * 0.18), 100)
    x1 = max(0, x_peak - half)
    x2 = min(w, x_peak + half)

    y1 = int(h * 0.15)
    y2 = int(h * 0.85)

    return slice(y1, y2), slice(x1, x2)


def roi_score(gray_u8: np.ndarray, sy: slice, sx: slice) -> float:
    """Оценивает «информативность» ROI: плотность границ."""
    roi = gray_u8[sy, sx]                                                                               # чем выше score, тем богаче ROI на контуры (чаще видны границы позвонков/пластинок)-потенциально проще разметить.

    # фильтры только для uint8/float32 → приводим явно
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)                                                         # uint8 -> uint8
    edges = cv2.Canny(roi_blur, 40, 120, L2gradient=True)
    density = float(np.count_nonzero(edges)) / float(edges.size)
    return density


def find_dicoms(input_dir: Path) -> List[Path]:
    exts = (".dcm", ".dicom")
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in exts])


def main():
    ap = argparse.ArgumentParser(description="Выбор снимков для ручной разметки (active learning seed).")
    ap.add_argument("input", type=str, help="Папка с DICOM")
    ap.add_argument("--k", type=int, default=10, help="Сколько выбрать")
    ap.add_argument("--save-crops", type=str, default=None, help="Папка для кропов ROI (опционально)")
    ap.add_argument("--manifest", type=str, default="to_label_manifest.csv", help="CSV с выбранными файлами")
    ap.add_argument("--strategy", type=str, default="random", choices=["random", "topk"],
                    help="random — случайные; topk — по метрике ROI (информативные)")
    args = ap.parse_args()

    input_dir = Path(args.input)
    save_dir = Path(args.save_crops) if args.save_crops else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    files = find_dicoms(input_dir)
    if not files:
        print("no dicom files found")
        return

    rows = []
    scored = []
    for p in files:
        try:
            img = read_dcm_uint8(p)                                                                     # uint8 строго
            sy, sx = central_roi_ap(img)
            score = roi_score(img, sy, sx)
            scored.append((p, score, sy, sx))

            if save_dir:
                crop = img[sy, sx]
                outp = save_dir / f"{p.stem}_crop.png"
                cv2.imwrite(str(outp), crop)
        except Exception as e:
            print(f"[skip] {p.name}: {e}")

    if not scored:
        print("no images processed")
        return

                                                                                                        # выбор
    if args.strategy == "random":
        random.shuffle(scored)
        picked = scored[: args.k]                                                                       # перемешивает scored и берёт первые k
    else:                                                                                               # topk сортирует по score по убыванию и берёт k
        picked = sorted(scored, key=lambda t: t[1], reverse=True)[: args.k]

                                                                                                        # записать манифест
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "y1", "y2", "x1", "x2", "score"])
        for p, s, sy, sx in picked:
            wr.writerow([str(p), sy.start, sy.stop, sx.start, sx.stop, f"{s:.6f}"])

    print(f"selected: {len(picked)} → {args.manifest}")
    if save_dir:
        print(f"crops saved to: {save_dir}")


if __name__ == "__main__":
    main()
