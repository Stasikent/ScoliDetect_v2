# cobb_two_borders_v9.py
# -*- coding: utf-8 -*-
"""
Автоматическое определение границ позвоночника и измерение угла Кобба.
Версия v9 — стабильная: совместима с OpenCV 4.x и scikit-image 0.22+.
"""

import argparse
import os
import math
import csv
from pathlib import Path
import numpy as np
import pydicom
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.morphology import white_tophat, disk
import cv2


# ------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ------------------------

def read_dicom_float(path):
    """Читает DICOM, нормализует в [0..1] и инвертирует при MONOCHROME1."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = np.max(arr) - arr

    vmin, vmax = np.percentile(arr, (0.5, 99.5))
    arr = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return arr


def preprocess_for_edges(img01):
    """Подготовка к поиску границ позвоночника: удаление фона, усиление контраста."""
    radius = max(15, int(min(img01.shape) * 0.01))
    # Совместимость с новыми версиями skimage
    try:
        bg_removed = white_tophat(img01, footprint=disk(radius))
    except TypeError:
        bg_removed = white_tophat(img01, selem=disk(radius))

    clahe = equalize_adapthist(bg_removed, clip_limit=0.02)

    # Приведение к uint8 перед medianBlur (иначе OpenCV падает)
    if clahe.dtype != np.uint8:
        clahe_u8 = cv2.normalize(clahe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        clahe_u8 = clahe

    clahe_u8 = cv2.medianBlur(clahe_u8, 3)
    clahe_f = clahe_u8.astype(np.float32) / 255.0
    return clahe_f


def vertical_profile_min_center(img01):
    """Находим центр позвоночника как минимум вертикального профиля."""
    h, w = img01.shape
    y0, y1 = int(h * 0.15), int(h * 0.85)
    prof = np.mean(img01[y0:y1, :], axis=0)
    prof_s = cv2.GaussianBlur(prof.reshape(1, -1).astype(np.float32), (1, 31), 0).ravel()
    x_center = int(np.argmin(prof_s))
    return x_center, prof_s


def refine_center_by_symmetry(img01, x_guess, half_width):
    """Точная коррекция центра по симметрии текстур слева/справа."""
    h, w = img01.shape
    best_x = x_guess
    best_score = -1.0
    search = range(max(half_width, x_guess - int(w * 0.05)),
                   min(w - half_width, x_guess + int(w * 0.05)))
    y0, y1 = int(h * 0.2), int(h * 0.8)
    for x in search:
        L = img01[y0:y1, max(0, x - 2 * half_width):x]
        R = img01[y0:y1, x:min(w, x + 2 * half_width)]
        Rf = np.flip(R, axis=1)
        if L.size == 0 or Rf.size == 0 or L.shape != Rf.shape:
            continue
        v1 = L.reshape(-1).astype(np.float32)
        v2 = Rf.reshape(-1).astype(np.float32)
        num = float(np.dot(v1, v2))
        den = float(np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        score = num / den
        if score > best_score:
            best_score = score
            best_x = x
    return best_x


def detect_borders(img01, x_center, roi_half_width):
    """Находит левый и правый край позвоночника в вертикальном окне."""
    h, w = img01.shape
    x0 = max(0, x_center - roi_half_width)
    x1 = min(w, x_center + roi_half_width)
    roi = img01[:, x0:x1]

    gx = cv2.Sobel((roi * 255).astype(np.uint8), cv2.CV_32F, 1, 0, ksize=3)
    gx = gx / 255.0

    left_idx = np.full(h, np.nan, dtype=np.float32)
    right_idx = np.full(h, np.nan, dtype=np.float32)
    thr = max(0.03, np.percentile(np.abs(gx), 90))

    for y in range(h):
        row = gx[y, :]
        left_zone = row[:roi_half_width]
        right_zone = row[roi_half_width:]

        if left_zone.size > 0:
            li = int(np.argmin(left_zone))
            if left_zone[li] < -thr:
                left_idx[y] = x0 + li

        if right_zone.size > 0:
            ri = int(np.argmax(right_zone))
            if right_zone[ri] > thr:
                right_idx[y] = x0 + roi_half_width + ri

    def clean_track(x):
        yy = np.arange(h, dtype=np.float32)
        good = ~np.isnan(x)
        if good.sum() < max(10, h * 0.1):
            return None
        xs = x.copy()
        xs[~good] = np.interp(yy[~good], yy[good], x[good])
        xs_u8 = cv2.normalize(xs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        xs_u8 = cv2.medianBlur(xs_u8, 9)
        xs = xs_u8.astype(np.float32)
        return xs

    L = clean_track(left_idx)
    R = clean_track(right_idx)
    return L, R, (x0, 0, x1 - x0, h)


def cobb_from_tracks(L, R):
    """Вычисляем угол Кобба по наклону средней линии."""
    if L is None or R is None:
        return 0.0
    h = len(L)
    yy = np.arange(h, dtype=np.float32)
    C = (L + R) / 2.0
    a, b = np.polyfit(yy, C, 1)
    angle = math.degrees(math.atan(a))
    return float(angle)


def draw_overlay(base_img01, L, R, roi_rect, angle_deg, out_path):
    """Отрисовывает наложение границ и угол Кобба на изображении."""
    h, w = base_img01.shape
    bg = (base_img01 * 255).astype(np.uint8)
    bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    x, y, ww, hh = roi_rect
    cv2.rectangle(bg, (x, y), (x + ww, y + hh), (0, 255, 255), 2)

    overlay = bg.copy()
    cv2.rectangle(overlay, (x + 1, y), (x + ww - 1, y + hh),
                  (0, 255, 255), -1)
    bg = cv2.addWeighted(overlay, 0.12, bg, 0.88, 0)

    if L is not None:
        for yy in range(h - 1):
            cv2.line(bg, (int(L[yy]), yy), (int(L[yy + 1]), yy + 1),
                     (0, 220, 0), 2)
    if R is not None:
        for yy in range(h - 1):
            cv2.line(bg, (int(R[yy]), yy), (int(R[yy + 1]), yy + 1),
                     (0, 50, 255), 2)

    label = f"Cobb: {angle_deg:.1f} deg" if (L is not None and R is not None) else "Cobb: 0.0 deg (fallback)"
    cv2.rectangle(bg, (10, 10), (10 + 420, 60), (40, 40, 40), -1)
    cv2.putText(bg, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, bg)


def process_one(path, out_dir):
    try:
        img = read_dicom_float(path)
        prep = preprocess_for_edges(img)
        h, w = prep.shape
        x_center, _ = vertical_profile_min_center(prep)
        roi_half = max(20, int(w * 0.03))
        x_center = refine_center_by_symmetry(prep, x_center, roi_half)
        L, R, rect = detect_borders(prep, x_center, roi_half)

        if L is None or R is None:
            roi_half2 = min(int(roi_half * 1.6), max(40, int(w * 0.06)))
            L2, R2, rect2 = detect_borders(prep, x_center, roi_half2)
            if L is None and L2 is not None:
                L, rect = L2, rect2
            if R is None and R2 is not None:
                R, rect = R2, rect2

        angle = cobb_from_tracks(L, R)
        out_png = os.path.join(out_dir, Path(path).stem + "_overlay.png")
        os.makedirs(out_dir, exist_ok=True)
        draw_overlay(img, L, R, rect, angle, out_png)

        ok = (L is not None and R is not None)
        return {"file": os.path.basename(path),
                "status": "ok" if ok else "fallback",
                "angle_deg": f"{angle:.2f}"}
    except Exception as e:
        return {"file": os.path.basename(path),
                "status": f"error: {e}",
                "angle_deg": "0.00"}


def main():
    ap = argparse.ArgumentParser(description="Auto spine borders & Cobb (v9)")
    ap.add_argument("input", help="Папка с .dcm файлами")
    ap.add_argument("--out", default="results_v9.csv", help="CSV с результатами")
    ap.add_argument("--save-overlays", dest="overlays", default="out_imgs_v9",
                    help="Папка для сохранения PNG")
    args = ap.parse_args()

    in_dir = Path(args.input)
    files = sorted([str(p) for p in in_dir.glob("*.dcm")])
    print(f"Найдено файлов: {len(files)}")

    rows = []
    ok_cnt, err_cnt = 0, 0
    for f in files:
        name = Path(f).stem
        print(f"→ {name}: читаю и нормализую…")
        print(f"→ {name}: автоизмерение…")
        res = process_one(f, args.overlays)
        rows.append(res)
        if res["status"].startswith("error"):
            print(f"[ERROR] {os.path.basename(f)}: {res['status']}")
            err_cnt += 1
        else:
            if res["status"] == "ok":
                ok_cnt += 1

    with open(args.out, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["file", "status", "angle_deg"])
        w.writeheader()
        w.writerows(rows)

    print(f"Готово: записано {len(rows)} строк в {args.out}.  "
          f"Успешно: {ok_cnt}, ошибок/фолбэков: {len(rows)-ok_cnt}.")


if __name__ == "__main__":
    main()
