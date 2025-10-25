# thor_scoli_app/cli.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from glob import glob
from typing import List, Tuple, Optional

import numpy as np
import cv2

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
except Exception:
    pydicom = None  # позволим обрабатывать PNG/JPG даже без pydicom

# наш алгоритм
from .auto_cobb import auto_cobb_from_gray


def _read_gray_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        if pydicom is None:
            raise RuntimeError("pydicom не установлен, а вход — DICOM")
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)

        # LUT (учитывает RescaleSlope/Intercept)
        try:
            arr = apply_modality_lut(arr, ds).astype(np.float32)
        except Exception:
            pass

        # инверсия, если это MONOCHROME1
        if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
            arr = np.max(arr) - arr

        # в uint8 с робастной нормировкой
        p1, p99 = np.percentile(arr, [1.0, 99.0])
        if p99 <= p1:
            p1, p99 = float(np.min(arr)), float(np.max(arr))
        img = (arr - p1) / (p99 - p1 + 1e-6)
        img = np.clip(img, 0, 1)
        img = (img * 255.0).astype(np.uint8)
        return img
    else:
        # обычные изображения
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"не удалось прочитать изображение: {path}")
        return img


def _collect_files(inp: str) -> List[str]:
    if os.path.isdir(inp):
        pats = ["*.dcm", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        files = []
        for p in pats:
            files.extend(glob(os.path.join(inp, p)))
        return sorted(files)
    else:
        return [inp]


def main():
    ap = argparse.ArgumentParser(description="Cobb auto-measure CLI")
    ap.add_argument("input", help="Папка или файл с рентгенами")
    ap.add_argument("--mode", default="auto", choices=["auto"], help="режим работы")
    ap.add_argument("--out", default="results.csv", help="CSV с результатами")
    ap.add_argument("--save-overlays", default=None, help="папка для сохранения оверлеев")
    args = ap.parse_args()

    files = _collect_files(args.input)
    print(f"Найдено файлов: {len(files)}")

    if args.save_overlays:
        os.makedirs(args.save_overlays, exist_ok=True)

    results_rows: List[Tuple[str, str, str, str]] = []
    ok_cnt, err_cnt = 0, 0

    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"→ {fname}: читаю и нормализую…")
        try:
            gray = _read_gray_image(fpath)
            print(f"→ {fname}: автоизмерение…")

            res = auto_cobb_from_gray(gray)  # ← наш алгоритм

            angle = float(res.get("angle_deg", float("nan")))
            overlay = res.get("overlay_bgr", None)

            if args.save_overlays and overlay is not None:
                out_png = os.path.join(args.save_overlays,
                                       os.path.splitext(fname)[0] + "_cobb.png")
                cv2.imwrite(out_png, overlay)

            results_rows.append((fname, f"{angle:.3f}", "ok", ""))
            ok_cnt += 1

        except Exception as e:
            err_cnt += 1
            etype, eval_, etb = sys.exc_info()
            tb_txt = "".join(traceback.format_exception(etype, eval_, etb)).rstrip()
            # Коротко в консоль:
            msg = str(e).strip() or e.__class__.__name__
            print(f"[ERROR] {fname}: {msg}")
            # И полный стек в CSV:
            results_rows.append((fname, "", "error", tb_txt))

    # запись CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "angle_deg", "status", "error"])
        wr.writerows(results_rows)

    print("Измерение Cobb " + "━" * 47 + f" 100%")
    print(f"Готово: записано {len(results_rows)} строк в {args.out}.  Успешно: {ok_cnt}, ошибок: {err_cnt}.")


if __name__ == "__main__":
    main()
