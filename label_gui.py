# -*- coding: utf-8 -*-
"""
Простой GUI для ручной разметки угла Кобба.
Управление:
  - ЛКМ: поставить/переставить 4 точки (p1,p2 для верхней линии; p3,p4 для нижней)
  - R: сбросить точки на текущем снимке
  - S: сохранить результат (true_angle_deg) и перейти к следующему
  - N / Пробел: следующий снимок
  - B: предыдущий снимок
  - O: сохранить оверлей текущего снимка (PNG)
  - Q / ESC: выход

Сохранение:
  - labels.csv (в текущей папке)
  - Формат: file,true_angle_deg,notes
    (если строка для файла уже была — перезапишется)

Источник файлов:
  - Можно дать папку с .dcm / .png / .jpg.
  - Если разметка по кропам из select_for_labeling.py,
    скрипт попробует прочитать to_label_manifest.csv для
    маппинга crop -> оригинальный DICOM, чтобы в labels.csv
    писать исходный путь.
"""

from __future__ import annotations
import os
import math
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2

try:
    import pydicom
    HAS_PYDICOM = True
except Exception:
    HAS_PYDICOM = False


                                                                                                                            # Утилиты работы с файлами

def list_images(root: Path) -> List[Path]:
    exts = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    return files


def load_manifest(mapping_csv: Path) -> Dict[str, str]:
    """
    Загружает to_label_manifest.csv, если есть.
    Ожидаемые колонки: crop_file,orig_file
    Возвращает: {crop_stem: orig_full_path}
    """
    if not mapping_csv.exists():
        return {}
    mapping = {}
    with mapping_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        lowcols = [c.strip().lower() for c in rd.fieldnames or []]
        try:
            i_crop = lowcols.index("crop_file")
            i_orig = lowcols.index("orig_file")
        except Exception:
            return {}

        f.seek(0)
        rd = csv.reader(f)
        header = next(rd)
        for row in rd:
            try:
                crop = row[i_crop]
                orig = row[i_orig]
            except Exception:
                continue
                                                                                                                                # ключом сделаем stem без расширения (или полный относительный путь)
            stem = Path(crop).stem
            mapping[stem] = orig
    return mapping


                                                                                                                                # Чтение изображений

def read_any_image(path: Path) -> np.ndarray:
    """
    Возвращает grayscale uint8 изображение.
    - DICOM: pydicom.pixel_array -> нормализация [0..255], монохром1 -> инверсия
    - PNG/JPG: cv2.imread -> gray
    """
    suf = path.suffix.lower()
    if suf in (".dcm", ".dicom"):
        if not HAS_PYDICOM:
            raise RuntimeError("pydicom не установлен, не могу читать DICOM")
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        arr -= arr.min()
        m = float(arr.max())
        if m > 0:
            arr /= m
        img = (arr * 255.0).clip(0, 255).astype(np.uint8)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            img = 255 - img
        return img
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("не удалось прочитать изображение")
        return img



                                                                                                                                # Геометрия: угол между двумя линиями

def angle_between_lines(p1: Tuple[int, int], p2: Tuple[int, int],
                        p3: Tuple[int, int], p4: Tuple[int, int]) -> float:
    """
    Возвращает угол между прямыми p1-p2 и p3-p4 в градусах,
    острый (<= 90).
    """
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p4[0] - p3[0], p4[1] - p3[1])

    def norm(v):
        return math.sqrt(v[0]*v[0] + v[1]*v[1]) + 1e-12

    a1 = math.atan2(-v1[1], v1[0])                                                                                              # -y, чтобы верх был положительным
    a2 = math.atan2(-v2[1], v2[0])
    ang = abs((a1 - a2) * 180.0 / math.pi)
    if ang > 180:
        ang = 360 - ang
    if ang > 90:
        ang = 180 - ang
    return float(ang)


                                                                                                                                # Рисование и масштаб

def fit_to_screen(img: np.ndarray, max_w: int = 1600, max_h: int = 1000) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img2 = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        return img2, scale
    return img, 1.0


def draw_overlay(base_gray: np.ndarray,
                 points: List[Tuple[int, int]],
                 angle: Optional[float] = None,
                 info: str = "") -> np.ndarray:
    """
    Рисует точки/линии и подпись угла на копии изображения.
    points: [p1,p2,p3,p4] в координатах base_gray
    """
    bg = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)

                                                                                                                                # инструкция
    cv2.rectangle(bg, (0, 0), (bg.shape[1], 40), (0, 0, 0), -1)
    txt = "ЛКМ: ставь 4 точки | R: сброс | S: сохранить | N: след. | B: назад | O: оверлей | Q/ESC: выход"
    cv2.putText(bg, txt, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)

                                                                                                                                # инфо (имя файла)
    if info:
        cv2.putText(bg, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 230, 180), 2, cv2.LINE_AA)

    col_pts = (0, 215, 255)
    col_l1 = (80, 220, 60)
    col_l2 = (0, 0, 220)

                                                                                                                                # точки
    for i, p in enumerate(points):
        cv2.circle(bg, p, 5, col_pts, -1)
        cv2.putText(bg, f"P{i+1}", (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_pts, 1, cv2.LINE_AA)

                                                                                                                                # линии
    if len(points) >= 2:
        cv2.line(bg, points[0], points[1], col_l1, 2, cv2.LINE_AA)
    if len(points) >= 4:
        cv2.line(bg, points[2], points[3], col_l2, 2, cv2.LINE_AA)
        if angle is not None:
            cv2.putText(bg, f"Cobb (manual): {angle:.1f} deg",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return bg


                                                                                                                                # CSV helpers

def load_labels_csv(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Возвращает словарь: file -> {true_angle_deg, notes}
    """
    if not csv_path.exists():
        return {}
    data: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            file = row.get("file", "").strip()
            if not file:
                continue
            data[file] = {
                "true_angle_deg": row.get("true_angle_deg", "").strip(),
                "notes": row.get("notes", "").strip(),
            }
    return data


def save_labels_csv(csv_path: Path, rows: Dict[str, Dict[str, str]]) -> None:
    """
    Перезаписывает labels.csv из словаря.
    """
    fieldnames = ["file", "true_angle_deg", "notes"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for fpath, cols in rows.items():
            wr.writerow({
                "file": fpath,
                "true_angle_deg": cols.get("true_angle_deg", ""),
                "notes": cols.get("notes", ""),
            })


                                                                                                                                # Основной цикл

def main():
    ap = argparse.ArgumentParser("Label Cobb angle with 4 clicks")
    ap.add_argument("input", type=str, help="Папка с DICOM/PNG/JPG")
    ap.add_argument("--labels", type=str, default="labels.csv", help="CSV для записи результатов")
    ap.add_argument("--overlays", type=str, default="overlays", help="Папка для PNG-оверлеев")
    ap.add_argument("--manifest", type=str, default="to_label_manifest.csv",
                    help="Если размечаем кропы: csv для маппинга crop->оригинал")
    args = ap.parse_args()

    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        print(f"[ERR] Папка не найдена: {root}")
        return

    files = list_images(root)
    if not files:
        print("[ERR] Файлов не найдено")
        return

    overlays_dir = Path(args.overlays)
    overlays_dir.mkdir(parents=True, exist_ok=True)

                                                                                                                                # карта для кропов -> оригинальный путь
    mapping = load_manifest(Path(args.manifest))

                                                                                                                                # подгружаем существующие метки (если есть)
    labels_path = Path(args.labels)
    rows = load_labels_csv(labels_path)

    idx = 0
    points: List[Tuple[int, int]] = []
    win = "Cobb Labeler"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def current_target_path(p: Path) -> str:
        """
        Что писать в labels.csv в колонку 'file'.
        Если есть manifest и это кроп, возьмём исходный файл.
        Иначе — полный путь к текущему p.
        """
        stem = p.stem
        if stem in mapping:
            return str(Path(mapping[stem]).resolve())
        return str(p.resolve())

    def on_mouse(event, x, y, flags, userdata):
        nonlocal points, show_img
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
            else:
                                                                                                                                # перезапуск точек с первой
                points = [(x, y)]
            redraw()

    def redraw():
        nonlocal show_img, scale, img, disp, points
                                                                                                                                # пересчёт точек в оригинальные координаты
        scaled_pts = [(int(p[0] / scale), int(p[1] / scale)) for p in points]
        ang = None
        if len(scaled_pts) == 4:
            ang = angle_between_lines(scaled_pts[0], scaled_pts[1],
                                      scaled_pts[2], scaled_pts[3])
        overlay = draw_overlay(img, scaled_pts, ang, info=str(files[idx].name))
        disp, scale2 = fit_to_screen(overlay)
        scale = 1.0 / scale2                                                                                                    # обратный (для событий мыши мы используем координаты дисплея)
        show_img = disp.copy()
        cv2.imshow(win, show_img)

    cv2.setMouseCallback(win, on_mouse)

    while True:
                                                                                                                                # загрузить изображение
        cur = files[idx]
        try:
            img = read_any_image(cur)
        except Exception as e:
            print(f"[skip] {cur.name}: {e}")
                                                                                                                                # перейти к следующему
            idx = (idx + 1) % len(files)
            continue

                                                                                                                                # масштабируем для экрана
        disp, scale2 = fit_to_screen(img)
        scale = 1.0 / scale2                                                                                                    # сохраним обратный масштаб для on_mouse
        points = []
        show_img = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        redraw()

        cv2.imshow(win, show_img)

                                                                                                                                # цикл на кадре
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or key in (ord('q'), ord('Q')):  # ESC/Q
                                                                                                                                # выход, сначала сохранить labels
                save_labels_csv(labels_path, rows)
                cv2.destroyAllWindows()
                print(f"[OK] Сохранено: {labels_path}")
                return

            elif key in (ord('r'), ord('R')):
                points = []
                redraw()

            elif key in (ord('b'), ord('B')):
                                                                                                                                # назад
                idx = (idx - 1) % len(files)
                break

            elif key in (ord('n'), ord('N'), ord(' ')):
                                                                                                                                # вперёд без сохранения
                idx = (idx + 1) % len(files)
                break

            elif key in (ord('o'), ord('O')):
                                                                                                                                # сохранить только оверлей, даже если угла нет
                scaled_pts = [(int(p[0] / scale), int(p[1] / scale)) for p in points]
                ang = None
                if len(scaled_pts) == 4:
                    ang = angle_between_lines(scaled_pts[0], scaled_pts[1],
                                              scaled_pts[2], scaled_pts[3])
                ov = draw_overlay(img, scaled_pts, ang, info=str(cur.name))
                out_png = overlays_dir / f"{cur.stem}_overlay.png"
                cv2.imwrite(str(out_png), ov)
                print(f"[overlay] {out_png}")

            elif key in (ord('s'), ord('S')):
                if len(points) != 4:
                    print("[warn] Нужно 4 точки: две на верхнем позвонке и две на нижнем.")
                    continue

                scaled_pts = [(int(p[0] / scale), int(p[1] / scale)) for p in points]
                ang = angle_between_lines(scaled_pts[0], scaled_pts[1],
                                          scaled_pts[2], scaled_pts[3])
                                                                                                                                # сохранить в labels
                canonical_path = current_target_path(cur)
                rows[canonical_path] = {
                    "true_angle_deg": f"{ang:.3f}",
                    "notes": "",
                }
                                                                                                                                # и сохранить оверлей
                ov = draw_overlay(img, scaled_pts, ang, info=str(cur.name))
                out_png = overlays_dir / f"{cur.stem}_overlay.png"
                cv2.imwrite(str(out_png), ov)
                print(f"[save] {cur.name}: {ang:.2f}°  -> labels.csv; overlay: {out_png}")

                                                                                                                                # перейти дальше
                idx = (idx + 1) % len(files)
                break


if __name__ == "__main__":
    main()
