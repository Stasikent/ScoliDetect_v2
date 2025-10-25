# thor_scoli_app/auto_cobb.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import cv2
import numpy as np


@dataclass
class CobbResult:
    angle_deg: float
    overlay_bgr: Optional[np.ndarray]
    diagnostics: Dict[str, float]


# --------------------------- helpers ---------------------------
def _savgol_1d(y: np.ndarray, win: int = 15, poly: int = 2) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    win = max(5, int(win) | 1)               # нечётное
    k = win // 2
    yy = np.pad(y, (k, k), mode="edge")
    x = np.arange(-k, k + 1, dtype=np.float64)
    V = np.vander(x, N=poly + 1, increasing=True)
    A = np.linalg.pinv(V)
    w = A[0]
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = float(np.dot(w, yy[i:i + win]))
    return out


def _normalize_u8(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    g -= np.min(g)
    m = float(np.max(g))
    if m > 0:
        g /= m
    return (g * 255.0).clip(0, 255).astype(np.uint8)


# ------------------------ ROI search --------------------------
def _central_column_roi(gray: np.ndarray) -> Tuple[slice, slice]:
    """
    Определение ROI по максимуму вертикального градиента (более точно по центру позвоночника).
    """
    h, w = gray.shape
    cx1, cx2 = int(w * 0.25), int(w * 0.75)
    central = gray[:, cx1:cx2]

    # Контрастное выравнивание
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2 = clahe.apply(central)

    # Вертикальный градиент |dI/dy|, усреднённый по оси Y
    gy = np.abs(cv2.Sobel(g2, cv2.CV_32F, 0, 1, ksize=3))
    grad_profile = gy.mean(axis=0)

    # Немного сглаживаем и ищем максимум градиента — середина столба
    grad_profile = cv2.GaussianBlur(grad_profile, (0, 0), 9)
    x_rel = int(np.argmax(grad_profile))
    x_center = cx1 + x_rel

    # Ширина ROI адаптируется: чем слабее контраст, тем шире берём
    contrast = np.std(g2) / 255
    half = int(max(w * (0.08 + 0.04 * (1 - contrast)), 60))
    rx1 = max(0, x_center - half)
    rx2 = min(w, x_center + half)

    # Высота — центральные 85% (немного выше, чем раньше)
    ry1, ry2 = int(h * 0.07), int(h * 0.92)
    return slice(ry1, ry2), slice(rx1, rx2)



# ----------------- centerline & cobb angle --------------------
def _fill_nans_1d(x: np.ndarray) -> np.ndarray:
    """Заполнить NaN: интерполяция + ffill/bfill + константа по центру."""
    x = x.astype(np.float32, copy=True)
    h = len(x)
    valid = np.isfinite(x)
    if valid.sum() == 0:
        # всё пропало — вернём вертикаль по центру
        return np.full(h, float(np.nan), dtype=np.float32)

    yi = np.arange(h, dtype=np.float32)
    # интерполяция по валидным
    x[~valid] = np.interp(yi[~valid], yi[valid], x[valid])

    # ffill/bfill — на случай крайних NaN после интерполяции
    # (обычно не нужно, но оставим как страховку)
    isn = ~np.isfinite(x)
    if isn.any():
        # ffill
        last = np.nan
        for i in range(h):
            if np.isfinite(x[i]):
                last = x[i]
            elif np.isfinite(last):
                x[i] = last
        # bfill
        last = np.nan
        for i in range(h - 1, -1, -1):
            if np.isfinite(x[i]):
                last = x[i]
            elif np.isfinite(last):
                x[i] = last

    # если вдруг остались NaN — заменим средней
    if (~np.isfinite(x)).any():
        mean_val = float(np.nanmean(x))
        x = np.where(np.isfinite(x), x, mean_val).astype(np.float32)
    return x


def _centerline_from_edges(roi: np.ndarray, canny_low: int, canny_high: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = cv2.Canny(roi, canny_low, canny_high, L2gradient=True)
    h, w = edges.shape
    xs = np.full(h, np.nan, dtype=np.float32)

    v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8), iterations=1)

    for y in range(h):
        cols = np.where(v[y] > 0)[0]
        if cols.size < 8:
            continue
        q1, q9 = np.percentile(cols, [5, 95])
        left, right = int(q1), int(q9)
        if right - left < max(30, int(w * 0.15)):
            continue
        xs[y] = (left + right) * 0.5

    # если валидных точек очень мало — вернём как есть
    if np.isfinite(xs).sum() < h * 0.2:
        return xs, edges

    xs = _fill_nans_1d(xs)
    xs = _savgol_1d(xs, win=max(15, h // 20), poly=2).astype(np.float32)
    return xs, edges


def _cobb_from_centerline(xs: np.ndarray) -> Tuple[float, Tuple[int, int, float, float]]:
    h = len(xs)
    dx = np.gradient(xs)
    dx = _savgol_1d(dx, win=max(15, h // 25), poly=2)
    angles = np.degrees(np.arctan(dx))

    k = max(20, h // 12)
    abs_a = cv2.GaussianBlur(np.abs(angles).astype(np.float32), (0, 0), 5)
    top = int(np.argmax(abs_a[:h // 2]))
    bottom = int(np.argmax(abs_a[h // 2:])) + h // 2

    a1 = float(np.median(angles[max(0, top - k):min(h, top + k)]))
    a2 = float(np.median(angles[max(0, bottom - k):min(h, bottom + k)]))
    cobb = abs(a1 - a2)
    return float(cobb), (top, bottom, float(a1), float(a2))


# ---------------------------- main API ------------------------
def auto_cobb_from_gray(
    gray: np.ndarray,
    *,
    return_overlay: bool = True,
    canny: Tuple[int, int] | None = None,
    debug: bool = False,
    **_
) -> CobbResult:
    g = _normalize_u8(gray)

    sy, sx = _central_column_roi(g)
    roi = g[sy, sx].copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi = clahe.apply(roi)

    if canny is None:
        med = float(np.median(roi))
        low = int(max(10, min(180, 0.66 * med)))
        high = int(max(30, min(255, 1.33 * med)))
        canny = (low, high)

    xs, edges = _centerline_from_edges(roi, canny[0], canny[1])

    if not np.isfinite(xs).any():
        overlay = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if return_overlay else None
        if return_overlay:
            cv2.rectangle(overlay, (sx.start, sy.start), (sx.stop - 1, sy.stop - 1), (0, 215, 255), 2)
            cv2.putText(overlay, "Cobb: 0.0 deg (fallback)", (sx.start + 10, sy.start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return CobbResult(0.0, overlay, {"ok": 0.0})

    cobb, (iy1, iy2, a1, a2) = _cobb_from_centerline(xs)

    overlay = None
    if return_overlay:
        overlay = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(overlay, (sx.start, sy.start), (sx.stop - 1, sy.stop - 1), (0, 215, 255), 2)

        # Рисуем центральную линию, ПРОПУСКАЯ NaN
        for i in range(sy.start, sy.stop):
            x_val = xs[i - sy.start]
            if not np.isfinite(x_val):
                continue
            x = int(round(float(x_val))) + sx.start
            overlay = cv2.circle(overlay, (x, i), 1, (255, 255, 255), -1)

        # Точки измерения: аккуратно обрабатываем NaN
        x1v = xs[iy1] if np.isfinite(xs[iy1]) else np.nanmean(xs)
        x2v = xs[iy2] if np.isfinite(xs[iy2]) else np.nanmean(xs)
        if not np.isfinite(x1v) or not np.isfinite(x2v):
            x1v = x2v = (sx.stop - sx.start) / 2.0

        x1 = int(round(float(x1v))) + sx.start
        x2 = int(round(float(x2v))) + sx.start
        y1 = iy1 + sy.start
        y2 = iy2 + sy.start
        cv2.line(overlay, (x1, y1 - 40), (x1, y1 + 40), (80, 220, 60), 3)
        cv2.line(overlay, (x2, y2 - 40), (x2, y2 + 40), (0, 0, 220), 3)

        txt = f"Cobb: {cobb:.1f} deg"
        cv2.putText(overlay, txt, (sx.start + 12, sy.start + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if debug:
            dbg = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            dbg = cv2.resize(dbg, (sx.stop - sx.start, sy.stop - sy.start))
            overlay[sy, sx] = cv2.addWeighted(overlay[sy, sx], 0.8, dbg, 0.4, 0)

    diagnostics = {
        "roi_x1": float(sx.start), "roi_x2": float(sx.stop),
        "roi_y1": float(sy.start), "roi_y2": float(sy.stop),
        "a1_deg": float(a1), "a2_deg": float(a2),
        "ok": 1.0
    }
    return CobbResult(float(cobb), overlay, diagnostics)
