# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CobbResult:
    angle_deg: float
    overlay_bgr: Optional[np.ndarray]
    diagnostics: Dict[str, float]


# --------------------
# ВСПОМОГАТЕЛЬНЫЕ
# --------------------

def _savgol_1d(y: np.ndarray, win: int = 15, poly: int = 2) -> np.ndarray:
    """Простейшее сглаживание наподобие Savitzky–Golay без SciPy."""
    win = max(5, int(win) | 1)  # нечетное >=5
    k = win // 2
    yy = np.pad(y.astype(np.float64), (k, k), mode="edge")
    out = np.empty_like(y, dtype=np.float64)

    # матрица Вандермонда
    x = np.arange(-k, k + 1, dtype=np.float64)
    V = np.vander(x, N=poly + 1, increasing=True)  # [1, x, x^2, ...]
    # коэффициенты оценки значения в центре окна
    A = np.linalg.pinv(V)  # (poly+1) x win
    w = A[0]               # веса для усреднения

    for i in range(len(y)):
        out[i] = float(np.dot(w, yy[i:i + win]))
    return out


def _normalize_u8(gray: np.ndarray) -> np.ndarray:
    """Нормализация входа в uint8 [0..255]."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    g -= float(g.min())
    mx = float(g.max())
    if mx > 0:
        g /= mx
    g = (g * 255.0).clip(0, 255).astype(np.uint8)
    return g


def _central_roi(gray_u8: np.ndarray) -> Tuple[slice, slice]:
    """
    Грубая локализация грудного отдела:
    - выравниваем контраст (CLAHE)
    - берём горизонтальную «инвертированную» проекцию, ищем «столб»
    - ограничиваем по ширине и по высоте (центральные 70%)
    """
    h, w = gray_u8.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2 = clahe.apply(gray_u8)

    inv = cv2.bitwise_not(g2)
    col = inv.mean(axis=0)
    col = cv2.GaussianBlur(col.astype(np.float32), (0, 0), 5)
    x_peak = int(np.argmax(col))

    half = max(int(w * 0.18), 100)  # чуть шире прежнего
    x1 = max(0, x_peak - half)
    x2 = min(w, x_peak + half)

    y1 = int(h * 0.15)
    y2 = int(h * 0.85)
    return slice(y1, y2), slice(x1, x2)


def _gradient_map(roi: np.ndarray) -> np.ndarray:
    """Карта силы края (модуль градиента)."""
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # нормировка в [0..1]
    m = float(mag.max()) if mag.size else 0.0
    if m > 1e-6:
        mag /= m
    return mag


def _find_dual_edges(roi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Поиск левой/правой границы позвонкового столба в каждой строке ROI.
    Возвращает (xL, xR, center) длины h (NaN где не найдено).
    """
    h, w = roi.shape
    mag = _gradient_map(roi)
    # «центр» берём из инвертированной яркости (более тёмная кость → светлая инверсия)
    inv = cv2.bitwise_not(roi).astype(np.float32)
    center_col = np.argmax(cv2.GaussianBlur(inv, (0, 0), 5), axis=1).astype(np.int32)

    xL = np.full(h, np.nan, dtype=np.float32)
    xR = np.full(h, np.nan, dtype=np.float32)

    # допустимая половина ширины столба (в пикселях)
    half_band = max(int(w * 0.45), 70)   # шире, чтобы не промахнуться
    min_span = max(int(w * 0.25), 40)    # минимальная ширина между краями

    # построчно: ищем локальные максимумы градиента слева и справа от «центра»
    for y in range(h):
        c = int(center_col[y])
        left_zone = mag[y, max(0, c - half_band):c]
        right_zone = mag[y, c:min(w, c + half_band)]

        if left_zone.size >= 10:
            # пик сильного градиента = левый край
            xi = int(np.argmax(left_zone))
            xL[y] = max(0, c - half_band) + xi

        if right_zone.size >= 10:
            xi = int(np.argmax(right_zone))
            xR[y] = c + xi

        # проверка на ширину
        if np.isfinite(xL[y]) and np.isfinite(xR[y]):
            if (xR[y] - xL[y]) < min_span:
                # отбрасываем как нестабильную строку
                xL[y] = np.nan
                xR[y] = np.nan

    # заполнение пробелов интерполяцией и сглаживание
    yi = np.arange(h, dtype=np.float32)
    for arr in (xL, xR):
        ok = np.isfinite(arr)
        if ok.sum() >= 5:
            arr[~ok] = np.interp(yi[~ok], yi[ok], arr[ok])
            arr[:] = _savgol_1d(arr, win=max(15, h // 25), poly=2).astype(np.float32)

    center = (xL + xR) * 0.5
    return xL, xR, center


def _single_centerline(roi: np.ndarray) -> np.ndarray:
    """Фолбэк-метод: один центр из бинаризованного столба (устойчиво, но грубо)."""
    h, w = roi.shape
    inv = cv2.bitwise_not(roi)
    col = inv.astype(np.float32)
    col = cv2.GaussianBlur(col, (0, 0), 3)
    center = np.argmax(col, axis=1).astype(np.float32)

    ok = np.ones(h, dtype=bool)
    center[~ok] = np.nan
    # сглаживаем
    center = _savgol_1d(center, win=max(15, h // 25), poly=2).astype(np.float32)
    return center


def _cobb_from_centerline(xs: np.ndarray) -> Tuple[float, Tuple[int, int, float, float]]:
    """
    Угол Кобба по центр-линии:
    - производная → угловой профиль
    - два «экстремума» в верхней/нижней половинах
    """
    h = len(xs)
    dx = np.gradient(xs)
    dx = _savgol_1d(dx, win=max(15, h // 25), poly=2)
    angles = np.degrees(np.arctan(dx)).astype(np.float32)

    k = max(20, h // 12)
    # верхний экстремум
    top = int(np.nanargmax(np.abs(angles[:h // 2])))
    # нижний экстремум
    bottom = int(np.nanargmax(np.abs(angles[h // 2:]))) + h // 2

    a1 = float(np.nanmedian(angles[max(0, top - k):min(h, top + k)]))
    a2 = float(np.nanmedian(angles[max(0, bottom - k):min(h, bottom + k)]))
    cobb = abs(a1 - a2)
    return float(cobb), (top, bottom, a1, a2)


# --------------------
# ГЛАВНАЯ ФУНКЦИЯ
# --------------------

def auto_cobb_from_gray(
    gray: np.ndarray,
    *,
    edge_mode: str = "dual",         # "dual" (две границы) или "center"
    return_overlay: bool = True,
    debug: bool = False,
) -> CobbResult:
    """
    Автоматическое измерение угла Кобба на AP-снимке (грудной отдел).
    Возвращает CobbResult(angle_deg, overlay_bgr, diagnostics)
    """
    g = _normalize_u8(gray)

    # ROI
    sy, sx = _central_roi(g)
    roi = g[sy, sx].copy()
    roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)

    # Попытка «dual-edges»
    used_dual = False
    if edge_mode.lower() == "dual":
        xL, xR, xc = _find_dual_edges(roi)
        # валидность: процент строк с валидными краями и адекватная ширина
        widths = (xR - xL)
        valid = np.isfinite(widths)
        ok_ratio = float(np.count_nonzero(valid)) / float(len(widths) + 1e-6)

        if ok_ratio >= 0.5 and np.nanmedian(widths) > 20:
            xs = xc
            used_dual = True
        else:
            xs = _single_centerline(roi)
    else:
        xs = _single_centerline(roi)

    # Если вообще пусто — отдаём 0° и только рамку
    if not np.isfinite(xs).any():
        overlay = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if return_overlay else None
        if return_overlay:
            cv2.rectangle(overlay, (sx.start, sy.start), (sx.stop - 1, sy.stop - 1), (0, 215, 255), 2)
            cv2.putText(overlay, "Cobb: 0.0 deg (no centerline)", (sx.start + 10, sy.start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return CobbResult(0.0, overlay, {"ok": 0.0, "used_dual": 0.0})

    cobb, (iy1, iy2, a1, a2) = _cobb_from_centerline(xs)

    # Оверлей
    overlay = None
    if return_overlay:
        overlay = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        # ROI
        cv2.rectangle(overlay, (sx.start, sy.start), (sx.stop - 1, sy.stop - 1), (0, 215, 255), 2)
        # Рисуем центр-линию
        for i in range(sy.start, sy.stop):
            x = int(float(xs[i - sy.start])) + sx.start
            overlay = cv2.circle(overlay, (x, i), 1, (255, 255, 255), -1)

        # Если были посчитаны 2 края — нарисуем их
        if edge_mode.lower() == "dual" and used_dual:
            xL, xR, _ = _find_dual_edges(roi)  # повторим быстро для рисовки (дешёво)
            for i in range(sy.start, sy.stop):
                xi = i - sy.start
                if 0 <= xi < len(xL) and np.isfinite(xL[xi]) and np.isfinite(xR[xi]):
                    overlay = cv2.circle(overlay, (int(xL[xi]) + sx.start, i), 1, (80, 220, 60), -1)
                    overlay = cv2.circle(overlay, (int(xR[xi]) + sx.start, i), 1, (0, 0, 220), -1)

        # «уровни» измерения
        x1 = int(xs[iy1]) + sx.start
        x2 = int(xs[iy2]) + sx.start
        y1 = iy1 + sy.start
        y2 = iy2 + sy.start
        cv2.line(overlay, (x1, max(0, y1 - 50)), (x1, min(overlay.shape[0]-1, y1 + 50)), (80, 220, 60), 3)
        cv2.line(overlay, (x2, max(0, y2 - 50)), (x2, min(overlay.shape[0]-1, y2 + 50)), (0, 0, 220), 3)

        txt = f"Cobb: {cobb:.1f} deg"
        cv2.putText(overlay, txt, (sx.start + 12, sy.start + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if debug:
            # добавим полупрозрачный градиент внутри ROI
            mag = (_gradient_map(roi) * 255).astype(np.uint8)
            mag = cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
            patch = overlay[sy, sx].copy()
            overlay[sy, sx] = cv2.addWeighted(patch, 0.75, mag, 0.25, 0)

    diagnostics = {
        "ok": 1.0,
        "used_dual": 1.0 if used_dual else 0.0,
        "roi_x1": float(sx.start), "roi_x2": float(sx.stop),
        "roi_y1": float(sy.start), "roi_y2": float(sy.stop),
        "a1_deg": float(a1), "a2_deg": float(a2),
        "cobb_deg": float(cobb),
    }
    return CobbResult(float(cobb), overlay, diagnostics)
