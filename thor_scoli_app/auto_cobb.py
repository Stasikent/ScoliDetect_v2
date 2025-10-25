# thor_scoli_app/auto_cobb.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Tuple, Dict, Any

import numpy as np
import cv2

try:
    from scipy.signal import savgol_filter  # опционально
    _HAS_SG = True
except Exception:
    _HAS_SG = False


# -----------------------------
# ВСПОМОГАТЕЛЬНОЕ: сглаживание 1D
# -----------------------------
def _gauss1d_kernel(half: int, sigma: float) -> np.ndarray:
    x = np.arange(-half, half + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    k /= np.sum(k)
    return k.astype(np.float32)


def _smooth_1d(x: np.ndarray,
               sigma_frac: float = 0.012,
               savgol_win_frac: float = 0.022,
               savgol_poly: int = 3) -> np.ndarray:
    """
    Плавное сглаживание профиля без фазового сдвига.
    - Если доступен scipy: Savitzky–Golay (хорошо держит форму).
    - Иначе: гауссово свёрточное сглаживание.
    Параметры зашиты как доля высоты профиля.
    """
    n = len(x)
    x = x.astype(np.float32, copy=False)

    if n < 8:
        return x.copy()

    if _HAS_SG:
        win = max(5, int(round(n * savgol_win_frac)) | 1)  # нечётное
        win = min(win, n - (1 - n % 2))  # окно не больше длины
        if win <= savgol_poly:  # защита
            win = savgol_poly + 2 + (savgol_poly + 2) % 2
        return savgol_filter(x, window_length=win, polyorder=savgol_poly, mode="mirror").astype(np.float32)
    else:
        # гаусс: длина окна ~= 6*sigma
        sigma = max(1.0, float(n) * sigma_frac)
        half = int(max(2, round(3.0 * sigma)))
        k = _gauss1d_kernel(half, sigma)
        return np.convolve(x, k, mode="same").astype(np.float32)


# -----------------------------
# ROI и центрлайн
# -----------------------------
def _compute_roi_box(width: int, height: int, roi_margin: float) -> Tuple[int, int, int, int]:
    cx = width // 2
    half = max(20, int(round(width * float(roi_margin))))
    x0 = max(0, cx - half)
    x1 = min(width - 1, cx + half)
    y0, y1 = 0, height - 1
    return (x0, y0, x1, y1)


def _extract_centerline(gray01: np.ndarray,
                        roi_margin: float = 0.18,
                        alpha: float = 0.25) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """
    Возвращает (y, x, roi_box):
      - y: массив [0..H-1] (int32)
      - x: float32 столбец каждой точки центрлайна в координатах кадра
      - roi_box: (x0,y0,x1,y1)

    Энергия: E = 1 - |∇I| внутри центрального ROI.
    DP с малым штрафом `alpha` за шаги влево/вправо снижает «пилу».
    """
    g = gray01.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    g = np.clip(g, 0.0, 1.0)

    H, W = g.shape[:2]
    roi_box = _compute_roi_box(W, H, roi_margin)
    x0, y0, x1, y1 = roi_box
    roi = g[y0:y1 + 1, x0:x1 + 1]

    # лёгкое шумоподавление
    blur = cv2.GaussianBlur(roi, (0, 0), 1.3)

    # градиенты
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    mmin, mmax = float(np.min(grad)), float(np.max(grad))
    if not np.isfinite(mmin) or not np.isfinite(mmax) or (mmax - mmin) < 1e-6:
        y = np.arange(H, dtype=np.int32)
        x = np.full(H, (x0 + x1) // 2, dtype=np.float32)
        return y, x, roi_box

    E = 1.0 - (grad - mmin) / (mmax - mmin + 1e-6)

    h, w = E.shape
    C = np.zeros_like(E, dtype=np.float32)
    P = np.zeros((h, w), dtype=np.int16)  # -1,0,+1
    C[0, :] = E[0, :]

    # DP: штраф `alpha` за горизонтальные переходы
    for i in range(1, h):
        left = np.pad(C[i - 1, :], (1, 0), mode='edge')[:-1] + alpha
        mid = C[i - 1, :]
        right = np.pad(C[i - 1, :], (0, 1), mode='edge')[1:] + alpha
        stack = np.stack([left, mid, right], axis=0)  # (3, w)
        idx = np.argmin(stack, axis=0)                # 0/1/2
        C[i, :] = E[i, :] + stack[idx, np.arange(w)]
        P[i, :] = (idx.astype(np.int16) - 1)

    j = int(np.argmin(C[-1, :]))
    path_x = np.zeros(h, dtype=np.int32)
    path_x[-1] = j
    for i in range(h - 2, -1, -1):
        j = int(np.clip(j + P[i + 1, j], 0, w - 1))
        path_x[i] = j

    # координаты кадра
    y = np.arange(H, dtype=np.int32)
    x = (x0 + path_x).astype(np.float32)

    # итоговое сглаживание центрлайна (адаптивное)
    x = _smooth_1d(x)

    return y, x, roi_box


# -----------------------------
# COBB из центрлайна
# -----------------------------
def _cobb_from_centerline(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, float, int, int]:
    H = int(y[-1]) + 1
    dx = np.gradient(x.astype(np.float32))

    # сглаживаем производную чуть сильнее (устойчивее углы)
    if len(dx) > 7:
        dx = _smooth_1d(dx, sigma_frac=0.02, savgol_win_frac=0.035, savgol_poly=2)

    k = max(40, H // 12)
    top_slice = slice(0, min(k, len(dx)))
    bot_slice = slice(max(0, len(dx) - k), len(dx))

    top_tan = float(np.nanmedian(dx[top_slice]))
    bot_tan = float(np.nanmedian(dx[bot_slice]))

    top_deg = math.degrees(math.atan(top_tan))
    bot_deg = math.degrees(math.atan(bot_tan))
    angle = abs(top_deg - bot_deg)

    top_y = int(np.clip(int(np.median(y[top_slice])), 0, H - 1))
    bot_y = int(np.clip(int(np.median(y[bot_slice])), 0, H - 1))
    return float(angle), float(top_deg), float(bot_deg), top_y, bot_y


# -----------------------------
# ОВЕРЛЕЙ
# -----------------------------
def _draw_overlay(gray01: np.ndarray,
                  y: np.ndarray,
                  x: np.ndarray,
                  roi_box: Tuple[int, int, int, int],
                  angle_deg: float,
                  top_y: int,
                  bot_y: int) -> np.ndarray:
    g = gray01
    if g.max() <= 1.5:
        img = (np.clip(g, 0, 1) * 255.0).astype(np.uint8)
    else:
        img = g.astype(np.uint8)
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        bgr = img.copy()

    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = roi_box
    cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 215, 255), 2)

    # сверхдискретизация траектории (визуально более гладко)
    if len(x) >= 2:
        yy = np.linspace(0, len(y) - 1, num=min(5000, 4 * len(y)), dtype=np.float32)
        xx = np.interp(yy, np.arange(len(x), dtype=np.float32), x.astype(np.float32))
        pts = np.stack([xx, yy], axis=1).astype(np.int32)
    else:
        pts = np.stack([x, y], axis=1).astype(np.int32)

    cv2.polylines(bgr, [pts], isClosed=False, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    half = max(10, (x1 - x0) // 12)
    tx = int(np.clip(x[top_y], 0, W - 1))
    bx = int(np.clip(x[bot_y], 0, W - 1))
    cv2.line(bgr, (tx - half, top_y), (tx + half, top_y), (80, 230, 110), 6, cv2.LINE_AA)
    cv2.line(bgr, (bx - half, bot_y), (bx + half, bot_y), (40, 40, 230), 6, cv2.LINE_AA)

    label = f"Cobb: {angle_deg:.1f} deg"
    cv2.putText(bgr, label, (int(0.05 * W), int(0.06 * H)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(bgr, label, (int(0.05 * W), int(0.06 * H)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return bgr


# -----------------------------
# ПУБЛИЧНЫЙ API
# -----------------------------
def auto_cobb_from_gray(gray: np.ndarray,
                        roi_margin: float = 0.18) -> Dict[str, Any]:
    """
    Главная функция для CLI.
    Параметры:
      gray — 2D numpy-массив (uint8 0..255 или float 0..1).
      roi_margin — половина ширины центрального ROI как доля ширины кадра.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)

    p1, p99 = np.percentile(g, [1.0, 99.0])
    if p99 <= p1:
        p1, p99 = float(np.min(g)), float(np.max(g))
    if p99 > p1:
        g = (g - p1) / (p99 - p1)
    else:
        g = (g - np.min(g)) / (np.max(g) - np.min(g) + 1e-6)
    g = np.clip(g, 0.0, 1.0)

    y, x, roi_box = _extract_centerline(g, roi_margin=roi_margin, alpha=0.15)

    angle_deg, top_deg, bot_deg, top_y, bot_y = _cobb_from_centerline(y, x)

    overlay = _draw_overlay(g, y, x, roi_box, angle_deg, top_y, bot_y)

    return {
        'angle_deg': float(angle_deg),
        'roi_box': tuple(map(int, roi_box)),
        'centerline_x': x.astype(np.float32),
        'centerline_y': y.astype(np.int32),
        'overlay_bgr': overlay,
        'top_y': int(top_y),
        'bot_y': int(bot_y),
        'top_deg': float(top_deg),
        'bot_deg': float(bot_deg),
    }
