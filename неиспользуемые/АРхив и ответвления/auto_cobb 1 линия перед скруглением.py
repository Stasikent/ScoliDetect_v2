# thor_scoli_app/auto_cobb.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Tuple, Dict, Any

import numpy as np
import cv2


# -----------------------------
# ВСПОМОГАТЕЛЬНОЕ: ROI и центрлайн
# -----------------------------
def _compute_roi_box(width: int, height: int, roi_margin: float) -> Tuple[int, int, int, int]:
    """
    Центральный вертикальный ROI вокруг позвоночника.
    roi_margin — доля ширины кадра от центра влево/вправо (0.12..0.25 обычно).
    """
    cx = width // 2
    half = max(20, int(round(width * float(roi_margin))))
    x0 = max(0, cx - half)
    x1 = min(width - 1, cx + half)
    y0, y1 = 0, height - 1
    return (x0, y0, x1, y1)


def _extract_centerline(gray01: np.ndarray, roi_margin: float = 0.18):
    """
    Возвращает (y, x, roi_box):
      - y: массив [0..H-1] (int32)
      - x: float32 столбец каждой точки центрлайна в координатах КАДРА
      - roi_box: (x0,y0,x1,y1)

    Алгоритм: seam-carving по "энергии" E = 1 - |∇I| внутри центрального ROI.
    """
    g = gray01.astype(np.float32)
    if g.max() > 1.5:  # на всякий случай: если пришли 0..255
        g = g / 255.0
    g = np.clip(g, 0.0, 1.0)

    H, W = g.shape[:2]
    roi_box = _compute_roi_box(W, H, roi_margin)
    x0, y0, x1, y1 = roi_box
    roi = g[y0:y1 + 1, x0:x1 + 1]

    # лёгкое шумоподавление
    blur = cv2.GaussianBlur(roi, (0, 0), 1.2)

    # градиенты
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    # нормировка
    mmin, mmax = float(np.min(grad)), float(np.max(grad))
    if not np.isfinite(mmin) or not np.isfinite(mmax) or (mmax - mmin) < 1e-6:
        # на случай "пустого" изображения — ровно середина ROI
        y = np.arange(H, dtype=np.int32)
        x = np.full(H, (x0 + x1) // 2, dtype=np.float32)
        return y, x, roi_box

    grad01 = (grad - mmin) / (mmax - mmin + 1e-6)
    E = 1.0 - grad01  # хотим идти по контрастным границам (минимизируем E)

    h, w = E.shape
    C = np.zeros_like(E, dtype=np.float32)  # накопленная стоимость
    P = np.zeros_like(E, dtype=np.int16)    # backpointers: -1,0,+1

    C[0, :] = E[0, :]
    P[0, :] = 0

    # Динамическое программирование с шагами -1/0/+1 по x
    for i in range(1, h):
        left = np.pad(C[i - 1, :], (1, 0), mode='edge')[:-1]
        mid = C[i - 1, :]
        right = np.pad(C[i - 1, :], (0, 1), mode='edge')[1:]
        stack = np.stack([left, mid, right], axis=0)  # (3, w)
        idx = np.argmin(stack, axis=0)                # 0/1/2
        C[i, :] = E[i, :] + stack[idx, np.arange(w)]
        P[i, :] = (idx.astype(np.int16) - 1)          # -1/0/+1

    # заканчиваем в точке с минимальной стоимостью внизу
    j = int(np.argmin(C[-1, :]))
    path_x = np.zeros(h, dtype=np.int32)
    path_x[-1] = j
    for i in range(h - 2, -1, -1):
        j = int(np.clip(j + P[i + 1, j], 0, w - 1))
        path_x[i] = j

    # координаты полного кадра
    y = np.arange(H, dtype=np.int32)
    x = (x0 + path_x).astype(np.float32)

    # сгладим «лесенку» вдоль y
    if len(x) > 7:
        ker = np.ones(9, np.float32) / 9.0
        x = np.convolve(x, ker, mode="same").astype(np.float32)

    return y, x, roi_box


# -----------------------------
# COBB из центрлайна
# -----------------------------
def _cobb_from_centerline(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, float, int, int]:
    """
    Расчёт Cobb как разности углов касательных центрлайна вверху и внизу.
    Возвращает:
      angle_deg, top_deg, bot_deg, top_y, bot_y
    """
    H = int(y[-1]) + 1
    # производная dx/dy (тангенс угла к вертикали)
    dx = np.gradient(x.astype(np.float32))
    # сгладим
    if len(dx) > 7:
        ker = np.ones(11, np.float32) / 11.0
        dx = np.convolve(dx, ker, mode="same")

    # окна сверху/снизу
    k = max(40, H // 12)  # около 8–10% высоты
    top_slice = slice(0, min(k, len(dx)))
    bot_slice = slice(max(0, len(dx) - k), len(dx))

    top_tan = float(np.nanmedian(dx[top_slice]))
    bot_tan = float(np.nanmedian(dx[bot_slice]))

    # в градусах: угол к вертикали = arctan(tan) в градусах
    top_deg = math.degrees(math.atan(top_tan))
    bot_deg = math.degrees(math.atan(bot_tan))
    angle = abs(top_deg - bot_deg)

    # позиции для рисования горизонтальных «шкал»
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
    """
    Рисует:
      - белую линию центрлайна,
      - жёлтую рамку ROI,
      - зелёную/красную короткие метки для верх/низ,
      - надпись "Cobb: NN.N deg".
    Возвращает BGR-изображение uint8.
    """
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
    # ROI рамка
    x0, y0, x1, y1 = roi_box
    cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 215, 255), 2)  # жёлтый

    # центрлайн
    pts = np.stack([x, y], axis=1).astype(np.int32)
    for i in range(1, len(pts)):
        cv2.line(bgr, tuple(pts[i - 1]), tuple(pts[i]), (255, 255, 255), 2, cv2.LINE_AA)

    # верх/низ — маленькие поперечные риски
    half = max(10, (x1 - x0) // 12)
    # top
    tx = int(np.clip(x[top_y], 0, W - 1))
    cv2.line(bgr, (tx - half, top_y), (tx + half, top_y), (80, 230, 110), 6, cv2.LINE_AA)
    # bottom
    bx = int(np.clip(x[bot_y], 0, W - 1))
    cv2.line(bgr, (bx - half, bot_y), (bx + half, bot_y), (40, 40, 230), 6, cv2.LINE_AA)

    # текст
    label = f"Cobb: {angle_deg:.1f} deg"
    # белый с лёгкой тенью
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

    Возвращает словарь:
      {
        'angle_deg': float,
        'roi_box': (x0,y0,x1,y1),
        'centerline_x': np.ndarray(float32, shape=(H,)),
        'centerline_y': np.ndarray(int32,   shape=(H,)),
        'overlay_bgr': np.ndarray(uint8, shape=(H,W,3)),
      }
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    # приведение к 0..1 по робастной нормировке
    p1, p99 = np.percentile(g, [1.0, 99.0])
    if p99 <= p1:
        p1, p99 = float(np.min(g)), float(np.max(g))
    if p99 > p1:
        g = (g - p1) / (p99 - p1)
    else:
        g = (g - np.min(g)) / (np.max(g) - np.min(g) + 1e-6)
    g = np.clip(g, 0.0, 1.0)

    # центрлайн
    y, x, roi_box = _extract_centerline(g, roi_margin=roi_margin)

    # угол Cobb
    angle_deg, top_deg, bot_deg, top_y, bot_y = _cobb_from_centerline(y, x)

    # оверлей
    overlay = _draw_overlay(g, y, x, roi_box, angle_deg, top_y, bot_y)

    return {
        'angle_deg': float(angle_deg),
        'roi_box': tuple(map(int, roi_box)),
        'centerline_x': x.astype(np.float32),
        'centerline_y': y.astype(np.int32),
        'overlay_bgr': overlay,
        # на всякий случай добавим вспомогательные поля — вдруг пригодятся
        'top_y': int(top_y),
        'bot_y': int(bot_y),
        'top_deg': float(top_deg),
        'bot_deg': float(bot_deg),
    }
