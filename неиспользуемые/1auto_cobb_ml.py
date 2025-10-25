# auto_cobb_ml.py
# -*- coding: utf-8 -*-
# реализует ML-модифицированный авто-измеритель угла Cobb.
# DP-трассировка идёт не только по физическому контрасту, но и по обученной вероятности положения позвоночного канала.
# В результате траектория центрлайна становится более устойчивой к шуму, костям рёбер и затемнениям, если модель обучена на правильных данных.
from __future__ import annotations
import math, joblib
from typing import Dict, Any, Tuple
import numpy as np, cv2

from auto_cobb import _smooth_1d, _compute_roi_box, _draw_overlay, _cobb_from_centerline 


def _roi_features(gray_u8: np.ndarray, box: Tuple[int,int,int,int]) -> np.ndarray:                                          # Создаёт признаки (features) для каждого пикселя внутри ROI.
    x0,y0,x1,y1 = box                                                                                                       # Вырезает ROI по координатам (x0,y0,x1,y1).
    roi = gray_u8[y0:y1+1, x0:x1+1].astype(np.float32) / 255.0
    h,w = roi.shape                                                                                                         # Преобразует в float 0–1, делает Gaussian Blur.
    blur = cv2.GaussianBlur(roi, (0,0), 1.2)                                                                                # Считает локальные фильтры:
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)                                                                         # градиенты (Sobel)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)                                                                                             # величина градиента
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)                                                                          # лапласиан (второй производной контраст)
    k = max(3, int(round(min(h,w) * 0.01)) | 1)
    mean = cv2.blur(roi, (k,k))                                                                                             # mean и std - локальные среднее и стандартное отклонение яркости (через блур)
    sqr = cv2.blur(roi*roi, (k,k))
    var = np.maximum(sqr - mean*mean, 0.0)
    std = np.sqrt(var)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx - (w-1)/2.0) / max(1.0, (w-1)/2.0)
    yy = (yy - (h-1)/2.0) / max(1.0, (h-1)/2.0)
    feats = np.stack([roi, blur, gx, gy, mag, lap, mean, std, xx, yy], axis=-1)
    return feats                                                                                                            # (h,w,F)


def _ml_energy(gray01: np.ndarray,                                                                                          # Формирует энергетическую карту E(y,x) для DP-поиска центрлайна.
               roi_box: Tuple[int,int,int,int],
               model: Dict[str,Any],
               w_grad: float = 0.6,
               w_ml: float = 0.4) -> np.ndarray:
    x0,y0,x1,y1 = roi_box
    roi = gray01[y0:y1+1, x0:x1+1].astype(np.float32)

                                                                                                                            # нормализованный градиент
    g = cv2.GaussianBlur(roi, (0,0), 1.3)
    gx = cv2.Sobel(g, cv2.CV_32F, 1,0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0,1, ksize=3)
    grad = cv2.magnitude(gx,gy)
    mn, mx = float(grad.min()), float(grad.max())
    grad_n = (grad - mn) / (mx - mn + 1e-6)
    e_grad = 1.0 - grad_n

                                                                                                                            # ML вероятность
    gray_u8 = (np.clip(gray01*255.0, 0, 255)).astype(np.uint8) if gray01.max()<=1.5 else gray01.astype(np.uint8)
    feats = _roi_features(gray_u8, roi_box)                                                                                 # (h,w,F)
    h,w,F = feats.shape
    X = feats.reshape(-1, F).astype(np.float32)

    clf = model["clf"]
                                                                                                                            # батч-предсказание (экономит память)
    proba = np.zeros((X.shape[0],), dtype=np.float32)                                                                       
    bs = 200_000
    for s in range(0, X.shape[0], bs):
        e = min(s+bs, X.shape[0])
        pp = clf.predict_proba(X[s:e])[:,1]
        proba[s:e] = pp.astype(np.float32)
    p = proba.reshape(h, w)                                                                                                 # Получается карта вероятностей p(y,x) = «вероятность, что этот пиксель — часть границы позвоночника»
                                                                                                                            # Из неё делает ML-энергию
    e_ml = 1.0 - p                                                                                                          # хотим притягивать путь туда, где p высокое

                                                                                                                            # Линейно объединяет обе
    E = np.clip(w_grad * e_grad + w_ml * e_ml, 0.0, 1.0).astype(np.float32)                                                 # Возвращает итоговую энергию E (чем меньше — тем «учше путь)
    return E


def auto_cobb_from_gray_ml(gray: np.ndarray,                                                                                # Главная функция. Повторяет общую структуру auto_cobb_from_gray
                           roi_margin: float = 0.18,
                           ml_model_path: str|None = None,
                           alpha: float = 0.15,
                           w_grad: float = 0.6,
                           w_ml: float = 0.4) -> Dict[str,Any]:
    """
    Авто-Cobb с опциональным ML.
    - gray: 2D numpy (uint8 0..255 или float 0..1)
    - ml_model_path: путь к .joblib, сохранённому train_border_ml.py (опционально).
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
                                                                                                                            # robust normalize
    p1, p99 = np.percentile(g, [1.0, 99.0])
    if p99 <= p1:
        p1, p99 = float(np.min(g)), float(np.max(g))
    g = (g - p1) / (p99 - p1 + 1e-6)
    g = np.clip(g, 0.0, 1.0)

    H,W = g.shape
    roi_box = _compute_roi_box(W, H, roi_margin)

                                                                                                                            # энергия
    if ml_model_path:
        model = joblib.load(ml_model_path)
        E = _ml_energy(g, roi_box, model, w_grad=w_grad, w_ml=w_ml)
    else:
                                                                                                                            # если модели нет — используем только градиент (падение к auto_cobb)
        x0,y0,x1,y1 = roi_box
        roi = g[y0:y1+1, x0:x1+1]
        blur = cv2.GaussianBlur(roi, (0,0), 1.3)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1,0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0,1, ksize=3)
        grad = cv2.magnitude(gx,gy)
        mn, mx = float(grad.min()), float(grad.max())
        E = 1.0 - (grad - mn) / (mx - mn + 1e-6)

                                                                                                                            # DP в ROI (как в твоей версии с «антизигзагом»)
    x0,y0,x1,y1 = roi_box
    h,w = E.shape
    C = np.zeros_like(E, dtype=np.float32)
    P = np.zeros((h, w), dtype=np.int16)
    C[0,:] = E[0,:]
    for i in range(1, h):
        left = np.pad(C[i-1,:], (1,0), mode='edge')[:-1] + alpha
        mid  = C[i-1,:]
        right= np.pad(C[i-1,:], (0,1), mode='edge')[1:] + alpha
        stack = np.stack([left, mid, right], axis=0)
        idx = np.argmin(stack, axis=0)
        C[i,:] = E[i,:] + stack[idx, np.arange(w)]
        P[i,:] = (idx.astype(np.int16) - 1)

    j = int(np.argmin(C[-1,:]))
    path_x = np.zeros(h, dtype=np.int32); path_x[-1] = j
    for i in range(h-2, -1, -1):
        j = int(np.clip(j + P[i+1, j], 0, w-1))
        path_x[i] = j

    y = np.arange(H, dtype=np.int32)
    x = (x0 + path_x).astype(np.float32)
    x = _smooth_1d(x)                                                                                                       # финишное сглаживание

    angle_deg, top_deg, bot_deg, top_y, bot_y = _cobb_from_centerline(y, x)
    overlay = _draw_overlay(g, y, x, roi_box, angle_deg, top_y, bot_y)

    return {
        "angle_deg": float(angle_deg),
        "roi_box": tuple(map(int, roi_box)),
        "centerline_x": x.astype(np.float32),
        "centerline_y": y.astype(np.int32),
        "overlay_bgr": overlay,
        "top_y": int(top_y),
        "bot_y": int(bot_y),
        "top_deg": float(top_deg),
        "bot_deg": float(bot_deg),
    }
