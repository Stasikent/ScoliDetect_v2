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


# ---------- small helpers ----------

def __smooth1d(a: np.ndarray, sigma: float) -> np.ndarray:
    """Безопасное 1D сглаживание через 2D Gaussian (нет 'object too deep')."""
    a = np.asarray(a, dtype=np.float32).reshape(1, -1)
    out = cv2.GaussianBlur(a, (0, 0), max(0.1, float(sigma)))
    return out.ravel()


def _savgol_1d(y: np.ndarray, win: int = 15, poly: int = 2) -> np.ndarray:
    win = max(5, int(win) | 1)  # нечётное
    k = win // 2
    yy = np.pad(np.asarray(y, dtype=np.float64), (k, k), mode="edge")
    x = np.arange(-k, k + 1, dtype=np.float64)
    V = np.vander(x, N=poly + 1, increasing=True)
    A = np.linalg.pinv(V)
    w = A[0]
    out = np.empty_like(y, dtype=np.float64)
    for i in range(len(y)):
        out[i] = float(np.dot(w, yy[i:i + win]))
    return out


def _normalize_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    g -= g.min()
    m = g.max()
    if m > 0:
        g /= m
    return (g * 255.0).clip(0, 255).astype(np.uint8)


# ---------- ROI finder ----------

def _locate_spine_roi(
    gray: np.ndarray,
    center_prior_scale: float = 0.22,
    roi_rel_width: float = 0.28,
    top_cut: float = 0.12,
    bottom_cut: float = 0.88,
) -> Tuple[slice, slice]:
    h, w = gray.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    g = cv2.GaussianBlur(g, (0, 0), 1.4)

    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    col = np.abs(gx).mean(axis=0).astype(np.float32)
    col = __smooth1d(col, 7)

    win = int(np.clip(0.07 * w, 50, 140))
    pad = win + 1
    prof = np.pad(col, (pad, pad), mode="reflect")

    xs = np.arange(w)
    sym = np.zeros(w, dtype=np.float32)
    for i, x in enumerate(xs):
        L = prof[pad + x - win : pad + x]
        R = prof[pad + x + 1 : pad + x + 1 + win][::-1]
        den = float(np.linalg.norm(L) * np.linalg.norm(R)) + 1e-6
        sym[i] = float(np.dot(L, R)) / den
    sym = __smooth1d(sym, 9)

    cen = (w - 1) / 2.0
    center_prior = np.exp(-0.5 * ((np.arange(w, dtype=np.float32) - cen) / (center_prior_scale * w)) ** 2)

    col /= (col.max() + 1e-6)
    sym /= (sym.max() + 1e-6)
    score = 0.55 * sym + 0.25 * col + 0.35 * center_prior
    x_peak = int(score.argmax())

    half = max(int(w * (roi_rel_width * 0.5)), 80)
    x1 = int(np.clip(x_peak - half, 0, w - 1))
    x2 = int(np.clip(x_peak + half, x1 + 40, w))
    y1 = int(h * top_cut)
    y2 = int(h * bottom_cut)
    return slice(y1, y2), slice(x1, x2)


# ---------- centerline from edges (NaN-safe) ----------

def _centerline_from_edges(roi: np.ndarray, canny_low: int, canny_high: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = cv2.Canny(roi, canny_low, canny_high, L2gradient=True)
    h, w = edges.shape
    xs = np.full(h, np.nan, dtype=np.float32)

    v = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8), iterations=1)
    for y in range(h):
        cols = np.where(v[y] > 0)[0]
        if cols.size < 10:
            continue
        q1, q9 = np.percentile(cols, [5, 95])
        left = int(q1); right = int(q9)
        if right - left < max(30, int(w * 0.15)):
            continue
        xs[y] = 0.5 * (left + right)

    valid = np.isfinite(xs)
    if valid.sum() < h * 0.30:
        return xs, edges

    yi = np.arange(h)
    xs[~valid] = np.interp(yi[~valid], yi[valid], xs[valid])  # внутр. дырки

    # жёстко заполняем хвосты (если первые/последние NaN)
    if not np.isfinite(xs[0]):
        xs[0] = xs[np.where(np.isfinite(xs))[0][0]]
    if not np.isfinite(xs[-1]):
        xs[-1] = xs[np.where(np.isfinite(xs))[0][-1]]

    # forward/backward fill на всякий случай
    last = np.nan
    for i in range(h):
        if np.isfinite(xs[i]): last = xs[i]
        else: xs[i] = last
    last = np.nan
    for i in range(h - 1, -1, -1):
        if np.isfinite(xs[i]): last = xs[i]
        else: xs[i] = last

    if not np.isfinite(xs).all():
        # слишком много дыр — вернём пустой результат
        return np.full(h, np.nan, dtype=np.float32), edges

    xs = _savgol_1d(xs, win=max(15, h // 20), poly=2).astype(np.float32)
    return xs, edges


# ---------- Cobb from centerline ----------

def _cobb_from_centerline(xs: np.ndarray) -> Tuple[float, Tuple[int, int, float, float]]:
    h = len(xs)
    dx = np.gradient(xs)
    dx = _savgol_1d(dx, win=max(15, h // 25), poly=2)
    angles = np.degrees(np.arctan(dx)).astype(np.float32)
    strength = __smooth1d(np.abs(angles), 5)

    top = int(np.argmax(strength[: h // 2]))
    bottom = int(np.argmax(strength[h // 2 :])) + h // 2

    win = max(20, h // 12)
    a1 = float(np.median(angles[max(0, top - win) : min(h, top + win)]))
    a2 = float(np.median(angles[max(0, bottom - win) : min(h, bottom + win)]))
    cobb = abs(a1 - a2)
    return float(cobb), (int(top), int(bottom), a1, a2)


# ---------- public API ----------

def auto_cobb_from_gray(
    gray: np.ndarray,
    *,
    return_overlay: bool = True,
    canny: Tuple[int, int] | None = None,
    debug: bool = False,
    **_: dict,
) -> CobbResult:
    g = _normalize_uint8(gray)

    sy, sx = _locate_spine_roi(g)
    roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g[sy, sx].copy())

    if canny is None:
        med = float(np.median(roi))
        canny = (
            int(max(10, min(180, 0.66 * med))),
            int(max(30, min(255, 1.33 * med))),
        )

    xs, edges = _centerline_from_edges(roi, canny[0], canny[1])

    if not np.isfinite(xs).any():
        overlay = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) if return_overlay else None
        if return_overlay:
            cv2.rectangle(overlay, (sx.start, sy.start), (sx.stop - 1, sy.stop - 1), (0, 215, 255), 2)
            cv2.putText(overlay, "Cobb: 0.0 deg (fallback)",
                        (sx.start + 10, sy.start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return CobbResult(0.0, overlay, {"ok": 0.0})

    cobb, (iy1, iy2, a1, a2) = _cobb_from_centerline(xs)

    overlay = None
    if return_overlay:
        overlay = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        # ROI рамка
        cv2.rectangle(overlay, (sx.start, sy.start), (sx.stop - 1, sy.stop - 1), (0, 215, 255), 2)

        # центральная линия (рисуем только валидные точки)
        for i in range(sy.start, sy.stop):
            x_val = xs[i - sy.start]
            if np.isfinite(x_val):
                overlay = cv2.circle(overlay, (int(x_val) + sx.start, i), 1, (255, 255, 255), -1)

        # места измерения — берём ближайшие валидные, иначе fallback
        def _safe_int_from_xs(idx: int) -> Optional[int]:
            if 0 <= idx < len(xs) and np.isfinite(xs[idx]):
                return int(xs[idx]) + sx.start
            # поиск ближайшего валидного
            left = idx - 1
            right = idx + 1
            while left >= 0 or right < len(xs):
                if left >= 0 and np.isfinite(xs[left]):
                    return int(xs[left]) + sx.start
                if right < len(xs) and np.isfinite(xs[right]):
                    return int(xs[right]) + sx.start
                left -= 1; right += 1
            return None

        x1 = _safe_int_from_xs(iy1)
        x2 = _safe_int_from_xs(iy2)
        y1 = iy1 + sy.start
        y2 = iy2 + sy.start

        if x1 is None or x2 is None:
            # не удалось безопасно получить точки — подпишем fallback
            cv2.putText(overlay, "Cobb: 0.0 deg (fallback)",
                        (sx.start + 10, sy.start + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            return CobbResult(0.0, overlay, {"ok": 0.0})

        cv2.line(overlay, (x1, y1 - 40), (x1, y1 + 40), (80, 220, 60), 3)
        cv2.line(overlay, (x2, y2 - 40), (x2, y2 + 40), (0, 0, 220), 3)

        cv2.putText(overlay, f"Cobb: {cobb:.1f} deg",
                    (sx.start + 12, sy.start + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    diagnostics = {
        "roi_x1": float(sx.start), "roi_x2": float(sx.stop),
        "roi_y1": float(sy.start), "roi_y2": float(sy.stop),
        "a1_deg": float(a1), "a2_deg": float(a2), "ok": 1.0,
    }
    return CobbResult(float(cobb), overlay, diagnostics)
