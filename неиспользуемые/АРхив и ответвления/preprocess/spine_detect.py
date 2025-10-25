import numpy as np
import cv2 as cv

def normalize_uint8(img_f32: np.ndarray) -> np.ndarray:
    """Scale float32 [0,1] to uint8 [0,255]."""
    x = np.clip(img_f32, 0, 1)
    return (x * 255.0).astype(np.uint8)

def estimate_spine_roi(img_f32: np.ndarray):
    """
    Heuristic ROI finder for thoracic column on AP X-ray.
    Steps:
      - convert to uint8, CLAHE for local contrast
      - vertical gradient + threshold
      - vertical projection to find spine centerline
      - crop a central vertical strip
    Returns: roi_img (uint8), (x0,y0,x1,y1)
    """
    u8 = normalize_uint8(img_f32)
    h, w = u8.shape[:2]
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(u8)

    # Edge-like enhancement (Sobel vertical)
    sobelx = cv.Sobel(eq, cv.CV_32F, 1, 0, ksize=3)
    sobelx = np.abs(sobelx).astype(np.float32)
    sobelx = (255 * (sobelx / (sobelx.max() + 1e-6))).astype(np.uint8)

    # Vertical projection
    proj = np.sum(sobelx, axis=0).astype(np.float32)
    # Smooth
    proj_smooth = cv.GaussianBlur(proj.reshape(1,-1), (0,0), sigmaX=10).flatten()

    cx = int(np.argmax(proj_smooth))  # spine center x
    strip_w = max( int(0.25 * w), 80 )  # 25% of width or min 80px
    x0 = max(0, cx - strip_w//2)
    x1 = min(w, cx + strip_w//2)

    # For thoracic, prefer mid-height region (exclude pelvis/neck)
    y0 = int(0.15 * h)
    y1 = int(0.85 * h)

    roi = eq[y0:y1, x0:x1].copy()
    return roi, (x0, y0, x1, y1)
