import math
from typing import Tuple, List
import numpy as np
import cv2 as cv

def angle_between_lines(p1a, p1b, p2a, p2b) -> float:
    """Return absolute angle (deg) between two lines defined by two points each."""
    v1 = np.array([p1b[0]-p1a[0], p1b[1]-p1a[1]], dtype=float)
    v2 = np.array([p2b[0]-p2a[0], p2b[1]-p2a[1]], dtype=float)
    def angle(v):
        return math.degrees(math.atan2(v[1], v[0]))
    a1 = angle(v1)
    a2 = angle(v2)
    ang = abs(a1 - a2)
    if ang > 180: ang = 360 - ang
    if ang > 90:  # use acute
        ang = 180 - ang
    return float(abs(ang))

def _hough_lines(roi_u8: np.ndarray) -> List[Tuple[Tuple[int,int], Tuple[int,int]]]:
    edges = cv.Canny(roi_u8, 50, 150, apertureSize=3, L2gradient=True)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=max(20, roi_u8.shape[1]//5), maxLineGap=10)
    segs = []
    if lines is not None:
        for l in lines[:,0,:]:
            x1,y1,x2,y2 = map(int, l.tolist())
            segs.append(((x1,y1),(x2,y2)))
    return segs

def _segment_angle(p1, p2):
    v = (p2[0]-p1[0], p2[1]-p1[1])
    return math.degrees(math.atan2(v[1], v[0]))

def auto_cobb_from_roi(roi_u8: np.ndarray):
    """
    Heuristic:
     - detect many line segments
     - keep near-horizontal segments (|angle| < ~30deg)
     - pick two with max angular separation
    Returns (angle_deg, (p1a,p1b), (p2a,p2b))
    """
    segs = _hough_lines(roi_u8)
    if not segs:
        raise RuntimeError("No lines detected in ROI")

    cand = []
    for (a,b) in segs:
        ang = _segment_angle(a,b)
        a_norm = ((ang + 180) % 180)
        if a_norm > 90:
            a_norm = 180 - a_norm
        if abs(a_norm) <= 30:
            cand.append(((a,b), a_norm))

    if len(cand) < 2:
        cand = [((a,b), _segment_angle(a,b)) for (a,b) in segs]

    best = None
    best_sep = -1
    for i in range(len(cand)):
        for j in range(i+1, len(cand)):
            ang1 = cand[i][1]
            ang2 = cand[j][1]
            sep = abs(ang1 - ang2)
            if sep > best_sep:
                best_sep = sep
                best = (cand[i][0], cand[j][0])

    (p1a,p1b), (p2a,p2b) = best
    deg = angle_between_lines(p1a,p1b,p2a,p2b)
    return deg, (p1a,p1b), (p2a,p2b)
