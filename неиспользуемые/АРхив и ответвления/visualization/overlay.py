import cv2 as cv
import numpy as np

def draw_line(img: np.ndarray, p1, p2, color=(0,255,0), thickness=2):
    cv.line(img, tuple(map(int,p1)), tuple(map(int,p2)), color, thickness, cv.LINE_AA)

def put_text(img: np.ndarray, text: str, org=(10,30)):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv.LINE_AA)

def overlay_cobb(base_u8: np.ndarray, roi_xyxy, line1, line2, angle_deg: float):
    x0,y0,x1,y1 = roi_xyxy
    vis = cv.cvtColor(base_u8, cv.COLOR_GRAY2BGR)
    cv.rectangle(vis, (x0,y0), (x1,y1), (0,200,255), 2)
    p1a = (line1[0][0]+x0, line1[0][1]+y0)
    p1b = (line1[1][0]+x0, line1[1][1]+y0)
    p2a = (line2[0][0]+x0, line2[0][1]+y0)
    p2b = (line2[1][0]+x0, line2[1][1]+y0)
    draw_line(vis, p1a, p1b, (0,255,0), 2)
    draw_line(vis, p2a, p2b, (0,0,255), 2)
    put_text(vis, f"Cobb: {angle_deg:.1f} deg", (x0+10, max(30,y0+30)))
    return vis
