# annotate_borders.py
# -*- coding: utf-8 -*-
# маленькое интерактивное приложение на OpenCV для ручной разметки двух продольных линий (левой и правой границ позвоночного столба) на уже подготовленных кропах (фрагментах снимков).
# Оно читает список таких кропов из index.json (файл генерируется скриптом select_for_labeling.py), показывает их по одному и позволяет мышью проставлять опорные точки для двух ломаных (left/right). 
# На выходе сохраняет бинарную маску PNG с этими линиями.
from __future__ import annotations
import os, json, argparse
import numpy as np, cv2

def _draw_poly(img, pts, color, thick=1):                                                               # Рисует ломаную (polyline) по списку точек pts на изображении img.
    for i in range(1, len(pts)):
        cv2.line(img, pts[i-1], pts[i], color, thick, cv2.LINE_AA)                                      # цикл из cv2.line для соседних пар точек
                                                                                                        # color — BGR-цвет, thick — толщина линии (по умолчанию 1 пиксель)
def _rasterize_mask(h, w, left_pts, right_pts):                                                         # Создаёт пустую маску размера h×w (uint8) и прорисовывает на ней две ломаные
    m = np.zeros((h,w), np.uint8)
    if len(left_pts) >= 2:                                                                              # левая (зелёная в GUI)  — но в маске всегда белым (255).
        for i in range(1, len(left_pts)):
            cv2.line(m, left_pts[i-1], left_pts[i], 255, 1, cv2.LINE_AA)
    if len(right_pts) >= 2:                                                                             # правая (оранжевая в GUI) — но в маске всегда белым (255).
        for i in range(1, len(right_pts)):
            cv2.line(m, right_pts[i-1], right_pts[i], 255, 1, cv2.LINE_AA)                              # Толщина линий в маске — 1 пиксель (параметр зашит).
    return m

def main():                                                                                             # Парсинг аргументов командной строки
    ap = argparse.ArgumentParser("Interactive border annotator")
    ap.add_argument("index_json", help="index.json from select_for_labeling.py")
    ap.add_argument("--offset-px", type=int, default=9, help="auto centerline offset for 'a'")
    args = ap.parse_args()

    with open(args.index_json, "r", encoding="utf-8") as f:                                             # Чтение списка кропов
        items = json.load(f)

    i = 0
    while 0 <= i < len(items):
        item = items[i]
        crop_path = item["crop"]                                                                        # Читает картинку crop_path в градациях серого.
        img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[skip] can't read {crop_path}"); i += 1; continue
        h,w = img.shape
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                                                   # Делает цветную версию для визулизации

        left_pts, right_pts = [], []
        active = "left"                                                                                 # кто получает клики: 'left' (ЛКМ) или 'right' (ПКМ)

                                                                                                        # обработчик мыши
        def on_mouse(event, x, y, flags, userdata):
            nonlocal left_pts, right_pts, active
            if event == cv2.EVENT_LBUTTONDOWN:
                left_pts.append((x,y)); active = "left"
            elif event == cv2.EVENT_RBUTTONDOWN:
                right_pts.append((x,y)); active = "right"

        win = "Annotate (L=left border, R=right border) | s=save, n=next, z=undo, r=reset, a=auto-start"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)

        while True:
            vis = color.copy()
            _draw_poly(vis, left_pts, (0,255,0), 2)
            _draw_poly(vis, right_pts,(0,128,255), 2)
            cv2.putText(vis, f"{os.path.basename(crop_path)}   active:{active}",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(200,200,200),2,cv2.LINE_AA)
            cv2.imshow(win, vis)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'):
                cv2.destroyAllWindows()
                return
            if k == ord('n'):
                break
            if k == ord('r'):
                left_pts, right_pts = [], []
            if k == ord('z'):
                if active == "left" and left_pts: left_pts.pop()
                elif active == "right" and right_pts: right_pts.pop()
            if k == ord('a'):
                                                                                                        # авто-инициализация: вертикальная линия по центру как центрлайн -> смещаем
                cx = w//2
                left_pts  = [(cx - args.offset_px, y) for y in range(0, h, 10)]
                right_pts = [(cx + args.offset_px, y) for y in range(0, h, 10)]
            if k == ord('s'):
                m = _rasterize_mask(h,w,left_pts,right_pts)
                out_mask = os.path.splitext(crop_path)[0].replace("_crop","") + "_mask.png"
                cv2.imwrite(out_mask, m)
                print(f"[saved] {out_mask}")
                break

        i += 1                                                                                          # Переход к следующему элементу

if __name__ == "__main__":
    main()
