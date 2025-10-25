import cv2
import numpy as np

# -----------------------
# 1) Предобработка: CLAHE + подавление рёбер, усиление вертикальных границ
# -----------------------
def _preprocess_for_spine(x):
    # x: float32 [0..1] или uint8
    if x.dtype != np.uint8:
        x8 = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    else:
        x8 = x

    # CLAHE для локального контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    x8 = clahe.apply(x8)

    # Мягкое шумоподавление, не размывая границ
    x8 = cv2.bilateralFilter(x8, d=5, sigmaColor=35, sigmaSpace=7)

    # Подавление рёбер (горизонталей) и усиление вертикального
    # Используем Габор с вертикальной ориентацией и антирёберный с горизонтальной
    g_vert = cv2.getGaborKernel(ksize=(31,31), sigma=6, theta=np.pi/2, lambd=10, gamma=0.5, psi=0)
    g_horz = cv2.getGaborKernel(ksize=(31,31), sigma=6, theta=0,       lambd=10, gamma=0.5, psi=0)

    v_resp = cv2.filter2D(x8, cv2.CV_32F, g_vert)
    h_resp = cv2.filter2D(x8, cv2.CV_32F, g_horz)

    # Усиливаем вертикальное и вычитаем горизонтальное
    resp = v_resp - 0.7 * h_resp

    # Нормируем к [0..1]
    resp = resp - resp.min()
    if resp.max() > 0:
        resp = resp / resp.max()
    return resp.astype(np.float32)  # [0..1]


# -----------------------
# 2) Карта "стоимости" для швов по левой/правой стенке
#    Берём горизонтальный градиент от вертикально-усиленного изображения
# -----------------------
def _edge_cost_maps(resp):
    # Горизонтальный градиент: левый край -> отрицательный, правый -> положительный
    gx = cv2.Sobel(resp, cv2.CV_32F, 1, 0, ksize=3)
    gx_blur = cv2.GaussianBlur(gx, (3,3), 0)

    # Стоимость = -|граница|, но отдельно знакополярно:
    #   слева хотим переход "тёмное->светлое" (gx > 0), справа "светлое->тёмное" (gx < 0)
    left_cost  = -np.clip(gx_blur,  0, None)   # хорошие большие положительные gx -> маленькая стоимость
    right_cost =  np.clip(gx_blur, None, 0)    # большие отрицательные gx -> маленькая стоимость (после инверсии знак)

    # Приводим к положительным и нормируем
    def norm_pos(c):
        c = c - c.min()
        c = c / (c.max() + 1e-6)
        return c + 1e-3  # чтобы нули не ломали DP

    return norm_pos(-left_cost), norm_pos(-right_cost)  # чем меньше, тем «лучше»


# -----------------------
# 3) Поиск двух связанных швов DP с ограничением ширины
# -----------------------
def _trace_coupled_seams(left_cost, right_cost, x0, w0, width_bounds=(40, 140),
                         lam_smooth=6.0, lam_width=0.3):
    """
    left_cost/right_cost: HxW карты стоимости (меньше = лучше)
    x0: центр ROI по X (в пикселях), w0: ожидаемая ширина (px)
    width_bounds: (w_min, w_max) в пикселях
    """
    H, W = left_cost.shape
    w_min, w_max = width_bounds

    # Инициализация: разрешённое окно вокруг ожидаемого центра
    # Чтобы не уехать к краям снимка
    cx = int(np.clip(x0, 20, W-21))
    # DP по строкам (y), состояние = (x_left, width)
    # но хранить всю матрицу 2D дорого -> пойдём от центра: будем перебирать x_left в разумном окне
    win = 120  # половина окна по x вокруг центра
    x_left_min = max(0, cx - win)
    x_left_max = min(W-1, cx + win)

    # Подготовим большие «плохие» стоимости вне допустимого
    BIG = 1e6
    prev_cost = None
    prev_ptr  = {}

    # Начальная строка
    y0 = 0
    cur_cost = {}
    for xl in range(x_left_min, x_left_max+1):
        for w in range(w_min, w_max+1):
            xr = xl + w
            if xr >= W: 
                continue
            c = left_cost[y0, xl] + right_cost[y0, xr] + lam_width * ((w - w0)**2)
            cur_cost[(xl, w)] = c
    prev_cost = cur_cost
    prev_ptr[y0] = {k: None for k in cur_cost.keys()}

    # Проходим вниз
    for y in range(1, H):
        cur_cost = {}
        cur_ptr  = {}
        for (xl, w), c_prev in prev_cost.items():
            xr = xl + w
            # разрешаем небольшой сдвиг по x и ширине (гладкость)
            for dx in (-2, -1, 0, 1, 2):
                xl2 = xl + dx
                if xl2 < x_left_min or xl2 >= x_left_max: 
                    continue
                for dw in (-2, -1, 0, 1, 2):
                    w2 = w + dw
                    if w2 < w_min or w2 > w_max:
                        continue
                    xr2 = xl2 + w2
                    if xr2 >= W:
                        continue
                    # локальная стоимость
                    data_term = left_cost[y, xl2] + right_cost[y, xr2] + lam_width*((w2 - w0)**2)
                    smooth = lam_smooth*(dx*dx + 0.5*dw*dw)
                    c = c_prev + data_term + smooth

                    key = (xl2, w2)
                    if (key not in cur_cost) or (c < cur_cost[key]):
                        cur_cost[key] = c
                        cur_ptr[key]  = (xl, w)
        prev_cost = cur_cost
        prev_ptr[y] = cur_ptr

        # ранняя отсечка: оставим top-K путей
        if len(prev_cost) > 40000:
            # оставим 10k лучших
            items = sorted(prev_cost.items(), key=lambda kv: kv[1])[:10000]
            prev_cost = dict(items)

    # Завершение: берём лучший на последней строке
    if not prev_cost:
        raise RuntimeError("DP failed: no paths")
    last_key = min(prev_cost, key=lambda k: prev_cost[k])

    # Обратный проход
    xl_path = np.zeros(H, np.int32)
    xr_path = np.zeros(H, np.int32)
    w_path  = np.zeros(H, np.int32)

    key = last_key
    for y in range(H-1, -1, -1):
        xl, w = key
        xl_path[y] = xl
        w_path[y]  = w
        xr_path[y] = xl + w
        key = prev_ptr[y][key] if prev_ptr[y][key] is not None else key

    return xl_path, xr_path


# -----------------------
# 4) Публичная функция
# -----------------------
def detect_spine_borders(img_u8, roi=None, expected_width_px=120,
                         width_bounds=(60, 180),
                         lam_smooth=6.0, lam_width=0.3):
    """
    img_u8: X-ray как uint8 (0..255) или float32 (0..1)
    roi: (x1,y1,x2,y2) в пикселях. Если None, берём центральную колонку шириной ~40% кадра.
    Возвращает: dict с ключами:
        left_x, right_x, center_x  — по одному x на каждую строку (np.ndarray длиной H_roi)
        overlay_bgr                — картинка для отрисовки
    """
    if img_u8.ndim == 3:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

    H, W = img_u8.shape

    # ROI по умолчанию
    if roi is None:
        w2 = int(W * 0.2)
        cx = W // 2
        roi = (max(0, cx - w2), 0, min(W, cx + w2), H)

    x1, y1, x2, y2 = roi
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    crop = img_u8[y1:y2, x1:x2]

    # 1) Предобработка
    resp = _preprocess_for_spine(crop)

    # 2) Стоимости для левой/правой границы
    left_cost, right_cost = _edge_cost_maps(resp)

    # 3) DP для двух связанных швов
    cx0 = (x2 - x1) / 2.0
    xl, xr = _trace_coupled_seams(
        left_cost, right_cost,
        x0=cx0,
        w0=float(expected_width_px),
        width_bounds=width_bounds,
        lam_smooth=lam_smooth,
        lam_width=lam_width
    )

    # 4) Сглаживание траекторий и центрлайн
    def smooth1d(a, k=13):
        k = max(3, k | 1)
        return cv2.GaussianBlur(a.astype(np.float32).reshape(-1,1), (1,k), 0).ravel()

    xl_s = smooth1d(xl,  nine_tap(13))
    xr_s = smooth1d(xr,  nine_tap(13))
    ctr  = 0.5*(xl_s + xr_s)

    # 5) Overlay
    ov = cv2.cvtColor(crop if crop.dtype==np.uint8 else (crop*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    yy = np.arange(y2 - y1, dtype=np.int32)

    for y, x in zip(yy, xl_s.astype(int)):
        cv2.circle(ov, (int(x), int(y)), 1, (0,255,0), -1)    # левая — зелёная
    for y, x in zip(yy, xr_s.astype(int)):
        cv2.circle(ov, (int(x), int(y)), 1, (0,0,255), -1)    # правая — красная
    for y, x in zip(yy, ctr.astype(int)):
        cv2.circle(ov, (int(x), int(y)), 1, (255,255,255), -1)  # центр — белая

    # положим обратно в полноразмер для удобной отрисовки поверх исходника
    overlay_full = cv2.cvtColor(img_u8 if img_u8.dtype==np.uint8 else (img_u8*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay_full[y1:y2, x1:x2] = cv2.addWeighted(overlay_full[y1:y2, x1:x2], 0.4, ov, 0.6, 0)

    # абсолютные координаты X относительно исходника
    xl_abs = xl_s + x1
    xr_abs = xr_s + x1
    ctr_abs = ctr + x1

    return {
        "left_x":   xl_abs.astype(np.float32),
        "right_x":  xr_abs.astype(np.float32),
        "center_x": ctr_abs.astype(np.float32),
        "roi":      (x1,y1,x2,y2),
        "overlay_bgr": overlay_full
    }

def nine_tap(k):
    # просто helper, чтобы гарантированно получить нечётное k
    return k | 1
