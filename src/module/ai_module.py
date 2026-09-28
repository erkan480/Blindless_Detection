import cv2
import numpy as np

from src.module.config import IMG_SIZE


def apply_preprocessing(img, img_size=IMG_SIZE):
    # 1. BGR'den RGB'ye çevir
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]

    # 2. Kontur ile içerik kırpma (orijinal boyutta)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)[:, 0, :]
        x1, y1 = np.min(contour, axis=0)
        x2, y2 = np.max(contour, axis=0)

        if (x2 - x1 > w_orig // 6) and (y2 - y1 > h_orig // 6):
            x_margin = int((x2 - x1) * 0.08)
            y_margin = int((y2 - y1) * 0.08)

            x1 = max(0, x1 - x_margin)
            y1 = max(0, y1 - y_margin)
            x2 = min(w_orig, x2 + x_margin)
            y2 = min(h_orig, y2 + y_margin)

            img_rgb = img_rgb[y1:y2, x1:x2]

    # 3. Resize
    img_rgb = cv2.resize(img_rgb, (img_size, img_size))

    # 4. CLAHE uygulama (istenirse)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    """# 5. Blur çıkarımı (isteğe bağlı — test et)
    med_blur = cv2.medianBlur(img_clahe, 3)
    background = cv2.medianBlur(img_clahe, 37)
    mask = cv2.subtract(med_blur, background)
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
    img_final = cv2.bitwise_and(mask, med_blur)"""
    return img_clahe
