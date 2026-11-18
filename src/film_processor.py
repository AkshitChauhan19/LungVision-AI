import os
import cv2
import numpy as np
from PIL import Image

def rectify_perspective(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (img_bgr.shape[0] * img_bgr.shape[1] * 0.25):
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            (tl, tr, br, bl) = rect

            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            max_w = max(int(width_a), int(width_b))

            height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_h = max(int(height_a), int(height_b))

            dst = np.array([
                [0, 0],
                [max_w - 1, 0],
                [max_w - 1, max_h - 1],
                [0, max_h - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img_bgr, M, (max_w, max_h))
            return warped

    return img_bgr


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def enhance_medical_contrast(img_bgr: np.ndarray) -> np.ndarray:
    denoised = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=25, sigmaSpace=25)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    bg_illum = cv2.GaussianBlur(l_channel, (101, 101), 0)
    norm_l = cv2.divide(l_channel, bg_illum, scale=128)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(norm_l)

    merged = cv2.merge([enhanced_l, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def detect_and_decompose_grid(film_img: np.ndarray, rows: int = None, cols: int = None):
    h, w = film_img.shape[:2]
    gray = cv2.cvtColor(film_img, cv2.COLOR_BGR2GRAY)

    if rows is not None and cols is not None:
        return decompose_fixed_grid(film_img, rows, cols)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 25, 5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    min_area = (h * w) / 60.0
    max_area = (h * w) / 3.0

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        aspect = float(cw) / float(ch)
        if min_area <= area <= max_area and 0.70 <= aspect <= 1.40:
            candidates.append((x, y, cw, ch))

    filtered = filter_overlapping_boxes(candidates)

    if len(filtered) >= 4:
        sorted_boxes = sort_boxes_reading_order(filtered, row_thresh=int(h * 0.05))
        slices = []
        for x, y, cw, ch in sorted_boxes:
            crop = film_img[y:y+ch, x:x+cw]
            cleaned = clean_slice_tile(crop)
            slices.append((cleaned, (x, y, cw, ch)))
        return slices

    # Not a multi-slice grid film sheet - do not force grid decomposition on single CT slices
    return []


def estimate_grid_dims(h: int, w: int):
    aspect = float(w) / float(h)
    if aspect >= 1.2:
        return 4, 6
    elif aspect <= 0.85:
        return 5, 4
    else:
        return 4, 4


def decompose_fixed_grid(film_img: np.ndarray, rows: int, cols: int):
    h, w = film_img.shape[:2]
    tile_h = h // rows
    tile_w = w // cols
    slices = []
    for r in range(rows):
        for c in range(cols):
            y1 = r * tile_h
            y2 = (r + 1) * tile_h if r < rows - 1 else h
            x1 = c * tile_w
            x2 = (c + 1) * tile_w if c < cols - 1 else w
            crop = film_img[y1:y2, x1:x2]
            cleaned = clean_slice_tile(crop)
            slices.append((cleaned, (x1, y1, x2 - x1, y2 - y1)))
    return slices


def clean_slice_tile(tile: np.ndarray) -> np.ndarray:
    th, tw = tile.shape[:2]
    pad_y = int(th * 0.06)
    pad_x = int(tw * 0.06)
    roi = tile[pad_y:th - pad_y, pad_x:tw - pad_x]

    rh, rw = roi.shape[:2]
    scale = 240.0 / max(rh, rw)
    nh, nw = int(rh * scale), int(rw * scale)
    resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_CUBIC)

    canvas = np.zeros((240, 240, 3), dtype=np.uint8)
    dy = (240 - nh) // 2
    dx = (240 - nw) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas


def filter_overlapping_boxes(boxes, iou_thresh=0.3):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    selected = []
    for b in boxes:
        x1, y1, w1, h1 = b
        overlap = False
        for s in selected:
            x2, y2, w2, h2 = s
            inter_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            inter_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter_area = inter_x * inter_y
            union_area = (w1 * h1) + (w2 * h2) - inter_area
            if union_area > 0 and (inter_area / union_area) > iou_thresh:
                overlap = True
                break
        if not overlap:
            selected.append(b)
    return selected


def sort_boxes_reading_order(boxes, row_thresh=30):
    boxes = sorted(boxes, key=lambda b: b[1])
    rows = []
    current_row = [boxes[0]]
    for b in boxes[1:]:
        if abs(b[1] - current_row[0][1]) < row_thresh:
            current_row.append(b)
        else:
            rows.append(sorted(current_row, key=lambda x: x[0]))
            current_row = [b]
    if current_row:
        rows.append(sorted(current_row, key=lambda x: x[0]))
    
    sorted_boxes = []
    for r in rows:
        sorted_boxes.extend(r)
    return sorted_boxes
