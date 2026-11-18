import cv2
import numpy as np

def validate_ct_scan(img_bgr: np.ndarray, threshold_lung: float = 0.04):
    """
    Two-Stage Medical Modality & Organ Validator:
    - Automatically detects Multi-Slice Physical CT Film Sheets (photographed with hands / ambient light).
    - Verifies Single Thoracic CT Slices (DICOM / PNG / JPEG) while blocking code, screenshots, and non-medical images.
    """
    if img_bgr is None or img_bgr.size == 0:
        return False, "INVALID_FILE", "Image file is empty or could not be decoded."

    h, w = img_bgr.shape[:2]
    if h < 50 or w < 50:
        return False, "INVALID_SIZE", "Image resolution is too low for radiological analysis."

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    mean_sat = sat.mean()

    # Reject obviously colorful non-medical photos (saturation > 65)
    if mean_sat > 65.0:
        return False, "NOT_CT", (
            f"Image has excessive color saturation ({mean_sat:.1f} > 65.0).\n"
            "• Medical thoracic CT examinations are grayscale radiographs.\n"
            "• Please upload a thoracic CT scan or physical film photo."
        )

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if np.std(gray) < 8.0:
        return False, "NOT_CT", (
            "Image lacks radiographic contrast dynamic range.\n"
            "• The image appears to be a blank or uniform graphic.\n"
            "• Please upload a valid CT scan."
        )

    # -------------------------------------------------------------
    # 1. Multi-Slice Physical CT Film Sheet Detection
    # -------------------------------------------------------------
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
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

    # Filter overlapping boxes
    filtered = []
    if candidates:
        candidates = sorted(candidates, key=lambda b: b[2] * b[3], reverse=True)
        for b in candidates:
            x1, y1, w1, h1 = b
            overlap = False
            for s in filtered:
                x2, y2, w2, h2 = s
                inter_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                inter_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                inter_area = inter_x * inter_y
                union_area = (w1 * h1) + (w2 * h2) - inter_area
                if union_area > 0 and (inter_area / union_area) > 0.30:
                    overlap = True
                    break
            if not overlap:
                filtered.append(b)

    if len(filtered) >= 4:
        # Multi-slice CT film sheet verified!
        return True, "OK", f"Input validation passed: Multi-slice CT Film Sheet verified ({len(filtered)} tiles detected)."

    # -------------------------------------------------------------
    # 2. Single Thoracic CT Slice Verification
    # -------------------------------------------------------------
    colored_pixels_ratio = (sat > 35).mean()
    if colored_pixels_ratio > 0.05 or mean_sat > 28.0:
        return False, "NOT_CT", (
            "The uploaded image is NOT a medical thoracic CT scan.\n"
            f"• Detected non-radiographic colored elements ({colored_pixels_ratio*100:.1f}% colored pixels like code syntax, UI, or photo).\n"
            "• Thoracic CT scans are strictly grayscale radiographs.\n"
            "• Please upload an authentic thoracic CT scan (PNG/JPEG/DICOM) or physical film photo."
        )

    _, body_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    body_pixels = np.sum(body_mask > 0)
    if body_pixels < (h * w * 0.05):
        return False, "NOT_CT", (
            "No medical cross-section or film sheet detected.\n"
            "• The image does not contain the signature anatomy of a CT scan."
        )

    kernel_size = max(5, int(min(h, w) * 0.03))
    inner_body = cv2.erode(body_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    lung_mask = ((gray < 85) & (inner_body > 0)).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lung_mask, connectivity=8)
    min_chamber_sz = body_pixels * 0.008
    max_chamber_sz = body_pixels * 0.70
    valid_chambers = [i for i in range(1, num_labels) if min_chamber_sz <= stats[i, cv2.CC_STAT_AREA] <= max_chamber_sz]

    if len(valid_chambers) == 0:
        return False, "NOT_LUNG", (
            "The image does not contain thoracic lung anatomy or lung air cavities.\n"
            "• Detected non-thoracic graphic, text document, or non-lung scan.\n"
            "• This CAD system specializes exclusively in thoracic lung cancer screening."
        )

    return True, "OK", "Input validation passed: Single Thoracic CT Slice verified."
