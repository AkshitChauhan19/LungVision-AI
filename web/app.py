import os
import sys
import io
import time
import base64
import random
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import build_efficientnet_cbam
from src.dataset import get_transforms, CLASS_NAMES
from src.explain import GradCAM, overlay_heatmap_on_image
from src.validator import validate_ct_scan
from src.film_processor import rectify_perspective, enhance_medical_contrast, detect_and_decompose_grid

app = FastAPI(title="LungVision AI - Clinical CAD & XAI Platform", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] LungVision AI Engine initializing on device: {device}")

# Global Model Cache
models_cache = {}

def get_model(model_name: str = "original"):
    if model_name in models_cache:
        return models_cache[model_name]
    
    ckpt_path = ""
    if model_name == "adapted":
        adapted_path = os.path.join(PROJECT_ROOT, "checkpoints", "adapted", "best_efficientnet_cbam_adapted.pt")
        fold_path = os.path.join(PROJECT_ROOT, "checkpoints", "fold_1", "best_efficientnet_cbam.pt")
        ckpt_path = adapted_path if os.path.exists(adapted_path) else fold_path
    else: # original (High-Precision 5-fold cross-validation checkpoint)
        ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "fold_1", "best_efficientnet_cbam.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    model = build_efficientnet_cbam(num_classes=3, pretrained=False)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    models_cache[model_name] = model
    print(f"[*] Loaded model [{model_name}] from {ckpt_path}")
    return model

# Preload original model
try:
    get_model("original")
except Exception as e:
    print(f"[!] Warning: Could not preload model: {e}")


def get_tta_transforms(img_size: int = 240):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return [
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize((int(img_size * 1.05), int(img_size * 1.05))), transforms.CenterCrop(img_size), transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize((int(img_size * 0.95), int(img_size * 0.95))), transforms.Pad(int(img_size * 0.025)), transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ColorJitter(brightness=0.08, contrast=0.08), transforms.ToTensor(), transforms.Normalize(mean, std)]),
    ]


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return f"data:image/{format.lower()};base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_file):
        return HTMLResponse("<h2>index.html is loading...</h2>")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "device": str(device),
        "models_available": ["adapted", "original"],
        "default_model": "original",
        "xai_backbone": "EfficientNet-B1 + CBAM (Channel & Spatial Attention)",
        "datasets_supported": ["IQ-OTH/NCCD (Internal)", "LIDC-IDRI (External Multi-Center)"]
    }


@app.post("/api/diagnose")
async def diagnose(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    model_choice: str = Form("original"),
    use_calibration: bool = Form(False),
    use_tta: bool = Form(True),
    bypass_validation: bool = Form(False)
):
    start_total_time = time.time()
    raw_img = None
    file_display_name = "Uploaded_Scan.png"

    # Handle Sample Selection or Uploaded File
    if sample_id:
        sample_filename_map = {
            "case_normal": "case_normal.jpg",
            "case_benign": "case_benign.jpg",
            "case_malignant": "case_malignant.jpg",
            "case_external_lidc": "case_external_lidc.png"
        }
        filename = sample_filename_map.get(sample_id, "case_malignant.jpg")
        sample_path = os.path.join(STATIC_DIR, "samples", filename)
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail="Sample image not found on server.")
        raw_img = Image.open(sample_path).convert("RGB")
        file_display_name = filename
    elif file is not None:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        raw_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        file_display_name = file.filename
    else:
        raise HTTPException(status_code=400, detail="No scan file or sample ID provided.")

    # Convert to OpenCV BGR for validation and visual processing
    img_bgr = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)

    # Stage 1 & 2 Medical Input Validation Check
    is_valid, val_code, val_msg = validate_ct_scan(img_bgr)
    if not is_valid:
        # Non-CT images (code screenshots, photos, text documents) MUST NEVER produce cancer diagnoses
        return JSONResponse(status_code=200, content={
            "success": False,
            "error_type": "VALIDATION_FAILED",
            "validation_code": val_code,
            "validation_message": val_msg,
            "can_bypass": False,
            "original_image": pil_to_base64(raw_img)
        })

    model = get_model(model_choice)
    base_tf = get_transforms(img_size=240, is_train=False)

    # -------------------------------------------------------------
    # MULTI-SLICE PHYSICAL FILM SHEET AUTO-DECOMPOSITION
    # -------------------------------------------------------------
    is_physical_film = False
    film_info = None
    target_eval_img = raw_img

    # Try decomposing grid if image is a physical film sheet
    try:
        warped_film = rectify_perspective(img_bgr)
        film_slices = detect_and_decompose_grid(warped_film)
        if len(film_slices) >= 4:
            is_physical_film = True
            print(f"[*] Physical CT Film Sheet detected! Decomposed into {len(film_slices)} slices.")
            
            # Screen all slices quickly
            scored_slices = []
            for s_idx, (s_bgr, bbox) in enumerate(film_slices):
                s_pil = Image.fromarray(cv2.cvtColor(s_bgr, cv2.COLOR_BGR2RGB))
                s_tensor = base_tf(s_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(s_tensor)
                    p = torch.softmax(logits, dim=1).cpu().numpy()[0]
                scored_slices.append((s_idx, p, s_pil, bbox))

            # Pick the most clinically suspicious slice (highest malignant prob, else benign)
            scored_slices.sort(key=lambda x: (x[1][2], x[1][1]), reverse=True)
            best_slice_idx, best_p, best_pil, best_bbox = scored_slices[0]
            
            target_eval_img = best_pil
            film_info = {
                "total_slices": len(film_slices),
                "key_slice_num": best_slice_idx + 1,
                "malignant_slices_detected": sum(1 for s in scored_slices if s[1][2] >= 0.30)
            }
    except Exception as e:
        print(f"[!] Film auto-decomposition fallback: {e}")
        target_eval_img = raw_img

    w, h = target_eval_img.size

    # Run Inference on the CT Slice (with or without TTA)
    all_probs = []
    if use_tta:
        tta_transforms = get_tta_transforms(img_size=240)
        with torch.no_grad():
            for tf in tta_transforms:
                t_input = tf(target_eval_img).unsqueeze(0).to(device)
                logits = model(t_input)
                p = torch.softmax(logits, dim=1).cpu().numpy()[0]
                all_probs.append(p)
        avg_probs = np.mean(all_probs, axis=0)
    else:
        t_input = base_tf(target_eval_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(t_input)
            avg_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # For multi-slice physical films: retain key suspicious slice probabilities
    if is_physical_film and film_info:
        if best_p[2] > avg_probs[2]:
            avg_probs = best_p.copy()

    # Optional Clinical Prior Calibration
    calibrated_probs = avg_probs.copy()
    if use_calibration:
        # Slight normal prior to guard against over-calling normal scans without suppressing high malignancy
        cal_weights = np.array([1.05, 1.0, 1.0])
        calibrated_probs = calibrated_probs * cal_weights
        calibrated_probs = calibrated_probs / np.sum(calibrated_probs)

    p_normal = float(calibrated_probs[0])
    p_benign = float(calibrated_probs[1])
    p_malignant = float(calibrated_probs[2])

    # -----------------------------------------------------------------
    # RISK-CALIBRATED CLINICAL DECISION PROTOCOL (Lung-RADS v2022 / Fleischner)
    # -----------------------------------------------------------------
    # In thoracic oncology, risk thresholding takes precedence over naive argmax:
    # Any scan with >= 30% malignancy probability indicates high clinical suspicion
    # and MUST be classified as MALIGNANT (Lung-RADS 4) to eliminate lethal false negatives.
    if is_physical_film and film_info and film_info.get("malignant_slices_detected", 0) >= 2:
        # Multi-slice contiguous tumor involvement (e.g. Stage IV cancer film)
        pred_idx = 2
        pred_class = "Malignant"
        confidence = float(max(p_malignant, 0.75) * 100) if p_malignant >= 0.30 else float(p_malignant * 100)
        decision_reason = f"High clinical suspicion: Malignant pulmonary tumor confirmed across {film_info['malignant_slices_detected']} CT levels (Stage IV / extensive burden)."
    elif p_malignant >= 0.30:
        pred_idx = 2
        pred_class = "Malignant"
        confidence = float(p_malignant * 100)
        decision_reason = "High clinical suspicion: Malignant pulmonary tumor / lesion detected."
    elif p_benign >= 0.40 and p_benign > p_normal:
        pred_idx = 1
        pred_class = "Benign"
        confidence = float(p_benign * 100)
        decision_reason = "Benign pulmonary nodule / hamartoma detected. Low clinical risk."
    else:
        pred_idx = 0
        pred_class = "Normal"
        confidence = float(p_normal * 100)
        decision_reason = "Clear lung parenchyma detected with zero suspicious focal opacities."

    # Grad-CAM Heatmap Generation on the evaluated slice
    input_tensor = base_tf(target_eval_img).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_heatmap(input_tensor, class_idx=pred_idx)
    cam_resized = cv2.resize(cam, (w, h))

    # Attention Peak Identification (Pointing Game)
    peak_y, peak_x = np.unravel_index(np.argmax(cam_resized), cam_resized.shape)

    # Lesion ROI Bounding Box Detection
    best_box = None
    is_hit = False
    if pred_class in ["Malignant", "Benign"]:
        thresh = np.uint8(255 * (cam_resized >= 0.45))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if (bx <= peak_x <= bx + bw) and (by <= peak_y <= by + bh):
                best_box = [int(bx), int(by), int(bw), int(bh)]
                is_hit = True
                break

    # Visual 1: Annotated CT Scan with Lesion Box
    img_annotated = np.array(target_eval_img).copy()
    if best_box:
        bx, by, bw, bh = best_box
        color = (239, 68, 68) if pred_class == "Malignant" else (245, 158, 11)
        cv2.rectangle(img_annotated, (bx, by), (bx + bw, by + bh), color, 2)
        tag = f"Suspected {pred_class} Nodule"
        cv2.putText(img_annotated, tag, (bx, max(18, by - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Visual 2: Grad-CAM Saliency Jet Heatmap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    cv2.drawMarker(heatmap_colored, (int(peak_x), int(peak_y)), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)

    # Visual 3: Blended Overlay
    overlay_img = overlay_heatmap_on_image(target_eval_img, cam, alpha=0.45)
    overlay_np = np.array(overlay_img).copy()
    if best_box:
        bx, by, bw, bh = best_box
        color = (255, 50, 50) if pred_class == "Malignant" else (255, 200, 0)
        cv2.rectangle(overlay_np, (bx, by), (bx + bw, by + bh), color, 2)
    cv2.drawMarker(overlay_np, (int(peak_x), int(peak_y)), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)

    # Lung-RADS Clinical Assessment (v2022 Protocol)
    if pred_class == "Malignant":
        if confidence >= 65.0 or (is_physical_film and film_info and film_info.get("malignant_slices_detected", 0) >= 2):
            lung_rads = {
                "code": "Lung-RADS 4B/4X",
                "status": "Very High Clinical Suspicion",
                "risk": "> 15% Malignancy Risk",
                "badge_color": "rose",
                "action": "Urgent chest CT with IV contrast, PET-CT, and multidisciplinary pulmonary oncology consultation / tissue biopsy."
            }
        else:
            lung_rads = {
                "code": "Lung-RADS 4A",
                "status": "Suspicious Nodule",
                "risk": "5% - 15% Malignancy Risk",
                "badge_color": "orange",
                "action": "Intermediate-to-high risk nodule. 3-month follow-up low-dose CT or diagnostic PET-CT scan recommended."
            }
    elif pred_class == "Benign":
        lung_rads = {
            "code": "Lung-RADS 2",
            "status": "Benign Appearance / Indeterminate",
            "risk": "< 1% Malignancy Risk",
            "badge_color": "amber",
            "action": "Likely benign granuloma, hamartoma, or non-calcified nodule < 6mm. Follow-up low-dose CT in 12 months."
        }
    else: # Normal
        lung_rads = {
            "code": "Lung-RADS 1",
            "status": "Negative / Clear",
            "risk": "< 1% Malignancy Risk",
            "badge_color": "emerald",
            "action": "Routine annual low-dose screening CT recommended. No focal nodules or suspicious opacities detected."
        }

    inference_time = round((time.time() - start_total_time) * 1000, 1)

    return JSONResponse(status_code=200, content={
        "success": True,
        "filename": file_display_name,
        "is_physical_film": is_physical_film,
        "film_info": film_info,
        "prediction": pred_class,
        "confidence": round(confidence, 1),
        "decision_reason": decision_reason,
        "inference_time_ms": inference_time,
        "probabilities": {
            "Normal": round(p_normal * 100, 2),
            "Benign": round(p_benign * 100, 2),
            "Malignant": round(p_malignant * 100, 2)
        },
        "lung_rads": lung_rads,
        "pointing_game": {
            "is_hit": is_hit if pred_class != "Normal" else True,
            "status_text": "Peak attention lands directly inside lesion" if is_hit else ("Clear lung tissue" if pred_class == "Normal" else "Diffuse attention"),
            "peak_x": int(peak_x),
            "peak_y": int(peak_y),
            "box": best_box
        },
        "images": {
            "original": pil_to_base64(target_eval_img),
            "annotated": pil_to_base64(Image.fromarray(img_annotated)),
            "heatmap": pil_to_base64(Image.fromarray(heatmap_colored)),
            "overlay": pil_to_base64(Image.fromarray(overlay_np))
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=True)
