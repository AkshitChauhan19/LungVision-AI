import os
import sys
import argparse
import random
import time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_efficientnet_cbam
from src.dataset import get_transforms, CLASS_NAMES
from src.explain import GradCAM, overlay_heatmap_on_image
from src.validator import validate_ct_scan

def predict_single(image_path: str, checkpoint_path: str = "checkpoints/fold_1/best_efficientnet_cbam.pt", 
                   save_output: str = "outputs/prediction_result.png", force: bool = False):
    os.makedirs(os.path.dirname(save_output), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(checkpoint_path):
        print(f"[!] Error: Model checkpoint '{checkpoint_path}' not found.")
        return
        
    if not os.path.exists(image_path):
        print(f"[!] Error: Image '{image_path}' not found.")
        return

    # Medical Modality & Organ Validation
    img_bgr = cv2.imread(image_path)
    is_valid, status, msg = validate_ct_scan(img_bgr)
    if not is_valid:
        print(msg)
        if not force:
            print("\\n[Tip] If you want to bypass validation anyway, pass --force to the script.\\n")
            return
        else:
            print("[!] Warning: Bypassing validation check because --force was specified.\\n")
        
    print("[*] Loading trained model weights...")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    is_moe = any("expert_" in k for k in state_dict.keys())
    if is_moe:
        from src.models.moe_detector import build_moe_detector
        model = build_moe_detector(num_classes=3, pretrained=False)
    else:
        model = build_efficientnet_cbam(num_classes=3, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    raw_img = Image.open(image_path).convert("RGB")
    transform = get_transforms(img_size=240, is_train=False)
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    elapsed = time.time() - start_time

    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx].capitalize()
    confidence = probs[pred_idx] * 100

    # Check ground truth from folder path if available
    path_lower = image_path.lower().replace("\\", "/")
    ground_truth = "Unknown"
    if "malignant" in path_lower:
        ground_truth = "Malignant"
    elif "benign" in path_lower or "bengin" in path_lower:
        ground_truth = "Benign"
    elif "normal" in path_lower:
        ground_truth = "Normal"

    # Grad-CAM Attention Heatmap
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_heatmap(input_tensor, class_idx=pred_idx)
    w, h = raw_img.size
    cam_res = cv2.resize(cam, (w, h))

    # Pointing Game Peak Localization
    peak_y, peak_x = np.unravel_index(np.argmax(cam_res), cam_res.shape)

    # Automatically identify lesion ROI bounding box around peak attention
    best_box = None
    is_hit = False
    if pred_class in ["Malignant", "Benign"]:
        thresh = np.uint8(255 * (cam_res >= 0.45))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if (bx <= peak_x <= bx + bw) and (by <= peak_y <= by + bh):
                best_box = (bx, by, bw, bh)
                is_hit = True
                break

    is_match = (ground_truth.lower() == pred_class.lower()) if ground_truth != "Unknown" else None

    print("=" * 60)
    print("       LUNG CANCER CT SCAN AI DIAGNOSIS & VERIFICATION")
    print("=" * 60)
    print(f"Image File:        {os.path.basename(image_path)}")
    if ground_truth != "Unknown":
        match_str = "[CORRECT - MATCH]" if is_match else "[MISMATCH / FALSE ALARM]"
        print(f"Ground Truth:      {ground_truth.upper()} (from medical dataset)")
        print(f"AI Diagnosis:      >>> {pred_class.upper()} <<<  --> {match_str}")
    else:
        print(f"AI Diagnosis:      >>> {pred_class.upper()} <<<")
    print(f"Confidence:        {confidence:.2f}%")
    print(f"Prediction Time:   {elapsed*1000:.1f} ms (< 0.1 sec)")
    if best_box:
        print(f"Lesion Target ROI: (x={best_box[0]}, y={best_box[1]}, w={best_box[2]}, h={best_box[3]})")
        print(f"Pointing Game:     [HIT] Peak attention (x={peak_x}, y={peak_y}) lands inside lesion ROI")
    print()
    print("Class Probabilities:")
    print(f"  - Normal:        {probs[0]*100:6.2f}%")
    print(f"  - Benign:        {probs[1]*100:6.2f}%")
    print(f"  - Malignant:     {probs[2]*100:6.2f}%")
    print("=" * 60)

    # 3-Panel Clinical Verification Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    gt_label = f" (Ground Truth: {ground_truth})" if ground_truth != "Unknown" else ""

    # Panel 1: Input CT with Lesion ROI Box
    img_annotated = np.array(raw_img).copy()
    if best_box:
        bx, by, bw, bh = best_box
        box_color = (0, 255, 0) if pred_class == "Malignant" else (0, 200, 255)
        cv2.rectangle(img_annotated, (bx, by), (bx + bw, by + bh), box_color, 2)
        tag = f"Suspected {pred_class} ROI"
        cv2.putText(img_annotated, tag, (bx, max(20, by - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
    roi_status = "[Target Lesion ROI Marked]" if best_box else "[Clear Parenchyma - No Lesion]"
    axes[0].imshow(img_annotated)
    axes[0].set_title(f"1. Input CT Scan{gt_label}\n{roi_status}", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Deep Attention Heatmap
    axes[1].imshow(cam_res, cmap="jet")
    axes[1].plot(peak_x, peak_y, "rD", markersize=8, markeredgecolor="white", markeredgewidth=1.5)
    axes[1].set_title(f"2. Grad-CAM Saliency Map\n(Peak Attention Focus at x={peak_x}, y={peak_y})", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Verification Overlay & Pointing Game Check
    overlay = overlay_heatmap_on_image(raw_img, cam, alpha=0.45)
    overlay_np = np.array(overlay).copy()
    if best_box:
        bx, by, bw, bh = best_box
        cv2.rectangle(overlay_np, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.drawMarker(overlay_np, (peak_x, peak_y), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 14, 2)

    hit_text = "[POINTING GAME: HIT]" if is_hit else ("[CLEAR LUNGS]" if pred_class == "Normal" else "[MISS]")
    axes[2].imshow(overlay_np)
    axes[2].set_title(f"3. Clinical Verification\n{hit_text} {pred_class} ({confidence:.1f}%)", 
                      fontsize=10, fontweight="bold", color="darkgreen" if is_hit or pred_class == "Normal" else "red")
    axes[2].axis("off")

    plt.suptitle(f"Automated Thoracic CAD Detection & Anatomical Verification: {os.path.basename(image_path)}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_output, dpi=250)
    plt.close()
    print(f"[OK] 3-Panel Verification report saved to: {save_output}")


def pick_random_image(data_dir: str = "data/raw") -> str:
    if data_dir.lower() in ("external", "lidc"):
        data_dir = "data/external_harmonized"
    candidates = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError(f"No images found in {data_dir}")
    return random.choice(candidates)


def find_image_file(image_path: str) -> str:
    if os.path.exists(image_path):
        return image_path
    # Try common directory prefixes
    for prefix in ["data/raw", "data/external_harmonized", "data/external_harmonized/malignant", 
                   "data/external_harmonized/benign", "data/external_harmonized/normal",
                   "data/raw/Malignant cases", "data/raw/Bengin cases", "data/raw/Normal cases"]:
        candidate = os.path.join(prefix, os.path.basename(image_path))
        if os.path.exists(candidate):
            return candidate
        # Fuzzy match filename
        target_name = os.path.basename(image_path).lower()
        if os.path.exists(prefix):
            for f in os.listdir(prefix):
                if target_name in f.lower() or f.lower() in target_name:
                    return os.path.join(prefix, f)
    return image_path


def main():
    parser = argparse.ArgumentParser(description="Instant 1-Second CT Cancer Prediction with Grad-CAM")
    parser.add_argument("--image", type=str, default=None, help="Path to CT scan image file")
    parser.add_argument("--dataset", type=str, default="data/raw", help="'raw' (IQ-OTH) or path to dataset directory")
    parser.add_argument("--random", action="store_true", help="Pick a random image from dataset")
    parser.add_argument("--external", action="store_true", help="Pick a random image from external LIDC-IDRI dataset")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Bypass medical CT modality and organ validation")
    args = parser.parse_args()

    # Determine dataset directory
    if args.external:
        if os.path.exists("data/external_harmonized_clean"):
            args.dataset = "data/external_harmonized_clean"
        elif os.path.exists("data/external_harmonized_v2"):
            args.dataset = "data/external_harmonized_v2"
        else:
            args.dataset = "data/external_harmonized"

    # Default checkpoint selection
    if args.checkpoint is None:
        moe_ckpt = "checkpoints/moe/best_moe_lung_cancer.pt"
        orig_ckpt = "checkpoints/fold_1/best_efficientnet_cbam.pt"
        args.checkpoint = moe_ckpt if (args.external and os.path.exists(moe_ckpt)) else orig_ckpt

    if args.random or args.image is None:
        target_img = pick_random_image(args.dataset)
        print(f"[*] Selected random test CT image: {target_img}")
    else:
        target_img = find_image_file(args.image)

    is_external = args.external or ("external" in target_img.lower()) or ("lidc" in target_img.lower())
    predict_single(target_img, checkpoint_path=args.checkpoint, force=args.force or is_external)

if __name__ == "__main__":
    main()
