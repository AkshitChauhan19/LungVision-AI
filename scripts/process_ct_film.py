import os
import sys
import argparse
import time
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_efficientnet_cbam
from src.dataset import get_transforms, CLASS_NAMES
from src.explain import GradCAM, overlay_heatmap_on_image
from src.film_processor import rectify_perspective, enhance_medical_contrast, detect_and_decompose_grid
from src.validator import validate_ct_scan

def analyze_ct_film(film_path: str, rows: int = None, cols: int = None, 
                    output_dir: str = "outputs/film_analysis",
                    checkpoint: str = "checkpoints/fold_1/best_efficientnet_cbam.pt",
                    force: bool = False):
    
    if not os.path.exists(film_path):
        print(f"[!] Error: CT film image '{film_path}' not found.")
        print("    Please place your patient CT film image at that path.")
        return

    print("=" * 65)
    print("      AUTOMATED PHYSICAL CT SCAN FILM ANALYSIS ENGINE")
    print("=" * 65)
    print(f"Film Image:        {film_path}")
    print(f"Model Checkpoint:  {checkpoint}")

    # Stage 1: Validate Film-Level Modality
    raw_film = cv2.imread(film_path)
    if raw_film is None:
        print(f"[!] Error: Could not read image from '{film_path}'.")
        return

    hsv = cv2.cvtColor(raw_film, cv2.COLOR_BGR2HSV)
    sat_mean = np.mean(hsv[:, :, 1])
    if sat_mean > 52.0 and not force:
        print("\n" + "=" * 65)
        print("                 INPUT VALIDATION REJECTED")
        print("=" * 65)
        print("[!] REASON: The uploaded image is NOT a medical CT scan.")
        print(f"    - Detected high color saturation ({sat_mean:.1f} > 52.0 threshold).")
        print("    - Medical CT scans are grayscale radiographic images.")
        print("    - Please upload a physical CT scan film sheet.")
        print("=" * 65)
        print("[Tip] If you want to bypass validation anyway, pass --force to the script.\n")
        return

    # Check if the user accidentally passed a SINGLE CT slice (e.g. Abdomen or Brain) instead of a film sheet
    is_single_valid, single_status, single_msg = validate_ct_scan(raw_film)
    if single_status == "NOT_LUNG" and not force:
        print("\n" + single_msg)
        print("[Tip] To diagnose a single 2D CT slice, please use: python scripts/predict.py --image <path>\n")
        return

    # Sketch / Manga / Line Art detection on whole image
    gray_film = cv2.cvtColor(raw_film, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray_film, cv2.CV_64F).var()
    if lap_var > 3000.0 and not force:
        print("\n" + "=" * 65)
        print("                 INPUT VALIDATION REJECTED")
        print("=" * 65)
        print("[!] REASON: Artwork / Drawing / Sketch wallpaper detected, NOT a medical CT scan.")
        print(f"    - Detected high-frequency line art edges (Variance: {lap_var:.0f} > 3000).")
        print("    - Please upload a genuine medical CT scan film sheet.")
        print("=" * 65)
        print("[Tip] If you want to bypass validation anyway, pass --force to the script.\n")
        return

    os.makedirs(output_dir, exist_ok=True)
    slices_dir = os.path.join(output_dir, "extracted_slices")
    heatmaps_dir = os.path.join(output_dir, "gradcam_heatmaps")
    os.makedirs(slices_dir, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)

    # Step 1 & 2: Load and Enhance
    print("[*] Step 1/4: Rectifying camera perspective and borders...")
    rectified = rectify_perspective(raw_film)
    print("[*] Step 2/4: Applying CLAHE contrast restoration & noise filtering...")
    enhanced = enhance_medical_contrast(rectified)

    # Step 3: Decompose grid
    print("[*] Step 3/4: Decomposing film sheet into individual CT slices...")
    slice_data = detect_and_decompose_grid(enhanced, rows=rows, cols=cols)
    raw_num = len(slice_data)
    print(f"    --> Extracted {raw_num} candidate tiles from grid.")

    # Stage 2: Filter out non-lung tiles (hands, neck, abdomen, black borders)
    valid_slices = []
    for idx, (clean_slice, bbox) in enumerate(slice_data):
        is_valid, status, msg = validate_ct_scan(clean_slice, threshold_lung=0.08)
        if is_valid or force:
            valid_slices.append((idx + 1, clean_slice, bbox))
        else:
            print(f"    [-] Tile {idx+1:02d} filtered out: {status} (Non-thoracic/artifact)")

    num_slices = len(valid_slices)
    # A real physical CT film sheet contains at least 3 valid lung slices and >= 25% of grid tiles
    if (num_slices < 3 or (num_slices / float(max(1, raw_num)) < 0.20)) and not force:
        print("\n" + "=" * 65)
        print("                 INPUT VALIDATION REJECTED")
        print("=" * 65)
        print("[!] REASON: The uploaded image is NOT a medical CT scan film sheet.")
        print(f"    - Only {num_slices} of {raw_num} grid tiles resembled thoracic anatomy.")
        print("    - A medical CT film sheet contains a recurring series of thoracic CT scans.")
        print("    - Uploaded file appears to be a single photo, artwork, or non-medical wallpaper.")
        print("=" * 65)
        print("[Tip] If you want to bypass validation anyway, pass --force to the script.\n")
        return

    print(f"    --> Verified {num_slices} true thoracic lung slices for diagnosis.")

    # Step 4: Run AI inference & Grad-CAM
    print(f"[*] Step 4/4: Running EfficientNet-B1 + CBAM AI diagnostics on {num_slices} lung slices...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_efficientnet_cbam(num_classes=3, pretrained=False)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = get_transforms(img_size=240, is_train=False)
    grad_cam = GradCAM(model, model.features[-1])

    results = []
    start_t = time.time()

    for orig_slice_num, clean_slice, bbox in valid_slices:
        # Save extracted slice
        slice_save_path = os.path.join(slices_dir, f"slice_{orig_slice_num:02d}.png")
        cv2.imwrite(slice_save_path, clean_slice)

        # PyTorch Inference
        pil_img = Image.fromarray(cv2.cvtColor(clean_slice, cv2.COLOR_BGR2RGB))
        tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_idx].capitalize()
        confidence = probs[pred_idx] * 100

        # Generate Grad-CAM overlay
        cam = grad_cam.generate_heatmap(tensor, class_idx=pred_idx)
        overlay = overlay_heatmap_on_image(pil_img, cam, alpha=0.5)

        hmap_save_path = os.path.join(heatmaps_dir, f"heatmap_slice_{orig_slice_num:02d}.png")
        cv2.imwrite(hmap_save_path, cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR))

        results.append({
            "slice_num": orig_slice_num,
            "pred_class": pred_class,
            "confidence": confidence,
            "probs": probs,
            "logits": logits.cpu().numpy()[0],
            "slice_img": pil_img,
            "overlay_img": overlay
        })

    elapsed = time.time() - start_t

    # Volumetric Patient-Level Pooling (Equation 6: Top-k Pooling, k=3)
    k = min(3, num_slices)
    malignant_scores = [r["probs"][2] for r in results]
    top_k_indices = np.argsort(malignant_scores)[-k:]

    pooled_logits = np.mean([results[i]["logits"] for i in top_k_indices], axis=0)
    exp_logits = np.exp(pooled_logits - np.max(pooled_logits))
    patient_probs = exp_logits / np.sum(exp_logits)
    patient_pred_idx = np.argmax(patient_probs)
    patient_diagnosis = CLASS_NAMES[patient_pred_idx].upper()
    patient_conf = patient_probs[patient_pred_idx] * 100

    critical_slices = [results[i]["slice_num"] for i in top_k_indices]
    crit_str = ", ".join([f"Slice #{s}" for s in critical_slices]) if critical_slices else "None"

    # Print Clinical Patient Report
    print()
    print("=" * 65)
    print("          PATIENT-LEVEL CLINICAL DIAGNOSTIC REPORT")
    print("=" * 65)
    print(f"Total Lung Slices Processed: {num_slices} slices in {elapsed:.2f} seconds")
    print(f"Patient Overall Verdict:     >>> {patient_diagnosis} <<<")
    print(f"Patient Confidence:          {patient_conf:.2f}%")
    print()
    print("Patient Risk Distribution (Top-k Volumetric Pooling):")
    print(f"  - Normal:                  {patient_probs[0]*100:6.2f}%")
    print(f"  - Benign:                  {patient_probs[1]*100:6.2f}%")
    print(f"  - Malignant:               {patient_probs[2]*100:6.2f}%")
    print(f"Critical Slices:             {crit_str}")
    print("=" * 65)

    print()
    print("--- Verified Lung Slice Breakdown ---")
    print("Slice #  | Diagnosis   | Confidence | Normal% | Benign% | Malignant%")
    print("-----------------------------------------------------------------")
    for r in results:
        star = " <-- CRITICAL" if r["slice_num"] in critical_slices else ""
        print(f"Slice {r['slice_num']:02d} | {r['pred_class']:<11} | {r['confidence']:6.2f}%   | {r['probs'][0]*100:5.1f}%  | {r['probs'][1]*100:5.1f}%  | {r['probs'][2]*100:5.1f}% {star}")
    print("=" * 65)

    # Generate Unified High-Resolution Visual Report
    build_visual_report(results, patient_diagnosis, patient_conf, critical_slices, 
                        output_path=os.path.join(output_dir, "film_diagnostic_report.png"))
    
    print(f"[OK] Full visual diagnostic report saved to: {output_dir}/film_diagnostic_report.png")
    print(f"[OK] All {num_slices} verified lung slices saved to: {slices_dir}/")
    print(f"[OK] Individual Grad-CAM heatmaps saved to: {heatmaps_dir}/")


def build_visual_report(results, patient_diag, patient_conf, critical_slices, output_path):
    n = len(results)
    grid_cols = min(4, n)
    grid_rows = int(np.ceil(n / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4 * grid_cols, 4.2 * grid_rows))
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([[axes]])
    elif grid_rows == 1:
        axes = np.array([axes])
    elif grid_cols == 1:
        axes = np.array([[ax] for ax in axes])

    banner_color = "#d9534f" if patient_diag == "MALIGNANT" else ("#f0ad4e" if patient_diag == "BENIGN" else "#5cb85c")
    crit_str = ", ".join([f"#{s}" for s in critical_slices])
    fig.suptitle(f"CT Film CAD Analysis: Patient Verdict = {patient_diag} ({patient_conf:.1f}%)\nCritical Slices: {crit_str}",
                 fontsize=15, fontweight="bold", color=banner_color, y=0.99)

    for i in range(grid_rows * grid_cols):
        r, c = i // grid_cols, i % grid_cols
        ax = axes[r, c]
        if i < n:
            item = results[i]
            overlay = item["overlay_img"]
            ax.imshow(overlay)
            
            is_crit = item["slice_num"] in critical_slices
            star = " [CRITICAL]" if is_crit else ""
            color = "red" if item["pred_class"] == "Malignant" else ("orange" if item["pred_class"] == "Benign" else "green")
            title_txt = f"Slice {item['slice_num']:02d}: {item['pred_class']} ({item['confidence']:.1f}%){star}"
            ax.set_title(title_txt, fontsize=9, color=color, fontweight="bold" if is_crit else "normal")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Process Physical Hardcopy CT Scan Film Sheet")
    parser.add_argument("--film", type=str, default="data/patient_film.jpg", help="Path to CT film photo")
    parser.add_argument("--rows", type=int, default=None, help="Optional manual row count in film grid")
    parser.add_argument("--cols", type=int, default=None, help="Optional manual col count in film grid")
    parser.add_argument("--output_dir", type=str, default="outputs/film_analysis", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fold_1/best_efficientnet_cbam.pt")
    parser.add_argument("--force", action="store_true", help="Bypass CT modality and lung organ validation")
    args = parser.parse_args()

    analyze_ct_film(args.film, rows=args.rows, cols=args.cols, 
                    output_dir=args.output_dir, checkpoint=args.checkpoint, force=args.force)

if __name__ == "__main__":
    main()
