import os
import sys
import argparse
import zipfile
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import build_efficientnet_cbam
from src.dataset import get_transforms, CLASS_NAMES
from src.explain import GradCAM, overlay_heatmap_on_image


def find_lidc_zip() -> str:
    candidates = [
        os.path.join(os.path.expanduser('~'), 'Downloads', 'archive (1).zip'),
        os.path.join(os.path.expanduser('~'), 'Downloads', 'archive.zip'),
        'data/archive (1).zip'
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError('LIDC archive (archive (1).zip) not found in Downloads or data/')


def evaluate_lidc_slice(slice_query: str = None, checkpoint_path: str = 'checkpoints/fold_1/best_efficientnet_cbam.pt',
                        output_path: str = 'outputs/lidc_verification_result.png'):
    zip_path = find_lidc_zip()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('[*] Opening LIDC-IDRI archive...')
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_names = z.namelist()

        target_img_name = None
        if slice_query:
            for n in all_names:
                if '/images/' in n and slice_query.lower() in n.lower():
                    target_img_name = n
                    break
            if not target_img_name:
                print(f"[!] Slice query '{slice_query}' not found. Selecting a high-consensus LIDC slice...")

        if not target_img_name:
            default_target = 'LIDC-IDRI-slices/LIDC-IDRI-0003/nodule-1/images/slice-3.png'
            if default_target in all_names:
                target_img_name = default_target
            else:
                for n in all_names:
                    if '/images/' in n:
                        target_img_name = n
                        break

        base_dir = target_img_name.split('images/')[0]
        slice_file = target_img_name.split('images/')[1]

        with z.open(target_img_name) as f:
            raw_img = Image.open(f).convert('RGB')

        radiologist_boxes = []
        masks = []
        for i in range(4):
            mask_name = f'{base_dir}mask-{i}/{slice_file}'
            if mask_name in all_names:
                with z.open(mask_name) as mf:
                    m_arr = np.array(Image.open(mf))
                    if m_arr.sum() > 0:
                        ys, xs = np.where(m_arr > 0)
                        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                        radiologist_boxes.append((i, box, len(xs)))
                        masks.append(m_arr)
                    else:
                        radiologist_boxes.append((i, None, 0))
            else:
                radiologist_boxes.append((i, None, 0))

        consensus_mask = np.zeros((raw_img.size[1], raw_img.size[0]), dtype=np.uint8)
        for m_arr in masks:
            consensus_mask = np.maximum(consensus_mask, (m_arr > 0).astype(np.uint8))

        agreeing_count = len(masks)
        if agreeing_count > 0:
            c_ys, c_xs = np.where(consensus_mask > 0)
            ground_truth_box = (int(c_xs.min()), int(c_ys.min()), int(c_xs.max()), int(c_ys.max()))
            ground_truth_class = 'MALIGNANT' if agreeing_count >= 3 else ('BENIGN' if agreeing_count == 1 else 'AMBIGUOUS')
        else:
            ground_truth_box = None
            ground_truth_class = 'NORMAL'

    print('[*] Loading trained model weights...')
    model = build_efficientnet_cbam(num_classes=3, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = get_transforms(img_size=240, is_train=False)
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx].capitalize()
    confidence = probs[pred_idx] * 100

    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_heatmap(input_tensor, class_idx=pred_idx)
    cam_res = cv2.resize(cam, raw_img.size)

    peak_y, peak_x = np.unravel_index(np.argmax(cam_res), cam_res.shape)

    is_hit = False
    if ground_truth_box:
        gx1, gy1, gx2, gy2 = ground_truth_box
        is_hit = (gx1 <= peak_x <= gx2) and (gy1 <= peak_y <= gy2)

    print('=' * 70)
    print('      LIDC-IDRI EXTERNAL BENCHMARK: TRUE RADIOLOGIST VERIFICATION')
    print('=' * 70)
    print(f'LIDC Slice:            {target_img_name}')
    print(f'Image Dimensions:      {raw_img.size[0]} x {raw_img.size[1]} px')
    print('-' * 70)
    print('4-RADIOLOGIST GROUND-TRUTH ANNOTATIONS:')
    for r_idx, r_box, p_cnt in radiologist_boxes:
        if r_box:
            rx1, ry1, rx2, ry2 = r_box
            w = rx2 - rx1
            h = ry2 - ry1
            print(f'  [+] Radiologist #{r_idx+1}:  Box: (x={rx1}..{rx2}, y={ry1}..{ry2}) | Size: {w}x{h} px | Pixels: {p_cnt}')
        else:
            print(f'  [-] Radiologist #{r_idx+1}:  Did not contour / Dismissed as non-suspicious')
    print(f'Radiologist Consensus: {agreeing_count} of 4 Radiologists confirmed nodule')
    if ground_truth_box:
        gx1, gy1, gx2, gy2 = ground_truth_box
        print(f'Consensus Bounding Box: Top-Left=({gx1}, {gy1}), Bottom-Right=({gx2}, {gy2})')
        print(f'Ground Truth Class:     {ground_truth_class}')
    else:
        print('Consensus Bounding Box: None (Healthy / Normal tissue)')
    print('-' * 70)
    print('AI MODEL DIAGNOSIS (UNSEEN EXTERNAL CT):')
    print(f'  AI Diagnosis:        >>> {pred_class.upper()} <<< (Confidence: {confidence:.2f}%)')
    print(f'  Normal Probability:    {probs[0]*100:6.2f}%')
    print(f'  Benign Probability:    {probs[1]*100:6.2f}%')
    print(f'  Malignant Probability: {probs[2]*100:6.2f}%')
    print('-' * 70)
    print('POINTING GAME LOCALIZATION EVALUATION:')
    print(f'  Grad-CAM Peak Focus:   (x={peak_x}, y={peak_y})')
    if ground_truth_box:
        hit_tag = '[HIT - TRUE POSITIVE]' if is_hit else '[MISS]'
        print(f'  Pointing Game Result:  {hit_tag}')
        if is_hit:
            print(f'  --> The AI\'s peak attention landed squarely inside the 4 doctors\' marked box!')
        else:
            print(f'  --> The AI\'s peak attention fell outside the 4 doctors\' marked box.')
    print('=' * 70)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    img_radiologist = np.array(raw_img).copy()
    if ground_truth_box:
        gx1, gy1, gx2, gy2 = ground_truth_box
        cv2.rectangle(img_radiologist, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        cv2.putText(img_radiologist, f"ROI: ({gx1},{gy1})-({gx2},{gy2})", 
                    (gx1, max(8, gy1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

    axes[0].imshow(img_radiologist)
    axes[0].set_title(f"1. LIDC-IDRI Input CT\n[True Radiologist Box (Green): {agreeing_count}/4 Doctors Agreed]", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cam_res, cmap="jet")
    axes[1].plot(peak_x, peak_y, "rD", markersize=8, markeredgecolor="white", markeredgewidth=1.5)
    axes[1].set_title(f"2. Grad-CAM Deep Attention Heatmap\n(AI Peak Focus: x={peak_x}, y={peak_y})", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    overlay = overlay_heatmap_on_image(raw_img, cam, alpha=0.45)
    overlay_np = np.array(overlay).copy()
    if ground_truth_box:
        gx1, gy1, gx2, gy2 = ground_truth_box
        cv2.rectangle(overlay_np, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        marker_color = (255, 255, 255) if is_hit else (0, 0, 255)
        cv2.drawMarker(overlay_np, (peak_x, peak_y), marker_color, cv2.MARKER_TILTED_CROSS, 10, 2)

    hit_status = "[POINTING GAME: HIT]" if is_hit else "[POINTING GAME: MISS]"
    status_color = "darkgreen" if is_hit else "red"
    axes[2].imshow(overlay_np)
    axes[2].set_title(f"3. Verification: Radiologist Box vs AI Focus\n{hit_status} - Diagnosis: {pred_class} ({confidence:.1f}%)", 
                      fontsize=10, fontweight="bold", color=status_color)
    axes[2].axis("off")

    patient_id = os.path.basename(os.path.dirname(os.path.dirname(target_img_name)))
    plt.suptitle(f'External Benchmark Verification on Unseen LIDC-IDRI CT: {patient_id} ({slice_file})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()
    print(f'[OK] Scientific verification image saved to: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model on LIDC-IDRI CT slice with actual radiologist ground-truth coordinates')
    parser.add_argument('--query', type=str, default='0003/nodule-1/images/slice-3', help='Patient/nodule/slice search substring')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fold_1/best_efficientnet_cbam.pt')
    parser.add_argument('--output', type=str, default='outputs/lidc_verification_result.png')
    args = parser.parse_args()

    evaluate_lidc_slice(slice_query=args.query, checkpoint_path=args.checkpoint, output_path=args.output)
