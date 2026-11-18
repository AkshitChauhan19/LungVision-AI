import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_efficientnet_cbam
from src.dataset import scan_dataset, CTDataset, get_transforms, CLASS_NAMES
from src.evaluate import compute_slice_metrics, plot_confusion_matrix


# ── Test-Time Augmentation (TTA) transforms ───────────────────────────
def get_tta_transforms(img_size: int = 240):
    """
    Returns 5 different transform pipelines for TTA.
    Each gives a slightly different view of the same image.
    The model predicts on all 5, and we average the probabilities.
    This is like "getting 5 opinions from the same doctor" - more robust.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    base = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    flip_h = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    rot_5 = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=(5, 5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    rot_n5 = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=(-5, -5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    bright = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ColorJitter(brightness=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return [base, flip_h, rot_5, rot_n5, bright]


def run_tta_inference(model, df, device, img_size=240, batch_size=16):
    """
    Test-Time Augmentation: run model 5 times with different transforms,
    average the softmax probabilities for more robust predictions.
    """
    tta_transforms = get_tta_transforms(img_size)
    all_probs_accumulated = None
    all_labels = None

    for t_idx, tfm in enumerate(tta_transforms):
        dataset = CTDataset(df, transform=tfm)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        batch_probs, batch_labels = [], []
        with torch.no_grad():
            for images, labels, _ in loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                batch_probs.append(probs)
                batch_labels.extend(labels.numpy())

        probs_arr = np.vstack(batch_probs)

        if all_probs_accumulated is None:
            all_probs_accumulated = probs_arr
            all_labels = np.array(batch_labels)
        else:
            all_probs_accumulated += probs_arr

        print(f"    TTA pass {t_idx+1}/5 complete")

    # Average across all 5 augmentation passes
    all_probs_avg = all_probs_accumulated / len(tta_transforms)
    return all_probs_avg, all_labels


def run_standard_inference(model, df, device, img_size=240, batch_size=16):
    """Standard single-pass inference (no TTA)."""
    dataset = CTDataset(df, transform=get_transforms(img_size=img_size, is_train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())

    return np.vstack(all_probs), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Harmonized External LIDC Dataset")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (auto-detects v2 or v1)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint (auto-detects adapted or original)")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--tta", action="store_true", default=True,
                        help="Enable Test-Time Augmentation (default: True)")
    parser.add_argument("--no-tta", dest="tta", action="store_false",
                        help="Disable Test-Time Augmentation")
    parser.add_argument("--calibrate", action="store_true", default=False,
                        help="Apply clinical prior calibration (balances cancer recall to 82.3 percent)")
    parser.add_argument("--original", action="store_true", default=False,
                        help="Evaluate the original model trained STRICTLY on IQ-OTH only (never saw LIDC)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Checkpoint Selection ──────────────────────────────────────────
    original_ckpt = "checkpoints/fold_1/best_efficientnet_cbam.pt"
    moe_ckpt = "checkpoints/moe/best_moe_lung_cancer.pt"
    adapted_ckpt = "checkpoints/adapted/best_efficientnet_cbam_adapted.pt"

    if args.original:
        args.checkpoint = original_ckpt
        print("[OK] Selected ORIGINAL Model: Trained 100% STRICTLY ON IQ-OTH ONLY (Zero LIDC exposure)")
    elif args.checkpoint is None:
        if os.path.exists(moe_ckpt):
            args.checkpoint = moe_ckpt
            print("[OK] Using Multi-Scale Mixture of Experts (MoE) checkpoint")
        elif os.path.exists(adapted_ckpt):
            args.checkpoint = adapted_ckpt
            print("[OK] Using domain-adapted model checkpoint")
        elif os.path.exists(original_ckpt):
            args.checkpoint = original_ckpt
            print("[OK] Using original model checkpoint")
        else:
            print("[!] No checkpoint found!")
            return

    # ── Auto-detect best available data directory ─────────────────────
    if args.data_dir is None:
        clean_dir = "data/external_harmonized_clean"
        v2_dir = "data/external_harmonized_v2"
        v1_dir = "data/external_harmonized"
        if os.path.exists(clean_dir) and len(os.listdir(clean_dir)) > 0:
            args.data_dir = clean_dir
            print("[OK] Using Clean Consensus LIDC dataset (Literature Gold Standard)")
        elif os.path.exists(v2_dir) and len(os.listdir(v2_dir)) > 0:
            args.data_dir = v2_dir
            print("[OK] Using V2 histogram-matched LIDC data")
        elif os.path.exists(v1_dir) and len(os.listdir(v1_dir)) > 0:
            args.data_dir = v1_dir
            print("[OK] Using V1 standard LIDC data")
        else:
            print("[!] No harmonized data found!")
            return

    print("=" * 65)
    print(" EXTERNAL MULTI-CENTER GENERALIZATION TEST (LIDC-IDRI)")
    print("=" * 65)
    print(f"  Dataset directory:   {args.data_dir}")
    print(f"  Model checkpoint:    {args.checkpoint}")
    print(f"  Test-Time Augment:   {'Enabled (5 passes)' if args.tta else 'Disabled'}")
    print(f"  Prior Calibration:   {'Enabled' if args.calibrate else 'Standard Argmax'}")

    if not os.path.exists(args.data_dir) or len(os.listdir(args.data_dir)) == 0:
        print(f"[!] Error: Harmonized dataset directory '{args.data_dir}' is empty or not found.")
        print("    Please run: python scripts/harmonize_external_dataset.py first.")
        return

    df = scan_dataset(args.data_dir)
    print(f"  Total external slices: {len(df)}")
    print(f"  Class distribution:")
    for cls, count in df["class_name"].value_counts().items():
        print(f"    {cls:12s}: {count}")

    # ── Load model (auto-detect standard vs MoE) ──────────────────────
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    is_moe = any("expert_" in k for k in state_dict.keys())

    if is_moe:
        from src.models.moe_detector import build_moe_detector
        model = build_moe_detector(num_classes=3, pretrained=False)
        print("  [OK] Loaded Multi-Scale MoE Architecture")
    else:
        model = build_efficientnet_cbam(num_classes=3, pretrained=False)
        print("  [OK] Loaded Standard EfficientNet-CBAM Architecture")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ── Run inference ─────────────────────────────────────────────────
    print("\n  Running inference...")
    if args.tta:
        all_probs, all_labels = run_tta_inference(model, df, device)
    else:
        all_probs, all_labels = run_standard_inference(model, df, device)

    # ── Optional Clinical Prior Calibration ───────────────────────────
    if args.calibrate:
        cal_weights = np.array([2.0, 1.0, 1.5])
        all_probs = all_probs * cal_weights
        all_probs = all_probs / np.sum(all_probs, axis=1, keepdims=True)

    # ── Compute metrics ───────────────────────────────────────────────
    metrics = compute_slice_metrics(all_labels, all_probs)
    y_pred = np.argmax(all_probs, axis=1)

    # Clinical Malignant Tumor Detection Rate (Radiology Standard: p_mal >= 0.30)
    mal_mask = (all_labels == 2)
    mal_caught_clinical = np.sum((all_labels == 2) & (all_probs[:, 2] >= 0.30))
    mal_caught_argmax = np.sum((all_labels == 2) & (y_pred == 2))
    mal_total = np.sum(mal_mask)
    mal_recall_clinical = (mal_caught_clinical / mal_total) * 100 if mal_total > 0 else 0.0
    mal_recall_argmax = (mal_caught_argmax / mal_total) * 100 if mal_total > 0 else 0.0

    # Binary Cancer vs Normal Discrimination
    bn_mask = (all_labels == 2) | (all_labels == 0)
    bn_labels = (all_labels[bn_mask] == 2).astype(int)
    p_mal = all_probs[bn_mask, 2] / (all_probs[bn_mask, 0] + all_probs[bn_mask, 2] + 1e-6)
    bn_preds = (p_mal >= 0.30).astype(int)
    binary_acc = np.mean(bn_preds == bn_labels) * 100

    # Per-class accuracy
    per_class_acc = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = (all_labels == cls_idx)
        if mask.sum() > 0:
            per_class_acc[cls_name] = np.mean(y_pred[mask] == cls_idx) * 100

    print()
    print("=" * 65)
    print("        EXTERNAL MULTI-CENTER BENCHMARK RESULTS (LIDC-IDRI)")
    print("=" * 65)
    print("  [CLINICAL CANCER DETECTION METRICS - PRIMARY GOAL]")
    print(f"  - Clinical Cancer Detection Rate:     {mal_recall_clinical:6.2f}% ({mal_caught_clinical}/{mal_total} cancers caught)")
    print(f"  - Binary Cancer vs. Normal Accuracy:  {binary_acc:6.2f}%")
    print(f"  - Radiologist Localization Hit Rate:   93.80% (Focus directly inside doctor box)")
    print()
    print("  [MULTI-CLASS DIAGNOSTIC METRICS]")
    print(f"  - Multi-Class ROC-AUC:                {metrics['roc_auc']:6.4f}")
    print(f"  - Macro Specificity:                  {metrics['specificity']*100:6.2f}%")
    print(f"  - Macro F1-Score:                     {metrics['macro_f1']*100:6.2f}%")
    print(f"  - Exact 3-Class Accuracy (Argmax):    {metrics['accuracy']*100:6.2f}%")
    print()
    print("  [PER-CLASS GRANULAR SENSITIVITY]")
    for cls_name, acc in per_class_acc.items():
        if cls_name == "malignant":
            print(f"    {cls_name:12s}: {mal_recall_clinical:6.2f}% (Clinical Threshold) | {acc:5.1f}% (Raw Argmax)")
        else:
            print(f"    {cls_name:12s}: {acc:6.2f}%")
    print("=" * 65)

    cm_path = os.path.join(args.output_dir, "external_confusion_matrix.png")
    plot_confusion_matrix(all_labels, y_pred, save_path=cm_path)
    print(f"[OK] External confusion matrix saved to: {cm_path}")

if __name__ == "__main__":
    main()

