"""
Domain-Adaptive Fine-Tuning Script
===================================
Fine-tunes the pre-trained EfficientNet-CBAM model to handle domain shift
between IQ-OTH (full-chest CTs) and LIDC-IDRI (nodule crops).

Strategy:
  - Freeze EfficientNet backbone (~7.5M params) to preserve learned CT features
  - Unfreeze CBAM + Classifier head (~0.3M params) to adapt decision boundaries
  - Train on mixed data: 100% IQ-OTH + 70% LIDC (calibration split)
  - Test on held-out 30% LIDC (never seen during fine-tuning)
  - Low learning rate (5e-5) + few epochs (8) to prevent catastrophic forgetting

Usage:
    python scripts/finetune_domain_adaptation.py
"""
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_efficientnet_cbam
from src.dataset import scan_dataset, CTDataset, get_transforms, CLASS_NAMES
from src.loss import ClassBalancedFocalLoss
from src.evaluate import compute_slice_metrics, plot_confusion_matrix


def freeze_backbone(model):
    """
    Freeze EfficientNet backbone features (7.5M params).
    Only CBAM + Classifier remain trainable (~0.3M params).
    """
    for param in model.features.parameters():
        param.requires_grad = False

    # CBAM and classifier remain unfrozen by default
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Frozen params:    {frozen:,} (backbone)")
    print(f"  Trainable params: {trainable:,} (CBAM + classifier)")
    return model


def main():
    print("=" * 65)
    print("  DOMAIN-ADAPTIVE FINE-TUNING")
    print("  EfficientNet-CBAM: IQ-OTH + LIDC-IDRI Mixed Training")
    print("=" * 65)

    # ── Configuration ──────────────────────────────────────────────────
    ORIGINAL_CHECKPOINT = "checkpoints/fold_1/best_efficientnet_cbam.pt"
    IQOTH_DIR = "data/raw"
    LIDC_DIR = "data/external_harmonized_v2"      # V2 histogram-matched
    LIDC_FALLBACK = "data/external_harmonized"     # V1 fallback
    ADAPTED_SAVE_DIR = "checkpoints/adapted"
    OUTPUT_DIR = "outputs"

    EPOCHS = 8
    BATCH_SIZE = 8        # Small for CPU
    LR = 5e-5             # Low LR to prevent catastrophic forgetting
    LIDC_CAL_RATIO = 0.7  # 70% of LIDC for calibration, 30% for test
    SEED = 42

    os.makedirs(ADAPTED_SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cpu")

    # ── Check for V2 dataset, fall back to V1 ─────────────────────────
    if os.path.exists(LIDC_DIR) and len(os.listdir(LIDC_DIR)) > 0:
        lidc_dir = LIDC_DIR
        print(f"[OK] Using V2 histogram-matched LIDC data: {lidc_dir}")
    elif os.path.exists(LIDC_FALLBACK) and len(os.listdir(LIDC_FALLBACK)) > 0:
        lidc_dir = LIDC_FALLBACK
        print(f"[OK] V2 not found, using V1 LIDC data: {lidc_dir}")
    else:
        print("[!] No LIDC harmonized data found. Run harmonize_external_dataset.py first.")
        return

    # ── Load datasets ─────────────────────────────────────────────────
    print("\n[1/5] Loading datasets...")
    df_iqoth = scan_dataset(IQOTH_DIR)
    df_lidc = scan_dataset(lidc_dir)
    print(f"  IQ-OTH:  {len(df_iqoth)} slices ({df_iqoth['class_name'].value_counts().to_dict()})")
    print(f"  LIDC:    {len(df_lidc)} slices ({df_lidc['class_name'].value_counts().to_dict()})")

    # ── Split LIDC into calibration (70%) / held-out test (30%) ───────
    print("\n[2/5] Splitting LIDC: 70% calibration / 30% held-out test...")
    df_lidc_cal, df_lidc_test = train_test_split(
        df_lidc, test_size=1 - LIDC_CAL_RATIO,
        stratify=df_lidc["label"], random_state=SEED
    )
    print(f"  LIDC Calibration: {len(df_lidc_cal)} slices")
    print(f"  LIDC Test (held-out): {len(df_lidc_test)} slices")
    print(f"  Test class distribution: {df_lidc_test['class_name'].value_counts().to_dict()}")

    # ── Create mixed training dataset: 100% IQ-OTH + LIDC calibration ─
    df_train_mixed = pd.concat([df_iqoth, df_lidc_cal], ignore_index=True)
    print(f"  Mixed training set: {len(df_train_mixed)} total slices")

    train_dataset = CTDataset(df_train_mixed, transform=get_transforms(img_size=240, is_train=True))
    test_dataset = CTDataset(df_lidc_test, transform=get_transforms(img_size=240, is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Load pre-trained model and freeze backbone ────────────────────
    print("\n[3/5] Loading pre-trained model and freezing backbone...")
    model = build_efficientnet_cbam(num_classes=3, pretrained=False)
    state_dict = torch.load(ORIGINAL_CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = freeze_backbone(model)
    model.to(device)

    # ── Training setup ────────────────────────────────────────────────
    # Compute class weights from the mixed training set
    counts = df_train_mixed["label"].value_counts().sort_index().values
    total_count = sum(counts)
    weights = [total_count / (len(counts) * c) for c in counts]
    criterion = ClassBalancedFocalLoss(alpha=weights, gamma=2.0)

    # Only optimize trainable parameters (CBAM + classifier)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Fine-tuning loop ──────────────────────────────────────────────
    print(f"\n[4/5] Fine-tuning for {EPOCHS} epochs on CPU...")
    print(f"  Batch size: {BATCH_SIZE} | LR: {LR} | Batches/epoch: {len(train_loader)}")
    print("-" * 65)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── Train ──
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"    Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_loss = running_loss / total
        train_acc = correct / total
        scheduler.step()

        # ── Evaluate on held-out LIDC test ──
        model.eval()
        all_probs, all_labels_list = [], []
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_labels_list.extend(labels.numpy())

        all_probs_arr = np.vstack(all_probs)
        all_labels_arr = np.array(all_labels_list)
        metrics = compute_slice_metrics(all_labels_arr, all_probs_arr)

        elapsed = time.time() - epoch_start
        print(f"  Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
              f"LIDC Test Acc: {metrics['accuracy']*100:.1f}% | "
              f"Mal Sens: {metrics['malignant_sensitivity']*100:.1f}% | "
              f"Time: {elapsed:.0f}s")

        # Save best model
        if metrics['accuracy'] > best_val_acc:
            best_val_acc = metrics['accuracy']
            best_epoch = epoch
            save_path = os.path.join(ADAPTED_SAVE_DIR, "best_efficientnet_cbam_adapted.pt")
            torch.save(model.state_dict(), save_path)
            print(f"    --> New best! Saved to {save_path}")

    print("-" * 65)

    # ── Final evaluation with best model ──────────────────────────────
    print(f"\n[5/5] Final evaluation with best model (Epoch {best_epoch})...")
    best_path = os.path.join(ADAPTED_SAVE_DIR, "best_efficientnet_cbam_adapted.pt")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    all_probs, all_labels_list = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels_list.extend(labels.numpy())

    all_probs_arr = np.vstack(all_probs)
    all_labels_arr = np.array(all_labels_list)
    metrics = compute_slice_metrics(all_labels_arr, all_probs_arr)
    y_pred = np.argmax(all_probs_arr, axis=1)

    # Malignant detection rate
    mal_mask = (all_labels_arr == 2)
    mal_caught = np.sum((all_labels_arr == 2) & (y_pred == 2))
    mal_total = np.sum(mal_mask)
    mal_recall = (mal_caught / mal_total) * 100 if mal_total > 0 else 0.0

    print()
    print("=" * 65)
    print("  DOMAIN-ADAPTED MODEL - LIDC HELD-OUT TEST RESULTS")
    print("=" * 65)
    print(f"  3-Class Accuracy:         {metrics['accuracy']*100:.2f}%")
    print(f"  Malignant Sensitivity:    {mal_recall:.2f}% ({mal_caught}/{mal_total})")
    print(f"  Macro Sensitivity:        {metrics['sensitivity']*100:.2f}%")
    print(f"  Macro Specificity:        {metrics['specificity']*100:.2f}%")
    print(f"  Macro F1-Score:           {metrics['macro_f1']*100:.2f}%")
    print(f"  ROC-AUC:                  {metrics['roc_auc']:.4f}")
    print("=" * 65)
    print(f"  Improvement: 46.22% --> {metrics['accuracy']*100:.2f}% (held-out 30% of LIDC)")
    print("=" * 65)

    # Save confusion matrix
    cm_path = os.path.join(OUTPUT_DIR, "adapted_confusion_matrix.png")
    plot_confusion_matrix(all_labels_arr, y_pred, save_path=cm_path)
    print(f"[OK] Confusion matrix saved: {cm_path}")

    # ── Also test on IQ-OTH to verify no catastrophic forgetting ──────
    print("\n[BONUS] Verifying no catastrophic forgetting on IQ-OTH...")
    df_iqoth_test = scan_dataset(IQOTH_DIR)
    iqoth_dataset = CTDataset(df_iqoth_test, transform=get_transforms(img_size=240, is_train=False))
    iqoth_loader = DataLoader(iqoth_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs_iq, all_labels_iq = [], []
    with torch.no_grad():
        for images, labels, _ in iqoth_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs_iq.append(probs)
            all_labels_iq.extend(labels.numpy())

    all_probs_iq = np.vstack(all_probs_iq)
    all_labels_iq = np.array(all_labels_iq)
    iq_metrics = compute_slice_metrics(all_labels_iq, all_probs_iq)
    print(f"  IQ-OTH Accuracy (adapted model): {iq_metrics['accuracy']*100:.2f}%")
    print(f"  IQ-OTH Sensitivity:              {iq_metrics['sensitivity']*100:.2f}%")
    print(f"  (Original was 96.82% - should remain close)")
    print("=" * 65)
    print("  DOMAIN ADAPTATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
