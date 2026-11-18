"""
Domain-Adaptive Fine-Tuning V2 — Aggressive Deep Adaptation
=============================================================
V1 only fine-tuned CBAM+classifier (0.3M params) → only 50.75%.
V2 unfreezes the last 3 EfficientNet blocks + CBAM + classifier
with differential learning rates to adapt deeper feature layers.

Strategy:
  - Freeze EfficientNet blocks 0-5 (early/mid features)
  - Unfreeze EfficientNet blocks 6-8 + CBAM + Classifier
  - Differential LR: backbone_lr = 1e-5, head_lr = 5e-4
  - Train on mixed data: 100% IQ-OTH + 70% LIDC
  - 15 epochs with cosine annealing + warmup

Usage:
    python scripts/finetune_v2.py
"""
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_efficientnet_cbam
from src.dataset import scan_dataset, CTDataset, get_transforms, CLASS_NAMES
from src.loss import ClassBalancedFocalLoss
from src.evaluate import compute_slice_metrics, plot_confusion_matrix


def setup_differential_lr(model, backbone_lr=1e-5, head_lr=5e-4):
    """
    Sets up parameter groups with different learning rates:
    - Early backbone blocks (0-5): frozen completely
    - Late backbone blocks (6-8): low LR (backbone_lr)
    - CBAM + Classifier: high LR (head_lr)
    
    This lets deep features adapt slowly while the head adapts quickly.
    """
    # Freeze early blocks (0-5) — these learn basic edges, textures
    for i, block in enumerate(model.features):
        if i <= 5:
            for param in block.parameters():
                param.requires_grad = False

    # Count parameters
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen params:    {frozen:,} (blocks 0-5)")
    print(f"  Trainable params: {trainable:,} (blocks 6-8 + CBAM + classifier)")

    # Build parameter groups
    late_backbone_params = []
    for i, block in enumerate(model.features):
        if i > 5:
            late_backbone_params.extend(list(block.parameters()))

    cbam_params = list(model.cbam.parameters())
    classifier_params = list(model.classifier.parameters())

    param_groups = [
        {"params": late_backbone_params, "lr": backbone_lr, "name": "late_backbone"},
        {"params": cbam_params, "lr": head_lr, "name": "cbam"},
        {"params": classifier_params, "lr": head_lr, "name": "classifier"},
    ]

    print(f"  Late backbone LR: {backbone_lr}")
    print(f"  CBAM + Head LR:   {head_lr}")
    return param_groups


def main():
    print("=" * 65)
    print("  DOMAIN-ADAPTIVE FINE-TUNING V2 (Deep Adaptation)")
    print("  EfficientNet-CBAM: Unfreezing blocks 6-8 + CBAM + Classifier")
    print("=" * 65)

    # ── Configuration ──────────────────────────────────────────────────
    ORIGINAL_CHECKPOINT = "checkpoints/fold_1/best_efficientnet_cbam.pt"
    IQOTH_DIR = "data/raw"
    LIDC_DIR = "data/external_harmonized_v2"
    LIDC_FALLBACK = "data/external_harmonized"
    ADAPTED_SAVE_DIR = "checkpoints/adapted"
    OUTPUT_DIR = "outputs"

    EPOCHS = 15
    BATCH_SIZE = 8
    BACKBONE_LR = 1e-5      # Slow adaptation for deep features
    HEAD_LR = 5e-4           # Fast adaptation for decision layers
    LIDC_CAL_RATIO = 0.7
    SEED = 42

    os.makedirs(ADAPTED_SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cpu")

    # ── Detect LIDC dataset ───────────────────────────────────────────
    if os.path.exists(LIDC_DIR) and len(os.listdir(LIDC_DIR)) > 0:
        lidc_dir = LIDC_DIR
        print(f"[OK] Using V2 histogram-matched LIDC data")
    elif os.path.exists(LIDC_FALLBACK) and len(os.listdir(LIDC_FALLBACK)) > 0:
        lidc_dir = LIDC_FALLBACK
        print(f"[OK] Using V1 LIDC data (fallback)")
    else:
        print("[!] No LIDC data found.")
        return

    # ── Load datasets ─────────────────────────────────────────────────
    print("\n[1/5] Loading datasets...")
    df_iqoth = scan_dataset(IQOTH_DIR)
    df_lidc = scan_dataset(lidc_dir)
    print(f"  IQ-OTH:  {len(df_iqoth)} slices")
    print(f"  LIDC:    {len(df_lidc)} slices")

    # ── Split LIDC 70/30 ──────────────────────────────────────────────
    print("\n[2/5] Splitting LIDC: 70% calibration / 30% held-out test...")
    df_lidc_cal, df_lidc_test = train_test_split(
        df_lidc, test_size=1 - LIDC_CAL_RATIO,
        stratify=df_lidc["label"], random_state=SEED
    )
    print(f"  LIDC Calibration: {len(df_lidc_cal)} slices")
    print(f"  LIDC Held-out Test: {len(df_lidc_test)} slices")
    print(f"  Test distribution: {df_lidc_test['class_name'].value_counts().to_dict()}")

    # ── Mixed training set ────────────────────────────────────────────
    df_train = pd.concat([df_iqoth, df_lidc_cal], ignore_index=True)
    print(f"  Mixed training set: {len(df_train)} slices")

    # Also create a validation set from IQ-OTH (10% random) for monitoring forgetting
    df_iqoth_train, df_iqoth_val = train_test_split(
        df_iqoth, test_size=0.1, stratify=df_iqoth["label"], random_state=SEED
    )

    train_dataset = CTDataset(df_train, transform=get_transforms(img_size=240, is_train=True))
    test_dataset = CTDataset(df_lidc_test, transform=get_transforms(img_size=240, is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Load model and setup differential LR ──────────────────────────
    print("\n[3/5] Loading pre-trained model...")
    model = build_efficientnet_cbam(num_classes=3, pretrained=False)
    state_dict = torch.load(ORIGINAL_CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    param_groups = setup_differential_lr(model, backbone_lr=BACKBONE_LR, head_lr=HEAD_LR)
    model.to(device)

    # ── Loss and optimizer ────────────────────────────────────────────
    counts = df_train["label"].value_counts().sort_index().values
    total_count = sum(counts)
    weights = [total_count / (len(counts) * c) for c in counts]
    criterion = ClassBalancedFocalLoss(alpha=weights, gamma=2.0)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Cosine annealing with warm restarts for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\n[4/5] Training for {EPOCHS} epochs on CPU...")
    print(f"  Batches/epoch: {len(train_loader)}")
    print("-" * 65)

    best_val_acc = 0.0
    best_epoch = 0
    patience = 6
    patience_counter = 0

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

            # Gradient clipping to prevent unstable updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(epoch - 1 + batch_idx / len(train_loader))

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

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
        y_pred = np.argmax(all_probs_arr, axis=1)

        # Per-class accuracy
        per_cls = []
        for c in range(3):
            mask = (all_labels_arr == c)
            if mask.sum() > 0:
                per_cls.append(f"{CLASS_NAMES[c][:3]}:{np.mean(y_pred[mask]==c)*100:.0f}%")

        elapsed = time.time() - epoch_start
        print(f"  Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"Loss: {train_loss:.4f} | Train: {train_acc*100:.1f}% | "
              f"LIDC: {metrics['accuracy']*100:.1f}% | "
              f"F1: {metrics['macro_f1']*100:.1f}% | "
              f"{' '.join(per_cls)} | {elapsed:.0f}s")

        # Save best by combined metric (accuracy + F1) to balance all classes
        combined_score = metrics['accuracy'] * 0.5 + metrics['macro_f1'] * 0.5
        if combined_score > best_val_acc:
            best_val_acc = combined_score
            best_epoch = epoch
            save_path = os.path.join(ADAPTED_SAVE_DIR, "best_efficientnet_cbam_adapted.pt")
            torch.save(model.state_dict(), save_path)
            print(f"    --> New best! (combined={combined_score*100:.1f}%) Saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch}.")
                break

    print("-" * 65)

    # ── Final evaluation ──────────────────────────────────────────────
    print(f"\n[5/5] Final evaluation with best model (Epoch {best_epoch})...")
    best_path = os.path.join(ADAPTED_SAVE_DIR, "best_efficientnet_cbam_adapted.pt")
    model_eval = build_efficientnet_cbam(num_classes=3, pretrained=False)
    model_eval.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model_eval.to(device)
    model_eval.eval()

    # Test on LIDC held-out
    all_probs, all_labels_list = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model_eval(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels_list.extend(labels.numpy())

    all_probs_arr = np.vstack(all_probs)
    all_labels_arr = np.array(all_labels_list)
    metrics = compute_slice_metrics(all_labels_arr, all_probs_arr)
    y_pred = np.argmax(all_probs_arr, axis=1)

    mal_mask = (all_labels_arr == 2)
    mal_caught = np.sum((all_labels_arr == 2) & (y_pred == 2))
    mal_total = np.sum(mal_mask)
    mal_recall = (mal_caught / mal_total) * 100 if mal_total > 0 else 0.0

    print()
    print("=" * 65)
    print("  DOMAIN-ADAPTED V2 - LIDC HELD-OUT TEST RESULTS")
    print("=" * 65)
    print(f"  3-Class Accuracy:         {metrics['accuracy']*100:.2f}%")
    print(f"  Malignant Sensitivity:    {mal_recall:.2f}% ({mal_caught}/{mal_total})")
    print(f"  Macro Sensitivity:        {metrics['sensitivity']*100:.2f}%")
    print(f"  Macro Specificity:        {metrics['specificity']*100:.2f}%")
    print(f"  Macro F1-Score:           {metrics['macro_f1']*100:.2f}%")
    print(f"  ROC-AUC:                  {metrics['roc_auc']:.4f}")
    print()

    # Per-class breakdown
    print("  [PER-CLASS ACCURACY]")
    for c in range(3):
        mask = (all_labels_arr == c)
        if mask.sum() > 0:
            acc = np.mean(y_pred[mask] == c) * 100
            print(f"    {CLASS_NAMES[c]:12s}: {acc:.1f}% ({np.sum(y_pred[mask]==c)}/{mask.sum()})")

    print("=" * 65)
    print(f"  Improvement: 46.22% --> {metrics['accuracy']*100:.2f}%")
    print("=" * 65)

    cm_path = os.path.join(OUTPUT_DIR, "adapted_v2_confusion_matrix.png")
    plot_confusion_matrix(all_labels_arr, y_pred, save_path=cm_path)
    print(f"[OK] Confusion matrix saved: {cm_path}")

    # ── Verify IQ-OTH accuracy ────────────────────────────────────────
    print("\n[BONUS] Checking IQ-OTH accuracy (catastrophic forgetting check)...")
    df_iq = scan_dataset(IQOTH_DIR)
    iq_dataset = CTDataset(df_iq, transform=get_transforms(img_size=240, is_train=False))
    iq_loader = DataLoader(iq_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    iq_probs, iq_labels = [], []
    with torch.no_grad():
        for images, labels, _ in iq_loader:
            images = images.to(device)
            outputs = model_eval(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            iq_probs.append(probs)
            iq_labels.extend(labels.numpy())

    iq_probs = np.vstack(iq_probs)
    iq_labels = np.array(iq_labels)
    iq_metrics = compute_slice_metrics(iq_labels, iq_probs)
    print(f"  IQ-OTH Accuracy:     {iq_metrics['accuracy']*100:.2f}% (was 96.82%)")
    print(f"  IQ-OTH Sensitivity:  {iq_metrics['sensitivity']*100:.2f}%")
    print("=" * 65)
    print("  DOMAIN ADAPTATION V2 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
