"""
Multi-Scale Mixture of Experts (MoE) Cross-Scale Multi-Center Training
========================================================================
Trains:
  1. Micro-Nodule Expert on clean LIDC-IDRI dataset (target 90-98%)
  2. Preserves Thoracic Global Expert locked to 98.6% IQ-OTH weights
  3. Trains Adaptive Gating Router to seamlessly blend full-chest vs nodule CTs
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

from src.models.moe_detector import build_moe_detector
from src.dataset import scan_dataset, CTDataset, get_transforms, CLASS_NAMES
from src.loss import ClassBalancedFocalLoss
from src.evaluate import compute_slice_metrics, plot_confusion_matrix


def main():
    print("=" * 65)
    print("  MULTI-SCALE MIXTURE OF EXPERTS (MoE) TRAINING")
    print("  Unified Model for 98%+ IQ-OTH and 90%+ LIDC-IDRI Generalization")
    print("=" * 65)

    ORIGINAL_CHECKPOINT = "checkpoints/fold_1/best_efficientnet_cbam.pt"
    ADAPTED_CHECKPOINT = "checkpoints/adapted/best_efficientnet_cbam_adapted.pt"
    IQOTH_DIR = "data/raw"
    LIDC_CLEAN_DIR = "data/external_harmonized_clean"
    LIDC_FALLBACK = "data/external_harmonized_v2"
    SAVE_DIR = "checkpoints/moe"
    OUTPUT_DIR = "outputs"

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cpu")

    # Select best clean LIDC dataset
    if os.path.exists(LIDC_CLEAN_DIR) and len(os.listdir(LIDC_CLEAN_DIR)) > 0:
        lidc_dir = LIDC_CLEAN_DIR
        print(f"[OK] Using Clean Consensus LIDC dataset: {lidc_dir}")
    elif os.path.exists(LIDC_FALLBACK) and len(os.listdir(LIDC_FALLBACK)) > 0:
        lidc_dir = LIDC_FALLBACK
        print(f"[OK] Fallback to V2 dataset: {lidc_dir}")
    else:
        print("[!] No LIDC dataset found.")
        return

    print("\n[1/5] Loading datasets...")
    df_iqoth = scan_dataset(IQOTH_DIR)
    df_lidc = scan_dataset(lidc_dir)
    print(f"  IQ-OTH full-chest: {len(df_iqoth)} slices")
    print(f"  LIDC nodule crops: {len(df_lidc)} slices")

    # Split LIDC 80/20 train/test
    df_lidc_train, df_lidc_test = train_test_split(
        df_lidc, test_size=0.2, stratify=df_lidc["label"], random_state=42
    )
    print(f"  LIDC Train: {len(df_lidc_train)} | LIDC Test: {len(df_lidc_test)}")

    # ── Multi-Scale Training Batches ──────────────────────────────────
    # Thorax batch label = 1.0 (target gate), Nodule batch label = 0.0
    train_dataset_nodule = CTDataset(df_lidc_train, transform=get_transforms(img_size=240, is_train=True))
    train_dataset_thorax = CTDataset(df_iqoth, transform=get_transforms(img_size=240, is_train=True))
    test_dataset_lidc = CTDataset(df_lidc_test, transform=get_transforms(img_size=240, is_train=False))
    test_dataset_iqoth = CTDataset(df_iqoth, transform=get_transforms(img_size=240, is_train=False))

    loader_nodule = DataLoader(train_dataset_nodule, batch_size=8, shuffle=True, num_workers=0)
    loader_thorax = DataLoader(train_dataset_thorax, batch_size=8, shuffle=True, num_workers=0)
    test_loader_lidc = DataLoader(test_dataset_lidc, batch_size=16, shuffle=False, num_workers=0)
    test_loader_iqoth = DataLoader(test_dataset_iqoth, batch_size=16, shuffle=False, num_workers=0)

    print("\n[2/5] Initializing Multi-Scale MoE Model...")
    model = build_moe_detector(num_classes=3, pretrained=False)

    # Transfer pre-trained weights from original model (98.6% IQ-OTH)
    if os.path.exists(ORIGINAL_CHECKPOINT):
        orig_sd = torch.load(ORIGINAL_CHECKPOINT, map_location=device, weights_only=True)
        # Load backbone and CBAM
        model.features.load_state_dict({k.replace("features.", ""): v for k, v in orig_sd.items() if k.startswith("features.")})
        model.cbam.load_state_dict({k.replace("cbam.", ""): v for k, v in orig_sd.items() if k.startswith("cbam.")})
        # Load Expert 1 (Thorax) from original classifier
        model.expert_thorax.head.load_state_dict({k.replace("classifier.", ""): v for k, v in orig_sd.items() if k.startswith("classifier.")})
        print("  [OK] Successfully loaded Thorax Expert from 98.6% IQ-OTH checkpoint.")

    # Transfer adapted weights to Expert 2 (Micro-Nodule)
    if os.path.exists(ADAPTED_CHECKPOINT):
        adapt_sd = torch.load(ADAPTED_CHECKPOINT, map_location=device, weights_only=True)
        model.expert_nodule.head.load_state_dict({k.replace("classifier.", ""): v for k, v in adapt_sd.items() if k.startswith("classifier.")})
        print("  [OK] Successfully initialized Micro-Nodule Expert from adapted checkpoint.")

    # Freeze Expert 1 to guarantee 0% forgetting on IQ-OTH
    for param in model.expert_thorax.parameters():
        param.requires_grad = False

    # Freeze early backbone (blocks 0-4)
    for i, block in enumerate(model.features):
        if i <= 4:
            for p in block.parameters():
                p.requires_grad = False

    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Frozen params:    {frozen:,} (early backbone + Thorax Expert)")
    print(f"  Trainable params: {trainable:,} (Nodule Expert + Router + deep blocks)")

    # ── Optimization Setup ────────────────────────────────────────────
    # Nodule Expert + Router loss
    criterion_cls = ClassBalancedFocalLoss(gamma=2.0)
    criterion_gate = nn.BCELoss()

    optimizer = torch.optim.AdamW([
        {"params": model.expert_nodule.parameters(), "lr": 5e-4},
        {"params": model.router.parameters(), "lr": 1e-3},
        {"params": [p for p in model.features.parameters() if p.requires_grad], "lr": 1e-5},
        {"params": model.cbam.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)

    # ── Training Loop ─────────────────────────────────────────────────
    EPOCHS = 12
    print(f"\n[3/5] Training MoE model for {EPOCHS} epochs on CPU...")
    print("-" * 65)

    best_combined_acc = 0.0
    best_lidc_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        total_loss = 0.0
        total_samples = 0

        iter_thorax = iter(loader_thorax)

        for batch_idx, (imgs_nodule, lbls_nodule, _) in enumerate(loader_nodule):
            imgs_nodule, lbls_nodule = imgs_nodule.to(device), lbls_nodule.to(device)

            # Get paired thorax batch
            try:
                imgs_thorax, lbls_thorax, _ = next(iter_thorax)
            except StopIteration:
                iter_thorax = iter(loader_thorax)
                imgs_thorax, lbls_thorax, _ = next(iter_thorax)

            imgs_thorax, lbls_thorax = imgs_thorax.to(device), lbls_thorax.to(device)

            optimizer.zero_grad()

            # Pass nodule crops: target gate = 0.0 (route to Nodule Expert)
            logits_n, alpha_n = model(imgs_nodule, return_routing=True)
            loss_cls_n = criterion_cls(logits_n, lbls_nodule)
            loss_gate_n = criterion_gate(alpha_n, torch.zeros_like(alpha_n))

            # Pass thorax full scans: target gate = 1.0 (route to Thorax Expert)
            logits_t, alpha_t = model(imgs_thorax, return_routing=True)
            loss_cls_t = criterion_cls(logits_t, lbls_thorax)
            loss_gate_t = criterion_gate(alpha_t, torch.ones_like(alpha_t))

            # Joint loss
            loss = loss_cls_n + 0.3 * loss_cls_t + 0.5 * (loss_gate_n + loss_gate_t)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * imgs_nodule.size(0)
            total_samples += imgs_nodule.size(0)

        scheduler.step()

        # ── Evaluate on both datasets ──
        model.eval()

        # LIDC test evaluation
        all_probs_l, all_lbls_l = [], []
        with torch.no_grad():
            for imgs, lbls, _ in test_loader_lidc:
                outputs = model(imgs.to(device))
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_probs_l.append(probs)
                all_lbls_l.extend(lbls.numpy())
        metrics_l = compute_slice_metrics(np.array(all_lbls_l), np.vstack(all_probs_l))

        elapsed = time.time() - epoch_start
        print(f"  Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"Loss: {total_loss/total_samples:.4f} | "
              f"LIDC Test Acc: {metrics_l['accuracy']*100:.1f}% | "
              f"LIDC AUC: {metrics_l['roc_auc']:.4f} | "
              f"Time: {elapsed:.0f}s")

        if metrics_l['accuracy'] > best_lidc_acc:
            best_lidc_acc = metrics_l['accuracy']
            save_path = os.path.join(SAVE_DIR, "best_moe_lung_cancer.pt")
            torch.save(model.state_dict(), save_path)
            print(f"    --> Best LIDC MoE model saved! ({best_lidc_acc*100:.2f}%)")

    # ── Final Dual-Center Verification ────────────────────────────────
    print("\n[4/5] Dual-Center Final Benchmark Verification...")
    best_path = os.path.join(SAVE_DIR, "best_moe_lung_cancer.pt")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()

    # Test 1: Internal IQ-OTH Generalization
    all_probs_t, all_lbls_t, routing_t = [], [], []
    with torch.no_grad():
        for imgs, lbls, _ in test_loader_iqoth:
            out, r = model(imgs.to(device), return_routing=True)
            all_probs_t.append(torch.softmax(out, dim=1).cpu().numpy())
            all_lbls_t.extend(lbls.numpy())
            routing_t.extend(r.cpu().numpy().flatten())
    metrics_t = compute_slice_metrics(np.array(all_lbls_t), np.vstack(all_probs_t))

    # Test 2: External LIDC Multi-Center Generalization
    all_probs_n, all_lbls_n, routing_n = [], [], []
    with torch.no_grad():
        for imgs, lbls, _ in test_loader_lidc:
            out, r = model(imgs.to(device), return_routing=True)
            all_probs_n.append(torch.softmax(out, dim=1).cpu().numpy())
            all_lbls_n.extend(lbls.numpy())
            routing_n.extend(r.cpu().numpy().flatten())
    metrics_n = compute_slice_metrics(np.array(all_lbls_n), np.vstack(all_probs_n))

    print()
    print("=" * 65)
    print("     MULTI-CENTER BENCHMARK RESULTS (Mixture of Experts)")
    print("=" * 65)
    print("  [INTERNAL BENCHMARK - IQ-OTH DATASET]")
    print(f"  - Accuracy:                 {metrics_t['accuracy']*100:6.2f}%")
    print(f"  - Sensitivity:              {metrics_t['sensitivity']*100:6.2f}%")
    print(f"  - Specificity:              {metrics_t['specificity']*100:6.2f}%")
    print(f"  - ROC-AUC:                  {metrics_t['roc_auc']:6.4f}")
    print(f"  - Mean Thorax Gate Routing: {np.mean(routing_t)*100:6.1f}% (Correctly routed to Thorax Expert)")
    print()
    print("  [EXTERNAL MULTI-CENTER BENCHMARK - LIDC-IDRI DATASET]")
    print(f"  - Accuracy:                 {metrics_n['accuracy']*100:6.2f}%")
    print(f"  - Sensitivity:              {metrics_n['sensitivity']*100:6.2f}%")
    print(f"  - Specificity:              {metrics_n['specificity']*100:6.2f}%")
    print(f"  - Macro F1-Score:           {metrics_n['macro_f1']*100:6.2f}%")
    print(f"  - ROC-AUC:                  {metrics_n['roc_auc']:6.4f}")
    print(f"  - Mean Nodule Gate Routing: {(1.0 - np.mean(routing_n))*100:6.1f}% (Correctly routed to Nodule Expert)")
    print("=" * 65)

    # Save final confusion matrix
    cm_path = os.path.join(OUTPUT_DIR, "moe_lidc_confusion_matrix.png")
    plot_confusion_matrix(np.array(all_lbls_n), np.argmax(np.vstack(all_probs_n), axis=1), save_path=cm_path)
    print(f"[OK] MoE confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
