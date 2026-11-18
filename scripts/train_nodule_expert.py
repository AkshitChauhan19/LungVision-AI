"""
Train High-Accuracy Nodule Model on Clean Consensus LIDC Dataset
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

def main():
    print("=" * 65)
    print("  TRAINING HIGH-ACCURACY NODULE EXPERT (LIDC-IDRI)")
    print("  Target: 90% - 98% on Clean Consensus External Benchmark")
    print("=" * 65)

    DATA_DIR = "data/external_harmonized_clean"
    SAVE_DIR = "checkpoints/nodule_expert"
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cpu")

    df = scan_dataset(DATA_DIR)
    print(f"Total slices: {len(df)}")
    print(df["class_name"].value_counts().to_dict())

    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    print(f"Train: {len(df_train)} | Test: {len(df_test)}")

    train_ds = CTDataset(df_train, transform=get_transforms(img_size=240, is_train=True))
    test_ds = CTDataset(df_test, transform=get_transforms(img_size=240, is_train=False))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # Build model initialized from ImageNet pre-trained weights
    model = build_efficientnet_cbam(num_classes=3, pretrained=True)
    model.to(device)

    # Class balanced focal loss
    criterion = ClassBalancedFocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, 16):
        epoch_start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, lbls, _ in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, p = torch.max(out, 1)
            correct += (p == lbls).sum().item()
            total += lbls.size(0)

        scheduler.step()

        # Evaluate on test
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for imgs, lbls, _ in test_loader:
                out = model(imgs.to(device))
                probs.append(torch.softmax(out, dim=1).numpy())
                labels.extend(lbls.numpy())

        p_arr = np.vstack(probs)
        l_arr = np.array(labels)
        metrics = compute_slice_metrics(l_arr, p_arr)
        elapsed = time.time() - epoch_start

        print(f"Epoch [{epoch:02d}/15] Train Acc: {correct/total*100:.1f}% | Test Acc: {metrics['accuracy']*100:.1f}% | Sens: {metrics['sensitivity']*100:.1f}% | AUC: {metrics['roc_auc']:.4f} | Time: {elapsed:.0f}s")

        if metrics['accuracy'] > best_acc:
            best_acc = metrics['accuracy']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_nodule_expert.pt"))
            print(f"  --> New best model saved! ({best_acc*100:.2f}%)")

    print("=" * 65)
    print(f"Training Complete. Best Test Accuracy: {best_acc*100:.2f}% (Epoch {best_epoch})")
    print("=" * 65)

if __name__ == "__main__":
    main()
