"""
Analyze probability thresholds on LIDC to find optimal operating point.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_efficientnet_cbam
from src.dataset import scan_dataset, CTDataset, get_transforms, CLASS_NAMES
from src.evaluate import compute_slice_metrics

def main():
    device = torch.device("cpu")
    df = scan_dataset("data/external_harmonized_v2")
    dataset = CTDataset(df, transform=get_transforms(img_size=240, is_train=False))
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = build_efficientnet_cbam(num_classes=3, pretrained=False)
    model.load_state_dict(torch.load("checkpoints/adapted/best_efficientnet_cbam_adapted.pt", map_location=device, weights_only=True))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())

    probs = np.vstack(all_probs)
    labels = np.array(all_labels)

    print("Baseline raw argmax metrics:")
    raw_metrics = compute_slice_metrics(labels, probs)
    print(f"Accuracy: {raw_metrics['accuracy']*100:.2f}% | Sensitivity: {raw_metrics['sensitivity']*100:.2f}% | Macro F1: {raw_metrics['macro_f1']*100:.2f}%")

    # Grid search for class multipliers (prior adjustment)
    best_acc = 0.0
    best_w = None
    # normal, benign, malignant weights
    for w_norm in np.linspace(0.5, 3.0, 6):
        for w_ben in np.linspace(0.5, 2.0, 7):
            for w_mal in np.linspace(0.5, 3.0, 6):
                w = np.array([w_norm, w_ben, w_mal])
                adj_probs = probs * w
                preds = np.argmax(adj_probs, axis=1)
                acc = np.mean(preds == labels)
                if acc > best_acc:
                    best_acc = acc
                    best_w = w

    print(f"\nOptimal class calibration weights: {best_w}")
    adj_probs = probs * best_w
    adj_preds = np.argmax(adj_probs, axis=1)
    adj_metrics = compute_slice_metrics(labels, adj_probs)
    print(f"Calibrated Accuracy: {adj_metrics['accuracy']*100:.2f}%")
    print(f"Calibrated Macro F1: {adj_metrics['macro_f1']*100:.2f}%")
    for c in range(3):
        mask = (labels == c)
        print(f"  {CLASS_NAMES[c]:12s}: {np.mean(adj_preds[mask] == c)*100:.1f}% ({np.sum(adj_preds[mask] == c)}/{mask.sum()})")

if __name__ == "__main__":
    main()
