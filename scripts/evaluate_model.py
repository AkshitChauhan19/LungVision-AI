import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import build_efficientnet_cbam
from src.dataset import scan_dataset, create_patient_folds, CTDataset, get_transforms, CLASS_NAMES
from src.evaluate import compute_slice_metrics, aggregate_patient_topk, plot_confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Model on Full Validation Set")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/fold_1/best_efficientnet_cbam.pt")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(" FULL MODEL EVALUATION & CONFUSION MATRIX")
    print("=" * 60)
    print(f"Loading checkpoint: {args.checkpoint}")

    model = build_efficientnet_cbam(num_classes=3, pretrained=False)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    df = scan_dataset(args.data_dir)
    folds = create_patient_folds(df, n_splits=5)
    train_df, val_df = folds[0]

    print(f"Validation Patients: {val_df['patient_id'].nunique()} ({len(val_df)} slices)")

    val_dataset = CTDataset(val_df, transform=get_transforms(img_size=240, is_train=False))
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    all_probs, all_labels, all_patients = [], [], []
    with torch.no_grad():
        for images, labels, pids in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())
            all_patients.extend(pids)

    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)

    slice_metrics = compute_slice_metrics(all_labels, all_probs)

    df_res = pd.DataFrame({
        "patient_id": all_patients,
        "label": all_labels,
        "prob": list(all_probs)
    })
    patient_metrics = aggregate_patient_topk(df_res, k=3)

    print()
    print("=" * 60)
    print("             FINAL CLINICAL PERFORMANCE REPORT")
    print("=" * 60)
    print("2D Slice-Level Metrics:")
    print(f"  - Slice Accuracy:          {slice_metrics['accuracy']*100:6.2f}%")
    print(f"  - Slice Sensitivity:       {slice_metrics['sensitivity']*100:6.2f}%")
    print(f"  - Slice Specificity:       {slice_metrics['specificity']*100:6.2f}%")
    print(f"  - Slice Macro-F1:          {slice_metrics['macro_f1']*100:6.2f}%")
    print(f"  - Slice ROC-AUC:           {slice_metrics['roc_auc']:6.4f}")
    print("-" * 60)
    print("Patient-Level Metrics (Top-k Pooling, k=3):")
    print(f"  - Patient Diagnosis Acc:   {patient_metrics['accuracy']*100:6.2f}%")
    print(f"  - Patient Sensitivity:     {patient_metrics['sensitivity']*100:6.2f}% (Cancer Catch Rate)")
    print(f"  - Patient Specificity:     {patient_metrics['specificity']*100:6.2f}% (Healthy Tissue Rate)")
    print(f"  - Patient Macro-F1:        {patient_metrics['macro_f1']*100:6.2f}%")
    print(f"  - Patient ROC-AUC:         {patient_metrics['roc_auc']:6.4f}")
    print("=" * 60)

    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    y_pred = np.argmax(all_probs, axis=1)
    plot_confusion_matrix(all_labels, y_pred, save_path=cm_path)
    print(f"[OK] Confusion matrix plot saved to: {cm_path}")

if __name__ == "__main__":
    main()
