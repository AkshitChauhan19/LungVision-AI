from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = ["Normal", "Benign", "Malignant"]

def compute_slice_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    """Computes all standard IEEE benchmark metrics for 2D slices."""
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    # Sensitivity (Recall) per class & Macro
    sensitivity_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    macro_sensitivity = np.mean(sensitivity_per_class)
    
    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    specificities = []
    for i in range(3):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    macro_specificity = np.mean(specificities)
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    
    # ROC-AUC (one-vs-rest)
    try:
        auc = roc_auc_score(y_true, y_pred_prob, multi_class="ovr")
    except Exception:
        auc = 0.0
        
    return {
        "accuracy": acc,
        "sensitivity": macro_sensitivity,
        "specificity": macro_specificity,
        "precision": precision,
        "macro_f1": macro_f1,
        "roc_auc": auc,
        "malignant_sensitivity": sensitivity_per_class[2] if len(sensitivity_per_class) > 2 else 0.0
    }


def aggregate_patient_topk(df_results: pd.DataFrame, k: int = 3) -> Dict[str, float]:
    """
    Top-k Volumetric Slice Pooling (Equation 6 in paper):
    Y_patient = (1/k) * sum_{j=1}^k Top_j({p_s})
    
    Aggregates slice-level probabilities into true patient diagnoses.
    """
    patient_records = []
    for patient_id, group in df_results.groupby("patient_id"):
        true_label = group["label"].iloc[0]
        
        # Probabilities shape: [num_slices, 3]
        probs = np.vstack(group["prob"].values)
        
        # If fewer than k slices, use all
        actual_k = min(k, len(probs))
        
        # Top-k pooling for each class
        topk_probs = np.zeros(3)
        for c in range(3):
            sorted_c = np.sort(probs[:, c])[::-1]
            topk_probs[c] = np.mean(sorted_c[:actual_k])
            
        patient_records.append({
            "patient_id": patient_id,
            "true_label": true_label,
            "pred_prob": topk_probs,
            "pred_label": np.argmax(topk_probs)
        })
        
    df_patient = pd.DataFrame(patient_records)
    y_true = df_patient["true_label"].values
    y_prob = np.vstack(df_patient["pred_prob"].values)
    
    return compute_slice_metrics(y_true, y_prob)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """Plots and saves an IEEE-style confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix (Normal / Benign / Malignant)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
