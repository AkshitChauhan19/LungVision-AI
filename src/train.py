import os
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from .dataset import CTDataset, get_transforms, create_patient_folds, scan_dataset
from .models import build_efficientnet_cbam, build_resnet50
from .loss import ClassBalancedFocalLoss
from .evaluate import compute_slice_metrics, aggregate_patient_topk

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _ in loader:
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
        
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
                   device: torch.device) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs, all_patients = [], [], [], []
    total = 0
    
    for images, labels, pids in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        all_probs.append(probs)
        all_labels.extend(labels.cpu().numpy())
        all_patients.extend(pids)
        total += labels.size(0)
        
    all_probs = np.vstack(all_probs)
    all_labels = np.array(all_labels)
    
    metrics = compute_slice_metrics(all_labels, all_probs)
    
    df_results = pd.DataFrame({
        "patient_id": all_patients,
        "label": all_labels,
        "prob": list(all_probs)
    })
    return running_loss / total, metrics, df_results


def run_fold_training(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      model_name: str = "efficientnet_cbam", epochs: int = 25, 
                      batch_size: int = 16, lr: float = 1e-4, 
                      device: str = "cuda" if torch.cuda.is_available() else "cpu",
                      save_dir: str = "checkpoints") -> Dict:
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)
    print(f"Training on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    
    # Dataset & Loaders
    img_size = 240 if "efficientnet" in model_name else 224
    train_dataset = CTDataset(train_df, transform=get_transforms(img_size=img_size, is_train=True))
    val_dataset = CTDataset(val_df, transform=get_transforms(img_size=img_size, is_train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    if model_name == "efficientnet_cbam":
        model = build_efficientnet_cbam(num_classes=3, pretrained=True)
    else:
        model = build_resnet50(num_classes=3, pretrained=True)
    model.to(device)
    
    # Compute class weights for Focal Loss based on training frequencies
    counts = train_df["label"].value_counts().sort_index().values
    total_count = sum(counts)
    weights = [total_count / (len(counts) * c) for c in counts]
    criterion = ClassBalancedFocalLoss(alpha=weights, gamma=2.0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    best_metrics = {}
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, df_res = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        # Calculate Top-k patient level metrics
        patient_metrics = aggregate_patient_topk(df_res, k=3)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Loss: Train {train_loss:.4f} | Val {val_loss:.4f} - "
              f"Slice Acc: {val_metrics['accuracy']*100:.1f}% | "
              f"Patient Acc: {patient_metrics['accuracy']*100:.1f}% | "
              f"Patient Sens: {patient_metrics['sensitivity']*100:.1f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "epoch": epoch,
                "slice_metrics": val_metrics,
                "patient_metrics": patient_metrics
            }
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_{model_name}.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
                
    return best_metrics
