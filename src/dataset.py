import os
import re
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold

CLASS_NAMES = ["normal", "benign", "malignant"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def extract_patient_id(filename: str) -> str:
    """
    Extracts Patient ID from standard CT filename conventions.
    Examples:
      'Normal case (12).jpg' -> 'norm_012'
      'Bengin case (3).jpg'  -> 'ben_003'
      'Malignant case (24).jpg' -> 'mal_024'
    """
    match = re.search(r'\((\d+)\)', filename)
    prefix = filename[:3].lower()
    if match:
        return f"{prefix}_{int(match.group(1)):04d}"
    
    # Fallback: extract any digits in filename
    num_match = re.search(r'(\d+)', filename)
    if num_match:
        return f"{prefix}_{int(num_match.group(1)):04d}"
    return filename.split(".")[0]


def scan_dataset(data_dir: str) -> pd.DataFrame:
    """
    Scans data directory structured into class folders:
      data_dir/
        ├── Normal/
        ├── Benign/
        └── Malignant/
    Returns a pandas DataFrame with columns: [filepath, label, class_name, patient_id]
    """
    records = []
    for class_name, label in CLASS_TO_IDX.items():
        # Match case-insensitively
        found_dir = None
        for d in os.listdir(data_dir):
            if d.lower().startswith(class_name[:3]):
                found_dir = os.path.join(data_dir, d)
                break
                
        if found_dir and os.path.isdir(found_dir):
            for fname in os.listdir(found_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    fpath = os.path.join(found_dir, fname)
                    pid = extract_patient_id(fname)
                    records.append({
                        "filepath": fpath,
                        "filename": fname,
                        "label": label,
                        "class_name": class_name,
                        "patient_id": pid
                    })
    df = pd.DataFrame(records)
    return df


def get_transforms(img_size: int = 240, is_train: bool = True):
    """
    Medical image transformations.
    Training uses affine rotations and flips strictly on training folds.
    Validation uses clean resizing and normalization.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


class CTDataset(Dataset):
    """PyTorch Dataset loading CT slices."""
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = int(row["label"])
        patient_id = row["patient_id"]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, patient_id


def create_patient_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Creates leakage-free Stratified Group K-Fold cross-validation splits.
    Guarantees that 100% of slices for any single patient are strictly
    confined to ONE fold (0% data leakage).
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    X = df.index.values
    y = df["label"].values
    groups = df["patient_id"].values
    
    folds = []
    for train_idx, val_idx in sgkf.split(X, y, groups=groups):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        # Sanity check: Assert zero overlap
        train_p = set(train_df["patient_id"])
        val_p = set(val_df["patient_id"])
        assert len(train_p.intersection(val_p)) == 0, "DATA LEAKAGE DETECTED! Overlapping patient in train and val."
        
        folds.append((train_df, val_df))
    return folds
