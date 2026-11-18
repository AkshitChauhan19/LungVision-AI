import os
import sys
import argparse
import pandas as pd

# Add root project dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import scan_dataset, create_patient_folds
from src.train import run_fold_training

def main():
    parser = argparse.ArgumentParser(description="Train Lung Cancer CT CAD Framework")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw IQ-OTH/NCCD dataset folder")
    parser.add_argument("--model", type=str, default="efficientnet_cbam", choices=["efficientnet_cbam", "resnet50"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=1, help="Number of folds to train (1 for quick run, 5 for full cross-val)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic CT slices for instant verification")
    args = parser.parse_args()

    if args.synthetic:
        print("\n[*] Generating synthetic CT slices for instant pipeline demonstration...")
        from PIL import Image
        import numpy as np
        synth_dir = "data/sample"
        for cname in ["Normal cases", "Bengin cases", "Malignant cases"]:
            os.makedirs(os.path.join(synth_dir, cname), exist_ok=True)
            for p in range(1, 6):
                for s in range(1, 5):
                    fname = f"{cname.split()[0]} case ({p})_slice{s}.jpg"
                    fpath = os.path.join(synth_dir, cname, fname)
                    if not os.path.exists(fpath):
                        # Create synthetic lung CT image with soft circle
                        arr = np.full((240, 240, 3), 40, dtype=np.uint8)
                        # Add simulated parenchymal lung ellipse
                        rr, cc = np.ogrid[:240, :240]
                        mask = ((rr - 120)**2 / (90**2) + (cc - 120)**2 / (70**2)) <= 1
                        arr[mask] = 120
                        Image.fromarray(arr).save(fpath)
        args.data_dir = synth_dir

    print("=" * 60)
    print(" LUNG CANCER CT DETECTION - LEAKAGE-FREE TRAINING")
    print("=" * 60)
    print(f"Dataset directory: {args.data_dir}")
    print(f"Model architecture: {args.model}")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}")

    if not os.path.exists(args.data_dir) or len(os.listdir(args.data_dir)) == 0:
        print(f"\n[!] Data directory '{args.data_dir}' is empty or not found.")
        print("    Please place IQ-OTH/NCCD dataset folders inside 'data/raw/':")
        print("      data/raw/Normal cases/")
        print("      data/raw/Bengin cases/")
        print("      data/raw/Malignant cases/")
        print("    Or run with '--synthetic' for an instant demo:")
        print("      python scripts/run_training.py --synthetic --epochs 2")
        print("    (Run 'python scripts/download_dataset.py' for dataset instructions).")
        return

    df = scan_dataset(args.data_dir)
    print(f"Total CT slices scanned: {len(df)}")
    print(f"Class distribution:\n{df['class_name'].value_counts()}")
    print(f"Total unique patients: {df['patient_id'].nunique()}")

    folds = create_patient_folds(df, n_splits=5)
    print(f"Successfully generated 5 leakage-free patient-stratified folds.")

    for fold_idx in range(args.folds):
        print(f"\n--- Running Fold {fold_idx + 1}/5 ---")
        train_df, val_df = folds[fold_idx]
        print(f"Train slices: {len(train_df)} ({train_df['patient_id'].nunique()} patients) | "
              f"Val slices: {len(val_df)} ({val_df['patient_id'].nunique()} patients)")
        
        results = run_fold_training(
            train_df=train_df,
            val_df=val_df,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_dir=f"checkpoints/fold_{fold_idx + 1}"
        )
        print(f"\nFold {fold_idx + 1} Best Results (Epoch {results['epoch']}):")
        print(f"  Slice Accuracy:   {results['slice_metrics']['accuracy']*100:.2f}%")
        print(f"  Patient Accuracy: {results['patient_metrics']['accuracy']*100:.2f}%")
        print(f"  Patient Macro-F1: {results['patient_metrics']['macro_f1']*100:.2f}%")

if __name__ == "__main__":
    main()
