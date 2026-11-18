import os

def main():
    print("=" * 60)
    print(" IQ-OTH/NCCD LUNG CANCER DATASET INSTRUCTIONS")
    print("=" * 60)
    print(r"""
The IQ-OTH/NCCD dataset is publicly available on Kaggle and Mendeley Data.
Dataset Name: 'The IQ-OTH/NCCD lung cancer dataset'

Instructions:
1. Download from Kaggle:
   https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset

2. Extract the downloaded archive. You will see three folders:
   - 'Bengin cases'
   - 'Malignant cases'
   - 'Normal cases'

3. Copy or move those 3 folders directly into:
   lung-cancer-ct-detection/data/raw/

Expected structure:
   data/raw/
   |-- Bengin cases/
   |-- Malignant cases/
   \-- Normal cases/

Once copied, verify by running:
   python scripts/run_training.py
""")

if __name__ == "__main__":
    main()
