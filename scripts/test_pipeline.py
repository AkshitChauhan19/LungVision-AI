import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import build_efficientnet_cbam, build_resnet50, CBAM
from src.loss import ClassBalancedFocalLoss
from src.dataset import extract_patient_id, create_patient_folds
from src.evaluate import compute_slice_metrics, aggregate_patient_topk
from src.explain import GradCAM

def test_all():
    print('=' * 60)
    print(' RUNNING AUTOMATED UNIT & PIPELINE TESTS')
    print('=' * 60)
    
    # 1. Hardware / CUDA test
    cuda_avail = torch.cuda.is_available()
    print(f'[1] CUDA Available: {cuda_avail}')
    if cuda_avail:
        print(f'    GPU Device: {torch.cuda.get_device_name(0)}')
        print(f'    VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB')
    
    # 2. CBAM Module Test
    cbam = CBAM(in_channels=1280, reduction_ratio=16, kernel_size=7)
    x = torch.randn(2, 1280, 7, 7)
    out = cbam(x)
    assert out.shape == x.shape, f'CBAM shape mismatch: {out.shape}'
    print('[2] CBAM Module: PASS (Shape preserved [2, 1280, 7, 7])')
    
    # 3. EfficientNet-B1 + CBAM Forward Pass Test
    eff_model = build_efficientnet_cbam(num_classes=3, pretrained=False)
    x_img = torch.randn(2, 3, 240, 240)
    eff_out = eff_model(x_img)
    assert eff_out.shape == (2, 3), f'EfficientNet output shape mismatch: {eff_out.shape}'
    print('[3] EfficientNet-B1 + CBAM: PASS (Output shape [2, 3])')
    
    # 4. ResNet50 Baseline Test
    resnet_model = build_resnet50(num_classes=3, pretrained=False)
    x_img_res = torch.randn(2, 3, 224, 224)
    res_out = resnet_model(x_img_res)
    assert res_out.shape == (2, 3), f'ResNet50 output shape mismatch: {res_out.shape}'
    print('[4] ResNet50 Baseline: PASS (Output shape [2, 3])')
    
    # 5. Class-Balanced Focal Loss Test
    criterion = ClassBalancedFocalLoss(alpha=[1.0, 3.0, 1.5], gamma=2.0)
    target = torch.tensor([0, 2])
    loss = criterion(eff_out, target)
    loss.backward()
    assert not torch.isnan(loss), 'Loss is NaN'
    print(f'[5] Class-Balanced Focal Loss: PASS (Loss = {loss.item():.4f}, Gradients computed)')
    
    # 6. Patient ID Extraction Test
    p1 = extract_patient_id('Normal case (12).jpg')
    p2 = extract_patient_id('Bengin case (3).jpg')
    p3 = extract_patient_id('Malignant case (24).jpg')
    assert p1 == 'nor_0012', f'Patient ID mismatch: {p1}'
    assert p2 == 'ben_0003', f'Patient ID mismatch: {p2}'
    assert p3 == 'mal_0024', f'Patient ID mismatch: {p3}'
    print(f'[6] Patient ID Parser: PASS ({p1}, {p2}, {p3})')
    
    # 7. Leakage-Free StratifiedGroupKFold Test
    records = []
    # Simulate 110 patients with 10 slices each = 1,100 slices
    for p_idx in range(110):
        cls = 0 if p_idx < 55 else (1 if p_idx < 70 else 2)
        pid = f'patient_{p_idx:03d}'
        for s in range(10):
            records.append({'patient_id': pid, 'label': cls, 'slice_idx': s})
    df_sim = pd.DataFrame(records)
    
    folds = create_patient_folds(df_sim, n_splits=5)
    print('[7] Leakage-Free StratifiedGroupKFold: PASS')
    for f_idx, (train_f, val_f) in enumerate(folds):
        train_pts = set(train_f['patient_id'])
        val_pts = set(val_f['patient_id'])
        overlap = train_pts.intersection(val_pts)
        assert len(overlap) == 0, f'Leakage in fold {f_idx}: {overlap}'
        print(f'    Fold {f_idx + 1}: Train {len(train_pts)} pts ({len(train_f)} slices) | Val {len(val_pts)} pts ({len(val_f)} slices) | Leakage: 0%')
        
    # 8. Top-k Pooling Test
    df_res_sim = pd.DataFrame({
        'patient_id': ['pat_1', 'pat_1', 'pat_1', 'pat_2', 'pat_2'],
        'label': [2, 2, 2, 0, 0],
        'prob': [
            np.array([0.1, 0.1, 0.8]),
            np.array([0.2, 0.1, 0.7]),
            np.array([0.1, 0.2, 0.7]),
            np.array([0.9, 0.05, 0.05]),
            np.array([0.85, 0.1, 0.05])
        ]
    })
    topk_res = aggregate_patient_topk(df_res_sim, k=3)
    assert topk_res['accuracy'] == 1.0, 'Top-k accuracy mismatch'
    print(f'[8] Top-k Volumetric Pooling: PASS (Aggregated patient accuracy: {topk_res["accuracy"]*100:.1f}%)')
    
    # 9. Grad-CAM Hook Test
    target_layer = eff_model.features[-1]
    grad_cam = GradCAM(eff_model, target_layer)
    cam = grad_cam.generate_heatmap(x_img[0:1], class_idx=2)
    assert cam.ndim == 2 and cam.shape[0] > 0, f'CAM shape mismatch: {cam.shape}'
    print(f'[9] Grad-CAM Activation Generator: PASS (CAM feature map: {cam.shape})')
    
    print('=' * 60)
    print(' ALL 9 PIPELINE TESTS PASSED WITH ZERO ERRORS!')
    print('=' * 60)

if __name__ == '__main__':
    test_all()
