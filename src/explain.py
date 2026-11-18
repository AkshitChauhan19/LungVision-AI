import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image

class GradCAM:
    """
    Grad-CAM Visual Explainability Generator.
    Calculates gradient-weighted activation maps from the final
    rectified convolutional layer of the backbone.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        # Global Average Pooling on gradients (Eq. Grad-CAM weights)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        
        # Apply ReLU to retain only positive contributions
        cam = torch.clamp(cam, min=0)
        cam = cam.cpu().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam


def overlay_heatmap_on_image(original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlays the Grad-CAM heatmap onto the original CT slice."""
    w, h = original_img.size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply JET colormap (red = high attention, blue = low)
    color_cam = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_cam = cv2.cvtColor(color_cam, cv2.COLOR_BGR2RGB)
    
    img_np = np.array(original_img)
    blended = np.uint8(alpha * color_cam + (1 - alpha) * img_np)
    return Image.fromarray(blended)


def evaluate_pointing_game(cam: np.ndarray, bbox: tuple) -> bool:
    """
    Pointing Game Protocol (Equation 9 in review paper).
    Returns True (Hit) if peak activation falls inside bounding box (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = bbox
    peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
    return (x1 <= peak_x <= x2) and (y1 <= peak_y <= y2)
