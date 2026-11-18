import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss (Equation 4 in review paper).
    Replaces pixel-space SMOTE to handle medical class imbalance
    without creating unnatural ghost artifacts.
    
    L_Focal = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: list = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: raw logits [BatchSize, NumClasses]
        targets: class index labels [BatchSize]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # probability of true class
        
        focal_weight = (1.0 - pt) ** self.gamma
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha[targets]
            focal_loss = at * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
            
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
