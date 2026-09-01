"""
Fairness-Aware and Standard Loss Functions for Deepfake Detection.
Includes demographic subgroup reweighting and Equalized Odds regularization.
"""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubgroupReweightedLoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss with Demographic Subgroup Balancing.
    Weights each sample inversely proportional to its demographic frequency.
    """

    def __init__(self, group_weights: Optional[Dict[str, float]] = None, pos_weight: Optional[float] = None):
        """
        Args:
            group_weights: Dictionary mapping subgroup keys (e.g., 'MST_1', 'Female') to scalar loss multipliers.
            pos_weight: Optional positive class weighting for class imbalance.
        """
        super().__init__()
        self.group_weights = group_weights or {}
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        subgroups: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted raw logits of shape (B, 1) or (B,).
            targets: Binary ground truth labels of shape (B,) or (B, 1).
            subgroups: List of demographic subgroup string identifiers of length B.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight, reduction="none"
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        if subgroups is not None and self.group_weights:
            sample_weights = torch.tensor(
                [self.group_weights.get(sg, 1.0) for sg in subgroups],
                device=logits.device,
                dtype=torch.float32,
            )
            # Normalize sample weights in the batch to avoid scale shifts
            sample_weights = sample_weights / (sample_weights.mean() + 1e-8)
            loss = (bce_loss * sample_weights).mean()
        else:
            loss = bce_loss.mean()

        return loss


class EqualizedOddsRegularizedLoss(nn.Module):
    """
    Combines Binary Cross-Entropy with a differentiable surrogate Equalized Odds penalty.
    Penalizes variance in subgroup False Positive Rates and True Positive Rates.
    """

    def __init__(self, lambda_fair: float = 0.5, pos_weight: Optional[float] = None):
        """
        Args:
            lambda_fair: Weight multiplier for the demographic fairness regularization term.
            pos_weight: Optional positive class weighting.
        """
        super().__init__()
        self.lambda_fair = lambda_fair
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        skin_tones: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, 1) or (B,) predicted logits.
            targets: (B,) or (B, 1) binary labels (0 or 1).
            skin_tones: (B,) tensor containing Monk Skin Tone integer scale (1-10).
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Primary classification loss
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets)

        if skin_tones is None or self.lambda_fair <= 0.0:
            return bce_loss

        probs = torch.sigmoid(logits)
        unique_groups = torch.unique(skin_tones)

        if len(unique_groups) < 2:
            return bce_loss

        group_fprs = []
        group_tprs = []

        eps = 1e-7
        for grp in unique_groups:
            grp_mask = skin_tones == grp
            neg_mask = grp_mask & (targets == 0)
            pos_mask = grp_mask & (targets == 1)

            # Soft FPR surrogate: mean predicted probability on real images
            if neg_mask.sum() > 0:
                soft_fpr = probs[neg_mask].mean()
                group_fprs.append(soft_fpr)

            # Soft TPR surrogate: mean predicted probability on fake images
            if pos_mask.sum() > 0:
                soft_tpr = probs[pos_mask].mean()
                group_tprs.append(soft_tpr)

        fairness_penalty = torch.tensor(0.0, device=logits.device)

        if len(group_fprs) >= 2:
            fpr_tensor = torch.stack(group_fprs)
            fairness_penalty = fairness_penalty + torch.var(fpr_tensor)

        if len(group_tprs) >= 2:
            tpr_tensor = torch.stack(group_tprs)
            fairness_penalty = fairness_penalty + torch.var(tpr_tensor)

        total_loss = bce_loss + (self.lambda_fair * fairness_penalty)
        return total_loss


def build_criterion(
    loss_type: str = "bce",
    lambda_fair: float = 0.5,
    pos_weight: Optional[float] = None,
    group_weights: Optional[Dict[str, float]] = None,
) -> nn.Module:
    """
    Factory function to construct loss criteria.
    """
    loss_type = loss_type.lower()
    if loss_type == "bce":
        pos_w = torch.tensor([pos_weight]) if pos_weight is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pos_w)
    elif loss_type in ["subgroup_reweighted", "reweighted", "fair_reweight"]:
        return SubgroupReweightedLoss(group_weights=group_weights, pos_weight=pos_weight)
    elif loss_type in ["equalized_odds", "eo_penalty", "fair_regularized"]:
        return EqualizedOddsRegularizedLoss(lambda_fair=lambda_fair, pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose from 'bce', 'subgroup_reweighted', 'equalized_odds'.")
