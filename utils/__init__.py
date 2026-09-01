"""
Utilities for Deepfake Detection & Algorithmic Fairness Benchmark.
"""

from utils.dataset_loader import (
    DeepfakeFairnessDataset,
    MST_GROUPS,
    create_dataloaders,
    get_default_transforms,
)
from utils.loss import (
    EqualizedOddsRegularizedLoss,
    SubgroupReweightedLoss,
    build_criterion,
)
from utils.metrics import (
    FairnessAuditor,
    compute_binary_metrics,
    compute_equal_error_rate,
    evaluate_fairness_disparities,
)

__all__ = [
    "DeepfakeFairnessDataset",
    "MST_GROUPS",
    "create_dataloaders",
    "get_default_transforms",
    "FairnessAuditor",
    "compute_binary_metrics",
    "compute_equal_error_rate",
    "evaluate_fairness_disparities",
    "SubgroupReweightedLoss",
    "EqualizedOddsRegularizedLoss",
    "build_criterion",
]
