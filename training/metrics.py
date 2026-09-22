"""
Evaluation metrics for intrusion detection.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from typing import Dict, Tuple


class MetricsCalculator:
    """Computes classification metrics."""
    
    def __init__(self, num_classes: int, average: str = "macro"):
        self.num_classes = num_classes
        self.average = average
        
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            y_true: (N,) ground truth labels
            y_pred: (N,) predicted labels
            y_prob: (N, C) predicted probabilities (optional)
        Returns:
            metrics: dict of metric names to values
        """
        metrics = {}
        
        # Accuracy
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1
        metrics["precision"] = precision_score(
            y_true, y_pred, average=self.average, zero_division=0
        )
        metrics["recall"] = recall_score(
            y_true, y_pred, average=self.average, zero_division=0
        )
        metrics["f1"] = f1_score(
            y_true, y_pred, average=self.average, zero_division=0
        )
        
        # Per-class metrics
        per_class_precision = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        per_class_recall = recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        per_class_f1 = f1_score(
            y_true, y_pred, average=None, zero_division=0
        )
        
        for c in range(len(per_class_f1)):
            metrics[f"precision_class_{c}"] = per_class_precision[c]
            metrics[f"recall_class_{c}"] = per_class_recall[c]
            metrics[f"f1_class_{c}"] = per_class_f1[c]
        
        # AUC (binary only)
        if y_prob is not None and self.num_classes == 2:
            try:
                metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
            except ValueError:
                metrics["auc"] = 0.0
        
        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        
        return metrics
    
    def aggregate(
        self,
        all_metrics: list,
    ) -> Dict[str, float]:
        """
        Aggregate metrics across multiple runs.
        
        Args:
            all_metrics: list of metric dicts
        Returns:
            aggregated: dict with mean and std
        """
        aggregated = {}
        
        # Get all numeric keys (exclude confusion_matrix)
        keys = [k for k in all_metrics[0].keys() if k != "confusion_matrix"]
        
        for key in keys:
            values = [m[key] for m in all_metrics]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
        
        return aggregated


def compute_binary_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Compute binary AUC for benign vs malicious."""
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return 0.0
