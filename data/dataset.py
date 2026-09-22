"""
PyTorch Dataset and DataLoader for network intrusion detection.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class IntrusionDetectionDataset(Dataset):
    """Dataset for network intrusion detection."""
    
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        masks: Optional[np.ndarray] = None,
    ):
        """
        Args:
            sequences: (N, T, D) input sequences
            labels: (N,) class labels
            masks: (N, T) padding masks
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.masks = torch.FloatTensor(masks) if masks is not None else None
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> dict:
        item = {
            "input": self.sequences[idx],
            "label": self.labels[idx],
        }
        if self.masks is not None:
            item["mask"] = self.masks[idx]
        return item


def create_dataloaders(
    train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    
    train_dataset = IntrusionDetectionDataset(*train_data)
    val_dataset = IntrusionDetectionDataset(*val_data)
    test_dataset = IntrusionDetectionDataset(*test_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequencies.
    w_c = N / (C * N_c)
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    N = len(labels)
    C = num_classes
    
    weights = N / (C * np.maximum(class_counts, 1))
    return torch.FloatTensor(weights)
