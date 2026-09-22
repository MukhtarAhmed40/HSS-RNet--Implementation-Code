"""
Data preprocessing pipeline for HSS-RNet.
Implements leakage-resistant preprocessing with train-only statistics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline:
        1. Invalid value handling
        2. Feature-wise mean imputation (train-only)
        3. Z-score normalization (train-only)
        4. Temporal ordering
        5. Sequence construction with padding/segmentation
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        stride: int = 30,
        pad_value: float = 0.0,
    ):
        self.sequence_length = sequence_length
        self.stride = stride
        self.pad_value = pad_value
        
        # Statistics computed from training data only
        self.feature_means = None
        self.feature_stds = None
        self.is_fitted = False
        
    def fit(self, X_train: np.ndarray) -> "PreprocessingPipeline":
        """
        Fit preprocessing statistics on training data only.
        
        Args:
            X_train: (num_samples, num_features) raw features
        """
        # Handle invalid values
        X_clean = self._handle_invalid_values(X_train)
        
        # Compute imputation statistics
        self.feature_means = np.nanmean(X_clean, axis=0)
        
        # Impute missing values
        X_imputed = self._impute_missing(X_clean)
        
        # Compute normalization statistics
        self.feature_stds = np.std(X_imputed, axis=0)
        self.feature_stds[self.feature_stds == 0] = 1.0  # Avoid division by zero
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to data.
        
        Args:
            X: (num_samples, num_features) raw features
        Returns:
            X_normalized: (num_samples, num_features) normalized features
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        # Handle invalid values
        X_clean = self._handle_invalid_values(X)
        
        # Impute using training statistics
        X_imputed = self._impute_missing(X_clean)
        
        # Z-score normalization using training statistics
        X_normalized = (X_imputed - self.feature_means) / self.feature_stds
        
        return X_normalized
    
    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit and transform training data."""
        self.fit(X_train)
        return self.transform(X_train)
    
    def _handle_invalid_values(self, X: np.ndarray) -> np.ndarray:
        """Replace inf and -inf with NaN."""
        X_clean = X.copy().astype(np.float64)
        X_clean[np.isinf(X_clean)] = np.nan
        return X_clean
    
    def _impute_missing(self, X: np.ndarray) -> np.ndarray:
        """Impute missing values using feature-wise means."""
        X_imputed = X.copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X_imputed[:, j])
            if mask.any():
                X_imputed[mask, j] = self.feature_means[j]
        return X_imputed
    
    def construct_sequences(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Construct fixed-length sequences from temporal data.
        
        Args:
            X: (num_samples, num_features) normalized features in temporal order
            y: (num_samples,) labels (optional)
        Returns:
            sequences: (num_sequences, seq_len, num_features)
            labels: (num_sequences,) or None
            masks: (num_sequences, seq_len) padding masks
        """
        num_samples, num_features = X.shape
        num_sequences = max(1, (num_samples - self.sequence_length) // self.stride + 1)
        
        sequences = []
        labels = [] if y is not None else None
        masks = []
        
        for i in range(num_sequences):
            start = i * self.stride
            end = start + self.sequence_length
            
            if end <= num_samples:
                # Full sequence
                seq = X[start:end]
                mask = np.ones(self.sequence_length, dtype=np.float32)
            else:
                # Pad sequence
                valid_len = num_samples - start
                seq = np.full((self.sequence_length, num_features), self.pad_value)
                seq[:valid_len] = X[start:]
                mask = np.zeros(self.sequence_length, dtype=np.float32)
                mask[:valid_len] = 1.0
            
            sequences.append(seq)
            masks.append(mask)
            
            if y is not None:
                # Use label of last timestep or majority vote
                if end <= num_samples:
                    label = y[end - 1]
                else:
                    label = y[-1]
                labels.append(label)
        
        sequences = np.array(sequences, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)
        labels = np.array(labels) if labels is not None else None
        
        return sequences, labels, masks
