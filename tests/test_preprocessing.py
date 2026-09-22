"""Unit tests for preprocessing pipeline."""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import PreprocessingPipeline
from data.feature_extraction import TrafficFeatureExtractor


class TestPreprocessingPipeline:
    """Tests for PreprocessingPipeline."""
    
    def test_fit_transform(self):
        """Test fit and transform."""
        X_train = np.random.randn(1000, 11)
        pipeline = PreprocessingPipeline(sequence_length=30, stride=30)
        
        X_norm = pipeline.fit_transform(X_train)
        
        # Check normalization
        assert np.allclose(X_norm.mean(axis=0), 0, atol=1e-5)
        assert np.allclose(X_norm.std(axis=0), 1, atol=1e-1)
    
    def test_imputation(self):
        """Test missing value imputation."""
        X_train = np.random.randn(1000, 11)
        X_train[0, 0] = np.nan
        X_train[1, 1] = np.inf
        
        pipeline = PreprocessingPipeline()
        pipeline.fit(X_train)
        
        assert not np.isnan(pipeline.feature_means).any()
    
    def test_sequence_construction(self):
        """Test sequence construction with padding."""
        X = np.random.randn(100, 11)
        y = np.random.randint(0, 2, 100)
        
        pipeline = PreprocessingPipeline(sequence_length=30, stride=30)
        X_norm = pipeline.fit_transform(X)
        
        sequences, labels, masks = pipeline.construct_sequences(X_norm, y)
        
        assert sequences.shape[1:] == (30, 11)
        assert len(sequences) == len(labels) == len(masks)
        assert masks.shape[1] == 30


class TestFeatureExtractor:
    """Tests for feature extraction."""
    
    def test_feature_count(self):
        """Test correct number of features."""
        extractor = TrafficFeatureExtractor()
        assert extractor.num_features == 11
    
    def test_feature_names(self):
        """Test feature names."""
        extractor = TrafficFeatureExtractor()
        assert len(extractor.FEATURE_NAMES) == 11


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
