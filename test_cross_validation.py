import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
import pytest
from cross_validation import cross_validation


def test_cross_validation():
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=100, n_features=20, n_classes=2, random_state=42
    )

    # Initialize model
    model = KNeighborsClassifier(n_neighbors=5)

    # Test with k-fold CV
    mean_score, std_score = cross_validation(model, X, y, nFolds=5)
    print(mean_score, std_score)
    assert isinstance(mean_score, float)
    assert isinstance(std_score, float)
    assert 0 <= mean_score <= 1
    assert std_score >= 0
    assert np.isclose(0.95, mean_score)
    assert np.isclose(0.044, std_score, atol=1e-3)
    # Test LOO CV
    mean_score_loo, std_score_loo = cross_validation(model, X, y, nFolds=-1)
    assert isinstance(mean_score_loo, float)
    assert isinstance(std_score_loo, float)
    assert 0 <= mean_score_loo <= 1
    assert std_score_loo >= 0
    assert np.isclose(0.93, mean_score_loo)
    assert np.isclose(0.255, std_score_loo, atol=1e-2)
