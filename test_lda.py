import pytest
import numpy as np
import sys
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib


matplotlib.use("Agg")

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Add it to sys.path
sys.path.insert(0, project_root)

# Import the LDA module to test
from lda import LinearDiscriminant as LDA


@pytest.fixture
def sample_data():
    """Provide sample data for testing LDA"""
    # Create a simple synthetic dataset
    X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
    y = np.array([0, 0, 0, 1, 1, 1])

    return X, y


@pytest.fixture
def iris_binary_data():
    """Fixture for binary classification with iris dataset"""
    iris = datasets.load_iris()
    X = iris.data[:100]  # Take only first two classes
    y = iris.target[:100]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def test_lda_initialization():
    """Test LDA model initialization"""
    lda = LDA()
    assert hasattr(lda, "fit"), "LDA class should have a fit method"
    assert hasattr(lda, "predict"), "LDA class should have a predict method"


def test_lda_fit(sample_data):
    """Test LDA fit method with simple data"""
    X, y = sample_data
    lda = LDA()
    lda.fit(X, y)

    # Check for class means (could be named means_, mean_vectors, class_means, etc.)
    means_attr = next(
        (
            attr
            for attr in dir(lda)
            if attr in ["means_", "mean_vectors", "class_means"]
        ),
        None,
    )
    assert means_attr is not None, "LDA should compute class means"
    means = getattr(lda, means_attr)

    # Check for class priors (could be named priors_, priors, prior_probabilities, etc.)
    priors_attr = next(
        (
            attr
            for attr in dir(lda)
            if attr in ["priors_", "priors", "prior_probabilities"]
        ),
        None,
    )
    assert priors_attr is not None, "LDA should compute class priors"
    priors = getattr(lda, priors_attr)

    # Check for coefficients (could be named coef_, weights_, weights, etc.)
    coef_attr = next(
        (
            attr
            for attr in dir(lda)
            if attr in ["coef_", "weights_", "weights", "coefficients"]
        ),
        None,
    )
    assert coef_attr is not None, "LDA should compute coefficients"

    # Check shapes of the attributes
    unique_classes = np.unique(y)
    assert len(priors) == len(
        unique_classes
    ), "Number of priors should match number of classes"

    # Check shape of means - more flexible check
    if means_attr:
        # Convert to numpy array if it's not already
        if not isinstance(means, np.ndarray):
            try:
                means = np.array(means)
            except:
                pass  # If conversion fails, we'll check differently

        # Flexible shape checking - just make sure there's a dimension with n_classes entries
        n_classes = len(unique_classes)
        n_features = X.shape[1]

        if isinstance(means, np.ndarray):
            # Check if any dimension has length n_classes
            has_class_dim = any(dim == n_classes for dim in means.shape)
            assert (
                has_class_dim
            ), f"Means should have a dimension with {n_classes} entries (one per class)"

            # For 2D arrays, check that the other dimension matches n_features
            if means.ndim == 2:
                has_feature_dim = any(dim == n_features for dim in means.shape)
                assert (
                    has_feature_dim
                ), f"For 2D means, one dimension should have {n_features} entries (one per feature)"
        else:
            # If it's not a numpy array, just check that there are n_classes items
            assert (
                len(means) == n_classes
            ), f"Means should have {n_classes} entries (one per class)"


def test_lda_predict(sample_data):
    """Test LDA predict method with simple data"""
    X, y = sample_data
    lda = LDA()
    lda.fit(X, y)

    # Make predictions
    y_pred = lda.predict(X)

    # Check predictions
    assert (
        y_pred.shape == y.shape
    ), "Predictions should have same shape as original labels"
    assert np.all(
        np.isin(y_pred, np.unique(y))
    ), "Predictions should be valid class labels"

    # Should have reasonable accuracy on training data
    accuracy = np.mean(y_pred == y)
    assert accuracy >= 0.8, f"Expected accuracy >= 80% but got {accuracy*100:.2f}%"


def test_lda_with_iris(iris_binary_data):
    """Test LDA on the iris dataset (binary classification)"""
    X_train, X_test, y_train, y_test = iris_binary_data

    lda = LDA()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    # Check predictions
    assert (
        y_pred.shape == y_test.shape
    ), "Predictions should have same shape as test labels"
    assert np.all(
        np.isin(y_pred, np.unique(y_train))
    ), "Predictions should be valid class labels"

    # Should have good accuracy on iris
    accuracy = np.mean(y_pred == y_test)
    assert accuracy >= 0.75, f"Expected accuracy >= 75% but got {accuracy*100:.2f}%"


def test_lda_predict_proba(iris_binary_data):
    """Test predict_proba method if implemented"""
    X_train, X_test, y_train, _ = iris_binary_data

    lda = LDA()
    lda.fit(X_train, y_train)

    # Only run this test if predict_proba is implemented
    if hasattr(lda, "predict_proba"):
        probs = lda.predict_proba(X_test)

        # Check basic probability properties
        assert (
            probs.shape[0] == X_test.shape[0]
        ), "Should have one probability row per sample"
        assert probs.shape[1] == len(
            np.unique(y_train)
        ), "Should have one probability column per class"
        assert np.allclose(
            np.sum(probs, axis=1), 1
        ), "Probabilities should sum to 1 for each sample"
        assert np.all(probs >= 0) and np.all(
            probs <= 1
        ), "Probabilities should be between 0 and 1"
