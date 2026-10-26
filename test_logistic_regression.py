import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import your LogisticRegressor class
try:
    from logistic_regression import LogisticRegressor
except ImportError:
    from logistic_regression import LogisticRegressor


@pytest.fixture
def sample_data():
    """Create a simple dataset for testing"""
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_fit_basic(sample_data):
    """Test basic fitting without regularization"""
    model = LogisticRegressor()
    X_train, _, y_train, _ = sample_data
    model.fit(X_train, y_train, num_iterations=100)

    assert model.weights is not None
    assert model.weights.shape == (X_train.shape[1],)
    assert isinstance(model.bias, (float, np.float64))


def test_predict_proba(sample_data):
    """Test probability predictions"""
    model = LogisticRegressor()
    X_train, X_test, y_train, _ = sample_data
    model.fit(X_train, y_train, num_iterations=100)

    probas = model.predict_proba(X_test)
    assert probas.shape == (X_test.shape[0],)
    assert np.all((probas >= 0) & (probas <= 1))


def test_predict(sample_data):
    """Test class predictions"""
    model = LogisticRegressor()
    X_train, X_test, y_train, _ = sample_data
    model.fit(X_train, y_train, num_iterations=100)

    predictions = model.predict(X_test)
    assert predictions.shape == (X_test.shape[0],)
    assert np.all(np.unique(predictions) == np.array([0, 1]))


def test_lasso_regularization():
    """Test L1 (Lasso) regularization implementation"""
    # Initialize test values
    model = LogisticRegressor()
    model.weights = np.array([1.0, -2.0, 0.0, 3.0])
    dw = np.array([0.1, 0.2, -0.1, 0.3])
    m = 100
    C = 1.0

    # Calculate expected result
    # Get actual result
    actual_result = model.lasso_regularization(dw, m, C)
    expected_result = [0.11, 0.19, -0.1, 0.31]
    # Verify results
    np.testing.assert_array_almost_equal(actual_result, expected_result)

    # Test with different C values
    C_strong = 1  # stronger regularization
    C_weak = 0.1  # weaker regularization

    result_strong = model.lasso_regularization(dw, m, C_strong)
    result_weak = model.lasso_regularization(dw, m, C_weak)

    # Verify that stronger regularization has less impact on gradient
    assert np.all(abs(result_strong - dw) >= abs(result_weak - dw))


def test_ridge_regularization():
    """Test L2 (Ridge) regularization implementation"""
    # Initialize test values
    model = LogisticRegressor()
    model.weights = np.array([1.0, -2.0, 0.0, 3.0])
    dw = np.array([0.1, 0.2, -0.1, 0.3])
    m = 100
    C = 1.0

    # Calculate expected result

    # Get actual result
    actual_result = model.ridge_regularization(dw, m, C)
    expected_result = [0.11, 0.18, -0.1, 0.33]
    # Verify results
    np.testing.assert_array_almost_equal(actual_result, expected_result)

    # Test with different C values
    C_strong = 10.0  # stronger regularization
    C_weak = 0.1  # weaker regularization

    result_strong = model.ridge_regularization(dw, m, C_strong)
    result_weak = model.ridge_regularization(dw, m, C_weak)

    # Verify that stronger regularization has less impact on gradient
    assert np.all(abs(result_strong - dw) >= abs(result_weak - dw))

    # Verify that zero weights result in zero regularization
    model.weights = np.zeros_like(model.weights)
    zero_result = model.ridge_regularization(dw, m, C)
    np.testing.assert_array_almost_equal(zero_result, dw)


def test_elasticnet_regularization():
    """Test Elastic Net regularization implementation"""
    # Initialize test values
    model = LogisticRegressor()
    model.weights = np.array([1.0, -2.0, 0.0, 3.0])
    dw = np.array([0.1, 0.2, -0.1, 0.3])
    m = 100
    C = 1.0
    l1_ratio = 0.5

    # Get actual result
    actual_result = model.elasticnet_regularization(dw, m, C, l1_ratio)
    expected_result = [0.11, 0.185, -0.1, 0.32]
    # Verify results
    np.testing.assert_array_almost_equal(actual_result, expected_result)

    # Test boundary conditions for l1_ratio
    # When l1_ratio = 1, should be equivalent to Lasso
    elasticnet_as_lasso = model.elasticnet_regularization(dw, m, C, l1_ratio=1.0)
    np.testing.assert_array_almost_equal(elasticnet_as_lasso, [0.11, 0.19, -0.1, 0.31])

    # When l1_ratio = 0, should be equivalent to Ridge

    elasticnet_as_ridge = model.elasticnet_regularization(dw, m, C, l1_ratio=0.0)
    np.testing.assert_array_almost_equal(elasticnet_as_ridge, [0.11, 0.18, -0.1, 0.33])

    # Test with different C values
    C_strong = 10.0  # stronger regularization
    C_weak = 0.1  # weaker regularization

    result_strong = model.elasticnet_regularization(dw, m, C_strong, l1_ratio)
    result_weak = model.elasticnet_regularization(dw, m, C_weak, l1_ratio)

    # Verify that stronger regularization has less impact on gradient
    assert np.all(abs(result_strong - dw) >= abs(result_weak - dw))


@pytest.mark.parametrize("penalty", [None, "lasso", "ridge", "elasticnet"])
def test_regularization(sample_data, penalty):
    """Test different regularization methods"""
    model = LogisticRegressor()
    X_train, X_test, y_train, y_test = sample_data

    if penalty == "elasticnet":
        model.fit(
            X_train, y_train, num_iterations=100, penalty=penalty, l1_ratio=0.5, C=1.0
        )
    else:
        model.fit(X_train, y_train, num_iterations=100, penalty=penalty, C=1.0)

    # Check if model can make predictions
    predictions = model.predict(X_test)
    assert predictions.shape == (X_test.shape[0],)


def test_sigmoid():
    """Test the sigmoid function"""
    test_values = np.array([-np.inf, -1, 0, 1, np.inf])
    results = LogisticRegressor.sigmoid(test_values)

    assert np.allclose(results, [0, 0.26894142, 0.5, 0.73105858, 1], rtol=1e-7)
    assert np.all((results >= 0) & (results <= 1))


def test_maximum_likelihood():
    """Test the maximum likelihood loss calculation"""
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.8, 0.2])

    loss = LogisticRegressor.log_likelihood(y_true, y_pred)
    assert isinstance(loss, float)
    assert loss >= 0
