"""Additional checks added for the portfolio; original model code is unchanged."""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from linear_regression import LinearRegressor
from gradient_descent import LinearRegressor as GradientRegressor
from logistic_regression import LogisticRegressor
from cross_validation import cross_validation


def test_least_squares_matches_sklearn_on_full_rank_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 4))
    y = X @ np.array([2., -1., .5, 3.]) + 4 + rng.normal(0, .1, 120)
    custom = LinearRegressor()
    custom.fit_multiple(X, y)
    reference = LinearRegression().fit(X, y)
    np.testing.assert_allclose(custom.predict(X), reference.predict(X), atol=1e-10)


def test_gradient_descent_reaches_least_squares_solution():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(100, 3))
    y = 2 + X @ np.array([1., -2., 3.])
    np.random.seed(42)
    custom = GradientRegressor()
    custom.fit(X, y, method='gradient_descent', learning_rate=.05, iterations=2000)
    reference = LinearRegression().fit(X, y)
    np.testing.assert_allclose(custom.predict(X), reference.predict(X), atol=1e-6)


def test_logistic_probabilities_match_unregularized_reference():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 3))
    p = 1 / (1 + np.exp(-(X @ np.array([1., -1., .5]) + .3)))
    y = rng.binomial(1, p)
    custom = LogisticRegressor()
    custom.fit(X, y, learning_rate=.1, num_iterations=4000)
    reference = LogisticRegression(penalty=None, tol=1e-10, max_iter=2000).fit(X, y)
    np.testing.assert_allclose(custom.predict_proba(X), reference.predict_proba(X)[:, 1], atol=1e-5)
    assert log_loss(y, custom.predict_proba(X)) < log_loss(y, np.full(len(y), .5))


def test_regularization_strength_follows_implemented_convention():
    model = LogisticRegressor()
    model.weights = np.array([2., -3.])
    zero = np.zeros(2)
    # This educational implementation uses C directly, unlike sklearn's inverse C.
    np.testing.assert_allclose(model.ridge_regularization(zero, 10, 4), [.8, -1.2])
    np.testing.assert_allclose(model.lasso_regularization(zero, 10, 4), [.4, -.4])


def test_cross_validation_matches_kfold_for_even_folds():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3))
    y = (X[:, 0] > 0).astype(int)
    scores = cross_val_score(KNeighborsClassifier(3), X, y, cv=KFold(5, shuffle=False))
    mean, std = cross_validation(KNeighborsClassifier(3), X, y, nFolds=5)
    np.testing.assert_allclose([mean, std], [scores.mean(), scores.std()])
