"""Regression tests for the small fixes applied to the original exercises."""
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from knn import knn, plot_roc_curve
from lda import LinearDiscriminant


@pytest.mark.parametrize('labels', [np.array([10, 20, 20]), np.array(['a', 'b', 'b'])])
def test_knn_votes_for_labels_not_row_indices(labels):
    X = np.array([[0.], [1.], [2.]])
    model = knn()
    model.fit(X, labels, k=1)
    assert model.predict(np.array([[2.]]))[0] == labels[2]
    np.testing.assert_allclose(model.predict_proba(np.array([[2.]])), [[0., 1.]])


@pytest.mark.parametrize('p', [1, 2, 3])
def test_knn_matches_reference(p):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 4))
    y = np.array(['blue', 'green', 'red'])[rng.integers(0, 3, 80)]
    queries = rng.normal(size=(15, 4))
    custom = knn()
    custom.fit(X, y, k=5, p=p)
    reference = KNeighborsClassifier(n_neighbors=5, p=p, algorithm='kd_tree').fit(X, y)
    np.testing.assert_array_equal(custom.predict(queries), reference.predict(queries))
    np.testing.assert_allclose(custom.predict_proba(queries), reference.predict_proba(queries))


def test_knn_rejects_more_neighbors_than_samples():
    with pytest.raises(ValueError, match='cannot exceed'):
        knn().fit(np.zeros((2, 1)), np.array([0, 1]), k=3)


def test_roc_supports_string_positive_label():
    result = plot_roc_curve(np.array(['yes', 'no']), np.array([.9, .1]), 'yes')
    assert result['tpr'][5] == 1
    assert result['fpr'][5] == 0


@pytest.mark.parametrize('labels', [np.array([10, 10, 10, 20, 20, 20]), np.array(['a', 'a', 'a', 'b', 'b', 'b'])])
def test_lda_is_invariant_to_label_names(labels):
    X = np.array([[-2., -1.], [-1., -1.], [-1., -2.], [1., 1.], [1., 2.], [2., 1.]])
    model = LinearDiscriminant()
    model.fit(X, labels)
    np.testing.assert_array_equal(model.predict(X), labels)
    scores = model.predict_proba(X)
    np.testing.assert_allclose(scores.sum(axis=1), 1)
    np.testing.assert_array_equal(model.class_labels[scores.argmax(axis=1)], model.predict(X))
    extreme_scores = model.predict_proba(np.array([[1e6, 1e6]]))
    assert np.isfinite(extreme_scores).all()
    np.testing.assert_allclose(extreme_scores.sum(axis=1), 1)


def test_lda_refit_recomputes_default_dimension():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(60, 3))
    model = LinearDiscriminant()
    model.fit(X, np.repeat([0, 1], 30))
    assert model.transform(X).shape == (60, 1)
    model.fit(X, np.repeat([0, 1, 2], 20))
    assert model.transform(X).shape == (60, 2)


@pytest.mark.parametrize('n_components', [0, 2])
def test_lda_validates_dimension(n_components):
    with pytest.raises(ValueError, match='n_components'):
        LinearDiscriminant(n_components).fit(np.array([[0., 1.], [2., 3.]]), np.array([0, 1]))
