"""Reproducible demonstration of the original models, using synthetic data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegressor
from gradient_descent import LinearRegressor as GradientRegressor
from logistic_regression import LogisticRegressor


def main():
    rng = np.random.default_rng(42)
    np.random.seed(42)
    X = rng.uniform(-2, 2, (180, 1))
    y = 1.5 + 2.2 * X[:, 0] + rng.normal(0, .5, len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    least_squares = LinearRegressor()
    least_squares.fit_simple(X_train[:, 0], y_train)
    descent = GradientRegressor()
    descent.fit(X_train, y_train, method='gradient_descent', learning_rate=.05, iterations=2000)
    print(f'Linear regression test RMSE: {np.sqrt(mean_squared_error(y_test, least_squares.predict(X_test))):.4f}')
    print(f'Gradient descent test RMSE: {np.sqrt(mean_squared_error(y_test, descent.predict(X_test))):.4f}')

    Z = rng.normal(size=(600, 2))
    probabilities = 1 / (1 + np.exp(-(1.8 * Z[:, 0] - 1.2 * Z[:, 1] + .2)))
    labels = rng.binomial(1, probabilities)
    Z_train, Z_test, z_train, z_test = train_test_split(Z, labels, random_state=42, stratify=labels)
    classifier = LogisticRegressor()
    classifier.fit(Z_train, z_train, learning_rate=.1, num_iterations=3000)
    print(f'Logistic regression test accuracy: {accuracy_score(z_test, classifier.predict(Z_test)):.4f}')
    print(f'Logistic regression test log-loss: {log_loss(z_test, classifier.predict_proba(Z_test)):.4f}')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    line = np.linspace(-2, 2, 150).reshape(-1, 1)
    axes[0].scatter(X_test[:, 0], y_test, s=25, alpha=.65, label='Held-out samples')
    axes[0].plot(line[:, 0], least_squares.predict(line), color='#de744a', label='Least squares')
    axes[0].plot(line[:, 0], descent.predict(line), '--', color='#253e67', label='Gradient descent')
    axes[0].set(title='Linear regression: two fitting methods', xlabel='Feature', ylabel='Target')
    axes[0].legend()
    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, 120), np.linspace(-3, 3, 120))
    probs = classifier.predict_proba(np.c_[grid_x.ravel(), grid_y.ravel()]).reshape(grid_x.shape)
    axes[1].contourf(grid_x, grid_y, probs, levels=np.linspace(0, 1, 11), cmap='RdBu', alpha=.45)
    axes[1].contour(grid_x, grid_y, probs, levels=[.5], colors=['#253e67'])
    axes[1].scatter(Z_test[:, 0], Z_test[:, 1], c=z_test, cmap='RdBu', s=18, edgecolor='white', linewidth=.3)
    axes[1].set(title='Logistic regression: decision boundary', xlabel='Feature 1', ylabel='Feature 2')
    output = Path(__file__).with_name('demo.png')
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f'Figure: {output.name}')


if __name__ == '__main__':
    main()
