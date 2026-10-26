import numpy as np

class LinearDiscriminant:
    """
    Linear Discriminant Analysis (LDA) class.

    This class implements the LDA algorithm for dimensionality reduction and 
    finding the linear combination of features that best separates two or more 
    classes of objects or events.

    Attributes:
        weights (np.ndarray): The linear discriminants (eigenvectors) that can
            be used to transform the data into a lower-dimensional space.
    """

    def __init__(self, n_components=None):
        """
        Initializes the LinearDiscriminant instance with the number of components.
        
        Args:
            n_components (int, optional): Number of linear discriminants to retain. If None,
                                        all components are kept.
        """
        self.n_components = n_components
        self.weights = None
        self.class_means = None
        self.priors = None
        self.class_labels = None

    def fit(self, X, y):
        """
        Fit the LDA model according to the given training data.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features), 
                where n_samples is the number of samples and n_features is 
                the number of features.
            y (np.ndarray): Target values, shape (n_samples,), where n_samples
                is the number of samples.

        Returns:
            None
        """
        n_features = X.shape[1]  # Number of features in X
        self.class_labels = np.unique(y)  # Find the unique class labels in y
        n_classes = len(self.class_labels)  # Determine the number of unique classes

        # Check if n_components is not set, use min(n_features, n_classes - 1)
        max_components = min(n_features, n_classes - 1)
        if n_classes < 2:
            raise ValueError("LDA requires at least two classes.")
        n_components = max_components if self.n_components is None else self.n_components
        if not isinstance(n_components, (int, np.integer)) or not 1 <= n_components <= max_components:
            raise ValueError("n_components must be between 1 and min(n_features, n_classes - 1).")

        # Calculate the mean vectors for each class
        mean_vectors = []
        for label in self.class_labels:
            mean_vectors.append(np.mean(X[y == label], axis=0))
        self.class_means = np.array(mean_vectors)  # Store as numpy array

        # Compute the overall mean of the data
        overall_mean = np.mean(X, axis=0)

        # Compute S_W and S_B
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        for i, mean_vec in enumerate(self.class_means):
            # Get samples for class i
            class_samples = X[y == self.class_labels[i]]
            n_samples_class = class_samples.shape[0]
            
            # Compute within-class scatter matrix
            for sample in class_samples:
                diff = (sample - mean_vec).reshape(-1, 1)
                S_W += diff @ diff.T
            
            # Compute between-class scatter matrix
            mean_diff = (mean_vec - overall_mean).reshape(-1, 1)
            S_B += n_samples_class * (mean_diff @ mean_diff.T)

        # Solve the eigenvalue problem
        try:
            eigen_values, eigen_vectors = np.linalg.eig(np.linalg.inv(S_W) @ S_B)
        except np.linalg.LinAlgError:
            # If S_W is singular, use pseudo-inverse
            eigen_values, eigen_vectors = np.linalg.eig(np.linalg.pinv(S_W) @ S_B)

        # Retain the real part of the scatter-matrix eigenproblem solution.
        eigen_values = np.real(eigen_values)
        eigen_vectors = np.real(eigen_vectors)

        # Sort eigenvectors by eigenvalues in descending order
        eigen_pairs = [(eigen_values[i], eigen_vectors[:, i]) for i in range(len(eigen_values))]
        eigen_pairs.sort(key=lambda x: x[0], reverse=True)

        # Extract the top n_components eigenvectors
        self.weights = np.array([eigen_pairs[i][1] for i in range(n_components)]).T

        # Store transformed class means for prediction
        self.transformed_means = self.class_means @ self.weights

        # Calculate priors as a numpy array
        self.priors = np.zeros(n_classes)
        total_samples = X.shape[0]
        for i, label in enumerate(self.class_labels):
            self.priors[i] = np.sum(y == label) / total_samples

    def transform(self, X):
        """
        Project the data onto the top linear discriminants.
        
        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features).
        
        Returns:
            X_transformed (np.ndarray): Data projected onto the selected linear discriminants,
                                        shape (n_samples, n_components).
        """
        X_transformed = X @ self.weights
        return X_transformed

    def fit_transform(self, X, y):
        """
        Fit to data, then transform it.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values, shape (n_samples,).

        Returns:
            X_lda (np.ndarray): Transformed data, shape (n_samples, n_components),
                where n_components <= n_classes - 1.
        """
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        This method applies the learned linear discriminant analysis model to
        predict the class labels of the given samples. The prediction is based
        on which class mean (in the transformed space) the sample is closest to.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features), where
                n_samples is the number of samples and n_features is the number
                of features.

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,), where each
                entry is the predicted class label for the corresponding sample
                in X.
        """
        # Project data onto the lower-dimensional space
        X_transformed = self.transform(X)

        # Calculate distances to each class mean in transformed space
        distances = []
        for sample in X_transformed:
            sample_distances = []
            for i, label in enumerate(self.class_labels):
                dist = np.linalg.norm(sample - self.transformed_means[i])
                sample_distances.append(dist)
            distances.append(sample_distances)
        distances = np.array(distances)

        # Find the class with minimum distance for each point
        predictions = self.class_labels[np.argmin(distances, axis=1)]

        return predictions

    def predict_proba(self, X):
        """
        Return normalized distance scores (not calibrated LDA posteriors).

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted class probabilities, shape (n_samples, n_classes).
        """
        # Project data onto the lower-dimensional space
        X_transformed = self.transform(X)
        
        # Calculate distances to each class mean in transformed space
        distances = np.zeros((X_transformed.shape[0], len(self.class_labels)))
        for i, label in enumerate(self.class_labels):
            distances[:, i] = np.linalg.norm(
                X_transformed - self.transformed_means[i], axis=1
            )
        
        # Convert distances to probabilities using softmax
        # Negate distances so closer points have higher probability
        exp_dists = np.exp(-(distances - distances.min(axis=1, keepdims=True)))
        probabilities = exp_dists / np.sum(exp_dists, axis=1, keepdims=True)
        
        return probabilities
    
    # EXTRA: To use Cross Validation
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters:
        - X: array-like of shape (n_samples, n_features)
            Test samples.
        - y: array-like of shape (n_samples,)
            True labels for X.
        
        Returns:
        - score: float
            Mean accuracy of self.predict(X) with respect to y.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)