import numpy as np

class GaussianAnomalyDetector:
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.epsilon = None

    def fit(self, X_train):
        """Estimate mean and standard deviation for each feature."""
        self.mu = X_train.mean()
        self.sigma = X_train.std()
        assert np.all(self.sigma > 0), "All standard deviations must be positive"

    def pdf(self, x):
        """Compute the probability density function for a given observation."""
        exponent = np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * exponent

    def predict(self, X, epsilon):
        """Predict anomalies based on a probability threshold."""
        probs = np.prod(self.pdf(X), axis=1)
        return (probs < epsilon).astype(int)