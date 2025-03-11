import numpy as np

# Kernel Logistic Regression method
class KernelLogisticRegression:
    def __init__(self, kernel, lambda_reg=1e-3, learning_rate=0.01, num_iters=1000, verbose=False):
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.verbose = verbose

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        n_samples = X.shape[0]
        K = self.kernel(X, X)
        self.alpha = np.zeros(n_samples)
        for i in range(self.num_iters):
            f = np.dot(K, self.alpha)
            p = self._sigmoid(f)
            error = p - y
            grad = np.dot(K, error) + self.lambda_reg * np.dot(K, self.alpha)
            self.alpha -= self.learning_rate * grad
            if self.verbose and i % 100 == 0:
                loss = -np.mean(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8)) + (self.lambda_reg / 2) * np.dot(self.alpha, np.dot(K, self.alpha))
                print(f"Iteration {i}, loss = {loss:.6f}")

    def decision_function(self, X):
        K = self.kernel(self.X_train, X)
        return np.dot(self.alpha, K)

    def predict_proba(self, X):
        f = self.decision_function(X)
        proba = self._sigmoid(f)
        return np.vstack([1 - proba, proba]).T

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

# Normal logistic regression
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iters=1000, lambda_reg=1e-3, verbose=False):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.lambda_reg = lambda_reg
        self.verbose = verbose

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for i in range(self.num_iters):
            linear_model = np.dot(X, self.w) + self.b
            p = self._sigmoid(linear_model)
            error = p - y
            dw = (np.dot(X.T, error) / n_samples) + self.lambda_reg * self.w
            db = np.mean(error)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            if self.verbose and i % 100 == 0:
                loss = -np.mean(y * np.log(p + 1e-8) + (1 - y) * np.log(1 - p + 1e-8)) + (self.lambda_reg / 2) * np.dot(self.w, self.w)
                print(f"Iteration {i}, loss = {loss:.6f}")

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict_proba(self, X):
        linear_model = self.decision_function(X)
        p = self._sigmoid(linear_model)
        return np.vstack([1 - p, p]).T

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)