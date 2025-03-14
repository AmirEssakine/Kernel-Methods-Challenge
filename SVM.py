import numpy as np
from scipy import optimize
from cvxopt import matrix, solvers

import numpy as np
from scipy import optimize
import cvxopt
import cvxopt.solvers


class C_SVM:
    def __init__(self, kernel,features_map=None, C=10., solver='BFGS'):
        
        self.C = C
        self.solver = solver
        self.kernel = kernel
        self.features_map = features_map
        self.alpha = None
        self.b = 0  # Bias term

    def loss(self, alpha):
        """Loss function for optimization"""
        return 0.5 * np.dot(alpha.T, np.dot(self.K, alpha)) - np.sum(alpha)

    def jac(self, alpha):
        """Gradient of the loss function"""
        return np.dot(self.K, alpha) - np.ones_like(alpha)

    def fit(self, X, y, suffix='test', save=False):
        """Train the SVM model"""
        self.X = X
        self.y = y

        # Compute Gram Matrix
        if self.kernel == None :
            self.K = self.features_map
        else: 
            self.K = self.kernel(self.X, self.X) 
            
        if save:
            np.save(suffix, self.K)
            print(f"Saving kernel matrix: {suffix}")

        n = self.K.shape[0]

        if self.solver == 'BFGS':
            # Initialization
            alpha0 = np.random.rand(n)

            # SVM constraints: 0 <= alpha_i <= C
            bounds = [(0, self.C) for _ in range(n)]

            # Solve optimization
            res = optimize.fmin_l_bfgs_b(self.loss, alpha0, fprime=self.jac, bounds=bounds)
            self.alpha = res[0]

        elif self.solver == 'CVX':
            cvxopt.solvers.options['show_progress'] = False

            P = cvxopt.matrix(self.K.astype(float))
            q = cvxopt.matrix(-np.ones(n))
            G = cvxopt.matrix(np.vstack([-np.eye(n), np.eye(n)]))
            h = cvxopt.matrix(np.hstack([np.zeros(n), np.full(n, self.C)]))
            A = cvxopt.matrix(y.astype(float), (1, n))
            b = cvxopt.matrix(0.0)

            sol = cvxopt.solvers.qp(P, q, G, h, A, b)
            self.alpha = np.ravel(sol['x'])

        # Compute bias term (b)
        support_vector_indices = (self.alpha > 1e-5)
        self.b = np.mean(
            y[support_vector_indices] - np.dot(self.K[support_vector_indices], self.alpha * y)
        )

    def predict(self, x_test,features_map_test):
        """Make predictions"""
        if self.kernel == None :
            K = features_map_test
        else :
            K = self.kernel(x_test, self.X)
        return np.sign(np.dot(K, self.alpha * self.y) + self.b)

    def score(self, pred, labels):
        """Compute accuracy"""
        return np.mean(pred == labels)