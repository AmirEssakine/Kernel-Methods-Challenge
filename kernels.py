import numpy as np
from collections import Counter
from numba import njit, prange
import itertools
from tqdm import tqdm
from scipy import sparse


# The Convolutional Kitchen Sinks 
class ConvNgramKernel:
    def __init__(self, gamma=10.0, n=8, k=4, M=4096, random_state=None):
        self.gamma = gamma
        self.n = n
        self.k = k
        self.M = M
        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random
        self.ws = self.rng.randn(self.M, self.k * self.n).astype(np.float32) * self.gamma
        self.bs = self.rng.uniform(0, 2 * np.pi, self.M).astype(np.float32)
    
    def one_hot_encode(self, sequences):
        mapping = {"A": [1, 0, 0, 0],
                   "T": [0, 1, 0, 0],
                   "G": [0, 0, 1, 0],
                   "C": [0, 0, 0, 1]}
        encoded = np.array([[mapping[char] for char in seq] for seq in sequences], dtype=np.float32)
        return encoded.reshape(len(sequences), -1)
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def compute_features(X, ws, bs, start, end, M, n, k):
        batch_size = end - start
        d = X.shape[1] // k
        phi_batch = np.zeros((batch_size, M), dtype=np.float32)
        for i in prange(batch_size):
            seq_idx = start + i
            for j in prange(M):
                sum_cos = 0.0
                for t in range(d - n + 1):
                    dot_val = 0.0
                    for l in range(n * k):
                        dot_val += X[seq_idx, t * k + l] * ws[j, l]
                    sum_cos += np.cos(dot_val + bs[j])
                phi_batch[i, j] = np.sqrt(2.0 / M) * sum_cos
        return phi_batch
    
    def map(self, sequences, batch_size=1):
        X = np.ascontiguousarray(self.one_hot_encode(sequences))
        N = X.shape[0]
        phi = np.zeros((N, self.M), dtype=np.float32)
        for start in tqdm(range(0, N, batch_size), desc="Computing features", unit="batch"):
            end = min(start + batch_size, N)
            phi[start:end] = self.compute_features(X, self.ws, self.bs, start, end, self.M, self.n, self.k)
        return phi
    
    def kernel(self, sequences1, sequences2):
        phi_X = self.map(sequences1)
        if sequences1 is sequences2:
            phi_Y = phi_X
        else:
            phi_Y = self.map(sequences2)
        gram_matrix = np.dot(phi_X, phi_Y.T)
        return gram_matrix

# Spectrum kernel
class SpectrumKernel:
    def __init__(self, k=3):
        self.k = k
        
        
    def make_phi(self, seq, k=None):
        assert (len(seq) >= k)
        dict_ = {}
        for i in range(len(seq) - k + 1):
            target = tuple(seq[i : i + k])
            if target not in dict_:
                dict_[target] = 1
            else:
                dict_[target] += 1
        return dict_


    def kernel(self, sequences1, sequences2):
        dict1 = self.make_phi(sequences1, self.k)
        dict2 = self.make_phi(sequences2, self.k)
    
        keys = list(set(dict1.keys()) & set(dict2.keys()))
        output = 0
        for key in keys:
            output += dict1[key] * dict2[key]
        return output

class MismatchKernel:
    def __init__(self, sequences1, sequences2, k=1,n=8, charset='ATCG'):
        self.X1 = sequences1
        self.X2 = sequences2
        self.k = k
        self.charset = charset
        self.n = n
        self.all_patterns = [''.join(p) for p in itertools.product(charset, repeat=self.n)]
        # Precompute neighbors for every possible pattern
        self.neighbors = {p: self._generate_neighbors(p, self.k) for p in self.all_patterns}
    
    def _generate_neighbors(self, pattern, k):
        if k == 0:
            return {pattern}
        neighbors = set()
        for i in range(len(pattern)):
            for c in self.charset:
                new_pattern = pattern[:i] + c + pattern[i+1:]
                neighbors.add(new_pattern)
        if k > 1:
            for neighbor in list(neighbors):
                neighbors.update(self._generate_neighbors(neighbor, k - 1))
        return neighbors
    
    def _compute_counts(self, seq):
        counts = {p: 0 for p in self.all_patterns}
        L = len(seq)
        for i in range(L - self.n + 1):
            sub = seq[i:i+self.n]
            # Increase count for all neighbors of the observed substring
            for neighbor in self.neighbors.get(sub, []):
                counts[neighbor] += 1
        return counts
    
    def kernel(self,sequences1,sequences2):
        counts1 = [self._compute_counts(seq) for seq in tqdm(sequences1, desc="Counting X1")]
        counts2 = [self._compute_counts(seq) for seq in tqdm(sequences2, desc="Counting X2")]
        feat1 = np.array([[c[p] for p in self.all_patterns] for c in counts1], dtype=np.float32)
        feat2 = np.array([[c[p] for p in self.all_patterns] for c in counts2], dtype=np.float32)
        return np.dot(feat1, feat2.T)
    

class MismatchKernelWeighted:
    def __init__(self, sequences1, sequences2, k=1,n=8 ,charset='ATCG'):
        self.X1 = sequences1
        self.X2 = sequences2
        self.k = k
        self.charset = charset
        self.n =n
        self.all_patterns = [''.join(p) for p in itertools.product(charset, repeat=self.n)]
        self.neighbors = {p: self._generate_neighbors(p, self.k) for p in self.all_patterns}
    
    def _generate_neighbors(self, pattern, k):
        if k == 0:
            return {pattern}
        neighbors = set()
        for i in range(len(pattern)):
            for c in self.charset:
                new_pattern = pattern[:i] + c + pattern[i+1:]
                neighbors.add(new_pattern)
        if k > 1:
            for neighbor in list(neighbors):
                neighbors.update(self._generate_neighbors(neighbor, k - 1))
        return neighbors

    def _compute_counts_and_positions(self, seq):
        counts = {p: 0 for p in self.all_patterns}
        pos_sum = {p: 0 for p in self.all_patterns}
        L = len(seq)
        for i in range(L - self.n + 1):
            sub = seq[i:i+self.n]
            for neighbor in self.neighbors.get(sub, []):
                counts[neighbor] += 1
                pos_sum[neighbor] += i  
        return counts, pos_sum

    def kernel(self,sequences1,sequences2):
        data1 = [self._compute_counts_and_positions(seq) for seq in tqdm(sequences1, desc="Counting X1 (weighted)")]
        data2 = [self._compute_counts_and_positions(seq) for seq in tqdm(sequences2, desc="Counting X2 (weighted)")]
        counts1 = np.array([[d[0][p] for p in self.all_patterns] for d in data1], dtype=np.float32)
        pos1 = np.array([[d[1][p] for p in self.all_patterns] for d in data1], dtype=np.float32)
        counts2 = np.array([[d[0][p] for p in self.all_patterns] for d in data2], dtype=np.float32)
        pos2 = np.array([[d[1][p] for p in self.all_patterns] for d in data2], dtype=np.float32)
        
        # Normalize each set of features
        norm_counts1 = np.linalg.norm(counts1, axis=1, keepdims=True) + 1e-8
        norm_counts2 = np.linalg.norm(counts2, axis=1, keepdims=True) + 1e-8
        norm_pos1 = np.linalg.norm(pos1, axis=1, keepdims=True) + 1e-8
        norm_pos2 = np.linalg.norm(pos2, axis=1, keepdims=True) + 1e-8
        
        feat1 = counts1 / norm_counts1
        feat2 = counts2 / norm_counts2
        pos_feat1 = pos1 / norm_pos1
        pos_feat2 = pos2 / norm_pos2
        
        K_counts = np.dot(feat1, feat2.T)
        K_pos = np.dot(pos_feat1, pos_feat2.T)
        return K_counts * K_pos


class GappedKmerKernel:
    def __init__(self, k, gap):
        self.k = k
        self.gap = gap

    def kernel(self, sequences1, sequences2):
        subseq_set = set(sequences1[i:i+self.k] for i in range(len(sequences1) - self.k + 1))
        return sum(sequences2[i:i+self.k] in subseq_set for i in range(0, len(sequences2) - self.k + 1, self.gap))


class PolynomialKernel:
    def __init__(self, degree=3):
        self.degree = degree

    def kernel(self, x, y):
        return (np.dot(x, y) + 1) ** self.degree
    
class PolynomialKernel:
    def __init__(self, degree=3):
        self.degree = degree

    def kernel(self, x, y):
        return (np.dot(x, y) + 1) ** self.degree

class LinearKernel:
    def kernel(self, x, y):
        return np.dot(x, y.T)

class GaussianKernel:
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def kernel(self, x, y):
        diff = x[:, None] - y[None, :]
        return np.exp(-self.gamma * np.sum(diff * diff, axis=-1))

class SigmoidKernel:
    def __init__(self, gamma=0.1, coef0=1):
        self.gamma = gamma
        self.coef0 = coef0
    def kernel(self, X, Y):
        return np.tanh(self.gamma * np.dot(X, Y.T) + self.coef0)

class LaplacianKernel:
    def __init__(self, gamma=1.0):
        self.gamma = gamma
    def kernel(self, X, Y):
        K = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            diff = np.abs(X[i] - Y)
            K[i] = np.exp(-self.gamma * np.sum(diff, axis=1))
        return K

class CosineKernel:
    def kernel(self, X, Y):
        X_norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        Y_norm = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8
        return np.dot(X / X_norm, (Y / Y_norm).T)

class ChiSquareKernel:
    def __init__(self, gamma=1.0):
        self.gamma = gamma
    def kernel(self, X, Y):
        K = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                numerator = (X[i] - Y[j]) ** 2
                denominator = X[i] + Y[j] + 1e-8
                chi = np.sum(numerator / denominator)
                K[i, j] = np.exp(-self.gamma * chi)
        return K