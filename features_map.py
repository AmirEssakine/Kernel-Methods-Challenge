### Feature map
import numpy as np
from numba import njit, prange
from tqdm import tqdm
import itertools
from collections import deque
from joblib import Parallel, delayed
from scipy.sparse import lil_matrix


# Caclulating features using spectrum embedding 
class SpectrumEmbedding:
    def __init__(self, k=3, alphabet="ATGC"):
        self.k = k
        self.alphabet = alphabet
        self.kmers = self.get_all_kmers()

    def get_all_kmers(self):
        return [''.join(p) for p in itertools.product(self.alphabet, repeat=self.k)]

    def spectrum_embedding(self, seq):
        vec = np.zeros(len(self.kmers))
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i+self.k]
            if kmer in self.kmers:
                idx = self.kmers.index(kmer)
                vec[idx] += 1
        if vec.sum() > 0:
            vec = vec / vec.sum()
        return vec

    def map(self, sequences):
        return np.array([self.spectrum_embedding(seq) for seq in sequences])
    
    
    
# Caclulating features using mismatch embedding 
class MismatchEmbedding:
    def __init__(self, k, m, alph="ATGC"):
        self.k = k
        self.m = m
        self.alph = alph
        self.precomputed = {}

    def get_kmer_set(self, sequences, k):
        kmer_set = {}
        idx = 0
        for x in tqdm(sequences, desc="Populating k-mer set"):
            for j in range(len(x) - k + 1):
                kmer = x[j: j + k]
                if kmer not in kmer_set:
                    kmer_set[kmer] = idx
                    idx += 1
        return kmer_set

    def m_neighborhood(self, kmer, m):
        mismatch_list = deque([(0, "")])
        for letter in kmer:
            num_candidates = len(mismatch_list)
            for _ in range(num_candidates):
                mismatches, candidate = mismatch_list.popleft()
                if mismatches < m:
                    for a in self.alph:
                        mismatch_list.append((mismatches + (a != letter), candidate + a))
                else:
                    mismatch_list.append((mismatches, candidate + letter))
        return [candidate for mismatches, candidate in mismatch_list]

    def map(self,sequences):
        n_samples = len(self.sequences)
        self.kmer_set = self.get_kmer_set(sequences, self.k)
        embedding = lil_matrix((n_samples, len(self.kmer_set)), dtype=np.float32)
        
        def process_sequence(i, x):
            row = lil_matrix((1, len(self.kmer_set)), dtype=np.float32)
            for j in range(len(x) - self.k + 1):
                kmer = x[j: j + self.k]
                if kmer not in self.precomputed:
                    neighborhood = self.m_neighborhood(kmer, self.m)
                    self.precomputed[kmer] = [self.kmer_set[n] for n in neighborhood if n in self.kmer_set]
                for idx in self.precomputed[kmer]:
                    row[0, idx] += 1
            return i, row

        results = Parallel(n_jobs=-1)(
            delayed(process_sequence)(i, x) for i, x in enumerate(self.sequences)
        )
        for i, row in results:
            embedding[i] = row
        return embedding.tocsr()

