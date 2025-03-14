from scipy.linalg import eigh
import numpy as np 

def kernel_pca(X,kernel, gamma=0.01, degree=3, n_components=10):       
    
    K = kernel(X)
            
    # Center the Gram matrix.    
    N = K.shape[0]
    one_n = np.ones((N,N)) / N    
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n) 
           
    eigvals, eigvecs = eigh(K)    
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
            
    X_pc = np.column_stack([eigvecs[:, i] for i in range(n_components)])        
    return X_pc

