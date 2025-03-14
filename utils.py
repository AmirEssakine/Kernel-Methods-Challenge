import numpy as np
from collections import Counter
from numba import njit, prange
import os
import pandas as pd

def load_data(datapath):
    df_dict = {}
    for filename in os.listdir(datapath):
        if filename.endswith('.csv'):
            df_dict[filename] = pd.read_csv(os.path.join(datapath, filename))
    
    data = {}
    for i in range(3):
        tf_key = f"TF{i}"
        X_train = np.array(df_dict[f"Xtr{i}.csv"]['seq'])
        X_test  = np.array(df_dict[f"Xte{i}.csv"]['seq'])
        
        y_train = df_dict[f"Ytr{i}.csv"]['Bound'].values.ravel()
        # Convert zeros to -1 for classification
        y_train = np.where(y_train == 0, -1, y_train)
        
        data[tf_key] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train
        }
    return data

def combine_kernels(kernel_list, weights=None):
    #Combine multiple kernel matrices by weighted sum.
    n_kernels = len(kernel_list)
    if weights is None:
        weights = [1.0/n_kernels] * n_kernels
    combined_kernel = np.zeros_like(kernel_list[0])
    for w, K in zip(weights, kernel_list):
        combined_kernel += w * K
    return combined_kernel

def compute_features_from_kernel (method,X_train,X_val,X_test,params_list) :
    kernel_train_list = []
    kernel_val_list = []
    kernel_test_list = []
    for params in params_list:
        print(f"Computing kernel with parameters: {params}")
        conv_kernel = method(**params)
        K_train_temp = conv_kernel.kernel(X_train, X_train)
        K_val_temp = conv_kernel.kernel(X_val, X_train)
        K_test_temp  = conv_kernel.kernel(X_test, X_train)
        kernel_train_list.append(K_train_temp)
        kernel_val_list.append(K_val_temp)
        kernel_test_list.append(K_test_temp)
    return kernel_train_list,kernel_val_list,kernel_test_list

def combine_kernels(kernel_list, weights=None):
    #Combine multiple kernel matrices by weighted sum.
    n_kernels = len(kernel_list)
    if weights is None:
        weights = [1.0/n_kernels] * n_kernels
    combined_kernel = np.zeros_like(kernel_list[0])
    for w, K in zip(weights, kernel_list):
        combined_kernel += w * K
    return combined_kernel

def score(pred, labels):
    return np.mean(pred == labels)