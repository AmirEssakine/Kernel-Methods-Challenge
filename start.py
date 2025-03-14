import numpy as np
import pandas as pd
from data_loading import load_data
from kernels import *
from utils import * 
from features_map import *
from Kernel_PCA import *
from SVM import C_SVM

# Set data path and output path
path = r"C:\Users\Essakine\Downloads\CT\Kernel methods\Kernel_methods_data"
out_path = "./"

# Set parameters
C = 1.
# Paramaeters of the kernels 
params_list = [ 
        {'gamma': 32.0, 'n': 9,  'k': 3, 'M': 8192},
        {'gamma': 32.0, 'n': 12,  'k': 3, 'M': 8192},
        {'gamma': 32.0, 'n': 13,  'k': 3, 'M': 8192}]
# Weights of the kernels Here we are taking K = K_1 + K_2 + K_3 as a kernel
weights = None


use_pca = False
# Here we calculate the features using the kernel and then do a kernel classification(by a linear kernel)
use_kernel = True
# Here we calculate the features using embeddings and then do a kernel classification(by a linear or gaussian kernel)
use_embeddings = False

val_ratio = 0.2

data = load_data(path)
print("Size of dataset: ", data['TF2']['X_train'].shape, data['TF2']['X_test'].shape)


pred_tests = {}

if use_kernel :
    for j in range(3):
        # Load dataset
        dataset_key = f'TF{j}'
        X_full = data[dataset_key]['X_train']
        y_full = data[dataset_key]['y_train']
        X_test = data[dataset_key]['X_test']   
        # Partition dataset to train and validation 
        n = X_full.shape[0]
        p = int(val_ratio * n) 
        idx_val = np.random.choice(n, p, replace=False)
        idx_train = np.setdiff1d(np.arange(n), idx_val)
        X_train, y_train = X_full[idx_train], y_full[idx_train]
        X_val, y_val = X_full[idx_val], y_full[idx_val]
        
        kernel = ConvNgramKernel
        kernel_train_list,kernel_val_list,kernel_test_list = compute_features_from_kernel(kernel,X_train,X_val,X_test,params_list)
        K_train_combined = combine_kernels(kernel_train_list, weights)
        K_val_combined = combine_kernels(kernel_val_list, weights)
        K_test_combined = combine_kernels(kernel_test_list, weights)
                
        print('--Begin fitting')
        algo = C_SVM(kernel=None,features_map=K_train_combined)
        algo.fit(X_train,y_train)
        pred_train = algo.predict(X_train,K_train_combined)
        print('Train dataset accuracy:', score(pred_train, y_train))
        
        # Evaluate on the evaluation set
        y_pred = algo.predict(x_test=None,features_map_test=K_val_combined)
        acc = score(y_val, y_pred)
        print(f"Combined Kernel SVM Classification Accuracy: {acc:.4f}")
        
        # Prediction on the test dataset
        pred_test = algo.predict(x_test=None,features_map_test=K_test_combined)
        pred_tests[dataset_key] = pred_test

if use_embeddings == True :        
    for j in range(3):
        # Load dataset
        dataset_key = f'TF{j}'
        X_full = data[dataset_key]['X_train']
        y_full = data[dataset_key]['y_train']
        X_test = data[dataset_key]['X_test']    
        n = X_full.shape[0]
        p = int(val_ratio * n) 
        idx_val = np.random.choice(n, p, replace=False)
        idx_train = np.setdiff1d(np.arange(n), idx_val)
        X_train, y_train = X_full[idx_train], y_full[idx_train]
        X_val, y_val = X_full[idx_val], y_full[idx_val]
        
        #Compute embedding
        features_map = ConvNgramKernel_features(gamma=10.0, n=8, k=4, M=4096, random_state=None)
        X_train_embed = features_map(X_train)
        X_val_embed = features_map(X_val)
        X_test_embed = features_map(X_test)
      
        print('--Begin fitting')
        kernel = LinearKernel()
        algo = C_SVM(kernel)
        algo.fit(X_train_embed, y_train)
        pred_train = algo.predict(X_train_embed)
        print('Train dataset accuracy:', score(pred_train, y_train))
        
        # Evaluate on the evaluation set
        y_pred = algo.predict(X_val_embed)
        acc = score(y_val, y_pred)
        print(f"Combined Kernel SVM Classification Accuracy: {acc:.4f}")
            
        # Prediction on the test dataset
        pred_test = algo.predict(X_test_embed)
        pred_tests[dataset_key] = pred_test
        

filename = "Yte"
saving = True
if saving:
    final_preds = np.concatenate([pred_tests[f'TF{j}'] for j in range(3)])
    data_id = np.arange(final_preds.shape[0])
    data_val = np.where(final_preds == -1, 0, 1)
    
    df = pd.DataFrame({"Id": data_id, "Bound": data_val})
    df.to_csv(out_path + filename + ".csv", index=False)
    print(f"Best test predictions saved to {out_path + filename}.csv")