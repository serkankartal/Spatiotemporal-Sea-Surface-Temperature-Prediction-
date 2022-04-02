from MLForcasting import mlp,lstm,Allmodels
import os
import numpy as np
np.random.seed(1)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DATA_DIR = './data/'
MODEL_DIR="./models/svm"

train_data= np.load(DATA_DIR+"train_full_2020_6.npz")["arr_0"]
test_data= np.load(DATA_DIR+"test_full_2020_6.npz")["arr_0"]
mask= np.load(DATA_DIR+"mask.npz")["arr_0"]

train_data=train_data[np.lexsort((train_data[:, 1], train_data[:, 0], train_data[:, 3],train_data[:,2]))]
test_data=test_data[np.lexsort((test_data[:, 1], test_data[:, 0], test_data[:, 3],test_data[:,2]))]

# Allmodels.CALLFunc(train_v=train_data, test_v=test_data, mask=mask,svm_degree=1)
# lstm_test2.CALLFunc(train_v=train_data, test_v=test_data, mask=mask)
mlp.CALLFunc(train_v=train_data, test_v=test_data, mask=mask)