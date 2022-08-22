import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_data(dataset='MSL', path='./dataset/'):
    if dataset in ['MSL', 'PSM', 'SMAP', 'SMD', 'SWaT', 'WADI']:
        data = {}
        train_valid_data = np.load(path + dataset + '/' + dataset + '_train.npy')
        train_valid_label = np.zeros(len(train_valid_data))

        thre_test_data = np.load(path + dataset + '/' + dataset + '_test.npy')
        thre_test_label = np.load(path + dataset + '/' + dataset + '_test_label.npy').astype(int)

        scaler = StandardScaler()
        train_valid_data = pd.DataFrame(scaler.fit_transform(train_valid_data))
        thre_test_data = pd.DataFrame(scaler.transform(thre_test_data))

        train_valid_data = train_valid_data.fillna(0).values
        thre_test_data = thre_test_data.fillna(0).values

        data['train_data'] = train_valid_data[:int(0.7 * len(train_valid_data)), :]
        data['train_label'] = train_valid_label[:int(0.7 * len(train_valid_label))]
        data['valid_data'] = train_valid_data[int(0.7 * len(train_valid_data)):, :]
        data['valid_label'] = train_valid_label[int(0.7 * len(train_valid_label)):]
        data['thre_data'] = thre_test_data[:int(0.3 * len(thre_test_data)), :]
        data['thre_label'] = thre_test_label[:int(0.3 * len(thre_test_label))]
        data['test_data'] = thre_test_data[int(0.3 * len(thre_test_data)):, :]
        data['test_label'] = thre_test_label[int(0.3 * len(thre_test_label)):]
        return data

    else:
        print('Dataset name must in ["MSL", "PSM", "SMAP", "SMD", "SWaT", "WADI"]')
