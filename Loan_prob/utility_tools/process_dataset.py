# -*- coding:utf-8 -*-
'''
Created on 2016年9月2日

@author: YinongLong

数据集中的数值-1均代表空值

根据产品ID，将用户基本信息数据分为两部分进行使用
'''
from __future__ import print_function

import os.path as op

import numpy as np

default_path = 'E:/dataset/dataseriesv2/data'
mode_data_path = 'E:/dataset/dataseriesv2/model_data'

def separate_user_basic_info(filename='user_info.txt',
                             id_1_col=[0,1,2,3,5,6,7,8,9,11,12,15,21],
                             id_2_col=[0,1,2,3,4,5,6,12,21]):
    """
    载入用户的基本信息，并且根绝产品ID将数据分为两个部分
    并且在每一个部分中提取id_1_col和id_2_col指定的数据列
    """
    file_path = op.join(default_path, filename)
    data_id1_path = op.join(mode_data_path, 'pro_1_user_info.csv')
    data_id2_path = op.join(mode_data_path, 'pro_2_user_info.csv')
    if not op.exists(data_id1_path) or not op.exists(data_id2_path):
        dataset = np.loadtxt(file_path, dtype=np.str, delimiter=',')
        dataset_by_id1 = dataset[dataset[:, -3] == '1'][:, id_1_col]
        dataset_by_id1_names = dataset[dataset[:, -3] == 'product_id'][:, id_1_col]
        dataset_by_id1 = np.concatenate((dataset_by_id1_names, dataset_by_id1), axis=0)
        
        dataset_by_id2 = dataset[dataset[:, -3] == '2'][:, id_2_col]
        dataset_by_id2_names = dataset[dataset[:, -3] == 'product_id'][:, id_2_col]
        dataset_by_id2 = np.concatenate((dataset_by_id2_names, dataset_by_id2), axis=0)
        np.savetxt(data_id1_path, dataset_by_id1, '%s', delimiter=',')
        np.savetxt(data_id2_path, dataset_by_id2, '%s', delimiter=',')
    print('separating user basic info finished!')

def process_uid_data(dataset):
    nums_feature = np.shape(dataset)[1]
    dataset = dataset.tolist()
    dataset = sorted(dataset, key=lambda item: int(item[-1]))
    for i in range(1, nums_feature-1):
        
        pass
    pass

def process_product_model_dataset(filename):
    """
    载入产品模型的数据，并且进行处理
    """
    file_path = op.join(mode_data_path, filename)
    dataset = np.loadtxt(file_path, dtype=np.str, delimiter=',')
    
    nums_feature = np.shape(dataset)[1]
    for line in dataset[1:]:
        for i in range(1, nums_feature-1):
            if line[i] == 'NA' or line[i] == 'NONE':
                line[i] = '0'
                
    
    tansformed_dataset = []
    uids_set = set(dataset[1:, 0].tolist())
    nums_uid = len(uids_set)
    for i in range(nums_uid):
        uid = uids_set.pop()
        uid_data = dataset[dataset[:, 0] == uid]
        
        pass
    
    train = load_train_data()
    test = load_test_data()
    
    train_key_dict = {}
    for uid, label in train[1:]:
        train_key_dict[uid] = label
    
    test_key_set = set(test[1:, 0].tolist())
    
    train_data = []
    test_data = []
    for line in dataset[1:]:
        line = line.tolist()
        if line[0] in test_key_set:
            test_data.append(line)
        elif line[0] in train_key_dict:
            line.insert(1, train_key_dict[line[0]])
            train_data.append(line)
    return np.array(train_data), np.array(test_data)
    
    
def output_csv_dataset(dataset, filename):
    file_path = op.join(mode_data_path, filename + '.csv')
    np.savetxt(file_path, dataset, '%s', delimiter=',')
    print('%s, output Finished!' % filename)

def load_train_data(filename='train.txt'):
    """
    载入训练样本的ID和label
    """
    file_path = op.join(default_path, filename)
    dataset = np.loadtxt(file_path, dtype=np.str, delimiter=',')
    return dataset

def load_test_data(filename='test.txt'):
    """
    载入预测样本的ID
    """
    file_path = op.join(default_path, filename)
    dataset = np.loadtxt(file_path, dtype=np.str, delimiter=',')
    return dataset

def load_consumption_hidden_feature(filename='consumption_hidden_feature.csv'):
    file_path = op.join(default_path, filename)
    dataset = np.loadtxt(file_path, dtype=np.str, delimiter=',')
    return dataset

def main():
    separate_user_basic_info()
#     check_dataset()
#     load_user_basic_info()
#     pass

if __name__ == '__main__':
    main()
