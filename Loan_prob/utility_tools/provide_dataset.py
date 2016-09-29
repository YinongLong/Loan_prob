# -*- coding:utf-8 -*-
'''
Created on 2016年9月24日

@author: YinongLong

用来给学习模型提供数据的模块，直接从本地保存的处理好的文件入手
'''
from __future__ import print_function

from process_dataset import process_product_model_dataset

import os.path as op

import numpy as np

import h5py

default_dir = 'E:/dataset/dataseriesv2/model_data'
raw_data_dir = 'E:/dataset/dataseriesv2/data'

def get_train_dataset(model_id):
    filename = 'pro_%s_user_info.csv' % model_id
    save_path = op.join(default_dir, model_id)
    if op.exists(save_path):
        data_file = h5py.File(save_path, 'r')
        train = np.array(data_file.get('train'))
        test = np.array(data_file.get('test'))
        return train, test
    else:
        train, test = process_product_model_dataset(filename)
        data_file = h5py.File(save_path, 'w')
        data_file.create_dataset('train', data=train)
        data_file.create_dataset('test', data=test)
        print('%s model data have saved!' % model_id)
        return train, test

def main():
    pass

if __name__ == '__main__':
    main()