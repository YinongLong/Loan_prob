# -*- coding:utf-8 -*-
'''
Created on 2016年9月13日

@author: YinongLong
'''
from __future__ import print_function

import os.path as op
import numpy as np
import csv

default_path = 'E:/dataset/dataseriesv2/data'
explore_path = 'E:/dataset/dataseriesv2/explore_data'

def print_categories_num(dataset, special_col=None, silent=False):
    if silent: return
    print('=='*10, 'categories num', '=='*10)
    num_cols = np.size(dataset, axis=1)
    for i in range(num_cols):
        set_feature = set(dataset[1:, i])
        set_feature.discard('NA')
        set_feature.discard('')
        if special_col != None and i in special_col:
            set_feature.discard('0')
        if i == 1:
            set_feature.discard('NONE')
        print('%s:%f' % (dataset[0, i], len(set_feature)))
        if len(set_feature) <= 10:
            print(set_feature)

def print_null_ratio(dataset, special_col=None, silent=False):
    """
    输出数据集每一列的缺失比例
    """
    if silent: return
    print('=='*10, 'null_ratio', '=='*10)
    num_cols = np.size(dataset, axis=1)
    num_sample = np.size(dataset, axis=0)
    for i in range(num_cols):
        null_sum = 0
        null_sum += np.sum(dataset[1:, i] == 'NA')
        null_sum += np.sum(dataset[1:, i] == '')
        if special_col != None and i in special_col:
            null_sum += np.sum(dataset[1:, i] == '0')
        if i == 1:
            null_sum += np.sum(dataset[1:, 2] == 'NONE')
        print('%s:%f' % (dataset[0, i], null_sum * 1.0 / num_sample))

def display_feature_info(pid, dataset, i, nums_all, fea_name):
    nums_zero = np.sum(dataset[:, i] == '0')
    nums_null = np.sum(dataset[:, i] == '')
    nums_na = np.sum(dataset[:, i] == 'NA')
    nums_none = np.sum(dataset[:, i] == 'NONE')
    result = '%s%s%s\n' % ('==' * 10, pid, '=' * 17)
    result += '%d:%s "0" : %d/%d  %f\n' % (i, fea_name, nums_zero, nums_all, nums_zero*1.0/nums_all)
    result += '%d:%s "空值" : %d/%d  %f\n' % (i, fea_name, nums_null, nums_all, nums_null*1.0/nums_all)
    result += '%d:%s "NA" : %d/%d  %f\n' % (i, fea_name, nums_na, nums_all, nums_na*1.0/nums_all)
    result += '%d:%s "NONE" : %d/%d  %f\n' % (i, fea_name, nums_none, nums_all, nums_none*1.0/nums_all)
    return result

def explore_basic_info(filename='user_info.txt'):
    """
    对用户的基本信息进行分析
    """
    file_path = op.join(default_path, filename)
    dataset = np.loadtxt(file_path, dtype=np.str, delimiter=',')
    # 根据产品的ID对数据的每一个属性进行处理
    data_id1 = dataset[dataset[:, -2] == '1']
    nums_id1_sample = np.shape(data_id1)[0]
    data_id2 = dataset[dataset[:, -2] == '2']
    nums_id2_sample = np.shape(data_id2)[0]
    # 对每一个产品下的用户基本信息数据显示统计信息
    nums_feature = np.shape(dataset)[1]
    info_static_id1 = []
    info_static_id2 = []
    for i in range(1, nums_feature-2):
        info_static_id1.append(display_feature_info('1', data_id1, i, nums_id1_sample, dataset[0, i]))
        info_static_id2.append(display_feature_info('2', data_id2, i, nums_id2_sample, dataset[0, i]))
    save_path = op.join(explore_path, 'basic_info_vals.txt')
    data_file = open(save_path, 'wb')
    for item_1, item_2 in zip(info_static_id1, info_static_id2):
        data_file.write(item_1)
        data_file.write(item_2)
    data_file.close()
    print('exploring user basic info finished!')
    
def explore_train_info(filename='train.txt'):
    """
    对给定的训练信息进行分析，其中仅有用户的ID和类别label
    正负样本的比例很好，接近1：1，且无重复的ID
    负样本（0）：13141
    正样本（1）：12859
    样本总数：26000
    """
    file_path = op.join(default_path, filename)
    dataset = np.loadtxt(file_path, dtype=np.str, delimiter=',')
    num_sample = np.size(dataset[1:], axis=0)
    num_ids = len(set(dataset[1:, 0]))
    num_zero = np.sum(dataset[1:, 1] == '0')
    num_one = np.sum(dataset[1:, 1] == '1')
    # 输出正负样本的个数及比例
    print('0: %d , 1: %d , ratio: %f' % (num_zero, num_one, num_zero * 1.0 / num_one))
    print('ids num : %d' % num_ids)
    print('sample num : %d' % num_sample)
    return dataset
    
def explore_test_info(filename='test.txt'):
    """
    只是检查预测样本的个数，为12261
    """
    file_path = op.join(default_path, filename)
    dataset = np.loadtxt(file_path, dtype=np.str, delimiter=',')
    num_ids = len(set(dataset[1:]))
    num_sample = np.size(dataset[1:], axis=0)
    print('ids num : %d' % num_ids)
    print('sample num : %d' % num_sample)
    return dataset

def explore_consumption_record(filename='consumption_recode.txt'):
    """
    分析部分用户的消费记录数据
    """
    file_path = op.join(default_path, filename)
    dataset = np.loadtxt(file_path, dtype=np.str, delimiter=',')
    
    save_path = op.join(explore_path, 'consumption_explore.csv')
    if not op.exists(save_path):
        data_file = open(save_path, 'wb')
        csv_writer = csv.writer(data_file)
        for line in dataset[:1001]:
            csv_writer.writerow(line)
        data_file.close()
        print('Finished!')
    # 检测所有消费的记录条数
    num_sample = np.size(dataset[1:], axis=0)
    print('sample num : %d' % num_sample)
    # 检测总共的ID数目
    num_ids = len(set(dataset[1:, 0]))
    print('ids num : %d' % num_ids)
    # 检测不同消费记录的条数
    num_record = len(set(dataset[1:, 1]))
    print('record num : %d' % num_record)
    # 检测类别属性值的个数
    col_category = [8, 18, 22, 23]
    for i in col_category:
        num_feaValue = len(set(dataset[1:, i]))
        print('%s : %d' % (dataset[0, i], num_feaValue))

def main():
    explore_basic_info()
#     explore_train_info()
#     explore_test_info()
#     explore_consumption_record()
#     pass

if __name__ == '__main__':
    main()
