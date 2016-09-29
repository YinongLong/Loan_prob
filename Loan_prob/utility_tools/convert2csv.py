# -*- coding:utf-8 -*-
'''
Created on 2016年9月23日

@author: YinongLong

将txt格式给定的数据转换为csv格式，方便查看
'''
from __future__ import print_function

import os.path as op

import numpy as np

default_dir = 'E:/dataset/dataseriesv2/data'

def convert(filename='user_info'):
    file_path = op.join(default_dir, filename + '.txt')
    dataset = np.loadtxt(file_path, dtype=np.str, delimiter=',')
    save_path = op.join(default_dir, filename + '.csv')
    np.savetxt(save_path, dataset, fmt='%s',delimiter=',')
    print('Finished!')

def concatenate_convert(filename='user_info', labelname='train'):
    """
    将用户的基本信息数据和训练集的标签集合起来一起输出查看
    """
    label_path = op.join(default_dir, labelname + '.txt')
    label_data = np.loadtxt(label_path, dtype=np.str, delimiter=',', skiprows=1)
    label_dict = {}
    for uid, label in label_data:
        label_dict[uid] = label
    
    file_path = op.join(default_dir, filename + '.txt')
    dataset = np.loadtxt(file_path, dtype=np.str, delimiter=',')
    new_dataset = []
    ishead = True
    for line in dataset:
        temp = line.tolist()
        if ishead:
            temp.append('label')
            ishead = False
        else:
            label = label_dict.get(line[0])
            if label != None:
                temp.append(label)
            else:
                temp.append('')
        new_dataset.append(temp)
    new_dataset = np.array(new_dataset)
    save_path = op.join(default_dir, filename + '.csv')
    np.savetxt(save_path, new_dataset, fmt='%s',delimiter=',')
    print('Finished!')

def main():
#     convert(filename='rong_tag')
    concatenate_convert()

if __name__ == '__main__':
    main()