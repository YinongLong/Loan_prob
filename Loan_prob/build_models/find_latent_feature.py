# -*- coding:utf-8 -*-
'''
Created on 2016年9月16日

@author: YinongLong

    处理用户的消费记录数据，使用自编码神经网络对用户的
    消费记录数据进行特征的提取。

'''
from __future__ import print_function

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.activations import relu
import numpy as np

import h5py

import os.path as op
import csv

default_dir = 'E:/dataset/dataseriesv2/data'
model_data_path = 'E:/dataset/dataseriesv2/model_data'

def load_consumption_record(filename='consumption_recode.txt'):
    file_path = op.join(default_dir, filename)
    dataset = np.loadtxt(file_path, dtype=np.str, delimiter=',')
    return dataset

def convert_dataset(dataset, train=True):
    """
    根据参数train，来决定转换数据的方式，当train=True，代表生成用来训练神经网络的数据集；
    当train=False，代表转化为用来生成用户隐含特征的数据集
    """
    if not train:
        user_ids = np.copy(dataset[1:, 0])
    dataset = np.copy(dataset[1:, 2:])
    dataset = dataset.astype(np.float)
    # 对非离散变量进行标准化
    special_col = [6, 16, 20, 21]
    num_sample = np.size(dataset, axis=0)
    num_feature = np.size(dataset, axis=1)
    min_max_scaler = MinMaxScaler()
    dataset_col = []
    for i in range(num_feature):
        if i in special_col:
            dataset_col.append(dataset[:, i].reshape((num_sample, 1)))
        else:     
            dataset_col.append(min_max_scaler.fit_transform(dataset[:, i].reshape((num_sample, 1))))
    # 在这里根据转化的需要，分为训练自编码和预测隐含特征，分别返回不同的值
    if train:
        dataset = np.concatenate(dataset_col, axis=1)
        np.random.shuffle(dataset)
        X_dataset = dataset
        Y_dataset = np.copy(dataset)
        return X_dataset, Y_dataset
    else:
        dataset_col.insert(0, user_ids.reshape((num_sample, 1)));
        dataset = np.concatenate(dataset_col, axis=1)
        return dataset

def train_autoEncoder(X, Y, test=False, weights_filename='saved_weights'):
    """
    训练自编码器神经网络，返回输入到隐藏层的权重和偏置参数
    """
    # 将数据划分为训练集和测试集
    num_sample = np.size(X, axis=0)
    num_train = int(num_sample * 0.7)
    X_train = X[0:num_train]
    Y_train = Y[0:num_train]
    X_test = X[num_train:]
    Y_test = Y[num_train:]
    
    # 构建模型
    model = Sequential()
    model.add(Dense(output_dim=10, input_dim=26))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=26))
    model.add(Activation('relu'))
    
    sgd = SGD(lr=0.01, momentum=0.1)
    model.compile(optimizer=sgd, loss='mae', metrics=['mae'])
    if test:
        model.fit(X_train, Y_train, batch_size=50, nb_epoch=250, verbose=1)
        result_eval = model.evaluate(X_test, Y_test)
        print(result_eval)
        Y_pred = model.predict(X_test)
        output_pred_result(Y_pred, Y_test)
    else:
        model.fit(X, Y, batch_size=50, nb_epoch=250, verbose=0)
        weights_filename = '%s_consumption' % weights_filename
        file_path = op.join(model_data_path, weights_filename)
        model.save(file_path)
        return model
    
def generate_user_feature(dataset, autoEncoder=None):
    """
    根据训练的自编码器，来生成所有拥有消费记录用户的隐含特征
    """
    dataset = convert_dataset(dataset, train=False)
    if autoEncoder == None:
        data_path = op.join(model_data_path, 'saved_weights_consumption')
        if op.isfile(data_path):
            # 载入之前保存的模型的参数
            data_file = h5py.File(data_path, mode='r')
            weights = np.array(data_file.get('model_weights').get('dense_1').get('dense_1_W'))
            bias = np.array(data_file.get('model_weights').get('dense_1').get('dense_1_b'))
        else:
            return None
    else:
        weights = autoEncoder.get_weights()[0]
        bias = autoEncoder.get_weights()[1]
        del autoEncoder
    # 针对每一条消费记录生成一个隐含特征，且将同一个用户ID的隐含特征放到一个列表中
    dict_user = {}
    for line in dataset:
        if line[0] not in dict_user:
            dict_user[line[0]] = []
        hidden_feature = caculate_hidden_feature(line[1:].astype(np.float), weights, bias)
        dict_user[line[0]].append(hidden_feature.reshape(1, -1))
    # 对每一个隐含特征多余一个的用户，对其所有的隐含特征计算均值
    dataset = []
    for user_id, features in dict_user.items():
        feature = np.concatenate(features, axis=0)
        feature = np.mean(feature, axis=0)
        feature = feature.tolist()
        feature.insert(0, user_id)
        dataset.append(feature)
    return dataset

def caculate_hidden_feature(sample, weights, bias):
    return relu(np.dot(weights.T, sample) + bias)
    
def output_pred_result(Y_pred, Y_test):
    """
    输出模型预测的结果，与原本的目标进行比较
    """
    save_path = op.join(default_dir, 'con_pred.csv')
    data_file = open(save_path, 'wb')
    csv_writer = csv.writer(data_file)
    for pre, rea in zip(Y_pred, Y_test):
        pre = pre.tolist()
        pre.append(-1)
        rea = rea.tolist()
        pre.extend(rea)
        csv_writer.writerow(pre)
    data_file.close()
    print('Finished!')
    
def output_hidden_feature(dataset, filename='consumption_hidden_feature.csv'):
    """
    将生成的隐含特征输出到文件中进行保存，方便后续的使用
    """
    file_path = op.join(model_data_path, filename)
    data_file = open(file_path, 'wb')
    csv_writer = csv.writer(data_file)
    for line in dataset:
        csv_writer.writerow(line)
    data_file.close()
    print('Finished!')

def main():
    dataset = load_consumption_record()
    X, Y = convert_dataset(dataset)
    # 生成自编码器，然后对所有的有消费记录的用户生成其隐含特征
    train_autoEncoder(X, Y)
    fea_dataset = generate_user_feature(dataset)
    output_hidden_feature(fea_dataset)

if __name__ == '__main__':
    main()