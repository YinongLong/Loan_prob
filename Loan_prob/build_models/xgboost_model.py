# -*- coding:utf-8 -*-
'''
Created on 2016年9月13日

@author: YinongLong
'''
from __future__ import print_function

import os.path as op
import csv

from proutility_tools_dataset.process_datasetport generate_dataset

import numpy as np

import xgboost as xgb


default_path = 'E:/dataset/dataseriesv2/data'


def xgboost_model(train, test):
    """
    使用xgboost模型
    :type train: np.array(float64)
    :type test: np.array(float64)
    """
    dtrain = xgb.DMatrix(train[:,:-1], label=train[:, -1], missing=-1.0)
    dtest = xgb.DMatrix(test, missing=-1.0)
    
    params = {'max_depth':6, 'eta':0.5, 'silent':0, 'objective':'binary:logistic'}
    num_round = 5
    
    bst = xgb.train(params, dtrain, num_round)
    preds = bst.predict(dtest)
    return preds

def xgboost_cv(train):
    """
    对xgboost使用交叉验证，来选择最优参数
    """
    dtrain = xgb.DMatrix(train[:, :-1], label=train[:, -1], missing=-1.0)
    
    params = {'max_depth':6, 'eta':0.5, 'silent':0, 'objective':'binary:logistic'}
    num_round = 20
    print('Running cross-validation')
    result = xgb.cv(params, dtrain, num_round, nfold=10, metrics={'auc'}, early_stopping_rounds=20,
           callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
    print(result)
    
def output_result(uids, preds, filename='test_preds.txt'):
    file_path = op.join(default_path, filename)
    data_file = open(file_path, 'wb')
    csv_writer = csv.writer(data_file)
    csv_writer.writerow(['user_id','probability'])
    for uid, pred in zip(uids, preds):
        csv_writer.writerow([uid, pred])
    data_file.close()
    print('Finished!')

def main():
    train, test = generate_dataset()
    
    uids = train[:, 0]
    train = train[:, 1:].astype(np.float64)
    print(np.sum(train[:, -1]))
    test = test[:, 1:].astype(np.float64)
#     xgboost_cv(train)
    preds = xgboost_model(train, test)
    output_result(uids, preds)

if __name__ == '__main__':
    main()