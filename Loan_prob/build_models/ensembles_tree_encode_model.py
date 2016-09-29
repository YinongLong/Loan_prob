# -*- coding:utf-8 -*-
'''
Created on 2016年9月24日

@author: YinongLong
'''
from __future__ import print_function

# from proutility_tools_dataset.provide_datasetport get_train_dataset

from utility_tools.provide_dataset import get_train_dataset

import matplotlib.pyplot as plt
import numpy as np
import csv
import os.path as op

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

default_dir = '/Users/Yinong/Downloads/dataseriesv2'

def output_result(uids, preds, filename='pred_resut.txt'):
    file_path = op.join(default_dir, filename)
    data_file = open(file_path, 'wb')
    csv_writer = csv.writer(data_file)
    csv_writer.writerow(['user_id','probability'])
    for uid, pred in zip(uids, preds):
        csv_writer.writerow([uid, pred])
    data_file.close()
    print('Finished!')
    
def get_train_test(train, test):
    X_train = train[:, 2:].astype(np.float)
    y_train = train[:, 1].astype(np.float)
    X_test = test[:, 1:]
    X_ids = test[:, 0]
    return X_train, y_train, X_test, X_ids

def grd_lr_model(train, test):
    X_train, y_train, X_test, X_ids = get_train_test(train, test)
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
    
    grd = GradientBoostingClassifier(n_estimators=10)
    grd_enc = OneHotEncoder()
    grd_lm = LogisticRegression()
    grd.fit(X_train, y_train)
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
    y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
    output_result(X_ids, y_pred_grd_lm)

def rt_lr_model(train, test):
    X_train, y_train, X_test, X_ids = get_train_test(train, test)
    rt = RandomTreesEmbedding(max_depth=3, n_estimators=10, random_state=0)
    rt_lm = LogisticRegression()
    pipeline = make_pipeline(rt, rt_lm)
    pipeline.fit(X_train, y_train)
    y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
    output_result(X_ids, y_pred_rt)

def rf_lr_model(train, test):
    X_train, y_train, X_test, X_ids = get_train_test(train, test)
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
    
    rf = RandomForestClassifier(max_depth=4, n_estimators=10)
    rf_enc = OneHotEncoder()
    rf_lm = LogisticRegression()
    rf.fit(X_train, y_train)
    rf_enc.fit(rf.apply(X_train))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
    y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
    output_result(X_ids, y_pred_rf_lm)

def check_model(train):
    X_train, X_test, y_train, y_test = train_test_split(train[:, 2:].astype(np.float), train[:, 1].astype(np.float), test_size=0.3)
    
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
    
    # 使用GBT进行特征编码，然后使用LogisticRegression
    grd = GradientBoostingClassifier(n_estimators=10)
    grd_enc = OneHotEncoder()
    grd_lm = LogisticRegression()
    grd.fit(X_train, y_train)
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
    y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:,:,0]))[:, 1]
    fpr_grd_lm, tpr_grd_lm , _ = roc_curve(y_test, y_pred_grd_lm)
    # 直接使用GBT进行预测
    y_pred_grd = grd.predict_proba(X_test)[:, 1]
    fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)
    
    # 完全使用随机树进行非监督的编码
    rt = RandomTreesEmbedding(max_depth=4, n_estimators=10, random_state=0)
    rt_lm = LogisticRegression()
    pipeline = make_pipeline(rt, rt_lm)
    pipeline.fit(X_train, y_train)
    y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)
    
    # 基于随机森林的监督特征转换
    rf = RandomForestClassifier(max_depth=5, n_estimators=10)
    rf_enc = OneHotEncoder()
    rf_lm = LogisticRegression()
    rf.fit(X_train, y_train)
    rf_enc.fit(rf.apply(X_train))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
    y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
    fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)
    
    plt.figure(1)
    plt.plot([0, 1],[0, 1], 'k--')
    plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
    plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
    plt.plot(fpr_grd, tpr_grd, label='GBT')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

def main():
    train, test = get_train_dataset()
    check_model(train)
#     rf_lr_model(train, test)
#     grd_lr_model(train, test)

if __name__ == '__main__':
    main()