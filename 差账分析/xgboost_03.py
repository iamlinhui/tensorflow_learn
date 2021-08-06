# -*- coding=utf-8 -*-
import time

import numpy as np
import pandas as pd
import xgboost as xgb

now = time.time()
dataset = pd.read_csv('./train.csv')
train = dataset.iloc[:, 1:].values
labels = dataset.iloc[:, :1].values
tests = pd.read_csv('./test.csv')
test = tests.iloc[:, :].values
paras = {
    'booster': 'gbtree',
    # 一个多类的问题，因此采用multisoft多分类器
    'objective': 'multi:softmax',
    'num_class': 10,
    'gamma': 0.05,  # 树的叶子节点下一个区分的最小损失，越大算法模型越保守
    'max_depth': 12,
    'lambda': 450,  # L2正则项权重
    'subsample': 0.4,  # 采样训练数据，设置为0.5
    'colsample_bytree': 0.7,  # 构建树时的采样比率
    'min_child_weight': 12,  # 节点的最少特征数
    'silent': 1,
    'eta': 0.005,  # 类似学习率
    'seed': 700,
    'nthread': 4,  # cpu线程数
}

offset = 35000  # 训练集中数据50000,划分35000用作训练，15000用作验证
num_rounds = 500  # 迭代次数
xgtest = xgb.DMatrix(test)  # 加载数据可以是numpy的二维数组形式，也可以是xgboost的二进制的缓存文件，加载的数据存储在对象DMatrix中
xgtrain = xgb.DMatrix(train[:offset, :], label=labels[:offset])  # 将训练集的二维数组加入到里面
xgval = xgb.DMatrix(train[offset:, :], label=labels[offset:])  # 将验证集的二维数组形式的数据加入到DMatrix对象中

watchlist = [(xgtrain, 'train'), (xgval, 'val')]  # return训练和验证的错误率
# 超参数放到集合plst中;
model = xgb.train(paras, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
preds = model.predict(xgtest, ntree_limit=model.best_iteration)
np.savetxt('submission_xgb_MultiSoftMax.csv', np.c_[range(1, len(test) + 1), preds],
           # np._c[]的作用就是将preds与前面的随机数两两配对，放到一块，看我的csdn整理的用法
           delimiter=',', header='ImageId,Label', comments='', fmt='%d')
# header标题为ImagerId和label就是列的名字为这俩个，看submission_xgb_MultiSoftMax.csv就都明白了
cost_time = time.time() - now
print("end...", '\n', "cost time", cost_time, "(s)...")
