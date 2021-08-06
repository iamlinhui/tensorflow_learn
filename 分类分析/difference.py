# -*-encoding: utf-8-*-
import pickle

import pandas as pd
import xgboost as xgb
from matplotlib import pylab as plt
from sklearn.metrics import accuracy_score  # 准确率
from sklearn.model_selection import train_test_split
from xgboost import plot_importance, plot_tree, XGBClassifier
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签

# 数据文件路径
path = 'data.csv'

data = pd.read_csv(path, header=None)  # load data
x, y = data[list(range(15))], data[15]

x.columns = ['我方本金', '我方利息', '资方本金', '资金方利息', '借款金额', '借款期数', '放款日期年', '放款日期月', '放款日期日', '首期应还款日年', '首期应还款日月',
             '首期应还款日日', '借款利率', '间隔天数', '是否足月', ]

# 数据集分割  test_size：样本占比，如果是整数的话就是样本的数量 random_state：是随机数的种子。 填0或不填，每次都会不一样 | 填1，其他参数一样的情况下你得到的随机数组是一样的
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

data_train = xgb.DMatrix(x_train, label=y_train)
data_test = xgb.DMatrix(x_test, label=y_test)
watch_list = [(data_test, 'eval'), (data_train, 'train')]

# 算法参数
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 采用softmax目标函数处理多分类问题，同时需要设置参数num_class（类别个数）
    'num_class': 8,  # 类别个数，与 multi:softmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,  # 构建树的深度，越大越容易过拟合
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.1,  # 如同学习率
    'seed': 1000,
    'nthread': 4,  # cpu 线程数
}

# 生成数据集格式
d_train = xgb.DMatrix(x_train, y_train)
# xgboost模型训练 num_boost_round 这是指提升迭代的次数，也就是生成多少基模型
model = xgb.train(params, d_train, num_boost_round=500, evals=watch_list)

# 对测试集进行预测
d_test = xgb.DMatrix(x_test)
y_pred = model.predict(d_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print('accuarcy:%.2f%%' % (accuracy * 100))

# importance = model.get_fscore()
# print(importance)
#
# # 显示重要特征
# plot_importance(model)
# plt.show()

# plot_tree(model, num_trees=13)
# plt.show()

src = xgb.to_graphviz(model, num_trees=13, leaf_node_params={'shape': 'plaintext'})
src.view("tree")

# for i in range(20):
#     plot_tree(model, num_trees=i, fmap='xgb.fmap')
#     plt.show()

# pickle.dump(model, open("pima.pickle.dat", "wb"))

# source = xgb.DMatrix(x)
# y_pred = model.predict(source)
# accuracy = accuracy_score(y, y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
