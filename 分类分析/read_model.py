import pickle

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score  # 准确率
from xgboost import plot_tree

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

loaded_model = pickle.load(open("pima.pickle.dat", "rb"))

plot_tree(loaded_model, num_trees=0)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.show()

# 记载样本数据集
path = 'data.csv'  # 数据文件路径

# 我方本金,我方利息,资方本金,资金方利息,借款金额,借款期数,放款日期年,放款日期月,放款日期日,首期应还款日年,首期应还款日月,首期应还款日日,借款利率,间隔天数,是否足月,问题分类
# load data
# data = pd.read_csv(path, header=None)
data = pd.read_csv(path, header=None)
x, y = data[list(range(15))], data[15]
x.columns = ['我方本金', '我方利息', '资方本金', '资金方利息', '借款金额', '借款期数', '放款日期年', '放款日期月', '放款日期日', '首期应还款日年', '首期应还款日月',
             '首期应还款日日', '借款利率', '间隔天数', '是否足月', ]

# make predictions for test data
source = xgb.DMatrix(x)
y_pred = loaded_model.predict(source)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
