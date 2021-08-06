import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split  # cross_validation
from xgboost import plot_importance, plot_tree
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签

if __name__ == "__main__":
    # 记载样本数据集
    path = 'data.csv'  # 数据文件路径

    # 我方本金,我方利息,资方本金,资金方利息,借款金额,借款期数,放款日期年,放款日期月,放款日期日,首期应还款日年,首期应还款日月,首期应还款日日,借款利率,间隔天数,是否足月,问题分类
    # load data
    data = pd.read_csv(path, header=None)
    x, y = data[list(range(15))], data[15]

    x.columns = ['我方本金', '我方利息', '资方本金', '资金方利息', '借款金额', '借款期数', '放款日期年', '放款日期月', '放款日期日', '首期应还款日年', '首期应还款日月',
                 '首期应还款日日', '借款利率', '间隔天数', '是否足月', ]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=50)

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 2, 'eta': 0.3, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 16}

    bst = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)
    y_hat = bst.predict(data_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_hat)
    print('accuarcy:%.2f%%' % (accuracy * 100))

    importance = bst.get_fscore()
    feature_num = 10  # 想要查看top10的特征名称及权重，这里设置想要查看前多少个特征及其权重
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    importance = importance[:feature_num]
    print(importance)

    plot_tree(bst, num_trees=0)
    plt.show()

    # src = xgb.to_graphviz(bst, num_trees=0, leaf_node_params={'shape': 'plaintext'})
    # src.view("tree")
