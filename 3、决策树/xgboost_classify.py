import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split  # cross_validation
from xgboost import plot_importance, plot_tree
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    path = 'iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    data = pd.read_csv(path, header=None)
    x, y = data[list(range(4))], data[4]
    y = pd.Categorical(y).codes
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=50)

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 2, 'eta': 0.3, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}

    bst = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)
    y_hat = bst.predict(data_test)

    print(y_hat)
    print(y_test.reshape(1, -1))
    result = y_test.reshape(1, -1) == y_hat
    print('正确率:\t', float(np.sum(result)) / len(y_hat))

    importance = bst.get_fscore()
    feature_num = 10  # 想要查看top10的特征名称及权重，这里设置想要查看前多少个特征及其权重
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    importance = importance[:feature_num]
    print(importance)

    plot_tree(bst, num_trees=0)
    plt.show()

    src = xgb.to_graphviz(bst, num_trees=0, leaf_node_params={'shape': 'plaintext'})
    src.view("tree")
