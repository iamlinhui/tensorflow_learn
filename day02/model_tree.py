# -*-encoding: utf-8-*-
import os
import pickle

import xgboost as xgb
from matplotlib import pylab as plt
from xgboost import plot_tree

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
plt.rcParams['font.sans-serif'] = ['FangSong']  # 显示中文标签

model = pickle.load(open("difference.dat", "rb"))

importance = model.get_fscore()
print(importance)

plot_tree(model, num_trees=13)
plt.show()

# dot -Tpng -o hello.png tmp.dot
digraph = xgb.to_graphviz(model, num_trees=13, leaf_node_params={'shape': 'plaintext'})
digraph.format = 'png'
digraph.view("tree")
