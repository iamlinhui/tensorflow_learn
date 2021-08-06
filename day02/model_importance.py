# -*-encoding: utf-8-*-
import pickle

from matplotlib import pylab as plt
from xgboost import plot_importance

plt.rcParams['font.sans-serif'] = ['FangSong']  # 显示中文标签

model = pickle.load(open("difference.dat", "rb"))

importance = model.get_fscore()
print(importance)

# 显示重要特征
plot_importance(model)
plt.show()
