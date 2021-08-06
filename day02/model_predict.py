# -*-encoding: utf-8-*-
import pickle

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

model = pickle.load(open("difference.dat", "rb"))

path = 'data.csv'

data = pd.read_csv(path, header=None)
x, y = data[list(range(15))], data[15]
x.columns = ['我方本金', '我方利息', '资方本金', '资金方利息', '借款金额', '借款期数', '放款日期年', '放款日期月', '放款日期日', '首期应还款日年', '首期应还款日月',
             '首期应还款日日', '借款利率', '间隔天数', '是否足月', ]

source = xgb.DMatrix(x)
y_pred = model.predict(source)

accuracy = accuracy_score(y, y_pred)
# 预测精准度: 99.22%
print("预测精准度: %.2f%%" % (accuracy * 100.0))
