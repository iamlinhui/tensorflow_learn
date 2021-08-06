# -*-encoding: utf-8-*-
import pandas as pd
import xgboost as xgb
from matplotlib import pylab as plt
from sklearn.metrics import accuracy_score  # 准确率
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

# 记载样本数据集
path = 'aaa.txt'  # 数据文件路径

# 我方本金,我方利息,资方本金,资金方利息,借款金额,借款期数,放款日期(年月日),首期应还款日(年月日),借款利率
data = pd.read_csv(path, header=None)
x, y = data[list(range(13))], data[13]
# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123457)

# 算法参数
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class': 2,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 6,  # 构建树的深度，越大越容易过拟合
    'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.7,  # 生成树时进行的列采样
    'min_child_weight': 3,
    'slient': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.1,  # 如同学习率
    'seed': 1000,
    'nthread': 4,  # cpu 线程数
}

# 生成数据集格式
dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 500
# xgboost模型训练
model = xgb.train(params, dtrain, num_rounds)

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
y_pred = model.predict(dtest)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print('accuarcy:%.2f%%' % (accuracy * 100))

importance = model.get_fscore()
print(importance)

# 显示重要特征
plot_importance(model)
plt.show()
