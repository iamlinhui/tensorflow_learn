# -*-encoding: utf-8-*-
import pandas as pd
import xgboost as xgb
from matplotlib import pylab as plt
from sklearn.metrics import accuracy_score  # 准确率
from sklearn.model_selection import train_test_split
from xgboost import plot_importance

# 记载样本数据集
path = 'iris.data'  # 数据文件路径
data = pd.read_csv(path, header=None)
x, y = data[list(range(4))], data[4]
y = pd.Categorical(y).codes
# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123457)

# 算法参数
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 3,
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'slient': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
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
