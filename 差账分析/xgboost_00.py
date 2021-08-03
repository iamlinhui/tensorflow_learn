import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split  # cross_validation

if __name__ == "__main__":
    path = 'aaa.txt'  # 数据文件路径

    # 我方本金,我方利息,资方本金,资金方利息,借款金额,借款期数,放款日期,首期应还款日,借款利率
    data = pd.read_csv(path, header=None)
    x, y = data[list(range(13))], data[13]
    # x = pd.Categorical(x).codes
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=500)

    data_train = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    data_test = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 9}

    bst = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)
    y_hat = bst.predict(data_test)
    print(y_hat)

    importance = bst.get_fscore()
    print(importance)
