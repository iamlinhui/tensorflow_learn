import matplotlib.pyplot as plt
import pandas as pd

dftrain_raw = pd.read_csv('./data/train.csv')
dftest_raw = pd.read_csv('./data/test.csv')
dftrain_raw.head(10)

# %matplotlib inline
# %config InlineBackend.figure_format = 'png'

# label分布情况
ax = dftrain_raw['Survived'].value_counts().plot(kind='bar', figsize=(12, 8), fontsize=15, rot=0)
ax.set_ylabel('Counts', fontsize=15)
ax.set_xlabel('Survived', fontsize=15)
plt.show()

# 年龄分布情况
ax = dftrain_raw['Age'].plot(kind='hist', bins=20, color='purple', figsize=(12, 8), fontsize=15)
ax.set_ylabel('Frequency', fontsize=15)
ax.set_xlabel('Age', fontsize=15)
plt.show()

ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
dftrain_raw.query('Survived == 1')['Age'].plot(kind='density', figsize=(12, 8), fontsize=15)
ax.legend(['Survived==0', 'Survived==1'], fontsize=12)
ax.set_ylabel('Density', fontsize=15)
ax.set_xlabel('Age', fontsize=15)
plt.show()
