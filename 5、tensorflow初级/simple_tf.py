import numpy as np
import tensorflow as tf

# 用 NumPy 随机生成 100 个数据
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300

tf.compat.v1.disable_eager_execution()

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.compat.v1.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.compat.v1.global_variables_initializer()

# 启动图 (graph)
sess = tf.compat.v1.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201):
    l, _ = sess.run([loss, train])
    print(l)

w_result, b_result = sess.run([W, b])
print(w_result, b_result)

sess.close()
