import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
from scipy import misc,ndimage

mnist = input_data.read_data_sets('./MNIST_data')

batch_size = 100
width,height = 28,28
mnist_dim = width*height
random_dim = 10
epochs = 1000000

def my_init(size):
    return tf.random_uniform(size, -0.05, 0.05)

# 定义判别器 三层Dense网络 relu激活函数
D_W1 = tf.Variable(my_init([mnist_dim, 128]))
D_b1 = tf.Variable(tf.zeros([128]))
D_W2 = tf.Variable(my_init([128, 32]))
D_b2 = tf.Variable(tf.zeros([32]))
D_W3 = tf.Variable(my_init([32, 1]))
D_b3 = tf.Variable(tf.zeros([1]))
D_variables = [D_W1, D_b1, D_W2, D_b2, D_W3, D_b3]

# 定义生成器 三层Dense网络 reluctant激活函数 
# 在这里 生成器和判别器要求网络复杂度相当->是指参数规模还是要求网络结构和参数一致？
G_W1 = tf.Variable(my_init([random_dim, 32]))
G_b1 = tf.Variable(tf.zeros([32]))
G_W2 = tf.Variable(my_init([32, 128]))
G_b2 = tf.Variable(tf.zeros([128]))
G_W3 = tf.Variable(my_init([128, mnist_dim]))
G_b3 = tf.Variable(tf.zeros([mnist_dim]))
G_variables = [G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]

# 判别器最后直接输出结果
def D(X):
    X = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    X = tf.nn.relu(tf.matmul(X, D_W2) + D_b2)
    X = tf.matmul(X, D_W3) + D_b3
    return X

# 生成器最后使用sigmoid函数约束结果为（0，1）
def G(X):
    X = tf.nn.relu(tf.matmul(X, G_W1) + G_b1)
    X = tf.nn.relu(tf.matmul(X, G_W2) + G_b2)
    X = tf.nn.sigmoid(tf.matmul(X, G_W3) + G_b3)
    return X

# 真实数据和随机数发生器的维度是一样的
real_X = tf.placeholder(tf.float32, shape=[batch_size, mnist_dim])
random_X = tf.placeholder(tf.float32, shape=[batch_size, random_dim])
random_Y = G(random_X)

# eps是作为插值的随机数，构建一个介于Y和Z的分布，其中Y是生成的分布，Z是真实的分布
eps = tf.random_uniform([batch_size, 1], minval=0., maxval=1.)
# 插值
X_inter = eps*real_X + (1. - eps)*random_Y
# 利用tf计算梯度，算出判别器关于两个插值的差分
grad = tf.gradients(D(X_inter), [X_inter])[0]
# 标准化差分：差分的平方，按行求和，再取平方根
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
# 惩罚项是标准化之后的差分-1，再整体求和，这一处与论文中不是特别一致，可能在本质上差不多，
# 乘以10应该是与随机数发生器的维度一致
grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))

# 定义判别器的损失函数，Z真实样本的求和（等价于求平均值），减去通过生成器生成的Y的求和，加上惩罚项
D_loss = tf.reduce_mean(D(real_X)) - tf.reduce_mean(D(random_Y)) + grad_pen
# 定义生成器的损失，就是生成的结果Y的求和，以上两个Loss都是在判别器视角下计算的，也就是D()
G_loss = tf.reduce_mean(D(random_Y))

# 定义优化函数，使用Adam
D_solver = tf.train.AdamOptimizer(1e-4, 0.5).minimize(D_loss, var_list=D_variables)
G_solver = tf.train.AdamOptimizer(1e-4, 0.5).minimize(G_loss, var_list=G_variables)

# 标准的tf会话
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

# 训练100000个epochs，这么多?
for e in range(epochs):
    for i in range(5):
        real_batch_X,_ = mnist.train.next_batch(batch_size)
        # 随机数发生器按照batch产生(-1,1)的随机数,维度为random_dim
        random_batch_X = np.random.uniform(-1, 1, (batch_size, random_dim))
        # 运行5次判别器的优化,输入为真实数据和随机数
        _,D_loss_ = sess.run([D_solver,D_loss], feed_dict={real_X:real_batch_X, random_X:random_batch_X})
    # 
    random_batch_X = np.random.uniform(-1, 1, (batch_size, random_dim))
    # 运行5次判别优化之后,运行1次生成器优化,输入随机数
    _,G_loss_ = sess.run([G_solver,G_loss], feed_dict={random_X:random_batch_X})
    if e % 1000 == 0:
        # 运行1000个epochs之后,显示两个loss
        print('epoch %s, D_loss: %s, G_loss: %s'%(e, D_loss_, G_loss_))
        # 
        n_rows = 6
        # 将生成的数据Y绘制出来,首先reshape成28*28的矩阵
        check_imgs = sess.run(random_Y, feed_dict={random_X:random_batch_X}).reshape((batch_size, width, height))[:n_rows*n_rows]
        # 
        imgs = np.ones((width*n_rows+5*n_rows+5, height*n_rows+5*n_rows+5))
        # 这是做了一个图像的切割么...
        for i in range(n_rows*n_rows):
            imgs[5+5*(i%n_rows)+width*(i%n_rows):5+5*(i%n_rows)+width+width*(i%n_rows), 5+5*(i/n_rows)+height*(i/n_rows):5+5*(i/n_rows)+height+height*(i/n_rows)] = check_imgs[i]
        misc.imsave('out/%s.png'%(e/1000), imgs)
